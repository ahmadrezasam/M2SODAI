import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Set sharing strategy to 'file_system' to prevent shared memory (/dev/shm) exhaustion crashes when using workers > 0
mp.set_sharing_strategy('file_system')

# Set number of PyTorch CPU threads to 1 to prevent CPU core over-subscription and thrashing inside worker processes
torch.set_num_threads(1)

# =====================================================================
# 1. Monkey-Patching for PyTorch 2.6+ & Ultralytics .npy Compatibility
# =====================================================================

# Monkey-patch torch.load for safe weights loading in PyTorch 2.6+
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

# Monkey-patch Ultralytics to recognize .npy files
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

# Monkey-patch PIL Image.open to handle .npy files without crashes
from PIL import Image
original_image_open = Image.open

class DummyImage:
    def __init__(self):
        self.format = 'PNG'
        self.size = (640, 640)
    def verify(self):
        pass
    def getexif(self):
        return None

def custom_image_open(fp, *args, **kwargs):
    if str(fp).endswith('.npy'):
        return DummyImage()
    return original_image_open(fp, *args, **kwargs)

Image.open = custom_image_open

# Monkey-patch Ultralytics plot_images to prevent 'Can't call numpy() on Tensor that requires grad' in background threads
import ultralytics.utils.plotting as plotting_module
original_plot_images = plotting_module.plot_images

def custom_plot_images(labels, *args, **kwargs):
    if isinstance(labels, dict) and "img" in labels:
        img_val = labels["img"]
        if isinstance(img_val, torch.Tensor) and img_val.requires_grad:
            labels = labels.copy()
            labels["img"] = img_val.detach()
            
    if "images" in kwargs:
        imgs = kwargs["images"]
        if isinstance(imgs, torch.Tensor) and imgs.requires_grad:
            kwargs["images"] = imgs.detach()
    elif len(args) > 0:
        args_list = list(args)
        imgs = args_list[0]
        if isinstance(imgs, torch.Tensor) and imgs.requires_grad:
            args_list[0] = imgs.detach()
        args = tuple(args_list)
        
    return original_plot_images(labels, *args, **kwargs)

plotting_module.plot_images = custom_plot_images

# Global Monkey-patching for build_yolo_dataset to return HSIDataset for target/validation data
import ultralytics.data.build as build_module
import ultralytics.data as data_module
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.models.yolo.obb.val as obb_val
from hsi_dataset import HSIDataset

original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"[Dataset] Building HSIDataset for HSI path: {img_path} at 224x224 native resolution")
        return HSIDataset(
            img_path=img_path,
            imgsz=224,               # Load and augment at native 224x224 to eliminate CPU resize bottlenecks
            batch_size=batch,
            augment=mode == 'train',
            hyp=cfg,
            rect=rect,
            cache=cfg.cache,
            single_cls=cfg.single_cls,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            data=data,
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
            task=cfg.task,
        )
    return original_build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=rect, stride=stride, multi_modal=multi_modal)

build_module.build_yolo_dataset = custom_build_yolo_dataset
data_module.build_yolo_dataset = custom_build_yolo_dataset
detect_val.build_yolo_dataset = custom_build_yolo_dataset
if hasattr(obb_val, "build_yolo_dataset"):
    obb_val.build_yolo_dataset = custom_build_yolo_dataset

# =====================================================================
# 2. Monkey-Patching OBBModel Forward to Inject the Spectral Adaptor
# =====================================================================
from ultralytics.nn.tasks import OBBModel

if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

def new_forward(self_model, x, *args, **kwargs):
    if isinstance(x, dict):
        img = x["img"]
    else:
        img = x
        
    is_hsi = img.size(1) == 30
    has_adaptor = hasattr(self_model, "spectral_adaptor")

    if has_adaptor and is_hsi:
        # Match tensor type (Float vs Half/AMP)
        if next(self_model.spectral_adaptor.parameters()).dtype != img.dtype:
            self_model.spectral_adaptor = self_model.spectral_adaptor.to(img.dtype)
        # Pass 30 channels through adaptor -> 3 channels pseudo-RGB
        img = self_model.spectral_adaptor(img)
        
        # GPU-side resize to 640x640 for spatial feature transfer compatibility with pre-trained yolo26s-obb
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
        
    # Shallow copy to prevent leaking requires_grad=True to YOLO's plotting thread
    if isinstance(x, dict):
        x = x.copy()
        x["img"] = img
    else:
        x = img

    # Match main model's parameters weight type to prevent precision mismatches
    main_dtype = torch.float32
    for name, param in self_model.named_parameters():
        if "spectral_adaptor" not in name:
            main_dtype = param.dtype
            break

    if isinstance(x, dict):
        x["img"] = x["img"].to(main_dtype)
    else:
        x = x.to(main_dtype)

    # Pass 3-channel pseudo-RGB to standard YOLO OBB forward
    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = new_forward

# =====================================================================
# 3. Spectral Adaptor (30-channels to 3-channels)
# =====================================================================
class Spectral3DCNN(nn.Module):
    """
    1x1 2D Convolutional Spectral Adaptor and BatchNorm2d.
    Projects 30 PCA-compressed spectral channels down to 3 standard channels.
    Initializes weight matrix to map the first three primary components to R, G, B,
    and applies batch normalization to standardise output ranges.
    """
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        with torch.no_grad():
            self.conv.weight.zero_()
            # Map PC1 -> R (channel 0)
            self.conv.weight[0, 0, 0, 0] = 1.0
            # Map PC2 -> G (channel 1)
            self.conv.weight[1, 1, 0, 0] = 1.0
            # Map PC3 -> B (channel 2)
            self.conv.weight[2, 2, 0, 0] = 1.0
            # Initialize BatchNorm to output [0, 1] range initially for perfect pre-trained backbone compatibility
            self.bn.weight.fill_(1.0)
            self.bn.bias.fill_(0.0)

    def forward(self, x):
        x = self.conv(x)
        
        # Vectorized batch-wise min-max normalization per-image to [0.0, 1.0]
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        im_min = x_flat.min(dim=1, keepdim=True)[0]
        im_max = x_flat.max(dim=1, keepdim=True)[0]
        x = (x - im_min.view(B, 1, 1, 1)) / (im_max.view(B, 1, 1, 1) - im_min.view(B, 1, 1, 1) + 1e-8)
        
        # Force BatchNorm to use evaluation mode (constant mean=0, var=1)
        # to prevent train-eval discrepancy and ensure perfect stability.
        self.bn.eval()
        x = self.bn(x)
        return x

# =====================================================================
# 4. Supervised 30-Channel OBB Trainer
# =====================================================================
from ultralytics.models.yolo.obb import OBBTrainer, OBBValidator

# =====================================================================
# 4. Monkey-Patching OBBTrainer & OBBValidator for low-memory GPU Adaptor + Resize
# =====================================================================
original_obb_trainer_preprocess = OBBTrainer.preprocess_batch
def custom_obb_trainer_preprocess(self_train, batch):
    batch = original_obb_trainer_preprocess(self_train, batch)
    img = batch["img"]
    if img.size(1) == 30 and hasattr(self_train, "model") and hasattr(self_train.model, "spectral_adaptor"):
        # Match data type of adaptor weights to avoid precision mismatch in float16/float32
        adaptor = self_train.model.spectral_adaptor
        if next(adaptor.parameters()).dtype != img.dtype:
            self_train.model.spectral_adaptor = adaptor.to(img.dtype)
            adaptor = self_train.model.spectral_adaptor
            
        # 1. Project 30 channels -> 3 channels (Pseudo-RGB)
        img = adaptor(img)
        
        # 2. Resize 3 channels -> 640x640 (saves 90% memory vs 30 channels!)
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(
                img, size=(640, 640), mode='bilinear', align_corners=False
            )
        batch["img"] = img
    return batch
OBBTrainer.preprocess_batch = custom_obb_trainer_preprocess

# Save model on validator instance during call to access it in preprocess
original_obb_validator_call = OBBValidator.__call__
def custom_obb_validator_call(self_val, *args, **kwargs):
    trainer = kwargs.get("trainer", None)
    if len(args) > 0:
        trainer = args[0]
    model = kwargs.get("model", None)
    if len(args) > 1:
        model = args[1]
        
    if trainer is not None:
        self_val.model = getattr(trainer, "ema", None) and trainer.ema.ema or trainer.model
    elif model is not None:
        self_val.model = model
        
    return original_obb_validator_call(self_val, *args, **kwargs)
OBBValidator.__call__ = custom_obb_validator_call

original_obb_validator_preprocess = OBBValidator.preprocess
def custom_obb_validator_preprocess(self_val, batch):
    batch = original_obb_validator_preprocess(self_val, batch)
    img = batch["img"]
    if img.size(1) == 30 and hasattr(self_val, "model") and hasattr(self_val.model, "spectral_adaptor"):
        adaptor = self_val.model.spectral_adaptor
        if next(adaptor.parameters()).dtype != img.dtype:
            self_val.model.spectral_adaptor = adaptor.to(img.dtype)
            adaptor = self_val.model.spectral_adaptor
            
        # 1. Project 30 channels -> 3 channels (Pseudo-RGB)
        img = adaptor(img)
        
        # 2. Resize 3 channels -> 640x640 (saves 90% memory vs 30 channels!)
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(
                img, size=(640, 640), mode='bilinear', align_corners=False
            )
        batch["img"] = img
    return batch
OBBValidator.preprocess = custom_obb_validator_preprocess

class Supervised30chTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # 1. Load the pre-trained spatial model
        model = super().get_model(cfg, weights, verbose)
        
        # 2. Attach the Spectral Adaptor to map HSI inputs into standard YOLO input space
        print("[Trainer] Attaching Spectral3DCNN adaptor to model...")
        model.spectral_adaptor = Spectral3DCNN(in_channels=30, out_channels=3).to(self.device)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        # Override to ensure HSIDataset is built for all HSI paths at native 224x224 resolution
        print(f"[Trainer] Building HSIDataset for path: {img_path} at 224x224 native resolution")
        return HSIDataset(
            img_path=img_path,
            imgsz=224,               # Load and augment at native 224x224 to eliminate CPU resize bottlenecks
            batch_size=batch,
            augment=mode == 'train',
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.0 if mode == 'train' else 0.5,
            data=self.data,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == 'train' else 1.0,
            task=self.args.task,
        )

# =====================================================================
# 5. Launch Supervised Training and Evaluation
# =====================================================================
def run_supervised_hsi_training():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"

    print("\n" + "="*80)
    print("STARTING SINGLE-BRANCH SUPERVISED HSI TRAINING ON ALL IMAGES")
    print("="*80 + "\n")

    # Labeled HSI data YAML (full train and validation sets)
    train_yaml = os.path.join(project_root, "baseline_official/yolo26_pca30_full.yaml")
    
    # Target Evaluation: PCA 30-channel test dataset
    test_yaml = os.path.join(project_root, "baseline_official/yolo26_pca30_test.yaml")

    # Starting weights: We start from standard pre-trained yolo26s-obb weights
    weights_path = os.path.join(project_root, "yolo26s-obb.pt")
    if not os.path.exists(weights_path):
        print(f"Pre-trained weights not found at root, checking default directories...")
        # fallback if model weights are located elsewhere
        weights_path = "yolo26s-obb.pt"

    overrides = {
        "model": weights_path,
        "data": train_yaml,
        "epochs": 100,               # Set training epochs
        "imgsz": 640,
        "batch": 16,                # Optimized batch size (16) for RTX 2080 Ti
        "task": "obb",
        "device": 0,                # GPU ID
        "single_cls": True,
        "project": os.path.join(project_root, "runs/supervised_30ch"),
        "name": "yolo26s_hsi_supervised_full",
        "exist_ok": True,
        "patience": 30,             # Early stopping patience
        "save_period": 10,
        "plots": True,
        "cos_lr": True,
        "lr0": 0.001,               # Initial learning rate
        "optimizer": "AdamW",
        "workers": 0,               # 0 workers is deadlock-free and extremely fast due to RAM caching
    }

    # Initialize the customized supervised trainer
    trainer = Supervised30chTrainer(overrides=overrides)
    
    # Train the combined model
    trainer.train()


    # Once trained, evaluate on the PCA30 test set
    print("\n" + "="*80)
    print("TRAINING COMPLETED. EVALUATING ON PCA30 TEST SET...")
    print("="*80 + "\n")

    best_checkpoint = os.path.join(
        project_root, 
        "runs/supervised_30ch/yolo26s_hsi_supervised_full/weights/best.pt"
    )

    if os.path.exists(best_checkpoint):
        from ultralytics import YOLO
        # Load the best trained model checkpoint (which automatically saves the adaptor state)
        model = YOLO(best_checkpoint)
        
        # Run validation/test
        results = model.val(
            data=test_yaml,
            imgsz=640,
            batch=16,
            device=0,
            single_cls=True,
            plots=True,
            save_json=False,
            workers=0,
            project=os.path.join(project_root, "runs/supervised_30ch/eval_pca_test")
        )
        
        metrics = results.results_dict
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        print("\n" + "="*80)
        print("HSI PCA30 TEST RESULTS")
        print("="*80)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"mAP50:     {map50:.4f}")
        print(f"mAP50-95:  {map50_95:.4f}")
        print("="*80 + "\n")
    else:
        print(f"Error: Best checkpoint not found at {best_checkpoint}. Skipping evaluation.")

if __name__ == "__main__":
    run_supervised_hsi_training()

