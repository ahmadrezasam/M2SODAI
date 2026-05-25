import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import cv2

# Disable OpenCV multithreading to prevent memory leaks in dataloader workers
cv2.setNumThreads(0)

# Set sharing strategy to 'file_system' to prevent shared memory (/dev/shm) exhaustion crashes when using workers > 0
mp.set_sharing_strategy('file_system')

# Set number of PyTorch CPU threads to 1 to prevent CPU core over-subscription and thrashing inside worker processes
torch.set_num_threads(1)

try:
    from ultralytics.utils.torch_utils import autocast
except ImportError:
    from torch.cuda.amp import autocast
import types


# Monkey-patch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

from uda.mt_obb_trainer import MeanTeacherOBBTrainer
from hsi_dataset import HSIDataset

# Monkey-patch Ultralytics to recognize .npy files
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

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

original_build_yolo_dataset = build_module.build_yolo_dataset

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
        print(f"[Dataset] Building HSIDataset for target HSI path: {img_path} at 224x224 native resolution")
        return HSIDataset(
            img_path=img_path,
            imgsz=224,               # Load and cache at native 224x224 to eliminate CPU resize bottlenecks
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

# ==========================================
# 0. Global OBBModel Monkey-patching
# ==========================================
from ultralytics.nn.tasks import OBBModel

if not hasattr(OBBModel, "_original_forward"):
    OBBModel._original_forward = OBBModel.forward

def new_forward(self_model, x, *args, **kwargs):
    # Ultralytics sometimes passes a dict (training), sometimes a tensor (inference)
    if isinstance(x, dict):
        img = x["img"]
    else:
        img = x
        
    is_hsi = img.size(1) == 30
    has_adaptor = hasattr(self_model, "spectral_adaptor")

    if has_adaptor and is_hsi:
        adaptor = self_model.spectral_adaptor
        # Run adaptor in float32 for BatchNorm numerical stability.
        # Do NOT cast the adaptor module permanently to fp16 — it breaks BN running stats.
        was_half = next(adaptor.parameters()).dtype == torch.float16
        if was_half:
            adaptor = adaptor.float()
        img = adaptor(img.float())
        if was_half:
            adaptor = adaptor.half()
        # Ensure spatial dimension is 640x640 for YOLO backbone compatibility.
        if img.shape[-2:] != (640, 640):
            img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
        # Match output dtype to the backbone's dtype (e.g. half if the model is half)
        backbone_dtype = img.dtype
        for name, param in self_model.named_parameters():
            if "spectral_adaptor" not in name:
                backbone_dtype = param.dtype
                break
        img = img.to(backbone_dtype)

    if isinstance(x, dict):
        x = x.copy()
        x["img"] = img
    else:
        x = img

    # Pass to standard YOLO using the un-monkey-patched class-level forward method
    # AMP autocast handles dtype casting automatically for backbone ops.
    return self_model._original_forward(x, *args, **kwargs)

OBBModel.forward = new_forward

# =====================================================================
# Monkey-Patching OBBTrainer & OBBValidator for low-memory GPU Adaptor + Resize
# =====================================================================
from ultralytics.models.yolo.obb import OBBTrainer, OBBValidator

original_obb_trainer_preprocess = OBBTrainer.preprocess_batch
def custom_obb_trainer_preprocess(self_train, batch):
    img = batch["img"]
    is_hsi = img.shape[1] == 30

    if is_hsi:
        # HSI PCA data is already float32 in [-8, +8] range — do NOT divide by 255.
        # Dividing PCA values by 255 collapses them to near-zero, destroying all signal.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self_train.device, non_blocking=self_train.device.type == "cuda")
        batch["img"] = batch["img"].float()

        if hasattr(self_train, "model") and hasattr(self_train.model, "spectral_adaptor"):
            img = self_train.model.spectral_adaptor(batch["img"].float())
            if img.shape[-2:] != (640, 640):
                img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
            batch["img"] = img
    else:
        # RGB uint8 data: standard YOLO preprocessing (includes /255)
        batch = original_obb_trainer_preprocess(self_train, batch)
    return batch
OBBTrainer.preprocess_batch = custom_obb_trainer_preprocess

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
    img = batch["img"]
    is_hsi = img.shape[1] == 30

    if is_hsi:
        # HSI PCA data is already float32 — do NOT divide by 255.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self_val.device, non_blocking=self_val.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self_val.args.half else batch["img"].float())

        if hasattr(self_val, "model") and hasattr(self_val.model, "spectral_adaptor"):
            adaptor = self_val.model.spectral_adaptor
            # Run adaptor in float32 for BN stability, even if validator set model to half
            was_half = next(adaptor.parameters()).dtype == torch.float16
            if was_half:
                adaptor = adaptor.float()
            img = adaptor(batch["img"].float())
            if was_half:
                adaptor = adaptor.half()
            if img.shape[-2:] != (640, 640):
                img = torch.nn.functional.interpolate(img, size=(640, 640), mode='bilinear', align_corners=False)
            
            # Determine the rest of the model's dtype (backbone dtype)
            backbone_dtype = img.dtype
            for name, param in self_val.model.named_parameters():
                if "spectral_adaptor" not in name:
                    backbone_dtype = param.dtype
                    break
            batch["img"] = img.to(backbone_dtype)
    else:
        # RGB data: standard YOLO validation preprocessing (includes /255)
        batch = original_obb_validator_preprocess(self_val, batch)
    return batch
OBBValidator.preprocess = custom_obb_validator_preprocess

OBBModel.forward = new_forward

# ==========================================
# 1. Spectral-Specific Adaptor (1x1 2D CNN)
# ==========================================
class Spectral3DCNN(nn.Module):
    """
    Spectral Adaptor: 1x1 Conv2d + BatchNorm + Sigmoid.
    Maps 30-channel PCA-compressed HSI data (float, range ~[-8, +8])
    to 3-channel [0, 1] output for YOLO backbone.

    Why not clamp(0, 1)?
      PCA data is zero-centered with negative values.  clamp kills all negatives,
      and the small positive residuals (~0.02 after the old /255 bug) gave the
      backbone near-zero inputs.  BatchNorm standardizes the conv output, and
      Sigmoid provides a smooth, gradient-friendly mapping to [0, 1].

    Initialization: PC1→R, PC2→G, PC3→B as a warm start.
    """
    def __init__(self, in_channels=30, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            # Map PC1 -> R (channel 0)
            self.conv.weight[0, 0, 0, 0] = 1.0
            # Map PC2 -> G (channel 1)
            self.conv.weight[1, 1, 0, 0] = 1.0
            # Map PC3 -> B (channel 2)
            self.conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        # x shape: (B, 30, H, W) — raw PCA float values (NOT /255 normalized)
        x = self.conv(x)
        x = self.bn(x)
        return torch.sigmoid(x)  # Smooth [0, 1] for YOLO backbone

# ==========================================
# 2. Custom Trainer Wrapper
# ==========================================
class DualBranchMTTrainer(MeanTeacherOBBTrainer):
    
    def build_dataset(self, img_path, mode="train", batch=None):
        # The MT trainer builds both the source (RGB) and target (HSI) datasets.
        # We want source to be standard YOLO, and target to be HSIDataset.
        if "hsi" in str(img_path).lower() or "pca" in str(img_path).lower():
            print(f"Building HSIDataset for target/validation: {img_path} at 224x224 native resolution")
            return HSIDataset(
                img_path=img_path,
                imgsz=224,           # Load at native 224x224
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
        else:
            return super().build_dataset(img_path, mode, batch)


    def get_model(self, cfg=None, weights=None, verbose=True):
        # 1. Load the standard YOLO model (with pre-trained RGB weights)
        model = super().get_model(cfg, weights, verbose)
        
        # 2. Attach the Spectral Adaptor to the model
        model.spectral_adaptor = Spectral3DCNN(in_channels=30, out_channels=3).to(self.device)

        # 3. Manually load custom spectral_adaptor weights from the checkpoint if present
        if weights and str(weights).endswith('.pt') and os.path.exists(weights):
            try:
                ckpt = torch.load(weights, map_location=self.device)
                state_dict = ckpt.get('model', ckpt)
                if state_dict is not None and hasattr(state_dict, "state_dict"):
                    state_dict = state_dict.state_dict()
                
                # Check if 'spectral_adaptor' parameters are present in the checkpoint
                adaptor_state = {k.replace('spectral_adaptor.', ''): v for k, v in state_dict.items() if k.startswith('spectral_adaptor.')}
                if adaptor_state:
                    print(f"\n[+] FOUND spectral_adaptor weights in checkpoint: {weights}")
                    print(f"[+] Loading spectral_adaptor state dict: {list(adaptor_state.keys())}")
                    model.spectral_adaptor.load_state_dict(adaptor_state)
                else:
                    print(f"\n[-] No spectral_adaptor weights found in checkpoint: {weights} (using PC1-guided initialization)")
            except Exception as e:
                print(f"\n[!] Error loading spectral_adaptor weights from checkpoint: {e}")
                
        return model

    @torch.no_grad()
    def _generate_pseudo_labels(self, target_batch: dict, conf_thresh: float):
        """Run teacher on target HSI images and produce robust oriented pseudo-labels."""
        from ultralytics.utils.nms import non_max_suppression
        from torchvision.ops import nms as box_nms
        import numpy as np

        # 1. Dynamic Teacher-Aware Thresholding (Safe Curriculum)
        target_conf_thresh = conf_thresh
        if self.running_teacher_conf < 0.65:
            # If running confidence drops, RAISE the threshold to protect from hallucination/noise loops
            target_conf_thresh = max(conf_thresh, 0.65 - (self.running_teacher_conf - 0.50))
        target_conf_thresh = float(np.clip(target_conf_thresh, self.conf_end, self.conf_start))

        teacher = self.ema.ema  # YOLO's built-in EMA model (teacher)
        teacher.eval()

        # Preprocess target images (same as YOLO's preprocess_batch)
        # HSI PCA data is already float — do NOT /255 (that's for uint8 RGB only)
        imgs = target_batch["img"].to(self.device, non_blocking=True).float()

        # Teacher forward on clean HSI images
        with autocast(self.amp):
            teacher_out = teacher(imgs)

        # Determine if the model is end-to-end
        is_end2end = getattr(teacher, "end2end", False) or getattr(getattr(teacher, "model", None), "end2end", False)

        # Clean image NMS pass
        detections = non_max_suppression(
            teacher_out,
            conf_thres=target_conf_thresh,
            iou_thres=0.45,
            nc=1,
            max_det=50,           # cap pseudo-labels per HSI image
            max_time_img=0.5,     # timeout per image
            rotated=True,
            end2end=is_end2end,
            max_nms=1000,         # cap NMS candidates to prevent rotated polygon RAM spikes
        )

        # Clean image detections are used directly (TTA is disabled for fast oriented-NMS execution)
        combined_detections = detections

        # 3. Format detections with size/aspect filters
        batch_idx_list, cls_list, bboxes_list = [], [], []
        conf_list = []
        for i, dets in enumerate(combined_detections):
            if len(dets) == 0:
                continue

            valid_dets = []
            for det in dets:
                x_c, y_c, box_w, box_h, conf, cls, angle = det

                # Calculate normalized dimensions
                norm_w = box_w / self.args.imgsz
                norm_h = box_h / self.args.imgsz

                # relative area filter (max 15% of image area)
                relative_area = norm_w * norm_h
                if relative_area > 0.15:
                    continue

                # aspect ratio filter (max 8.0 to prevent skewed line hallucinations)
                if norm_h > 0 and norm_w > 0:
                    aspect_ratio = max(norm_w / norm_h, norm_h / norm_w)
                    if aspect_ratio > 8.0:
                        continue

                valid_dets.append(det)

            if len(valid_dets) == 0:
                continue

            valid_dets = torch.stack(valid_dets)
            bboxes = valid_dets[:, :4] / self.args.imgsz  # normalize to [0, 1]
            angle = valid_dets[:, 6:7]                     # radians
            cls = valid_dets[:, 5:6]

            batch_idx_list.append(torch.full((len(valid_dets), 1), i, device=self.device))
            cls_list.append(cls)
            bboxes_list.append(torch.cat([bboxes, angle], dim=1))
            conf_list.append(valid_dets[:, 4])  # Gather confidence scores

        if not bboxes_list:
            return None, 0, 0.0

        n_pls = sum(len(b) for b in bboxes_list)
        mean_conf = torch.cat(conf_list).mean().item()

        # Update teacher running confidence with moving average
        self.running_teacher_conf = 0.95 * self.running_teacher_conf + 0.05 * mean_conf

        pseudo_batch = {
            "cls": torch.cat(cls_list, dim=0),
            "bboxes": torch.cat(bboxes_list, dim=0),
            "batch_idx": torch.cat(batch_idx_list, dim=0).view(-1),
        }
        return pseudo_batch, n_pls, mean_conf

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):

        """Construct an optimizer with custom differential learning rates for UDA."""
        if name == "auto":
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        import torch.optim as optim
        from ultralytics.utils.torch_utils import unwrap_model
        from ultralytics.utils import colorstr, LOGGER

        # Partition parameters into Main group and Trainable Spectral group
        g_main_decay = []
        g_main_no_decay = []
        g_main_bias = []
        
        g_spectral_decay = []
        g_spectral_no_decay = []

        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                is_spectral = "spectral_adaptor" in fullname

                if is_spectral:
                    param.requires_grad = True  # Active!
                    # Put ALL spectral parameters in no_decay to prevent weights from shrinking to 0
                    g_spectral_no_decay.append(param)
                    continue

                if "bias" in fullname:
                    g_main_bias.append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g_main_no_decay.append(param)
                else:
                    g_main_decay.append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        opt_name = {x.lower(): x for x in optimizers}.get(name.lower(), "AdamW")

        # Main backbone/head parameters are optimized at full learning rate;
        # Spectral Adaptor is optimized at 10x slower rate (lr * 0.1) for extreme stability.
        lr_main = lr
        lr_spectral = lr * 0.1

        if opt_name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args_main = dict(lr=lr_main, betas=(momentum, 0.999), weight_decay=0.0)
            optim_args_spectral = dict(lr=lr_spectral, betas=(momentum, 0.999), weight_decay=0.0)
        elif opt_name == "RMSProp":
            optim_args_main = dict(lr=lr_main, momentum=momentum)
            optim_args_spectral = dict(lr=lr_spectral, momentum=momentum)
        elif opt_name == "SGD":
            optim_args_main = dict(lr=lr_main, momentum=momentum, nesterov=True)
            optim_args_spectral = dict(lr=lr_spectral, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not supported here.")

        param_groups = []
        if g_main_decay:
            param_groups.append({"params": g_main_decay, **optim_args_main, "weight_decay": decay, "param_group": "main_weight"})
        if g_main_no_decay:
            param_groups.append({"params": g_main_no_decay, **optim_args_main, "weight_decay": 0.0, "param_group": "main_bn"})
        if g_main_bias:
            param_groups.append({"params": g_main_bias, **optim_args_main, "param_group": "main_bias"})
            
        if g_spectral_decay:
            param_groups.append({"params": g_spectral_decay, **optim_args_spectral, "weight_decay": decay, "param_group": "spectral_weight"})
        if g_spectral_no_decay:
            param_groups.append({"params": g_spectral_no_decay, **optim_args_spectral, "weight_decay": 0.0, "param_group": "spectral_bn"})

        optimizer_cls = getattr(optim, opt_name)
        optimizer = optimizer_cls(param_groups)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with Trainable Spectral Adaptor + Spatial Backbone:\n"
            f"  - Spatial Backbone/Heads: lr={lr_main:.2e} (decay={len(g_main_decay)}, no_decay={len(g_main_no_decay)}, bias={len(g_main_bias)})\n"
            f"  - Spectral Adaptor (Unfrozen): lr={lr_spectral:.2e} (decay={len(g_spectral_decay)}, no_decay={len(g_spectral_no_decay)})"
        )
        return optimizer

    def _model_train(self):
        """Set model in training mode and ensure all parameters are active."""
        super()._model_train()
        
        # Keep both backbone/heads and spectral adaptor active for joint training
        active_count = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            active_count += 1

        # Log periodically to confirm settings
        from ultralytics.utils import LOGGER
        if getattr(self, "epoch", 0) % 5 == 0:
            LOGGER.info(
                f"Active Joint Training UDA (Epoch {self.epoch}): "
                f"Total active params = {active_count}"
            )


# ==========================================
# 3. Launch Script
# ==========================================
def run_integrated_training():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"

    print("\n" + "="*80)
    print("STARTING DUAL-BRANCH MT INTEGRATED TRAINING ON ALL HSI IMAGES")
    print("="*80 + "\n")

    # Source: RGB Fold (labeled) -> 3 channels (using fold 2 source for consistency)
    source_data = os.path.join(project_root, "baseline_official/yolo26_fold2_obb.yaml")

    # Target: Unified full HSI Training Set (all 988 images) -> 30 channels
    target_data = os.path.join(project_root, "baseline_official/yolo26_pca30_full.yaml")

    # Validation: PCA 30-channel labeled dataset (built by prepare_pca_30ch_eval.py)
    val_data = os.path.join(project_root, "baseline_official/yolo26_pca30_val.yaml")

    # Starting weights: Converged RGB model for this fold
    best_rgb_weights = os.path.join(
        project_root,
        "baseline_official/runs_yolo26s_obb/yolo26s_fold2_obb/weights/best.pt",
    )

    overrides = {
        "model": best_rgb_weights,
        "data": source_data,        # Source domain (supervised RGB)
        "epochs": 120,
        "imgsz": 640,
        "batch": 4,                 # Safe batch size of 4 to prevent CUDA OOM on RTX 2080 Ti
        "task": "obb",
        "device": 0,
        "single_cls": True,
        "project": os.path.join(project_root, "runs/mt_dual_branch"),
        "name": "dual_branch_mt_full",
        "exist_ok": True,
        "patience": 0,              # Train full duration
        "save_period": 10,
        "plots": False,             # Set to False to completely prevent plotting memory leaks and CPU/GPU overhead
        "cos_lr": True,
        "lr0": 0.0005,              # Sweet spot for UDA AdamW learning rate
        "optimizer": "AdamW",
        "workers": 0,               # Set to 0 to run loading in main process, completely eliminating OOM memory leaks and crashes
    }

    # Initialize our custom Dual Branch MT Trainer with stricter curriculum confidence bounds
    trainer = DualBranchMTTrainer(
        overrides=overrides,
        target_data=target_data,
        val_data=val_data,
        lambda_pseudo=0.40,         # Increased from 0.20 to 0.40 to provide a stronger learning signal from the target domain
        conf_start=0.40,            # Lowered from 0.60 to 0.40 to bootstrap early pseudo-label learning safely (protected by max_nms=1000)
        conf_end=0.25,              # Lowered from 0.40 to 0.25 to capture fine features as the model converges
        lambda_warmup_frac=0.0,     # Keep lambda active from epoch 0
        burn_in_epochs=0,
    )

    trainer.train()

if __name__ == "__main__":
    run_integrated_training()

