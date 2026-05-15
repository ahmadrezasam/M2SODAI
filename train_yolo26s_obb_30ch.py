from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBTrainer
from hsi_dataset import HSIDataset
import os

class HSITrainer(OBBTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return HSIDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.0,
            data=self.data,
            classes=self.args.classes,
            fraction=self.args.fraction
        )

def train_hsi():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    model_yaml = os.path.join(project_root, 'baseline_official', 'yolo26s-obb-30ch.yaml')
    
    # Loop through folds (starting with fold 2 as in original script)
    for fold_num in [2]:
        print(f"\n{'='*30}\nSTARTING HSI TRAINING FOR FOLD {fold_num}\n{'='*30}\n")
        
        data_yaml = os.path.join(project_root, 'baseline_official', f'yolo26_fold{fold_num}_hsi.yaml')
        
        # Initialize a fresh 30-channel model
        # We use the YAML but ensure ch=30 is passed to the underlying model
        model = YOLO(model_yaml, task='obb')
        
        # Verify and fix in_channels if needed (Ultralytics sometimes defaults to 3)
        if model.model.model[0].conv.in_channels != 30:
            print("Fixing in_channels to 30...")
            from ultralytics.nn.tasks import OBBModel
            model.model = OBBModel(model_yaml, ch=30).to(model.device)
        
        # Start training
        # We use the custom trainer to handle .npy and 30 channels
        model.train(
            data=data_yaml,
            epochs=300,
            imgsz=640,
            batch=16,
            device=0,
            single_cls=True,
            project=os.path.join(project_root, 'baseline_official', 'runs_hsi_upperbound'),
            name=f'yolo26s_fold{fold_num}_hsi_30ch',
            trainer=HSITrainer,
            # Disable HSV augmentations for HSI
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0
        )

if __name__ == "__main__":
    train_hsi()
