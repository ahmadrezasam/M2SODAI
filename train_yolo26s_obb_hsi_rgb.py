from ultralytics import YOLO
import os

# Paths
PROJECT_ROOT = "/home/ahmadreza/Downloads/Research/M2SODAI"
DATA_YAML = os.path.join(PROJECT_ROOT, "baseline_official", "hsi_rgb_640_v3_bilinear_obb.yaml")
MODEL_CHECKPOINT = "yolo26s-obb.pt"

def train_hsi_rgb():
    print(f"\n{'='*50}\nSTARTING TRAINING ON HSI-EXTRACTED RGB (v3-Bilinear)\n{'='*50}\n")
    
    # 1. Load the YOLO26 OBB model
    model = YOLO(MODEL_CHECKPOINT)

    # 2. Start training
    results = model.train(
        model=MODEL_CHECKPOINT,
        data=DATA_YAML,
        epochs=300,        # Standard training length
        imgsz=640,         # Match upscaled size
        batch=16,          # Adjust based on GPU memory
        device=0,          # GPU ID
        single_cls=True,   # Ship detection is single class
        project=os.path.join(PROJECT_ROOT, 'baseline_official/runs_yolo26'),
        name='yolo26s_hsi_rgb_v3_bilinear_obb',
        save=True,
        cache=True         # Cache images for faster training
    )

if __name__ == "__main__":
    train_hsi_rgb()
