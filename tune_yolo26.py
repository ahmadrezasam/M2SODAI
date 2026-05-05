from ultralytics import YOLO
import os

# Configuration
base_dir = '/home/ahmadreza/Downloads/Research/M2SODAI'

# Weights - Start from a standard OBB model
# Using yolo11s-obb as it matches the size/complexity of yolo26s
model_path = 'yolo26s-obb.pt' 

# Data
data_yaml = os.path.join(base_dir, 'baseline_official/yolo26_fold1_obb.yaml')

# Load the model
model = YOLO(model_path)

# Start Tuning
# This will run multiple training iterations to evolve the best augmentations
# Results will be saved to baseline_official/tuning/yolo26_obb_tune
model.tune(
    data=data_yaml,
    epochs=50,        # Sufficient epochs to see convergence
    iterations=30,    # Number of genetic evolutions
    imgsz=640,
    batch=16,
    device=0,
    single_cls=True,
    optimizer='AdamW',
    project=os.path.join(base_dir, 'baseline_official/tuning'),
    name='yolo26_obb_tune',
    use_ray=False     # Use the built-in Genetic Algorithm
)
