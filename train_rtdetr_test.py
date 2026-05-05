import os
from ultralytics import RTDETR

def train_rtdetr_3fold():
    # Configuration
    # Use relative path for baseline_official as it's in the current directory
    base_dir = "baseline_official" 
    
    # Use the supported RT-DETR Large model (HGNetv2 based)
    # Note: rtdetr-l (66MB) is actually smaller and more efficient than r18vd (81MB)
    model_name = "rtdetr-l.pt" 
    
    # Project results will be saved in baseline_official/runs_rtdetr
    project_name = os.path.join(base_dir, 'runs_rtdetr')
    
    epochs = 1
    imgsz = 640
    batch_size = 16 # Adjust if you encounter OOM

    # Folds to train
    folds = [1]

    # Create the project directory if it doesn't exist
    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    for fold in folds:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR FOLD {fold}")
        print(f"{'='*50}\n")

        # Path to the data yaml for this fold
        data_yaml = os.path.abspath(os.path.join(base_dir, f"fold_{fold}.yaml"))
        
        if not os.path.exists(data_yaml):
            print(f"Warning: {data_yaml} not found. Skipping...")
            continue

        # Initialize the RT-DETR model
        model = RTDETR(model_name)

        # Start training
        # RT-DETR is a transformer and benefits from AdamW and a stable learning rate
        model.train(
            data=data_yaml,
            epochs=epochs,
            patience=100,
            imgsz=imgsz,
            batch=batch_size,
            project=project_name,
            name=f"rtdetr_l_fold_{fold}",
            device=0,  # Use 0 for GPU
            exist_ok=True,
            optimizer='AdamW', 
            lr0=0.0001,        # Recommended for RT-DETR
            seed=42,
            save=True,
            val=True
        )

        print(f"\nFinished training Fold {fold}.\n")

if __name__ == "__main__":
    train_rtdetr_3fold()

