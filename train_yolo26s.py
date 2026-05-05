from ultralytics import YOLO

# Loop through all 3 folds
for fold_num in [3]:
    print(f"\n{'='*30}\nSTARTING TRAINING FOR FOLD {fold_num}\n{'='*30}\n")
    
    # 1. Load a standard pretrained YOLOv26s model
    # This model natively outputs 4 box coordinates
    model = YOLO('yolo26s.pt')

    # 2. Start training with standard fold configurations
    results = model.train(
        model='yolo26s.pt',
        data=f'/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/fold_{fold_num}.yaml',
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        seed=0, 
        project='/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/runs_yolo26s',
        name=f'yolo26s_fold{fold_num}_hbb'
    )

    print(f"\nFold {fold_num}  training completed.\n")
