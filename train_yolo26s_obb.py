from ultralytics import YOLO

# Loop through all 3 folds
for fold_num in [2]:
    print(f"\n{'='*30}\nSTARTING TRAINING FOR FOLD {fold_num}\n{'='*30}\n")
    
    # 1. Load a fresh pretrained model for each fold
    model = YOLO('yolo26s-obb.pt')

    # 2. Start training with fold-specific paths
    results = model.train(
        model='yolo26s-obb.pt',
        data=f'/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/yolo26_fold{fold_num}_obb.yaml',
        # cfg='/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/hyp_yolo26_obb.yaml',
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project='/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/runs_yolo26',
        name=f'yolo26s_fold{fold_num}_obb'
    )
