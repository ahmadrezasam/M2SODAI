from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

# Define paths
base_dir = '/home/ahmadreza/Downloads/Research/M2SODAI'
weights_dir = os.path.join(base_dir, 'baseline_official/runs_yolo26s')
test_yaml = os.path.join(base_dir, 'baseline_official/test_hsi_rgb_640_v3_bilinear.yaml')
project_dir = os.path.join(base_dir, 'baseline_official/eval_results')

# Folds to evaluate
folds = [1, 2, 3]

all_results = []

for fold in folds:
    print(f"\n{'='*40}")
    print(f"EVALUATING FOLD {fold}")
    print(f"{'='*40}\n")
    
    # Path to best weights for this fold
    weights_path = os.path.join(weights_dir, f'yolo26s_fold{fold}/weights/best.pt')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        continue
        
    # Load the model
    model = YOLO(weights_path)
    
    # Run validation on the test set
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=project_dir,
        name=f'eval_test_fold{fold}_v3_bilinear'
    )
    
    # Extract metrics
    # results.results_dict contains the metrics
    map50 = results.results_dict['metrics/mAP50(B)']
    map50_95 = results.results_dict['metrics/mAP50-95(B)']
    
    all_results.append({
        'Fold': fold,
        'mAP50': map50,
        'mAP50-95': map50_95
    })
    
    print(f"\nFold {fold} evaluation completed.")

# Summary Table
if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*40}")
    print("SUMMARY RESULTS")
    print(f"{'='*40}")
    print(df.to_string(index=False))
    print(f"{'-'*40}")
    
    mean_map50 = df['mAP50'].mean()
    std_map50 = df['mAP50'].std()
    mean_map50_95 = df['mAP50-95'].mean()
    std_map50_95 = df['mAP50-95'].std()
    
    print(f"Mean mAP50:     {mean_map50:.4f} ± {std_map50:.4f}")
    print(f"Mean mAP50-95:  {mean_map50_95:.4f} ± {std_map50_95:.4f}")
    print(f"{'='*40}")
