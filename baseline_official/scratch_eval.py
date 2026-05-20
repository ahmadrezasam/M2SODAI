from ultralytics import YOLO
import os
import pandas as pd

base_dir = '/home/ahmadreza/Downloads/Research/M2SODAI'
weights_dir = os.path.join(base_dir, 'baseline_official/runs_yolo26s_obb')
test_yaml = os.path.join(base_dir, 'baseline_official/hsi_rgb_640_v3_bilinear_obb.yaml')
project_dir = os.path.join(base_dir, 'baseline_official/eval_results')

folds = [1, 2, 3]
all_results = []

for fold in folds:
    weights_path = os.path.join(weights_dir, f'yolo26s_fold{fold}_obb/weights/best.pt')
    if not os.path.exists(weights_path):
        continue
    model = YOLO(weights_path)
    results = model.val(
        data=test_yaml,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=project_dir,
        name=f'eval_scratch_fold{fold}',
        verbose=False,
        plots=False
    )
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50(M)', 0))
    map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95(M)', 0))
    all_results.append({
        'Fold': fold,
        'mAP50': map50,
        'mAP50-95': map50_95
    })

print("SCRATCH RESULTS:")
for r in all_results:
    print(f"Fold {r['Fold']}: mAP50 = {r['mAP50']:.4f}")
