from ultralytics import YOLO
import os
import numpy as np

def eval_mt_all_folds():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    test_yaml = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb.yaml')
    
    models = [
        {
            "fold": 1,
            "name": "Mean Teacher UDA Fold 1 (Adaptive)",
            "weights": os.path.join(project_root, 'runs', 'mt_integrated', 'yolo26s_mt_fold1_adaptive', 'weights', 'best.pt')
        },
        {
            "fold": 2,
            "name": "Mean Teacher UDA Fold 2 (Adaptive)",
            "weights": os.path.join(project_root, 'runs', 'mt_integrated', 'yolo26s_mt_fold2_adaptive', 'weights', 'best.pt')
        },
        {
            "fold": 3,
            "name": "Mean Teacher UDA Fold 3 (Adaptive)",
            "weights": os.path.join(project_root, 'runs', 'mt_integrated', 'yolo26s_mt_fold3_adaptive', 'weights', 'best.pt')
        }
    ]

    print(f"\n{'='*70}")
    print(f"EVALUATING MEAN TEACHER UDA ALL FOLDS ON HSI TEST SET")
    print(f"{'='*70}\n")

    results_list = []
    
    for m in models:
        print(f"\n>>> Evaluating {m['name']}...")
        if not os.path.exists(m['weights']):
            print(f"Skipping: Weights not found at {m['weights']}")
            continue
            
        model = YOLO(m['weights'])
        results = model.val(
            data=test_yaml,
            imgsz=640,
            batch=16,
            device=0,
            single_cls=True,
            plots=False,
            save_json=False,
            verbose=False
        )
        
        metrics = results.results_dict
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        results_list.append({
            "fold": m['fold'],
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95
        })
        
        print(f"Result for {m['name']}:")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")

    if len(results_list) > 0:
        precisions = [r['precision'] for r in results_list]
        recalls = [r['recall'] for r in results_list]
        map50s = [r['map50'] for r in results_list]
        map50_95s = [r['map50_95'] for r in results_list]
        
        print(f"\n{'='*70}")
        print(f"SUMMARY OF 3-FOLD CROSS-VALIDATION ON HSI TEST SET")
        print(f"{'='*70}")
        print(f"{'Fold':<10} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
        print(f"{'-'*70}")
        for r in results_list:
            print(f"Fold {r['fold']:<5} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['map50']:<10.4f} | {r['map50_95']:<10.4f}")
        print(f"{'-'*70}")
        print(f"Mean       | {np.mean(precisions):<10.4f} | {np.mean(recalls):<10.4f} | {np.mean(map50s):<10.4f} | {np.mean(map50_95s):<10.4f}")
        print(f"Std        | {np.std(precisions):<10.4f} | {np.std(recalls):<10.4f} | {np.std(map50s):<10.4f} | {np.std(map50_95s):<10.4f}")
        print(f"{'='*70}\n")
    else:
        print("No results evaluated.")

if __name__ == "__main__":
    eval_mt_all_folds()
