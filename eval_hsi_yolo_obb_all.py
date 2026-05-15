from ultralytics import YOLO
import os

def eval_hsi_yolo_obb():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    test_yaml = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb.yaml')
    
    models = [
        {
            "name": "YOLO OBB Fold 1 (Base RGB)",
            "weights": os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb', 'yolo26s_fold1_obb', 'weights', 'best.pt')
        },
        {
            "name": "YOLO OBB Fold 1 (UDA Round 1)",
            "weights": os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda', 'yolo26s_fold1_obb_pseudo', 'weights', 'best.pt')
        },
        {
            "name": "YOLO OBB Fold 1 (UDA Round 2)",
            "weights": os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda', 'yolo26s_fold1_obb_pseudo2', 'weights', 'best.pt')
        }
    ]

    print(f"\n{'='*60}")
    print(f"EVALUATING YOLO OBB ON HSI TEST SET")
    print(f"{'='*60}\n")

    for m in models:
        print(f"\n>>> Testing {m['name']}...")
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
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        print(f"Result for {m['name']}:")
        print(f"  mAP50:     {map50:.4f}")
        print(f"  mAP50-95:  {map50_95:.4f}")

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    eval_hsi_yolo_obb()
