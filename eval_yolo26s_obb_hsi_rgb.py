from ultralytics import YOLO
import os
import pandas as pd

# Define paths
PROJECT_ROOT = '/home/ahmadreza/Downloads/Research/M2SODAI'
# The directory where the training results are stored
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'baseline_official/runs_yolo26/yolo26s_hsi_rgb_v3_bilinear_obb/weights/best.pt')
# The test YAML for the HSI OBB test set
TEST_YAML = os.path.join(PROJECT_ROOT, 'baseline_official/test_hsi_rgb_640_v3_bilinear_obb.yaml')
# Evaluation results directory
PROJECT_DIR = os.path.join(PROJECT_ROOT, 'baseline_official/eval_results')

def eval_hsi_rgb():
    print(f"\n{'='*50}")
    print(f"EVALUATING YOLO26-OBB ON HSI TEST SET (v3-Bilinear)")
    print(f"{'='*50}\n")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        print("Please ensure training is complete before running evaluation.")
        return
        
    # 1. Load the trained model
    model = YOLO(WEIGHTS_PATH)
    
    # 2. Run validation on the test set
    results = model.val(
        data=TEST_YAML,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=PROJECT_DIR,
        name='eval_hsi_rgb_v3_bilinear_obb'
    )
    
    # 3. Extract and print metrics
    metrics = results.results_dict
    # For OBB, metrics usually have (B) for box or (M) for mask/obb depending on version
    map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50(M)', 0))
    map50_95 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95(M)', 0))
    
    print(f"\n{'='*40}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"mAP@50:     {map50:.4f}")
    print(f"mAP@50-95:  {map50_95:.4f}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    eval_hsi_rgb()
