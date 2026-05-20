from ultralytics import YOLO
import os

def test():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    model_path = os.path.join(project_root, "runs/mt_integrated/yolo26s_mt_v9_static_conf/weights/best.pt")
    test_data = os.path.join(project_root, "baseline_official/test_hsi_rgb_640_v3_bilinear_obb.yaml")

    model = YOLO(model_path)
    
    print(f"Testing model: {model_path}")
    print(f"On dataset: {test_data}")
    
    results = model.val(
        data=test_data,
        split='val', # The test YAML uses 'val' split for its images
        device=0,
        imgsz=640,
        task='obb'
    )
    
    print("\nFinal Test Results on HSI:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")

if __name__ == "__main__":
    test()
