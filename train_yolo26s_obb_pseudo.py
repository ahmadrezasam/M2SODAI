from ultralytics import YOLO
import os

def fine_tune_uda():
    print(f"\n{'='*40}\nSTARTING UDA FINE-TUNING FOR FOLD 1\n{'='*40}\n")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    
    # 1. Load the pre-trained model from Fold 1 (source domain)
    best_weights = os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb', 'yolo26s_fold1_obb', 'weights', 'best.pt')
    
    if not os.path.exists(best_weights):
        print(f"Error: Could not find pre-trained weights at {best_weights}")
        return
        
    model = YOLO(best_weights)

    # 2. Fine-tune on the combined dataset (RGB + Pseudo-labeled HSI)
    data_yaml = os.path.join(project_root, 'baseline_official', 'yolo26_fold1_obb_pseudo.yaml')
    
    results = model.train(
        data=data_yaml,
        epochs=100,           # 100 epochs is usually sufficient for fine-tuning UDA
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda'),
        name='yolo26s_fold1_obb_pseudo'
    )
    
    print("\nUDA Fine-Tuning Completed!")

if __name__ == "__main__":
    fine_tune_uda()
