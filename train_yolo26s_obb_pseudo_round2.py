from ultralytics import YOLO
import os

def fine_tune_uda_round2():
    print(f"\n{'='*50}\nSTARTING UDA ROUND 2 FINE-TUNING FOR FOLD 1\n{'='*50}\n")
    
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    
    # We start from the ORIGINAL RGB weights to prevent confirmation bias compounding
    # Alternatively, you could start from the round 1 adapted weights.
    best_weights = os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb', 'yolo26s_fold1_obb', 'weights', 'best.pt')
    
    if not os.path.exists(best_weights):
        print(f"Error: Could not find pre-trained weights at {best_weights}")
        return
        
    model = YOLO(best_weights)

    # Fine-tune on the Round 2 combined dataset
    data_yaml = os.path.join(project_root, 'baseline_official', 'yolo26_fold1_obb_pseudo_round2.yaml')
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        single_cls=True,
        project=os.path.join(project_root, 'baseline_official', 'runs_yolo26s_obb_uda'),
        name='yolo26s_fold1_obb_pseudo_round2'
    )
    
    print("\nUDA Round 2 Fine-Tuning Completed!")

if __name__ == "__main__":
    fine_tune_uda_round2()
