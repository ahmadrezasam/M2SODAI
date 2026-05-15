import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Swin-T Cross-Validation Folds')
    parser.add_argument('--out_dir', type=str, default='baseline_official/eval_results_swin', help='Directory to save results')
    return parser.parse_args()

def eval_swin_cv():
    args = parse_args()
    
    base_dir = "baseline_official"
    runs_dir = os.path.join(base_dir, 'runs_swin_hbb')
    folds = [1, 2, 3]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STARTING EVALUATION FOR {len(folds)} FOLDS")
    print(f"{'='*60}\n")

    for fold in folds:
        print(f"\n--- Evaluating Fold {fold} ---")
        
        config_path = f"configs/faster_rcnn/fold_{fold}_swin_t_3x_hbb.py"
        work_dir = os.path.join(runs_dir, f"swin_t_hbb_fold_{fold}")
        
        # Priority 1: Best mAP checkpoint
        # Priority 2: Latest checkpoint
        checkpoint_path = os.path.join(work_dir, 'best_bbox_mAP.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(work_dir, 'latest.pth')
            
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: No checkpoint found for Fold {fold} in {work_dir}")
            continue

        print(f"Using checkpoint: {checkpoint_path}")
        
        output_pkl = os.path.join(args.out_dir, f"fold_{fold}_results.pkl")
        
        # MMDetection test command
        test_cmd = [
            "python", "./tools/test.py",
            config_path,
            checkpoint_path,
            "--eval", "bbox",
            "--out", output_pkl
        ]
        
        print(f"Executing: {' '.join(test_cmd)}")
        
        try:
            subprocess.run(test_cmd, check=True)
            print(f"Successfully evaluated Fold {fold}.")
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating Fold {fold}: {e}")

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE. Results saved to {args.out_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    eval_swin_cv()
