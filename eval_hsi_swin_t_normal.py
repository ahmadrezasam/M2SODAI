import os
import subprocess

def eval_hsi_swin_t_normal():
    hsi_dir = "baseline_official/test_hsi_rgb_640_v3_bilinear"
    ann_file = os.path.abspath(os.path.join(hsi_dir, "annotations.json"))
    img_prefix = os.path.abspath(os.path.join(hsi_dir, "images/"))
    
    runs_dir = "baseline_official/runs_swin_hbb"
    folds = [1, 2, 3]
    
    python_path = "/home/ahmadreza/miniconda3/envs/Maritime/bin/python"

    print(f"\n{'='*60}")
    print(f"RUNNING HSI EVALUATION FOR SWIN-T (NORMAL FOLDS)")
    print(f"{'='*60}\n")

    for fold in folds:
        print(f"\n>>> Testing Swin-T Fold {fold}...")
        
        # Determine checkpoint path
        fold_dir = os.path.join(runs_dir, f"swin_t_hbb_fold_{fold}")
        
        # Use best checkpoint if exists, otherwise latest.pth
        if fold == 1:
            checkpoint = os.path.join(fold_dir, "best_bbox_mAP_epoch_250.pth")
        else:
            checkpoint = os.path.join(fold_dir, "latest.pth")
            
        config = os.path.join(fold_dir, f"fold_{fold}_swin_t_3x_hbb.py")
        
        if not os.path.exists(checkpoint):
            print(f"Error: Checkpoint not found at {checkpoint}")
            continue
        if not os.path.exists(config):
            print(f"Error: Config not found at {config}")
            continue

        cmd = [
            python_path, "./tools/test.py",
            config,
            checkpoint,
            "--eval", "bbox",
            "--cfg-options",
            f'data.test.ann_file="{ann_file}"',
            f'data.test.img_prefix="{img_prefix}"'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print results
        output = process.stdout
        print(output[-600:])
        
        if process.returncode != 0:
            print(f"Error running test for Fold {fold}")
            print(process.stderr)

    print(f"\n{'='*60}")
    print(f"SWIN-T NORMAL EVALUATION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    eval_hsi_swin_t_normal()
