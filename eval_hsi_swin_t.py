import os
import subprocess

def eval_hsi():
    runs_dir = "baseline_official/runs_swin_hbb"
    hsi_dir = "baseline_official/test_hsi_rgb_640_v3_bilinear"
    ann_file = os.path.join(hsi_dir, "annotations.json")
    img_prefix = os.path.join(hsi_dir, "images/")
    
    folds = [1, 2, 3]
    output_dir = "baseline_official/eval_results_hsi"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUNNING HSI DOMAIN GENERALIZATION TEST (125 Images)")
    print(f"{'='*60}\n")

    for fold in folds:
        print(f"\n--- Testing Fold {fold} on HSI dataset ---")
        
        config = f"configs/faster_rcnn/fold_{fold}_swin_t_3x_hbb.py"
        checkpoint = os.path.join(runs_dir, f"swin_t_hbb_fold_{fold}/latest.pth")
        
        if not os.path.exists(checkpoint):
            print(f"Skipping Fold {fold} (checkpoint not found)")
            continue

        python_path = "/home/ahmadreza/miniconda3/envs/Maritime/bin/python"
        cmd = [
            python_path, "./tools/test.py",
            config,
            checkpoint,
            "--eval", "bbox",
            "--cfg-options",
            f'data.test.ann_file="{ann_file}"',
            f'data.test.img_prefix="{img_prefix}"'
        ]
        
        print(f"Executing evaluation...")
        subprocess.run(cmd)

if __name__ == "__main__":
    eval_hsi()
