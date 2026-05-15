import os
import subprocess
import json

def eval_hsi_r50():
    hsi_dir = "baseline_official/test_hsi_rgb_640_v3_bilinear"
    ann_file = os.path.abspath(os.path.join(hsi_dir, "annotations.json"))
    img_prefix = os.path.abspath(os.path.join(hsi_dir, "images/"))
    
    python_path = "/home/ahmadreza/miniconda3/envs/Maritime/bin/python"
    
    tasks = [
        {
            "name": "Fold 1 (No YOLO Aug)",
            "config": "configs/faster_rcnn/fold_1_r50_300e_hbb.py", # This was updated to YOLO aug later, but I can override pipeline if needed. 
            # Actually, I'll use the 20260510 run's config which was v1.
            "checkpoint": "baseline_official/runs_r50_hbb/r50_hbb_fold_1_20260510/best_bbox_mAP_epoch_60.pth"
        },
        {
            "name": "Fold 2 (With YOLO Aug)",
            "config": "configs/faster_rcnn/fold_2_r50_300e_hbb.py",
            "checkpoint": "baseline_official/runs_r50_hbb/r50_hbb_fold_2_20260511/best_bbox_mAP_epoch_240.pth"
        },
        {
            "name": "Fold 3 (With YOLO Aug)",
            "config": "configs/faster_rcnn/fold_3_r50_300e_hbb.py",
            "checkpoint": "baseline_official/runs_r50_hbb/r50_hbb_fold_3_20260511/best_bbox_mAP_epoch_260.pth"
        }
    ]

    print(f"\n{'='*60}")
    print(f"RUNNING HSI EVALUATION FOR RESNET-50 R-CNN")
    print(f"{'='*60}\n")

    results_all = {}

    for task in tasks:
        print(f"\n>>> Testing {task['name']}...")
        
        if not os.path.exists(task['checkpoint']):
            print(f"Error: Checkpoint not found at {task['checkpoint']}")
            continue

        # We need to ensure Fold 1 evaluation uses the correct pipeline if the config file was changed.
        # However, for testing, standard test_pipeline is usually consistent.
        
        cmd = [
            python_path, "./tools/test.py",
            task['config'],
            task['checkpoint'],
            "--eval", "bbox",
            "--cfg-options",
            f'data.test.ann_file="{ann_file}"',
            f'data.test.img_prefix="{img_prefix}"'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract mAP from output
        # MMDetection prints a dict-like string at the end: OrderedDict([('bbox_mAP', 0.07), ...])
        output = process.stdout
        print(output[-500:]) # Print last bit of output
        
        if process.returncode != 0:
            print(f"Error running test for {task['name']}")
            print(process.stderr)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    eval_hsi_r50()
