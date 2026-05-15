import os
import subprocess
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN R50 with Cross-Validation')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    return parser.parse_args()

def train_r50_cv():
    args = parse_args()
    
    rsp_root_dir = "." 
    base_dir = "baseline_official"
    project_name = os.path.abspath(os.path.join(base_dir, 'runs_r50_hbb'))
    
    folds = [1]

    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    for fold in folds:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR FOLD {fold} (Faster R-CNN R50)")
        print(f"{'='*50}\n")
        
        from datetime import datetime
        current_date = datetime.now().strftime('%Y%m%d')
        config_path = f"configs/faster_rcnn/fold_{fold}_r50_300e_hbb.py"
        work_dir = os.path.join(project_name, f"r50_hbb_fold_{fold}_{current_date}")
        
        latest_ckpt = os.path.join(work_dir, 'latest.pth')
        resume_arg = []
        if os.path.exists(latest_ckpt):
            print(f"Found existing checkpoint at {latest_ckpt}. Resuming...")
            resume_arg = ["--resume-from", latest_ckpt]
        
        python_path = "/home/ahmadreza/miniconda3/envs/Maritime/bin/python"
        
        train_cmd = [
            python_path, "-m", "torch.distributed.launch", 
            "--nproc_per_node=1", 
            "--master_port=50016", 
            os.path.join(rsp_root_dir, "tools/train.py"),
            config_path,
            "--work-dir", work_dir,
            "--launcher", "pytorch",
            "--cfg-options", 
            f"runner.max_epochs={args.epochs}",
            f"data.samples_per_gpu={args.batch_size}",
            f"evaluation.interval={args.eval_interval}"
        ] + resume_arg
        
        print(f"Executing command:\n{' '.join(train_cmd)}\n")
        
        if not os.path.exists(config_path):
            print(f"WARNING: Config file not found: {config_path}")
            continue
            
        try:
            subprocess.run(train_cmd, check=True)
            print(f"\nFinished training Fold {fold} successfully.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError during training Fold {fold}: {e}")
            break
        
        print(f"Cleaning up memory and cooling down for 15 seconds...")
        import gc
        gc.collect() 
        time.sleep(15)

if __name__ == "__main__":
    train_r50_cv()
