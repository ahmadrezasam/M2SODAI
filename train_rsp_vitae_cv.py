import os
import subprocess
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train RSP ViTAE with Cross-Validation')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    return parser.parse_args()

def train_rsp_vitae_3fold():
    args = parse_args()
    
    # Path to the RSP/OBBDetection root directory
    rsp_root_dir = "." 
    
    # Where to save the results
    base_dir = "baseline_official"
    project_name = os.path.abspath(os.path.join(base_dir, 'runs_vitae_hbb'))
    
    folds = [1, 2, 3]

    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    for fold in folds:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR FOLD {fold} (ViTAE-Tiny)")
        print(f"{'='*50}\n")
        
        # Tiny Config Path
        config_path = f"configs/faster_rcnn/fold_{fold}_vitae_t_3x_hbb.py"
        
        # Directory to save the weights and logs for this fold
        work_dir = os.path.join(project_name, f"vitae_t_hbb_fold_{fold}")
        
        # The training command using torch.distributed.launch
        train_cmd = [
            "python", "-m", "torch.distributed.launch", 
            "--nproc_per_node=1", 
            "--master_port=50011", 
            os.path.join(rsp_root_dir, "tools/train.py"),
            config_path,
            "--work-dir", work_dir,
            "--launcher", "pytorch",
            "--cfg-options", 
            f"runner.max_epochs={args.epochs}",
            f"data.samples_per_gpu={args.batch_size}",
            f"evaluation.interval={args.eval_interval}"
        ]
        
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
        
        # --- Memory Cleanup Logic ---
        # 1. Subprocess exit automatically clears GPU VRAM.
        # 2. We wait a bit longer to ensure the GPU driver has fully released all handles.
        # 3. We trigger Python's garbage collector for the orchestrator itself.
        print(f"Cleaning up memory and cooling down for 15 seconds...")
        import gc
        gc.collect() 
        time.sleep(15)

if __name__ == "__main__":
    train_rsp_vitae_3fold()
