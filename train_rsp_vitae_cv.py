import os
import subprocess
import time

def train_rsp_vitae_3fold():
    """
    This script automates the 3-fold cross-validation training for RSP-ViTAEv2-S
    using the OBBDetection/RSP framework, matching the logic of train_rtdetr_cv.py.
    
    IMPORTANT PREREQUISITES:
    1. You must clone the RSP repository and install OBBDetection (mmcv, mmdet, BboxToolkit).
    2. You must convert your fold_1, fold_2, fold_3 data into DOTA format 
       (since OBBDetection expects DOTA-style labels, not YOLO txt files).
    3. You must create 3 config files (one for each fold) based on the original:
       'configs/obb/oriented_rcnn/faster_rcnn_orpn_our_rsp_vitae_fpn_3x_hrsc.py'
    """
    
    # Path to the RSP/OBBDetection root directory (User must update this)
    # Example: '/home/ahmadreza/Downloads/Research/RSP/Object Detection'
    rsp_root_dir = "./RSP_Object_Detection" 
    
    # Where to save the results
    base_dir = "baseline_official"
    project_name = os.path.abspath(os.path.join(base_dir, 'runs_rsp_vitae'))
    
    epochs = 36 # In mmdet, 3x schedule is usually 36 epochs. 
    
    folds = [1, 2, 3]

    if not os.path.exists(project_name):
        os.makedirs(project_name, exist_ok=True)

    for fold in folds:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR FOLD {fold}")
        print(f"{'='*50}\n")
        
        # In MMDet/OBBDetection, everything is controlled by a config file.
        # You must create these config files manually where data paths point to Fold 1, 2, or 3.
        config_path = os.path.join(rsp_root_dir, f"configs/obb/oriented_rcnn/fold_{fold}_rsp_vitae_fpn_3x.py")
        
        # Directory to save the weights and logs for this fold
        work_dir = os.path.join(project_name, f"vitae_s_fold_{fold}")
        
        # The training command using torch.distributed.launch (as required by RSP)
        # We use subprocess to call the training tool
        train_cmd = [
            "python", "-m", "torch.distributed.launch", 
            "--nproc_per_node=1", # Change to number of GPUs you have
            "--master_port=50002", 
            os.path.join(rsp_root_dir, "tools/train.py"),
            config_path,
            "--work-dir", work_dir,
            "--launcher", "pytorch",
            "--options", "find_unused_parameters=True"
        ]
        
        print(f"Executing command:\n{' '.join(train_cmd)}\n")
        
        # Wait for user to ensure config exists (optional safeguard)
        if not os.path.exists(config_path):
            print(f"WARNING: Config file not found: {config_path}")
            print("Please create the config file pointing to this fold's dataset.")
            break
            
        try:
            # Run the training process
            subprocess.run(train_cmd, check=True)
            print(f"\nFinished training Fold {fold} successfully.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError during training Fold {fold}: {e}")
            break
            
        # Give the system a moment to clean up resources
        time.sleep(5)

if __name__ == "__main__":
    print("NOTE: OBBDetection does not use a simple Python API like Ultralytics.")
    print("This script uses subprocess to call the training tools.")
    train_rsp_vitae_3fold()
