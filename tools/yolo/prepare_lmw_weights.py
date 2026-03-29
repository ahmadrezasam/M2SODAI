import os
import sys

def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lmw_yolo_dir = os.path.join(project_root, "mmdet/models/LMW-YOLO")
    
    # Add LMW-YOLO directory to sys.path so that custom modules can be found
    if lmw_yolo_dir not in sys.path:
        sys.path.insert(0, lmw_yolo_dir)

    # Now import ultralytics and torch
    # By inserting at 0, we prefer the local clone in mmdet/models/LMW-YOLO
    import torch
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    from ultralytics.modifiednn.modules.block import LKCA, MSDP
    
    # Registering them in the globals of ultralytics.nn.tasks
    setattr(tasks, 'LKCA', LKCA)
    setattr(tasks, 'MSDP', MSDP)
    
    yaml_path = os.path.join(lmw_yolo_dir, "ultralytics/cfg/models/v11/LMW-YOLO.yaml")
    output_path = os.path.join(project_root, "yolo11n_lmw_init.pt")
    
    print(f"Loading official YOLO11n weights...")
    base = YOLO("yolo11n.pt")
    
    print(f"Loading LMW-YOLO architecture from {yaml_path}...")
    if not os.path.exists(yaml_path):
        print(f"Error: Architecture file not found at {yaml_path}")
        return

    # Use the absolute path to the yaml file
    # YOLO() might need to be run from the directory containing 'ultralytics' folder if it's not installed
    # But since 'ultralytics' IS installed, we might need to make sure the yaml is compatible.
    try:
        lmw = YOLO(yaml_path)
    except Exception as e:
        print(f"Error loading LMW-YOLO model: {e}")
        print("Attempting to load from the LMW-YOLO directory...")
        # Try changing directory to mmdet/models/LMW-YOLO
        old_cwd = os.getcwd()
        os.chdir(lmw_yolo_dir)
        try:
            lmw = YOLO("ultralytics/cfg/models/v11/LMW-YOLO.yaml")
        finally:
            os.chdir(old_cwd)

    # Transfer matching weights, skip mismatched layers (LKCA, MSDP)
    base_state  = base.model.state_dict()
    lmw_state   = lmw.model.state_dict()

    transferred = {}
    skipped     = []

    for k, v in lmw_state.items():
        if k in base_state and base_state[k].shape == v.shape:
            transferred[k] = base_state[k]
        else:
            transferred[k] = v          # keep random init for new modules
            skipped.append(k)

    lmw.model.load_state_dict(transferred)
    
    print(f"Transferred: {len(transferred) - len(skipped)} layers")
    print(f"Randomly initialized (new modules): {len(skipped)} layers")
    
    if len(skipped) > 0:
        print("Skipped layers (new modules):")
        for s in skipped[:10]:  # Show first 10
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Save the initialized model
    lmw.save(output_path)
    print(f"Successfully saved LMW-YOLO initial weights to {output_path}")

if __name__ == "__main__":
    main()
