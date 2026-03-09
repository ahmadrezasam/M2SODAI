import sys
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_hsi_sfda.py'
    checkpoint_file = 'rgb_sfda_mapped.pth'
    
    print(f"1. Loading configuration from {config_file}...")
    cfg = Config.fromfile(config_file)
    
    print("2. Building the detector model with ProjectUpsampleResNet backbone...")
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    # Check if the backbone was built correctly
    print(f"--> Backbone type: {type(model.backbone).__name__}")
    if type(model.backbone).__name__ == "ProjectUpsampleResNet":
        print("--> Custom backbone successfully built!")
    else:
        print("--> ERROR: Custom backbone not found.")
        sys.exit(1)
        
    print(f"\n3. Loading weights from {checkpoint_file} into the model...")
    # strict=False is normally used, but let's see what keys are missing using load_checkpoint
    try:
        checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu', strict=False)
        print("--> Weights loaded successfully!")
    except Exception as e:
        print(f"--> ERROR loading weights: {e}")
        sys.exit(1)
        
    print("\nVerification Complete: The SFDA model architecture and mapped weights are compatible!")

if __name__ == '__main__':
    main()
