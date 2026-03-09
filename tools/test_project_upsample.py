import torch
import sys
sys.path.append('.')
from mmdet.models.backbones.project_upsample_resnet import ProjectUpsampleResNet

# Create a mockup ResNet config
resnet_cfg = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch'
)

def main():
    print("Building ProjectUpsampleResNet...")
    model = ProjectUpsampleResNet(in_channels=30, resnet_cfg=resnet_cfg, target_size=(1600, 1600))
    model.eval()
    
    # 224x224 HSI image, batch size 2, 30 channels
    print("Creating dummy HSI input tensor (2, 30, 224, 224)...")
    dummy_input = torch.randn(2, 30, 224, 224)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print("\nForward pass successful! Output shapes:")
    for i, out in enumerate(outputs):
        print(f"Stage {i} output shape: {out.shape}")
        
    print("\nExpected output shapes (based on 1600x1600 input to standard ResNet-50):")
    print("Stage 0 (stride 4): (2, 256, 400, 400)")
    print("Stage 1 (stride 8): (2, 512, 200, 200)")
    print("Stage 2 (stride 16): (2, 1024, 100, 100)")
    print("Stage 3 (stride 32): (2, 2048, 50, 50)")

if __name__ == '__main__':
    main()
