import torch
import sys
import os
sys.path.append('.')

def test_weight_conversion():
    print("Testing weight conversion script...")
    
    # Create a mockup RGB state_dict
    rgb_state_dict = {
        'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
        'backbone.layer1.0.conv1.weight': torch.randn(64, 64, 1, 1),
        'neck.lateral_convs.0.conv.weight': torch.randn(256, 256, 1, 1),
        'roi_head.bbox_head.fc_cls.weight': torch.randn(3, 1024)
    }
    src_path = 'tools_new/mock_rgb.pth'
    dst_path = 'tools_new/mock_hsi.pth'
    
    torch.save({'state_dict': rgb_state_dict}, src_path)
    
    # Run the conversion script
    print(f"Running conversion: python tools_new/convert_rgb_to_hsi_projector.py {src_path} {dst_path}")
    os.system(f"python tools_new/convert_rgb_to_hsi_projector.py {src_path} {dst_path}")
    
    # Verify the output
    print("\nVerifying converted checkpoint...")
    hsi_checkpoint = torch.load(dst_path, map_location='cpu')
    hsi_state_dict = hsi_checkpoint['state_dict']
    
    # Expected key remapping: backbone. -> backbone.resnet.
    assert 'backbone.resnet.conv1.weight' in hsi_state_dict
    assert 'backbone.resnet.layer1.0.conv1.weight' in hsi_state_dict
    
    # Check projector keys
    assert 'backbone.channel_projector.0.weight' in hsi_state_dict
    assert 'backbone.channel_projector.1.weight' in hsi_state_dict
    
    # Check projector values
    projector_weight = hsi_state_dict['backbone.channel_projector.0.weight']
    print(f"Projector weight shape: {projector_weight.shape}")
    print(f"Projector weight mean: {projector_weight.mean().item():.6f}")
    assert torch.allclose(projector_weight, torch.ones(3, 30, 1, 1) / 30.0)
    
    print("\nWeight conversion test passed!")
    
    # Clean up
    if os.path.exists(src_path): os.remove(src_path)
    if os.path.exists(dst_path): os.remove(dst_path)

if __name__ == '__main__':
    test_weight_conversion()
