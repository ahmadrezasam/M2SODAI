import torch
import os
import argparse

def convert_weights(src_path, dst_path):
    print(f"Loading checkpoint from {src_path}...")
    checkpoint = torch.load(src_path, map_location='cpu')
    
    # Checkpoint structure usually has 'state_dict' as a key
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Remap backbone keys to backbone.resnet
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', 'backbone.resnet.', 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # Initialize the 1x1 projector to average the 30 bands
    # shape: [out_channels, in_channels, kernel_size, kernel_size] = [3, 30, 1, 1]
    print("Initializing 1x1 channel projector weights with band averaging...")
    projector_weight = torch.ones(3, 30, 1, 1) / 30.0
    projector_bias = torch.zeros(3)
    
    new_state_dict['backbone.channel_projector.weight'] = projector_weight
    new_state_dict['backbone.channel_projector.bias'] = projector_bias
    
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict
        
    print(f"Saving new checkpoint to {dst_path}...")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(checkpoint, dst_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert RGB model weights to HSI-projected architecture")
    parser.add_argument('src', type=str, help='Path to the source generic RGB .pth file')
    parser.add_argument('dst', type=str, help='Path to save the new architecture .pth file')
    args = parser.parse_args()
    
    convert_weights(args.src, args.dst)
