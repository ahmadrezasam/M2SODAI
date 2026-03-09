import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert RGB weights to SFDA ProjectUpsampleResNet format')
    parser.add_argument('src', help='source rgb checkpoint file')
    parser.add_argument('dst', help='destination checkpoint file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print(f"Loading weights from {args.src}")
    checkpoint = torch.load(args.src, map_location='cpu')
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # Maps backbone.conv1 to backbone.resnet.conv1
            new_key = k.replace('backbone.', 'backbone.resnet.')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # Put the modified state dict back into the checkpoint structure
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict
        
    print(f"Saving converted weights to {args.dst}")
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(checkpoint, args.dst)
    
    print("Conversion complete!")

if __name__ == '__main__':
    main()
