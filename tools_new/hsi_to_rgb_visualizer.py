import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
from pickle import load
from tqdm import tqdm

def load_pca_model(model_path='./data/model.pkl'):
    if os.path.exists(model_path):
        return load(open(model_path, 'rb'))
    return None

def hsi_to_rgb(hsi_data, method='average', checkpoint_path=None, pca_model=None):
    """
    Convert HSI data to RGB.
    hsi_data: numpy array of shape (H, W, C)
    """
    h, w, c = hsi_data.shape
    
    if method == 'average':
        # Simple averaging into 3 channels
        step = c // 3
        r = hsi_data[:, :, 0:step].mean(axis=2)
        g = hsi_data[:, :, step:2*step].mean(axis=2)
        b = hsi_data[:, :, 2*step:].mean(axis=2)
        rgb = np.stack([r, g, b], axis=2)
        
    elif method == 'projector' and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        weight_key = 'backbone.channel_projector.0.weight'
        if weight_key not in state_dict:
            for k in state_dict.keys():
                if 'channel_projector.0.weight' in k:
                    weight_key = k
                    break
        
        if weight_key in state_dict:
            weights = state_dict[weight_key].squeeze().numpy()
            hsi_tensor = hsi_data.reshape(-1, c)
            rgb_tensor = hsi_tensor @ weights.T
            rgb = rgb_tensor.reshape(h, w, 3)
            
            # Apply BatchNorm if exists
            bias_key = weight_key.replace('.0.weight', '.1.bias')
            mean_key = weight_key.replace('.0.weight', '.1.running_mean')
            var_key = weight_key.replace('.0.weight', '.1.running_var')
            weight_bn_key = weight_key.replace('.0.weight', '.1.weight')
            if bias_key in state_dict and mean_key in state_dict:
                b = state_dict[bias_key].numpy()
                m = state_dict[mean_key].numpy()
                v = state_dict[var_key].numpy()
                w_bn = state_dict[weight_bn_key].numpy()
                rgb = (rgb - m) / np.sqrt(v + 1e-5) * w_bn + b
        else:
            print(f"Warning: Projector weights not found. Using average.")
            return hsi_to_rgb(hsi_data, method='average')

    elif method == 'pca_select' or c == 127:
        # If it's already 127 bands, we just select
        if c == 127:
            hsi_127 = hsi_data.reshape(-1, 127)
        elif c == 30 and pca_model:
            if os.path.exists('./data/pca_mean_std.mat'):
                pca_meta = sio.loadmat('./data/pca_mean_std.mat')
                mean_pca = pca_meta['mean']
                std_pca = pca_meta['std']
                hsi_normalized = hsi_data.reshape(-1, 30)
                hsi_unnorm = hsi_normalized * std_pca + mean_pca
                hsi_127 = pca_model.inverse_transform(hsi_unnorm)
            else:
                print("Warning: pca_mean_std.mat not found. Using average.")
                return hsi_to_rgb(hsi_data, method='average')
        else:
            print(f"Warning: Cannot use pca_select with {c} channels without model. Using average.")
            return hsi_to_rgb(hsi_data, method='average')
            
        # R: ~650nm (index ~53), G: ~550nm (index ~32), B: ~450nm (index ~11)
        r = hsi_127[:, 53]
        g = hsi_127[:, 32]
        b = hsi_127[:, 11]
        rgb = np.stack([r, g, b], axis=1).reshape(h, w, 3)
            
    else:
        return hsi_to_rgb(hsi_data, method='average')

    # Normalize to 0-1 for visualization
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

    # Normalize to 0-1 for visualization
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def process_file(input_path, output_path, method, checkpoint_path, pca_model):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    data = sio.loadmat(input_path)
    if 'data' not in data:
        # Check for other common HSI keys
        for k in data.keys():
            if not k.startswith('__'):
                hsi_data = data[k]
                break
        else:
            print(f"Error: No HSI data key found in {input_path}.")
            return
    else:
        hsi_data = data['data']
    
    rgb = hsi_to_rgb(hsi_data, method=method, checkpoint_path=checkpoint_path, pca_model=pca_model)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, rgb)

def main():
    parser = argparse.ArgumentParser(description='Convert HSI .mat to RGB image (Batch supported)')
    parser.add_argument('input', help='Input .mat file or directory')
    parser.add_argument('--method', default='average', choices=['average', 'projector', 'pca_select'], help='Conversion method')
    parser.add_argument('--checkpoint', help='Checkpoint path for projector method')
    parser.add_argument('--out', default='output_rgb', help='Output image path or directory')
    args = parser.parse_args()

    pca_model = load_pca_model() if args.method == 'pca_select' else None

    if os.path.isdir(args.input):
        # Batch mode
        os.makedirs(args.out, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.endswith('.mat')]
        print(f"Batch processing {len(files)} files from {args.input}...")
        for f in tqdm(files):
            process_file(os.path.join(args.input, f), 
                         os.path.join(args.out, f.replace('.mat', '.png')),
                         args.method, args.checkpoint, pca_model)
    else:
        # Single file mode
        out_path = args.out if args.out.endswith(('.png', '.jpg', '.jpeg')) else os.path.join(args.out, os.path.basename(args.input).replace('.mat', '.png'))
        process_file(args.input, out_path, args.method, args.checkpoint, pca_model)
        print(f"Saved RGB image to {out_path}")

if __name__ == '__main__':
    main()
