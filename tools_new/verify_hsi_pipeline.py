import mmcv
from mmdet.datasets.pipelines import Compose
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_pipeline():
    # Define the pipeline
    img_norm_cfg = dict(
        mean=[123.6, 116.2, 103.5], std=[58.39, 56.12, 57.3], to_rgb=False)
    
    pipeline = Compose([
        dict(type='LoadRawHSI'),
        dict(type='ExtractRGBBandsFromHSI', r_idx=53, g_idx=32, b_idx=11),
        dict(type='Resize', img_scale=(1600, 1600), keep_ratio=True),
        # Collect usually expects some meta keys, we can skip it or provide them
        # dict(type='Collect', keys=['img'])
    ])

    # Sample HSI info
    results = dict(
        img_prefix='./data/',
        img_info=dict(filename='test_coco/0.jpg') # This will be transformed to data/test/0.mat
    )

    print("Running pipeline...")
    try:
        data = pipeline(results)
        img = data['img']
        print(f"Extracted image shape: {img.shape}")
        print(f"Image min/max: {img.min()}, {img.max()}")

        # Normalize for visualization [0, 1]
        img_vis = (img - img.min()) / (img.max() - img.min() + 1e-6)
        
        # Save the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis)
        plt.title("Extracted Pseudo-RGB (Bands 53, 32, 11)")
        plt.axis('off')
        
        save_path = 'work_dirs/hsi_lower_bound_verification.png'
        os.makedirs('work_dirs', exist_ok=True)
        plt.savefig(save_path)
        print(f"Verification image saved to {save_path}")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_pipeline()
