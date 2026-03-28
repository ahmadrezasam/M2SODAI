import mmcv
from mmdet.apis import init_detector, inference_detector
import torch
import os

def debug_inference():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_hsi_lower_bound.py'
    checkpoint_file = 'work_dirs/faster_rcnn_r50_rgb/rgb.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Init detector
    print(f"Loading model on {device}...")
    model = init_detector(config_file, checkpoint_file, device=device)
    
    # Sample image info
    # We need a real HSI file path that the loader can handle
    # LoadRawHSI expects results['img_info']['filename'] and prepends img_prefix
    # In config: img_prefix = 'data/test_coco/', filename from JSON
    img_path = 'data/test_coco/JPEGImages/993.jpg' # This is what the loader gets as input
    
    print(f"Running inference on {img_path}...")
    result = inference_detector(model, img_path)
    
    # Inspect result
    # result is a list of [bbox_results] for each class
    for i, class_result in enumerate(result):
        if class_result.shape[0] > 0:
            print(f"Class {i} has {class_result.shape[0]} detections")
            scores = class_result[:, 4]
            print(f"  Max score: {scores.max():.4f}, Min score: {scores.min():.4f}")
            print(f"  Detections with score > 0.05: {(scores > 0.05).sum()}")
        else:
            print(f"Class {i} has NO detections")

if __name__ == '__main__':
    debug_inference()
