import argparse
import os
import json
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import inference_detector
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for SFDA')
    parser.add_argument('config', help='train config file path for Target domain (e.g. HSI)')
    parser.add_argument('checkpoint', help='pretrained RGB checkpoint file')
    parser.add_argument('--out', default='data/train_coco/pseudo_labels_HSI.json', help='output JSON path')
    parser.add_argument('--conf-thr', type=float, default=0.8, help='confidence threshold')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading config from {args.config}")
    cfg = Config.fromfile(args.config)

    # 1. Build the model
    # We set test_cfg to None to avoid inference failures on uninitialized stuff, but let's just use the config
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    # 2. Load the checkpoint
    print(f"Loading weights from {args.checkpoint}")
    # The checkpoint contains keys for the inner resnet mostly. 
    # Because our new backbone is ProjectUpsampleResNet -> self.resnet,
    # the keys in the checkpoint (which start with backbone.xxx) might need to be mapped to backbone.resnet.xxx
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Modify state dict keys dynamically to fit the new wrapper
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # Maps backbone.conv1 to backbone.resnet.conv1
            new_key = k.replace('backbone.', 'backbone.resnet.')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # Load state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Send model to GPU and put in eval mode
    model.cuda()
    model.eval()

    # 3. Load the dataset (HSI train dataset)
    # We need to build the dataset to iterate through the images easily
    print("Building HSI dataset for inference...")
    # Temporarily modify config to use test pipeline for the train dataset so we don't apply flip/crop
    cfg.data.train.pipeline = cfg.data.test.pipeline
    dataset = build_dataset(cfg.data.train)

    print(f"Found {len(dataset)} images in the training dataset.")

    # Prepare COCO format dictionary
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "ship"},
            {"id": 2, "name": "floatingmatter"}
        ]
    }

    # Extract existing classes to ID mapping
    # MMDetection classes are 0-indexed, COCO ids are 1-indexed
    class_name_to_id = {name: i + 1 for i, name in enumerate(dataset.CLASSES)}
    
    ann_id = 1
    
    # We must load images manually using inference_detector to apply the exact same test_pipeline transforms
    img_prefix = cfg.data.train.img_prefix
    
    for img_info in tqdm(dataset.data_infos, desc="Generating Pseudo-labels"):
        img_id = img_info['id']
        filename = img_info['filename']
        img_path = os.path.join(img_prefix, filename)
        
        # Add to images dict
        coco_output["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_info['width'],
            "height": img_info['height']
        })
        
        # Run inference
        with torch.no_grad():
            result = inference_detector(model, img_path)
            
        # result is a list of numpy arrays, one for each class
        for class_idx, class_result in enumerate(result):
            # MMDetection returns bounding boxes as [x1, y1, x2, y2, score]
            if len(class_result) == 0:
                continue
                
            # Filter by confidence threshold
            confident_preds = class_result[class_result[:, 4] > args.conf_thr]
            
            for pred in confident_preds:
                x1, y1, x2, y2, score = pred
                w = x2 - x1
                h = y2 - y1
                
                # Add annotation
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_idx + 1,  # 1-indexed
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "score": float(score)  # keeping score for reference if needed
                })
                ann_id += 1

    # Save output
    print(f"Generated {len(coco_output['annotations'])} pseudo-labels.")
    print(f"Saving to {args.out}...")
    with open(args.out, 'w') as f:
        json.dump(coco_output, f)
        
    print("Done!")

if __name__ == '__main__':
    main()
