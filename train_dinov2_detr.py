import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    DeformableDetrConfig, 
    DeformableDetrForObjectDetection, 
    AutoImageProcessor
)
from tqdm import tqdm
import torchvision
from pycocotools.cocoeval import COCOeval
import numpy as np
import time
import csv
import logging

# Configuration
BASE_DIR = "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official"
IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 300
LEARNING_RATE = 1e-4

BACKBONE = "vit_small_patch16_dinov3.lvd1689m"

class CocoDetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor=None):
        super(CocoDetrDataset, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetrDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target_dict = {'image_id': image_id, 'annotations': target}
        
        # Original size for evaluation
        w, h = img.size
        
        if self.processor:
            encoding = self.processor(images=img, annotations=target_dict, return_tensors="pt")
            item = {k: v[0] for k, v in encoding.items()}
            item["orig_size"] = torch.tensor([h, w])
            item["image_id"] = torch.tensor([image_id])
            return item
            
        return img, target_dict

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    
    batch_pixel_values = torch.stack(pixel_values, dim=0)
    # The processor expects 'labels' to be a list of dictionaries
    return {
        "pixel_values": batch_pixel_values, 
        "labels": labels, 
        "orig_sizes": orig_sizes,
        "image_ids": image_ids
    }

def get_gpu_mem():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_reserved() / 1e9:.3f}G"
    return "0G"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Processor
    processor = AutoImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        size={"shortest_edge": IMG_SIZE, "longest_edge": IMG_SIZE}
    )
    
    for fold_num in [1, 2, 3]:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR FOLD {fold_num}")
        print(f"{'='*50}\n")
        
        fold_dir = os.path.join(BASE_DIR, f"fold_{fold_num}")
        save_dir = f"models/fold_{fold_num}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        log_path = os.path.join(save_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(f"Fold_{fold_num}")
        
        csv_path = os.path.join(save_dir, "results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'mAP50', 'mAP50-95'])
            
        train_img_dir = os.path.join(fold_dir, "train", "images")
        train_ann_file = os.path.join(fold_dir, "train", "annotations.json")
        val_img_dir = os.path.join(fold_dir, "val", "images")
        val_ann_file = os.path.join(fold_dir, "val", "annotations.json")

        train_dataset = CocoDetrDataset(train_img_dir, train_ann_file, processor=processor)
        val_dataset = CocoDetrDataset(val_img_dir, val_ann_file, processor=processor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        # 2. Initialize Model with DINOv2/v3 backbone
        print(f"Initializing Deformable DETR with {BACKBONE} backbone...")
        config = DeformableDetrConfig(
            use_timm_backbone=True,
            backbone=BACKBONE,
            num_labels=1,
            ignore_mismatched_sizes=True,
            use_pretrained_backbone=True,
            out_indices=[3, 6, 9, 11]  # Better feature spread for 12-layer ViT
        )
        
        model = DeformableDetrForObjectDetection(config)
        model.to(device)
        
        # Differential Learning Rate: Backbone should be 10x smaller than head
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": LEARNING_RATE,
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": LEARNING_RATE / 10,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
        
        best_map50 = 0.0
        
        # YOLO-like Header
        logger.info(f"\n{'Epoch':>10} {'GPU_mem':>10} {'Loss':>10} {'Instances':>10} {'Size':>10}")
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"{epoch+1}/{EPOCHS}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                num_instances = sum([len(l["class_labels"]) for l in labels])
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar with YOLO-like info
                mem = get_gpu_mem()
                pbar.set_description(f"{epoch+1:10}/ {EPOCHS:10} {mem:>10} {loss.item():10.4f} {num_instances:10} {IMG_SIZE:10}")
            
            # Validation / mAP Calculation
            model.eval()
            results = []
            logger.info(f"\n{'Class':>10} {'Images':>10} {'Instances':>10} {'mAP50':>10} {'mAP50-95':>10}")
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    pixel_values = batch["pixel_values"].to(device)
                    outputs = model(pixel_values=pixel_values)
                    
                    orig_target_sizes = torch.stack(batch["orig_sizes"]).to(device)
                    processed_results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.01)
                    
                    for i, res in enumerate(processed_results):
                        image_id = batch["image_ids"][i].item()
                        boxes = res["boxes"].cpu().numpy()
                        scores = res["scores"].cpu().numpy()
                        labels = res["labels"].cpu().numpy()
                        
                        for box, score, label in zip(boxes, scores, labels):
                            x_min, y_min, x_max, y_max = box
                            results.append({
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                "score": float(score)
                            })

            if results:
                try:
                    coco_dt = val_dataset.coco.loadRes(results)
                    coco_eval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    
                    mAP = coco_eval.stats[0]
                    mAP50 = coco_eval.stats[1]
                except Exception as e:
                    print(f"Eval error: {e}")
                    mAP, mAP50 = 0, 0
            else:
                mAP, mAP50 = 0, 0
            
            num_val_images = len(val_dataset)
            num_val_instances = len(val_dataset.coco.getAnnIds())
            avg_loss = total_loss / len(train_loader)
            logger.info(f"{'all':>10} {num_val_images:10} {num_val_instances:10} {mAP50:10.4f} {mAP:10.4f}\n")
            
            # Save CSV log
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, mAP50, mAP])
            
            # Save checkpoints (last and best)
            os.makedirs(save_dir, exist_ok=True)
            
            # Always save as last
            last_dir = os.path.join(save_dir, "last")
            model.save_pretrained(last_dir)
            processor.save_pretrained(last_dir)
            
            # Save best if mAP50 improved
            if mAP50 > best_map50:
                best_map50 = mAP50
                best_dir = os.path.join(save_dir, "best")
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                logger.info(f"New best model saved to {best_dir} (mAP50: {best_map50:.4f})")
            
        logger.info(f"Training Complete for Fold {fold_num}!")

if __name__ == "__main__":
    train()
