"""
Mean Teacher UDA Diagnostic & Debugging Tool for YOLOv26-OBB
Allows visual inspection of teacher pseudo-labels, confidence distributions,
and classification-to-localization alignment on HSI target images.
Saves numerical statistics to CSV for distribution analysis.
"""

import os
import cv2
import csv
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

def draw_obb(image, box_coords, color=(0, 255, 0), thickness=2):
    """
    Draws a single oriented bounding box on the image using OpenCV.
    box_coords: [x_c, y_c, w, h, angle_rad] in pixel coordinates.
    """
    x_c, y_c, w, h, angle = box_coords
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Half width and height
    hw = w / 2
    hh = h / 2
    
    # Local offsets relative to center
    dx1 = cos_a * hw - sin_a * hh
    dy1 = sin_a * hw + cos_a * hh
    
    dx2 = cos_a * hw + sin_a * hh
    dy2 = sin_a * hw - cos_a * hh
    
    # Four rotated corner points
    corners = np.array([
        [x_c - dx1, y_c - dy1],
        [x_c + dx2, y_c + dy2],
        [x_c + dx1, y_c + dy1],
        [x_c - dx2, y_c - dy2]
    ], dtype=np.int32)
    
    # Draw rotated bounding box
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=thickness)
    # Highlight the center point
    cv2.circle(image, (int(x_c), int(y_c)), 3, (0, 0, 255), -1)

def run_diagnostics():
    project_root = "/home/ahmadreza/Downloads/Research/M2SODAI"
    
    # Input directories
    weights_path = os.path.join(project_root, 'runs', 'mt_integrated', 'yolo26s_mt_fold1_adaptive', 'weights', 'best.pt')
    images_dir = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb', 'images')
    ground_truth_dir = os.path.join(project_root, 'baseline_official', 'test_hsi_rgb_640_v3_bilinear_obb', 'labels')
    
    # Output debug directory
    output_dir = os.path.join(project_root, 'runs', 'mt_debug_pseudo_labels')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("RUNNING MEAN TEACHER UDA VISUAL DIAGNOSTICS & DEBUGGING PIPELINE")
    print("="*80)
    
    if not os.path.exists(weights_path):
        print(f"Error: Trained MT weights not found at {weights_path}")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    print(f"Loading Model Weights from: {weights_path}")
    model = YOLO(weights_path)
    model.to(device)
    model.model.eval() # Put model in eval mode
    
    # We will sample some images to visualize and evaluate
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Total HSI target test images available: {len(image_files)}")
    
    # Select 20 representative samples for debugging
    sample_images = sorted(image_files)[:20]
    
    conf_thresholds = [0.35, 0.50, 0.70]
    
    # Prepare list for structured numerical stats
    csv_rows = []
    
    for conf_th in conf_thresholds:
        print(f"\n>>> Running diagnostic sweep with Confidence Threshold = {conf_th}...")
        conf_out_dir = os.path.join(output_dir, f"conf_{int(conf_th*100)}")
        os.makedirs(conf_out_dir, exist_ok=True)
        
        total_detections = 0
        total_conf = 0.0
        
        for img_name in tqdm(sample_images):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            # Preprocess image to tensor matching YOLO OBB input format
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            
            with torch.no_grad():
                # Get raw model outputs
                raw_out = model.model(img_tensor)
                
            # Perform non-max suppression matching our mt_obb_trainer.py
            is_end2end = getattr(model.model, "end2end", False)
            detections = non_max_suppression(
                raw_out,
                conf_thres=conf_th,
                iou_thres=0.45,
                nc=1,
                max_det=50,
                max_time_img=0.5,
                rotated=True,
                end2end=is_end2end
            )
            
            dets = detections[0] # Single image batch
            
            # Copy original image to draw on
            canvas = img.copy()
            
            # Count ground truth objects
            gt_count = 0
            gt_path = os.path.join(ground_truth_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9: # OBB polygon format
                            gt_count += 1
                            poly = np.array([float(x) for x in parts[1:9]]).reshape(-1, 2) * w
                            cv2.polylines(canvas, [poly.astype(np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
            
            # Count pseudo-labels and draw them
            pl_count = len(dets)
            img_conf_sum = 0.0
            
            if pl_count > 0:
                for det in dets:
                    # det: [x_c, y_c, w, h, conf, cls, angle] in pixels
                    x_c, y_c, box_w, box_h, conf, cls, angle = det.cpu().numpy()
                    draw_obb(canvas, [x_c, y_c, box_w, box_h, angle], color=(0, 255, 0), thickness=2)
                    
                    # Write confidence score
                    cv2.putText(canvas, f"{conf:.2f}", (int(x_c - box_w/2), int(y_c - box_h/2 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    img_conf_sum += conf
                    total_detections += 1
                    total_conf += conf
            
            # Save debug canvas
            cv2.imwrite(os.path.join(conf_out_dir, f"debug_{img_name}"), canvas)
            
            # Add image-level stat row
            img_avg_conf = img_conf_sum / pl_count if pl_count > 0 else 0.0
            csv_rows.append({
                "Threshold": conf_th,
                "Image": img_name,
                "GT_Count": gt_count,
                "Pseudo_Label_Count": pl_count,
                "Avg_Confidence": round(img_avg_conf, 4)
            })
            
        avg_c = total_conf / total_detections if total_detections > 0 else 0.0
        print(f"  Summary for Conf={conf_th}: Total pseudo-labels={total_detections} | Avg Confidence={avg_c:.3f}")
    
    # Save statistics to CSV
    csv_path = os.path.join(output_dir, "sweep_statistics.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Threshold", "Image", "GT_Count", "Pseudo_Label_Count", "Avg_Confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC VISUALIZATIONS & NUMERICAL NUMBERS GENERATED SUCCESSFULLY!")
    print(f"Check output images in: {output_dir}")
    print(f"Check CSV numerical stats in: {csv_path}")
    print("  - Cyan bounding boxes represent Ground Truth (HSI).")
    print("  - Green bounding boxes represent Teacher's predicted pseudo-labels with confidence.")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_diagnostics()
