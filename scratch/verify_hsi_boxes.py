import json
import cv2
import os
import numpy as np

def verify_batch():
    hsi_dir = 'baseline_official/test_hsi_rgb_640_v3_bilinear'
    json_path = os.path.join(hsi_dir, 'annotations.json')
    img_dir = os.path.join(hsi_dir, 'images')
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    # Get 10 random images with annotations
    images_with_anns = list(set([ann['image_id'] for ann in coco['annotations']]))[:10]
    
    vis_imgs = []
    for img_id in images_with_anns:
        img_info = next(img for img in coco['images'] if img['id'] == img_id)
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Draw boxes
        for ann in coco['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, img_info['file_name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        vis_imgs.append(img)
        
    # Create a 2x5 grid
    if vis_imgs:
        top_row = np.hstack(vis_imgs[:5])
        bottom_row = np.hstack(vis_imgs[5:])
        grid = np.vstack([top_row, bottom_row])
        cv2.imwrite('hsi_verification_grid.jpg', grid)
        print("Saved hsi_verification_grid.jpg")

if __name__ == "__main__":
    verify_batch()
