import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_coco_eval(gt_path, pred_path):
    # Load ground truth
    coco_gt = COCO(gt_path)
    
    # Create mapping from basename to GT image_id
    # GT file_name usually look like 'JPEGImages/0.jpg'
    basename_to_id = {os.path.basename(img['file_name']): img['id'] for img in coco_gt.dataset['images']}
    
    # Load predictions
    with open(pred_path, 'r') as f:
        preds = json.load(f)
    
    # Adjust predictions
    valid_preds = []
    for p in preds:
        # 1. Adjust category_id (Ultralytics +1)
        p['category_id'] -= 1
        
        # 2. Map image_id using file_name basename
        # Pred file_name usually look like '0.jpg'
        pred_basename = os.path.basename(p['file_name'])
        if pred_basename in basename_to_id:
            p['image_id'] = basename_to_id[pred_basename]
            valid_preds.append(p)
        else:
            # Fallback to direct image_id if basename mapping fails
            # and check if it's in GT
            if p['image_id'] in coco_gt.imgs:
                 valid_preds.append(p)
            
    if not valid_preds:
         print("Error: No valid predictions found after ID mapping.")
         return

    # Standard COCO evaluation requires a temporary file
    tmp_pred_path = "tmp_preds_adjusted.json"
    with open(tmp_pred_path, 'w') as f:
        json.dump(valid_preds, f)
        
    # Load adjusted predictions into COCO API
    coco_dt = coco_gt.loadRes(tmp_pred_path)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Cleanup
    if os.path.exists(tmp_pred_path):
        os.remove(tmp_pred_path)

if __name__ == "__main__":
    gt = "data/test_coco/annotations.json"
    # Looking for the latest val run with predictions.json
    pred = "/home/ahmadreza/Downloads/Research/M2SODAI/mmdet/models/LMW-YOLO/runs/detect/val10/predictions.json"
    
    if os.path.exists(gt) and os.path.exists(pred):
        run_coco_eval(gt, pred)
    else:
        print(f"Error: Could not find GT or Predictions file.")
