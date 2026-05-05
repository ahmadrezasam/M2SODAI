import os
import cv2
from tqdm import tqdm

def yolo_obb_to_dota(label_dir, image_dir, output_dir):
    """
    Converts YOLO-OBB format (normalized x1 y1 x2 y2 x3 y3 x4 y4) 
    to DOTA format (absolute pixels x1 y1 x2 y2 x3 y3 x4 y4 class difficulty).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files, desc=f"Converting {os.path.basename(label_dir)}"):
        img_name = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg', '.tif']:
                if os.path.exists(os.path.join(image_dir, label_file.replace('.txt', ext))):
                    img_path = os.path.join(image_dir, label_file.replace('.txt', ext))
                    break
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        with open(os.path.join(output_dir, label_file), 'w') as f_out:
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                cls_id = int(parts[0])
                # Ship detection only
                if cls_id != 0:
                    continue
                cls_name = 'ship'
                
                coords = [float(x) for x in parts[1:9]]
                abs_coords = []
                for i in range(0, 8, 2):
                    abs_coords.append(coords[i] * w)
                    abs_coords.append(coords[i+1] * h)
                
                out_line = " ".join([f"{x:.1f}" for x in abs_coords]) + f" {cls_name} 0\n"
                f_out.write(out_line)

if __name__ == "__main__":
    base_path = "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official"
    folds = [1, 2, 3]
    
    for fold in folds:
        for split in ['train', 'val']:
            l_dir = f"{base_path}/fold_{fold}_obb/{split}/labels"
            i_dir = f"{base_path}/fold_{fold}_obb/{split}/images"
            o_dir = f"{base_path}/fold_{fold}_obb/dota/{split}/labelTxt"
            
            if os.path.exists(l_dir):
                yolo_obb_to_dota(l_dir, i_dir, o_dir)
                img_out_dir = f"{base_path}/fold_{fold}_obb/dota/{split}/images"
                if not os.path.exists(img_out_dir):
                    os.makedirs(os.path.dirname(img_out_dir), exist_ok=True)
                    os.symlink(i_dir, img_out_dir)

    print("\nConversion complete (Ships Only). DOTA format labels are in baseline_official/fold_x_obb/dota/")
