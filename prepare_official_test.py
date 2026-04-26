import argparse
import json
import os
import shutil

def prepare_official_hsi_rgb_test(data_dir):
    """
    Prepares the official test_rgb dataset structure for YOLOv5 baseline evaluation.
    This creates the structure expected by the baseline testing tools by symlinking
    the raw test images.
    """
    output_dir = os.path.join(data_dir, '..', 'baseline_official', 'test_official_hsi_rgb')
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Note: We use the already provided dataset's test_rgb for HSI-to-RGB images
    raw_images_dir = os.path.join(data_dir, 'test_rgb', 'JPEGImages')
    print(f"Creating symlinks for official HSI-RGB test images from {raw_images_dir} to {images_output_dir}")

    for img_file in os.listdir(raw_images_dir):
        if img_file.endswith('.jpg'):
            src_path = os.path.join(raw_images_dir, img_file)
            dst_path = os.path.join(images_output_dir, img_file)
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

    json_path = os.path.join(data_dir, 'test_rgb', 'annotations_RGB.json')
    print(f"Parsing labels from {json_path}")
    
    with open(json_path) as f:
        coco_data = json.load(f)

    images_dict = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}
    category_id_to_idx = {cat['id']: cat['name'] for cat in coco_data['categories']}
    assert 0 in category_id_to_idx or 'ship' in category_id_to_idx.values()

    # The paper trains on 'ship' category only. For binary evaluation, ensure cls=0
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        category_id = ann['category_id']
        if category_id != 0:
            continue
        
        file_name, width, height = images_dict[img_id]
        bbox = ann['bbox'] # [x_min, y_min, width, height]
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w = bbox[2] / width
        h = bbox[3] / height
        
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_output_dir, txt_name)
        
        with open(label_path, 'a') as lf:
            lf.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Done. Labels saved to {labels_output_dir}")


def prepare_official_hsi_pca_test(data_dir):
    """
    Prepares the official test_pca dataset structure for YOLOv5 baseline evaluation.
    This creates 3-channel pseudo-RGB representations using the first 3 principal components
    as was officially evaluated for baseline models in the paper.
    """
    import scipy.io as sio
    import numpy as np
    from cv2 import resize
    from PIL import Image

    output_dir = os.path.join(data_dir, '..', 'baseline_official', 'test_official_hsi_pca_v2')
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    raw_mat_dir = os.path.join(data_dir, 'test_pca')
    print(f"Processing PCA MAT files from {raw_mat_dir} to {images_output_dir}")

    for mat_file in os.listdir(raw_mat_dir):
        if mat_file.endswith('.mat'):
            mat_path = os.path.join(raw_mat_dir, mat_file)
            mat_data = sio.loadmat(mat_path)
            
            cube = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    cube = mat_data[key]
                    break
            
            if cube is None: continue
            
            # Use first 3 PCs and map to 0-255 scale
            pc_img = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.uint8)
            num_channels = min(3, cube.shape[2])
            for i in range(num_channels):
                band = cube[:, :, i]
                # Normalize band to 0-255
                b_min, b_max = band.min(), band.max()
                if b_max > b_min:
                    band_norm = (band - b_min) / (b_max - b_min) * 255.0
                else:
                    band_norm = np.zeros_like(band)
                pc_img[:, :, i] = band_norm.astype(np.uint8)
                
            img = Image.fromarray(pc_img)
            dst_name = mat_file.replace('.mat', '.jpg')
            img.save(os.path.join(images_output_dir, dst_name))
            
    json_path = os.path.join(data_dir, 'test_coco', 'annotations_HSI.json')
    print(f"Parsing labels from {json_path}")
    
    with open(json_path) as f:
        coco_data = json.load(f)

    images_dict = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}

    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        category_id = ann['category_id']
        if category_id != 0: continue
            
        file_name, width, height = images_dict[img_id]
        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w = bbox[2] / width
        h = bbox[3] / height
        
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_output_dir, txt_name)
        
        with open(label_path, 'a') as lf:
            lf.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Done. Labels saved to {labels_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/ahmadreza/Downloads/Research/M2SODAI/data')
    parser.add_argument('--task', type=str, default='all', choices=['all', 'rgb', 'pca'])
    args = parser.parse_args()
    
    if args.task in ['all', 'rgb']:
        prepare_official_hsi_rgb_test(args.data_dir)
        
    if args.task in ['all', 'pca']:
        prepare_official_hsi_pca_test(args.data_dir)
