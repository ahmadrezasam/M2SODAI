# Setup for Faster R-CNN R50 RGB > HSI (No Adaptation)

### **Evaluation Command**
```bash
/home/ahmadreza/miniconda3/envs/Maritime/bin/python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_rgb_with_1x1_projector.py \
    work_dirs/faster_rcnn_r50_rgb/rgb_projected.pth \
    --eval bbox
```

### **1. Architecture Settings**
*   **Backbone**: `ProjectUpsampleResNet` (Includes a 1x1 conv layer before ResNet).
*   **Input Channels**: **30** (Expecting PCA-compressed `.mat` files).
*   **Spatial Scaling**: 
    *   Backbone: `target_size=(1600, 1600)`
    *   Data Pipeline: `img_scale=(1600, 1600)`

### **2. Projector Weight Initialization**
*   **Method**: **Band Averaging** (Deterministic).
*   **Logic**: Every one of the 30 input PCA bands is multiplied by **1/30** ($\approx 0.033$) and summed into each RGB channel.
*   **Weights Initialization**: `torch.ones(3, 30, 1, 1) / 30.0`.
*   **Bias/BN**: BatchNorm initialized to identity (scale 1.0, bias 0.0).

### **3. Data Pipeline (`coco_detection_hsi.py`)**
*   **Loading**: `LoadImageFromHSI` (loads the `'data'` key from `.mat`).
*   **Normalization**: `mean=0.0`, `std=1.0` for all 30 channels (effectively no normalization).
*   **Resize**: Scaled up to **1600x1600** before entering the model.
--------------------------------
