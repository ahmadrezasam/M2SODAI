import numpy as np
import cv2
import os
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, LetterBox, Format
import torch

class HSIDataset(YOLODataset):
    """
    Custom Dataset for 30-channel Hyperspectral Images.
    Expects .npy files instead of standard images.
    """
    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, original_shape, resized_shape)."""
        f = self.im_files[i]
        
        # Determine .npy path
        if f.endswith(".jpg"):
            npy_path = f[:-4] + ".npy"
        elif f.endswith(".png"):
            npy_path = f[:-4] + ".npy"
        else:
            npy_path = f + ".npy"
        
        if os.path.exists(npy_path) and npy_path.endswith(".npy"):
            im = np.load(npy_path) # Shape (H, W, 30)
        else:
            im = cv2.imread(f)

        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        
        h0, w0 = im.shape[:2]
        
        # Resize if necessary
        # Note: cv2.resize might only work for up to 512 channels, but let's check
        # For 30 channels, it should be fine.
        r = self.imgsz / max(h0, w0)
        if r != 1:
            # Resize each channel or use cv2.resize if it supports 30 channels
            # To be safe, we resize channel by channel if it fails
            try:
                new_size = (int(w0 * r), int(h0 * r))
                im = cv2.resize(im, new_size, interpolation=cv2.INTER_LINEAR)
            except:
                new_im = []
                for c in range(im.shape[2]):
                    new_im.append(cv2.resize(im[:,:,c], (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR))
                im = np.stack(new_im, axis=2)
                
        return im, (h0, w0), im.shape[:2]

    def build_transforms(self, hyp=None):
        """
        Build transforms for HSI data.
        Removes RGB-specific augmentations and problematic Mosaic/Mixup for small datasets.
        """
        if hyp is not None:
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            hyp.hsv_h = 0.0
            hyp.hsv_s = 0.0
            hyp.hsv_v = 0.0
            
        return super().build_transforms(hyp)
