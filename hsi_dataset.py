import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import cv2
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, LetterBox, Format
import torch
import torch.multiprocessing as mp

# Disable OpenCV multithreading to prevent memory leaks in dataloader workers
cv2.setNumThreads(0)

# Set sharing strategy to 'file_system' to prevent shared memory (/dev/shm) exhaustion crashes when using workers > 0
mp.set_sharing_strategy('file_system')

class HSIDataset(YOLODataset):
    """
    Custom Dataset for 30-channel Hyperspectral Images.
    Expects .npy files instead of standard images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_shared_cache = False
        
        # Only initialize the shared memory cache if cache is explicitly enabled (e.g. 'ram' or True)
        cache_mode = getattr(self, "cache", False)
        if cache_mode in (True, "ram"):
            print(f"[HSIDataset] Initializing HSI Shared Memory cache for {len(self.im_files)} files...")
            
            first_img = None
            for f in self.im_files:
                if f.endswith(".npy"):
                    npy_path = f
                elif f.endswith(".jpg"):
                    npy_path = f[:-4] + ".npy"
                elif f.endswith(".png"):
                    npy_path = f[:-4] + ".npy"
                else:
                    npy_path = f + ".npy"
                    
                if os.path.exists(npy_path):
                    try:
                        first_img = np.load(npy_path, mmap_mode='r')
                        break
                    except Exception as e:
                        print(f"[HSIDataset] Error loading first HSI {npy_path}: {e}")
                    
            if first_img is not None:
                h0, w0, c0 = first_img.shape
                print(f"[HSIDataset] Detected HSI shape: {h0}x{w0}x{c0}")
                
                # Pre-allocate contiguous CPU shared tensor as float16 to save 50% memory
                self.shared_cache = torch.zeros((len(self.im_files), h0, w0, c0), dtype=torch.float16)
                
                loaded_count = 0
                for idx, f in enumerate(self.im_files):
                    if f.endswith(".npy"):
                        npy_path = f
                    elif f.endswith(".jpg"):
                        npy_path = f[:-4] + ".npy"
                    elif f.endswith(".png"):
                        npy_path = f[:-4] + ".npy"
                    else:
                        npy_path = f + ".npy"
                        
                    if os.path.exists(npy_path):
                        try:
                            im = np.load(npy_path, mmap_mode='r')
                            self.shared_cache[idx] = torch.from_numpy(np.array(im, dtype=np.float16))
                            loaded_count += 1
                        except Exception as e:
                            print(f"[HSIDataset] Error loading {npy_path} at index {idx}: {e}")
                    else:
                        # Fallback/placeholder if file not found
                        pass
                
                # Call share_memory_() so memory is shared via shm handles
                self.shared_cache.share_memory_()
                self.use_shared_cache = True
                print(f"[HSIDataset] HSI Shared Memory cache successfully created and shared! Loaded {loaded_count}/{len(self.im_files)} files. Total Cache Size: {self.shared_cache.element_size() * self.shared_cache.nelement() / 1024 / 1024:.2f} MB")
        else:
            print(f"[HSIDataset] HSI Shared Memory cache is DISABLED (cache={cache_mode}). Loading on-the-fly from disk to preserve System RAM.")

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, original_shape, resized_shape)."""
        if hasattr(self, "use_shared_cache") and self.use_shared_cache:
            # Convert float16 shared tensor back to float32 numpy array on the fly
            im = self.shared_cache[i].to(torch.float32).numpy()
        else:
            f = self.im_files[i]
            
            # Determine .npy path
            if f.endswith(".npy"):
                npy_path = f
            elif f.endswith(".jpg"):
                npy_path = f[:-4] + ".npy"
            elif f.endswith(".png"):
                npy_path = f[:-4] + ".npy"
            else:
                npy_path = f + ".npy"
            
            if os.path.exists(npy_path) and npy_path.endswith(".npy"):
                im = np.load(npy_path) # Shape (H, W, 30)
                im = np.array(im, dtype=np.float32)   # Copy to memory
            else:
                im = cv2.imread(f)
                if im is not None:
                    im = im.astype(np.float32)

            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

        h0, w0 = im.shape[:2]
        
        # 2. Vectorized PyTorch interpolation for high-dimensional spectral channels
        r = self.imgsz / max(h0, w0)
        if r != 1:
            new_h, new_w = int(h0 * r), int(w0 * r)
            # Convert NumPy array (H, W, C) -> PyTorch tensor (1, C, H, W)
            tensor_im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
            
            # Perform vectorized bilinear interpolation
            tensor_im = torch.nn.functional.interpolate(
                tensor_im, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            
            # Convert back to NumPy array (new_H, new_W, C)
            im = tensor_im.squeeze(0).permute(1, 2, 0).numpy()
                
        return im, (h0, w0), im.shape[:2]

    def build_transforms(self, hyp=None):
        """
        Build transforms for HSI data.
        Fixes:
        - Disables RGB-specific augmentations (mosaic, mixup, HSV jitter)
        - Uses zero-padding instead of 114-gray for PCA-normalized data
        """
        if hyp is not None:
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            hyp.hsv_h = 0.0
            hyp.hsv_s = 0.0
            hyp.hsv_v = 0.0

        transforms = super().build_transforms(hyp)

        # Fix LetterBox: PCA data is zero-centered, padding with 114 (gray) is wrong
        for t in transforms.transforms:
            if isinstance(t, LetterBox):
                t.color = (0, 0, 0)

        return transforms
