import os
import sys
import torch

sys.path.append("/home/ahmadreza/Downloads/Research/M2SODAI")

# Monkey-patch torch.load
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = custom_torch_load

# Monkey-patch Ultralytics to recognize .npy files
import ultralytics.data.utils as data_utils
if 'npy' not in data_utils.IMG_FORMATS:
    data_utils.IMG_FORMATS.add('npy')

# Monkey-patch PIL Image.open to handle .npy files
from PIL import Image
original_image_open = Image.open
class DummyImage:
    def __init__(self):
        self.format = 'PNG'
        self.size = (640, 640)
    def verify(self):
        pass
    def getexif(self):
        return None

def custom_image_open(fp, *args, **kwargs):
    if str(fp).endswith('.npy'):
        return DummyImage()
    return original_image_open(fp, *args, **kwargs)
Image.open = custom_image_open

from hsi_dataset import HSIDataset

class DummyHyp:
    def __getattr__(self, name):
        if name == 'auto_augment':
            return None
        if name == 'copy_paste_mode':
            return 'flip'
        return 0.0

def main():
    img_path = "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/fold_2_hsi/train/images"
    print(f"Testing HSIDataset caching on: {img_path}")
    
    # We instantiate a tiny version of HSIDataset or run standard init
    # YOLODataset requires 'data' dict with names and classes
    data_dict = {
        "names": {0: "ship"},
        "nc": 1,
        "path": "/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/fold_2_hsi",
    }
    
    try:
        dataset = HSIDataset(
            img_path=img_path,
            imgsz=224,
            batch_size=8,
            augment=True,
            hyp=DummyHyp(),
            rect=False,
            cache=None,
            single_cls=True,
            stride=32,
            pad=0.0,
            data=data_dict,
            task="obb",
        )
        
        print("\nSUCCESS! Dataset instantiated successfully.")
        print(f"Dataset use_shared_cache: {getattr(dataset, 'use_shared_cache', False)}")
        if getattr(dataset, "use_shared_cache", False):
            print(f"Cache Tensor shape: {dataset.shared_cache.shape}")
            print(f"Cache is in shared memory: {dataset.shared_cache.is_shared()}")
            
            # Fetch a sample
            im, shape, size = dataset.load_image(0)
            print(f"Sample loaded successfully. Image shape: {im.shape}, shape metadata: {shape}, size: {size}")
            
    except Exception as e:
        print("\nERROR occurred during dataset caching initialization:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
