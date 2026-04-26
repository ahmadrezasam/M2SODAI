import torch
import sys

def inspect_pth(path):
    try:
        # Try loading with weights_only=False but handle the missing module
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Keys: {ckpt.keys()}")
        if 'model' in ckpt:
            print(f"Model type: {type(ckpt['model'])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_pth(sys.argv[1])
