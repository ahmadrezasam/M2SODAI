import torch

ckpt_path = "orcn-vitaev2-s-dota-latest.pth"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("First 20 keys:")
    for k in keys[:20]:
        print(k)
    
    # Check for backbone type if possible (metadata)
    if 'meta' in ckpt:
        print("\nMetadata found:")
        if 'config' in ckpt['meta']:
            print("Config found in metadata!")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
