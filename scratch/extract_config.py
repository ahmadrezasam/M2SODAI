import torch

ckpt_path = "orcn-vitaev2-s-dota-latest.pth"
ckpt = torch.load(ckpt_path, map_location='cpu')

if 'meta' in ckpt and 'config' in ckpt['meta']:
    print("EXTRACTED CONFIG:")
    print(ckpt['meta']['config'])
else:
    print("No config found in metadata.")
