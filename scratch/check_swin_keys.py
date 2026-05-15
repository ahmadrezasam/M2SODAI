import torch
from mmdet.models.backbones import SwinTransformer

model = SwinTransformer(
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7)

print("Model state_dict keys (first 10):")
print(list(model.state_dict().keys())[:10])
