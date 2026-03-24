import torch
from torch import nn
from mmdet.models.builder import BACKBONES, build_backbone
import torch.nn.functional as F
import copy

@BACKBONES.register_module()
class ProjectUpsampleResNet(nn.Module):
    """Custom backbone that projects input channels to 3, spatially upsamples, 
    and then passes the output to a standard RGB ResNet.

    Args:
        in_channels (int): The number of channels of the original HSI input (e.g., 30 or 127).
        resnet_cfg (dict): The configuration dictionary for the inner ResNet backbone.
        target_size (tuple[int, int]): The desired spatial size to upsample to (e.g., (1600, 1600)).
    """

    def __init__(self, in_channels, resnet_cfg, target_size=None):
        super().__init__()
        
        self.channel_projector = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )
        
        self.target_size = target_size  # Only set if spatial alignment truly needed

        resnet_cfg_copy = copy.deepcopy(resnet_cfg)
        resnet_cfg_copy['in_channels'] = 3
        self.resnet = build_backbone(resnet_cfg_copy)

    def forward(self, x):
        x = self.channel_projector(x)          # (N, C, H, W) -> (N, 3, H, W)
        
        if self.target_size and x.shape[2:] != tuple(self.target_size):
            x = F.interpolate(x, size=self.target_size, 
                            mode='bilinear', align_corners=False)
        
        return self.resnet(x)
