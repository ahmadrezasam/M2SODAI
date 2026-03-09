import torch
from torch import nn
from mmdet.models.builder import BACKBONES, build_backbone

@BACKBONES.register_module()
class ProjectUpsampleResNet(nn.Module):
    """Custom backbone that projects input channels to 3, spatially upsamples, 
    and then passes the output to a standard RGB ResNet.

    Args:
        in_channels (int): The number of channels of the original HSI input (e.g., 30 or 127).
        resnet_cfg (dict): The configuration dictionary for the inner ResNet backbone.
        target_size (tuple[int, int]): The desired spatial size to upsample to (e.g., (1600, 1600)).
    """
    def __init__(self, in_channels, resnet_cfg, target_size=(1600, 1600)):
        super(ProjectUpsampleResNet, self).__init__()
        
        # 1. Project channels from `in_channels` to 3
        # We use a 1x1 conv to merge the spectral bands.
        self.channel_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # 2. Spatial Upsampler
        self.target_size = target_size
        self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        
        # 3. Inner ResNet Backbone
        # We force in_channels to 3 because it will always receive the projected output
        resnet_cfg_copy = resnet_cfg.copy()
        resnet_cfg_copy['in_channels'] = 3
        self.resnet = build_backbone(resnet_cfg_copy)

    def init_weights(self):
        """Initialize the weights of the modules."""
        # Initialize the 1x1 projection layer
        nn.init.kaiming_normal_(self.channel_projector.weight, mode='fan_out', nonlinearity='relu')
        if self.channel_projector.bias is not None:
            nn.init.constant_(self.channel_projector.bias, 0)
            
        # Initialize the inner ResNet according to its config
        self.resnet.init_weights()

    def forward(self, x):
        """Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W) where C is `in_channels`
                        and (H, W) is typically small (e.g., 224, 224).
                        
        Returns:
            tuple[Tensor]: Features from the inner ResNet's output stages.
        """
        # Step 1: Spectral Projection (N, C, H, W) -> (N, 3, H, W)
        x = self.channel_projector(x)
        
        # Step 2: Spatial Alignment (N, 3, H, W) -> (N, 3, target_H, target_W)
        if x.shape[2:] != self.target_size:
             x = self.upsample(x)
             
        # Step 3: Pass through the frozen/pretrained RGB ResNet backbone
        outs = self.resnet(x)
        
        return outs
