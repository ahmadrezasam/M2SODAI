import torch
from torch import nn
from mmdet.models.builder import BACKBONES, build_backbone
import torch.nn.functional as F
import copy


class SpectralChannelAttention(nn.Module):
    """Squeeze-and-Excitation style attention on spectral channels.

    Learns which intermediate spectral band groups are most informative
    for the downstream detection task.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck. Default: 4.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = self.attention(x)  # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)


@BACKBONES.register_module()
class CascadedProjectUpsampleResNet(nn.Module):
    """Backbone with a cascaded spectral projector instead of a single 1×1 conv.

    The projector reduces HSI bands in two stages with channel attention
    between them, preserving much more spectral information:

        HSI (in_channels) → [1×1 Conv + BN + GELU] → mid_channels
                          → [Spectral Channel Attention]
                          → [1×1 Conv + BN] → 3
                          → [Bilinear Upsample]
                          → Pre-trained ResNet-50

    Args:
        in_channels (int): Number of HSI bands (e.g., 30).
        mid_channels (int): Intermediate channel count. Default: 16.
        resnet_cfg (dict): Config for the inner ResNet backbone.
        target_size (tuple[int, int]): Spatial upsample target. Default: None.
        attention_reduction (int): SE-attention reduction ratio. Default: 4.
    """

    def __init__(self, in_channels, resnet_cfg, mid_channels=16,
                 target_size=None, attention_reduction=4):
        super().__init__()

        # Stage 1: in_channels → mid_channels (with activation)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # Stage 2: Spectral channel attention
        self.spectral_attention = SpectralChannelAttention(
            mid_channels, reduction=attention_reduction)

        # Stage 3: mid_channels → 3 (output for pretrained backbone)
        self.stage2 = nn.Sequential(
            nn.Conv2d(mid_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )

        # Expose channel_projector as a property so existing code that
        # accesses self.backbone.channel_projector still works
        self.channel_projector = nn.Sequential(
            self.stage1,
            self.spectral_attention,
            self.stage2,
        )

        self.target_size = target_size

        resnet_cfg_copy = copy.deepcopy(resnet_cfg)
        resnet_cfg_copy['in_channels'] = 3
        self.resnet = build_backbone(resnet_cfg_copy)

    def init_stage1_from_pretrained(self, pretrained_weight):
        """Initialize stage1 from the pretrained 1×1 conv [3, in_ch, 1, 1].

        Spreads the 3 RGB-mapping rows across mid_channels by repeating
        and adding slight noise for symmetry breaking.

        Args:
            pretrained_weight (Tensor): Shape [3, in_channels, 1, 1].
        """
        mid_ch = self.stage1[0].weight.shape[0]
        in_ch = pretrained_weight.shape[1]
        # Repeat the 3 rows to fill mid_channels
        repeated = pretrained_weight.repeat(
            (mid_ch + 2) // 3, 1, 1, 1)[:mid_ch]
        # Add small noise for symmetry breaking
        noise = torch.randn_like(repeated) * 0.01
        self.stage1[0].weight.data.copy_(repeated + noise)

    def init_stage2_identity_like(self):
        """Initialize stage2 so mid_channels→3 approximates band selection.

        Maps the first 3 of mid_channels directly to the 3 output channels.
        """
        nn.init.zeros_(self.stage2[0].weight)
        # First 3 channels pass through (identity-like)
        for i in range(3):
            self.stage2[0].weight.data[i, i] = 1.0

    def forward(self, x):
        x = self.stage1(x)                     # (B, in_ch, H, W) → (B, mid, H, W)
        x = self.spectral_attention(x)         # reweight channels
        x = self.stage2(x)                     # (B, mid, H, W) → (B, 3, H, W)

        if self.target_size and x.shape[2:] != tuple(self.target_size):
            x = F.interpolate(x, size=self.target_size,
                              mode='bilinear', align_corners=False)

        return self.resnet(x)
