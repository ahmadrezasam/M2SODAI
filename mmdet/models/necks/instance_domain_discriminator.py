# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from ..builder import NECKS
from ..utils import GradientReversalLayer


@NECKS.register_module()
class InstanceDomainDiscriminator(nn.Module):
    """Instance-Level Domain Discriminator for Adversarial UDA.

    Unlike the image-level DomainDiscriminator which operates on FPN feature
    maps, this discriminator operates on RoI-pooled features (e.g., 256x7x7)
    to perform instance-level domain alignment.

    Args:
        in_channels (int): Number of channels of the RoI-pooled features.
            Default: 256.
        roi_feat_size (int): Spatial size of RoI features. Default: 7.
        hidden_dim (int): Hidden layer dimension. Default: 1024.
        grl_alpha (float): Scale factor for the gradient reversal layer.
            Default: 1.0.
    """

    def __init__(self,
                 in_channels=256,
                 roi_feat_size=7,
                 hidden_dim=1024,
                 grl_alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        feat_dim = in_channels * roi_feat_size * roi_feat_size
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),  # Binary: RGB (0) vs HSI (1)
        )

    def forward(self, roi_feats):
        """Forward pass on RoI-pooled features.

        Args:
            roi_feats (Tensor): Shape (N_rois, C, H, W) from RoI pooling.

        Returns:
            Tensor: Domain logits of shape (N_rois, 1).
        """
        x = self.grl(roi_feats)
        x = x.flatten(1)  # (N_rois, C*H*W)
        return self.classifier(x)
