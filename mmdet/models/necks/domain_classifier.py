# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from ..builder import NECKS
from ..utils import GradientReversalLayer

@NECKS.register_module()
class DomainDiscriminator(nn.Module):
    """Domain Discriminator for Adversarial UDA.
    
    It takes feature maps from multiple levels (e.g., from FPN) and
    predicts the domain labels.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=256,
                 num_convs=3,
                 grl_alpha=1.0):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        layers = []
        curr_channels = in_channels
        for _ in range(num_convs):
            layers.append(nn.Conv2d(curr_channels, hidden_channels, 3, padding=1))
            layers.append(nn.GroupNorm(32, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            curr_channels = hidden_channels
        
        self.convs = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, 1) # Binary classification: RGB (0) vs HSI (1)

    def forward(self, x):
        """Forward pass for a single feature level or a list of features.
        If x is a list (from FPN), we can process each level and average or just take one.
        Usually, aligning at multiple levels is better.
        """
        if isinstance(x, (list, tuple)):
            # Process multiple levels and average the domain scores
            scores = []
            for feat in x:
                feat = self.grl(feat)
                feat = self.convs(feat)
                feat = self.avg_pool(feat).view(feat.size(0), -1)
                scores.append(self.fc(feat))
            return sum(scores) / len(scores)
        else:
            x = self.grl(x)
            x = self.convs(x)
            x = self.avg_pool(x).view(x.size(0), -1)
            return self.fc(x)
