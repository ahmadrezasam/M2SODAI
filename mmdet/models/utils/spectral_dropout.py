# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class SpectralDropout(nn.Module):
    """Randomly zeros out entire spectral bands during training.

    This is analogous to Dropout but operates on the channel (spectral band)
    dimension. It forces the projector and backbone to not over-rely on any
    single spectral band, providing physically-motivated regularization.

    Args:
        p (float): Probability of zeroing out each band. Default: 0.3.
    """

    def __init__(self, p=0.3):
        super().__init__()
        assert 0.0 <= p < 1.0, f'Drop probability must be in [0, 1), got {p}'
        self.p = p

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): HSI input of shape (B, C_hsi, H, W).

        Returns:
            Tensor: Same shape, with random bands zeroed and scaled.
        """
        if not self.training or self.p == 0.0:
            return x
        # Create per-band mask: (1, C, 1, 1) — shared across batch and spatial
        mask = torch.bernoulli(
            torch.full((1, x.shape[1], 1, 1), 1.0 - self.p, device=x.device))
        # Inverted dropout scaling to maintain expected value
        return x * mask / (1.0 - self.p)

    def extra_repr(self):
        return f'p={self.p}'
