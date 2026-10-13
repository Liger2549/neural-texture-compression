"""M3: latent grid + tiny MLP decoder (see docs/M3-STEPS.md)."""
import torch
import torch.nn.functional as F
from torch import nn


class LatentGrid(nn.Module):
    """A learnable grid of codes, shape (1, channels, size, size). Each grid point holds `channels` numbers."""

    def __init__(self, channels: int = 8, size: int = 512):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1, channels, size, size) * 0.01)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """(N, 2) UVs in [0, 1] -> (N, channels) codes, bilinearly blended from the 4 nearest grid points."""
        coords = (uv * 2 - 1).reshape(1, -1, 1, 2)
        out = F.grid_sample(self.grid, coords, mode="bilinear", padding_mode="border", align_corners=False)
        return out.reshape(self.grid.shape[1], -1).T
        
