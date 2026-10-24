"""M3/M4: latent grid(s) + tiny MLP decoder (see docs/milestones/M3-STEPS.md and M4-STEPS.md)."""
import torch
import torch.nn.functional as F
from torch import nn


class LatentGrid(nn.Module):
    """A learnable grid of codes, shape (1, channels, size, size). Each grid point holds `channels` numbers."""

    def __init__(self, channels: int = 8, size: int = 512):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1, channels, size, size) * 0.01)
        self.bits = 16      # target bit depth (16 = off, no quantization)
        self.mode = "off"   # "off", "noise" or "ste"

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """(N, 2) UVs in [0, 1] -> (N, channels) codes, bilinearly blended from the 4 nearest grid points."""
        
        g = self.grid
        if self.mode == "noise":
            g = quantize_noise(g, self.bits)
        elif self.mode == "ste":
            g = quantize_ste(g, self.bits)
        
        coords = (uv * 2 - 1).reshape(1, -1, 1, 2)
        out = F.grid_sample(g, coords, mode="bilinear", padding_mode="border", align_corners=False)
        return out.reshape(g.shape[1], -1).T
        
class TinyMLP(nn.Module):
    """(N, in_dim) codes -> (N, out_dim) values in [0, 1]."""
    
    def __init__(self, in_dim: int = 8, hidden: int = 64, out_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.net(codes)
    
class NTCModel(nn.Module):
    """(N, 2) UVs -> (N, out_dim) values in [0, 1]: grid lookup, then the tiny MLP."""
    
    def __init__(self, out_dim: int, channels: list[int] = [8], sizes: list[int] = [512], hidden: int = 64):
        super().__init__()
        self.grid = MultiResGrid(channels, sizes)
        self.mlp = TinyMLP(sum(channels), hidden, out_dim)
        
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        codes = self.grid(uv)
        return self.mlp(codes)
    
    
def weighted_l2(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """pred, target (N, C); weights (C,) -> one number: per-channel MSE, weighted average."""
        
    err = (pred - target) ** 2
    per_channel = err.mean(dim=0)                           # average down the columns
    loss = (per_channel * weights).sum() / weights.sum()
    return loss

def grid_bpt(size: int, channels: int, bits_per_value: int, tex_size: int) -> float:
    """Bits per texel of one (size × size × channels) grid, each value stored at `bits_per_value` bits,
    for a tex_size × tex_size texture."""

    # (size² · channels · bits_per_value) / tex_size²: size counts squared, channels and bits linearly
    return channels * bits_per_value * pow(size / tex_size, 2)

class MultiResGrid(nn.Module):
    """Several latent grids at different sizes. (N, 2) UVs -> (N, sum of channels) codes."""
    
    def __init__(self, channels: list[int], sizes: list[int]):
        super().__init__()
        self.grids = nn.ModuleList([LatentGrid(c, s) for c, s in zip(channels, sizes)]) # one LatentGrid per (channels, size) pair, in an nn.ModuleList
        self.out_dim = sum(channels)
        
    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        codes = []
        for g in self.grids:
            codes.append(g(uv))
        return torch.cat(codes, dim=1)

def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Snap x onto 2**bits evenly spaced levels between its min and max, returned as floats."""
    
    lo, hi = x.min(), x.max()
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    q = torch.clamp(torch.round((x - lo) / step), 0, levels)
    return q * step + lo
     
def quantize_noise(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Training stand-in for quantize: add up to ±½ step of uniform noise instead of rounding."""
    
    lo, hi = x.min().detach(), x.max().detach()
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    q = torch.clamp((x - lo) / step + torch.rand_like(x) - 0.5, 0, levels)
    return q * step + lo
    
def quantize_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Training version of quantize: forward rounds onto the levels, backward passes the gradient straight through."""
    
    lo, hi = x.min().detach(), x.max().detach()
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    q = (x - lo) / step  
    
    q = q + (torch.round(q) - q).detach()
    
    # Clamp doesn't need, since its value is between min and max.    
    return  q * step + lo

    