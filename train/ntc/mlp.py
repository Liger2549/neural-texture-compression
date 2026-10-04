"""M2: a coordinate MLP that maps a UV position to a colour, optionally through Fourier features."""
import torch
from torch import nn


def fourier_features(uv: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """(N, 2) UVs in [0, 1] -> (N, 4 * num_freqs) features.

    For each frequency k = 0 .. num_freqs-1 and each of u, v:
        sin(2^k * pi * uv), cos(2^k * pi * uv)
    """
    freqs = 2 ** torch.arange(num_freqs, device=uv.device) * torch.pi
    
    X = uv[:, :, None] * freqs
    X = X.reshape(len(uv), -1)
    
    return torch.cat([torch.sin(X), torch.cos(X)], dim=1)


class CoordMLP(nn.Module):
    """(N, 2) UVs -> (N, out_channels) values in [0, 1].

    num_freqs == 0: the raw UV goes straight into the MLP ("plain").
    num_freqs > 0:  the UV is expanded with fourier_features first.
    """

    def __init__(self, out_channels: int = 3, hidden: int = 256, layers: int = 4, num_freqs: int = 0):
        super().__init__()
        self.num_freqs = num_freqs
        in_dim = 2 if num_freqs == 0 else 4 * num_freqs
        dims = [in_dim] + [hidden] * layers
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU()]
        blocks += [nn.Linear(hidden, out_channels), nn.Sigmoid()]  # sigmoid keeps the output in [0, 1]
        self.net = nn.Sequential(*blocks)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        x = uv if self.num_freqs == 0 else fourier_features(uv, self.num_freqs)
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
