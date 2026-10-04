import math

import torch

from ntc.mlp import CoordMLP, fourier_features


def test_fourier_shape():
    uv = torch.rand(10, 2)
    assert fourier_features(uv, 6).shape == (10, 24)


def test_fourier_values_at_origin():
    # uv = 0: every sin is 0 and every cos is 1, so exactly half the features are 1.
    f = fourier_features(torch.zeros(1, 2), 4)
    assert torch.allclose(f.sum(), torch.tensor(8.0))
    assert torch.all((f == 0) | (f == 1))


def test_fourier_contains_expected_frequency():
    # u = 0.25 at k = 1: sin(2 * pi * 0.25) = 1 must appear somewhere in the features.
    f = fourier_features(torch.tensor([[0.25, 0.0]]), 2)
    assert torch.isclose(f, torch.tensor(1.0), atol=1e-6).any()
    assert torch.isclose(f, torch.tensor(math.sin(math.pi * 0.25)), atol=1e-6).any()  # k = 0


def test_plain_mlp_output_shape_and_range():
    model = CoordMLP(out_channels=3, hidden=32, layers=2, num_freqs=0)
    y = model(torch.rand(100, 2))
    assert y.shape == (100, 3)
    assert torch.all((y >= 0) & (y <= 1))


def test_fourier_mlp_output_shape():
    model = CoordMLP(out_channels=3, hidden=32, layers=2, num_freqs=5)
    assert model(torch.rand(100, 2)).shape == (100, 3)
