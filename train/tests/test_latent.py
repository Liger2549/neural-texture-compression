import torch

from ntc.latent import LatentGrid, MultiResGrid, NTCModel, TinyMLP, grid_bpt, weighted_l2


def test_grid_lookup_shape():
    grid = LatentGrid(channels=8, size=16)
    assert grid(torch.rand(100, 2)).shape == (100, 8)


def test_grid_lookup_at_texel_center_returns_that_code():
    # UV of grid point (row i, column j) is ((j + 0.5) / S, (i + 0.5) / S): no blending, the exact code comes back.
    grid = LatentGrid(channels=3, size=4)
    i, j = 1, 2
    out = grid(torch.tensor([[(j + 0.5) / 4, (i + 0.5) / 4]]))
    assert torch.allclose(out[0], grid.grid[0, :, i, j], atol=1e-6)


def test_grid_lookup_halfway_is_average():
    # Halfway between two neighbouring grid points (same row): bilinear gives their mean.
    grid = LatentGrid(channels=3, size=4)
    out = grid(torch.tensor([[2 / 4, 0.5 / 4]]))  # between columns 1 and 2, on row 0
    expected = (grid.grid[0, :, 0, 1] + grid.grid[0, :, 0, 2]) / 2
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_tiny_mlp_shape_range_and_params():
    mlp = TinyMLP(in_dim=8, hidden=64, out_dim=7)
    y = mlp(torch.randn(100, 8) * 10)  # large inputs: sigmoid must still keep the output in [0, 1]
    assert y.shape == (100, 7)
    assert torch.all((y >= 0) & (y <= 1))
    assert sum(p.numel() for p in mlp.parameters()) == 5191  # (8*64+64) + (64*64+64) + (64*7+7)


def test_model_shape_and_params():
    model = NTCModel(out_dim=7, channels=[8], sizes=[16], hidden=64)
    assert model(torch.rand(50, 2)).shape == (50, 7)
    assert sum(p.numel() for p in model.parameters()) == 8 * 16 * 16 + 5191


def test_weighted_l2_hand_example():
    # Worked example from M3 step 6.
    pred = torch.full((2, 3), 0.5)
    target = torch.tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 0.5]])
    loss = weighted_l2(pred, target, torch.tensor([2.0, 1.0, 1.0]))
    assert torch.isclose(loss, torch.tensor(0.15625))


def test_weighted_l2_all_ones_is_mse():
    pred, target = torch.rand(64, 7), torch.rand(64, 7)
    loss = weighted_l2(pred, target, torch.ones(7))
    assert torch.isclose(loss, torch.mean((pred - target) ** 2))


def test_training_reduces_loss_and_updates_both_parts():
    # A few Adam steps on a tiny 8x8 target: the loss must drop, and both the grid and the MLP must change.
    torch.manual_seed(0)
    model = NTCModel(out_dim=3, channels=[4], sizes=[4], hidden=16)
    v, u = torch.meshgrid((torch.arange(8) + 0.5) / 8, (torch.arange(8) + 0.5) / 8, indexing="ij")
    uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)
    target = torch.rand(64, 3)
    weights = torch.ones(3)
    grid_before = model.grid.grids[0].grid.detach().clone()
    mlp_before = model.mlp.net[0].weight.detach().clone()
    opt = torch.optim.Adam([
        {"params": model.grid.parameters(), "lr": 1e-2},
        {"params": model.mlp.parameters(), "lr": 1e-3},
    ])

    first = weighted_l2(model(uvs), target, weights).item()
    for _ in range(200):
        loss = weighted_l2(model(uvs), target, weights)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < first * 0.8
    assert not torch.equal(model.grid.grids[0].grid, grid_before)
    assert not torch.equal(model.mlp.net[0].weight, mlp_before)


def test_grid_bpt():
    assert grid_bpt(512, 8, 16, 2048) == 8.0   # M3 grid, fp16
    assert grid_bpt(512, 8, 4, 2048) == 2.0    # same grid, 4-bit
    assert grid_bpt(1024, 4, 4, 2048) == 4.0   # 2x size = 4x points, half the channels -> 2x the cost


def test_multires_grid_shape_and_params():
    grid = MultiResGrid(sizes=[4, 16], channels=[3, 2])
    assert grid(torch.rand(10, 2)).shape == (10, 5)
    assert grid.out_dim == 5
    assert sum(p.numel() for p in grid.parameters()) == 4 * 4 * 3 + 16 * 16 * 2


def test_multires_grid_concatenates_in_order():
    # Columns are the first grid's codes, then the second grid's codes, for the same UVs.
    grid = MultiResGrid(sizes=[4, 16], channels=[3, 2])
    uv = torch.rand(10, 2)
    out = grid(uv)
    assert torch.allclose(out[:, :3], grid.grids[0](uv))
    assert torch.allclose(out[:, 3:], grid.grids[1](uv))


def test_model_multires_mlp_input_is_sum_of_channels():
    model = NTCModel(out_dim=7, channels=[8, 4], sizes=[4, 16], hidden=64)
    assert model(torch.rand(5, 2)).shape == (5, 7)
    assert model.mlp.net[0].in_features == 12
