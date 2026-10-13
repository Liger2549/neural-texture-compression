"""M3: train a latent grid + tiny MLP decoder on a whole material (all channels at once).

Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/train_ntc.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.material import load_config, load_material  # noqa: E402
from ntc.latent import LatentGrid # noqa: E402
from overfit import texel_uvs  # noqa: E402

MATERIAL = Path("train/configs/materials/Metal016.json")
BATCH = 1 << 16  # random texels per step (65,536)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(MATERIAL)
    ref = load_material(cfg)
    h, w, c = ref.shape

    target = torch.from_numpy(ref).to(device).reshape(-1, c)

    uvs = texel_uvs(h, w, device)

    idx = torch.randint(0, h * w, (BATCH,), device=device)
    uv_batch = uvs[idx]
    value_batch = target[idx]

    print("target", tuple(target.shape), target.device, "min", target.min().item(), "max", target.max().item())
    print("uv_batch", tuple(uv_batch.shape), "value_batch", tuple(value_batch.shape))

    grid = LatentGrid().to(device)
    print("grid", tuple(grid.grid.shape), f"{grid.grid.numel():,} values", "min", grid.grid.min().item(), "max", grid.grid.max().item())
    
    codes = grid(uv_batch)
    print("codes", tuple(codes.shape))

if __name__ == "__main__":
    main()
