"""M3/M4: train latent grid(s) + tiny MLP decoder on a whole material (all channels at once).

Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/train_ntc.py                    # M3 setup: one 512² grid, 8 channels
    .venv\\Scripts\\python.exe train/scripts/train_ntc.py --run fast --steps 500
    .venv\\Scripts\\python.exe train/scripts/train_ntc.py --run c8f4 --channels 8 4 --sizes 256 1024
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.material import load_config, load_material  # noqa: E402
from ntc.latent import NTCModel, weighted_l2  # noqa: E402
from ntc.metrics import psnr  # noqa: E402
from overfit import decode, texel_uvs  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, default=Path("train/configs/materials/Metal016.json"))
    p.add_argument("--milestone", default="m4", help="results go to results/<material>/<milestone>/<run>")
    p.add_argument("--run", default="base", help="run name")
    p.add_argument("--channels", type=int, nargs="+", default=[8], help="channels per latent grid, e.g. 8 4")
    p.add_argument("--sizes", type=int, nargs="+", default=[512], help="size of each latent grid, e.g. 256 1024")
    p.add_argument("--grid-bits", type=int, default=16, help="quantization-aware training to this many bits (16 = off)")
    p.add_argument("--ste-from", type=float, default=0.7, help="fraction of steps after which noise switches to STE (1 = noise only)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=1 << 16, help="random texels per step")
    p.add_argument("--lr-grid", type=float, default=1e-2, help="learning rate for the latent grid (higher)")
    p.add_argument("--lr-mlp", type=float, default=1e-3, help="learning rate for the MLP (lower)")
    p.add_argument("--normal-weight", type=float, default=2.0, help="loss weight for the normal channels")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.channels) != len(args.sizes):
        p.error("--channels and --sizes need one value per grid")
    cfg = load_config(args.material)
    out = Path("results") / cfg.name / args.milestone / args.run
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: the whole material as (H*W, C) values, plus the UV of every texel.
    ref = load_material(cfg)
    h, w, c = ref.shape
    target = torch.from_numpy(ref).to(device).reshape(-1, c)
    uvs = texel_uvs(h, w, device)

    # Step 6: per-channel loss weights from the config (normals count more).
    weights = torch.ones(c, device=device)
    for ch in cfg.channels:
        if ch.normal:
            weights[ch.start:ch.start + ch.count] = args.normal_weight

    model = NTCModel(out_dim=c, channels=args.channels, sizes=args.sizes).to(device)
    for g in model.grid.grids:
        g.bits = args.grid_bits
    params = sum(p.numel() for p in model.parameters())
    qat = f", QAT {args.grid_bits}-bit (STE from {args.ste_from:.0%})" if args.grid_bits < 16 else ""
    print(f"{args.run}: {params:,} params, {h}x{w}x{c} target, {device}{qat}")

    opt = torch.optim.Adam([
        {"params": model.grid.parameters(), "lr": args.lr_grid},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ])

    curve = []  # (step, PSNR over all channels on the full material)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        # ── YOUR PART (M4 6c): set every grid's mode for this step ("noise" early, "ste" late; skip if grid_bits is 16) ──
        if (args.grid_bits != 16):
            mode = "noise" if (step < args.steps * args.ste_from) else "ste"
            for g in model.grid.grids:
                g.mode = mode

        # ── YOUR PART (M3 7b): one training step ──
        idx = torch.randint(0, h * w, (args.batch,), device=device)
        pred = model(uvs[idx])
        loss = weighted_l2(pred, target[idx], weights)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            # Log the quality the stored model would really have: rounded grids (STE forward = quantize), no noise.
            modes = [g.mode for g in model.grid.grids]
            for g in model.grid.grids:
                g.mode = "ste" if args.grid_bits < 16 else "off"
            curve.append((step, psnr(ref, decode(model, uvs, h, w))))
            for g, m in zip(model.grid.grids, modes):
                g.mode = m
            print(f"step {step:6d}  loss {loss.item():.6f}  PSNR {curve[-1][1]:.2f} dB")
    seconds = time.perf_counter() - start

    torch.save(model.state_dict(), out / "model.pt")
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "material": cfg.name,
        "config": vars(args) | {"material": str(args.material)},
        "params": params,
        "train_seconds": round(seconds, 1),
        "curve": curve,
    }
    (out / "train.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done in {seconds:.0f}s: PSNR {curve[-1][1]:.2f} dB -> {out}")


if __name__ == "__main__":
    main()
