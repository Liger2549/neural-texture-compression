"""M2: overfit one texture map with a coordinate MLP, (u, v) -> colour. Logs PSNR over training.

Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/overfit.py --run plain
    .venv\\Scripts\\python.exe train/scripts/overfit.py --run fourier --num-freqs 10
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.material import load_config, load_image  # noqa: E402
from ntc.metrics import psnr, ssim  # noqa: E402
from ntc.mlp import CoordMLP  # noqa: E402

CROP = 256  # size of the checkpoint crops (same center crop as the M1 baseline)


def texel_uvs(h: int, w: int, device: str) -> torch.Tensor:
    """UV of every texel center, row by row: (H*W, 2), u = x direction, v = y direction."""
    v, u = torch.meshgrid((torch.arange(h, device=device) + 0.5) / h,
                          (torch.arange(w, device=device) + 0.5) / w, indexing="ij")
    return torch.stack([u, v], dim=-1).reshape(-1, 2)


@torch.no_grad()
def decode(model: CoordMLP, uvs: torch.Tensor, h: int, w: int, chunk: int = 1 << 18) -> np.ndarray:
    """Run the model on every texel (in chunks to save memory). Returns (H, W, C) float32, 8-bit quantized."""
    out = torch.cat([model(uvs[i:i + chunk]) for i in range(0, len(uvs), chunk)])
    img = out.reshape(h, w, -1).cpu().numpy()
    return np.round(img * 255) / 255  # measure what an 8-bit texture would store, like the BC baseline


def to_png(img: np.ndarray) -> Image.Image:
    a = np.round(img * 255).astype(np.uint8)
    return Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a)


def center_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = (h - CROP) // 2, (w - CROP) // 2
    return img[y:y + CROP, x:x + CROP]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, default=Path("train/configs/materials/Metal016.json"))
    p.add_argument("--channel", default="albedo", help="which map of the material to overfit")
    p.add_argument("--run", required=True, help="run name, e.g. plain or fourier")
    p.add_argument("--num-freqs", type=int, default=0, help="0 = plain MLP, >0 = Fourier features")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=1 << 16, help="random texels per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--image-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.material)
    ch = next(c for c in cfg.channels if c.name == args.channel)
    out = Path("results") / cfg.name / "m2" / args.run
    (out / "steps").mkdir(parents=True, exist_ok=True)

    # Target: the map as stored in the PNG (albedo stays sRGB-encoded, same as the BC baseline measures it).
    ref = load_image(cfg.folder / ch.file)[..., :ch.count]
    h, w, c = ref.shape
    target = torch.from_numpy(ref).to(device).reshape(-1, c)
    uvs = texel_uvs(h, w, device)

    model = CoordMLP(c, args.hidden, args.layers, args.num_freqs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.run}: {model.num_params():,} params, {h}x{w}x{c} target, {device}")

    curve = []  # (step, PSNR on the full image)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, h * w, (args.batch,), device=device)  # random texels this step
        loss = torch.mean((model(uvs[idx]) - target[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            img = decode(model, uvs, h, w)
            curve.append((step, psnr(ref, img)))
            print(f"step {step:6d}  loss {loss.item():.6f}  PSNR {curve[-1][1]:.2f} dB")
            if step % args.image_every == 0 or step == args.steps:
                to_png(center_crop(img)).save(out / "steps" / f"step_{step:06d}.png")
    seconds = time.perf_counter() - start

    img = decode(model, uvs, h, w)
    to_png(img).save(out / "decoded.png")
    torch.save(model.state_dict(), out / "model.pt")

    params = model.num_params()
    metrics = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "material": cfg.name,
        "channel": ch.name,
        "config": vars(args) | {"material": str(args.material)},
        "params": params,
        "size_bytes_fp32": params * 4,
        "size_bytes_fp16": params * 2,
        "bpt_fp16": params * 16 / (h * w),
        "psnr": psnr(ref, img),
        "ssim": ssim(ref, img),
        "train_seconds": round(seconds, 1),
        "curve": curve,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"done in {seconds:.0f}s: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.4f}, "
          f"{metrics['bpt_fp16']:.3f} bpt (fp16) -> {out}")


if __name__ == "__main__":
    main()
