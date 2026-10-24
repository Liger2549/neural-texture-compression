"""M3/M4 eval: decode trained latent grid(s) + MLP at full resolution and compare it per channel with the BC baseline.

Run after train_ntc.py (and baseline.py for the BC numbers). Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/eval_ntc.py
    .venv\\Scripts\\python.exe train/scripts/eval_ntc.py --run fast
    .venv\\Scripts\\python.exe train/scripts/eval_ntc.py --run c8f4 --grid-bits 4   # quantize the grids after training
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.latent import NTCModel, quantize  # noqa: E402
from ntc.material import load_config, load_image, load_material  # noqa: E402
from ntc.metrics import normal_angle_error, psnr, ssim  # noqa: E402
from baseline import CROP, to_rgb8  # noqa: E402
from overfit import decode, texel_uvs, to_png  # noqa: E402


def save_crop(ref: np.ndarray, bc: np.ndarray, ours: np.ndarray, normal: bool, path: Path) -> None:
    """Side by side: reference | BC | ours | |ours - ref| x8, from the image center, scaled 2x."""
    h, w = ref.shape[:2]
    y, x = (h - CROP) // 2, (w - CROP) // 2
    r, b, o = (a[y:y + CROP, x:x + CROP] for a in (ref, bc, ours))
    err = np.clip(np.abs(r - o).max(axis=-1, keepdims=True) * 8, 0, 1)
    row = np.concatenate([to_rgb8(r, normal), to_rgb8(b, normal), to_rgb8(o, normal), to_rgb8(err, False)], axis=1)
    Image.fromarray(row).resize((row.shape[1] * 2, row.shape[0] * 2), Image.NEAREST).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, default=Path("train/configs/materials/Metal016.json"))
    p.add_argument("--milestone", default="m4", help="milestone folder used in train_ntc.py")
    p.add_argument("--run", default="base", help="run name used in train_ntc.py")
    p.add_argument("--grid-bits", type=int, default=None,
                   help="bits per latent value (16 = fp16, <16 = quantize after loading; default: the run's --grid-bits)")
    p.add_argument("--mlp-bits", type=int, default=16, help="bits per MLP weight (16 = fp16)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.material)
    run_dir = Path("results") / cfg.name / args.milestone / args.run
    train_cfg = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))["config"]
    if args.grid_bits is None:
        args.grid_bits = train_cfg.get("grid_bits", 16)
    bc_dir = Path("results") / "tmp" / cfg.name / "nomips"  # BC-decoded PNGs written by baseline.py
    baseline = json.loads((Path("results") / cfg.name / "baseline.json").read_text(encoding="utf-8"))
    bc_rows = {r["map"]: r for r in baseline["maps"]}
    tag = "" if args.grid_bits == 16 else f"_q{args.grid_bits}"  # quantized evals get their own files
    (run_dir / f"crops{tag}").mkdir(parents=True, exist_ok=True)

    ref = load_material(cfg)
    h, w, c = ref.shape
    # Same grids as in training; M3 runs saved no grid flags and used one 512² grid with 8 channels.
    model = NTCModel(out_dim=c, channels=train_cfg.get("channels", [8]), sizes=train_cfg.get("sizes", [512])).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    if "grid.grid" in state:  # M3 checkpoint: one LatentGrid, before MultiResGrid existed
        state["grid.grids.0.grid"] = state.pop("grid.grid")
    model.load_state_dict(state)
    model.eval()
    if args.grid_bits < 16:
        with torch.no_grad():
            for g in model.grid.grids:
                g.grid.copy_(quantize(g.grid, args.grid_bits))
    ours = decode(model, texel_uvs(h, w, device), h, w)  # (H, W, C), 8-bit quantized like a stored texture

    params = sum(p.numel() for p in model.parameters())
    grid_values = sum(g.grid.numel() for g in model.grid.grids)
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    scale_offset_bits = 2 * 16 * len(model.grid.grids) if args.grid_bits < 16 else 0  # one fp16 min + step per grid
    total_bits = grid_values * args.grid_bits + scale_offset_bits + mlp_params * args.mlp_bits
    texels = h * w
    bpt = total_bits / texels

    rows = []
    for ch in cfg.channels:
        sl = slice(ch.start, ch.start + ch.count)
        r, o = ref[..., sl], ours[..., sl]
        bc = load_image(bc_dir / (Path(ch.file).stem + ".png"))[..., :ch.count]
        row = {"map": ch.name, "channels": ch.count, "psnr": psnr(r, o), "ssim": ssim(r, o)}
        if ch.normal:
            row["angle_error_deg"] = normal_angle_error(r, o)
        rows.append(row)
        save_crop(r, bc, o, ch.normal, run_dir / f"crops{tag}" / f"{ch.name}.png")
        to_png(o if not ch.normal else np.concatenate([o, np.zeros_like(o[..., :1])], axis=-1)).save(
            run_dir / f"decoded_{ch.name}{tag}.png")
        b = bc_rows[ch.name]
        print(f"{ch.name:10s} PSNR {row['psnr']:6.2f} dB (BC {b['psnr']:6.2f})  "
              f"SSIM {row['ssim']:.4f} (BC {b['ssim']:.4f})")

    summary = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "material": cfg.name,
        "run": args.run,
        "channels": train_cfg.get("channels", [8]),
        "sizes": train_cfg.get("sizes", [512]),
        "resolution": [w, h],
        "params": params,
        "grid_values": grid_values,
        "mlp_params": mlp_params,
        "grid_bits": args.grid_bits,
        "mlp_bits": args.mlp_bits,
        "size_bytes": total_bits // 8,
        "bpt": bpt,
        "bc_bpt": baseline["total_bpt"],
        "psnr_all": psnr(ref, ours),
        "maps": rows,
    }
    (run_dir / f"eval{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / f"eval{tag}.md").write_text(to_markdown(summary, bc_rows), encoding="utf-8")
    print(f"size       {bpt:.2f} bpt (BC {baseline['total_bpt']:.1f}), PSNR all channels {summary['psnr_all']:.2f} dB")
    print(f"wrote {run_dir / f'eval{tag}.md'}")


def to_markdown(s: dict, bc: dict) -> str:
    w, h = s["resolution"]
    lines = [
        f"# Eval — {s['material']}, run `{s['run']}`" + (f", grids quantized to {s['grid_bits']} bits after training" if s["grid_bits"] < 16 else ""),
        "",
        f"{w}×{h}, latent grids (channels {s['channels']}, sizes {s['sizes']}) + tiny MLP. Generated by `train/scripts/eval_ntc.py` on {s['date']}.",
        "Both sides measured on 8-bit values; albedo on sRGB-encoded values (same as the BC baseline).",
        "",
        "| Map | Ch | PSNR ours | PSNR BC | SSIM ours | SSIM BC | Normal angle ours | Normal angle BC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in s["maps"]:
        b = bc[r["map"]]
        a = f"{r['angle_error_deg']:.2f}°" if "angle_error_deg" in r else "–"
        ab = f"{b['angle_error_deg']:.2f}°" if "angle_error_deg" in b else "–"
        lines.append(f"| {r['map']} | {r['channels']} | {r['psnr']:.2f} | {b['psnr']:.2f} | "
                     f"{r['ssim']:.4f} | {b['ssim']:.4f} | {a} | {ab} |")
    lines += [
        "",
        f"**Size:** {s['grid_values']:,} latents × {s['grid_bits']} bits + {s['mlp_params']:,} MLP weights × "
        f"{s['mlp_bits']} bits = {s['size_bytes'] / 2**20:.2f} MiB "
        f"→ **{s['bpt']:.2f} bpt** for all 7 channels (BC: {s['bc_bpt']:.1f} bpt).",
        f"PSNR over all channels: {s['psnr_all']:.2f} dB.",
        "",
        "Crops (center 256×256, shown 2×): reference | BC | ours | |ours − reference| ×8 — see `crops/`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
