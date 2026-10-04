"""M2: side-by-side figure (reference | plain | Fourier), PSNR-over-steps plot and a results table.

Run after overfit.py has produced the runs. Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/compare_m2.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.material import load_config, load_image  # noqa: E402

CROP = 256


def center_crop(img):
    h, w = img.shape[:2]
    y, x = (h - CROP) // 2, (w - CROP) // 2
    return img[y:y + CROP, x:x + CROP]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, default=Path("train/configs/materials/Metal016.json"))
    p.add_argument("--runs", nargs="+", default=["plain", "fourier"])
    args = p.parse_args()

    cfg = load_config(args.material)
    root = Path("results") / cfg.name / "m2"
    metrics = {r: json.loads((root / r / "metrics.json").read_text(encoding="utf-8")) for r in args.runs}
    ch = next(c for c in cfg.channels if c.name == metrics[args.runs[0]]["channel"])
    ref = load_image(cfg.folder / ch.file)[..., :ch.count]

    # Side by side: full image on top, center crop below.
    images = [("reference", ref)] + [(f"{r}\n{metrics[r]['psnr']:.2f} dB", load_image(root / r / "decoded.png"))
                                     for r in args.runs]
    fig, axes = plt.subplots(2, len(images), figsize=(4 * len(images), 8.4))
    for col, (title, img) in enumerate(images):
        axes[0, col].imshow(img)
        axes[0, col].set_title(title)
        axes[1, col].imshow(center_crop(img), interpolation="nearest")
        axes[1, col].set_title(f"center {CROP}×{CROP}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)

    # PSNR over training steps.
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in args.runs:
        steps, values = zip(*metrics[r]["curve"])
        ax.plot(steps, values, label=r)
    ax.set_xlabel("training step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{cfg.name} {ch.name}: PSNR during training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "psnr_curve.png", dpi=150)
    plt.close(fig)

    # Results table.
    lines = [f"# M2 — {cfg.name} {ch.name}: coordinate MLP", "",
             "| Run | Fourier freqs | Params | Size (fp16) | bpt (fp16) | PSNR (dB) | SSIM | Train time |",
             "|---|---|---|---|---|---|---|---|"]
    for r in args.runs:
        m = metrics[r]
        lines.append(f"| {r} | {m['config']['num_freqs']} | {m['params']:,} | {m['size_bytes_fp16'] / 1024:.0f} KiB "
                     f"| {m['bpt_fp16']:.3f} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['train_seconds']:.0f} s |")
    lines += ["", "BC7 baseline for the same map: 8 bpt, 47.82 dB (`../baseline.md`).", "",
              "![comparison](comparison.png)", "", "![PSNR curve](psnr_curve.png)", ""]
    (root / "m2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {root / 'comparison.png'}, {root / 'psnr_curve.png'}, {root / 'm2.md'}")


if __name__ == "__main__":
    main()
