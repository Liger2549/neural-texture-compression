"""README figure: reference | BC | neural, center crops of albedo and normal, labelled with size and PSNR.

Run after baseline.py, train_ntc.py and eval_ntc.py. Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/readme_figure.py
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
from baseline import to_rgb8  # noqa: E402
from compare_m2 import center_crop  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, default=Path("train/configs/materials/Metal016.json"))
    p.add_argument("--run", default="base", help="M3 run name")
    p.add_argument("--maps", nargs="+", default=["albedo", "normal"])
    p.add_argument("--out", type=Path, default=Path("docs/images/comparison.png"))
    args = p.parse_args()

    cfg = load_config(args.material)
    res = Path("results") / cfg.name
    bc = json.loads((res / "baseline.json").read_text(encoding="utf-8"))
    ours = json.loads((res / "m3" / args.run / "eval.json").read_text(encoding="utf-8"))
    bc_rows = {r["map"]: r for r in bc["maps"]}
    our_rows = {r["map"]: r for r in ours["maps"]}

    fig, axes = plt.subplots(len(args.maps), 3, figsize=(10.5, 3.6 * len(args.maps)), squeeze=False)
    for row, name in zip(axes, args.maps):
        ch = next(c for c in cfg.channels if c.name == name)
        stem = Path(ch.file).stem
        images = [
            ("Reference", load_image(cfg.folder / ch.file)),
            (f"BC ({ch.bc_format.split('_')[0]})\n{bc_rows[name]['psnr']:.1f} dB",
             load_image(Path("results") / "tmp" / cfg.name / "nomips" / f"{stem}.png")),
            (f"Neural (latent grid + MLP)\n{our_rows[name]['psnr']:.1f} dB",
             load_image(res / "m3" / args.run / f"decoded_{name}.png")),
        ]
        for ax, (title, img) in zip(row, images):
            ax.imshow(to_rgb8(center_crop(img[..., :ch.count]), ch.normal), interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=12)

    fig.suptitle(f"{cfg.name}, 2048², center 256×256 crop. Size for all 7 channels: "
                 f"BC {bc['total_bpt']:.0f} bpt, neural {ours['bpt']:.1f} bpt", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
