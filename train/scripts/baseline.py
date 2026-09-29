"""M1 baseline: compress each map of a material with BC (texconv), decode it back, measure size and quality.

Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/baseline.py --material train/configs/materials/Metal016.json
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntc.material import load_config, load_image, rebuild_normal_z  # noqa: E402
from ntc.metrics import normal_angle_error, psnr, ssim  # noqa: E402

DDS_HEADER = 128        # "DDS " magic + DDS_HEADER
DDS_DX10_HEADER = 20    # extra header used by BC7 and other newer formats
CROP = 256              # size of the comparison crops


def texconv(exe: Path, args: list[str]) -> None:
    subprocess.run([str(exe), "-nologo", "-y", *args], check=True, capture_output=True, text=True)


def dds_payload_bytes(path: Path) -> int:
    """File size minus the DDS header(s): the bytes the GPU actually stores."""
    raw = path.read_bytes()
    header = DDS_HEADER + (DDS_DX10_HEADER if raw[84:88] == b"DX10" else 0)
    return len(raw) - header


def compress_and_decode(exe: Path, src: Path, bc_format: str, srgb: bool, mips: bool, out: Path) -> tuple[Path, Path]:
    """Encode src to a BC .dds, then decode that .dds back to an 8-bit PNG. Returns (dds, png)."""
    out.mkdir(parents=True, exist_ok=True)
    flags = ["-srgb"] if srgb else []  # tell texconv the data is already sRGB-encoded (no conversion)
    texconv(exe, [*flags, "-f", bc_format, "-m", "0" if mips else "1", "-o", str(out), str(src)])
    dds = out / (src.stem + ".dds")
    decoded_fmt = "R8G8B8A8_UNORM_SRGB" if srgb else "R8G8B8A8_UNORM"
    texconv(exe, [*flags, "-ft", "png", "-f", decoded_fmt, "-m", "1", "-o", str(out), str(dds)])
    return dds, out / (src.stem + ".png")


def to_rgb8(a: np.ndarray, normal: bool) -> np.ndarray:
    """Float [0,1] map with 1-3 channels -> uint8 RGB for viewing."""
    if normal:
        a = rebuild_normal_z(a) * 0.5 + 0.5
    elif a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_crop(ref: np.ndarray, test: np.ndarray, normal: bool, path: Path) -> None:
    """Side by side: reference | BC | |error| x8, from the image center, scaled 2x."""
    h, w = ref.shape[:2]
    y, x = (h - CROP) // 2, (w - CROP) // 2
    r, t = ref[y:y + CROP, x:x + CROP], test[y:y + CROP, x:x + CROP]
    err = np.clip(np.abs(r - t).max(axis=-1, keepdims=True) * 8, 0, 1)
    row = np.concatenate([to_rgb8(r, normal), to_rgb8(t, normal), to_rgb8(err, False)], axis=1)
    Image.fromarray(row).resize((row.shape[1] * 2, row.shape[0] * 2), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--material", type=Path, required=True, help="material config JSON")
    ap.add_argument("--texconv", type=Path, default=Path("tools/texconv.exe"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = load_config(args.material)
    out = args.out / cfg.name
    tmp = args.out / "tmp" / cfg.name  # .dds and decoded .png files (gitignored)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    rows = []
    for ch in cfg.channels:
        src = cfg.folder / ch.file
        srgb = ch.color_space == "srgb"
        ref = load_image(src)[..., :ch.count]
        h, w = ref.shape[:2]

        dds, png = compress_and_decode(args.texconv, src, ch.bc_format, srgb, mips=False, out=tmp / "nomips")
        dds_mips, _ = compress_and_decode(args.texconv, src, ch.bc_format, srgb, mips=True, out=tmp / "mips")
        test = load_image(png)[..., :ch.count]

        size = dds_payload_bytes(dds)
        size_mips = dds_payload_bytes(dds_mips)
        row = {
            "map": ch.name,
            "channels": ch.count,
            "format": ch.bc_format,
            "color_space": ch.color_space,
            "bytes": size,
            "bytes_with_mips": size_mips,
            "bpt": size * 8 / (w * h),
            "bpt_with_mips": size_mips * 8 / (w * h),
            "psnr": psnr(ref, test),
            "ssim": ssim(ref, test),
        }
        if ch.normal:
            row["angle_error_deg"] = normal_angle_error(ref, test)
        rows.append(row)
        save_crop(ref, test, ch.normal, out / "crops" / f"{ch.name}.png")
        print(f"{ch.name:10s} {ch.bc_format:15s} {row['bpt']:5.2f} bpt  PSNR {row['psnr']:6.2f} dB  SSIM {row['ssim']:.4f}")

    total = sum(r["bytes"] for r in rows)
    total_mips = sum(r["bytes_with_mips"] for r in rows)
    summary = {
        "material": cfg.name,
        "config": str(args.material),
        "resolution": [w, h],
        "date": datetime.now().isoformat(timespec="seconds"),
        "texconv": "default quality settings",
        "maps": rows,
        "total_bytes": total,
        "total_bytes_with_mips": total_mips,
        "total_bpt": total * 8 / (w * h),
        "total_bpt_with_mips": total_mips * 8 / (w * h),
    }
    (out / "baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "baseline.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"total      {summary['total_bpt']:.2f} bpt, {total / 2**20:.2f} MiB ({total_mips / 2**20:.2f} MiB with mips)")
    print(f"wrote {out / 'baseline.md'}")


def to_markdown(s: dict) -> str:
    w, h = s["resolution"]
    lines = [
        f"# BC baseline — {s['material']}",
        "",
        f"{w}×{h}, texconv {s['texconv']}. Generated by `train/scripts/baseline.py` on {s['date']}.",
        "Albedo metrics are computed on sRGB-encoded values; all other maps are linear.",
        "",
        "| Map | Ch | Format | Size (MiB) | bpt | Size w/ mips (MiB) | PSNR (dB) | SSIM | Normal angle error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["maps"]:
        angle = f"{r['angle_error_deg']:.2f}°" if "angle_error_deg" in r else "–"
        lines.append(
            f"| {r['map']} | {r['channels']} | {r['format']} | {r['bytes'] / 2**20:.2f} | {r['bpt']:.1f} | "
            f"{r['bytes_with_mips'] / 2**20:.2f} | {r['psnr']:.2f} | {r['ssim']:.4f} | {angle} |"
        )
    lines.append(
        f"| **Total** | {sum(r['channels'] for r in s['maps'])} | | **{s['total_bytes'] / 2**20:.2f}** | "
        f"**{s['total_bpt']:.1f}** | **{s['total_bytes_with_mips'] / 2**20:.2f}** | | | |"
    )
    lines += ["", "Crops (center 256×256, shown 2×): reference | BC | error ×8 — see `crops/`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
