"""M4 step 8: quality vs bits-per-texel chart + results table from the sweep (run sweep_m4.py first).

Writes results/Metal016/m4/sweep/rd_chart.png and results/Metal016/m4/sweep/sweep.md.
Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/plot_m4.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SWEEP = Path("results/Metal016/m4/sweep")
BASELINE = Path("results/Metal016/baseline.json")
MAPS = ["albedo", "normal", "roughness", "metalness"]

# Reference palette (dataviz skill), light mode; first three categorical slots validate all-pairs.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# qat_ste is left off the chart: within 0.2 dB of qat_noise everywhere except full2 4-bit (table has all four).
METHODS = {  # key: (legend label, color, marker, marker area; triangles/diamonds look smaller at the same area)
    "qat_noise": ("QAT, noise only", "#2a78d6", "o", 56),
    "ptq": ("Quantized after training", "#1baf7a", "^", 80),
    "fp16": ("fp16, no quantization", MUTED, "D", 50),
}


LABELS = {  # (config, bits): label offset in points
    ("m8", 4): (0, 9), ("c8f2", 4): (0, 9), ("c8f4", 4): (0, 9), ("3lvl", 4): (0, 9),
    ("3lvl", 8): (0, 9), ("full2", 4): (0, -15),
}


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(SWEEP.glob("*/eval*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        run = f.parent.name
        config = run.split("_")[0]
        if run.endswith("_fp16"):
            method = "fp16" if s["grid_bits"] == 16 else "ptq"
        else:
            method = "qat_" + run.split("_")[-1]
        rows.append({
            "config": config, "method": method, "bits": s["grid_bits"], "bpt": s["bpt"], "psnr": s["psnr_all"],
            "maps": {m["map"]: m["psnr"] for m in s["maps"]},
            "channels": s["channels"], "sizes": s["sizes"],
        })
    return rows


def bc_point() -> tuple[float, float]:
    """BC bpt and PSNR over all 7 channels: per-map MSE weighted by channel count (same as psnr over all channels)."""
    b = json.loads(BASELINE.read_text(encoding="utf-8"))
    mse = sum(m["channels"] * 10 ** (-m["psnr"] / 10) for m in b["maps"]) / sum(m["channels"] for m in b["maps"])
    return b["total_bpt"], 10 * math.log10(1 / mse)


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both size (lower) and quality (higher), sorted by size."""
    front, best = [], -math.inf
    for x, y in sorted(points):
        if y > best:
            front.append((x, y))
            best = y
    return front


def plot(rows: list[dict], bc: tuple[float, float], path: Path) -> None:
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for key, (label, color, marker, size) in METHODS.items():
        pts = [(r["bpt"], r["psnr"]) for r in rows if r["method"] == key]
        if not pts:
            continue
        front = pareto(pts)
        ax.plot(*zip(*front), color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.scatter(*zip(*pts), s=size, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=3,
                   label=label)

    # Direct labels on a few QAT points only (the table has every number).
    for r in rows:
        if r["method"] == "qat_noise" and (r["config"], r["bits"]) in LABELS:
            dx, dy = LABELS[(r["config"], r["bits"])]
            ax.annotate(f"{r['config']} {r['bits']}-bit", (r["bpt"], r["psnr"]), xytext=(dx, dy),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_2)

    ax.scatter([bc[0]], [bc[1]], s=300, marker="*", color=INK, edgecolors=SURFACE, linewidths=2, zorder=4,
               label="BC baseline (texconv)")
    ax.annotate(f"BC  {bc[1]:.1f} dB", bc, xytext=(-8, -16), textcoords="offset points", ha="right",
                fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(1.6, 40)
    ax.set_xlabel("bits per texel, all 7 channels (log scale, lower = smaller)", color=INK_2)
    ax.set_ylabel("PSNR over all channels (dB, higher = better)", color=INK_2)
    ax.set_title("Metal016: quality vs size, neural configs vs BC", loc="left", color=INK, fontsize=12, pad=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=9)
    fig.text(0.01, 0.01, "Lines connect each method's best trade-offs (Pareto front). 5,000 steps, seed 42, 2048² material.\n"
             "QAT noise → STE is within 0.2 dB of noise only (except full2 4-bit), see sweep.md.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def to_markdown(rows: list[dict], bc: tuple[float, float]) -> str:
    order = {k: i for i, k in enumerate(["fp16", "ptq", "qat_noise", "qat_ste"])}
    lines = [
        "# M4 sweep — Metal016, quality vs bits per texel",
        "",
        "Generated by `train/scripts/plot_m4.py` from `train/scripts/sweep_m4.py` runs (5,000 steps, seed 42).",
        "PSNR in dB on 8-bit values. Methods: fp16 = no quantization; ptq = grids quantized after training;",
        "qat_noise = quantization-aware training with noise only; qat_ste = noise, then STE from 70% of the steps.",
        "",
        f"BC baseline: {bc[0]:.0f} bpt, {bc[1]:.2f} dB over all channels.",
        "",
        "![quality vs bpt](rd_chart.png)",
        "",
        "| Config | Grids (channels × size²) | Method | Grid bits | bpt | PSNR all | " + " | ".join(MAPS) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(MAPS),
    ]
    for r in sorted(rows, key=lambda r: (r["config"], r["bits"], order[r["method"]])):
        grids = " + ".join(f"{c} × {s}²" for c, s in zip(r["channels"], r["sizes"]))
        lines.append(f"| {r['config']} | {grids} | {r['method']} | {r['bits']} | {r['bpt']:.2f} | {r['psnr']:.2f} | "
                     + " | ".join(f"{r['maps'][m]:.2f}" for m in MAPS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    bc = bc_point()
    plot(rows, bc, SWEEP / "rd_chart.png")
    (SWEEP / "sweep.md").write_text(to_markdown(rows, bc), encoding="utf-8")
    print(f"{len(rows)} results -> {SWEEP / 'rd_chart.png'}, {SWEEP / 'sweep.md'}")


if __name__ == "__main__":
    main()
