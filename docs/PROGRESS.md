# Progress Log

## 2026-10-21 — M3: Latent grid + tiny decoder ✅

**Done**
- Branch `m3-latent-grid` from `main` (after the M2 merge, #3). Step-by-step plan in `docs/milestones/M3-STEPS.md`
- `train/ntc/latent.py`, written by me:
  - `LatentGrid`: learnable `(1, 8, 512, 512)` grid with a bilinear lookup (`F.grid_sample`, texel-centre convention like GPUs), `(N, 2)` UVs → `(N, 8)` codes
  - `TinyMLP`: 8 → 64 → 64 → 7, ReLU, sigmoid output (5,191 params)
  - `NTCModel`: lookup → MLP, `(N, 2)` → `(N, 7)`
  - `weighted_l2`: per-channel MSE with configurable weights (normals ×2 by default)
- `train/scripts/train_ntc.py`: Adam with two learning rates (grid 1e-2, MLP 1e-3), random batches of 65,536 texels, PSNR log every 250 steps. Optimizer and training step written by me, the rest by the AI
- `train/scripts/eval_ntc.py`: full decode, per-channel PSNR/SSIM, normal angle error, bpt, crops vs BC. bpt line written by me, the rest by the AI
- 8 new pytest tests (`train/tests/test_latent.py`), 22 in total, all passing

**Results (Metal016, 2048², run `base`, 5,000 steps, 23 s on an RTX 3070 Laptop, `results/Metal016/m3/base/eval.md`)**

| Map | PSNR ours (dB) | PSNR BC (dB) | SSIM ours | SSIM BC |
|---|---|---|---|---|
| albedo | 31.02 | 47.82 | 0.835 | 0.997 |
| normal | 30.96 (3.31°) | 47.01 (0.54°) | 0.722 | 0.994 |
| roughness | 38.20 | 52.18 | 0.925 | 0.997 |
| metalness | 31.79 | 41.08 | 0.961 | 0.998 |
| **Size** | **8.02 bpt** (2.1M params, fp16) | **24 bpt** | | |

- Size: 8.02 bpt for all 7 channels, about 3× smaller than BC (24 bpt).
- Quality: roughness is the best channel (38.20 dB) and normal the worst (30.96 dB, albedo almost the same). Every channel is still 9–17 dB below BC.
- Look: our output is blurrier than the reference and fine detail is less visible. The error is highest where the texture has a lot of small detail (`crops/normal.png`).
- Why: the grid is 512×512 for a 2048×2048 texture, so one grid point covers 4×4 texels, and bilinear blending between grid points can't make details smaller than that. M4 adds multi-resolution grids (a finer grid for small details) and quantization to keep the size low.
**Open issues**
- Experiments not run yet: more steps (still +0.2 dB per 1,000 steps at step 5,000), learning rates, normal weight
- sRGB vs linear albedo → D9. Trained on sRGB-encoded values (same as the BC baseline); decision not written down yet

**Next:** M4 — multi-resolution grids + quantization.

## 2026-10-04 — M2: Single-texture MLP ✅

**Done**
- `train/ntc/mlp.py`: coordinate MLP `(u, v) → RGB`, optional Fourier features. `fourier_features` written by me
- `train/scripts/overfit.py`: trains on random texel batches, logs full-image PSNR every 250 steps, saves checkpoint crops, config and metrics to `results/Metal016/m2/<run>/`
- `train/scripts/compare_m2.py`: side-by-side figure, PSNR curve, results table (`results/Metal016/m2/m2.md`)
- `matplotlib` added (D8); 14 pytest tests passing
- `docs/concepts/spectral-bias.md`: why plain MLPs are blurry (written by me, restructured and grammar-checked with AI)

**Results (Metal016 albedo, 2048², 5000 steps, `results/Metal016/m2/m2.md`)**

| Run | Params | bpt (fp16) | PSNR (dB) | SSIM |
|---|---|---|---|---|
| plain MLP | 198,915 | 0.76 | 22.31 | 0.361 |
| Fourier features (10 freqs) | 208,643 | 0.80 | 25.91 | 0.469 |
| BC7 baseline | | 8 | 47.82 | 0.997 |

- Plain MLP: only the overall layout survives, a smooth blur (spectral bias).
- Fourier features: +3.6 dB at the same size; patches and spots appear, but close-up crops are still soft.
- One MLP can't memorize 4M texels of detail. This motivates M3: store the detail in a latent grid and keep the MLP small.

**Open issues**
- Linear vs sRGB albedo for training is still open (decide in M3, see D7/D8)
- Before M3: skim NVIDIA's "Random-Access Neural Compression of Material Textures" (overview, architecture, results)

**Next:** M3 — latent grid + tiny decoder.

## 2026-09-29 — M1: BC baseline ✅

**Done**
- Material config system: `train/configs/materials/Metal016.json` + config-driven loader (`train/ntc/material.py`). Multiple materials planned, each trained separately (DECISIONS.md D6)
- `rebuild_normal_z` and `psnr` written by me; SSIM and normal angle error added (`train/ntc/metrics.py`)
- `train/scripts/baseline.py`: texconv compress → decode → measure. Measurement choices in D7
- 9 pytest tests passing (`train/tests/`)
- `docs/concepts/bc-compression.md`: BC explainer (written by me)
- `docs/AI-USAGE.md`: what I did vs. what the AI assistant did

**Results (Metal016, 2048², `results/Metal016/baseline.md`)**

| Map | Format | bpt | PSNR (dB) | SSIM |
|---|---|---|---|---|
| albedo | BC7 sRGB | 8 | 47.82 | 0.9970 |
| normal | BC5 | 8 | 47.01 (0.54° angle error) | 0.9942 |
| roughness | BC4 | 4 | 52.18 | 0.9971 |
| metalness | BC4 | 4 | 41.08 | 0.9976 |
| **Total** | | **24** | 12 MiB (16 MiB with mips) | |

- Metalness has the lowest PSNR: error is only on patch edges, where one 4×4 BC4 block must span 0–255 with 8 levels.
- BC is already above 40 dB everywhere, so the realistic NTC goal is similar quality at fewer bits per texel.

**Open issues**
- Optional: A/B BC7 default vs `-bc x` quality (D7)

**Next:** M2 — overfit a single texture with an MLP (plain vs Fourier features).

## 2026-09-28 — M0: Repository setup ✅

**Environment**
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8 GB)
- Python 3.13, PyTorch 2.14.0 + CUDA 13.0 (GPU training confirmed working)
- Windows, Git LFS for textures and binaries

**Done**
- Repo structure created (train/, runtime/, assets/, results/, docs/)
- Git LFS tracking set up before adding any images
- Python virtual environment + requirements.txt (direct dependencies pinned; fresh install tested, CUDA works)
- Material chosen: `Metal016` (ambientCG, 2K PNG). 7 channels: albedo RGB, normal XY (NormalDX), roughness, metalness. No AO (DECISIONS.md D5)
- Other candidates (Ground054, Ground068, Marble016, Metal055A) compared and deleted before the first commit
- README, ROADMAP, PROGRESS, DECISIONS created

**Open issues**
- None

**Next:** M1 — BC baseline (compress the material with texconv, measure size and quality)