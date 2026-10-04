# Progress Log

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