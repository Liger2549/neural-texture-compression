# Progress Log

## 2026-10-25 — M4: Multi-resolution latents + quantization ✅

**Done**
- Branch `m4-multires-quant` from `main` (after the M3 merge, #4). Step plan in `docs/milestones/M4-STEPS.md`
- `train/ntc/latent.py`, written by me:
  - `grid_bpt(size, channels, bits_per_value, tex_size)`: size of one grid in bits per texel
  - `MultiResGrid`: several `LatentGrid`s in an `nn.ModuleList`; looks up each one and concatenates the codes. `NTCModel` takes lists `channels`, `sizes` (channels first everywhere); the MLP input is the sum of the channels
  - `quantize` (round onto `2**bits` levels between the grid's min and max), `quantize_noise` (±½ step uniform noise), `quantize_ste` (straight-through estimator; no clamp, because clamp gave 0 gradient to values on the edge levels)
  - `LatentGrid.mode` / `bits`: the grid is quantized before sampling
- `train_ntc.py`: the noise → STE schedule written by me; `--channels`, `--sizes`, `--grid-bits`, `--ste-from` flags and logging of the rounded model's PSNR by the AI
- `eval_ntc.py`: fp16 round trip for grids and MLP stored at 16 bits written by me; `--grid-bits` (quantize after loading) and the size breakdown (grids, scale/offset, MLP) by the AI
- `sweep_m4.py`, `plot_m4.py` (AI): 6 configs × 4 methods × 4/8-bit, chart and table in `results/Metal016/m4/sweep/`
- 11 new pytest tests, 33 in total, all passing
- Decisions: D10 (per-grid min/max range, 4/8-bit), D11 (noise-only QAT; default is now `--ste-from 1.0`)

**Results 1: one grid vs several (fp16, 5,000 steps, `results/Metal016/m4/<run>/train.json`)**

| Run | Grids (channels × size²) | bpt | PSNR all channels (dB) |
|---|---|---|---|
| `f2` | 2 × 1024² | 8 | 30.99 |
| `single` (M3 setup) | 8 × 512² | 8 | 31.67 |
| `f4` | 4 × 1024² | 16 | 33.28 |
| `c8f4` | 8 × 256² + 4 × 1024² | 18 | 33.88 |

- Same size (8 bpt): `single` beats `f2`. A fine grid alone is not enough; 2 numbers per texel are too few to describe all 7 channels.
- `f4` beats `single`: more numbers, and closer together (one grid point per 2×2 texels instead of 4×4), so smaller details survive.
- `c8f4` vs `f4`: the coarse grid adds +0.6 dB for only +2 bpt. The coarse grid gives many channels cheaply, the fine grid gives resolution with few channels.

**Results 2: quantizing `c8f4` (`results/Metal016/m4/`)**

| Grids | Method | bpt | PSNR all (dB) |
|---|---|---|---|
| fp16 | – | 18.02 | 33.88 |
| 8-bit | quantized after training | 9.02 | 33.86 |
| 4-bit | quantized after training | 4.52 | 30.22 |
| 4-bit | QAT noise → STE (seed 0 / 1) | 4.52 | 32.15 / 32.31 |
| 4-bit | QAT noise only (seed 0 / 1) | 4.52 | 32.23 / 32.39 |
| 4-bit | QAT STE only | 4.52 | 31.92 |
| 2-bit | quantized after training | 2.27 | 9.88 |

- MLP in fp16 instead of fp32: −0.02 bpt, PSNR unchanged (32.23 dB both). The output is stored as 8 bits anyway, so the fp16 error disappears in the final rounding.

**Results 3: sweep (6 configs, 5,000 steps, seed 42, `results/Metal016/m4/sweep/sweep.md`)**

![quality vs bpt](../results/Metal016/m4/sweep/rd_chart.png)

Best result per config and bit depth (QAT noise only), next to quantizing after training:

| Config | Grids | 4-bit bpt | 4-bit PSNR (after → QAT) | 8-bit bpt | 8-bit PSNR (QAT) |
|---|---|---|---|---|---|
| `m8` | 8 × 512² | 2.02 | 27.07 → 29.86 | 4.02 | 31.58 |
| `c8f2` | 8 × 256² + 2 × 1024² | 2.52 | 29.91 → 31.42 | 5.02 | 32.50 |
| `c8f4` | 8 × 256² + 4 × 1024² | 4.52 | 30.22 → 32.26 | 9.02 | 33.77 |
| `c16f4` | 16 × 256² + 4 × 1024² | 5.02 | 30.80 → 32.44 | 10.02 | 33.84 |
| `3lvl` | 16 × 128² + 8 × 512² + 4 × 1024² | 6.27 | 31.28 → 32.83 | 12.52 | 34.70 |
| `full2` | 2 × 2048² | 8.02 | 29.42 → 30.55 | 16.02 | 33.48 |
| **BC** | BC7 + BC5 + 2 × BC4 | | | **24** | **46.1** |

**My findings**
1. **8-bit:** QAT doesn't help at 8-bit. An 8-bit step is 17× smaller than a 4-bit step, so rounding barely changes the codes: quantizing after training is already within 0.05 dB of fp16.
2. **4-bit:** QAT gains 1.1–2.8 dB over quantizing after training. The best 4-bit model is `3lvl` with noise only (32.8 dB at 6.3 bpt). `m8` gains the most (+2.8 dB), probably because it has only one grid, so every rounding error goes straight into the MLP's input.
3. **More grid or more bits?** At ~5 bpt both are about equal on this material: `c8f2` 8-bit gets 32.50 dB and `c16f4` 4-bit gets 32.44 dB, a smaller difference than changing the seed (0.16 dB). More bits make each stored value more accurate; more grid gives more detail and more numbers for the coarse shapes. Which one wins probably depends on the material's patterns.
4. **Why `full2` is the worst:** it has one grid point per texel, so it can store fine detail everywhere, but only 2 numbers per texel for all 7 channels, and no coarse grid. The 2 numbers went mostly to albedo (its 8-bit albedo, 36.8 dB, is the best of all configs), while roughness and metalness are the worst of all configs. It also learns slowly: each of its 4M grid points is updated only about 300 times in 5,000 steps.
5. **Acceptance:** No, not yet. The best neural result is 34.8 dB at 12.5 bpt, against BC's 46.1 dB at 24 bpt. These models only trained for 5,000 steps. Longer training might improve them; if it doesn't, this is the best this setup can do. Viewed as a whole texture, the 34.8 dB result doesn't look bad, and it uses half of BC's memory on this material. I don't want to jump to conclusions from PSNR alone: the real test is how it looks in a real renderer (M7–M9).

**Open issues**
- Closing the gap to BC, ideas from the NTC paper (Vaidyanathan et al. 2023): much longer training (250k iterations with ~8× bigger batches, vs our 5,000), cosine learning-rate decay to 0, a fixed quantization range with clamping in the backward pass only, a final phase with quantized latents frozen and only the MLP fine-tuned, and positional encoding as extra MLP input
- Learning-rate schedule (cosine decay, or reduce-on-plateau + early stop): QAT runs flatten out around step 3,000 with a constant learning rate; fp16 runs are still improving at step 5,000
- Entropy coding of the quantized grids for the file on disk (4-bit fine grid: 1.7 bits/value entropy): idea for M6, see D10
- Fixed-range quantization (clip outliers) instead of min/max: might help 4-bit, see D10
- Roughness got worse than M3 in some configs while normals improved (normal loss weight ×2)
- sRGB vs linear albedo → D9, still not written down (from M3)

**Next:** M4b — closing the gap (ablations from the NTC paper), then M5 — mipmaps.

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