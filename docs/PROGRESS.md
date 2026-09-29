# Progress Log

## 2026-09-29 — M1: BC baseline (in progress)

**Done**
- Material config system: `train/configs/materials/Metal016.json` + config-driven loader (`train/ntc/material.py`). Multiple materials planned, each trained separately (DECISIONS.md D6)
- `rebuild_normal_z` and `psnr` written by me; SSIM and normal angle error added (`train/ntc/metrics.py`)
- `train/scripts/baseline.py`: texconv compress → decode → measure. Measurement choices in D7
- 9 pytest tests passing (`train/tests/`)
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
- `docs/concepts/bc-compression.md` explainer (me) not written yet
- Optional: A/B BC7 default vs `-bc x` quality (D7)

**Next:** finish the BC explainer, then close M1 and start M2 (single-texture MLP).

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