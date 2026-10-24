# Neural Texture Compression

Compressing PBR materials into learned latent grids + a tiny MLP, decoded in real time on the GPU, compared against traditional BC compression.

![Reference vs BC vs neural compression](docs/images/comparison.png)

**Status:** work in progress. With multi-resolution grids and 4-bit quantization-aware training (M4), one model stores all 7 channels of a material in **2.5–6 bpt, 4–10× smaller than BC (24 bpt)**, at 31–33 dB PSNR. BC is still far sharper (46 dB), see [results](#results-so-far). Next: mipmaps (M5), then a real-time GPU decoder (M7).

## How it works

```
         ┌─ coarse grid (256², 8 ch) ─► 8 numbers ─┐   big shapes
UV ──────┤                                           ├─ concat ─► 12 numbers ─► tiny MLP (12 → 64 → 64 → 7) ─► 7 channels
         └─ fine grid (1024², 4 ch)  ─► 4 numbers ─┘   small detail

each latent stored as a 4- or 8-bit integer (+ one fp16 min and step per grid)
```

- **Per-material overfitting.** Each material gets its own model, trained only on that material. Gradient descent is the encoder; the trained parameters *are* the compressed file.
- **Latent grids + shared decoder.** The detail lives in learnable grids of codes at several resolutions, sampled bilinearly like GPU textures. A coarse grid gives many numbers per texel cheaply; a fine grid gives resolution with few numbers. One tiny MLP (about 5,400 params) shared by every texel turns the codes into albedo RGB, normal XY, roughness and metalness at once.
- **Quantization-aware training.** The grids are stored as 4- or 8-bit integers. During training the grids get ±½ step of random noise (a stand-in for rounding that still has a gradient), so the model learns codes that survive the final rounding. At 4-bit this recovers about 2 dB of the 3.7 dB that rounding a normally trained model costs.
- **Training.** Random batches of 65,536 texels, per-channel weighted L2 loss (normals weighted ×2), Adam with a higher learning rate for the grid than for the MLP.
- **Random access.** Every texel decodes on its own, so the decoder can run per pixel in a shader.

## Results so far

Metal016 (2048², 7 channels), 5,000 steps per model (25–40 s on an RTX 3070 Laptop GPU). Full tables: [BC baseline](results/Metal016/baseline.md), [M2](results/Metal016/m2/m2.md), [M3](results/Metal016/m3/base/eval.md), [M4 sweep](results/Metal016/m4/sweep/sweep.md).

### Quality vs size (M4)

![Quality vs bits per texel: neural configs vs BC](results/Metal016/m4/sweep/rd_chart.png)

| Model | Grids (channels × size²) | Size, all 7 channels | PSNR, all channels |
|---|---|---|---|
| BC (BC7 + BC5 + 2 × BC4) | | 24 bpt | 46.1 dB |
| M3: one grid, fp16 | 8 × 512² | 8.02 bpt | 31.57 dB |
| M4 `c8f2`, 4-bit QAT | 8 × 256² + 2 × 1024² | 2.52 bpt | 31.42 dB |
| M4 `c8f4`, 4-bit QAT | 8 × 256² + 4 × 1024² | 4.52 bpt | 32.26 dB |
| M4 `3lvl`, 4-bit QAT | 16 × 128² + 8 × 512² + 4 × 1024² | 6.27 bpt | 32.83 dB |
| M4 `3lvl`, 8-bit | same | 12.52 bpt | 34.70 dB |

- Multi-resolution + 4-bit QAT reaches M3's quality at **a third of the size** (2.5 vs 8 bpt).
- 8-bit quantization is free (within 0.05 dB of fp16); 4-bit needs quantization-aware training.
- No neural config gets near BC's quality: the best is 34.8 dB at 12.5 bpt (`3lvl`, 8-bit), 11 dB below BC at half its size.

### M3 per map

| Map | BC: PSNR (dB) | Neural M3: PSNR (dB) |
|---|---|---|
| albedo | 47.82 | 31.02 |
| normal | 47.01 (0.54° angle error) | 30.96 (3.31°) |
| roughness | 52.18 | 38.20 |
| metalness | 41.08 | 31.79 |
| **Size, all 7 channels** | **24 bpt** | **8.02 bpt** (fp16) |

How the model got here (albedo):

| Step | Model | Albedo PSNR (dB) |
|---|---|---|
| M2 | one MLP, UV → colour | 22.31 |
| M2 | + Fourier features | 25.91 |
| M3 | latent grid + tiny MLP, all 7 channels | 31.02 |

M2 used a 10× smaller budget (about 0.8 bpt for albedo only), so the M2 → M3 jump is not like for like. M2 showed that a single MLP can't memorise fine detail (spectral bias); M3 moves the detail into the grid.

**Where it fails:** fine detail is still blurred compared with BC. In M3 the limit was the grid resolution (one grid point per 4×4 texels); in M4 finer grids help, but every config levels off around 32–35 dB in 5,000 steps.

## Milestones
- [X] M0 Repository setup
- [X] M1 BC baseline
- [X] M2 Single-texture MLP
- [X] M3 Latent grid + tiny decoder ([step plan](docs/milestones/M3-STEPS.md))
- [X] M4 Multi-resolution + quantization ([step plan](docs/milestones/M4-STEPS.md))
- [ ] M4b Closing the gap (ablations from the NTC paper)
- [ ] M5 Mipmaps
- [ ] M6 Export format + reference decoder
- [ ] M7 Real-time GPU decoder
- [ ] M8 Fallback mode + auto selection
- [ ] M9 Measurement + write-up

## Run it yourself

Windows, Python 3.13, NVIDIA GPU recommended (CPU works but is slow). Textures are stored with Git LFS.

```
git lfs pull
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python.exe train/scripts/baseline.py --material train/configs/materials/Metal016.json
.venv\Scripts\python.exe train/scripts/train_ntc.py
.venv\Scripts\python.exe train/scripts/eval_ntc.py

# M4: two grids, 4-bit quantization-aware training (noise only), then the full sweep + chart
.venv\Scripts\python.exe train/scripts/train_ntc.py --run c8f4_q4 --channels 8 4 --sizes 256 1024 --grid-bits 4
.venv\Scripts\python.exe train/scripts/eval_ntc.py --run c8f4_q4
.venv\Scripts\python.exe train/scripts/sweep_m4.py
.venv\Scripts\python.exe train/scripts/plot_m4.py
.venv\Scripts\python.exe -m pytest train/tests
```

The BC baseline needs `texconv.exe` from [DirectXTex](https://github.com/microsoft/DirectXTex/releases) in `tools/`.

## Docs

- [ROADMAP.md](docs/ROADMAP.md): goals, milestones, acceptance criteria
- [DECISIONS.md](docs/DECISIONS.md): design decisions and why
- [PROGRESS.md](docs/PROGRESS.md): what was done and measured, per milestone
- [concepts/](docs/concepts/): short explainers ([BC compression](docs/concepts/bc-compression.md), [spectral bias](docs/concepts/spectral-bias.md))

## AI assistance
This project is built with help from an AI coding assistant. What I did vs. what the assistant did is listed per milestone in [docs/AI-USAGE.md](docs/AI-USAGE.md).

## Credits

- Material: [Metal016](https://ambientcg.com/view?id=Metal016) from ambientCG (CC0)
- Technique: Vaidyanathan et al., [*Random-Access Neural Compression of Material Textures*](https://research.nvidia.com/labs/rtr/neural_texture_compression/), SIGGRAPH 2023
- BC compression: Microsoft [DirectXTex](https://github.com/microsoft/DirectXTex) (texconv)
