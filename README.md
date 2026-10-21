# Neural Texture Compression

Compressing PBR materials into learned latent grids + a tiny MLP, decoded in real time on the GPU, compared against traditional BC compression.

![Reference vs BC vs neural compression](docs/images/comparison.png)

**Status:** work in progress. The training pipeline works end to end (M3): one model stores all 7 channels of a material in **8 bpt, 3× smaller than BC (24 bpt)**, but it is still blurrier. Next: multi-resolution grids and quantization-aware training (M4), then a real-time GPU decoder (M7).

## How it works

```
UV ──► latent grid (512×512, 8 learned numbers per point) ──► 8 numbers ──► tiny MLP (8 → 64 → 64 → 7) ──► 7 channels
       "where things are"                                                    "how to read a code"
```

- **Per-material overfitting.** Each material gets its own model, trained only on that material. Gradient descent is the encoder; the trained parameters *are* the compressed file.
- **Latent grid + shared decoder.** The detail lives in a learnable grid of codes, sampled bilinearly like a GPU texture. One tiny MLP (5,191 params) shared by every texel turns a code into albedo RGB, normal XY, roughness and metalness at once.
- **Training.** Random batches of 65,536 texels, per-channel weighted L2 loss (normals weighted ×2), Adam with a higher learning rate for the grid than for the MLP.
- **Random access.** Every texel decodes on its own, so the decoder can run per pixel in a shader.

## Results so far

Metal016 (2048², 7 channels), 5,000 steps (23 s on an RTX 3070 Laptop GPU). Full tables: [BC baseline](results/Metal016/baseline.md), [M2](results/Metal016/m2/m2.md), [M3](results/Metal016/m3/base/eval.md).

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

**Where it fails:** details smaller than one grid cell (4×4 texels) get blurred, because bilinear blending can't make them. M4 adds finer grids at more resolutions, and quantization to keep the size down.

## Milestones
- [X] M0 Repository setup
- [X] M1 BC baseline
- [X] M2 Single-texture MLP
- [X] M3 Latent grid + tiny decoder ([step plan](docs/milestones/M3-STEPS.md))
- [ ] M4 Multi-resolution + quantization
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
