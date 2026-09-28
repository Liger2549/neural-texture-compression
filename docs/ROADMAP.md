# Neural Texture Compression — Roadmap

## 1. Project summary

**Goal:** Build a neural texture compression (NTC) system from scratch. A PBR material (albedo, normal, roughness, metalness) is compressed into small learned latent grids plus a tiny MLP decoder. The decoder runs in real time inside a low-level GPU renderer (DirectX 12 or Vulkan — decided after M6), with a fallback path for GPUs that are too slow for per-pixel inference.

**Why:** Neural texture compression is shipping AAA technology (Ubisoft shipped it in Assassin's Creed Mirage; NVIDIA, Intel and Microsoft are all building support). This project implements the technique end to end, from training to a real-time GPU decoder, with the engineering trade-offs a shipping game would face.

**Definition of done (whole project):**
- One material compressed with neural compression, decoded in a pixel shader in real time (DX12 or Vulkan).
- A decode-on-load fallback mode, plus automatic mode selection (feature check + benchmark + user override).
- A results table comparing BC baseline vs neural (size, bits per texel, PSNR/SSIM, frame time, VRAM).
- A README with images, the table, and design decisions that a reader can understand in 2 minutes.

## 2. Core concept (reference)

- Traditional BC formats compress each texture separately in 4x4 blocks. They cannot exploit correlation between channels (e.g. rust in albedo lines up with high roughness).
- NTC stores: (1) **latent grids** — low-resolution multi-channel textures of learned values; (2) a **tiny MLP** shared by the whole material.
- Decoding a texel at UV: sample latent grids (bilinear) → concatenate features → MLP → all material channels at once.
- Training = per-material overfitting. Gradient descent is the encoder. The network is NOT meant to generalize to other materials.
- **Random access:** every pixel decodes independently — required for GPU use.
- **Quantization-aware training:** latents and weights will be stored in 8-bit/4-bit/fp16, so training must simulate that.
- **Runtime modes:**
  - *In-shader decode* (inference on sample): saves VRAM, costs compute per pixel.
  - *Decode-on-load*: decode once at load into ordinary textures; works on any GPU, saves disk/download size only.
- **Filtering:** hardware can filter latents, but the MLP is nonlinear, so filtered-latent → MLP is not identical to filtering the decoded output. Handle mips explicitly.

## 3. Tech stack

**Training (Python)**
- Python 3.11+, PyTorch (CUDA if available, CPU fallback must work for small tests)
- numpy, imageio / Pillow, OpenEXR only if needed
- scikit-image or torchmetrics for PSNR/SSIM; NVIDIA FLIP if easily installable
- pytest for tests

**Baseline tools**
- `texconv` from Microsoft DirectXTex for BC1/BC4/BC5/BC7 baselines

**Runtime (C++)**
- C++20, CMake, Windows
- Graphics API: **DirectX 12 or Vulkan** — chosen at the decision point after M6. M0–M6 are API-independent.
- Shaders: HLSL compiled with DXC (DXIL for DX12, SPIR-V for Vulkan) so the decoder shader stays portable. Native 16-bit math where supported.
- Texture loading: DirectXTex (DX12) or a simple loader / KTX (Vulkan). Dear ImGui for the debug UI.
- The `.ntex` format and all decoder math are API-independent.

## 4. Repository layout

```
neural-texture-compression/
├── README.md               # public-facing summary, results, images
├── docs/
│   ├── ROADMAP.md          # this file
│   ├── PROGRESS.md         # running log: what was done, decisions, next step
│   ├── DECISIONS.md        # design decisions with reasons
│   └── concepts/           # short explainers written while learning
├── assets/                 # source materials (Git LFS)
├── train/
│   ├── ntc/                # python package: data, model, quantize, export, metrics
│   ├── scripts/            # train.py, eval.py, baseline.py, export.py
│   ├── configs/            # yaml/json experiment configs
│   └── tests/
├── runtime/
│   ├── src/                # C++ viewer
│   ├── shaders/            # HLSL
│   └── CMakeLists.txt
├── results/                # metrics tables (csv/md), comparison images
└── .gitattributes          # Git LFS rules
```

## 5. Milestones

Rough pace: about one milestone per week. M3 and M7 take longer.

---

### M0 — Repository setup

**Tasks**
- Create the repo layout above. `git init`, `git lfs install`, track `*.png *.jpg *.exr *.dds *.bin *.ntex` with LFS.
- Python environment (`requirements.txt` or `pyproject.toml`), `.gitignore` for Python, CMake build dirs, and results caches.
- Download one CC0 PBR material from ambientCG at 2K (something with visible detail: rusty metal, bricks, or worn wood). Put it in `assets/<material_name>/`.
- Write the first README: one-sentence goal, milestone checklist, license.
- Create empty `docs/PROGRESS.md` and `docs/DECISIONS.md`.

**Acceptance criteria**
- Repo on GitHub, clones cleanly, LFS works.
- `python -c "import torch; print(torch.cuda.is_available())"` runs in the environment.

**Key concepts:** what a PBR material set is and what each map does.

---

### M1 — Baseline (traditional BC compression)

**Tasks**
- Script `train/scripts/baseline.py`:
  - Load the material; define the channel layout: albedo RGB (3), normal XY (2, Z reconstructed), roughness (1), metalness (1) = 7 channels. Material: Metal016, NormalDX, no AO (see DECISIONS.md D5).
  - Compress with `texconv`: BC7 for albedo, BC5 for normal, BC4 for roughness / metalness.
  - Decode back and compute per-channel PSNR and SSIM vs the source.
  - Compute total size in MB and bits per texel (bpt), with and without mip chain.
- Save `results/baseline.md` and comparison crops.

**Acceptance criteria**
- A table: format per map, size, bpt, PSNR, SSIM.

**Key concepts:** how BC block compression works and why it cannot use cross-channel correlation.

---

### M2 — Overfit a single texture with an MLP

**Tasks**
- Train an MLP `(u, v) → RGB` on the albedo only. Observe the blur.
- Add Fourier features (sin/cos of UV at multiple frequencies). Compare.
- Log PSNR over training steps; save images at checkpoints.

**Acceptance criteria**
- Side-by-side image: plain MLP vs Fourier features vs reference, with PSNR.
- Short write-up in `docs/concepts/spectral-bias.md` (why plain MLPs are blurry).

**Key concepts:** spectral bias, positional encoding, overfitting as compression.

---

### M3 — Latent grid + tiny decoder (core milestone)

**Tasks**
- Model: learnable latent grid(s), sampled bilinearly at random UVs → MLP → 7 output channels.
  - Start config: one grid at 1/4 resolution with 8 latent channels (not the 7 output channels); MLP 2 hidden layers of 32–64 units; output with sigmoid (normals remapped to [-1, 1]).
- Loss: per-channel weighted L2 (make weights configurable; normals may need their own weight).
- Optimizer: Adam, separate learning rates for grids (higher) and MLP (lower).
- Training: random UV batches (tens of thousands per step), configurable step count.
- Decide and document: train albedo in sRGB-encoded or linear space.
- Eval script: full-resolution decode, per-channel PSNR/SSIM, bpt, comparison images vs BC baseline.

**Acceptance criteria**
- Full 7-channel material reconstructs recognizably.
- Results table row added next to the BC baseline (unquantized float latents are fine at this stage, but report their true size honestly).

**Key concepts:** why latents + small MLP beats a big MLP; how the decoder exploits cross-channel correlation.

---

### M4 — Multi-resolution latents + quantization

**Tasks**
- 2–3 latent grids at different resolutions (coarse + fine); concatenate their features.
- Quantization-aware training for latents: noise injection early, then straight-through rounding to the target bit depth (support 8-bit and 4-bit). Per-grid scale/offset.
- MLP weights stored as fp16; verify accuracy impact.
- Sweep a few configs (grid sizes, channels, bit depths) and plot quality vs bpt against the BC baseline point.

**Acceptance criteria**
- A quality vs bits-per-texel chart with at least one neural config that beats or matches BC at lower bpt, or an honest explanation if not.

**Key concepts:** quantization-aware training, straight-through estimator, rate–distortion trade-off.

---

### M5 — Mipmaps

**Tasks**
- Build target mip chain for the material (correct handling for normals: renormalize after downsampling).
- Implement one approach (document the choice in DECISIONS.md):
  - (a) latent pyramid: each mip level samples a matching latent level, shared decoder, trained on all mips; or
  - (b) decoder takes the mip level as an extra input.
- Evaluate quality at each mip level; render a receding plane to check for shimmering/aliasing.

**Acceptance criteria**
- Per-mip quality table; no obvious aliasing in a receding-plane test image.

**Key concepts:** why filtering latents is not the same as filtering outputs.

---

### M6 — Export format + reference decoder

**Tasks**
- Define a simple container, e.g. `material.ntex`: JSON header (grid sizes, channels, bit depths, scale/offset, MLP layer shapes, activation, channel layout, version) + binary blobs.
- Latents stored as ordinary texture data (e.g. R8G8B8A8_UNORM or R8 per channel group) so the GPU can sample them with hardware filtering.
- Weights as fp16 binary.
- Python **reference decoder** that loads only the exported file (no PyTorch model) and reproduces the output.
- pytest: exported decode matches the in-memory model within a tight tolerance.

**Acceptance criteria**
- Round trip test passes. File size reported and matches the bpt claims.

**Key concepts:** why the export format must mirror exactly what the shader will do.

---

### Decision point — Runtime graphics API (after M6)

Choose DirectX 12 or Vulkan for the runtime, and whether the viewer is standalone or built on an existing renderer. Record the choice and reasoning in `docs/DECISIONS.md`.

- **DX12:** main AAA PC API; Microsoft has announced plans for DirectX support of neural texture compression.
- **Vulkan:** cross-platform; strong cooperative-matrix/vector extension support.
- Porting to the other API later is listed as a stretch goal.

---

### M7 — Runtime: in-shader decoder

**Tasks**
- Minimal viewer in the chosen API (DX12 or Vulkan): window, orbit camera, a sphere and a plane, one directional/point light with animated movement, simple PBR (GGX) shading.
- Load `.ntex`: latents as textures, MLP weights in a structured/storage buffer or constant/uniform buffer.
- Pixel/fragment shader (HLSL via DXC unless decided otherwise): sample latents → MLP forward pass with plain loops (use 16-bit math when supported) → material channels → PBR shading.
- Reference path: the same material using BC textures from M1.
- ImGui: mode switch (BC reference / neural), split-screen comparison, per-channel debug view (show albedo only, normal only, etc.).

**Acceptance criteria**
- Neural material renders correctly and visually matches the Python reference decoder (screenshot diff).
- Runs in real time on the development GPU; frame time reported.

**Key concepts:** how an MLP maps to shader code; cost per pixel = (inputs × hidden + hidden × hidden + hidden × outputs) multiply-adds.

---

### M8 — Fallback mode + automatic selection

**Tasks**
- **Decode-on-load mode:** a compute shader decodes the neural material once into ordinary textures (start with uncompressed RGBA8/RG8; optional stretch: re-encode to BC on CPU, e.g. with DirectXTex). Rendering then uses the standard path.
- **Capability check at startup:** query device feature support and record GPU vendor/name. DX12: shader model, native 16-bit ops. Vulkan: shaderFloat16 / 16-bit storage features, and cooperative-matrix/vector extensions if present.
- **Startup micro-benchmark:** time in-shader decode on a test view; if above a threshold, pick decode-on-load.
- **User override:** ImGui / config setting "Neural textures: Auto / In-shader / Decode-on-load".
- Log which mode was chosen and why.

**Acceptance criteria**
- All three modes (BC reference, in-shader, decode-on-load) selectable and working.
- Auto mode chooses correctly on the development machine; forcing each mode works.

**Key concepts:** why real games ship fallbacks; VRAM vs compute trade-off; why low-end GPUs (e.g. older integrated graphics without matrix hardware) should use decode-on-load.

---

### M9 — Measurement + write-up

**Tasks**
- GPU timestamp queries around the material pass; average over many frames.
- VRAM accounting: sum of resource sizes per mode (plus a driver memory query for context: DXGI video memory info on DX12, VK_EXT_memory_budget on Vulkan).
- Final table in `results/final.md`: for each mode → disk size, VRAM, bpt, PSNR/SSIM per channel, frame time.
- README rewrite: hero image, 3-sentence summary, how it works diagram, results table, design decisions, limitations, future work, how to build and run.
- Optional: 1–2 minute video / GIF of the split-screen comparison.

**Acceptance criteria**
- A stranger can understand what was built, why, and how well it works in 2 minutes of reading.

---

## 6. Stretch goals (only after M9)

- **BC-compatible latents** (Ubisoft-style): simulate BC compression of latents inside training so latents can be stored as BC textures.
- **Stochastic texture filtering** for neural outputs.
- **Hardware-accelerated MLP** via cooperative vector / matrix features in the chosen API (where supported).
- **Port the runtime to the other API** (DX12 ↔ Vulkan) to show the technique in both.
- Multiple materials; shared vs per-material decoders.
- Compare against NVIDIA RTX NTC or Intel's texture set neural compression SDK.
- Streaming / partial decode of only the texture regions in use.

## 7. Known risks

- Normal maps are sensitive: small errors look bad under lighting. Check them in the renderer, not only by PSNR.
- sRGB vs linear mistakes will quietly ruin comparisons. Be explicit everywhere.
- Python decode and HLSL decode can drift (precision, sampling offsets, texel-center conventions). The M6 reference decoder exists to catch this.
- Scope creep: finish M9 before any stretch goal.