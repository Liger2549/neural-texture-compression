# Design Decisions

Each entry records what was decided, why, and what else was considered. Newest at the bottom.
Status: **Accepted** (in effect), **Open** (needs a choice), or **Superseded** (replaced by a later entry).

---

## D1 — Store binary assets with Git LFS

**Date:** 2026-09-28 · **Milestone:** M0 · **Status:** Accepted

**Decision:** Track `*.png *.jpg *.exr *.dds *.bin *.ntex` with Git LFS (see `.gitattributes`). LFS was set up before any image was added, so no binary ever lands in normal Git history.

**Why:** One 2K PBR material is tens of MB, and every experiment adds comparison images and exported files. In plain Git these would bloat the repo permanently, because deleted files stay in history.

**Alternatives considered:**
- *Plain Git:* simplest, but the repo grows with every image and can't shrink without rewriting history.
- *Don't commit assets; add a download script:* keeps the repo small, but the exact source files aren't versioned and ambientCG URLs could change, so results would stop being reproducible.

**Consequences:** Cloning needs `git lfs install`. GitHub's free LFS quota (storage and bandwidth) is limited, so keep large temporary outputs in `results/tmp/` (gitignored).

---

## D2 — Python 3.13 with PyTorch CUDA 13.0 wheels

**Date:** 2026-09-28 · **Milestone:** M0 · **Status:** Accepted

**Decision:** Use Python 3.13 and PyTorch 2.14.0 built for CUDA 13.0, installed from the PyTorch wheel index (`--extra-index-url https://download.pytorch.org/whl/cu130` in `requirements.txt`). Training runs on the RTX 3070 Laptop GPU (8 GB).

**Why:** It is a current PyTorch build with CUDA wheels for this GPU, and GPU training was confirmed working. The default PyPI index can install a CPU-only build on Windows, so the CUDA index is pinned in `requirements.txt`.

**Alternatives considered:**
- *Python 3.11/3.12:* the roadmap minimum is 3.11 and older versions have wider library support, but there was no need, since everything needed for M0 works on 3.13.
- *CPU-only PyTorch:* works everywhere but is far too slow for training runs in M3–M5. It is kept only as a fallback for small tests.

**Consequences:** Some optional packages (OpenEXR, NVIDIA FLIP) may not ship 3.13 wheels yet. Check before adding them. 8 GB of VRAM limits batch size and grid resolution, so keep that in mind for the sweeps in M4.

---

## D3 — Defer the runtime graphics API choice until after M6

**Date:** 2026-09-28 · **Milestone:** M0 · **Status:** Accepted

**Decision:** M0–M6 (training, quantization, export format, Python reference decoder) must not depend on DirectX 12 or Vulkan. The API is chosen at the checkpoint after M6 and recorded here as a new entry.

**Why:** The best choice depends on things not known yet: whether this can double as the Computer Graphics course project (which would favor Vulkan), the course deadline, and which existing code is the better base (the class Vulkan code or the personal DX12 engine). None of the work before M7 needs the API, so choosing early would only lock in a guess.

**Alternatives considered:**
- *Commit to DX12 now:* existing engine code, the AAA PC standard, and upcoming DirectX support for neural texture compression. Rejected for now because it could rule out using this as the course project.
- *Commit to Vulkan now:* matches the course, but it's too early to know if the course project allows this topic.

**Consequences:** The `.ntex` format and decoder math must stay API-neutral. Latents are stored as ordinary texture data, and weights are stored as plain fp16 arrays.

---

## D4 — Write shaders in HLSL, compiled with DXC

**Date:** 2026-09-28 · **Milestone:** M0 (plan) · **Status:** Accepted

**Decision:** Write the decoder and material shaders in HLSL and compile them with DXC: to DXIL for DX12 and to SPIR-V for Vulkan.

**Why:** A single shader source works with either API, so D3 doesn't force a shader rewrite, and a later port (a stretch goal) stays cheap. HLSL also supports native 16-bit types (`min16float` / `float16_t`) for the MLP.

**Alternatives considered:**
- *GLSL:* the natural choice for Vulkan, but it would mean a second shader source if DX12 is chosen. **Recheck at the API checkpoint** if the course requires GLSL.
- *Slang:* compiles to both targets and supports automatic differentiation, but it is one more tool to learn and the project doesn't need what it adds.

---

## D5 — Material: Metal016, 7 channels, NormalDX

**Date:** 2026-09-28 · **Milestone:** M0 → used from M1 · **Status:** Accepted

**Decision:**
- **Material:** `assets/Metal016_2K-PNG/` (ambientCG Metal016, 2K PNG, CC0).
- **Channel layout (7 channels):**

| # | Channel | Source map | Space | BC baseline format |
|---|---|---|---|---|
| 0–2 | Albedo RGB | `_Color.png` | sRGB-encoded (linear vs sRGB for training decided in M3) | BC7 |
| 3–4 | Normal XY | `_NormalDX.png` | [0,1] → [-1,1]; Z reconstructed | BC5 |
| 5 | Roughness | `_Roughness.png` | linear | BC4 |
| 6 | Metalness | `_Metalness.png` | linear | BC4 |

- **No AO channel.** Metal016 has no AO map.
- **Normal convention:** DirectX (Y down, `NormalDX`). `NormalGL` is not used.
- `_Displacement.png` is not used (it isn't part of the shaded material in this project).

**Why:**
- Metal016 is worn metal with non-metal patches, so its channels vary *together*: about 12% of texels are non-metal (metalness ≤ 127), 22% are transitions (metalness between 25 and 230), metalness vs roughness r = 0.45, and metalness vs color luminance r = 0.24. This cross-channel structure is what NTC exploits and per-texture BC can't, so it makes the comparison meaningful.
- No AO rather than a constant fake AO channel: a constant channel still costs BC a full BC4 texture (4 bpt) but costs NTC almost nothing, which would unfairly favor NTC.
- NormalDX matches the DirectX-style convention used by many engines (e.g. Unreal). The shader language and graphics API don't decide the Y direction; the tangent-space setup and UV convention do. If the runtime's tangent-space convention differs (expects Y-up), flip Y at load time. Don't switch the source map.

**Metal016 map statistics** (2048², 8-bit unless noted):

| Map | Range | Mean | Std dev |
|---|---|---|---|
| Color (RGB) | 78–255 | 220.0 | 22.2 |
| Metalness | 0–255 | 219.3 | 69.0 |
| Roughness | 72–241 | 128.9 | 24.2 |
| Displacement (16-bit) | 0–65535 | 14825 | 9829 |

**Alternatives considered:**

| Material | Color | Normal | Roughness | Metalness | AO | Why not |
|---|---|---|---|---|---|---|
| Ground054 | ✓ | ✓ | ✓ | – | ✓ | No metalness |
| Ground068 | ✓ | ✓ | ✓ | – | ✓ | No metalness |
| Marble016 | ✓ | ✓ | ✓ | – | – | No metalness or AO |
| Metal055A | ✓ | ✓ | ✓ | ✓ | – | Metalness nearly constant (mean 254.7, std 2.7), color flat (std 8.9); little cross-channel structure |

- *8 channels with a constant AO = 1.0:* keeps the roadmap's original layout but adds a fake channel that skews bpt comparisons (see above).
- *Look for a metal material that also has AO:* possible, but Metal016 already shows the property that matters. Missing AO doesn't weaken the comparison.

**Consequences:** The roadmap's 8-channel layout becomes 7 channels (M1 baseline, M3 decoder outputs). BC baseline total is BC7 + BC5 + 2 × BC4 = 8 + 8 + 4 + 4 = 24 bpt before mips. The other candidates were deleted before the first commit so they never enter Git LFS. They can be downloaded again from ambientCG when more materials are added (see D6).

---

## D6 — Multiple materials, each trained separately, combined in one scene

**Date:** 2026-09-29 · **Milestone:** M1 (affects M1, M6, M7, M9) · **Status:** Accepted

**Decision:** The project compresses 3–4 materials instead of one. Each material is trained on its own, with its own latent grids and decoder, and has its own channel layout. The M7 runtime shows them together in one scene.

- Development and tuning happen on **Metal016** only (D5). The other materials are added around M6, once the pipeline works end to end.
- Each material is described by a **config file** (`train/configs/materials/<name>.json`): folder, source map per channel, channel counts, color space, BC baseline format. No code hard-codes a channel count or layout.
- Pick the extra materials to have **different channel sets** (e.g. one with AO, one without metalness), so the per-material layout is actually exercised.

**Why:**
- This is how NTC is used in real games: every material (texture set) is compressed separately with its own decoder. A scene with several materials is closer to a shipping game than one sphere.
- Results on several materials show the method works in general, not just on one lucky material.
- Making the layout config-driven costs little now and avoids a rewrite later.

**Alternatives considered:**
- *One material only (original plan):* simplest, but a weaker demo and a weaker results claim. "Multiple materials" was a stretch goal.
- *Fixed 8-channel layout for all materials:* simple code, but missing maps need fake constant channels, which skews bpt comparisons (see D5), and it breaks on materials with other maps (opacity, emissive).
- *One shared decoder for all materials:* a research question (quality vs savings). It stays a stretch goal.

**Consequences:** Scripts take a `--material <config>` argument. Results are stored in `results/<material>/`. The `.ntex` header must record the channel layout (already planned in M6). The M7 shader needs a variant (or a max size) per output channel count. Training time grows with the number of materials, so keep it at 3–4.

---

## D7 — How the BC baseline is measured

**Date:** 2026-09-29 · **Milestone:** M1 · **Status:** Accepted

**Decision:**
- **Albedo metrics are computed on sRGB-encoded values** (the 0–255 numbers stored in the file). All other maps are linear data and measured as stored.
- **texconv runs with `-srgb` for sRGB channels**, on both encode and decode. Tested: with `-srgb` an uncompressed round trip is bit-exact; without it texconv applies a color conversion that shifts values (mean bias +0.12/255) before compression even starts.
- **texconv default quality settings** for all formats (no `-bc x` maximum-quality BC7 mode).
- **Size = DDS file size minus the DDS header(s)**, i.e. the bytes the GPU stores. bpt is reported without and with the full mip chain.
- **Normals get an extra metric:** mean angle error in degrees between the rebuilt (and normalized) normals, because PSNR on XY doesn't say how wrong the lighting will be.

**Why:** These choices make the baseline fair and repeatable. sRGB-encoded values are what the file stores and what BC compresses, so it's the natural space to compare in. Whether training uses linear or sRGB albedo is a separate M3 decision; linear metrics can be added then.

**Alternatives considered:**
- *Albedo metrics in linear space:* closer to what lighting uses, but not what BC was optimized for. Revisit in M3.
- *Maximum BC7 quality (`-bc x`):* a stronger baseline, but much slower. Worth a quick A/B later to see if it changes the albedo row; if it does, use the stronger one so NTC isn't compared against a weak baseline.
- *Theoretical sizes from the format spec:* same numbers for these formats, but measuring the real file catches mistakes (wrong format, missing mips).
