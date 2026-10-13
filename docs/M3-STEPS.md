# M3 — Step by step

Goal: latent grid + tiny MLP decoder that reconstructs all 7 channels of Metal016 (see ROADMAP.md, M3).
For each step, Claude explains the tools and shapes, I write the code, then we review it and run a check.

## Concept in one picture

```
UV ──► latent grid (512×512, 8 numbers each) ──► 8 numbers ──► tiny MLP ──► 7 channels
       "map: WHERE things are"                                 "legend: WHAT they look like"
```

## Steps

- [x] **1. Training data** (`train/scripts/train_ntc.py`)
  Material → tensor `(H*W, 7)`, texel UVs `(H*W, 2)`, random batch of 65,536 texels.
  *Understand:* what one training sample looks like.
- [x] **2. Latent grid** (`train/ntc/latent.py`)
  An `nn.Parameter` of shape `(1, 8, 512, 512)`.
  *Understand:* what "learnable codes" means in code.
- [ ] **3. Bilinear lookup** (`train/ntc/latent.py`)
  `F.grid_sample`: UVs `(N, 2)` → features `(N, 8)`.
  *Understand:* the "look up the map" part, same as a GPU texture sampler.
- [ ] **4. Tiny MLP** (`train/ntc/latent.py`)
  8 → 64 → 64 → 7, sigmoid output.
  *Understand:* the "legend" part.
- [ ] **5. Full model** (`train/ntc/latent.py`)
  `forward(uv)` = lookup → MLP.
  *Understand:* the whole decoder.
- [ ] **6. Loss** (`train/ntc/latent.py`)
  Per-channel weighted L2.
  *Understand:* how "wrong" is measured, and why normals get their own weight.
- [ ] **7. Training loop** (`train/scripts/train_ntc.py`)
  Adam with two learning rates (grid higher, MLP lower), logging, saving.
  *Understand:* how training fills in the codes.
- [ ] **8. Eval** (`train/scripts/eval_ntc.py`)
  Full decode, per-channel PSNR/SSIM, normal angle error, bpt, comparison images vs BC.
  *Understand:* does it beat BC, and where does it fail?
- [ ] **9. Tests + write-up**
  pytest for the model, results row next to BC baseline, PROGRESS.md.

## Also to do

- [ ] Experiments after step 7: try learning rates and loss weights, compare numbers
- [ ] Decide sRGB vs linear albedo for training → DECISIONS.md D9 (recommended: sRGB, same as BC baseline)
- [ ] README: tick M3
