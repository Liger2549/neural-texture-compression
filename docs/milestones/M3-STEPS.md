# M3 — Step by step

Goal: latent grid + tiny MLP decoder that reconstructs all 7 channels of Metal016 (see ROADMAP.md, M3).
For each step, Claude explains the tools and shapes, I write the core model code (Claude writes the plumbing: logging, saving, eval, tests), then we review it and run a check.
Step plans start at M3, when I began writing the core model code myself. M1 and M2 are summarized in [PROGRESS.md](../PROGRESS.md).

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
- [x] **3. Bilinear lookup** (`train/ntc/latent.py`)
  `F.grid_sample`: UVs `(N, 2)` → features `(N, 8)`.
  *Understand:* the "look up the map" part, same as a GPU texture sampler.
- [x] **4. Tiny MLP** (`train/ntc/latent.py`)
  8 → 64 → 64 → 7, sigmoid output.
  *Understand:* the "legend" part.
  *In my words:* TinyMLP takes the 8-number code for a texel and turns it into the 7 channel values (albedo, normal, roughness, metalness). There is one MLP for the whole material, shared by every texel, because the per-texel detail lives in the grid codes and the MLP only learns how to read them.
- [x] **5. Full model** (`train/ntc/latent.py`)
  `forward(uv)` = lookup → MLP.
  *Understand:* the whole decoder.
- [x] **6. Loss** (`train/ntc/latent.py`)
  Per-channel weighted L2.
  *Understand:* how "wrong" is measured, and why normals get their own weight.
- [x] **7. Training loop** (`train/scripts/train_ntc.py`)
  Adam with two learning rates (grid higher, MLP lower), logging, saving.
  *Understand:* how training fills in the codes.
- [x] **8. Eval** (`train/scripts/eval_ntc.py`)
  Full decode, per-channel PSNR/SSIM, normal angle error, bpt, comparison images vs BC.
  *Understand:* does it beat BC, and where does it fail?
- [x] **9. Tests + write-up**
  pytest for the model, results row next to BC baseline, PROGRESS.md.
  Questions for my PROGRESS.md bullets:
  1. Size: how does our bpt compare with BC?
  2. Quality: which channel is best/worst, and how far are we from BC?
  3. What does it look like? (`crops/normal.png`: blurry or sharp, where is the error?)
  4. Why? (grid 512 vs texture 2048) and what fixes it in M4?

## Also to do

- [ ] Experiments after step 7: try learning rates and loss weights, compare numbers
- [ ] Decide sRGB vs linear albedo for training → DECISIONS.md D9 (recommended: sRGB, same as BC baseline)
- [x] README: tick M3
