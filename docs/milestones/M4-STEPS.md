# M4 — Step by step

Goal: multi-resolution latent grids + quantization-aware training, then a quality vs bits-per-texel chart against BC (see ROADMAP.md, M4).
Same way of working as M3: I write every line that carries the idea, Claude writes the plumbing (argparse, logging, eval, plots, tests). I try each step first; if I'm stuck, Claude shows a worked example and I type it.

## Concept in one picture

```
         ┌─ coarse grid (256², 8 ch) ─► 8 numbers ─┐   big shapes
UV ──────┤                                           ├─ cat ─► 12 numbers ─► tiny MLP ─► 7 channels
         └─ fine grid (1024², 4 ch)  ─► 4 numbers ─┘   small detail

each latent stored as a 4- or 8-bit integer (+ one scale/offset per grid), not fp16
```

Size of one grid: `bpt = channels × bits × (grid_size / texture_size)²`

## Steps

- [x] **1. bpt formula** (`train/ntc/latent.py`)
  `grid_bpt(size, channels, bits, tex_size)`.
  *Understand:* where the bits go, and why fine grids are expensive per channel.
- [x] **2. Multi-res grid** (`train/ntc/latent.py`)
  `MultiResGrid`: several `LatentGrid`s in an `nn.ModuleList`, look up each one, `torch.cat` the codes.
  *Understand:* how coarse + fine features become one input for the MLP.
- [x] **3. Model update** (`train/ntc/latent.py`, `train/scripts/train_ntc.py`)
  `NTCModel` uses `MultiResGrid`; MLP input size = sum of the grid channels.
  *Understand:* the full M4 decoder.
- [ ] **4. Quantize** (`train/ntc/latent.py`)
  `quantize(x, bits)`: scale/offset from the grid's range, round, clamp, back to float.
  *Understand:* what 4-bit or 8-bit storage does to the codes.
- [ ] **5. Noise injection**
  Add ±½ step random noise instead of rounding.
  *Understand:* why rounding has no useful gradient, and how noise imitates rounding error.
- [ ] **6. Straight-through estimator + schedule**
  `x + (round(x) - x).detach()`; noise early in training, STE late.
  *Understand:* forward = real rounding, backward = pretend it's identity.
- [ ] **7. fp16 MLP**
  Cast MLP weights to half and back, compare PSNR.
  *Understand:* why the MLP's precision barely matters for size.
- [ ] **8. Sweep + chart**
  A few configs (grid sizes, channels, bit depths), PSNR vs bpt plot with the BC point.
  *Understand:* the rate–distortion trade-off.
- [ ] **9. Tests + write-up**
  pytest, results table, PROGRESS.md, optional `docs/concepts/quantization.md`.
