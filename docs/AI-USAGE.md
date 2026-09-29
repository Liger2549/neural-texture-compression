# How AI was used in this project

This project was built with the help of an AI coding assistant (Claude Code). This page states which parts the assistant did and which parts I did myself, so readers can judge my own work accurately. It is updated at the end of every milestone.

## How the work is split

| Who | What |
|---|---|
| **Me** | Project goal and plan (roadmap, milestones), every design decision (see [DECISIONS.md](DECISIONS.md)), choosing materials, reading and questioning results, core pieces of code marked below, and the real-time runtime (C++ / HLSL, M7–M8) |
| **AI assistant** | Explaining concepts, proposing options with trade-offs, writing tooling code (scripts, file handling, tables, tests), checking my code, and drafting docs that I review |

Rules I followed:
- **Decisions are mine.** The assistant proposes options; I choose. Each choice is recorded with its reasons in DECISIONS.md.
- **No unmeasured claims.** Every quality or size number comes from a script in this repo that anyone can rerun.
- **I must be able to explain everything.** If I can't explain a part, it isn't done.

## Per milestone

### M0 — Repository setup
- **Me:** repo, Git LFS, Python environment, material research and final choice (Metal016, after comparing five candidates).
- **AI:** wrote DECISIONS.md entries from our discussion, measured map statistics to compare candidate materials, pinned `requirements.txt`.

### M1 — BC baseline
- **Me:** `rebuild_normal_z` (normal Z reconstruction) and `psnr` in `train/ntc/`; the BC explainer `docs/concepts/bc-compression.md` (AI filled in one sentence and reworded one); data inspection (channel layouts, NormalDX vs NormalGL, sRGB); interpreting the results; the decision to support multiple materials, each trained separately (D6).
- **AI:** `train/scripts/baseline.py` (texconv calls, size measurement, tables, crops), material config loader, SSIM and normal angle error metrics, tests, checking texconv's sRGB handling.
