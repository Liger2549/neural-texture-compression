# Progress Log

## 2026-09-28 — M0: Repository setup ✅

**Environment**
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8 GB)
- Python 3.13, PyTorch 2.14.0 + CUDA 13.0 (GPU training confirmed working)
- Windows, Git LFS for textures and binaries

**Done**
- Repo structure created (train/, runtime/, assets/, results/, docs/)
- Git LFS tracking set up before adding any images
- Python virtual environment + requirements.txt
- Material chosen: `Metal016` (ambientCG, 2K PNG). 7 channels: albedo RGB, normal XY (NormalDX), roughness, metalness. No AO (DECISIONS.md D5)
- Other candidates (Ground054, Ground068, Marble016, Metal055A) compared and deleted before the first commit
- README, ROADMAP, PROGRESS, DECISIONS created

**Open issues**
- `requirements.txt` only has the PyTorch index URL, no packages listed yet

**Next:** M1 — BC baseline (compress the material with texconv, measure size and quality)