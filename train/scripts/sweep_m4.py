"""M4 step 8: train + eval every config × quantization method, for the quality vs bpt chart.

Methods per config:
    fp16       train without quantization, eval at fp16
    ptq        the fp16 model, grids quantized after training (eval --grid-bits b)
    qat_noise  quantization-aware training, noise only (--ste-from 1.0)
    qat_ste    quantization-aware training, noise then STE (--ste-from 0.7)

Runs go to results/<material>/m4/sweep/<run>. Finished runs (eval json present) are skipped, so it can resume.
Usage (from the repo root):
    .venv\\Scripts\\python.exe train/scripts/sweep_m4.py
"""
import subprocess
import sys
from pathlib import Path

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent
OUT = Path("results/Metal016/m4/sweep")
SEED = 42
STEPS = 5000
BITS = [4, 8]

# name: (channels, sizes)
CONFIGS = {
    "m8": ([8], [512]),
    "c8f2": ([8, 2], [256, 1024]),
    "c8f4": ([8, 4], [256, 1024]),
    "c16f4": ([16, 4], [256, 1024]),
    "3lvl": ([16, 8, 4], [128, 512, 1024]),
    "full2": ([2], [2048]),
}


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def train(run_name: str, channels: list[int], sizes: list[int], extra: list[str]) -> None:
    if (OUT / run_name / "train.json").exists():
        return
    run([PY, str(SCRIPTS / "train_ntc.py"), "--milestone", "m4/sweep", "--run", run_name,
         "--channels", *map(str, channels), "--sizes", *map(str, sizes),
         "--steps", str(STEPS), "--seed", str(SEED), "--log-every", "1000", *extra])


def evaluate(run_name: str, grid_bits: int | None = None) -> None:
    tag = "" if grid_bits in (None, 16) else f"_q{grid_bits}"
    if (OUT / run_name / f"eval{tag}.json").exists():
        return
    extra = [] if grid_bits is None else ["--grid-bits", str(grid_bits)]
    run([PY, str(SCRIPTS / "eval_ntc.py"), "--milestone", "m4/sweep", "--run", run_name, *extra])


def main() -> None:
    for name, (channels, sizes) in CONFIGS.items():
        train(f"{name}_fp16", channels, sizes, [])
        evaluate(f"{name}_fp16")
        for b in BITS:
            evaluate(f"{name}_fp16", b)  # ptq
            for method, ste_from in (("noise", "1.0"), ("ste", "0.7")):
                run_name = f"{name}_q{b}_{method}"
                train(run_name, channels, sizes, ["--grid-bits", str(b), "--ste-from", ste_from])
                evaluate(run_name)
    print("sweep done", flush=True)


if __name__ == "__main__":
    main()
