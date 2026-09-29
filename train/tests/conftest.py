import sys
from pathlib import Path

# Make the `ntc` package importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
