"""Load a PBR material into its channel layout, described by a config file (see DECISIONS.md D5, D6)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Channel:
    name: str          # e.g. "albedo"
    file: str          # source PNG inside the material folder
    count: int         # how many channels to take from the file (normal: 2 = XY, Z is rebuilt)
    color_space: str   # "srgb" or "linear"
    bc_format: str     # texconv format for the BC baseline, e.g. "BC7_UNORM_SRGB"
    normal: bool = False
    start: int = 0     # first index in the stacked array (filled in by load_config)


@dataclass
class MaterialConfig:
    name: str
    folder: Path
    channels: list[Channel]

    @property
    def num_channels(self) -> int:
        return sum(c.count for c in self.channels)


def load_config(path: Path) -> MaterialConfig:
    """Read a material config JSON. Relative folders are resolved from the repo root."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    channels = [Channel(**c) for c in data["channels"]]
    start = 0
    for c in channels:
        c.start = start
        start += c.count
    return MaterialConfig(data["name"], Path(data["folder"]), channels)


def load_image(path: Path) -> np.ndarray:
    """Load an 8-bit PNG as float32 in [0, 1], always shape (H, W, C)."""
    a = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return a[..., None] if a.ndim == 2 else a


def load_material(cfg: MaterialConfig) -> np.ndarray:
    """Return an (H, W, cfg.num_channels) float32 array in [0, 1], channels in config order."""
    parts = [load_image(cfg.folder / c.file)[..., :c.count] for c in cfg.channels]
    return np.concatenate(parts, axis=-1)


def rebuild_normal_z(xy: np.ndarray) -> np.ndarray:
    """xy in [0,1] (H, W, 2) -> unit normals in [-1,1] (H, W, 3)."""
    x = (xy[..., 0] * 2 - 1)
    y = (xy[..., 1] * 2 - 1)
    z = np.sqrt(np.clip(1 - x**2 - y**2, 0, None))
    
    return np.stack([x, y, z], axis=-1)