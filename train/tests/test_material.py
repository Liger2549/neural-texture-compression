from pathlib import Path

import numpy as np

from ntc.material import load_config, load_material

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "materials" / "Metal016.json"
REPO = CONFIG.parents[3]


def test_config_layout():
    cfg = load_config(CONFIG)
    assert cfg.num_channels == 7
    assert [(c.name, c.start, c.count) for c in cfg.channels] == [
        ("albedo", 0, 3), ("normal", 3, 2), ("roughness", 5, 1), ("metalness", 6, 1),
    ]


def test_load_material_shape_and_range():
    cfg = load_config(CONFIG)
    cfg.folder = REPO / cfg.folder  # tests may run from any directory
    m = load_material(cfg)
    assert m.shape == (2048, 2048, 7)
    assert m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0
