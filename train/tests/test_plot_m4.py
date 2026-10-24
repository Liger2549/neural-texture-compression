import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_m4  # noqa: E402


def test_pareto_keeps_only_unbeaten_points():
    # (bpt, PSNR): (3, 29) is beaten by (2, 30): bigger and worse.
    points = [(4, 32), (2, 30), (3, 29), (8, 31), (6, 33)]
    assert plot_m4.pareto(points) == [(2, 30), (4, 32), (6, 33)]


def test_bc_point_weights_maps_by_channel_count(tmp_path, monkeypatch):
    # Two maps with the same PSNR -> the combined PSNR is that PSNR, whatever the channel counts.
    baseline = {"total_bpt": 24, "maps": [{"channels": 3, "psnr": 40.0}, {"channels": 1, "psnr": 40.0}]}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(plot_m4, "BASELINE", path)
    assert plot_m4.bc_point() == (24, 40.0)

    # 3 channels at MSE 1e-4 and 1 channel at MSE 1e-2 -> mean MSE (3e-4 + 1e-2) / 4.
    baseline["maps"] = [{"channels": 3, "psnr": 40.0}, {"channels": 1, "psnr": 20.0}]
    path.write_text(json.dumps(baseline), encoding="utf-8")
    _, db = plot_m4.bc_point()
    assert math.isclose(db, 10 * math.log10(4 / (3e-4 + 1e-2)))
