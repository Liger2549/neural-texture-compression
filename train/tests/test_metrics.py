import numpy as np

from ntc.material import rebuild_normal_z
from ntc.metrics import normal_angle_error, psnr, ssim


def test_psnr_identical_is_inf():
    a = np.random.default_rng(0).random((8, 8, 3))
    assert psnr(a, a) == float("inf")


def test_psnr_known_values():
    a = np.zeros((4, 4, 1))
    assert np.isclose(psnr(a, a + 0.1), 20.0)
    assert np.isclose(psnr(a, a + 0.01), 40.0)  # 10x smaller error -> +20 dB


def test_ssim_identical_is_one():
    a = np.random.default_rng(0).random((32, 32, 3))
    assert np.isclose(ssim(a, a), 1.0)


def test_rebuild_normal_flat():
    flat = np.full((1, 1, 2), 0.5)
    assert np.allclose(rebuild_normal_z(flat), [0.0, 0.0, 1.0])


def test_rebuild_normal_is_unit_length():
    # 45 degree tilt: x = y = 0.5 in [-1, 1] -> 0.75 in [0, 1]
    xy = np.full((1, 1, 2), 0.75)
    n = rebuild_normal_z(xy)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.isclose(n[0, 0, 2], np.sqrt(0.5))


def test_rebuild_normal_outside_circle_has_no_nan():
    xy = np.ones((1, 1, 2))  # x^2 + y^2 = 2 > 1: impossible, but can come out of compression
    assert not np.isnan(rebuild_normal_z(xy)).any()


def test_normal_angle_error():
    flat = np.full((1, 1, 2), 0.5)
    tilted = np.array([[[0.75, 0.5]]])  # x = 0.5 -> 30 degrees from straight up
    assert np.isclose(normal_angle_error(flat, flat), 0.0)
    assert np.isclose(normal_angle_error(flat, tilted), 30.0)
