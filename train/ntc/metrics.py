"""Quality metrics. All image inputs are float arrays in [0, 1] with shape (H, W, C)."""
import numpy as np
from skimage.metrics import structural_similarity

from .material import rebuild_normal_z


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for values in [0, 1]. Identical inputs -> inf."""
    mse = float(np.mean((ref - test) ** 2))
    if mse <= 0:
        return float("inf")
    return 10 * np.log10(1 / mse)  # peak value is 1, so peak^2 = 1


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM over all channels (scikit-image, default 7x7 window)."""
    return float(structural_similarity(ref, test, channel_axis=-1, data_range=1.0))


def normal_angle_error(ref_xy: np.ndarray, test_xy: np.ndarray) -> float:
    """Mean angle in degrees between normals rebuilt from XY in [0, 1]."""
    a = rebuild_normal_z(ref_xy)
    b = rebuild_normal_z(test_xy)
    # Normalize: compressed XY can land outside the unit circle, so rebuilt vectors may not be unit length.
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    cos = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)  # per-pixel dot product
    return float(np.degrees(np.arccos(cos)).mean())
