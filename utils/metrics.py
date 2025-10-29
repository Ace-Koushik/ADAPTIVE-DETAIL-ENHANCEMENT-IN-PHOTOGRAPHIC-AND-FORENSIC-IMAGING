from typing import Dict
import inspect
import numpy as np

# skimage import compatibility
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    # Older API fallback
    from skimage.measure import compare_ssim as ssim_fn  # type: ignore


def compute_mse(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.
    """
    if original.shape != enhanced.shape:
        raise ValueError("Images must have the same shape for MSE.")
    a = original.astype(np.float32)
    b = enhanced.astype(np.float32)
    err = np.mean((a - b) ** 2)
    return float(err)


def compute_psnr(original: np.ndarray, enhanced: np.ndarray, max_val: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    """
    mse = compute_mse(original, enhanced)
    if mse <= 1e-12:
        return float("inf")
    psnr = 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)
    return float(psnr)


def _safe_win_size_for_shape(shape) -> int:
    """
    Compute a safe odd window size for SSIM not exceeding the smaller spatial dimension.
    Preference order: 7, 5, 3. Returns 0 if image is smaller than 3x3.
    """
    if len(shape) == 2:
        h, w = shape
    else:
        h, w = shape[:2]
    min_side = int(min(h, w))
    for k in (7, 5, 3):
        if min_side >= k:
            return k
    return 0


def _data_range(img: np.ndarray) -> float:
    """
    Determine a safe data_range for SSIM based on dtype.
    """
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    if np.issubdtype(img.dtype, np.floating):
        # Assume normalized floats in [0,1]; if not normalized, fall back to dynamic range
        rng = float(img.max()) - float(img.min())
        return rng if rng > 0 else 1.0
    # Fallback to dynamic range
    rng = float(img.max()) - float(img.min())
    return rng if rng > 0 else 1.0


def _ssim_kwargs_for_version(original_ndim: int) -> Dict[str, object]:
    """
    Build keyword args compatible with both old and new skimage SSIM APIs.
    - Newer skimage uses 'channel_axis'
    - Older skimage uses 'multichannel'
    """
    sig = inspect.signature(ssim_fn)
    has_channel_axis = "channel_axis" in sig.parameters
    kwargs: Dict[str, object] = {}
    if original_ndim == 3:
        if has_channel_axis:
            kwargs["channel_axis"] = -1
        else:
            kwargs["multichannel"] = True
    else:
        # Grayscale
        if has_channel_axis:
            kwargs["channel_axis"] = None  # Explicitly tell skimage there is no channel axis
        else:
            kwargs["multichannel"] = False
    return kwargs


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) robustly across skimage versions and small images.
    Returns a value in [0, 1] (higher is better).
    """
    if original.shape != enhanced.shape:
        raise ValueError("Images must have the same shape for SSIM.")

    # Very small images cannot support SSIM window >= 3
    win_size = _safe_win_size_for_shape(original.shape)
    if win_size < 3:
        return 1.0 if np.array_equal(original, enhanced) else 0.0

    dr = _data_range(original)
    kwargs = _ssim_kwargs_for_version(original.ndim)

    # Attempt with preferred window; on failure, try smaller windows; finally, fall back.
    for w in (win_size, 5, 3):
        if w < 3:
            continue
        try:
            score = ssim_fn(
                original,
                enhanced,
                win_size=w,
                data_range=dr,
                **kwargs
            )
            return float(score)
        except Exception:
            continue

    # Final fallback to avoid crashing the app
    return 1.0 if np.array_equal(original, enhanced) else 0.0


def compute_all_metrics(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    """
    Compute PSNR, MSE, and SSIM metrics.
    """
    return {
        "psnr": compute_psnr(original, enhanced),
        "mse": compute_mse(original, enhanced),
        "ssim": compute_ssim(original, enhanced),
    }