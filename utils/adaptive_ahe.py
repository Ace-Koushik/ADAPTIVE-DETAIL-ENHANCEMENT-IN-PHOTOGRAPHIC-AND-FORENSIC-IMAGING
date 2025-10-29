import cv2
import numpy as np


def _to_ycrcb(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCrCb."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)


def _from_ycrcb(image_ycrcb: np.ndarray) -> np.ndarray:
    """Convert YCrCb image to RGB."""
    return cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB)


def standard_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Apply standard CLAHE on the luminance channel (Y) in YCrCb color space.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB.
    clip_limit : float
        Clip limit for CLAHE.
    tile_grid_size : tuple
        Tile grid size for CLAHE.

    Returns
    -------
    np.ndarray
        Contrast-enhanced RGB image.
    """
    if image.ndim == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
        out = clahe.apply(image)
        return out

    ycrcb = _to_ycrcb(image)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    rgb = _from_ycrcb(merged)
    return rgb


def clahe_bilateral(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply CLAHE followed by Bilateral filtering for edge-preserving smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    clip_limit : float
        CLAHE clip limit.
    tile_grid_size : tuple
        CLAHE tile grid size (w, h).
    bilateral_d : int
        Diameter of each pixel neighborhood used during filtering.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Enhanced RGB image.
    """
    clahe_out = standard_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    # Use bilateralFilter in BGR space for OpenCV speed
    bgr = cv2.cvtColor(clahe_out, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr, int(bilateral_d), float(sigma_color), float(sigma_space))
    rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return rgb


def clahe_unsharp_masking(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
    amount: float = 1.5,
    radius: float = 1.0,
    threshold: int = 0
) -> np.ndarray:
    """
    Apply CLAHE followed by Unsharp Masking for detail enhancement.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    clip_limit : float
        CLAHE clip limit.
    tile_grid_size : tuple
        CLAHE tile grid size (w, h).
    amount : float
        Strength of the sharpening effect.
    radius : float
        Gaussian blur radius (sigma) for generating the mask.
    threshold : int
        Threshold for minimal mask contribution (to avoid amplifying noise).

    Returns
    -------
    np.ndarray
        Enhanced RGB image.
    """
    clahe_out = standard_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    if clahe_out.ndim == 2:
        base = clahe_out.astype(np.float32)
        blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
        mask = base - blurred
        if threshold > 0:
            mask[np.abs(mask) < threshold] = 0
        sharpened = np.clip(base + amount * mask, 0, 255).astype(np.uint8)
        return sharpened

    # Color
    base = clahe_out.astype(np.float32)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
    mask = base - blurred

    if threshold > 0:
        mask[np.abs(mask) < threshold] = 0

    sharpened = np.clip(base + amount * mask, 0, 255).astype(np.uint8)
    return sharpened


def _local_variance_clip_limit(image: np.ndarray, base_clip: float, scale: float = 1.0) -> float:
    """
    Estimate local variance to slightly adjust CLAHE clip limit.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    base_clip : float
        Base clip limit.
    scale : float
        Scaling factor for adjustment.

    Returns
    -------
    float
        Adjusted clip limit within [0.5, 4.0] typical range.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    var = float(np.var(gray))
    # Normalize variance to a manageable range and adjust
    adj = base_clip + scale * (var / (255.0 ** 2)) * 2.0
    return float(np.clip(adj, 0.5, 4.0))


def hybrid_adaptive_hist_eq(
    image: np.ndarray, block_size: int = 8, limit: float = 2.0, mode: str = "standard"
) -> np.ndarray:
    """
    Dispatcher: Adaptive histogram equalization pipeline with optional variance-based clip limit tuning.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    block_size : int
        Tile grid block dimension (block_size x block_size).
    limit : float
        Base clip limit for CLAHE.
    mode : str
        'standard' | 'bilateral' | 'unsharp' | 'median'

    Returns
    -------
    np.ndarray
        Enhanced RGB image.
    """
    bs = max(2, int(block_size))
    tile_grid = (bs, bs)
    # Slightly adjust clip limit based on local variance
    adj_clip = _local_variance_clip_limit(image, base_clip=float(limit), scale=1.0)

    if mode == "standard":
        return standard_clahe(image, clip_limit=adj_clip, tile_grid_size=tile_grid)
    elif mode == "bilateral":
        return clahe_bilateral(image, clip_limit=adj_clip, tile_grid_size=tile_grid)
    elif mode == "unsharp":
        return clahe_unsharp_masking(image, clip_limit=adj_clip, tile_grid_size=tile_grid)
    elif mode == "median":
        # As a fallback baseline
        k = 5 if bs % 2 == 0 else bs
        from .filtering import median_filter_enhancement
        return median_filter_enhancement(image, k)
    else:
        return standard_clahe(image, clip_limit=adj_clip, tile_grid_size=tile_grid)