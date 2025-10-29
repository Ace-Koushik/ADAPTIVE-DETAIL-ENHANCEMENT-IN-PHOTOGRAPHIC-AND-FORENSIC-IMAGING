import cv2
import numpy as np


def median_filter_enhancement(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Enhance image using a Median Filter as a baseline method.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB or grayscale format.
    kernel_size : int
        Size of the median filter kernel. Must be odd and >= 1.

    Returns
    -------
    np.ndarray
        Enhanced image with median filtering applied.
    """
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    if image.ndim == 2:
        # Grayscale
        enhanced = cv2.medianBlur(image, k)
        return enhanced
    elif image.ndim == 3 and image.shape[2] == 3:
        # Color (apply per channel in BGR space for OpenCV efficiency)
        # Convert RGB->BGR for consistency with OpenCV operations
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(bgr)
        b = cv2.medianBlur(b, k)
        g = cv2.medianBlur(g, k)
        r = cv2.medianBlur(r, k)
        enhanced_bgr = cv2.merge([b, g, r])
        rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        return rgb
    else:
        raise ValueError("Unsupported image shape for median filtering.")