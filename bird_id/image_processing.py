"""
Loading a user's PNG drawing and extracting the opaque, colored pixels.

Author: Christine Tao
Date: 7/4/2026
"""

import numpy as np
from PIL import Image

def load_opaque_pixels(png_path: str, alpha_threshold: int = 128) -> np.ndarray:
    """
    Load a PNG and return only the RGB values of pixels that are "opaque enough"
    (i.e., actually drawn by the user, not transparent background).
 
    Args:
        png_path: path to the PNG file.
        alpha_threshold: alpha values >= this are treated as opaque (0-255 scale).
                         Anything below is excluded entirely. (Simplification of
                         the translucent-pixel edge case discussed earlier -- a
                         hard threshold rather than partial weighting.)
 
    Returns:
        np.ndarray of shape (num_opaque_pixels, 3) -- just the R, G, B values
        (no alpha) for each pixel that passed the threshold.
    """

    img = Image.open(png_path).convert("RGBA")
    array = np.asarray(img)
    alpha = array[:,:,3]
    mask = alpha >= alpha_threshold
    filtered = array[mask]

    return filtered[:, :3]
    
    
