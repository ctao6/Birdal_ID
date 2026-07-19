"""
color_vector.py
---------------
Turns a set of rgb pixels into one 24-dimensional vector for purpose of comparison
against a record in our training data table.

Author: Christine Tao
Date: 7/3/2026
"""

import numpy as np

def pixel_weights(pixel_rgb: np.ndarray, cluster_centers: np.ndarray, temperature = 1) -> np.ndarray:
    """
    Calculate soft cluster-membership weights for singular pixel.

    Args:
        pixel_rgb:shape (3,), pixel's rgb values.
        cluster_centers: shape (24,3), foxed cluster center rgb values

    Returns:
        np.ndarray of shape (24,) that sums to 1 -- this pixel's weight toward
        each of the 24 clusters, with closer clusters getting higher weight.
 
    Hints:
        - Compute Euclidean distance from this pixel to each of the 24 centers
          (np.linalg.norm along the right axis will do this in one line for
          all 24 centers at once -- no loop needed).
        - Converting "distance" into "weight" needs an inverse relationship:
          small distance -> large weight. A common trick is something like
          1 / (distance + small_epsilon), or a negative-exponential/softmax
          transform. Try the simplest version first, you can refine later.
        - Don't forget to normalize the 24 weights so they sum to 1.
    """

    diffs = cluster_centers - pixel_rgb
    squared = diffs**2
    summed = squared.sum(axis=1)
    distances = summed**(0.5)
    raw_weights = np.exp(-distances/temperature)
    weights = raw_weights/raw_weights.sum()

    return weights

def image_color_vector(pixels, cluster_centers, temperature = 1) -> np.ndarray:
    """
    Compute the full 24-dimensional color vector for an entire image.
 
    Args:
        pixels: shape (num_pixels, 3) -- all opaque pixels from the image
                (output of image_processing.load_opaque_pixels).
        cluster_centers: shape (24, 3).
 
    Returns:
        np.ndarray of shape (24,) summing to 1 -- the image's overall color
        vector, directly comparable to a species row in the reference table.

    """
    total = np.zeros(len(cluster_centers))

    for pixel in pixels:
        total += pixel_weights(pixel, cluster_centers, temperature)

    return total / len(pixels)

    """
    pixels_expanded = pixels[:, np.newaxis, :]
    diffs = cluster_centers - pixels_expanded
    squared = diffs**2
    summed = squared.sum(axis=2)
    distances = summed**0.5
    raw_weights = np.exp(-distances/temperature)

    weights = raw_weights/raw_weights.sum(axis=1)[:,np.newaxis]

    return weights.mean(axis=0)
    """
    
    
        


