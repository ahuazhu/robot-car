"""Single-frame illumination normalization and road-region classification.

Blurred moments stay in OpenCV. The optional C kernel fuses the remaining
per-pixel arithmetic and predicates without changing their thresholds.
"""

import cv2
import numpy as np

from robot.core.native_geometry import native_road_regions


def road_regions(hue, saturation, value, visible, window):
    vf = value.astype(np.float32)
    mean = cv2.blur(vf, (window, window))
    second = cv2.blur(vf * vf, (window, window))
    regions = native_road_regions(hue, saturation, value, visible, mean, second)
    if regions is not None:
        return regions
    return road_regions_reference(hue, saturation, value, visible, mean, second, value_float=vf)


def road_regions_reference(hue, saturation, value, visible, mean, second, *, value_float=None):
    """NumPy fallback; also the independent numeric reference for the C loop."""
    vf = value.astype(np.float32) if value_float is None else value_float
    visible = np.asarray(visible, bool)
    green = (hue >= 32) & (hue <= 95) & (saturation > 55) & (value > 25) & visible
    deviation = np.sqrt(np.maximum(0.0, second - mean * mean))
    contrast = np.clip(
        180 + 60 * (vf - mean) / np.maximum(deviation, 1.0), 0, 255
    ).astype(np.uint8)
    contrast[value < 35] = 0
    road = (saturation < 110) & visible & ~green
    raw_gray = road & (value >= 35) & (value < 210)
    normalized_gray = road & (contrast >= 35) & (contrast < 210)
    return green, contrast, raw_gray, normalized_gray
