# Copyright 2026 the VA-OPD authors. Licensed under Apache-2.0.
"""Matched-resolution counterfactual images for visual-advantage scoring."""

from __future__ import annotations

import math
from numbers import Real

from PIL import Image


def pixelate_image(image: Image.Image, ratio: float = 0.10) -> Image.Image:
    """Bilinear downsample each side to 10%, then nearest-neighbor upsample.

    Output size and mode match the input, preserving the processor's image-grid
    shape. Dimensions use floor with a minimum of one pixel. No input mutation.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image.")
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be a finite number in (0, 1].")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive.")
    # Pillow's palette/1-bit resize ignores bilinear interpolation. Work in a
    # continuous color mode first, then restore the original mode explicitly.
    source = image.convert("RGB") if image.mode == "P" else image.convert("L") if image.mode == "1" else image
    small = source.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.Resampling.BILINEAR)
    result = small.resize(image.size, Image.Resampling.NEAREST)
    return result.convert(image.mode) if result.mode != image.mode else result
