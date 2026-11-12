"""Crop and prepare plate images for OCR."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import cv2 as cv
import numpy as np

from vrpr.config import PreprocessConfig
from vrpr.models import BoundingBox, Frame

_DEFAULT_FILTERS = PreprocessConfig()


def crop_plate(frame: Frame, box: BoundingBox, padding: float = 0.0) -> Frame | None:
    """Return an independent copy of the plate region, or None if it is off-frame.

    The copy matters: a numpy slice shares memory with the frame, so anything
    drawn on the frame afterwards would appear in the crop.
    """
    height, width = frame.shape[:2]
    pad_x, pad_y = box.width * padding, box.height * padding
    padded = BoundingBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y)
    clipped = padded.clip(width, height)
    if clipped is None:
        return None
    x1, y1 = math.floor(clipped.x1), math.floor(clipped.y1)
    x2, y2 = math.ceil(clipped.x2), math.ceil(clipped.y2)
    return cast(Frame, frame[y1:y2, x1:x2].copy())


def prepare_for_ocr(
    crop: Frame, min_height: int, filters: PreprocessConfig = _DEFAULT_FILTERS
) -> Frame:
    """Prepare a crop for a general text reader: grayscale, enlarged if small, then filtered.

    With the default filters this only converts and enlarges.
    """
    gray = filters.grayscale or filters.threshold != "none"
    image = cast(Frame, cv.cvtColor(crop, cv.COLOR_BGR2GRAY)) if gray and crop.ndim == 3 else crop
    return apply_filters(enlarge(image, min_height), filters)


def prepare_in_colour(
    crop: Frame, min_height: int, filters: PreprocessConfig = _DEFAULT_FILTERS
) -> Frame:
    """Prepare a crop for a reader trained on colour plates at their own size.

    With no filters on, the crop is returned as it is. With filters on, it is enlarged first,
    as in ``prepare_for_ocr``, so each filter acts at the same scale for every reader.
    """
    if not filters.changes_image:
        return crop
    return apply_filters(enlarge(crop, min_height), filters)


def enlarge(image: Frame, min_height: int) -> Frame:
    """Scale an image up to ``min_height`` if it is shorter."""
    height = image.shape[0]
    if height >= min_height:
        return image
    scale = min_height / height
    return cast(Frame, cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC))


def apply_filters(image: Frame, filters: PreprocessConfig) -> Frame:
    """Apply the OpenCV filters in a fixed order: denoise, gamma, contrast, sharpen, threshold."""
    if not filters.changes_image:
        return image
    if filters.denoise == "bilateral":
        # Smooths flat areas but keeps character edges sharp.
        image = cast(Frame, cv.bilateralFilter(image, 7, 50, 50))
    elif filters.denoise == "median":
        image = cast(Frame, cv.medianBlur(image, 3))
    if filters.gamma != 1.0:
        table = (np.linspace(0.0, 1.0, 256) ** (1.0 / filters.gamma) * 255).round()
        image = cast(Frame, cv.LUT(image, table.astype(np.uint8)))
    if filters.contrast == "clahe":
        clahe = cv.createCLAHE(clipLimit=filters.clahe_clip_limit, tileGridSize=(8, 8))
        image = _on_lightness(image, clahe.apply)
    elif filters.contrast == "equalize":
        image = _on_lightness(image, cv.equalizeHist)
    if filters.sharpen > 0:
        blurred = cv.GaussianBlur(image, (0, 0), 3)
        image = cast(
            Frame, cv.addWeighted(image, 1 + filters.sharpen, blurred, -filters.sharpen, 0)
        )
    if filters.threshold != "none" and image.ndim == 3:
        # Thresholding needs a single channel.
        image = cast(Frame, cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    if filters.threshold == "otsu":
        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        image = cast(Frame, binary)
    elif filters.threshold == "adaptive":
        image = cast(
            Frame,
            cv.adaptiveThreshold(
                image,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                filters.adaptive_block_size,
                10,
            ),
        )
    if filters.invert:
        image = cast(Frame, cv.bitwise_not(image))
    return image


def _on_lightness(image: Frame, operation: Callable[[Any], Any]) -> Frame:
    """Apply a single-channel operation to a gray image, or to the lightness of a colour one."""
    if image.ndim == 2:
        return cast(Frame, operation(image))
    lab = cast(Any, cv.cvtColor(image, cv.COLOR_BGR2LAB))
    lab[:, :, 0] = operation(lab[:, :, 0])
    return cast(Frame, cv.cvtColor(lab, cv.COLOR_LAB2BGR))
