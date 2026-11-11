"""Crop and prepare plate images for OCR."""

from __future__ import annotations

import math
from typing import cast

import cv2 as cv

from vrpr.models import BoundingBox, Frame


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


def prepare_for_ocr(crop: Frame, min_height: int) -> Frame:
    """Convert a crop to grayscale and enlarge it if it is too small to read."""
    gray = crop if crop.ndim == 2 else cast(Frame, cv.cvtColor(crop, cv.COLOR_BGR2GRAY))
    height = gray.shape[0]
    if height >= min_height:
        return gray
    scale = min_height / height
    resized = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    return cast(Frame, resized)
