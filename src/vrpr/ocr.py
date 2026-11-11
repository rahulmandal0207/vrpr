"""Text recognition on plate crops.

Every engine takes the plate crop as cut from the frame, in OpenCV's BGR order, and prepares it
the way its own model needs.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import cv2 as cv

from vrpr.config import OcrConfig
from vrpr.exceptions import DependencyError, ModelError, OcrError
from vrpr.imaging import prepare_for_ocr
from vrpr.models import BoundingBox, Frame, OcrFragment
from vrpr.plates import read_plate

_PLATE_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class OcrEngine(Protocol):
    def read(self, image: Frame) -> list[OcrFragment]: ...


class EasyOcrEngine:
    """General text reader. It finds the text lines itself, so it copes with two-line plates."""

    def __init__(self, config: OcrConfig) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise DependencyError(
                "EasyOCR needs the ML dependencies. Install them with: pip install -e .[ml]"
            ) from exc
        self._min_height = config.min_crop_height
        self._reader = easyocr.Reader(list(config.languages), gpu=config.gpu)

    def read(self, image: Frame) -> list[OcrFragment]:
        try:
            prepared = prepare_for_ocr(image, self._min_height)
            raw = self._reader.readtext(prepared, allowlist=_PLATE_CHARACTERS)
        except Exception as exc:  # EasyOCR raises many internal types; one crop must not end a run.
            raise OcrError(f"EasyOCR failed: {exc}") from exc

        fragments = []
        for corners, text, confidence in raw:
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            fragments.append(OcrFragment(text, float(confidence), box))
        return fragments


class PlateOcrEngine:
    """A small model trained only on number plates (fast-plate-ocr, run with ONNX Runtime).

    It reads small and blurred single-line plates far better than a general text reader, and
    takes a few milliseconds on a CPU. It reads the whole crop as one line, so it does poorly on
    two-line plates and on boxes much larger than the plate.
    """

    def __init__(self, config: OcrConfig) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as exc:
            raise DependencyError(
                "The plate reader needs the ML dependencies. Install them with: "
                "pip install -e .[ml]"
            ) from exc
        try:
            # The model file is downloaded once, to the user's cache folder.
            self._recognizer: Any = LicensePlateRecognizer(config.plate_model, device="cpu")
        except Exception as exc:
            raise ModelError(
                f"Cannot load the plate reading model '{config.plate_model}': {exc}. "
                "The first use downloads it, which needs a network connection."
            ) from exc

    def read(self, image: Frame) -> list[OcrFragment]:
        try:
            # The model was trained on RGB images; OpenCV frames are BGR.
            code = cv.COLOR_GRAY2RGB if image.ndim == 2 else cv.COLOR_BGR2RGB
            rgb = cast(Frame, cv.cvtColor(image, code))
            prediction = self._recognizer.run(rgb, return_confidence=True)[0]
            text = str(prediction.plate).replace("_", "")
            probabilities = [float(value) for value in prediction.char_probs]
            confidence = sum(probabilities) / len(probabilities)
        except Exception as exc:  # One unreadable crop must not end a run.
            raise OcrError(f"The plate reader failed: {exc}") from exc
        if not text:
            return []
        height, width = image.shape[:2]
        return [OcrFragment(text, confidence, BoundingBox(0.0, 0.0, float(width), float(height)))]


class HybridOcrEngine:
    """Use the plate reader, and the general reader for the crops it cannot settle.

    The plate reader's answer is kept when it is a valid plate number read with enough
    confidence. Otherwise the crop goes to the general reader, which handles two-line plates
    and loosely boxed plates better.
    """

    def __init__(self, primary: OcrEngine, fallback: OcrEngine, fallback_confidence: float) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallback_confidence = fallback_confidence

    def read(self, image: Frame) -> list[OcrFragment]:
        fragments = self._primary.read(image)
        if read_plate(fragments, self._fallback_confidence) is not None:
            return fragments
        return self._fallback.read(image)


def build_ocr_engine(config: OcrConfig) -> OcrEngine:
    """Create the engine named by ``ocr.engine``."""
    if config.engine == "easyocr":
        return EasyOcrEngine(config)
    if config.engine == "plate":
        return PlateOcrEngine(config)
    return HybridOcrEngine(
        PlateOcrEngine(config), EasyOcrEngine(config), config.fallback_confidence
    )
