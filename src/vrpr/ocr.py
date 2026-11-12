"""Text recognition on plate crops.

Every engine takes the plate crop as cut from the frame, in OpenCV's BGR order, and prepares it
the way its own model needs. The ``preprocess`` filters are part of that preparation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import cv2 as cv

from vrpr.config import OcrConfig, PreprocessConfig
from vrpr.exceptions import DependencyError, ModelError, OcrError
from vrpr.imaging import prepare_for_ocr, prepare_in_colour
from vrpr.models import BoundingBox, Frame, OcrFragment
from vrpr.plates import read_plate

_PLATE_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NO_FILTERS = PreprocessConfig()


class OcrEngine(Protocol):
    def read(self, image: Frame) -> list[OcrFragment]: ...


@dataclass(frozen=True)
class TracedReading:
    """A reading with how it was made, for trying settings on one frame.

    ``prepared`` is the image the reader was given after its own preparation. ``set_aside`` is
    an earlier reader's attempt that was not kept.
    """

    reader: str
    prepared: Frame
    fragments: list[OcrFragment]
    set_aside: TracedReading | None = None


def trace_reading(engine: OcrEngine, image: Frame) -> TracedReading:
    """Read ``image`` and report which reader answered and the image it saw."""
    traced = getattr(engine, "read_traced", None)
    if traced is None:
        return TracedReading("", image, engine.read(image))
    return cast(TracedReading, traced(image))


class EasyOcrEngine:
    """General text reader. It finds the text lines itself, so it copes with two-line plates."""

    name = "EasyOCR"

    def __init__(self, config: OcrConfig, filters: PreprocessConfig = _NO_FILTERS) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise DependencyError(
                "EasyOCR needs the ML dependencies. Install them with: pip install -e .[ml]"
            ) from exc
        self._config = config
        self._filters = filters
        self._reader = easyocr.Reader(list(config.languages), gpu=config.gpu)

    @property
    def config(self) -> OcrConfig:
        return self._config

    def with_settings(
        self, config: OcrConfig, filters: PreprocessConfig = _NO_FILTERS
    ) -> EasyOcrEngine:
        """Return an engine that shares this one's loaded reader but reads with new settings.

        Languages and GPU use are the reader's own, so only reading settings change.
        """
        other = copy.copy(self)
        other._config = config
        other._filters = filters
        return other

    def readtext_options(self) -> dict[str, Any]:
        settings = self._config
        return {
            "allowlist": _PLATE_CHARACTERS,
            "decoder": settings.decoder,
            "beamWidth": settings.beam_width,
            "text_threshold": settings.text_threshold,
            "low_text": settings.low_text,
            "link_threshold": settings.link_threshold,
            "contrast_ths": settings.contrast_ths,
            "adjust_contrast": settings.adjust_contrast,
            "mag_ratio": settings.mag_ratio,
            "min_size": settings.min_size,
            "rotation_info": list(settings.rotation_info) or None,
        }

    def prepare(self, image: Frame) -> Frame:
        return prepare_for_ocr(image, self._config.min_crop_height, self._filters)

    def read(self, image: Frame) -> list[OcrFragment]:
        return self.read_traced(image).fragments

    def read_traced(self, image: Frame) -> TracedReading:
        try:
            prepared = self.prepare(image)
            raw = self._reader.readtext(prepared, **self.readtext_options())
        except Exception as exc:  # EasyOCR raises many internal types; one crop must not end a run.
            raise OcrError(f"EasyOCR failed: {exc}") from exc

        fragments = []
        for corners, text, confidence in raw:
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            fragments.append(OcrFragment(text, float(confidence), box))
        return TracedReading(self.name, prepared, fragments)


class PlateOcrEngine:
    """A small model trained only on number plates (fast-plate-ocr, run with ONNX Runtime).

    It reads small and blurred single-line plates far better than a general text reader, and
    takes a few milliseconds on a CPU. It reads the whole crop as one line, so it does poorly on
    two-line plates and on boxes much larger than the plate.
    """

    name = "Plate model"

    def __init__(self, config: OcrConfig, filters: PreprocessConfig = _NO_FILTERS) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as exc:
            raise DependencyError(
                "The plate reader needs the ML dependencies. Install them with: "
                "pip install -e .[ml]"
            ) from exc
        self._config = config
        self._filters = filters
        try:
            # The model file is downloaded once, to the user's cache folder.
            self._recognizer: Any = LicensePlateRecognizer(config.plate_model, device="cpu")
        except Exception as exc:
            raise ModelError(
                f"Cannot load the plate reading model '{config.plate_model}': {exc}. "
                "The first use downloads it, which needs a network connection."
            ) from exc

    @property
    def config(self) -> OcrConfig:
        return self._config

    def with_settings(
        self, config: OcrConfig, filters: PreprocessConfig = _NO_FILTERS
    ) -> PlateOcrEngine:
        """Return an engine that shares this one's loaded model but reads with new settings."""
        other = copy.copy(self)
        other._config = config
        other._filters = filters
        return other

    def prepare(self, image: Frame) -> Frame:
        # The model was trained on colour plates at their own size, so the crop is only changed
        # when a filter is on.
        return prepare_in_colour(image, self._config.min_crop_height, self._filters)

    def read(self, image: Frame) -> list[OcrFragment]:
        return self.read_traced(image).fragments

    def read_traced(self, image: Frame) -> TracedReading:
        try:
            prepared = self.prepare(image)
            # The model was trained on RGB images; OpenCV frames are BGR.
            code = cv.COLOR_GRAY2RGB if prepared.ndim == 2 else cv.COLOR_BGR2RGB
            rgb = cast(Frame, cv.cvtColor(prepared, code))
            prediction = self._recognizer.run(rgb, return_confidence=True)[0]
            text = str(prediction.plate).replace("_", "")
            probabilities = [float(value) for value in prediction.char_probs]
            confidence = sum(probabilities) / len(probabilities)
        except Exception as exc:  # One unreadable crop must not end a run.
            raise OcrError(f"The plate reader failed: {exc}") from exc
        if not text:
            return TracedReading(self.name, prepared, [])
        height, width = image.shape[:2]
        box = BoundingBox(0.0, 0.0, float(width), float(height))
        return TracedReading(self.name, prepared, [OcrFragment(text, confidence, box)])


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

    def read_traced(self, image: Frame) -> TracedReading:
        first = trace_reading(self._primary, image)
        if read_plate(first.fragments, self._fallback_confidence) is not None:
            return first
        second = trace_reading(self._fallback, image)
        return TracedReading(second.reader, second.prepared, second.fragments, set_aside=first)


def combine_readers(
    config: OcrConfig, plate: Callable[[], OcrEngine], general: Callable[[], OcrEngine]
) -> OcrEngine:
    """Return the reader named by ``ocr.engine``, loading only the models it needs."""
    if config.engine == "easyocr":
        return general()
    if config.engine == "plate":
        return plate()
    return HybridOcrEngine(plate(), general(), config.fallback_confidence)


def build_ocr_engine(config: OcrConfig, filters: PreprocessConfig = _NO_FILTERS) -> OcrEngine:
    """Create the engine named by ``ocr.engine``."""
    return combine_readers(
        config, lambda: PlateOcrEngine(config, filters), lambda: EasyOcrEngine(config, filters)
    )
