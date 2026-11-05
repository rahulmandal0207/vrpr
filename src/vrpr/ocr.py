"""Text recognition on plate crops."""

from __future__ import annotations

from typing import Protocol

from vrpr.config import OcrConfig
from vrpr.exceptions import DependencyError, OcrError
from vrpr.models import BoundingBox, Frame, OcrFragment

_PLATE_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class OcrEngine(Protocol):
    def read(self, image: Frame) -> list[OcrFragment]: ...


class EasyOcrEngine:
    def __init__(self, config: OcrConfig) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise DependencyError(
                "EasyOCR needs the ML dependencies. Install them with: pip install -e .[ml]"
            ) from exc
        self._reader = easyocr.Reader(list(config.languages), gpu=config.gpu)

    def read(self, image: Frame) -> list[OcrFragment]:
        try:
            raw = self._reader.readtext(image, allowlist=_PLATE_CHARACTERS)
        except Exception as exc:  # EasyOCR raises many internal types; one crop must not end a run.
            raise OcrError(f"EasyOCR failed: {exc}") from exc

        fragments = []
        for corners, text, confidence in raw:
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            fragments.append(OcrFragment(text, float(confidence), box))
        return fragments
