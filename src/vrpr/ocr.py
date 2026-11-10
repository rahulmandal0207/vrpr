"""Text recognition on plate crops."""

from __future__ import annotations

import copy
from typing import Any, Protocol

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
        self._config = config
        self._reader = easyocr.Reader(list(config.languages), gpu=config.gpu)

    @property
    def config(self) -> OcrConfig:
        return self._config

    def with_settings(self, config: OcrConfig) -> EasyOcrEngine:
        """Return an engine that shares this one's loaded reader but reads with ``config``.

        Languages and GPU use are the reader's own, so only reading settings change.
        """
        other = copy.copy(self)
        other._config = config
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

    def read(self, image: Frame) -> list[OcrFragment]:
        try:
            raw = self._reader.readtext(image, **self.readtext_options())
        except Exception as exc:  # EasyOCR raises many internal types; one crop must not end a run.
            raise OcrError(f"EasyOCR failed: {exc}") from exc

        fragments = []
        for corners, text, confidence in raw:
            xs = [float(point[0]) for point in corners]
            ys = [float(point[1]) for point in corners]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            fragments.append(OcrFragment(text, float(confidence), box))
        return fragments
