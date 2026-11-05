"""Plate detection. ML imports stay inside functions so the core runs without them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from vrpr.config import DetectorConfig
from vrpr.exceptions import DependencyError, ModelError
from vrpr.models import BoundingBox, Detection, Frame


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


def load_yolo(weights: str | Path) -> Any:
    """Load a YOLO model, or explain which dependency is missing."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DependencyError(
            "YOLO needs the ML dependencies. Install them with: pip install -e .[ml]"
        ) from exc
    return YOLO(str(weights))


class YoloDetector:
    def __init__(self, config: DetectorConfig) -> None:
        if not config.weights.is_file():
            raise ModelError(
                f"Plate detector weights not found at {config.weights}. "
                "Train a model with 'vrpr train', or set detector.weights."
            )
        self._config = config
        self._model = load_yolo(config.weights)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self._config.confidence,
            imgsz=self._config.image_size,
            device=self._config.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(BoundingBox(x1, y1, x2, y2), float(box.conf[0])))
        return detections
