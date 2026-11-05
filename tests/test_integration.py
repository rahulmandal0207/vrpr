"""Run the real libraries. Needs the ml extra and downloads models on first use.

Run with: pytest -m integration
"""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from vrpr.config import DetectorConfig, OcrConfig
from vrpr.detector import YoloDetector
from vrpr.imaging import prepare_for_ocr
from vrpr.ocr import EasyOcrEngine
from vrpr.plates import read_plate

pytestmark = pytest.mark.integration


def rendered_plate(text: str) -> np.ndarray:
    image = np.full((90, 420, 3), 255, np.uint8)
    cv.putText(image, text, (15, 65), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv.LINE_AA)
    return image


def test_easyocr_reads_a_rendered_plate() -> None:
    pytest.importorskip("easyocr")
    engine = EasyOcrEngine(OcrConfig(gpu=False))
    image = prepare_for_ocr(rendered_plate("HR26DK8337"), 64)
    reading = read_plate(engine.read(image))
    assert reading is not None
    assert reading.plate == "HR26DK8337"


def test_yolo_adapter_runs_on_real_weights(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("ultralytics")
    monkeypatch.chdir(tmp_path)
    from ultralytics import YOLO

    YOLO("yolov8n.pt")  # Downloads the weights. COCO classes are enough to exercise the adapter.
    detector = YoloDetector(DetectorConfig(weights=Path("yolov8n.pt"), device="cpu"))
    assert isinstance(detector.detect(np.zeros((480, 640, 3), np.uint8)), list)
