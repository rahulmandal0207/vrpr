"""Shared fakes. They stand in for the ML models so the suite needs no GPU or weights."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from vrpr.config import AppConfig, ConsensusConfig, OcrConfig, OutputConfig, TrackerConfig
from vrpr.models import BoundingBox, Detection, Frame, OcrFragment

FRAME_WIDTH = 320
FRAME_HEIGHT = 240


class ScriptedDetector:
    """Returns a prepared list of detections for each frame."""

    def __init__(self, script: list[list[BoundingBox]]) -> None:
        self._script = iter(script)

    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection(box, 0.9) for box in next(self._script, [])]


class StubOcr:
    """Returns prepared text, in order, repeating the last entry."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.images: list[Frame] = []

    def read(self, image: Frame) -> list[OcrFragment]:
        self.images.append(image.copy())
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        if not text:
            return []
        return [OcrFragment(text, 0.9, BoundingBox(0, 0, 100, 30))]


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in an empty folder, so the CLI never reads the developer's own .env."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ocr=OcrConfig(every_n_frames=1),
        tracker=TrackerConfig(max_age=1, min_hits=1),
        consensus=ConsensusConfig(min_agreement=2),
        output=OutputConfig(directory=tmp_path / "output"),
    )


@pytest.fixture
def make_video(tmp_path: Path):
    def _make(frames: int = 10, name: str = "clip.mp4") -> Path:
        path = tmp_path / name
        writer = cv.VideoWriter(
            str(path), cv.VideoWriter.fourcc(*"mp4v"), 10.0, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        assert writer.isOpened()
        for _ in range(frames):
            writer.write(np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 200, np.uint8))
        writer.release()
        return path

    return _make
