"""Try detection and OCR settings on a single frame, to see how each one changes the reading.

The web page uses this for its tuning view. It runs the same steps as the pipeline for one
frame, without tracking or voting, and keeps every intermediate image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2 as cv

from vrpr.config import AppConfig
from vrpr.detector import Detector
from vrpr.exceptions import OcrError
from vrpr.imaging import crop_plate
from vrpr.models import Detection, Frame, OcrFragment, PlateReading
from vrpr.ocr import OcrEngine, TracedReading, trace_reading
from vrpr.plates import read_plate

MAX_PLATES = 6
_BOX_COLOUR = (0, 200, 0)
_LABEL_COLOUR = (255, 255, 255)


@dataclass(frozen=True)
class PlateTrial:
    """What happened to one detected plate: the crop, what the reader saw, and the reading.

    ``reader`` names the reader that answered, and ``set_aside`` is an earlier reader's attempt
    that was not kept.
    """

    number: int
    detection: Detection
    crop: Frame
    prepared: Frame
    fragments: list[OcrFragment]
    reading: PlateReading | None
    error: str | None = None
    reader: str = ""
    set_aside: TracedReading | None = None


def read_frame(source: str, index: int) -> Frame | None:
    """Read one frame of a video file, or return None if it has no such frame."""
    if not Path(source).is_file():
        return None
    capture = cv.VideoCapture(source)
    try:
        capture.set(cv.CAP_PROP_POS_FRAMES, max(0, index))
        success, frame = capture.read()
    finally:
        capture.release()
    return cast(Frame, frame) if success else None


def try_frame(
    frame: Frame,
    detector: Detector,
    ocr: OcrEngine,
    config: AppConfig,
    max_plates: int = MAX_PLATES,
) -> list[PlateTrial]:
    """Detect plates in ``frame`` and read each one, strongest detection first."""
    detections = sorted(detector.detect(frame), key=lambda found: -found.confidence)
    trials: list[PlateTrial] = []
    for detection in detections[:max_plates]:
        crop = crop_plate(frame, detection.box, config.ocr.crop_padding)
        if crop is None:
            continue
        number = len(trials) + 1
        try:
            traced = trace_reading(ocr, crop)
        except OcrError as exc:
            trials.append(PlateTrial(number, detection, crop, crop, [], None, str(exc)))
            continue
        reading = read_plate(traced.fragments, config.ocr.min_confidence)
        trials.append(
            PlateTrial(
                number,
                detection,
                crop,
                traced.prepared,
                traced.fragments,
                reading,
                reader=traced.reader,
                set_aside=traced.set_aside,
            )
        )
    return trials


def draw_trials(frame: Frame, trials: list[PlateTrial]) -> Frame:
    """Return a copy of ``frame`` with each plate boxed and numbered as in the trial list."""
    drawn = frame.copy()
    scale = max(1, round(frame.shape[0] / 720))
    for trial in trials:
        box = trial.detection.box
        top_left = (int(box.x1), int(box.y1))
        cv.rectangle(drawn, top_left, (int(box.x2), int(box.y2)), _BOX_COLOUR, 2 * scale)
        label = str(trial.number)
        (width, height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8 * scale, 2)
        label_top = max(0, top_left[1] - height - 8 * scale)
        cv.rectangle(
            drawn,
            (top_left[0], label_top),
            (top_left[0] + width + 8 * scale, top_left[1]),
            _BOX_COLOUR,
            cv.FILLED,
        )
        cv.putText(
            drawn,
            label,
            (top_left[0] + 4 * scale, top_left[1] - 4 * scale),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            _LABEL_COLOUR,
            2,
        )
    return cast(Frame, drawn)
