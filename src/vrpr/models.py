"""Plain data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Return the intersection over union with another box."""
        overlap = BoundingBox(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )
        if overlap.width <= 0 or overlap.height <= 0:
            return 0.0
        union = self.area + other.area - overlap.area
        return overlap.area / union if union > 0 else 0.0

    def clip(self, frame_width: int, frame_height: int) -> BoundingBox | None:
        """Return this box limited to the frame, or None if nothing is left."""
        clipped = BoundingBox(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(float(frame_width), self.x2),
            min(float(frame_height), self.y2),
        )
        return clipped if clipped.width > 0 and clipped.height > 0 else None


@dataclass(frozen=True, slots=True)
class Detection:
    """A plate found by the detector in one frame."""

    box: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class Track:
    """A detection associated with a stable identity across frames."""

    track_id: int
    box: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class OcrFragment:
    """One run of text returned by an OCR engine."""

    text: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class PlateReading:
    """A validated plate number from a single crop."""

    plate: str
    plate_format: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PlateRecord:
    """The final result for one vehicle."""

    run_id: str
    track_id: int
    plate: str
    plate_format: str
    confidence: float
    votes: int
    total_reads: int
    first_frame: int
    last_frame: int
    timestamp_s: float
    crop_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
