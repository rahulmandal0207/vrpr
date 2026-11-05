"""Optional visual output: a live preview window and an annotated video file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol, cast

import cv2 as cv

from vrpr.exceptions import DisplayError
from vrpr.models import Frame, Track

_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 0, 255)
_WINDOW_NAME = "VRPR"
_QUIT_KEY = ord("q")


class FrameObserver(Protocol):
    def observe(self, frame: Frame) -> bool:
        """Handle an annotated frame. Return False to stop the run."""
        ...

    def close(self) -> None: ...


def annotate(frame: Frame, tracks: Sequence[Track], labels: Mapping[int, str]) -> Frame:
    """Return a copy of the frame with tracks drawn on it."""
    canvas = cast(Frame, frame.copy())
    for track in tracks:
        top_left = (int(track.box.x1), int(track.box.y1))
        bottom_right = (int(track.box.x2), int(track.box.y2))
        cv.rectangle(canvas, top_left, bottom_right, _BOX_COLOR, 2)
        label = labels.get(track.track_id, f"#{track.track_id}")
        origin = (top_left[0], max(20, top_left[1] - 10))
        cv.putText(canvas, label, origin, cv.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 2)
    return canvas


class WindowPreview:
    """Shows frames in a window. Pressing q stops the run."""

    def observe(self, frame: Frame) -> bool:
        try:
            cv.imshow(_WINDOW_NAME, frame)
            return cv.waitKey(1) & 0xFF != _QUIT_KEY
        except cv.error as exc:
            raise DisplayError(
                "Cannot open a preview window: this OpenCV build has no GUI support. "
                "See 'Live preview' in the README, or use --save-video instead."
            ) from exc

    def close(self) -> None:
        # This fails when the window never opened or the user already closed it.
        with suppress(cv.error):
            cv.destroyWindow(_WINDOW_NAME)


class AnnotatedVideoWriter:
    """Writes frames to a video file, sized from the first frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv.VideoWriter | None = None

    def observe(self, frame: Frame) -> bool:
        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv.VideoWriter.fourcc(*"mp4v")
            self._writer = cv.VideoWriter(str(self._path), fourcc, self._fps, (width, height))
        self._writer.write(frame)
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
