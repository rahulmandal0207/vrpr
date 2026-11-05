"""Read frames from video files, cameras and network streams."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

import cv2 as cv

from vrpr.config import SourceConfig
from vrpr.exceptions import SourceError
from vrpr.models import Frame

logger = logging.getLogger(__name__)

_FALLBACK_FPS = 30.0
_URL_CREDENTIALS = re.compile(r"://[^/@]+@")


@dataclass(frozen=True, slots=True)
class FramePacket:
    index: int
    timestamp_s: float
    frame: Frame


def is_live_source(source: str) -> bool:
    """Cameras are given by index and streams by URL. Anything else is a file."""
    return source.isdigit() or "://" in source


def redact_source(source: str) -> str:
    """Remove credentials from a stream URL so it is safe to log or store."""
    return _URL_CREDENTIALS.sub("://***@", source)


class VideoSource:
    """A context manager that yields frames and reconnects live sources."""

    def __init__(self, source: str, config: SourceConfig) -> None:
        self._source = source
        self.name = redact_source(source)
        self._config = config
        self._live = is_live_source(source)
        self._capture: cv.VideoCapture | None = None
        self.fps = _FALLBACK_FPS

    def __enter__(self) -> VideoSource:
        if not self._live and not Path(self._source).is_file():
            raise SourceError(f"Video file not found: {self.name}")
        self._capture = self._open()
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(f"Could not open video source: {self.name}")
        reported_fps = self._capture.get(cv.CAP_PROP_FPS)
        if reported_fps > 0:
            self.fps = reported_fps
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open(self) -> cv.VideoCapture:
        return cv.VideoCapture(int(self._source) if self._source.isdigit() else self._source)

    def frames(self) -> Iterator[FramePacket]:
        if self._capture is None:
            raise SourceError("VideoSource must be opened with a 'with' block")

        started = time.monotonic()
        index = 0
        failures = 0
        while True:
            success, frame = self._capture.read()
            if success:
                failures = 0
                elapsed = time.monotonic() - started if self._live else index / self.fps
                yield FramePacket(index, elapsed, cast(Frame, frame))
                index += 1
                continue

            if not self._live:
                return
            failures += 1
            if failures > self._config.max_read_failures:
                logger.error("Giving up on %s after %d failed reads", self.name, failures - 1)
                return
            logger.warning("Lost %s, reconnecting (attempt %d)", self.name, failures)
            time.sleep(self._config.retry_delay_seconds)
            self._capture.release()
            self._capture = self._open()
