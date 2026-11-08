"""Files and progress for the web interface. Nothing here imports Streamlit."""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv

from vrpr.exceptions import ConfigError
from vrpr.models import Frame
from vrpr.video import is_live_source

VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v"})
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class RunResults:
    run_id: str
    path: Path
    summary: dict[str, Any] | None
    plates: list[dict[str, str]]

    @property
    def complete(self) -> bool:
        """A run stopped part way has plates but no summary."""
        return self.summary is not None


def list_videos(folder: Path) -> list[Path]:
    """Return the video files directly inside a folder, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def list_runs(output_dir: Path) -> list[Path]:
    """Return run folders, newest first. Run ids start with a timestamp, so names sort by time."""
    if not output_dir.is_dir():
        return []
    runs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and ((path / "summary.json").is_file() or (path / "plates.csv").is_file())
    ]
    return sorted(runs, key=lambda path: path.name, reverse=True)


def load_run(run_dir: Path) -> RunResults:
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    )
    plates: list[dict[str, str]] = []
    plates_path = run_dir / "plates.csv"
    if plates_path.is_file():
        with plates_path.open(newline="", encoding="utf-8") as file:
            plates = list(csv.DictReader(file))
    return RunResults(run_dir.name, run_dir, summary, plates)


def save_upload(name: str, data: bytes, folder: Path) -> Path:
    """Save an uploaded video under a safe, unused name inside ``folder``."""
    original = Path(name.replace("\\", "/")).name
    suffix = Path(original).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(VIDEO_SUFFIXES))
        raise ConfigError(f"Unsupported video type '{suffix or original}'. Use one of: {allowed}")

    stem = _UNSAFE_CHARACTERS.sub("_", Path(original).stem).strip("._") or "upload"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{suffix}"
    counter = 1
    while path.exists():
        path = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    path.write_bytes(data)
    return path


def frame_count(source: str) -> int | None:
    """Return how many frames a video file has, or None for cameras, streams and unknowns."""
    if is_live_source(source) or not Path(source).is_file():
        return None
    capture = cv.VideoCapture(source)
    try:
        count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count if count > 0 else None


class ThrottledObserver:
    """Passes frames to a callback at most once per interval, so the page stays responsive.

    It counts every frame, and always reports the first one so the preview appears at once.
    """

    def __init__(
        self,
        on_update: Callable[[Frame, int], None],
        interval_seconds: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_update = on_update
        self._interval = interval_seconds
        self._clock = clock
        self._last_update: float | None = None
        self.frames = 0

    def observe(self, frame: Frame) -> bool:
        self.frames += 1
        now = self._clock()
        if self._last_update is None or now - self._last_update >= self._interval:
            self._last_update = now
            self._on_update(frame, self.frames)
        return True

    def close(self) -> None:
        pass
