"""Persist results. Records are flushed one by one so a crash loses nothing."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Protocol

import cv2 as cv

from vrpr.models import Frame, PlateRecord

logger = logging.getLogger(__name__)

CSV_FIELDS = [item.name for item in fields(PlateRecord)]


class ResultSink(Protocol):
    def write(self, record: PlateRecord) -> None: ...

    def close(self) -> None: ...


class CsvSink:
    def __init__(self, path: Path) -> None:
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, record: PlateRecord) -> None:
        self._writer.writerow(record.to_dict())
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: PlateRecord) -> None:
        self._file.write(json.dumps(record.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class CropStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, track_id: int, plate: str, crop: Frame) -> Path | None:
        path = self._directory / f"{track_id:06d}_{plate}.png"
        if cv.imwrite(str(path), crop):
            return path
        logger.warning("Could not write crop for track %d", track_id)
        return None
