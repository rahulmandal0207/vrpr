"""Score a run against a hand-labelled list of the plates in a video."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vrpr.exceptions import ConfigError
from vrpr.plates import edit_distance, normalize

_PLATE_COLUMN = "plate"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    expected: int
    predicted: int
    correct: int
    precision: float
    recall: float
    f1: float
    character_error_rate: float


def read_plates(path: Path) -> list[str]:
    """Read the plate column of a CSV file."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if _PLATE_COLUMN not in (reader.fieldnames or []):
                raise ConfigError(f"{path} has no '{_PLATE_COLUMN}' column")
            return [normalize(row[_PLATE_COLUMN]) for row in reader if row[_PLATE_COLUMN]]
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc


def evaluate(expected: list[str], predicted: list[str]) -> EvaluationResult:
    """Compare plates as multisets: order does not matter, repeats do."""
    matched = Counter(expected) & Counter(predicted)
    correct = sum(matched.values())
    missed = list((Counter(expected) - matched).elements())
    spare = list((Counter(predicted) - matched).elements())

    # Character errors show how close the misses were. Each plate left unread
    # counts in full, so an empty run scores 1.0 rather than a flattering 0.
    errors = 0
    for plate in missed:
        if not spare:
            errors += len(plate)
            continue
        nearest = min(spare, key=lambda candidate: edit_distance(plate, candidate))
        spare.remove(nearest)
        errors += edit_distance(plate, nearest)
    characters = sum(len(plate) for plate in expected)

    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationResult(
        expected=len(expected),
        predicted=len(predicted),
        correct=correct,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        character_error_rate=round(errors / characters, 4) if characters else 0.0,
    )
