"""Normalize OCR text and recognize Indian registration plate formats."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vrpr.models import OcrFragment, PlateReading

STANDARD = "standard"
BHARAT = "bharat"

STATE_CODES = frozenset(
    [
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "DD",
        "DL",
        "DN",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "OR",
        "PB",
        "PY",
        "RJ",
        "SK",
        "TG",
        "TN",
        "TR",
        "TS",
        "UK",
        "UA",
        "UP",
        "WB",
    ]
)

# Only Delhi writes its RTO code without a leading zero, as in DL 3C AB 1234.
_ONE_DIGIT_RTO_STATES = frozenset({"DL"})
_BHARAT_RE = re.compile(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}")

# OCR confuses these glyphs. Each map is applied only where the plate layout
# requires that character class, so a real letter is never rewritten in a letter slot.
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"})
_TO_DIGIT = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "A": "4",
        "S": "5",
        "G": "6",
        "B": "8",
        "T": "7",
    }
)

_NOT_ALNUM = re.compile(r"[^A-Z0-9]")
_COUNTRY_MARK = "IND"
_LONG_SERIES = 3


@dataclass(frozen=True, slots=True)
class ParsedPlate:
    plate: str
    plate_format: str
    corrections: int


def normalize(text: str) -> str:
    """Uppercase the text and drop everything except letters and digits."""
    return _NOT_ALNUM.sub("", text.upper())


def _as_letters(text: str) -> str:
    return text.translate(_TO_LETTER)


def _as_digits(text: str) -> str:
    return text.translate(_TO_DIGIT)


def _count_changes(original: str, corrected: str) -> int:
    return sum(a != b for a, b in zip(original, corrected, strict=True))


def _parse_standard(cleaned: str) -> ParsedPlate | None:
    if not 8 <= len(cleaned) <= 11:
        return None

    scored: list[tuple[int, ParsedPlate]] = []
    series_end = len(cleaned) - 4
    # The RTO code is one or two digits, so the layout is ambiguous until both are tried.
    for rto_length in (2, 1):
        rto_end = 2 + rto_length
        if rto_end > series_end:
            continue
        series = cleaned[rto_end:series_end]
        corrected_series = _as_letters(series)
        corrected = (
            _as_letters(cleaned[:2])
            + _as_digits(cleaned[2:rto_end])
            + corrected_series
            + _as_digits(cleaned[series_end:])
        )
        state, rto, number = corrected[:2], corrected[2:rto_end], corrected[series_end:]
        # Check each part in its own slot. A pattern for the whole plate let UP8CB0067
        # through the two-digit layout, because the pattern also allows one digit.
        # Series letters are required: without them, fragments of a longer plate pass.
        if not (
            state in STATE_CODES
            and rto.isdigit()
            and corrected_series.isalpha()
            and len(corrected_series) <= _LONG_SERIES
            and number.isdigit()
        ):
            continue
        # A dropped digit, as in UP8CB0067 for UP80CB0067, would otherwise read as a
        # valid one-digit RTO code.
        if rto_length == 1 and state not in _ONE_DIGIT_RTO_STATES:
            continue
        # Three-letter series are rare. Accept one only as read, never by repairing
        # a stray digit into it, and prefer any other layout that fits as well.
        is_long_series = len(series) == _LONG_SERIES
        if is_long_series and corrected_series != series:
            continue
        changes = _count_changes(cleaned, corrected)
        scored.append((changes + is_long_series, ParsedPlate(corrected, STANDARD, changes)))
    return min(scored, key=lambda item: item[0], default=(0, None))[1]


def _parse_bharat(cleaned: str) -> ParsedPlate | None:
    if len(cleaned) not in (9, 10):
        return None
    corrected = (
        _as_digits(cleaned[:2])
        + _as_letters(cleaned[2:4])
        + _as_digits(cleaned[4:8])
        + _as_letters(cleaned[8:])
    )
    if not _BHARAT_RE.fullmatch(corrected):
        return None
    return ParsedPlate(corrected, BHARAT, _count_changes(cleaned, corrected))


def parse_plate(text: str) -> ParsedPlate | None:
    """Return the plate in ``text``, repairing common OCR confusions, or None."""
    cleaned = normalize(text)
    if cleaned.startswith(_COUNTRY_MARK):
        cleaned = cleaned[len(_COUNTRY_MARK) :]
    return _parse_bharat(cleaned) or _parse_standard(cleaned)


def reading_order(fragments: Iterable[OcrFragment]) -> list[OcrFragment]:
    """Sort fragments into lines from top to bottom, each from left to right."""
    lines: list[list[OcrFragment]] = []
    for fragment in sorted(fragments, key=lambda f: (f.box.y1 + f.box.y2) / 2):
        center = (fragment.box.y1 + fragment.box.y2) / 2
        if lines:
            anchor = lines[-1][0].box
            if abs(center - (anchor.y1 + anchor.y2) / 2) <= anchor.height / 2:
                lines[-1].append(fragment)
                continue
        lines.append([fragment])
    return [fragment for line in lines for fragment in sorted(line, key=lambda f: f.box.x1)]


def _candidate_spans(count: int) -> list[tuple[int, int]]:
    """List contiguous fragment ranges, widest first."""
    return [
        (start, start + width)
        for width in range(count, 0, -1)
        for start in range(count - width + 1)
    ]


def read_plate(
    fragments: Sequence[OcrFragment], min_confidence: float = 0.0
) -> PlateReading | None:
    """Assemble OCR fragments into a plate.

    Two-line plates arrive as separate fragments, and plates often carry extra
    text such as a dealer name, so every contiguous run of fragments is tried.
    """
    usable = [
        fragment
        for fragment in reading_order(fragments)
        if fragment.confidence >= min_confidence and normalize(fragment.text) != _COUNTRY_MARK
    ]
    for start, end in _candidate_spans(len(usable)):
        span = usable[start:end]
        parsed = parse_plate("".join(fragment.text for fragment in span))
        if parsed is not None:
            confidence = sum(fragment.confidence for fragment in span) / len(span)
            return PlateReading(parsed.plate, parsed.plate_format, confidence)
    return None


def edit_distance(first: str, second: str) -> int:
    """Return the Levenshtein distance between two strings."""
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def mask_plate(plate: str) -> str:
    """Hide the middle of a plate so logs do not carry personal data."""
    if len(plate) <= 6:
        return "*" * len(plate)
    return plate[:4] + "*" * (len(plate) - 6) + plate[-2:]
