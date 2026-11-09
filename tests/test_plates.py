import pytest

from vrpr.models import BoundingBox, OcrFragment
from vrpr.plates import (
    BHARAT,
    STANDARD,
    mask_plate,
    normalize,
    parse_plate,
    read_plate,
    reading_order,
)


def fragment(text: str, y: float = 0, x: float = 0, confidence: float = 0.9) -> OcrFragment:
    return OcrFragment(text, confidence, BoundingBox(x, y, x + 50, y + 20))


@pytest.mark.parametrize(
    ("text", "expected", "plate_format"),
    [
        ("HR26DK8337", "HR26DK8337", STANDARD),
        ("HR 26 DK 8337", "HR26DK8337", STANDARD),
        ("hr-26-dk-8337", "HR26DK8337", STANDARD),
        ("HR26.DK8337 ", "HR26DK8337", STANDARD),
        ("DL 3C AB 1234", "DL3CAB1234", STANDARD),
        ("DL10CAB1234", "DL10CAB1234", STANDARD),
        ("UP80CB0067", "UP80CB0067", STANDARD),
        ("22 BH 1234 AA", "22BH1234AA", BHARAT),
        ("22BH1234A", "22BH1234A", BHARAT),
        ("IND HR26DK8337", "HR26DK8337", STANDARD),
    ],
)
def test_accepts_real_formats(text: str, expected: str, plate_format: str) -> None:
    parsed = parse_plate(text)
    assert parsed is not None
    assert (parsed.plate, parsed.plate_format) == (expected, plate_format)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HR2GDK8337", "HR26DK8337"),  # G in a digit slot
        ("HR26DK83B7", "HR26DK8387"),  # B in a digit slot
        ("0D02AB1234", "OD02AB1234"),  # 0 in a letter slot
        ("MH12ABI234", "MH12AB1234"),
        ("22BH12E4AA", None),  # E has no digit lookalike
    ],
)
def test_repairs_confusions_only_where_the_layout_demands(text: str, expected: str | None) -> None:
    parsed = parse_plate(text)
    assert (parsed.plate if parsed else None) == expected


def test_series_letter_is_not_rewritten_as_a_digit() -> None:
    parsed = parse_plate("DL3S1234")
    assert parsed is not None
    assert parsed.plate == "DL3S1234"
    assert parsed.corrections == 0


@pytest.mark.parametrize(
    "text", ["", "IND", "HR26", "HR26DK833", "HR26DK83377", "XX26DK8337", "ABCDEFGH", "12345678"]
)
def test_rejects_non_plates(text: str) -> None:
    assert parse_plate(text) is None


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("UP8CB0067", "UP80CB0067 with a digit dropped, misread on real dashcam footage"),
        ("PB0CD3389", "a fragment of UP80CD3389 from the same footage"),
        ("HR6KA8337", "HR26KA8337 with a digit dropped: only Delhi writes a one-digit RTO code"),
        ("KA051234", "no series letters: usually part of a longer plate"),
    ],
)
def test_rejects_fragments_that_look_like_plates(text: str, why: str) -> None:
    assert parse_plate(text) is None, why


def test_one_digit_rto_codes_are_kept_for_delhi() -> None:
    parsed = parse_plate("DL 3C AB 1234")
    assert parsed is not None
    assert parsed.plate == "DL3CAB1234"


def test_a_dropped_letter_repair_prefers_the_two_digit_code() -> None:
    """With one-digit codes limited to Delhi, MH1Z... can only be MH 12."""
    parsed = parse_plate("MH1ZAB1234")
    assert parsed is not None
    assert parsed.plate == "MH12AB1234"


def test_normalize_keeps_only_uppercase_alphanumerics() -> None:
    assert normalize(" hr-26.dk 8337\n") == "HR26DK8337"


def test_reading_order_handles_two_lines_given_out_of_order() -> None:
    fragments = [fragment("DK8337", y=30), fragment("26", y=0, x=60), fragment("HR", y=0)]
    assert [f.text for f in reading_order(fragments)] == ["HR", "26", "DK8337"]


def test_read_plate_joins_a_two_line_plate() -> None:
    reading = read_plate([fragment("DK8337", y=30), fragment("HR26", y=0)])
    assert reading is not None
    assert reading.plate == "HR26DK8337"


def test_read_plate_skips_the_ind_strip() -> None:
    reading = read_plate([fragment("IND", x=0), fragment("HR26DK8337", x=60)])
    assert reading is not None
    assert reading.plate == "HR26DK8337"


def test_read_plate_skips_an_ind_mark_between_two_lines() -> None:
    fragments = [fragment("HR26", y=0), fragment("IND", y=15), fragment("DK8337", y=30)]
    reading = read_plate(fragments)
    assert reading is not None
    assert reading.plate == "HR26DK8337"


def test_read_plate_ignores_surrounding_text() -> None:
    fragments = [fragment("SHARMA MOTORS", y=0), fragment("HR26DK8337", y=30)]
    reading = read_plate(fragments)
    assert reading is not None
    assert reading.plate == "HR26DK8337"


def test_read_plate_drops_low_confidence_fragments() -> None:
    assert read_plate([fragment("HR26DK8337", confidence=0.1)], min_confidence=0.5) is None


def test_read_plate_with_no_text() -> None:
    assert read_plate([]) is None


def test_mask_plate_hides_the_middle() -> None:
    assert mask_plate("HR26DK8337") == "HR26****37"
    assert mask_plate("ABCD") == "****"
