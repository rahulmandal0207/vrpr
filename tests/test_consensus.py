import numpy as np

from vrpr.config import ConsensusConfig
from vrpr.consensus import ConsensusEngine
from vrpr.models import PlateReading
from vrpr.plates import STANDARD

CROP = np.zeros((10, 30, 3), np.uint8)


def reading(plate: str = "HR26DK8337", confidence: float = 0.9) -> PlateReading:
    return PlateReading(plate, STANDARD, confidence)


def engine(**settings: float) -> ConsensusEngine:
    return ConsensusEngine(ConsensusConfig(**settings), "run")  # type: ignore[arg-type]


def test_reports_once_readings_agree_and_stops_ocr() -> None:
    consensus = engine(min_agreement=2)
    assert consensus.needs_ocr(1, 0)
    assert consensus.add_reading(1, reading(), CROP, 0, 0.0) is None
    finalized = consensus.add_reading(1, reading(), CROP, 1, 0.1)
    assert finalized is not None
    assert finalized.record.plate == "HR26DK8337"
    assert finalized.record.votes == 2
    assert not consensus.needs_ocr(1, 1)


def test_a_single_misread_is_outvoted() -> None:
    consensus = engine(min_agreement=3)
    consensus.add_reading(1, reading("HR26DK8331"), CROP, 0, 0.0)
    consensus.add_reading(1, reading(), CROP, 1, 0.1)
    consensus.add_reading(1, reading(), CROP, 2, 0.2)
    finalized = consensus.add_reading(1, reading(), CROP, 3, 0.3)
    assert finalized is not None
    assert finalized.record.plate == "HR26DK8337"
    assert finalized.record.total_reads == 4


def test_a_track_that_ends_early_still_reports_its_best_reading() -> None:
    consensus = engine(min_agreement=5)
    consensus.add_reading(1, reading("HR26DK8331", 0.3), CROP, 0, 0.0)
    consensus.add_reading(1, reading(confidence=0.8), CROP, 1, 0.1)
    finalized = consensus.close_track(1, 0.2)
    assert finalized is not None
    assert finalized.record.plate == "HR26DK8337"


def test_weak_track_is_not_reported_when_it_ends() -> None:
    consensus = engine(min_agreement=5, min_votes_on_close=2)
    consensus.add_reading(1, reading(), CROP, 0, 0.0)
    assert consensus.close_track(1, 0.1) is None
    assert consensus.tracks_unread == 1


def test_track_with_no_readings_counts_as_unread() -> None:
    consensus = engine()
    assert consensus.close_track(1, 0.0) is None
    assert consensus.tracks_unread == 1


def test_a_reported_track_is_not_reported_again_when_it_ends() -> None:
    consensus = engine(min_agreement=1)
    assert consensus.add_reading(1, reading(), CROP, 0, 0.0) is not None
    assert consensus.close_track(1, 0.1) is None
    assert consensus.tracks_unread == 0


def test_reacquired_vehicle_is_a_duplicate() -> None:
    consensus = engine(min_agreement=1, dedupe_window_seconds=30)
    assert consensus.add_reading(1, reading(), CROP, 0, 0.0) is not None
    assert consensus.add_reading(2, reading(), CROP, 50, 5.0) is None
    assert consensus.duplicates_suppressed == 1


def test_same_plate_after_the_window_is_a_new_sighting() -> None:
    consensus = engine(min_agreement=1, dedupe_window_seconds=30)
    assert consensus.add_reading(1, reading(), CROP, 0, 0.0) is not None
    assert consensus.add_reading(2, reading(), CROP, 900, 90.0) is not None


def test_highest_confidence_crop_is_kept() -> None:
    clear = np.ones((10, 30, 3), np.uint8)
    consensus = engine(min_agreement=2)
    consensus.add_reading(1, reading(confidence=0.5), CROP, 0, 0.0)
    finalized = consensus.add_reading(1, reading(confidence=0.9), clear, 1, 0.1)
    assert finalized is not None
    assert finalized.crop is clear


def test_settled_track_is_rechecked_periodically() -> None:
    consensus = engine(min_agreement=1, recheck_every_n_frames=10)
    consensus.add_reading(1, reading(), CROP, 0, 0.0)
    assert not consensus.needs_ocr(1, 9)
    assert consensus.needs_ocr(1, 10)


def test_track_that_moved_onto_another_vehicle_reports_both() -> None:
    """A settled track that inherits the next vehicle used to lose that plate."""
    consensus = engine(min_agreement=2)
    consensus.add_reading(1, reading("HR26DK8337"), CROP, 0, 0.0)
    assert consensus.add_reading(1, reading("HR26DK8337"), CROP, 1, 0.1) is not None

    assert consensus.add_reading(1, reading("MH12AB1234"), CROP, 16, 1.6) is None
    assert consensus.needs_ocr(1, 17)  # Reading at full rate again
    finalized = consensus.add_reading(1, reading("MH12AB1234"), CROP, 17, 1.7)
    assert finalized is not None
    assert finalized.record.plate == "MH12AB1234"


def test_new_vehicle_is_reported_if_its_track_ends_before_agreement() -> None:
    consensus = engine(min_agreement=3)
    for frame in range(3):
        consensus.add_reading(1, reading("HR26DK8337"), CROP, frame, 0.0)
    consensus.add_reading(1, reading("MH12AB1234"), CROP, 16, 1.6)
    finalized = consensus.close_track(1, 1.7)
    assert finalized is not None
    assert finalized.record.plate == "MH12AB1234"


def test_misread_of_a_reported_plate_is_not_a_new_vehicle() -> None:
    consensus = engine(min_agreement=1)
    consensus.add_reading(1, reading("HR26DK8337"), CROP, 0, 0.0)
    assert consensus.add_reading(1, reading("HR26DK8331"), CROP, 16, 1.6) is None
    assert not consensus.needs_ocr(1, 17)


def test_a_wild_misread_cannot_report_the_settled_plate_twice() -> None:
    consensus = engine(min_agreement=2)
    consensus.add_reading(1, reading("HR26DK8337"), CROP, 0, 0.0)
    assert consensus.add_reading(1, reading("HR26DK8337"), CROP, 1, 0.1) is not None
    consensus.add_reading(1, reading("MH12AB1234"), CROP, 16, 1.6)
    consensus.add_reading(1, reading("HR26DK8337"), CROP, 32, 3.2)
    assert consensus.add_reading(1, reading("HR26DK8337"), CROP, 33, 3.3) is None
    assert consensus.duplicates_suppressed == 1
