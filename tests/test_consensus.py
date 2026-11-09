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


def read(consensus: ConsensusEngine, track: int, plates: list[str], start: float = 0.0) -> None:
    """Add one reading per plate to a track, a tenth of a second apart."""
    for offset, plate in enumerate(plates):
        frame = round(start * 10) + offset
        consensus.add_reading(track, reading(plate), CROP, frame, start + offset / 10)


def plates(reports) -> list[str]:
    return [finalized.record.plate for finalized in reports]


# Reading a track


def test_a_track_stops_reading_once_readings_agree() -> None:
    consensus = engine(min_agreement=2)
    assert consensus.needs_ocr(1, 0)
    read(consensus, 1, ["HR26DK8337", "HR26DK8337"])
    assert not consensus.needs_ocr(1, 1)


def test_settled_track_is_rechecked_periodically() -> None:
    consensus = engine(min_agreement=1, recheck_every_n_frames=10)
    read(consensus, 1, ["HR26DK8337"])
    assert not consensus.needs_ocr(1, 9)
    assert consensus.needs_ocr(1, 10)


def test_misread_of_a_settled_plate_keeps_the_track_settled() -> None:
    consensus = engine(min_agreement=1)
    read(consensus, 1, ["HR26DK8337"])
    consensus.add_reading(1, reading("HR26DK8331"), CROP, 16, 1.6)
    assert not consensus.needs_ocr(1, 17)


# When a vehicle is reported


def test_vehicle_is_reported_once_it_has_been_quiet() -> None:
    consensus = engine(min_agreement=2, report_delay_seconds=3)
    read(consensus, 1, ["HR26DK8337", "HR26DK8337"])
    consensus.close_track(1)
    assert consensus.poll(2.0) == []
    reports = consensus.poll(3.2)
    assert plates(reports) == ["HR26DK8337"]
    assert reports[0].record.confirmed
    assert consensus.poll(10.0) == [], "a vehicle is reported only once"


def test_vehicle_still_in_view_is_not_reported() -> None:
    """The tracker lost the plate, but a new track already sees it again."""
    consensus = engine(min_agreement=2, report_delay_seconds=1)
    read(consensus, 1, ["HR26DK8337", "HR26DK8337"])
    consensus.close_track(1)
    read(consensus, 2, ["HR26DK8337"], start=0.5)
    assert consensus.poll(5.0) == []
    consensus.close_track(2)
    assert plates(consensus.poll(5.0)) == ["HR26DK8337"]


def test_flush_reports_everything_still_waiting() -> None:
    consensus = engine(report_delay_seconds=60)
    read(consensus, 1, ["HR26DK8337"])
    read(consensus, 2, ["MH12AB1234"])
    assert sorted(plates(consensus.flush())) == ["HR26DK8337", "MH12AB1234"]


# Pooling the tracks of one vehicle


def test_tracks_of_one_vehicle_are_pooled_into_one_report() -> None:
    """The tracker often restarts on the same plate; each restart used to be its own row."""
    consensus = engine(min_agreement=3)
    read(consensus, 1, ["HR26DK8337", "HR26DK8331"])
    consensus.close_track(1)
    read(consensus, 2, ["HR26DK8337", "HR26DK8337"], start=1.0)
    consensus.close_track(2)
    reports = consensus.flush()
    assert plates(reports) == ["HR26DK8337"]
    record = reports[0].record
    assert (record.votes, record.total_reads, record.track_id) == (3, 4, 1)
    assert record.confirmed


def test_near_identical_variants_vote_together() -> None:
    consensus = engine(min_agreement=3)
    read(consensus, 1, ["AP28CC6284"])
    consensus.close_track(1)
    read(consensus, 2, ["AP28CC4284", "AP28CC4284"], start=1.0)
    consensus.close_track(2)
    read(consensus, 3, ["AP28CC4284"], start=2.0)
    assert plates(consensus.flush()) == ["AP28CC4284"]


def test_different_plates_are_different_vehicles() -> None:
    consensus = engine()
    read(consensus, 1, ["HR26DK8337"])
    read(consensus, 2, ["MH12AB1234"])
    assert sorted(plates(consensus.flush())) == ["HR26DK8337", "MH12AB1234"]


def test_same_plate_after_the_window_is_a_new_sighting() -> None:
    consensus = engine(dedupe_window_seconds=30, report_delay_seconds=1)
    read(consensus, 1, ["HR26DK8337"])
    consensus.close_track(1)
    assert plates(consensus.poll(2.0)) == ["HR26DK8337"]
    read(consensus, 2, ["HR26DK8337"], start=90.0)
    consensus.close_track(2)
    assert plates(consensus.poll(92.0)) == ["HR26DK8337"]
    assert consensus.duplicates_suppressed == 0


def test_vehicle_seen_again_after_its_report_is_a_duplicate() -> None:
    consensus = engine(dedupe_window_seconds=30, report_delay_seconds=1)
    read(consensus, 1, ["HR26DK8337"])
    consensus.close_track(1)
    assert plates(consensus.poll(2.0)) == ["HR26DK8337"]
    read(consensus, 2, ["HR26DK8331"], start=10.0)
    consensus.close_track(2)
    assert consensus.poll(12.0) == []
    assert consensus.duplicates_suppressed == 1


def test_highest_confidence_crop_is_kept_across_tracks() -> None:
    clear = np.ones((10, 30, 3), np.uint8)
    consensus = engine()
    consensus.add_reading(1, reading(confidence=0.5), CROP, 0, 0.0)
    consensus.close_track(1)
    consensus.add_reading(2, reading(confidence=0.9), clear, 5, 0.5)
    assert consensus.flush()[0].crop is clear


# Confirmation


def test_a_single_reading_is_unconfirmed() -> None:
    consensus = engine(min_agreement=3)
    read(consensus, 1, ["HR26DK8337"])
    record = consensus.flush()[0].record
    assert record.plate == "HR26DK8337"
    assert not record.confirmed


def test_agreement_that_is_half_of_the_readings_is_confirmed() -> None:
    consensus = engine(min_agreement=3)
    read(consensus, 1, ["HR26DK8337"] * 3 + ["HR26DK8331", "HR26DK8381", "HR26DK8387"])
    assert consensus.flush()[0].record.confirmed


def test_agreement_outnumbered_by_rival_readings_is_unconfirmed() -> None:
    consensus = engine(min_agreement=3)
    read(consensus, 1, ["HR26DK8337"] * 3 + ["HR26DK8331"] * 2 + ["HR26DK8381"] * 2)
    record = consensus.flush()[0].record
    assert record.plate == "HR26DK8337"
    assert not record.confirmed


# Unread and switching vehicles


def test_weak_vehicle_is_not_reported() -> None:
    consensus = engine(min_votes_on_close=2)
    read(consensus, 1, ["HR26DK8337"])
    assert consensus.flush() == []
    assert consensus.tracks_unread == 1


def test_track_with_no_readings_counts_as_unread() -> None:
    consensus = engine()
    consensus.close_track(1)
    assert consensus.tracks_unread == 1


def test_track_that_moved_onto_another_vehicle_reports_both() -> None:
    """A settled track that inherits the next vehicle used to lose that plate."""
    consensus = engine(min_agreement=2)
    read(consensus, 1, ["HR26DK8337", "HR26DK8337"])
    consensus.add_reading(1, reading("MH12AB1234"), CROP, 16, 1.6)
    assert consensus.needs_ocr(1, 17), "reading at full rate again"
    consensus.add_reading(1, reading("MH12AB1234"), CROP, 17, 1.7)
    assert plates(consensus.flush()) == ["HR26DK8337", "MH12AB1234"]
