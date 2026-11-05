from vrpr.config import TrackerConfig
from vrpr.models import BoundingBox, Detection
from vrpr.tracking import IouTracker


def detection(x: float, width: float = 100) -> Detection:
    return Detection(BoundingBox(x, 100, x + width, 140), 0.9)


def test_track_keeps_its_id_while_moving() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=1))
    ids = {tracker.update([detection(x)]).active[0].track_id for x in range(0, 100, 20)}
    assert ids == {1}


def test_motion_estimate_follows_a_fast_plate() -> None:
    # 60px per frame on a 100px box drops raw IoU to 0.25, and 90px to about 0.05.
    tracker = IouTracker(TrackerConfig(min_hits=1, iou_threshold=0.2))
    tracker.update([detection(0)])
    tracker.update([detection(60)])
    assert tracker.update([detection(150)]).active[0].track_id == 1


def test_unconfirmed_tracks_are_hidden() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=3))
    assert tracker.update([detection(0)]).active == []
    assert tracker.update([detection(5)]).active == []
    assert [track.track_id for track in tracker.update([detection(10)]).active] == [1]


def test_short_gap_does_not_change_the_id() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=1, max_age=2))
    tracker.update([detection(0)])
    tracker.update([])
    assert tracker.update([detection(5)]).active[0].track_id == 1


def test_track_expires_after_max_age() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=1, max_age=1))
    tracker.update([detection(0)])
    assert tracker.update([]).expired == []
    assert tracker.update([]).expired == [1]


def test_noise_that_never_confirmed_does_not_expire_visibly() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=2, max_age=0))
    tracker.update([detection(0)])
    assert tracker.update([]).expired == []


def test_two_plates_keep_separate_ids() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=1))
    for _ in range(3):
        update = tracker.update([detection(0), detection(400)])
    assert sorted(track.track_id for track in update.active) == [1, 2]


def test_each_detection_matches_at_most_one_track() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=1))
    tracker.update([detection(0), detection(10)])
    assert len(tracker.update([detection(5)]).active) == 1


def test_flush_reports_only_confirmed_tracks() -> None:
    tracker = IouTracker(TrackerConfig(min_hits=2))
    tracker.update([detection(0)])
    tracker.update([detection(0), detection(400)])
    assert tracker.flush() == [1]
    assert tracker.flush() == []


def test_next_vehicle_does_not_inherit_a_coasting_track() -> None:
    """With default settings, a gap between vehicles ends the first track."""
    tracker = IouTracker(TrackerConfig(min_hits=1))
    for x in range(0, 60, 6):
        tracker.update([detection(x)])
    for _ in range(TrackerConfig().max_age + 1):
        tracker.update([])
    predicted_position = 60 + 6 * (TrackerConfig().max_age + 1)
    assert tracker.update([detection(predicted_position)]).active[0].track_id == 2
