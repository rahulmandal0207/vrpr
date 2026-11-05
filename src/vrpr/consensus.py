"""Turn many per-frame readings of a vehicle into one reported plate."""

from __future__ import annotations

from dataclasses import dataclass, field

from vrpr.config import ConsensusConfig
from vrpr.models import Frame, PlateReading, PlateRecord
from vrpr.plates import edit_distance

# Readings this close to a reported plate are treated as misreads of it.
_MISREAD_DISTANCE = 2


@dataclass(slots=True)
class _Tally:
    plate_format: str
    votes: int = 0
    confidence_sum: float = 0.0
    best_confidence: float = -1.0
    best_crop: Frame | None = None


@dataclass(slots=True)
class _TrackVotes:
    first_frame: int
    last_frame: int
    timestamp_s: float
    tallies: dict[str, _Tally] = field(default_factory=dict)
    total_reads: int = 0
    reported_plate: str | None = None


@dataclass(frozen=True, slots=True)
class FinalizedPlate:
    """A record ready to report, with the clearest crop of that plate."""

    record: PlateRecord
    crop: Frame | None


class ConsensusEngine:
    def __init__(self, config: ConsensusConfig, run_id: str) -> None:
        self._config = config
        self._run_id = run_id
        self._tracks: dict[int, _TrackVotes] = {}
        self._last_reported: dict[str, float] = {}
        self.duplicates_suppressed = 0
        self.tracks_unread = 0

    def needs_ocr(self, track_id: int, frame_index: int) -> bool:
        """Say whether to read a track on this frame.

        A settled track is only rechecked now and then, which saves OCR while
        still noticing if the tracker has moved it onto a different vehicle.
        """
        votes = self._tracks.get(track_id)
        if votes is None or votes.reported_plate is None:
            return True
        return frame_index % self._config.recheck_every_n_frames == 0

    def add_reading(
        self,
        track_id: int,
        reading: PlateReading,
        crop: Frame,
        frame_index: int,
        timestamp_s: float,
    ) -> FinalizedPlate | None:
        """Record a reading. Returns a result as soon as enough readings agree."""
        votes = self._tracks.get(track_id)
        if votes is None or self._is_different_vehicle(votes, reading.plate):
            # A false alarm costs little: if the old plate wins again, it is a repeat
            # and gets suppressed. Missing a real switch would lose a vehicle.
            votes = _TrackVotes(frame_index, frame_index, timestamp_s)
            self._tracks[track_id] = votes
        votes.last_frame = frame_index
        votes.total_reads += 1

        tally = votes.tallies.setdefault(reading.plate, _Tally(reading.plate_format))
        tally.votes += 1
        tally.confidence_sum += reading.confidence
        if reading.confidence > tally.best_confidence:
            tally.best_confidence = reading.confidence
            tally.best_crop = crop

        if votes.reported_plate is not None or tally.votes < self._config.min_agreement:
            return None
        return self._finalize(track_id, votes, reading.plate, timestamp_s)

    def close_track(self, track_id: int, timestamp_s: float) -> FinalizedPlate | None:
        """Report whatever a finished track collected, if it is strong enough."""
        votes = self._tracks.pop(track_id, None)
        if votes is None:
            self.tracks_unread += 1
            return None
        if votes.reported_plate is not None:
            return None

        plate = max(votes.tallies, key=lambda key: votes.tallies[key].confidence_sum)
        if votes.tallies[plate].votes < self._config.min_votes_on_close:
            self.tracks_unread += 1
            return None
        return self._finalize(track_id, votes, plate, timestamp_s)

    @staticmethod
    def _is_different_vehicle(votes: _TrackVotes, plate: str) -> bool:
        reported = votes.reported_plate
        return reported is not None and edit_distance(reported, plate) > _MISREAD_DISTANCE

    def _finalize(
        self, track_id: int, votes: _TrackVotes, plate: str, timestamp_s: float
    ) -> FinalizedPlate | None:
        votes.reported_plate = plate
        self._forget_old_plates(timestamp_s)
        if plate in self._last_reported:
            # A vehicle lost and re-acquired by the tracker gets a new id, not a new row.
            self._last_reported[plate] = timestamp_s
            self.duplicates_suppressed += 1
            return None
        self._last_reported[plate] = timestamp_s

        tally = votes.tallies[plate]
        record = PlateRecord(
            run_id=self._run_id,
            track_id=track_id,
            plate=plate,
            plate_format=tally.plate_format,
            confidence=round(tally.confidence_sum / tally.votes, 4),
            votes=tally.votes,
            total_reads=votes.total_reads,
            first_frame=votes.first_frame,
            last_frame=votes.last_frame,
            timestamp_s=round(votes.timestamp_s, 3),
        )
        return FinalizedPlate(record, tally.best_crop)

    def _forget_old_plates(self, now_s: float) -> None:
        window = self._config.dedupe_window_seconds
        self._last_reported = {
            plate: seen for plate, seen in self._last_reported.items() if now_s - seen <= window
        }
