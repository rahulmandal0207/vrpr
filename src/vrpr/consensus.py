"""Turn many readings of a vehicle, across all its tracks, into one reported plate.

The tracker often loses a plate for a moment and starts a new track for it, so one vehicle can
have many tracks, each with a few misreads of its own. Each track collects readings. When a track
ends, its readings join the vehicle whose plate it resembles, if that vehicle was seen recently,
or start a new vehicle. A vehicle is reported once nothing has joined it for
``report_delay_seconds``, with the plate most of its readings agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vrpr.config import ConsensusConfig
from vrpr.models import Frame, PlateReading, PlateRecord
from vrpr.plates import edit_distance

# Plates this close are treated as misreads of each other.
_MISREAD_DISTANCE = 2


@dataclass(slots=True)
class _Tally:
    plate_format: str
    votes: int = 0
    confidence_sum: float = 0.0
    best_confidence: float = -1.0
    best_crop: Frame | None = None

    def absorb(self, other: _Tally) -> None:
        self.votes += other.votes
        self.confidence_sum += other.confidence_sum
        if other.best_confidence > self.best_confidence:
            self.best_confidence = other.best_confidence
            self.best_crop = other.best_crop


@dataclass(slots=True)
class _Evidence:
    """Readings of one track, or of all the tracks of one vehicle."""

    first_frame: int
    last_frame: int
    first_seen_s: float
    last_seen_s: float
    tallies: dict[str, _Tally] = field(default_factory=dict)
    total_reads: int = 0

    def leading(self) -> str:
        return max(
            self.tallies,
            key=lambda plate: (self.tallies[plate].votes, self.tallies[plate].confidence_sum),
        )

    def absorb(self, other: _Evidence) -> None:
        self.first_frame = min(self.first_frame, other.first_frame)
        self.last_frame = max(self.last_frame, other.last_frame)
        self.first_seen_s = min(self.first_seen_s, other.first_seen_s)
        self.last_seen_s = max(self.last_seen_s, other.last_seen_s)
        self.total_reads += other.total_reads
        for plate, tally in other.tallies.items():
            self.tallies.setdefault(plate, _Tally(tally.plate_format)).absorb(tally)


@dataclass(slots=True)
class _Track:
    evidence: _Evidence
    settled: bool = False


@dataclass(slots=True)
class _Vehicle:
    evidence: _Evidence
    track_ids: list[int]
    reported: bool = False


@dataclass(frozen=True, slots=True)
class FinalizedPlate:
    """A record ready to report, with the clearest crop of that plate."""

    record: PlateRecord
    crop: Frame | None


class ConsensusEngine:
    def __init__(self, config: ConsensusConfig, run_id: str) -> None:
        self._config = config
        self._run_id = run_id
        self._tracks: dict[int, _Track] = {}
        self._vehicles: list[_Vehicle] = []
        self.duplicates_suppressed = 0
        self.tracks_unread = 0

    def needs_ocr(self, track_id: int, frame_index: int) -> bool:
        """Say whether to read a track on this frame.

        Once ``min_agreement`` readings of a track agree, it is only rechecked now and then,
        which saves OCR while still noticing if the tracker has moved onto another vehicle.
        """
        track = self._tracks.get(track_id)
        if track is None or not track.settled:
            return True
        return frame_index % self._config.recheck_every_n_frames == 0

    def add_reading(
        self,
        track_id: int,
        reading: PlateReading,
        crop: Frame,
        frame_index: int,
        timestamp_s: float,
    ) -> None:
        track = self._tracks.get(track_id)
        if (
            track is not None
            and track.settled
            and edit_distance(track.evidence.leading(), reading.plate) > _MISREAD_DISTANCE
        ):
            # The tracker has moved onto another vehicle: what it read so far belongs to
            # the first one, and this track starts afresh.
            self._file(track_id, track.evidence)
            track = None
        if track is None:
            track = _Track(_Evidence(frame_index, frame_index, timestamp_s, timestamp_s))
            self._tracks[track_id] = track

        evidence = track.evidence
        evidence.last_frame = frame_index
        evidence.last_seen_s = timestamp_s
        evidence.total_reads += 1
        tally = evidence.tallies.setdefault(reading.plate, _Tally(reading.plate_format))
        tally.absorb(_Tally(reading.plate_format, 1, reading.confidence, reading.confidence, crop))
        if tally.votes >= self._config.min_agreement:
            track.settled = True

    def close_track(self, track_id: int) -> None:
        """Hand a finished track's readings to its vehicle."""
        track = self._tracks.pop(track_id, None)
        if track is None:
            self.tracks_unread += 1
            return
        self._file(track_id, track.evidence)

    def poll(self, now_s: float) -> list[FinalizedPlate]:
        """Report vehicles nothing has joined for ``report_delay_seconds``."""
        reports = []
        for vehicle in self._vehicles:
            idle = now_s - vehicle.evidence.last_seen_s >= self._config.report_delay_seconds
            if not vehicle.reported and idle and not self._still_in_view(vehicle):
                finalized = self._report(vehicle)
                if finalized is not None:
                    reports.append(finalized)
        window = self._config.dedupe_window_seconds
        self._vehicles = [
            v for v in self._vehicles if not v.reported or now_s - v.evidence.last_seen_s <= window
        ]
        return reports

    def flush(self) -> list[FinalizedPlate]:
        """Report every vehicle still waiting, at the end of a run."""
        for track_id in list(self._tracks):
            self.close_track(track_id)
        reports = []
        for vehicle in self._vehicles:
            if not vehicle.reported:
                finalized = self._report(vehicle)
                if finalized is not None:
                    reports.append(finalized)
        return reports

    def _match(self, evidence: _Evidence) -> _Vehicle | None:
        plate = evidence.leading()
        window = self._config.dedupe_window_seconds
        candidates = [
            (edit_distance(vehicle.evidence.leading(), plate), -vehicle.evidence.last_seen_s, i)
            for i, vehicle in enumerate(self._vehicles)
            if evidence.first_seen_s - vehicle.evidence.last_seen_s <= window
        ]
        close = [c for c in candidates if c[0] <= _MISREAD_DISTANCE]
        return self._vehicles[min(close)[2]] if close else None

    def _file(self, track_id: int, evidence: _Evidence) -> None:
        vehicle = self._match(evidence)
        if vehicle is None:
            self._vehicles.append(_Vehicle(evidence, [track_id]))
            return
        if vehicle.reported:
            # Seen again after it was reported: one row per sighting, not per track.
            self.duplicates_suppressed += 1
        vehicle.evidence.absorb(evidence)
        vehicle.track_ids.append(track_id)

    def _still_in_view(self, vehicle: _Vehicle) -> bool:
        plate = vehicle.evidence.leading()
        return any(
            edit_distance(track.evidence.leading(), plate) <= _MISREAD_DISTANCE
            for track in self._tracks.values()
        )

    def _report(self, vehicle: _Vehicle) -> FinalizedPlate | None:
        vehicle.reported = True
        evidence = vehicle.evidence
        plate = evidence.leading()
        tally = evidence.tallies[plate]
        if tally.votes < self._config.min_votes_on_close:
            self.tracks_unread += len(vehicle.track_ids)
            return None
        record = PlateRecord(
            run_id=self._run_id,
            track_id=vehicle.track_ids[0],
            plate=plate,
            plate_format=tally.plate_format,
            confidence=round(tally.confidence_sum / tally.votes, 4),
            votes=tally.votes,
            total_reads=evidence.total_reads,
            first_frame=evidence.first_frame,
            last_frame=evidence.last_frame,
            timestamp_s=round(evidence.first_seen_s, 3),
            # Enough readings agree, and no rival readings outnumber them. This measures
            # consistency, not truth: OCR can repeat the same misread many times.
            confirmed=(
                tally.votes >= self._config.min_agreement
                and tally.votes * 2 >= evidence.total_reads
            ),
        )
        return FinalizedPlate(record, tally.best_crop)
