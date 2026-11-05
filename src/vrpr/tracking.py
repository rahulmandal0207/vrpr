"""Assign stable identities to plates across frames.

A greedy IoU tracker with a constant-velocity motion estimate. Plates are small
and few per frame, so this needs no assignment solver or Kalman filter. Tracks
report the detector's own box, which keeps them anchored to real pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vrpr.config import TrackerConfig
from vrpr.models import BoundingBox, Detection, Track

_VELOCITY_SMOOTHING = 0.5


@dataclass(slots=True)
class _TrackState:
    track_id: int
    box: BoundingBox
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    frames_missed: int = 0

    def predicted(self) -> BoundingBox:
        steps = self.frames_missed + 1
        dx, dy = self.velocity[0] * steps, self.velocity[1] * steps
        return BoundingBox(self.box.x1 + dx, self.box.y1 + dy, self.box.x2 + dx, self.box.y2 + dy)

    def observe(self, detection: Detection) -> None:
        steps = self.frames_missed + 1
        dx = (detection.box.x1 - self.box.x1) / steps
        dy = (detection.box.y1 - self.box.y1) / steps
        vx, vy = self.velocity
        self.velocity = (
            vx + _VELOCITY_SMOOTHING * (dx - vx),
            vy + _VELOCITY_SMOOTHING * (dy - vy),
        )
        self.box = detection.box
        self.confidence = detection.confidence
        self.hits += 1
        self.frames_missed = 0


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    """Tracks seen this frame, and the ids of tracks that have ended."""

    active: list[Track] = field(default_factory=list)
    expired: list[int] = field(default_factory=list)


class IouTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> TrackerUpdate:
        matches = self._match(detections)
        matched_tracks = {track.track_id for track, _ in matches}
        matched_detections = {id(detection) for _, detection in matches}

        for track, detection in matches:
            track.observe(detection)

        expired: list[int] = []
        surviving: list[_TrackState] = []
        for track in self._tracks:
            if track.track_id not in matched_tracks:
                track.frames_missed += 1
            if track.frames_missed <= self._config.max_age:
                surviving.append(track)
            elif self._is_confirmed(track):
                expired.append(track.track_id)
        self._tracks = surviving

        for detection in detections:
            if id(detection) not in matched_detections:
                self._start_track(detection)

        active = [
            Track(track.track_id, track.box, track.confidence)
            for track in self._tracks
            if track.frames_missed == 0 and self._is_confirmed(track)
        ]
        return TrackerUpdate(active, expired)

    def flush(self) -> list[int]:
        """End every track and return the ids of those that were confirmed."""
        ended = [track.track_id for track in self._tracks if self._is_confirmed(track)]
        self._tracks = []
        return ended

    def _is_confirmed(self, track: _TrackState) -> bool:
        return track.hits >= self._config.min_hits

    def _start_track(self, detection: Detection) -> None:
        self._tracks.append(_TrackState(self._next_id, detection.box, detection.confidence))
        self._next_id += 1

    def _match(self, detections: list[Detection]) -> list[tuple[_TrackState, Detection]]:
        pairs = [
            (track.predicted().iou(detection.box), index, position)
            for index, track in enumerate(self._tracks)
            for position, detection in enumerate(detections)
        ]
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        matches: list[tuple[_TrackState, Detection]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for overlap, index, position in pairs:
            if overlap < self._config.iou_threshold:
                break
            if index in used_tracks or position in used_detections:
                continue
            used_tracks.add(index)
            used_detections.add(position)
            matches.append((self._tracks[index], detections[position]))
        return matches
