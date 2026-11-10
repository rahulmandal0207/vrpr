"""Run frames through detection, tracking, OCR and consensus."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vrpr.config import AppConfig
from vrpr.consensus import ConsensusEngine, FinalizedPlate
from vrpr.detector import Detector, YoloDetector
from vrpr.exceptions import OcrError
from vrpr.imaging import crop_plate, prepare_for_ocr
from vrpr.models import Track
from vrpr.ocr import EasyOcrEngine, OcrEngine
from vrpr.plate_details import describe_plate
from vrpr.plates import mask_plate, read_plate
from vrpr.rendering import AnnotatedVideoWriter, FrameObserver, WindowPreview, annotate
from vrpr.sinks import CropStore, CsvSink, JsonlSink, ResultSink
from vrpr.tracking import IouTracker
from vrpr.video import FramePacket, VideoSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    source: str
    frames: int
    duration_s: float
    frames_per_second: float
    detections: int
    ocr_calls: int
    ocr_errors: int
    plates: int
    tracks_unread: int
    duplicates_suppressed: int
    stopped_early: bool


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        run_id: str,
        detector: Detector,
        ocr: OcrEngine,
        sinks: list[ResultSink],
        crop_store: CropStore | None = None,
        observers: list[FrameObserver] | None = None,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._detector = detector
        self._ocr = ocr
        self._sinks = sinks
        self._crop_store = crop_store
        self._observers = observers or []
        self._tracker = IouTracker(config.tracker)
        self._consensus = ConsensusEngine(config.consensus, run_id)
        self._labels: dict[int, str] = {}
        self._detections = 0
        self._ocr_calls = 0
        self._ocr_errors = 0
        self._plates = 0

    def run(self, source: VideoSource, stop: threading.Event | None = None) -> RunSummary:
        """Process the source until it ends, then settle every open track."""
        started = time.perf_counter()
        frames = 0
        stopped_early = False
        try:
            for packet in source.frames():
                frames += 1
                tracks = self._process(packet)
                if not self._notify_observers(packet, tracks) or (stop and stop.is_set()):
                    stopped_early = True
                    break
        finally:
            # Results already collected are saved even if a stage raised.
            for track_id in self._tracker.flush():
                self._close_track(track_id)
            for finalized in self._consensus.flush():
                self._emit(finalized)
            self._close_outputs()

        duration = time.perf_counter() - started
        return RunSummary(
            run_id=self._run_id,
            source=source.name,
            frames=frames,
            duration_s=round(duration, 3),
            frames_per_second=round(frames / duration, 2) if duration > 0 else 0.0,
            detections=self._detections,
            ocr_calls=self._ocr_calls,
            ocr_errors=self._ocr_errors,
            plates=self._plates,
            tracks_unread=self._consensus.tracks_unread,
            duplicates_suppressed=self._consensus.duplicates_suppressed,
            stopped_early=stopped_early,
        )

    def _process(self, packet: FramePacket) -> list[Track]:
        detections = self._detector.detect(packet.frame)
        self._detections += len(detections)
        update = self._tracker.update(detections)

        if packet.index % self._config.ocr.every_n_frames == 0:
            for track in update.active:
                self._read_track(track, packet)
        for track_id in update.expired:
            self._close_track(track_id)
        for finalized in self._consensus.poll(packet.timestamp_s):
            self._emit(finalized)
        return update.active

    def _read_track(self, track: Track, packet: FramePacket) -> None:
        if not self._consensus.needs_ocr(track.track_id, packet.index):
            return
        settings = self._config.ocr
        crop = crop_plate(packet.frame, track.box, settings.crop_padding)
        if crop is None:
            return

        self._ocr_calls += 1
        try:
            fragments = self._ocr.read(
                prepare_for_ocr(crop, settings.min_crop_height, self._config.preprocess)
            )
        except OcrError as exc:
            self._ocr_errors += 1
            logger.warning("OCR failed on track %d: %s", track.track_id, exc)
            return

        reading = read_plate(fragments, settings.min_confidence)
        if reading is None:
            return
        self._consensus.add_reading(track.track_id, reading, crop, packet.index, packet.timestamp_s)

    def _close_track(self, track_id: int) -> None:
        self._labels.pop(track_id, None)
        self._consensus.close_track(track_id)

    def _emit(self, finalized: FinalizedPlate) -> None:
        record = finalized.record
        details = describe_plate(record.plate)
        if details is not None:
            record = replace(
                record,
                state=details.state,
                rto_code=details.rto_code,
                rto_office=details.rto_office,
                rto_area=details.rto_area,
                registered_year=details.registered_year,
            )
        if self._crop_store is not None and finalized.crop is not None:
            path = self._crop_store.save(record.track_id, record.plate, finalized.crop)
            if path is not None:
                record = replace(record, crop_path=str(path))
        for sink in self._sinks:
            sink.write(record)
        self._plates += 1
        self._labels[record.track_id] = record.plate
        logger.info(
            "Plate %s on track %d (%d of %d reads agree, confidence %.2f)%s",
            mask_plate(record.plate),
            record.track_id,
            record.votes,
            record.total_reads,
            record.confidence,
            "" if record.confirmed else ", unconfirmed",
        )

    def _notify_observers(self, packet: FramePacket, tracks: list[Track]) -> bool:
        if not self._observers:
            return True
        annotated = annotate(packet.frame, tracks, self._labels)
        # Every observer gets the frame, even if an earlier one asks to stop.
        keep_going = [observer.observe(annotated) for observer in self._observers]
        return all(keep_going)

    def _close_outputs(self) -> None:
        closables: list[Any] = [*self._sinks, *self._observers]
        for closable in closables:
            closable.close()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"


def execute_run(
    config: AppConfig,
    source: str,
    display: bool = False,
    stop: threading.Event | None = None,
    detector: Detector | None = None,
    ocr: OcrEngine | None = None,
    observers: Sequence[FrameObserver] = (),
) -> RunSummary:
    """Run the full pipeline on a source and write results to a new run directory.

    ``observers`` receive every annotated frame, for example to show progress.
    """
    # Models load first: a missing file should fail before any output is created.
    detector = detector or YoloDetector(config.detector)
    ocr = ocr or EasyOcrEngine(config.ocr)

    with VideoSource(source, config.source) as video:
        run_id = new_run_id()
        run_dir: Path = config.output.directory / run_id
        run_dir.mkdir(parents=True)

        frame_observers: list[FrameObserver] = list(observers)
        if display:
            frame_observers.append(WindowPreview())
        if config.output.save_annotated_video:
            frame_observers.append(AnnotatedVideoWriter(run_dir / "annotated.mp4", video.fps))

        pipeline = Pipeline(
            config,
            run_id,
            detector,
            ocr,
            sinks=[CsvSink(run_dir / "plates.csv"), JsonlSink(run_dir / "plates.jsonl")],
            crop_store=CropStore(run_dir / "crops") if config.output.save_crops else None,
            observers=frame_observers,
        )
        logger.info("Run %s started on %s", run_id, video.name)
        summary = pipeline.run(video, stop)

    (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    logger.info(
        "Run %s finished: %d plates from %d frames at %.1f FPS. Results in %s",
        run_id,
        summary.plates,
        summary.frames,
        summary.frames_per_second,
        run_dir,
    )
    return summary
