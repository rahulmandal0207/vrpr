"""End-to-end runs on real video files. The first group pins the bugs in the old main.py."""

import csv
import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import ScriptedDetector, StubOcr
from vrpr.config import AppConfig, OutputConfig
from vrpr.exceptions import OcrError, SourceError
from vrpr.imaging import crop_plate, prepare_for_ocr
from vrpr.models import BoundingBox, Frame, OcrFragment, Track
from vrpr.pipeline import execute_run
from vrpr.rendering import annotate


def box(x: float, y: float = 100, width: float = 80) -> BoundingBox:
    return BoundingBox(x, y, x + width, y + 30)


def rows(config: AppConfig, run_id: str) -> list[dict[str, str]]:
    path = config.output.directory / run_id / "plates.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class OcrByRegion:
    """Reads different text depending on which track's crop arrives, by call order."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self._calls = 0

    def read(self, image: Frame) -> list[OcrFragment]:
        text = self._texts[self._calls % len(self._texts)]
        self._calls += 1
        return [OcrFragment(text, 0.9, BoundingBox(0, 0, 100, 30))] if text else []


def test_csv_keeps_every_plate(config: AppConfig, make_video) -> None:
    """The old code reopened the CSV in 'w' mode, keeping only the last plate."""
    detector = ScriptedDetector(
        [[box(10)]] * 3 + [[]] * 3 + [[box(150)]] * 3,
    )
    ocr = StubOcr("HR26DK8337", "HR26DK8337", "HR26DK8337", "MH12AB1234")
    summary = execute_run(config, str(make_video(9)), detector=detector, ocr=ocr)
    assert [row["plate"] for row in rows(config, summary.run_id)] == ["HR26DK8337", "MH12AB1234"]


def test_two_plates_in_one_frame_are_both_kept(config: AppConfig, make_video) -> None:
    """The old `obj_id > currentId` check dropped whichever plate came second."""
    detector = ScriptedDetector([[box(10), box(200)]] * 4)
    ocr = OcrByRegion("HR26DK8337", "MH12AB1234")
    summary = execute_run(config, str(make_video(4)), detector=detector, ocr=ocr)
    plates = {row["plate"] for row in rows(config, summary.run_id)}
    assert plates == {"HR26DK8337", "MH12AB1234"}


def test_plate_read_after_a_newer_one_is_kept(config: AppConfig, make_video) -> None:
    """An older track that finishes reading late used to be discarded."""
    detector = ScriptedDetector([[box(10), box(200)]] * 6)
    # Track 2 settles in the first two frames. Track 1 is unreadable until after that.
    ocr = OcrByRegion("", "HR26DK8337", "", "HR26DK8337", "MH12AB1234", "", "MH12AB1234", "")
    summary = execute_run(config, str(make_video(6)), detector=detector, ocr=ocr)
    assert [(row["track_id"], row["plate"]) for row in rows(config, summary.run_id)] == [
        ("2", "HR26DK8337"),
        ("1", "MH12AB1234"),
    ]


def test_next_vehicle_on_the_same_track_is_still_reported(config: AppConfig, make_video) -> None:
    """Found on real footage: a coasting track took over the next vehicle in the lane."""
    settings = replace(config, consensus=replace(config.consensus, recheck_every_n_frames=4))
    detector = ScriptedDetector([[box(10)]] * 12)
    ocr = StubOcr("HR26DK8337", "HR26DK8337", "MH12AB1234")
    summary = execute_run(settings, str(make_video(12)), detector=detector, ocr=ocr)
    assert [row["plate"] for row in rows(settings, summary.run_id)] == [
        "HR26DK8337",
        "MH12AB1234",
    ]


def test_box_outside_the_frame_does_not_crash(config: AppConfig, make_video) -> None:
    """A negative coordinate used to give an empty crop, which crashed cvtColor."""
    detector = ScriptedDetector([[box(-40)], [box(-60)], [box(-500)], [box(1000)]])
    summary = execute_run(config, str(make_video(4)), detector=detector, ocr=StubOcr("HR26DK8337"))
    assert summary.frames == 4


def test_partly_visible_plate_is_still_read(config: AppConfig, make_video) -> None:
    detector = ScriptedDetector([[box(-20)]] * 3)
    summary = execute_run(config, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337"))
    assert summary.plates == 1


def test_crop_plate_returns_a_copy() -> None:
    """A crop that shares memory with the frame picks up anything drawn on it later."""
    frame = np.zeros((100, 100, 3), np.uint8)
    crop = crop_plate(frame, BoundingBox(10, 10, 50, 50))
    assert crop is not None
    frame[:] = 255
    assert crop.max() == 0


def test_annotate_leaves_the_source_frame_clean() -> None:
    """Drawing on the frame itself used to put the box into the OCR input and saved crop."""
    frame = np.zeros((100, 100, 3), np.uint8)
    canvas = annotate(frame, [Track(1, BoundingBox(10, 10, 50, 50), 0.9)], {})
    assert frame.max() == 0
    assert canvas.max() > 0


def test_output_directories_are_created(tmp_path: Path, config: AppConfig, make_video) -> None:
    """A fresh clone has no output folder, which used to raise FileNotFoundError."""
    nested = replace(config, output=OutputConfig(directory=tmp_path / "does" / "not" / "exist"))
    detector = ScriptedDetector([[box(10)]] * 3)
    summary = execute_run(nested, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337"))
    assert (nested.output.directory / summary.run_id / "plates.csv").is_file()


def test_lost_and_reacquired_plate_is_one_row(config: AppConfig, make_video) -> None:
    detector = ScriptedDetector([[box(10)]] * 3 + [[]] * 3 + [[box(10)]] * 3)
    summary = execute_run(config, str(make_video(9)), detector=detector, ocr=StubOcr("HR26DK8337"))
    assert len(rows(config, summary.run_id)) == 1
    assert summary.duplicates_suppressed == 1


def test_ocr_stops_once_a_plate_is_settled(config: AppConfig, make_video) -> None:
    ocr = StubOcr("HR26DK8337")
    detector = ScriptedDetector([[box(10)]] * 10)
    execute_run(config, str(make_video(10)), detector=detector, ocr=ocr)
    assert len(ocr.images) == config.consensus.min_agreement


def test_ocr_runs_on_the_configured_cadence(config: AppConfig, make_video) -> None:
    settings = replace(config, ocr=replace(config.ocr, every_n_frames=3))
    ocr = StubOcr("")
    detector = ScriptedDetector([[box(10)]] * 9)
    execute_run(settings, str(make_video(9)), detector=detector, ocr=ocr)
    assert len(ocr.images) == 3


def test_track_cut_off_by_end_of_video_is_reported(config: AppConfig, make_video) -> None:
    settings = replace(config, consensus=replace(config.consensus, min_agreement=50))
    detector = ScriptedDetector([[box(10)]] * 3)
    summary = execute_run(
        settings, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337")
    )
    assert summary.plates == 1


def test_one_failed_ocr_call_does_not_end_the_run(config: AppConfig, make_video) -> None:
    class Flaky(StubOcr):
        def read(self, image: Frame) -> list[OcrFragment]:
            if not self.images:
                self.images.append(image)
                raise OcrError("boom")
            return super().read(image)

    detector = ScriptedDetector([[box(10)]] * 4)
    ocr = Flaky("HR26DK8337")
    summary = execute_run(config, str(make_video(4)), detector=detector, ocr=ocr)
    assert summary.ocr_errors == 1
    assert summary.plates == 1


def test_results_survive_a_detector_crash(config: AppConfig, make_video) -> None:
    class Crashing(ScriptedDetector):
        def __init__(self) -> None:
            super().__init__([[box(10)]] * 3)
            self.calls = 0

        def detect(self, frame: Frame):
            self.calls += 1
            if self.calls > 3:
                raise RuntimeError("GPU fell over")
            return super().detect(frame)

    with pytest.raises(RuntimeError):
        execute_run(config, str(make_video(6)), detector=Crashing(), ocr=StubOcr("HR26DK8337"))

    runs = list(config.output.directory.iterdir())
    assert len(rows(config, runs[0].name)) == 1


def test_stop_event_ends_the_run_cleanly(config: AppConfig, make_video) -> None:
    stop = threading.Event()
    stop.set()
    detector = ScriptedDetector([[box(10)]] * 5)
    summary = execute_run(
        config, str(make_video(5)), stop=stop, detector=detector, ocr=StubOcr("HR26DK8337")
    )
    assert summary.stopped_early
    assert summary.frames == 1


def test_run_writes_all_artifacts(config: AppConfig, make_video) -> None:
    settings = replace(config, output=replace(config.output, save_annotated_video=True))
    detector = ScriptedDetector([[box(10)]] * 3)
    summary = execute_run(
        settings, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337")
    )

    run_dir = settings.output.directory / summary.run_id
    row = rows(settings, summary.run_id)[0]
    assert Path(row["crop_path"]).is_file()
    assert (run_dir / "annotated.mp4").stat().st_size > 0
    line = json.loads((run_dir / "plates.jsonl").read_text(encoding="utf-8"))
    assert line["plate"] == "HR26DK8337"
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["plates"] == 1


def test_crops_can_be_turned_off(config: AppConfig, make_video) -> None:
    settings = replace(config, output=replace(config.output, save_crops=False))
    detector = ScriptedDetector([[box(10)]] * 3)
    summary = execute_run(
        settings, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337")
    )
    assert not (settings.output.directory / summary.run_id / "crops").exists()
    assert rows(settings, summary.run_id)[0]["crop_path"] == ""


def test_full_plates_stay_out_of_the_logs(config: AppConfig, make_video, caplog) -> None:
    caplog.set_level("INFO")
    detector = ScriptedDetector([[box(10)]] * 3)
    execute_run(config, str(make_video(3)), detector=detector, ocr=StubOcr("HR26DK8337"))
    assert "HR26****37" in caplog.text
    assert "HR26DK8337" not in caplog.text


def test_missing_video_is_reported_before_output_is_created(config: AppConfig) -> None:
    with pytest.raises(SourceError, match="not found"):
        execute_run(config, "missing.mp4", detector=ScriptedDetector([]), ocr=StubOcr(""))
    assert not config.output.directory.exists()


def test_prepare_for_ocr_upscales_small_crops() -> None:
    small = np.zeros((20, 60, 3), np.uint8)
    assert prepare_for_ocr(small, 64).shape == (64, 192)


def test_prepare_for_ocr_leaves_large_crops_alone() -> None:
    large = np.zeros((100, 300, 3), np.uint8)
    assert prepare_for_ocr(large, 64).shape == (100, 300)
