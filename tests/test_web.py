"""The web interface: file helpers, the page itself, and the vrpr web command."""

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from vrpr import cli
from vrpr.exceptions import ConfigError
from vrpr.models import BoundingBox, Detection, OcrFragment
from vrpr.web import runs

APP = Path(runs.__file__).with_name("app.py")
ENTRY_POINT = Path(__file__).resolve().parent.parent / "streamlit_app.py"


# Finding videos and runs


def test_list_videos_returns_only_videos_sorted(tmp_path: Path) -> None:
    for name in ("b.webm", "A.MP4", "notes.txt", "c.mov"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "folder.mp4").mkdir()
    assert [path.name for path in runs.list_videos(tmp_path)] == ["A.MP4", "b.webm", "c.mov"]


def test_list_videos_of_a_missing_folder(tmp_path: Path) -> None:
    assert runs.list_videos(tmp_path / "missing") == []


def write_run(
    output: Path,
    run_id: str,
    plates: list[str],
    summary: bool = True,
    unconfirmed: frozenset[str] = frozenset(),
) -> Path:
    run_dir = output / run_id
    run_dir.mkdir(parents=True)
    with (run_dir / "plates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["plate", "confidence", "confirmed"])
        writer.writeheader()
        writer.writerows(
            {"plate": plate, "confidence": "0.9", "confirmed": str(plate not in unconfirmed)}
            for plate in plates
        )
    if summary:
        (run_dir / "summary.json").write_text(json.dumps({"frames": 10}), encoding="utf-8")
    return run_dir


def test_list_runs_newest_first_and_skips_other_folders(tmp_path: Path) -> None:
    write_run(tmp_path, "20261108T100000Z-aaaaaa", [])
    write_run(tmp_path, "20261108T120000Z-bbbbbb", [], summary=False)
    (tmp_path / "empty").mkdir()
    (tmp_path / "stray.txt").write_text("", encoding="utf-8")
    names = [path.name for path in runs.list_runs(tmp_path)]
    assert names == ["20261108T120000Z-bbbbbb", "20261108T100000Z-aaaaaa"]


def test_load_run_reads_summary_and_plates(tmp_path: Path) -> None:
    results = runs.load_run(write_run(tmp_path, "run", ["HR26DK8337", "MH12AB1234"]))
    assert results.complete
    assert results.summary == {"frames": 10}
    assert [row["plate"] for row in results.plates] == ["HR26DK8337", "MH12AB1234"]


def test_run_stopped_part_way_is_incomplete(tmp_path: Path) -> None:
    results = runs.load_run(write_run(tmp_path, "run", ["HR26DK8337"], summary=False))
    assert not results.complete
    assert len(results.plates) == 1


# Uploads


def test_upload_name_cannot_escape_the_folder(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    saved = runs.save_upload("..\\..\\my clip (final).MP4", b"data", uploads)
    assert saved.parent == uploads
    assert saved.name == "my_clip_final.mp4"
    assert saved.read_bytes() == b"data"


def test_upload_never_overwrites(tmp_path: Path) -> None:
    first = runs.save_upload("clip.mp4", b"one", tmp_path)
    second = runs.save_upload("clip.mp4", b"two", tmp_path)
    assert (first.name, second.name) == ("clip.mp4", "clip_1.mp4")
    assert first.read_bytes() == b"one"


@pytest.mark.parametrize("name", ["notes.txt", "clip", "..mp4.exe"])
def test_upload_rejects_non_videos(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigError, match="Unsupported"):
        runs.save_upload(name, b"", tmp_path)


def test_upload_with_only_unsafe_characters_gets_a_name(tmp_path: Path) -> None:
    assert runs.save_upload("?*.mp4", b"", tmp_path).name == "upload.mp4"


# Progress


def test_frame_count_of_a_file(make_video) -> None:
    assert runs.frame_count(str(make_video(7))) == 7


@pytest.mark.parametrize("source", ["0", "rtsp://camera/stream", "missing.mp4"])
def test_frame_count_unknown_for_live_or_missing_sources(source: str) -> None:
    assert runs.frame_count(source) is None


def test_preview_is_shrunk_to_width_and_keeps_its_shape() -> None:
    frame = np.full((1080, 1920, 3), 120, np.uint8)
    data = runs.preview_jpeg(frame, 960)
    assert data[:2] == b"\xff\xd8", "should be a JPEG"
    decoded = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    assert decoded.shape == (540, 960, 3)


def test_small_preview_is_not_enlarged_and_frame_is_untouched() -> None:
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[10:20, 10:20] = (0, 0, 255)
    before = frame.copy()
    decoded = cv.imdecode(np.frombuffer(runs.preview_jpeg(frame, 960), np.uint8), cv.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert np.array_equal(frame, before), "the analysed frame must not change"
    assert decoded[15, 15, 2] > 200, "colours stay in OpenCV's BGR order"


def test_throttled_observer_reports_first_frame_then_waits() -> None:
    now = [0.0]
    updates: list[int] = []
    observer = runs.ThrottledObserver(
        lambda frame, count: updates.append(count), 0.5, lambda: now[0]
    )
    frame = np.zeros((2, 2, 3), np.uint8)
    for step in range(6):
        now[0] = step * 0.2
        assert observer.observe(frame)
    assert updates == [1, 4]
    assert observer.frames == 6


def test_extra_observers_see_every_frame(config, make_video) -> None:
    from conftest import ScriptedDetector, StubOcr
    from vrpr.pipeline import execute_run

    seen: list[int] = []
    observer = runs.ThrottledObserver(lambda frame, count: seen.append(count), interval_seconds=0)
    execute_run(
        config,
        str(make_video(4)),
        detector=ScriptedDetector([]),
        ocr=StubOcr(""),
        observers=[observer],
    )
    assert seen == [1, 2, 3, 4]


# The vrpr web command


def test_web_command_starts_streamlit_on_localhost_only(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    def fake_call(command: list[str], cwd: Path) -> int:
        calls.update(command=command, cwd=cwd)
        return 0

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "call", fake_call)
    assert cli.main(["web", "--port", "9000", "--root", str(tmp_path), "--no-browser"]) == 0

    command = calls["command"]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert Path(command[4]) == APP
    assert "--server.address=localhost" in command
    assert "--server.port=9000" in command
    assert "--server.headless=true" in command
    assert "--browser.gatherUsageStats=false" in command
    assert "--client.toolbarMode=minimal" in command
    assert calls["cwd"] == tmp_path


def test_web_command_explains_a_missing_streamlit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    assert cli.main(["web"]) == cli.EXIT_FAILURE
    assert ".[web]" in capsys.readouterr().err


def test_web_command_needs_an_existing_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    assert cli.main(["web", "--root", str(tmp_path / "missing")]) == cli.EXIT_USAGE


# The page, driven headlessly with Streamlit's test harness


class FakeYolo:
    def __init__(self, config) -> None:
        self.config = config

    def detect(self, frame) -> list[Detection]:
        return [Detection(BoundingBox(40, 100, 200, 140), 0.9)]


class FakeEasyOcr:
    def __init__(self, config) -> None:
        pass

    def read(self, image) -> list[OcrFragment]:
        return [OcrFragment("HR26DK8337", 0.95, BoundingBox(0, 0, 100, 30))]


def plates_shown(app_test) -> list[str]:
    """Plate numbers in the result cards, which are rendered as level-4 headings."""
    return [md.value[5:] for md in app_test.markdown if md.value.startswith("#### ")]


@pytest.fixture
def app_test(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st

    import vrpr.detector
    import vrpr.ocr

    monkeypatch.setattr(vrpr.detector, "YoloDetector", FakeYolo)
    monkeypatch.setattr(vrpr.ocr, "EasyOcrEngine", FakeEasyOcr)
    st.cache_resource.clear()
    return testing.AppTest.from_file(str(APP), default_timeout=60)


def test_page_loads_without_videos_or_runs(app_test) -> None:
    app_test.run()
    assert not app_test.exception
    assert "plate recognition" in app_test.title[0].value
    assert any("No videos found" in info.value for info in app_test.info)
    assert any("No runs yet" in info.value for info in app_test.info)


def test_page_runs_a_video_and_shows_the_plate(app_test, make_video, tmp_path: Path) -> None:
    videos = Path("data/videos")
    videos.mkdir(parents=True)
    make_video(8).replace(videos / "clip.mp4")
    Path("models").mkdir()
    Path("models/best.pt").write_bytes(b"fake")

    app_test.run()
    app_test.selectbox(key="video").select("clip.mp4")
    app_test.button(key="run").click().run()

    assert not app_test.exception, app_test.exception
    assert not app_test.error, [error.value for error in app_test.error]
    assert plates_shown(app_test) == ["HR26DK8337", "HR26DK8337"]  # Results, and newest in history
    assert str(app_test.metric[0].value) == "1"
    # The last live preview frame, then the plate image on each tab's card.
    assert len(app_test.image) == 3
    run_folders = list(Path("output").iterdir())
    assert (run_folders[0] / "summary.json").is_file()


def test_page_shows_missing_weights_as_a_message(app_test, make_video, monkeypatch) -> None:
    import vrpr.detector
    from vrpr.exceptions import ModelError

    videos = Path("data/videos")
    videos.mkdir(parents=True)
    make_video(3).replace(videos / "clip.mp4")

    def missing(config):
        raise ModelError("Plate detector weights not found at models/best.pt.")

    monkeypatch.setattr(vrpr.detector, "YoloDetector", missing)
    app_test.run()
    app_test.button(key="run").click().run()
    assert not app_test.exception
    assert any("weights not found" in error.value for error in app_test.error)


@pytest.mark.parametrize(
    ("value", "flagged"), [("False", True), ("false", True), ("True", False), ("", False)]
)
def test_is_unconfirmed(value: str, flagged: bool) -> None:
    assert runs.is_unconfirmed({"confirmed": value}) is flagged


def test_runs_from_before_the_confirmed_column_are_not_flagged() -> None:
    assert not runs.is_unconfirmed({"plate": "HR26DK8337"})


def test_page_marks_unconfirmed_plates(app_test) -> None:
    write_run(
        Path("output"),
        "20261108T100000Z-aaaaaa",
        ["MH12AB1234", "HR26DK8337"],
        unconfirmed=frozenset({"HR26DK8337"}),
    )
    app_test.run()
    assert not app_test.exception
    badges = [element for element in app_test.markdown if "Unconfirmed" in element.value]
    assert len(badges) == 1
    assert any("1 of these plates are unconfirmed" in caption.value for caption in app_test.caption)


def test_page_shows_past_runs(app_test) -> None:
    write_run(Path("output"), "20261108T100000Z-aaaaaa", ["MH12AB1234"])
    app_test.run()
    assert not app_test.exception
    assert app_test.selectbox(key="history_run").value == "20261108T100000Z-aaaaaa"
    assert plates_shown(app_test) == ["MH12AB1234"]


# The public demo


def test_public_demo_is_off_unless_switched_on(monkeypatch) -> None:
    monkeypatch.delenv(runs.PUBLIC_DEMO_VARIABLE, raising=False)
    assert not runs.public_demo()
    monkeypatch.setenv(runs.PUBLIC_DEMO_VARIABLE, "0")
    assert not runs.public_demo()
    monkeypatch.setenv(runs.PUBLIC_DEMO_VARIABLE, "1")
    assert runs.public_demo()


def test_old_session_folders_are_removed(tmp_path: Path) -> None:
    old = tmp_path / f"{runs.SESSION_PREFIX}old"
    new = tmp_path / f"{runs.SESSION_PREFIX}new"
    unrelated = tmp_path / "something-else"
    for folder in (old, new, unrelated):
        folder.mkdir()
    os.utime(old, (1000, 1000))
    os.utime(unrelated, (1000, 1000))
    assert runs.remove_old_sessions(tmp_path, max_age_seconds=3600, now=1000 + 7200) == 1
    assert not old.exists()
    assert new.exists() and unrelated.exists()


def test_frames_for_seconds(make_video) -> None:
    assert runs.frames_for_seconds(str(make_video(10)), 0.5) == 5  # the test videos are 10 fps


def test_frames_for_seconds_falls_back_for_unknown_rates() -> None:
    assert runs.frames_for_seconds("missing.mp4", 2) == 60


def test_frame_limit_stops_the_run_at_the_limit() -> None:
    limit = runs.FrameLimit(3)
    frame = np.zeros((2, 2, 3), np.uint8)
    assert [limit.observe(frame) for _ in range(3)] == [True, True, False]
    assert limit.reached


def test_progress_text_starts_with_the_frame_only() -> None:
    assert runs.progress_text(1, 240, 0.0) == "Frame 1 of 240"
    assert runs.progress_text(1, None, 0.0) == "Frame 1"


def test_progress_text_shows_speed_and_time_left() -> None:
    # 20 frames after the first in 10 seconds is 2 a second, and 219 frames are left.
    assert runs.progress_text(21, 240, 10.0) == (
        "Frame 21 of 240 · 2.0 frames a second · about 1 min 50 s left"
    )
    assert runs.progress_text(231, 240, 23.0) == (
        "Frame 231 of 240 · 10.0 frames a second · about 1 s left"
    )


def test_progress_text_past_an_estimated_total_shows_only_speed() -> None:
    assert runs.progress_text(251, 240, 50.0) == "Frame 251 · 5.0 frames a second"
    assert runs.progress_text(11, None, 5.0) == "Frame 11 · 2.0 frames a second"


@pytest.fixture
def public_page(app_test, monkeypatch, tmp_path: Path, make_video):
    """The page as a public host serves it, with a sample video and a private temp folder."""
    monkeypatch.setenv(runs.PUBLIC_DEMO_VARIABLE, "1")
    temp = tmp_path / "server-temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    Path("samples").mkdir()
    make_video(8).replace(Path("samples") / "sample.mp4")
    Path("models").mkdir()
    Path("models/best.pt").write_bytes(b"fake")
    return app_test


PUBLIC_SOURCE_OPTIONS = [
    "Sample video",
    "Upload a video",
    "Video in data/videos",
    "Camera or stream",
]


def test_public_page_shows_local_controls_greyed_out(public_page) -> None:
    public_page.run()
    assert not public_page.exception
    assert public_page.radio(key="source_mode").options == PUBLIC_SOURCE_OPTIONS
    for widget in (
        public_page.text_input(key="weights"),
        public_page.selectbox(key="device"),
        public_page.checkbox(key="ocr_gpu"),
        public_page.checkbox(key="save_video"),
    ):
        assert widget.disabled, widget.label
        assert widget.help and "Clone VRPR" in widget.help, widget.label
    assert public_page.selectbox(key="device").value == "CPU"
    assert not public_page.slider(key="confidence").disabled, "demo settings stay usable"
    assert any("personal data" in caption.value for caption in public_page.caption)


@pytest.mark.parametrize("source", ["Video in data/videos", "Camera or stream"])
def test_public_local_sources_are_shown_but_cannot_run(public_page, source: str) -> None:
    public_page.run()
    public_page.radio(key="source_mode").set_value(source).run()
    assert not public_page.exception
    locked = [*public_page.selectbox, *public_page.text_input]
    assert any(widget.disabled and "Clone VRPR" in (widget.help or "") for widget in locked)
    assert public_page.button(key="run").disabled


def test_public_page_ignores_local_settings_sent_anyway(public_page) -> None:
    # Disabled widgets stop people in the browser, not a client that sends values directly.
    public_page.run()
    public_page.session_state["weights"] = "C:/secret/other.pt"
    public_page.session_state["save_video"] = True
    public_page.session_state["device"] = "GPU"
    public_page.button(key="run").click().run()
    assert not public_page.exception
    assert not public_page.error, [error.value for error in public_page.error]
    assert plates_shown(public_page)


def test_public_page_shows_the_demo_notice_once(public_page) -> None:
    public_page.run()
    assert not public_page.exception
    notice = [md.value for md in public_page.markdown if "This page is a demo" in md.value]
    assert notice, "the pop-up opens on the first visit"
    assert any("git clone" in code.value for code in public_page.code)
    public_page.button(key="close_notice").click().run()
    assert not any("This page is a demo" in md.value for md in public_page.markdown)
    public_page.run()
    assert not any("This page is a demo" in md.value for md in public_page.markdown)


def test_public_runs_are_private_to_each_session(public_page) -> None:
    public_page.run()
    public_page.button(key="run").click().run()
    assert plates_shown(public_page), "the first visitor sees their plates"
    assert not any("Run folder" in caption.value for caption in public_page.caption)

    from streamlit.testing.v1 import AppTest

    other_visitor = AppTest.from_file(str(APP), default_timeout=60)
    other_visitor.run()
    assert any("No runs yet" in info.value for info in other_visitor.info)
    assert plates_shown(other_visitor) == []


def test_public_demo_analyses_only_the_first_seconds(public_page, monkeypatch) -> None:
    monkeypatch.setattr(runs, "frames_for_seconds", lambda source, seconds: 3)
    public_page.run()
    public_page.button(key="run").click().run()
    assert not public_page.exception
    assert str(public_page.metric[1].value) == "3", "frames analysed"
    assert any("analyses the first" in info.value for info in public_page.info)


def test_entry_point_serves_the_public_page(app_test, monkeypatch, tmp_path: Path) -> None:
    # The entry point sets the switch itself; registering it here restores it afterwards.
    monkeypatch.setenv(runs.PUBLIC_DEMO_VARIABLE, "0")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(ENTRY_POINT), default_timeout=60)
    page.run()
    assert not page.exception
    assert page.radio(key="source_mode").options == PUBLIC_SOURCE_OPTIONS
