"""Configuration, sinks, evaluation, video sources, adapters and the CLI."""

import csv
import json
import logging
import signal
import sys
import types
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import yaml

from vrpr import cli, video
from vrpr.config import (
    AppConfig,
    DetectorConfig,
    LoggingConfig,
    OcrConfig,
    SourceConfig,
    build_config,
    load_config,
)
from vrpr.detector import YoloDetector
from vrpr.evaluate import evaluate, read_plates
from vrpr.exceptions import (
    ConfigError,
    DependencyError,
    DisplayError,
    ModelError,
    OcrError,
    SourceError,
)
from vrpr.logging_setup import JsonFormatter
from vrpr.models import BoundingBox, PlateRecord
from vrpr.ocr import EasyOcrEngine
from vrpr.pipeline import RunSummary
from vrpr.plates import edit_distance
from vrpr.rendering import WindowPreview
from vrpr.sinks import CropStore, CsvSink, JsonlSink
from vrpr.training import TrainingOptions, download_dataset, train_model, validate_model
from vrpr.video import VideoSource, is_live_source, redact_source

REPO_ROOT = Path(__file__).resolve().parent.parent


def record(plate: str = "HR26DK8337") -> PlateRecord:
    return PlateRecord("run", 1, plate, "standard", 0.9, 3, 4, 0, 9, 0.5)


# Configuration


def test_shipped_defaults_match_the_code() -> None:
    assert load_config(REPO_ROOT / "configs" / "default.yaml") == AppConfig()


def test_partial_config_keeps_other_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tracker:\n  max_age: 40\n", encoding="utf-8")
    config = load_config(path)
    assert config.tracker.max_age == 40
    assert config.tracker.min_hits == AppConfig().tracker.min_hits


def test_empty_config_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == AppConfig()


@pytest.mark.parametrize(
    "data",
    [
        {"unknown": {}},
        {"tracker": {"typo": 1}},
        {"tracker": []},
        {"detector": {"confidence": 2}},
        {"ocr": {"every_n_frames": 0}},
        {"ocr": {"languages": []}},
        {"logging": {"level": "LOUD"}},
        {"logging": {"format": "xml"}},
    ],
)
def test_bad_settings_are_rejected(data: dict) -> None:
    with pytest.raises(ConfigError):
        build_config(data)


def test_missing_config_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("tracker: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


# Sinks


def test_csv_sink_writes_a_single_header_across_reopens(tmp_path: Path) -> None:
    path = tmp_path / "plates.csv"
    for plate in ("HR26DK8337", "MH12AB1234"):
        sink = CsvSink(path)
        sink.write(record(plate))
        sink.close()
    with path.open(newline="", encoding="utf-8") as file:
        assert [row["plate"] for row in csv.DictReader(file)] == ["HR26DK8337", "MH12AB1234"]


def test_sink_data_is_on_disk_before_close(tmp_path: Path) -> None:
    csv_path, jsonl_path = tmp_path / "plates.csv", tmp_path / "plates.jsonl"
    sinks = [CsvSink(csv_path), JsonlSink(jsonl_path)]
    for sink in sinks:
        sink.write(record())
    assert "HR26DK8337" in csv_path.read_text(encoding="utf-8")
    assert "HR26DK8337" in jsonl_path.read_text(encoding="utf-8")
    for sink in sinks:
        sink.close()


def test_crop_store_reports_failed_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cv, "imwrite", lambda *_: False)
    crop = np.zeros((10, 10, 3), np.uint8)
    assert CropStore(tmp_path).save(1, "HR26DK8337", crop) is None


# Evaluation


def test_edit_distance() -> None:
    assert edit_distance("HR26DK8337", "HR26DK8337") == 0
    assert edit_distance("HR26DK8337", "HR26DK8331") == 1
    assert edit_distance("", "ABC") == 3


def test_evaluate_perfect_run() -> None:
    result = evaluate(["HR26DK8337"], ["HR26DK8337"])
    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
    assert result.character_error_rate == 0.0


def test_evaluate_scores_a_near_miss_by_characters() -> None:
    result = evaluate(["HR26DK8337"], ["HR26DK8331"])
    assert result.correct == 0
    assert result.character_error_rate == 0.1


def test_evaluate_counts_repeats() -> None:
    result = evaluate(["HR26DK8337"], ["HR26DK8337", "HR26DK8337"])
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_evaluate_an_empty_run_is_a_total_miss() -> None:
    result = evaluate(["HR26DK8337"], [])
    assert result.f1 == 0.0
    assert result.character_error_rate == 1.0


def test_read_plates_needs_a_plate_column(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("number\nHR26DK8337\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="plate"):
        read_plates(path)


# Video sources


@pytest.mark.parametrize(
    ("source", "live"),
    [("0", True), ("rtsp://camera/stream", True), ("clip.mp4", False), ("C:/v/clip.mp4", False)],
)
def test_is_live_source(source: str, live: bool) -> None:
    assert is_live_source(source) is live


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("rtsp://admin:secret@camera/stream", "rtsp://***@camera/stream"),
        ("rtsp://camera/stream", "rtsp://camera/stream"),
        ("clip.mp4", "clip.mp4"),
    ],
)
def test_stream_credentials_are_redacted(source: str, expected: str) -> None:
    assert redact_source(source) == expected


def test_stream_credentials_stay_out_of_errors_and_logs(monkeypatch, caplog) -> None:
    class Closed(FakeCapture):
        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(video.cv, "VideoCapture", Closed)
    with (
        pytest.raises(SourceError) as error,
        VideoSource("rtsp://admin:secret@camera", SourceConfig()),
    ):
        pass
    assert "secret" not in str(error.value)
    assert "secret" not in caplog.text


def test_video_file_yields_every_frame(make_video) -> None:
    with VideoSource(str(make_video(5)), SourceConfig()) as source:
        packets = list(source.frames())
    assert [packet.index for packet in packets] == [0, 1, 2, 3, 4]
    assert packets[1].timestamp_s == pytest.approx(0.1)


def test_frames_needs_an_open_source() -> None:
    with pytest.raises(SourceError):
        list(VideoSource("clip.mp4", SourceConfig()).frames())


def test_corrupt_file_is_a_source_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(SourceError, match="Could not open"), VideoSource(str(path), SourceConfig()):
        pass


class FakeCapture:
    """A stream that drops after one frame, recovers once, then stays down."""

    opened = 0

    def __init__(self, _source: object) -> None:
        type(self).opened += 1
        self.generation = type(self).opened
        self.reads = 0

    def isOpened(self) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def release(self) -> None:
        pass

    def read(self):
        self.reads += 1
        if self.generation <= 2 and self.reads == 1:
            return True, np.zeros((4, 4, 3), np.uint8)
        return False, None


def test_live_source_reconnects_then_gives_up(monkeypatch) -> None:
    FakeCapture.opened = 0
    monkeypatch.setattr(video.cv, "VideoCapture", FakeCapture)
    config = SourceConfig(max_read_failures=2, retry_delay_seconds=0)
    with VideoSource("rtsp://camera", config) as source:
        frames = list(source.frames())
    assert len(frames) == 2
    assert FakeCapture.opened > 1


# Model adapters, against fakes of the libraries' interfaces


class FakeTensor:
    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


def install_fake_yolo(monkeypatch) -> types.SimpleNamespace:
    calls = types.SimpleNamespace(trained={}, validated={}, loaded=None)
    save_dir = {}

    class FakeYolo:
        def __init__(self, weights: str) -> None:
            calls.loaded = weights

        def predict(self, **kwargs):
            calls.predict = kwargs
            boxes = [types.SimpleNamespace(xyxy=[FakeTensor([1.0, 2.0, 3.0, 4.0])], conf=[0.75])]
            return [types.SimpleNamespace(boxes=boxes)]

        def train(self, **kwargs):
            calls.trained = kwargs
            return types.SimpleNamespace(save_dir=save_dir["path"])

        def val(self, **kwargs):
            calls.validated = kwargs
            box = types.SimpleNamespace(map50=0.9, map=0.7, mp=0.8, mr=0.85)
            return types.SimpleNamespace(box=box)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    calls.save_dir = save_dir
    return calls


def test_yolo_detector_converts_results(tmp_path: Path, monkeypatch) -> None:
    calls = install_fake_yolo(monkeypatch)
    weights = tmp_path / "best.pt"
    weights.touch()
    detector = YoloDetector(DetectorConfig(weights=weights, confidence=0.5))
    detection = detector.detect(np.zeros((4, 4, 3), np.uint8))[0]
    assert detection.box == BoundingBox(1, 2, 3, 4)
    assert detection.confidence == 0.75
    assert calls.predict["conf"] == 0.5
    assert calls.predict["verbose"] is False


def test_missing_weights_explain_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="vrpr train"):
        YoloDetector(DetectorConfig(weights=tmp_path / "missing.pt"))


def test_missing_ml_stack_explains_what_to_do(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    weights = tmp_path / "best.pt"
    weights.touch()
    with pytest.raises(DependencyError, match=r"\.\[ml\]"):
        YoloDetector(DetectorConfig(weights=weights))


def test_easyocr_engine_converts_results(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, languages: list, gpu: bool) -> None:
            self.gpu = gpu

        def readtext(self, image, allowlist: str):
            return [([[0, 1], [10, 1], [10, 5], [0, 5]], "HR26", 0.8)]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FakeReader))
    fragment = EasyOcrEngine(OcrConfig(gpu=False)).read(np.zeros((4, 4), np.uint8))[0]
    assert fragment.text == "HR26"
    assert fragment.box == BoundingBox(0, 1, 10, 5)


def test_easyocr_failures_become_ocr_errors(monkeypatch) -> None:
    class FailingReader:
        def __init__(self, languages: list, gpu: bool) -> None:
            pass

        def readtext(self, image, allowlist: str):
            raise ValueError("bad image")

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FailingReader))
    with pytest.raises(OcrError):
        EasyOcrEngine(OcrConfig()).read(np.zeros((4, 4), np.uint8))


# Training


def test_train_installs_the_best_weights(tmp_path: Path, monkeypatch) -> None:
    calls = install_fake_yolo(monkeypatch)
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({}), encoding="utf-8")
    best = tmp_path / "run" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_text("weights", encoding="utf-8")
    calls.save_dir["path"] = str(tmp_path / "run")

    output = tmp_path / "models" / "best.pt"
    assert train_model(TrainingOptions(data=data, output_weights=output, seed=7)) == output
    assert output.read_text(encoding="utf-8") == "weights"
    assert calls.trained["seed"] == 7


def test_train_needs_a_dataset(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Dataset"):
        train_model(TrainingOptions(data=tmp_path / "missing.yaml"))


def test_validate_returns_metrics(tmp_path: Path, monkeypatch) -> None:
    install_fake_yolo(monkeypatch)
    weights, data = tmp_path / "best.pt", tmp_path / "data.yaml"
    weights.touch()
    data.touch()
    assert validate_model(weights, data)["map50"] == 0.9


def test_download_needs_an_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ROBOFLOW_API_KEY"):
        download_dataset("workspace", "project", 1, tmp_path)


# Logging and the command line


def test_json_formatter_writes_one_object() -> None:
    entry = logging.LogRecord("vrpr", logging.INFO, "", 0, "hello %s", ("there",), None)
    assert json.loads(JsonFormatter().format(entry))["message"] == "hello there"


def test_logging_level_is_normalized() -> None:
    assert LoggingConfig(level="debug").level == "DEBUG"


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "vrpr" in capsys.readouterr().out


def test_cli_parse(capsys) -> None:
    assert cli.main(["parse", "HR 26 DK 8337"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["plate"] == "HR26DK8337"


def test_cli_parse_no_match() -> None:
    assert cli.main(["parse", "hello"]) == cli.EXIT_NO_MATCH


def test_cli_evaluate(tmp_path: Path, capsys) -> None:
    expected = tmp_path / "expected.csv"
    expected.write_text("plate\nHR26DK8337\n", encoding="utf-8")
    assert cli.main(["evaluate", "--expected", str(expected), "--predicted", str(expected)]) == 0
    assert json.loads(capsys.readouterr().out)["f1"] == 1.0


def test_cli_reports_config_errors(tmp_path: Path, capsys) -> None:
    status = cli.main(["run", "clip.mp4", "--config", str(tmp_path / "missing.yaml")])
    assert status == cli.EXIT_USAGE
    assert "Configuration error" in capsys.readouterr().err


def test_cli_reports_missing_weights_without_a_traceback(tmp_path: Path, capsys) -> None:
    status = cli.main(["run", "clip.mp4", "--weights", str(tmp_path / "missing.pt")])
    assert status == cli.EXIT_FAILURE
    assert "vrpr train" in capsys.readouterr().err


def test_cli_run_applies_flags_and_restores_signal_handlers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen = {}

    def fake_run(config: AppConfig, source: str, display: bool, stop: object) -> RunSummary:
        seen.update(config=config, source=source, display=display)
        return RunSummary("run", source, 1, 1.0, 1.0, 0, 0, 0, 0, 0, 0, False)

    monkeypatch.setattr(cli, "execute_run", fake_run)
    handler_before = signal.getsignal(signal.SIGINT)
    output = tmp_path / "results"
    status = cli.main(
        [
            "run",
            "clip.mp4",
            "--weights",
            "custom.pt",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--save-video",
            "--no-crops",
            "--log-format",
            "json",
            "--display",
        ]
    )

    assert status == cli.EXIT_OK
    config = seen["config"]
    assert config.detector.weights == Path("custom.pt")
    assert config.detector.device == "cpu"
    assert config.output.directory == output
    assert config.output.save_annotated_video
    assert not config.output.save_crops
    assert config.logging.format == "json"
    assert seen["display"] is True
    assert signal.getsignal(signal.SIGINT) is handler_before
    assert json.loads(capsys.readouterr().out)["run_id"] == "run"


def test_cli_train_and_validate_pass_options_through(tmp_path: Path, monkeypatch, capsys) -> None:
    seen = {}

    def fake_train(options: TrainingOptions) -> Path:
        seen["options"] = options
        return options.output_weights

    monkeypatch.setattr(cli, "train_model", fake_train)
    monkeypatch.setattr(cli, "validate_model", lambda *_: {"map50": 0.9})
    data = tmp_path / "data.yaml"

    assert cli.main(["train", "--data", str(data), "--epochs", "5"]) == cli.EXIT_OK
    assert seen["options"].epochs == 5
    assert cli.main(["validate", "--data", str(data)]) == cli.EXIT_OK
    assert '"map50": 0.9' in capsys.readouterr().out


def test_preview_explains_a_headless_opencv_build(monkeypatch) -> None:
    def no_gui(*_: object) -> None:
        raise cv.error("not implemented")

    monkeypatch.setattr(cv, "imshow", no_gui)
    with pytest.raises(DisplayError, match="save-video"):
        WindowPreview().observe(np.zeros((4, 4, 3), np.uint8))
