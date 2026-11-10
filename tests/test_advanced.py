"""Advanced settings: OpenCV filters, their configuration, and trying them on one frame."""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
import yaml

from conftest import StubOcr
from vrpr import tuning
from vrpr.config import (
    AppConfig,
    DetectorConfig,
    OcrConfig,
    PreprocessConfig,
    build_config,
    dump_config,
    load_config,
)
from vrpr.exceptions import ConfigError, OcrError
from vrpr.imaging import apply_filters, prepare_for_ocr
from vrpr.models import BoundingBox, Detection

REPOSITORY = Path(__file__).resolve().parent.parent


def plate_image() -> np.ndarray:
    """A small colour crop: dark text on a light, unevenly lit plate."""
    image = np.full((30, 90, 3), 180, np.uint8)
    image[:, :45] = 120  # a shadow over the left half
    cv.putText(image, "HR26", (5, 22), cv.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return image


# Filters


def test_default_preparation_is_unchanged() -> None:
    crop = plate_image()
    expected = cv.resize(
        cv.cvtColor(crop, cv.COLOR_BGR2GRAY),
        None,
        fx=128 / 30,
        fy=128 / 30,
        interpolation=cv.INTER_CUBIC,
    )
    assert np.array_equal(prepare_for_ocr(crop, 128), expected)
    assert np.array_equal(prepare_for_ocr(crop, 128, PreprocessConfig()), expected)


def test_colour_is_kept_when_grayscale_is_off() -> None:
    prepared = prepare_for_ocr(plate_image(), 64, PreprocessConfig(grayscale=False))
    assert prepared.shape == (64, 192, 3)


def test_thresholding_implies_grayscale_and_gives_black_and_white() -> None:
    for method in ("otsu", "adaptive"):
        settings = PreprocessConfig(grayscale=False, threshold=method)
        prepared = prepare_for_ocr(plate_image(), 64, settings)
        assert prepared.ndim == 2
        assert set(np.unique(prepared)) <= {0, 255}


@pytest.mark.parametrize(
    "settings",
    [
        PreprocessConfig(denoise="bilateral"),
        PreprocessConfig(denoise="median"),
        PreprocessConfig(gamma=2.0),
        PreprocessConfig(contrast="clahe", clahe_clip_limit=4.0),
        PreprocessConfig(contrast="equalize"),
        PreprocessConfig(sharpen=1.5),
        PreprocessConfig(threshold="adaptive", adaptive_block_size=11),
        PreprocessConfig(invert=True),
        PreprocessConfig(grayscale=False, contrast="clahe", sharpen=1.0, denoise="bilateral"),
    ],
)
def test_each_filter_keeps_the_image_shape_and_type(settings: PreprocessConfig) -> None:
    before = prepare_for_ocr(plate_image(), 64)
    if not settings.grayscale:
        before = prepare_for_ocr(plate_image(), 64, PreprocessConfig(grayscale=False))
    after = prepare_for_ocr(plate_image(), 64, settings)
    assert after.shape == before.shape
    assert after.dtype == np.uint8
    assert not np.array_equal(after, before)


def test_gamma_above_one_brightens() -> None:
    gray = cv.cvtColor(plate_image(), cv.COLOR_BGR2GRAY)
    assert apply_filters(gray, PreprocessConfig(gamma=2.0)).mean() > gray.mean()
    assert apply_filters(gray, PreprocessConfig(gamma=0.5)).mean() < gray.mean()


def test_invert_swaps_light_and_dark() -> None:
    gray = cv.cvtColor(plate_image(), cv.COLOR_BGR2GRAY)
    assert np.array_equal(apply_filters(gray, PreprocessConfig(invert=True)), 255 - gray)


# Settings


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("preprocess", {"denoise": "blur"}, "preprocess.denoise"),
        ("preprocess", {"threshold": "binary"}, "preprocess.threshold"),
        ("preprocess", {"gamma": 9}, "preprocess.gamma"),
        ("preprocess", {"adaptive_block_size": 30}, "odd"),
        ("ocr", {"decoder": "wordbeamsearch"}, "ocr.decoder"),
        ("ocr", {"rotation_info": [45]}, "rotation_info"),
        ("ocr", {"text_threshold": 1.5}, "ocr.text_threshold"),
        ("detector", {"iou": 2}, "detector.iou"),
    ],
)
def test_invalid_settings_are_explained(section: str, values: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_config({section: values})


def test_settings_round_trip_through_yaml() -> None:
    config = AppConfig(
        detector=DetectorConfig(weights=Path("models/other.pt"), iou=0.5, augment=True),
        ocr=OcrConfig(decoder="beamsearch", beam_width=9, rotation_info=(90, 270)),
        preprocess=PreprocessConfig(contrast="clahe", threshold="otsu", invert=True),
    )
    assert build_config(yaml.safe_load(dump_config(config))) == config


def test_the_shipped_defaults_match_the_code() -> None:
    # The web page's reset and the CLI both start from configs/default.yaml.
    shipped = load_config(REPOSITORY / "configs" / "default.yaml")
    assert shipped.preprocess == PreprocessConfig()
    assert (shipped.detector.iou, shipped.detector.augment) == (0.7, False)
    assert shipped.ocr.decoder == "greedy"


# Trying settings on one frame


class TwoPlates:
    def detect(self, frame):
        return [
            Detection(BoundingBox(10, 10, 60, 30), 0.5),
            Detection(BoundingBox(80, 40, 140, 60), 0.9),
        ]


def test_try_frame_reads_each_plate_strongest_first() -> None:
    frame = np.full((120, 160, 3), 200, np.uint8)
    ocr = StubOcr("HR26DK8337", "")
    trials = tuning.try_frame(frame, TwoPlates(), ocr, AppConfig())
    assert [trial.number for trial in trials] == [1, 2]
    assert [trial.detection.confidence for trial in trials] == [0.9, 0.5]
    assert trials[0].reading is not None and trials[0].reading.plate == "HR26DK8337"
    assert trials[1].reading is None and trials[1].fragments == []
    # The reader is given the filtered image, and the trial keeps both versions.
    assert trials[0].prepared.shape[0] == 128 and trials[0].crop.ndim == 3
    assert np.array_equal(ocr.images[0], trials[0].prepared)


def test_try_frame_applies_the_filters() -> None:
    frame = np.full((120, 160, 3), 200, np.uint8)
    config = AppConfig(preprocess=PreprocessConfig(invert=True))
    trial = tuning.try_frame(frame, TwoPlates(), StubOcr(""), config)[0]
    assert trial.prepared.max() == 55  # 255 - 200


def test_try_frame_reports_ocr_failures_per_plate() -> None:
    class Failing:
        def read(self, image):
            raise OcrError("EasyOCR failed: bad image")

    frame = np.full((120, 160, 3), 200, np.uint8)
    trials = tuning.try_frame(frame, TwoPlates(), Failing(), AppConfig())
    assert [trial.error for trial in trials] == ["EasyOCR failed: bad image"] * 2


def test_try_frame_limits_the_plates_read() -> None:
    frame = np.full((120, 160, 3), 200, np.uint8)
    assert len(tuning.try_frame(frame, TwoPlates(), StubOcr(""), AppConfig(), max_plates=1)) == 1


def test_draw_trials_boxes_a_copy() -> None:
    frame = np.full((120, 160, 3), 200, np.uint8)
    trials = tuning.try_frame(frame, TwoPlates(), StubOcr(""), AppConfig())
    drawn = tuning.draw_trials(frame, trials)
    assert drawn.shape == frame.shape
    assert not np.array_equal(drawn, frame)
    assert (frame == 200).all()


def test_read_frame(make_video) -> None:
    video = str(make_video(5))
    assert tuning.read_frame(video, 2).shape == (240, 320, 3)
    assert tuning.read_frame(video, 50) is None
    assert tuning.read_frame("missing.mp4", 0) is None
