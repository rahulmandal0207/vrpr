"""Typed, validated application configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from vrpr.exceptions import ConfigError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _require_fraction(name: str, value: float) -> None:
    _require(0.0 <= value <= 1.0, f"{name} must be between 0 and 1, got {value}")


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    _require(value in choices, f"{name} must be one of {', '.join(choices)}, got {value!r}")


DENOISE_METHODS = ("none", "bilateral", "median")
CONTRAST_METHODS = ("none", "clahe", "equalize")
THRESHOLD_METHODS = ("none", "otsu", "adaptive")
OCR_DECODERS = ("greedy", "beamsearch")
OCR_ROTATIONS = (90, 180, 270)


@dataclass(frozen=True)
class DetectorConfig:
    weights: Path = Path("models/best.pt")
    confidence: float = 0.4
    image_size: int = 640
    device: str | None = None
    iou: float = 0.7  # overlap above which YOLO keeps only the stronger of two boxes
    augment: bool = False  # test-time augmentation: flipped and scaled copies, slower

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", Path(self.weights))
        _require_fraction("detector.confidence", self.confidence)
        _require(self.image_size > 0, "detector.image_size must be positive")
        _require_fraction("detector.iou", self.iou)


@dataclass(frozen=True)
class OcrConfig:
    languages: tuple[str, ...] = ("en",)
    gpu: bool = True
    min_confidence: float = 0.2
    every_n_frames: int = 1
    crop_padding: float = 0.05
    min_crop_height: int = 128
    # Passed to EasyOCR's readtext. The defaults are EasyOCR's own.
    decoder: str = "greedy"
    beam_width: int = 5
    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5
    mag_ratio: float = 1.0
    min_size: int = 20
    rotation_info: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "languages", tuple(self.languages))
        object.__setattr__(self, "rotation_info", tuple(self.rotation_info))
        _require_choice("ocr.decoder", self.decoder, OCR_DECODERS)
        _require(self.beam_width >= 1, "ocr.beam_width must be at least 1")
        for name in ("text_threshold", "low_text", "link_threshold", "contrast_ths"):
            _require_fraction(f"ocr.{name}", getattr(self, name))
        _require_fraction("ocr.adjust_contrast", self.adjust_contrast)
        _require(self.mag_ratio > 0, "ocr.mag_ratio must be positive")
        _require(self.min_size >= 1, "ocr.min_size must be at least 1")
        _require(
            all(angle in OCR_ROTATIONS for angle in self.rotation_info),
            f"ocr.rotation_info may only contain {OCR_ROTATIONS}",
        )
        _require(bool(self.languages), "ocr.languages must not be empty")
        _require_fraction("ocr.min_confidence", self.min_confidence)
        _require(self.every_n_frames >= 1, "ocr.every_n_frames must be at least 1")
        _require(self.crop_padding >= 0, "ocr.crop_padding must not be negative")
        _require(self.min_crop_height >= 1, "ocr.min_crop_height must be at least 1")


@dataclass(frozen=True)
class PreprocessConfig:
    """OpenCV filters applied to each plate crop, in this order, before OCR.

    The defaults leave the crop as it is, apart from the grayscale conversion and upscaling in
    ``ocr``. Thresholding needs a single channel, so it implies grayscale.
    """

    grayscale: bool = True
    denoise: str = "none"
    gamma: float = 1.0  # above 1 brightens dark plates, below 1 darkens washed-out ones
    contrast: str = "none"
    clahe_clip_limit: float = 2.0
    sharpen: float = 0.0  # unsharp mask strength; 0 is off
    threshold: str = "none"
    adaptive_block_size: int = 31
    invert: bool = False  # for light text on a dark plate

    def __post_init__(self) -> None:
        _require_choice("preprocess.denoise", self.denoise, DENOISE_METHODS)
        _require_choice("preprocess.contrast", self.contrast, CONTRAST_METHODS)
        _require_choice("preprocess.threshold", self.threshold, THRESHOLD_METHODS)
        _require(0.1 <= self.gamma <= 5.0, "preprocess.gamma must be between 0.1 and 5")
        _require(self.clahe_clip_limit > 0, "preprocess.clahe_clip_limit must be positive")
        _require(0.0 <= self.sharpen <= 5.0, "preprocess.sharpen must be between 0 and 5")
        _require(
            self.adaptive_block_size >= 3 and self.adaptive_block_size % 2 == 1,
            "preprocess.adaptive_block_size must be an odd number of at least 3",
        )

    @property
    def is_default(self) -> bool:
        return self == PreprocessConfig()


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int = 5
    min_hits: int = 2
    iou_threshold: float = 0.2

    def __post_init__(self) -> None:
        _require(self.max_age >= 0, "tracker.max_age must not be negative")
        _require(self.min_hits >= 1, "tracker.min_hits must be at least 1")
        _require_fraction("tracker.iou_threshold", self.iou_threshold)


@dataclass(frozen=True)
class ConsensusConfig:
    min_agreement: int = 3
    min_votes_on_close: int = 1
    dedupe_window_seconds: float = 30.0
    recheck_every_n_frames: int = 16
    report_delay_seconds: float = 3.0

    def __post_init__(self) -> None:
        _require(self.min_agreement >= 1, "consensus.min_agreement must be at least 1")
        _require(self.min_votes_on_close >= 1, "consensus.min_votes_on_close must be at least 1")
        _require(
            self.recheck_every_n_frames >= 1, "consensus.recheck_every_n_frames must be at least 1"
        )
        _require(
            self.report_delay_seconds >= 0, "consensus.report_delay_seconds must not be negative"
        )
        _require(
            self.dedupe_window_seconds >= 0, "consensus.dedupe_window_seconds must not be negative"
        )


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("output")
    save_crops: bool = True
    save_annotated_video: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))


@dataclass(frozen=True)
class SourceConfig:
    max_read_failures: int = 30
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        _require(self.max_read_failures >= 0, "source.max_read_failures must not be negative")
        _require(self.retry_delay_seconds >= 0, "source.retry_delay_seconds must not be negative")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", self.level.upper())
        allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        _require(self.level in allowed, f"logging.level must be one of {allowed}")
        _require(self.format in ("text", "json"), "logging.format must be 'text' or 'json'")


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_SECTIONS: dict[str, Any] = {
    "detector": DetectorConfig,
    "ocr": OcrConfig,
    "preprocess": PreprocessConfig,
    "tracker": TrackerConfig,
    "consensus": ConsensusConfig,
    "output": OutputConfig,
    "source": SourceConfig,
    "logging": LoggingConfig,
}


def _build_section(name: str, values: Any) -> Any:
    section = _SECTIONS[name]
    _require(isinstance(values, dict), f"{name} must be a mapping")
    known = {item.name for item in fields(section)}
    unknown = sorted(set(values) - known)
    _require(not unknown, f"Unknown {name} settings: {', '.join(unknown)}")
    try:
        return section(**values)
    except (TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid value in {name}: {exc}") from exc


def build_config(data: dict[str, Any]) -> AppConfig:
    """Build a validated configuration from a nested mapping."""
    unknown = sorted(set(data) - set(_SECTIONS))
    _require(not unknown, f"Unknown configuration sections: {', '.join(unknown)}")
    return AppConfig(**{name: _build_section(name, values) for name, values in data.items()})


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, or return the defaults."""
    if path is None:
        return AppConfig()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return AppConfig()
    _require(isinstance(data, dict), f"{path} must contain a mapping")
    return build_config(data)


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Return the configuration as plain values, the shape ``build_config`` reads back."""
    return {name: _plain(asdict(getattr(config, name))) for name in _SECTIONS}


def dump_config(config: AppConfig) -> str:
    """Return the configuration as YAML, for use with ``vrpr run --config``."""
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)
