"""Train the plate detector, measure it and fetch its dataset."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from vrpr.detector import load_yolo
from vrpr.exceptions import ConfigError, DependencyError, ModelError

logger = logging.getLogger(__name__)

API_KEY_VARIABLE = "ROBOFLOW_API_KEY"


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    data: Path
    base_weights: str = "yolov8n.pt"
    epochs: int = 100
    image_size: int = 640
    batch: int = 16
    device: str | None = None
    seed: int = 0
    project: Path = Path("runs/detect")
    output_weights: Path = Path("models/best.pt")


def train_model(options: TrainingOptions) -> Path:
    """Train from pretrained weights and install the best result."""
    if not options.data.is_file():
        raise ConfigError(f"Dataset file not found: {options.data}")

    results = load_yolo(options.base_weights).train(
        data=str(options.data.resolve()),
        epochs=options.epochs,
        imgsz=options.image_size,
        batch=options.batch,
        device=options.device,
        seed=options.seed,
        deterministic=True,
        project=str(options.project.resolve()),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise ModelError(f"Training finished without producing {best}")
    options.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, options.output_weights)
    logger.info("Best weights copied to %s", options.output_weights)
    return options.output_weights


def validate_model(weights: Path, data: Path, device: str | None = None) -> dict[str, float]:
    """Return detection metrics for trained weights on a dataset."""
    if not weights.is_file():
        raise ModelError(f"Weights not found: {weights}")
    if not data.is_file():
        raise ConfigError(f"Dataset file not found: {data}")
    metrics = load_yolo(weights).val(data=str(data.resolve()), device=device, verbose=False)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def download_dataset(workspace: str, project: str, version: int, destination: Path) -> Path:
    """Download a YOLO-format dataset from Roboflow."""
    api_key = os.environ.get(API_KEY_VARIABLE)
    if not api_key:
        raise ConfigError(f"Set {API_KEY_VARIABLE} to download a dataset")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise DependencyError(
            "Downloading a dataset needs Roboflow. Install it with: pip install -e .[dataset]"
        ) from exc

    dataset = (
        Roboflow(api_key=api_key)
        .workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov8", location=str(destination))
    )
    return Path(dataset.location)
