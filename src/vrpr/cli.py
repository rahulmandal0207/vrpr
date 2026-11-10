"""The ``vrpr`` command line."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from vrpr import __version__
from vrpr.config import AppConfig, load_config
from vrpr.evaluate import evaluate, read_plates
from vrpr.exceptions import ConfigError, DependencyError, VrprError
from vrpr.logging_setup import configure_logging
from vrpr.pipeline import execute_run
from vrpr.plate_details import describe_plate
from vrpr.plates import parse_plate
from vrpr.training import TrainingOptions, download_dataset, train_model, validate_model

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_NO_MATCH = 3

_QUOTES = ("'", '"')


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=value lines from a .env file into the environment.

    Variables already set win, so a value exported in the shell overrides the file.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.removeprefix("export ").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrpr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"vrpr {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="recognize plates in a video, camera or stream")
    run.add_argument("source", help="video file, camera index or stream URL")
    run.add_argument("--config", type=Path, help="YAML settings file")
    run.add_argument("--weights", type=Path, help="plate detector weights")
    run.add_argument("--output", type=Path, help="directory for run results")
    run.add_argument("--device", help="inference device, for example cpu or 0")
    run.add_argument("--display", action="store_true", help="show a live preview")
    run.add_argument("--save-video", action="store_true", help="save an annotated video")
    run.add_argument("--no-crops", action="store_true", help="do not save plate images")
    run.add_argument("--log-format", choices=("text", "json"), help="log output format")
    run.set_defaults(handler=_run)

    train = commands.add_parser("train", help="train the plate detector")
    train.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    train.add_argument("--base-weights", default="yolov8n.pt", help="pretrained starting weights")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--image-size", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", type=Path, default=Path("models/best.pt"))
    train.set_defaults(handler=_train)

    validate = commands.add_parser("validate", help="measure detector accuracy on a dataset")
    validate.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    validate.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    validate.add_argument("--device")
    validate.set_defaults(handler=_validate)

    score = commands.add_parser("evaluate", help="score a run against labelled plates")
    score.add_argument("--expected", type=Path, required=True, help="CSV with a plate column")
    score.add_argument("--predicted", type=Path, required=True, help="a run's plates.csv")
    score.set_defaults(handler=_evaluate)

    dataset = commands.add_parser("dataset", help="download a dataset from Roboflow")
    dataset.add_argument("--workspace", required=True)
    dataset.add_argument("--project", required=True)
    dataset.add_argument("--version", type=int, required=True, dest="dataset_version")
    dataset.add_argument("--output", type=Path, default=Path("data/plates"))
    dataset.set_defaults(handler=_dataset)

    parse = commands.add_parser("parse", help="check how text would be read as a plate")
    parse.add_argument("text")
    parse.set_defaults(handler=_parse)

    web = commands.add_parser("web", help="open the browser interface")
    web.add_argument("--port", type=int, default=8501)
    web.add_argument("--root", type=Path, default=Path(), help="project folder to work in")
    web.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    web.set_defaults(handler=_web)

    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    detector = config.detector
    if args.weights:
        detector = replace(detector, weights=args.weights)
    if args.device:
        detector = replace(detector, device=args.device)

    output = config.output
    if args.output:
        output = replace(output, directory=args.output)
    if args.save_video:
        output = replace(output, save_annotated_video=True)
    if args.no_crops:
        output = replace(output, save_crops=False)

    logging_config = config.logging
    if args.log_format:
        logging_config = replace(logging_config, format=args.log_format)
    return replace(config, detector=detector, output=output, logging=logging_config)


def _run(args: argparse.Namespace) -> int:
    config = _apply_overrides(load_config(args.config), args)
    configure_logging(config.logging)

    stop = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("Received signal %d, finishing up", signum)
        stop.set()

    previous = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        summary = execute_run(config, args.source, display=args.display, stop=stop)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(asdict(summary), indent=2))
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    weights = train_model(
        TrainingOptions(
            data=args.data,
            base_weights=args.base_weights,
            epochs=args.epochs,
            image_size=args.image_size,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            output_weights=args.output,
        )
    )
    print(weights)
    return EXIT_OK


def _validate(args: argparse.Namespace) -> int:
    print(json.dumps(validate_model(args.weights, args.data, args.device), indent=2))
    return EXIT_OK


def _evaluate(args: argparse.Namespace) -> int:
    result = evaluate(read_plates(args.expected), read_plates(args.predicted))
    print(json.dumps(asdict(result), indent=2))
    return EXIT_OK


def _dataset(args: argparse.Namespace) -> int:
    print(download_dataset(args.workspace, args.project, args.dataset_version, args.output))
    return EXIT_OK


def _web(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise DependencyError(
            "The web interface needs Streamlit. Install it with: pip install -e .[web]"
        )
    if not args.root.is_dir():
        raise ConfigError(f"Project folder not found: {args.root}")
    app = Path(__file__).with_name("web") / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address=localhost",  # Plates are personal data: never serve them to the network.
        f"--server.port={args.port}",
        f"--server.headless={'true' if args.no_browser else 'false'}",
        "--server.maxUploadSize=2048",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",  # Hides "Deploy", which offers to publish the app online.
    ]
    try:
        return subprocess.call(command, cwd=args.root)  # noqa: S603 - fixed arguments, no shell
    except KeyboardInterrupt:
        return EXIT_OK


def _parse(args: argparse.Namespace) -> int:
    parsed = parse_plate(args.text)
    if parsed is None:
        print("Not a recognized plate")
        return EXIT_NO_MATCH
    result = asdict(parsed)
    details = describe_plate(parsed.plate)
    if details is not None:
        result.update(details.to_fields(), summary=details.summary, note=details.note)
    print(json.dumps(result, indent=2))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    try:
        status: int = args.handler(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except VrprError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    return status
