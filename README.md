# VRPR: Vehicle Registration Plate Recognition

![Project banner](vrpr.png)

VRPR finds Indian vehicle registration plates in video and reports each vehicle once. It works on
files, cameras and network streams, and writes results as CSV and JSON Lines with an image of each
plate.

It started as a Python Programming course project at GD Goenka University.

## How it works

```mermaid
flowchart LR
    A[Video source] --> B[YOLOv8 detector]
    B --> C[IoU tracker]
    C --> D[Crop and prepare]
    D --> E[EasyOCR]
    E --> F[Plate parser]
    F --> G[Consensus vote]
    G --> H[CSV, JSONL, crops]
```

- **Tracking** gives each plate a stable identity across frames.
- **The parser** handles standard, Bharat series and two-line plates, and repairs common OCR
  confusions such as `O` for `0`, but only where the plate layout requires a digit or a letter.
- **Consensus** takes the plate that several frames agree on, so one bad frame cannot produce a
  wrong result. OCR stops once a plate is settled, so it is not repeated needlessly.

See [docs/architecture.md](docs/architecture.md) for the design.

## Install

Python 3.10 to 3.12 is recommended for the ML dependencies.

```bash
git clone https://github.com/rahulmandal0207/vrpr.git
cd vrpr
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate`
elsewhere. Then install:

```bash
pip install -e ".[ml]"
```

Add `dataset` to the extras to download training data, and `dev` to run the tests.

## Quick start

A model is not included, so train one first or copy your own to `models/best.pt`.

```bash
vrpr run traffic.mp4
```

`vrpr run` also accepts a camera index or a stream URL:

```bash
vrpr run 0 --display
```

```bash
vrpr run rtsp://camera.local/stream --config configs/default.yaml --save-video
```

Each run writes to its own folder, `output/<run id>/`:

| File | Contents |
| --- | --- |
| `plates.csv`, `plates.jsonl` | One record per vehicle, written as it is found |
| `crops/` | The clearest image of each plate |
| `summary.json` | Frames, speed, OCR calls, plates found and tracks left unread |
| `annotated.mp4` | The video with boxes drawn, with `--save-video` |

To check how a piece of text would be read as a plate:

```bash
vrpr parse "HR 26 DK 8337"
```

## Train a model

The dataset is not in this repository. Download a YOLO-format dataset from Roboflow, or use any
dataset that has a `data.yaml`. For Roboflow, copy `.env.example` to `.env` and put your private API
key in it. Git ignores `.env`, and a `ROBOFLOW_API_KEY` set in your shell takes priority over it.

```bash
vrpr dataset --workspace <workspace> --project <project> --version 3
```

```bash
vrpr train --data data/plates/data.yaml --epochs 100
```

Training saves the best weights to `models/best.pt`, which is where `vrpr run` looks by default.

## Measure accuracy

There are two separate questions, and both need your own data.

**Does it find plates?** This reports mAP, precision and recall on the dataset:

```bash
vrpr validate --data data/plates/data.yaml
```

**Does it read them correctly?** List the plates in a video in a CSV with a `plate` column, then
score a run against it:

```bash
vrpr evaluate --expected labels.csv --predicted output/<run id>/plates.csv
```

## Configuration

Settings live in YAML. [configs/default.yaml](configs/default.yaml) documents every option, and
your file only needs the values it changes. Unknown or out-of-range settings stop the run with a
clear message.

## Live preview

`--display` needs a build of OpenCV with GUI support. Ultralytics installs `opencv-python` and
EasyOCR installs `opencv-python-headless`. They occupy the same `cv2` module and whichever installed
last wins, so if the preview reports no GUI support, reinstall the GUI build:

```bash
pip uninstall -y opencv-python opencv-python-headless
```

```bash
pip install opencv-python
```

Use `--save-video` instead on servers.

## Docker

```bash
docker build -t vrpr .
```

```bash
docker run --rm -v "$PWD/data:/data" vrpr run /data/traffic.mp4 --weights /data/best.pt --output /data/output --device cpu
```

The image runs on CPU as a non-root user.

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest --cov
```

```bash
ruff check . && ruff format --check . && mypy
```

The default test run uses fake models, so it needs no GPU or weights. `pytest -m integration` runs
against real YOLO and EasyOCR. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Privacy

Registration numbers and images of vehicles can identify people. Before using VRPR on real traffic,
read [docs/privacy.md](docs/privacy.md). Logs mask plate numbers, but the output files do not.

## License

VRPR's own code is under the MIT License. It depends on AGPL-3.0 software, which affects how you can
distribute it: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
