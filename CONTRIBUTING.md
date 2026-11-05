# Contributing

## Setup

```bash
pip install -e ".[dev]"
```

```bash
pre-commit install
```

## Before opening a pull request

All of these run in CI:

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
mypy
```

```bash
pytest --cov --cov-fail-under=90
```

If you change `detector.py` or `ocr.py`, also run the checks against the real libraries, which need
the `ml` extra and download models on first use:

```bash
pytest -m integration
```

## Guidelines

- Every bug fix comes with a test that fails without it.
- Tests use the fakes in `tests/conftest.py` rather than real models, so they stay fast.
- Keep ML imports inside `detector.py`, `ocr.py` and `training.py`, and inside functions.
- Never put a full plate number in a log message. Use `mask_plate`.
- Never commit weights, datasets, videos, run output or API keys.
- New settings need a default, validation, and an entry in `configs/default.yaml`.
