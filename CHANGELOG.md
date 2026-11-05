# Changelog

This project follows [Semantic Versioning](https://semver.org).

## 1.0.0 (unreleased)

The scripts in `Src/` became an installable package with a command line. Commands and output files
differ from before: see the README.

### Added

- The `vrpr` command with `run`, `train`, `validate`, `evaluate`, `dataset` and `parse`.
- YAML configuration, validated when loaded.
- Consensus across frames, and repeat suppression by plate number.
- Bharat series, three-letter series, one-digit RTO and two-line plate support.
- Position-aware repair of OCR confusions, and state code validation.
- Periodic rechecks of settled tracks, so a track that moves onto another vehicle cannot lose it.
- Reconnection for cameras and streams, and clean shutdown on Ctrl+C or `SIGTERM`.
- JSON Lines output, a run summary, an optional annotated video and JSON logs.
- Tests, type checks, linting, CI, a Dockerfile and documentation.

### Fixed

- The CSV kept only the last plate, because each write reopened it in `w` mode.
- A plate was dropped when a newer track had already been logged.
- A box outside the frame produced an empty crop and crashed the run.
- The green tracking box was drawn into the crop used for OCR and saved to disk.
- A fresh clone failed because the output folders did not exist.
- OCR read only the first text fragment, which broke two-line plates.
- OCR kept running on plates that had already been logged.
- A vehicle lost for two frames could be logged again under a new track.
- `python Src/main.py` failed outside PyCharm because `Utils` could not be imported.

### Changed

- Replaced the SORT tracker with a built-in IoU tracker, removing the GPL-3.0 file and the
  matplotlib, scikit-image and filterpy dependencies.
- Replaced `requirements.txt`, which was UTF-16, with `pyproject.toml`.
- Updated Ultralytics from 8.0 to 8.3 or later.
- Logs now mask plate numbers.

### Removed

- `Src/info.py`, the owner lookup: see [docs/privacy.md](docs/privacy.md).
- `Src/car_count.py`, which was unused and crashed on launch.
- `Src/img2txt.py`, left over from the Tesseract version.
