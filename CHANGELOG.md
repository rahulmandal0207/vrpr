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
- A `.env` file in the working folder is read on start, for settings such as `ROBOFLOW_API_KEY`.
- A browser interface, `vrpr web`, with a live preview, plate gallery and past runs.
- A public demo for Streamlit Community Cloud, kept on the `demo` branch.
- The trained model, `models/best.pt`, so a fresh clone runs without training.
- Speed and time left in the web page's progress bar.
- Advanced options on the web page for OpenCV image filters, YOLO detection and EasyOCR
  reading, with a view that tries them on one frame and shows each plate before and after
  the filters, and which reader read it. They are also config settings: `detector.iou`,
  `detector.augment`, EasyOCR's readtext settings under `ocr`, and a new `preprocess` section
  whose filters both readers apply. The defaults leave results unchanged.
- Plate details: the state, RTO office and area of each plate, and the year of Bharat series
  plates, from a bundled table of about 1,300 RTO codes. They are added to `plates.csv`,
  `plates.jsonl`, the web page's plate cards and `vrpr parse`, and need no network access.

- A plate reading model, `fast-plate-ocr`, read before EasyOCR. It reads small, blurred and
  low-resolution plates far better: on two test clips, vehicles reported with exactly the
  right plate rose from 4 to 9 of 10 at 1080p, from 2 to 8 at 720p, from 1 to 5 at 480p and
  from 0 to 3 at 360p. EasyOCR still reads the crops the model is unsure about, such as
  two-line plates. `ocr.engine`, `ocr.plate_model` and `ocr.fallback_confidence` control it.
  Existing installs need `pip install -e ".[ml]"` again, and the first run downloads the
  model (about 3 MB).

### Performance

- On the CPU, PyTorch's threads are limited to the cores a container may use. Ultralytics set
  up to 8 from the machine's core count, which slowed capped hosts such as Streamlit Community
  Cloud.
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
- Partial reads such as `UP8CB0067` (for `UP80CB0067`) were accepted as plates, because the
  whole-plate pattern let a dropped digit pass as a one-digit RTO code.
- One vehicle was reported several times with slightly different plates, because the tracker
  restarts on the same plate and each track reported its own misreads. Readings are now pooled
  per vehicle. On a real 34-second dashcam clip with a hand-checked answer key, rows fell from 16
  to 10, one per vehicle, and wrong rows from 12 to 6.

### Changed

- Replaced the SORT tracker with a built-in IoU tracker, removing the GPL-3.0 file and the
  matplotlib, scikit-image and filterpy dependencies.
- Replaced `requirements.txt`, which was UTF-16, with `pyproject.toml`.
- Updated Ultralytics from 8.0 to 8.3 or later.
- Logs now mask plate numbers.
- OCR reads every frame until a plate is settled, on crops upscaled to 128 pixels high, so more
  readings vote. One-digit RTO codes are accepted only for Delhi, and series letters are required.
- Results have a `confirmed` field: true when at least `min_agreement` readings agree and they are
  at least half of the vehicle's readings. The web page marks the others as unconfirmed.
- A vehicle is reported once it has left the picture for `report_delay_seconds` (3 by default),
  not as soon as its first track settles.

### Removed

- `Src/info.py`, the owner lookup: see [docs/privacy.md](docs/privacy.md).
- `Src/car_count.py`, which was unused and crashed on launch.
- `Src/img2txt.py`, left over from the Tesseract version.
