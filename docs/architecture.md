# Architecture

## Goals

- Report each vehicle once, with the plate most likely to be correct.
- Never lose results that were already found, including when a run crashes.
- Keep the logic testable without a GPU, model weights or a video of real traffic.

## Modules

| Module | Responsibility |
| --- | --- |
| `cli` | Commands, flags and exit codes |
| `config` | Typed settings, validated when loaded |
| `pipeline` | Runs the stages frame by frame and owns the run's lifecycle |
| `video` | Reads frames, and reconnects cameras and streams |
| `detector` | Wraps YOLO behind the `Detector` protocol |
| `tracking` | Gives detections a stable identity |
| `imaging` | Crops plates and prepares them for OCR |
| `ocr` | Wraps the plate reading model and EasyOCR behind the `OcrEngine` protocol |
| `plates` | Turns OCR text into a validated plate number |
| `consensus` | Combines readings of one vehicle into one result |
| `sinks` | Writes CSV, JSON Lines and crops |
| `rendering` | Optional preview window and annotated video |
| `web` | The Streamlit page started by `vrpr web`, and its file and progress helpers |
| `training` | Trains and measures the detector, and fetches datasets |
| `evaluate` | Scores a run against labelled plates |

`detector` and `ocr` import their ML libraries only when created, so everything else runs and tests
without them. To use a different model, implement the protocol and pass it to `execute_run`.

## Decisions

### One report per vehicle, after it leaves

The tracker often loses a plate for a moment, as the vehicle moves or the detector misses a frame,
and starts a new track for it. On a real 34-second dashcam clip, one car had 10 tracks. Reporting
per track turned each track's small misreads into separate plates for the same car.

So readings are pooled per vehicle. Each track collects readings, and stops reading once
`min_agreement` of them match, apart from occasional rechecks. When a track ends, its readings join
the vehicle whose plate is within two characters of its own, if that vehicle was seen within
`dedupe_window_seconds`; otherwise they start a new vehicle. A vehicle is reported once nothing has
joined it for `report_delay_seconds` and no open track still looks like it, with the plate that has
the most votes across all its tracks. A vehicle that joins one already reported counts as a
suppressed duplicate. On that clip, this cut 16 rows to 10, one per vehicle, and wrong rows from 12
to 6. The cost is that a vehicle appears a few seconds after it leaves the picture.

`confirmed` is true when at least `min_agreement` readings agree and they are at least half of all
the vehicle's readings. It measures consistency, not truth: EasyOCR repeats some misreads, such as
an `M` read as `H`, often enough to be confirmed. Treat it as a strong hint.

Agreement still matters because single frames are unreliable. EasyOCR read one car's plate
correctly in about 1 frame in 10, yet that was still its most common reading, which is why OCR reads
every frame until a plate settles (`ocr.every_n_frames: 1`) and upscales crops to 128 pixels high.
Contrast enhancement, beam-search decoding, extra crop padding and skipping EasyOCR's text detector
all made results worse or no better. Voting cannot fix a misread that EasyOCR makes most of the
time: on the clip above, it read `AP28CC4284` as `AP28CC6284` twice as often as correctly.

### A reader trained on plates, with a general reader behind it

That limit is why plates are read first by a model trained only on number plates
(`fast-plate-ocr`, run with ONNX Runtime). The detector was never the weak stage: it finds about
as many plates at 360p as at 1080p. Reading was. At 1080p a plate is about 43 pixels high, at
480p about 20, and at 360p about 15. On labelled crops from one clip the plate model read 68% of
single frames correctly at 1080p, 53% at 720p, 37% at 480p and 5% at 360p, against 15%, 14%, 3%
and under 1% for EasyOCR, in about 2 ms a crop on a CPU.

The plate model reads the crop as one line, so it fails on two-line plates and on boxes much
larger than the plate. `ocr.engine: hybrid` keeps its answer when it is a valid plate number
read with at least `ocr.fallback_confidence`, and otherwise sends the crop to EasyOCR, which
finds the text lines itself. Vehicles reported with exactly the right plate, of 10 across two
clips, each downscaled from 1080p:

| Video height | EasyOCR alone | Plate model alone | Hybrid |
| --- | --- | --- | --- |
| 1080 | 4 | 8 | 9 |
| 720 | 2 | 8 | 8 |
| 480 | 1 | 5 | 5 |
| 360 | 0 | 3 | 3 |
| 240 | 0 | 1 | 1 |

Wrong rows fell as well, from 6 to 1 at 1080p and from 7 to 2 at 720p. Ten vehicles is a small
sample, so read the table as a direction, not a rate. Below about 15 pixels of plate height most
frames are unreadable by any model tried, including RapidOCR. Sharpening, contrast
equalisation, averaging a track's crops across frames and voting per character instead of per
plate were all tried on the plate model and did not help reliably.

### Identity comes from the tracker, sightings from the plate

Pooling uses only the plate, so a vehicle that returns after `dedupe_window_seconds` is reported
again, which is what a log of sightings should show. Two different vehicles whose plates are within
two characters of each other, seen within the window, would be merged; that is rare, and the window
keeps it unlikely.

### Splitting a track is cheap, merging one is not

If the tracker splits one vehicle into two tracks, the consensus stage pools them and nothing is
lost. If it merges two vehicles into one track, the second vehicle is lost, because its track
already counts as settled. This happened on the first real end-to-end run: a track coasted across a
gap between vehicles and took over the next one in the lane.

Two things guard against it. `tracker.max_age` is kept low, so a track ends quickly instead of
coasting onto another vehicle. And a settled track is read again every
`consensus.recheck_every_n_frames` frames. If that reading is clearly a different plate, the
readings so far go to the first vehicle and the track starts afresh for the second. A reading
within two characters of the settled plate counts as a misread of it.

### Correct OCR only where the layout allows it

`O` and `0` are the same shape, but a plate's layout says which one each position holds. The parser
applies a correction only in a position that demands that character class, and refuses to build a
three-letter series out of a repaired digit. A wrong plate is worse than a missing one, because
nothing downstream would question it.

For the same reason, two shapes that OCR often produces from a partly read plate are refused:
a one-digit RTO code outside Delhi, which is the only state that writes one (`UP8CB0067` is
`UP80CB0067` with a digit dropped), and a plate without series letters. Each part of the plate is
checked in its own slot, rather than matching the whole text against one pattern, so a layout
cannot pass by borrowing a shape from another layout.

### A dedicated tracker

Plates are small and few, so a greedy IoU match with a constant-velocity estimate is enough. It
drops the dependencies the old SORT module brought in, which included matplotlib with a Tk backend,
scikit-image and filterpy, and it replaces GPL-3.0 code with code under the project's own license.
The trade-off is that there is no Kalman filter, so a plate moving very fast or very erratically
between frames is more likely to get a new track. Consensus limits the damage, because the
duplicate is dropped.

### Crops are copies

A numpy slice shares memory with its frame, so anything later drawn on the frame would end up in
the crop. Crops are copied, and annotation works on its own copy of the frame.

### Every run gets its own folder

Results are appended and flushed record by record, so a crash loses at most the plate being
written. The pipeline settles open tracks and closes outputs in a `finally` block, so tracks in
progress are saved as well. Separate folders mean a run can never overwrite an earlier one.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| Bad config, missing weights or missing video | Stops before creating output, with a clear message and a non-zero exit code |
| OCR fails on one crop | Logged and counted in the summary, and the run continues |
| Box outside the frame | Clipped, or skipped if nothing is left |
| Camera or stream drops | Reconnects up to `source.max_read_failures` times |
| Ctrl+C or `SIGTERM` | Finishes the current frame, settles tracks and writes the summary |
| Any other error | Saves collected results, then raises |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Runtime failure |
| 2 | Invalid configuration or arguments |
| 3 | `vrpr parse` did not find a plate |
