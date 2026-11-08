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
| `ocr` | Wraps EasyOCR behind the `OcrEngine` protocol |
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

### A track reports when readings agree, or when it ends

Taking the first valid reading makes one bad frame final. Waiting for the track to end delays every
result until the vehicle has left. VRPR reports as soon as `min_agreement` readings match, then
reads that track only occasionally. A track that ends first reports its highest-scoring plate, so
vehicles seen only briefly are not lost.

### Identity comes from the tracker, duplicates from the plate

The tracker cannot know that a vehicle it lost and found again is the same one, so that vehicle gets
a new track. The consensus stage remembers recently reported plates and drops a repeat inside
`dedupe_window_seconds`. This also means a plate is reported again if the same vehicle returns
later, which is usually what a log of sightings should show.

### Splitting a track is cheap, merging one is not

If the tracker splits one vehicle into two tracks, the consensus stage drops the repeat and nothing
is lost. If it merges two vehicles into one track, the second vehicle is lost, because its track
already counts as settled. This happened on the first real end-to-end run: a track coasted across a
gap between vehicles and took over the next one in the lane.

Two things guard against it. `tracker.max_age` is kept low, so a track ends quickly instead of
coasting onto another vehicle. And a settled track is read again every
`consensus.recheck_every_n_frames` frames. If that reading is clearly a different plate, voting
starts over on that track. A reading within two characters of the reported plate counts as a
misread of it.

### Correct OCR only where the layout allows it

`O` and `0` are the same shape, but a plate's layout says which one each position holds. The parser
applies a correction only in a position that demands that character class, and refuses to build a
three-letter series out of a repaired digit. A wrong plate is worse than a missing one, because
nothing downstream would question it.

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
