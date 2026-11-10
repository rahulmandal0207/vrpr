"""The VRPR web page, a Streamlit script. Start it with ``vrpr web`` from the project folder."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

from vrpr import tuning
from vrpr.config import AppConfig, DetectorConfig, OcrConfig, load_config
from vrpr.detector import YoloDetector
from vrpr.exceptions import VrprError
from vrpr.models import Frame
from vrpr.ocr import EasyOcrEngine
from vrpr.pipeline import execute_run
from vrpr.plate_details import describe_plate
from vrpr.web import advanced, runs

CONFIG_PATH = Path("configs/default.yaml")
VIDEO_DIR = Path("data/videos")
UPLOAD_DIR = Path("data/uploads")
DEVICES = {"Automatic": None, "GPU": "0", "CPU": "cpu"}
SOURCES = ("Video in data/videos", "Upload a video", "Camera or stream")
LAST_RUN = "last_run"
CARDS_PER_ROW = 4
# The preview is display only: the pipeline always analyses full-size frames.
PREVIEW_WIDTH = 960
PREVIEW_INTERVAL_SECONDS = 0.1


@st.cache_resource(show_spinner="Loading the plate detector...", max_entries=4)
def load_detector(weights: str, device: str | None, modified: float) -> YoloDetector:
    # ``modified`` is part of the cache key, so retrained weights are picked up. Prediction
    # settings are not: changing them reuses the loaded model.
    return YoloDetector(DetectorConfig(weights=Path(weights), device=device))


@st.cache_resource(show_spinner="Loading the text reader...", max_entries=4)
def load_ocr(languages: tuple[str, ...], gpu: bool) -> EasyOcrEngine:
    return EasyOcrEngine(OcrConfig(languages=languages, gpu=gpu))


def load_models(config: AppConfig) -> tuple[YoloDetector, EasyOcrEngine]:
    """The shared models, set up with this page's current settings."""
    weights = config.detector.weights
    modified = weights.stat().st_mtime if weights.is_file() else 0.0
    detector = load_detector(str(weights), config.detector.device, modified)
    ocr = load_ocr(config.ocr.languages, config.ocr.gpu)
    return detector.with_settings(config.detector), ocr.with_settings(config.ocr)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_frame(source: str, index: int, modified: float) -> Frame | None:
    # ``modified`` is part of the cache key, so a replaced file is read again.
    return tuning.read_frame(source, index)


def settings_sidebar(base: AppConfig) -> AppConfig:
    sidebar = st.sidebar
    sidebar.header("Settings")
    weights = sidebar.text_input("Detector weights", str(base.detector.weights), key="weights")
    device = sidebar.selectbox("Run the detector on", list(DEVICES), key="device")
    confidence = sidebar.slider(
        "Detection confidence", 0.05, 0.95, float(base.detector.confidence), 0.05, key="confidence"
    )
    ocr_gpu = sidebar.checkbox("Read text on the GPU", base.ocr.gpu, key="ocr_gpu")
    agreement = sidebar.number_input(
        "Readings that must agree",
        min_value=1,
        max_value=10,
        value=base.consensus.min_agreement,
        key="agreement",
    )
    save_crops = sidebar.checkbox("Save plate images", base.output.save_crops, key="save_crops")
    save_video = sidebar.checkbox(
        "Save an annotated video", base.output.save_annotated_video, key="save_video"
    )
    sidebar.caption("Plate numbers and images are personal data. Results stay on this computer.")

    return replace(
        base,
        detector=replace(
            base.detector,
            weights=Path(weights),
            confidence=confidence,
            device=DEVICES.get(device or ""),
        ),
        ocr=replace(base.ocr, gpu=ocr_gpu),
        consensus=replace(base.consensus, min_agreement=int(agreement)),
        output=replace(base.output, save_crops=save_crops, save_annotated_video=save_video),
    )


def choose_source() -> str | None:
    mode = st.radio("Source", SOURCES, horizontal=True, key="source_mode")
    if mode == SOURCES[0]:
        videos = runs.list_videos(VIDEO_DIR)
        if not videos:
            st.info(f"No videos found in {VIDEO_DIR}. Upload one, or add files to that folder.")
            return None
        chosen = st.selectbox("Video", [video.name for video in videos], key="video")
        return str(VIDEO_DIR / chosen) if chosen else None
    if mode == SOURCES[1]:
        upload = st.file_uploader(
            "Video file", type=sorted(suffix[1:] for suffix in runs.VIDEO_SUFFIXES), key="upload"
        )
        if upload is None:
            return None
        if st.session_state.get("upload_saved_for") != upload.file_id:
            saved = runs.save_upload(upload.name, upload.getvalue(), UPLOAD_DIR)
            st.session_state["upload_saved_for"] = upload.file_id
            st.session_state["upload_path"] = str(saved)
        return str(st.session_state["upload_path"])
    value = (st.text_input("Camera number or stream URL", "0", key="stream") or "").strip()
    return value or None


def run_pipeline(config: AppConfig, source: str) -> None:
    # Show progress before touching the file: OneDrive may still be downloading it.
    progress = st.progress(0.0, text="Preparing the video...")
    total = runs.frame_count(source)
    preview = st.empty()
    started: list[float] = []

    def update(frame: Frame, frames: int) -> None:
        if not started:
            started.append(time.monotonic())
        preview.image(
            runs.preview_jpeg(frame, PREVIEW_WIDTH),
            output_format="JPEG",
            width="stretch",
            alt="Live preview of the video being analysed, with detected plates boxed",
        )
        # The header count is an estimate, and can be wrong for files still being downloaded,
        # so past it the bar holds just short of full and only the frame number is shown.
        text = runs.progress_text(frames, total, time.monotonic() - started[0])
        value = frames / total if total and frames <= total else (0.99 if total else 0.0)
        progress.progress(value, text=text)

    try:
        detector, ocr = load_models(config)
        summary = execute_run(
            config,
            source,
            detector=detector,
            ocr=ocr,
            observers=[runs.ThrottledObserver(update, PREVIEW_INTERVAL_SECONDS)],
        )
    except VrprError as exc:
        progress.empty()
        st.error(str(exc))
        return
    progress.progress(1.0, text=f"Done: {summary.frames} frames")
    st.session_state[LAST_RUN] = str(config.output.directory / summary.run_id)


@st.fragment
def tuning_view(config: AppConfig, source: str | None) -> None:
    """Show how the current settings detect and read the plates on one frame of a video."""
    if source is None:
        st.info("Choose a video above first.")
        return
    total = runs.frame_count(source)
    if not total:
        st.info("This works on video files. Choose a video in data/videos, or upload one.")
        return
    index = st.slider(
        "Frame",
        0,
        total - 1,
        0,
        key=f"tune_frame-{source}",
        help="Pick a frame where a plate is clear and close, then change the advanced options "
        "in the side panel and watch the reading.",
    )
    frame = cached_frame(source, index, Path(source).stat().st_mtime)
    if frame is None:
        st.warning("Could not read that frame. Try another one.")
        return
    try:
        detector, ocr = load_models(config)
        with st.spinner("Reading the plates..."):
            trials = tuning.try_frame(frame, detector, ocr, config)
    except VrprError as exc:
        st.error(str(exc))
        return

    st.image(
        runs.preview_jpeg(tuning.draw_trials(frame, trials), PREVIEW_WIDTH),
        output_format="JPEG",
        width="stretch",
        alt=f"Frame {index} with {len(trials)} detected plates boxed and numbered",
    )
    if not trials:
        st.info(
            "No plates found on this frame. Try another frame, or lower the detection "
            "confidence in the side panel."
        )
        return
    st.caption(
        "This reads one frame only. A full run reads each plate on many frames and votes, so "
        "it is more reliable than any single reading here."
    )
    for trial in trials:
        show_trial(trial)


def show_trial(trial: tuning.PlateTrial) -> None:
    with st.container(border=True):
        crop_column, prepared_column, result_column = st.columns([1, 1, 1.2])
        crop_column.image(
            trial.crop,
            channels="BGR",
            caption=f"{trial.number}. Detected with confidence {trial.detection.confidence:.2f}",
            width="stretch",
            alt=f"Plate {trial.number} as detected",
        )
        prepared_column.image(
            trial.prepared,
            # Filtered plates are usually gray; colour ones are still in OpenCV's order.
            channels="BGR" if trial.prepared.ndim == 3 else "RGB",
            caption="What the text reader sees, after the image filters",
            width="stretch",
            alt=f"Plate {trial.number} after the image filters",
        )
        with result_column:
            if trial.error:
                st.error(trial.error)
            elif trial.reading is not None:
                st.markdown(f"#### {trial.reading.plate}")
                st.caption(
                    f"{trial.reading.plate_format} plate · "
                    f"confidence {trial.reading.confidence:.2f}"
                )
            else:
                st.markdown("#### No plate number")
                st.caption("The text read does not make a valid Indian plate number.")
            if trial.fragments:
                st.caption(
                    "EasyOCR read: "
                    + " · ".join(
                        f"`{fragment.text}` {fragment.confidence:.2f}"
                        for fragment in trial.fragments
                    )
                )
            elif not trial.error:
                st.caption("EasyOCR found no text.")


def crop_file(results: runs.RunResults, crop_path: str) -> Path | None:
    """Find a plate image inside the run folder, wherever the run was started from."""
    if not crop_path:
        return None
    path = results.path / "crops" / Path(crop_path).name
    return path if path.is_file() else None


def show_plate_cards(results: runs.RunResults) -> None:
    # A card gallery rather than st.dataframe: tables need pandas, which some locked-down
    # Windows machines refuse to load.
    for start in range(0, len(results.plates), CARDS_PER_ROW):
        columns = st.columns(CARDS_PER_ROW)
        for column, row in zip(
            columns, results.plates[start : start + CARDS_PER_ROW], strict=False
        ):
            with column.container(border=True):
                plate = row.get("plate", "?")
                image = crop_file(results, row.get("crop_path", ""))
                if image is not None:
                    st.image(str(image), width="stretch", alt=f"Number plate read as {plate}")
                st.markdown(f"#### {plate}")
                unconfirmed = runs.is_unconfirmed(row)
                if unconfirmed:
                    st.badge("Unconfirmed", icon=":material/help:", color="orange")
                # Details describe the number as read, so a misread plate would get another
                # number's. Show them only once enough readings agree. They are worked out from
                # the number itself, so runs from before these columns existed get them too.
                details = None if unconfirmed else describe_plate(plate)
                if details is not None and details.summary:
                    st.markdown(
                        f":material/location_on: {details.summary}", help=details.note or None
                    )
                st.caption(
                    f"Confidence {float(row.get('confidence') or 0):.2f} · "
                    f"{row.get('votes', '?')} of {row.get('total_reads', '?')} readings agree · "
                    f"{float(row.get('timestamp_s') or 0):.1f} s · {row.get('plate_format', '')}"
                )


def show_results(run_dir: Path, place: str) -> None:
    """Show one run. ``place`` keeps widget keys unique when a run appears in both tabs."""
    results = runs.load_run(run_dir)
    if not results.complete:
        st.warning(
            "This run stopped before it finished, so only the plates found so far are shown."
        )
    summary = results.summary or {}
    columns = st.columns(4)
    columns[0].metric("Plates", len(results.plates))
    columns[1].metric("Frames", summary.get("frames", "-"))
    columns[2].metric("Frames per second", summary.get("frames_per_second", "-"))
    columns[3].metric("Vehicles not read", summary.get("tracks_unread", "-"))
    unconfirmed = sum(runs.is_unconfirmed(row) for row in results.plates)
    if unconfirmed:
        st.caption(
            f"{unconfirmed} of these plates are unconfirmed: too few readings agreed, "
            "so treat them as best guesses."
        )

    if not results.plates:
        st.info("No plates were found in this run.")
    else:
        show_plate_cards(results)
        csv_path = run_dir / "plates.csv"
        st.download_button(
            "Download plates.csv",
            csv_path.read_bytes(),
            file_name=f"{results.run_id}-plates.csv",
            mime="text/csv",
            key=f"download-{place}-{results.run_id}",
        )
    if (run_dir / "annotated.mp4").is_file():
        st.caption("The annotated video is saved in the run folder. Open it with a video player.")
    st.caption(f"Run folder: {run_dir.resolve()}")


def main() -> None:
    st.set_page_config(page_title="VRPR", layout="wide")
    st.title("Vehicle registration plate recognition")

    config = settings_sidebar(load_config(CONFIG_PATH if CONFIG_PATH.is_file() else None))
    config = advanced.advanced_options(config)
    run_tab, history_tab = st.tabs(["Run", "Past runs"])

    with run_tab:
        source = choose_source()
        if st.button("Run", type="primary", disabled=source is None, key="run") and source:
            run_pipeline(config, source)
        tune = st.expander(
            "Try settings on one frame", icon=":material/tune:", on_change="rerun", key="tune"
        )
        if tune.open:
            with tune:
                tuning_view(config, source)
        last_run = st.session_state.get(LAST_RUN)
        if last_run and Path(last_run).is_dir():
            st.subheader("Results")
            show_results(Path(last_run), "latest")

    with history_tab:
        past = runs.list_runs(config.output.directory)
        if not past:
            st.info("No runs yet.")
        else:
            chosen = st.selectbox("Run", [path.name for path in past], key="history_run")
            if chosen:
                show_results(config.output.directory / chosen, "history")


try:
    main()
except VrprError as exc:  # A bad config file shows as a message rather than a traceback.
    st.error(str(exc))
