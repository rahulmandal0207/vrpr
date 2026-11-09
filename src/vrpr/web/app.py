"""The VRPR web page, a Streamlit script. Start it with ``vrpr web`` from the project folder."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from vrpr.config import AppConfig, DetectorConfig, OcrConfig, load_config
from vrpr.detector import YoloDetector
from vrpr.exceptions import VrprError
from vrpr.models import Frame
from vrpr.ocr import EasyOcrEngine
from vrpr.pipeline import execute_run
from vrpr.web import runs

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


@st.cache_resource(show_spinner="Loading the plate detector...")
def load_detector(
    weights: str, confidence: float, image_size: int, device: str | None, modified: float
) -> YoloDetector:
    # ``modified`` is part of the cache key, so retrained weights are picked up.
    return YoloDetector(
        DetectorConfig(
            weights=Path(weights), confidence=confidence, image_size=image_size, device=device
        )
    )


@st.cache_resource(show_spinner="Loading the text reader...")
def load_ocr(languages: tuple[str, ...], gpu: bool) -> EasyOcrEngine:
    return EasyOcrEngine(OcrConfig(languages=languages, gpu=gpu))


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

    def update(frame: Frame, frames: int) -> None:
        preview.image(
            runs.preview_jpeg(frame, PREVIEW_WIDTH),
            output_format="JPEG",
            width="stretch",
            alt="Live preview of the video being analysed, with detected plates boxed",
        )
        # The header count is an estimate, and can be wrong for files still being downloaded,
        # so past it the bar holds just short of full and only the frame number is shown.
        if total and frames <= total:
            progress.progress(frames / total, text=f"Frame {frames} of {total}")
        else:
            progress.progress(0.99 if total else 0.0, text=f"Frame {frames}")

    weights = config.detector.weights
    try:
        detector = load_detector(
            str(weights),
            config.detector.confidence,
            config.detector.image_size,
            config.detector.device,
            weights.stat().st_mtime if weights.is_file() else 0.0,
        )
        ocr = load_ocr(config.ocr.languages, config.ocr.gpu)
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
                if runs.is_unconfirmed(row):
                    st.badge("Unconfirmed", icon=":material/help:", color="orange")
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
    run_tab, history_tab = st.tabs(["Run", "Past runs"])

    with run_tab:
        source = choose_source()
        if st.button("Run", type="primary", disabled=source is None, key="run") and source:
            run_pipeline(config, source)
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
