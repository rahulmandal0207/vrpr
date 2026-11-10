"""The VRPR web page, a Streamlit script.

Start it with ``vrpr web`` from the project folder. ``streamlit_app.py`` at the repository root
serves the same page as a public demo, for hosting such as Streamlit Community Cloud.
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st

from vrpr.config import AppConfig, DetectorConfig, OcrConfig, load_config
from vrpr.detector import YoloDetector
from vrpr.exceptions import VrprError
from vrpr.models import Frame
from vrpr.ocr import EasyOcrEngine
from vrpr.pipeline import execute_run
from vrpr.rendering import FrameObserver
from vrpr.web import runs

CONFIG_PATH = Path("configs/default.yaml")
VIDEO_DIR = Path("data/videos")
UPLOAD_DIR = Path("data/uploads")
SAMPLE_DIR = Path("samples")
DEVICES = {"Automatic": None, "GPU": "0", "CPU": "cpu"}
LOCAL_SOURCES = ("Video in data/videos", "Upload a video", "Camera or stream")
# Sources that need the visitor's own computer. The demo lists them too, greyed out, so visitors
# see everything the project does.
LOCAL_ONLY_SOURCES = ("Video in data/videos", "Camera or stream")
PUBLIC_SOURCES = ("Sample video", "Upload a video", *LOCAL_ONLY_SOURCES)
LAST_RUN = "last_run"
SESSION_DIR = "session_dir"
CARDS_PER_ROW = 4
# The preview is display only: the pipeline always analyses full-size frames.
PREVIEW_WIDTH = 960
PREVIEW_INTERVAL_SECONDS = 0.1
# Public demo limits keep one visitor from tying up a small shared server.
PUBLIC_MAX_SECONDS = 15
PUBLIC_MAX_UPLOAD_MB = 50
PUBLIC_SESSION_HOURS = 2
# OCR is the slowest stage on a CPU. Reading every second frame needed 35-48% fewer reads and got
# the same plates right on two clips with answer keys. The detector still sees every frame.
PUBLIC_OCR_EVERY_N_FRAMES = 2
SOURCE_CODE_URL = "https://github.com/rahulmandal0207/vrpr"
CLONE_COMMAND = f"git clone {SOURCE_CODE_URL}.git"
# Shown on hover over each control the demo cannot offer.
CLONE_HELP = (
    f"Not available in this demo. [Clone VRPR from GitHub]({SOURCE_CODE_URL}) and run it on your "
    "own computer to use this."
)
NOTICE_SHOWN = "demo_notice_shown"

PUBLIC = runs.public_demo()


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


@st.cache_resource
def run_lock() -> threading.Lock:
    """The models are shared by every visitor and are not safe to run on two videos at once."""
    return threading.Lock()


def session_folder() -> Path:
    """A private folder for this browser session, so visitors never see each other's runs."""
    existing = st.session_state.get(SESSION_DIR)
    if existing and Path(existing).is_dir():
        return Path(existing)
    root = Path(tempfile.gettempdir())
    runs.remove_old_sessions(root, PUBLIC_SESSION_HOURS * 3600)
    folder = Path(tempfile.mkdtemp(prefix=runs.SESSION_PREFIX, dir=root))
    st.session_state[SESSION_DIR] = str(folder)
    return folder


def settings_sidebar(base: AppConfig) -> AppConfig:
    sidebar = st.sidebar
    sidebar.header("Settings")
    # In the demo these controls are shown greyed out, with a tooltip on how to get them.
    locked: dict[str, Any] = {"disabled": True, "help": CLONE_HELP} if PUBLIC else {}
    weights = sidebar.text_input(
        "Detector weights", str(base.detector.weights), key="weights", **locked
    )
    device = (
        sidebar.selectbox(
            "Run the detector on",
            list(DEVICES),
            index=list(DEVICES).index("CPU") if PUBLIC else 0,
            key="device",
            **locked,
        )
        or ""
    )
    confidence = sidebar.slider(
        "Detection confidence", 0.05, 0.95, float(base.detector.confidence), 0.05, key="confidence"
    )
    ocr_gpu = sidebar.checkbox(
        "Read text on the GPU", base.ocr.gpu and not PUBLIC, key="ocr_gpu", **locked
    )
    agreement = sidebar.number_input(
        "Readings that must agree",
        min_value=1,
        max_value=10,
        value=base.consensus.min_agreement,
        key="agreement",
    )
    save_video = sidebar.checkbox(
        "Save an annotated video",
        base.output.save_annotated_video and not PUBLIC,
        key="save_video",
        **locked,
    )
    if PUBLIC:
        # Disabled widgets keep their values, but the demo must never use the local settings.
        weights, device, ocr_gpu, save_video = str(base.detector.weights), "CPU", False, False
    else:
        sidebar.caption(
            "Plate numbers and images are personal data. Results stay on this computer."
        )

    output = base.output
    if PUBLIC:
        output = replace(output, directory=session_folder() / "output", save_crops=True)
    return replace(
        base,
        detector=replace(
            base.detector, weights=Path(weights), confidence=confidence, device=DEVICES.get(device)
        ),
        ocr=replace(
            base.ocr,
            gpu=ocr_gpu,
            every_n_frames=PUBLIC_OCR_EVERY_N_FRAMES if PUBLIC else base.ocr.every_n_frames,
        ),
        consensus=replace(base.consensus, min_agreement=int(agreement)),
        output=replace(output, save_annotated_video=save_video),
    )


def choose_upload() -> str | None:
    upload = st.file_uploader(
        "Video file",
        type=sorted(suffix[1:] for suffix in runs.VIDEO_SUFFIXES),
        key="upload",
        max_upload_size=PUBLIC_MAX_UPLOAD_MB if PUBLIC else None,
    )
    if upload is None:
        return None
    if st.session_state.get("upload_saved_for") != upload.file_id:
        folder = session_folder() / "uploads" if PUBLIC else UPLOAD_DIR
        saved = runs.save_upload(upload.name, upload.getvalue(), folder)
        st.session_state["upload_saved_for"] = upload.file_id
        st.session_state["upload_path"] = str(saved)
    return str(st.session_state["upload_path"])


def choose_from_folder(folder: Path) -> str | None:
    videos = runs.list_videos(folder)
    if not videos:
        st.info(f"No videos found in {folder}. Upload one, or add files to that folder.")
        return None
    chosen = st.selectbox("Video", [video.name for video in videos], key="video")
    return str(folder / chosen) if chosen else None


def locked_source(mode: str) -> None:
    """Show a local-only source's input, greyed out, so the demo still shows what it does."""
    if mode == LOCAL_ONLY_SOURCES[0]:
        st.selectbox(
            "Video", ["Videos from your own data/videos folder"], disabled=True, help=CLONE_HELP
        )
    else:
        st.text_input("Camera number or stream URL", "0", disabled=True, help=CLONE_HELP)


def choose_source() -> str | None:
    sources = PUBLIC_SOURCES if PUBLIC else LOCAL_SOURCES
    captions = [
        "Needs your own computer" if PUBLIC and source in LOCAL_ONLY_SOURCES else ""
        for source in sources
    ]
    mode = st.radio("Source", sources, horizontal=True, key="source_mode", captions=captions)
    if mode == "Upload a video":
        return choose_upload()
    if PUBLIC and mode is not None and mode in LOCAL_ONLY_SOURCES:
        locked_source(mode)
        return None
    if mode == sources[0]:
        return choose_from_folder(SAMPLE_DIR if PUBLIC else VIDEO_DIR)
    value = (st.text_input("Camera number or stream URL", "0", key="stream") or "").strip()
    return value or None


def run_pipeline(config: AppConfig, source: str) -> None:
    # Show progress before touching the file: OneDrive may still be downloading it.
    progress = st.progress(0.0, text="Preparing the video...")
    total = runs.frame_count(source)
    limit = runs.FrameLimit(runs.frames_for_seconds(source, PUBLIC_MAX_SECONDS)) if PUBLIC else None
    if limit is not None and total:
        total = min(total, limit.max_frames)
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

    observers: list[FrameObserver] = [runs.ThrottledObserver(update, PREVIEW_INTERVAL_SECONDS)]
    if limit is not None:
        observers.append(limit)

    lock = run_lock()
    if not lock.acquire(blocking=False):
        progress.progress(0.0, text="Waiting for another video to finish...")
        lock.acquire()
    try:
        weights = config.detector.weights
        detector = load_detector(
            str(weights),
            config.detector.confidence,
            config.detector.image_size,
            config.detector.device,
            weights.stat().st_mtime if weights.is_file() else 0.0,
        )
        ocr = load_ocr(config.ocr.languages, config.ocr.gpu)
        summary = execute_run(config, source, detector=detector, ocr=ocr, observers=observers)
    except VrprError as exc:
        progress.empty()
        st.error(str(exc))
        return
    finally:
        lock.release()
    progress.progress(1.0, text=f"Done: {summary.frames} frames")
    if limit is not None and limit.reached:
        st.info(f"This demo analyses the first {PUBLIC_MAX_SECONDS} seconds of each video.")
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
    if PUBLIC:
        return
    if (run_dir / "annotated.mp4").is_file():
        st.caption("The annotated video is saved in the run folder. Open it with a video player.")
    st.caption(f"Run folder: {run_dir.resolve()}")


@st.dialog("This is a demo", icon=":material/info:", width="medium")
def demo_notice() -> None:
    st.markdown(
        "This page is a demo of VRPR. It runs on a small shared server with no GPU, so it is "
        f"slow, reads plates less often and analyses only the first {PUBLIC_MAX_SECONDS} seconds "
        "of each video. Some settings and sources are greyed out because they need your own "
        "computer."
    )
    st.markdown(
        "To see how fast and accurate it really is, clone the repository and run it with a GPU:"
    )
    st.code(CLONE_COMMAND, language="bash")
    with st.container(horizontal=True):
        st.link_button("Open on GitHub", SOURCE_CODE_URL, icon=":material/open_in_new:")
        if st.button("Try the demo", type="primary", key="close_notice"):
            st.rerun()


def public_notice() -> None:
    st.caption(
        f"A demo of [VRPR]({SOURCE_CODE_URL}), open source under the MIT License with "
        "AGPL-3.0 dependencies. It runs on a small shared CPU, so it analyses at most "
        f"{PUBLIC_MAX_SECONDS} seconds of each video. Videos you upload are processed on this "
        f"server, visible only in your browser session, and deleted within about "
        f"{PUBLIC_SESSION_HOURS} hours. Number plates are personal data: only upload footage you "
        "are allowed to share. Readings can be wrong, so do not rely on them."
    )


def main() -> None:
    st.set_page_config(page_title="VRPR", layout="wide")
    st.title("Vehicle registration plate recognition")
    if PUBLIC:
        public_notice()
        # Mark it shown first, so closing it any way, including the X, keeps it closed.
        if not st.session_state.get(NOTICE_SHOWN):
            st.session_state[NOTICE_SHOWN] = True
            demo_notice()

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
