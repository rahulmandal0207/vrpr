"""The web page's advanced options: OpenCV filters, YOLO and text reader settings.

Each widget starts from the loaded configuration, so the defaults match ``vrpr run``. Widget
keys start with ``adv_``, which lets the reset button clear them all at once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import streamlit as st

from vrpr.config import (
    CONTRAST_METHODS,
    DENOISE_METHODS,
    OCR_DECODERS,
    OCR_ENGINES,
    OCR_ROTATIONS,
    THRESHOLD_METHODS,
    AppConfig,
    OcrConfig,
    dump_config,
)

PREFIX = "adv_"
IMAGE_SIZES = (320, 416, 480, 512, 640, 768, 960, 1280)
_LABELS = {
    "none": "Off",
    "bilateral": "Bilateral",
    "median": "Median",
    "clahe": "CLAHE",
    "equalize": "Equalize",
    "otsu": "Otsu",
    "adaptive": "Adaptive",
    "greedy": "Greedy",
    "beamsearch": "Beam search",
    "hybrid": "Both",
    "plate": "Plate model",
    "easyocr": "EasyOCR",
}


def reset_advanced() -> None:
    """Return every advanced option to the loaded configuration."""
    for key in [key for key in st.session_state if str(key).startswith(PREFIX)]:
        del st.session_state[key]


def _label(value: Any) -> str:
    return _LABELS.get(str(value), str(value))


def _choice(label: str, options: tuple[str, ...], default: str, key: str, help: str) -> str:
    chosen = st.segmented_control(
        label,
        options,
        default=default,
        required=True,
        format_func=_label,
        key=PREFIX + key,
        help=help,
    )
    return chosen or default


def changed_settings(base: AppConfig, config: AppConfig) -> int:
    """Count the advanced settings that differ from ``base``."""
    count = 0
    for section in ("detector", "ocr", "preprocess"):
        before, after = getattr(base, section), getattr(config, section)
        count += sum(
            getattr(before, name) != getattr(after, name)
            for name in vars(before)
            if name not in ("weights", "device", "confidence", "gpu", "languages", "every_n_frames")
        )
    return count


def image_filters(base: AppConfig) -> AppConfig:
    filters, ocr = base.preprocess, base.ocr
    grayscale = st.toggle(
        "Convert to grayscale",
        filters.grayscale,
        key=PREFIX + "grayscale",
        help="For EasyOCR, which reads gray images well. The plate model is given colour "
        "unless Black and white is on.",
    )
    min_height = st.slider(
        "Enlarge plates smaller than (pixels tall)",
        32,
        256,
        ocr.min_crop_height,
        16,
        key=PREFIX + "min_height",
        help="Small crops are scaled up to this height for EasyOCR, and for the plate model "
        "when a filter is on.",
    )
    padding = st.slider(
        "Margin around the plate",
        0.0,
        0.3,
        float(ocr.crop_padding),
        0.01,
        key=PREFIX + "padding",
        help="Share of the box added on each side. Too little can cut off characters.",
    )
    denoise = _choice(
        "Noise reduction",
        DENOISE_METHODS,
        filters.denoise,
        "denoise",
        "Bilateral smooths grain but keeps edges. Median removes speckles.",
    )
    gamma = st.slider(
        "Brightness (gamma)",
        0.3,
        3.0,
        float(filters.gamma),
        0.1,
        key=PREFIX + "gamma",
        help="Above 1 brightens dark plates; below 1 darkens washed-out ones.",
    )
    contrast = _choice(
        "Contrast",
        CONTRAST_METHODS,
        filters.contrast,
        "contrast",
        "CLAHE evens out shadows and glare locally. Equalize stretches the whole image.",
    )
    clip_limit = filters.clahe_clip_limit
    if contrast == "clahe":
        clip_limit = st.slider(
            "CLAHE strength",
            0.5,
            8.0,
            float(filters.clahe_clip_limit),
            0.5,
            key=PREFIX + "clahe",
            help="Higher values boost contrast more, and noise with it.",
        )
    sharpen = st.slider(
        "Sharpen",
        0.0,
        3.0,
        float(filters.sharpen),
        0.1,
        key=PREFIX + "sharpen",
        help="Unsharp mask. Helps motion blur a little; too much creates halos.",
    )
    threshold = _choice(
        "Black and white",
        THRESHOLD_METHODS,
        filters.threshold,
        "threshold",
        "Turns the plate into pure black and white. Otsu picks one cut-off for the whole "
        "plate; Adaptive picks one per area, which copes with uneven light.",
    )
    block_size = filters.adaptive_block_size
    if threshold == "adaptive":
        block_size = st.slider(
            "Adaptive area size (pixels)",
            3,
            99,
            filters.adaptive_block_size,
            2,
            key=PREFIX + "block",
            help="Should be larger than a character's stroke width.",
        )
    invert = st.toggle(
        "Invert colours",
        filters.invert,
        key=PREFIX + "invert",
        help="For light text on dark plates, such as black commercial plates.",
    )
    return replace(
        base,
        ocr=replace(base.ocr, min_crop_height=min_height, crop_padding=padding),
        preprocess=replace(
            filters,
            grayscale=grayscale,
            denoise=denoise,
            gamma=gamma,
            contrast=contrast,
            clahe_clip_limit=clip_limit,
            sharpen=sharpen,
            threshold=threshold,
            adaptive_block_size=block_size,
            invert=invert,
        ),
    )


def detector_settings(base: AppConfig) -> AppConfig:
    detector = base.detector
    sizes = sorted({*IMAGE_SIZES, detector.image_size})
    image_size = st.select_slider(
        "Detector image size",
        sizes,
        detector.image_size,
        key=PREFIX + "image_size",
        help="Frames are resized to this before detection. Larger finds smaller, distant "
        "plates but is slower. The model was trained at 640.",
    )
    iou = st.slider(
        "Overlap limit",
        0.1,
        0.95,
        float(detector.iou),
        0.05,
        key=PREFIX + "iou",
        help="When two boxes overlap more than this, YOLO keeps only the stronger one. Lower "
        "it if one plate gets two boxes.",
    )
    augment = st.toggle(
        "Test-time augmentation",
        detector.augment,
        key=PREFIX + "augment",
        help="Also detects on flipped and rescaled copies of each frame. Can find harder "
        "plates, but is about three times slower.",
    )
    return replace(
        base, detector=replace(detector, image_size=image_size, iou=iou, augment=augment)
    )


def reader_settings(base: AppConfig) -> AppConfig:
    ocr = base.ocr
    engine = _choice(
        "Reader",
        OCR_ENGINES,
        ocr.engine,
        "engine",
        "The plate model is trained on number plates and is best on small or blurred ones. "
        "EasyOCR is a general text reader that copes better with two-line plates. Both uses "
        "the plate model first, and EasyOCR when it is unsure.",
    )
    fallback_confidence = ocr.fallback_confidence
    if engine == "hybrid":
        fallback_confidence = st.slider(
            "Use EasyOCR when the plate model is below",
            0.0,
            1.0,
            float(ocr.fallback_confidence),
            0.05,
            key=PREFIX + "fallback_confidence",
            help="A plate model reading is kept when it is a valid plate number with at least "
            "this confidence. Otherwise EasyOCR reads the plate.",
        )
    min_confidence = st.slider(
        "Ignore text below this confidence",
        0.0,
        0.9,
        float(ocr.min_confidence),
        0.05,
        key=PREFIX + "min_confidence",
        help="Text the reader is less sure of is dropped before the plate is assembled.",
    )
    chosen = replace(
        ocr, engine=engine, fallback_confidence=fallback_confidence, min_confidence=min_confidence
    )
    if engine == "plate":
        st.caption("EasyOCR's own settings are hidden, because the plate model reads alone.")
        return replace(base, ocr=chosen)
    st.caption("EasyOCR's own settings:")
    return replace(base, ocr=easyocr_settings(chosen))


def easyocr_settings(ocr: OcrConfig) -> OcrConfig:
    decoder = _choice(
        "Decoder",
        OCR_DECODERS,
        ocr.decoder,
        "decoder",
        "Beam search weighs several candidate readings instead of the single likeliest "
        "character each time. Slower.",
    )
    beam_width = ocr.beam_width
    if decoder == "beamsearch":
        beam_width = st.slider(
            "Beam width",
            2,
            20,
            ocr.beam_width,
            1,
            key=PREFIX + "beam_width",
            help="Candidates kept at each step.",
        )
    text_threshold = st.slider(
        "Text confidence",
        0.1,
        0.95,
        float(ocr.text_threshold),
        0.05,
        key=PREFIX + "text_threshold",
        help="How sure EasyOCR's text finder must be that a region is text. Lower it if "
        "characters are missed.",
    )
    low_text = st.slider(
        "Character edge sensitivity",
        0.1,
        0.9,
        float(ocr.low_text),
        0.05,
        key=PREFIX + "low_text",
        help="EasyOCR's low_text. Lower values include fainter character edges.",
    )
    link_threshold = st.slider(
        "Character linking",
        0.1,
        0.9,
        float(ocr.link_threshold),
        0.05,
        key=PREFIX + "link_threshold",
        help="EasyOCR's link_threshold. Lower values join characters into one word more "
        "readily; raise it if two rows of a plate merge.",
    )
    mag_ratio = st.slider(
        "Magnification",
        0.5,
        3.0,
        float(ocr.mag_ratio),
        0.1,
        key=PREFIX + "mag_ratio",
        help="Enlarges the image inside EasyOCR's text finder. Above 1 helps small text.",
    )
    min_size = st.slider(
        "Smallest text (pixels)",
        5,
        60,
        ocr.min_size,
        1,
        key=PREFIX + "min_size",
        help="Text regions smaller than this are ignored.",
    )
    contrast_ths = st.slider(
        "Retry low-contrast text below",
        0.0,
        1.0,
        float(ocr.contrast_ths),
        0.05,
        key=PREFIX + "contrast_ths",
        help="Text EasyOCR reads with less confidence than this is read again with more contrast.",
    )
    adjust_contrast = st.slider(
        "Contrast for the retry",
        0.0,
        1.0,
        float(ocr.adjust_contrast),
        0.05,
        key=PREFIX + "adjust_contrast",
    )
    rotations = st.pills(
        "Also try rotated",
        OCR_ROTATIONS,
        selection_mode="multi",
        default=list(ocr.rotation_info),
        format_func=lambda angle: f"{angle}°",
        key=PREFIX + "rotation",
        help="Reads the text turned by these angles too, and keeps the most confident.",
    )
    return replace(
        ocr,
        decoder=decoder,
        beam_width=beam_width,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
        mag_ratio=mag_ratio,
        min_size=min_size,
        contrast_ths=contrast_ths,
        adjust_contrast=adjust_contrast,
        rotation_info=tuple(sorted(rotations or ())),
    )


def advanced_options(base: AppConfig) -> AppConfig:
    """Show the advanced options in the sidebar and return the configuration they describe."""
    sidebar = st.sidebar
    sidebar.subheader("Advanced options")
    sidebar.caption(
        "Change how plates are cleaned up, found and read. Use “Try settings on one frame” "
        "on the Run tab to see the effect before running a whole video."
    )
    config = base
    with sidebar.expander("Image filters", icon=":material/photo_filter:"):
        config = image_filters(config)
    with sidebar.expander("Plate detector", icon=":material/crop_free:"):
        config = detector_settings(config)
    with sidebar.expander("Text reader", icon=":material/text_fields:"):
        config = reader_settings(config)

    changed = changed_settings(base, config)
    if changed:
        sidebar.caption(f"{changed} advanced setting{'s' if changed != 1 else ''} changed.")
    with sidebar.container(horizontal=True):
        st.button(
            "Reset",
            icon=":material/restart_alt:",
            on_click=reset_advanced,
            disabled=not changed,
            key="advanced_reset",
            help="Return the advanced options to configs/default.yaml.",
        )
        st.download_button(
            "Save settings",
            dump_config(config),
            file_name="vrpr-settings.yaml",
            mime="application/x-yaml",
            icon=":material/download:",
            key="advanced_download",
            on_click="ignore",
            help="Download every setting as YAML, for vrpr run --config.",
        )
    return config
