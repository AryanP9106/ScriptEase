"""
preprocess.py
=============
Image preprocessing stage of the ScriptEase pipeline.

    PDF -> PDF Extraction -> [Image Preprocessing] -> Word Segmentation -> ...

Responsibilities of this module:
    1. Deskew scanned/photographed notebook pages.
    2. Remove shadows and uneven illumination.
    3. Denoise while preserving handwriting strokes.
    4. Enhance handwriting contrast (binarization / adaptive thresholding).
    5. Optionally remove notebook ruling (horizontal lines, margin lines).
    6. Save clean, uniform images ready for segmentation.

Design notes:
    - Every step is a standalone function that takes/returns a numpy array
      (OpenCV BGR or single-channel image), so steps can be unit-tested,
      reordered, or swapped independently.
    - `PreprocessConfig` centralizes tunable parameters so the pipeline
      can be adjusted without touching function internals.
    - `preprocess_page()` composes the steps into the full pipeline for a
      single image; `preprocess_folder()` batches over a directory of pages
      (e.g. the `dataset/pages/` folder produced by pdf_extractor.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessConfig:
    """Tunable parameters for the preprocessing pipeline."""

    # -- Deskew --
    deskew_enabled: bool = True
    deskew_angle_limit: float = 15.0  # degrees; ignore corrections beyond this (likely bad estimate)

    # -- Shadow removal / illumination correction --
    shadow_removal_enabled: bool = True
    shadow_blur_kernel: int = 35  # must be odd; larger = smoother illumination estimate

    # -- Denoising --
    denoise_enabled: bool = True
    denoise_strength: int = 7  # h parameter for fastNlMeansDenoising
    denoise_method: str = "nlmeans"  # "nlmeans" (slow, best quality) or "bilateral"/"median" (fast)

    # -- Speed / resizing --
    # Resize the long edge to this many pixels before heavy processing.
    # Big win on time: fastNlMeansDenoising cost scales with pixel count.
    # Set to None to keep original resolution.
    max_long_edge: Optional[int] = 1600

    # -- Handwriting enhancement / binarization --
    binarize_enabled: bool = True
    adaptive_block_size: int = 25  # must be odd
    adaptive_C: int = 15
    sharpen_enabled: bool = True

    # -- Notebook line removal (optional) --
    remove_lines_enabled: bool = False
    line_kernel_length: int = 40  # horizontal structuring element length (px)

    # -- Output --
    output_format: str = "png"
    target_dpi: Optional[int] = None  # if set, resize so long edge matches expected page size

    def validate(self) -> None:
        if self.shadow_blur_kernel % 2 == 0:
            self.shadow_blur_kernel += 1
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1
        if self.adaptive_block_size < 3:
            self.adaptive_block_size = 3


# --------------------------------------------------------------------------- #
# Individual pipeline steps
# --------------------------------------------------------------------------- #

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. No-op if already single channel."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_long_edge(image: np.ndarray, max_long_edge: Optional[int]) -> np.ndarray:
    """
    Downscale image so its longer edge is at most `max_long_edge` pixels.
    No-op if image is already smaller, or if max_long_edge is None.

    This is the single biggest speed lever in the pipeline: denoising and
    morphological ops scale with pixel count, so halving resolution can
    cut total processing time by ~4x.
    """
    if max_long_edge is None:
        return image
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def deskew(image: np.ndarray, angle_limit: float = 15.0) -> np.ndarray:
    """
    Estimate and correct page skew using the minimum-area bounding rectangle
    of foreground (dark) pixels.

    Works on grayscale or BGR input; returns same type as input.
    """
    gray = to_grayscale(image)

    # Invert + threshold so text/ruling becomes foreground (white) on black.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        logger.debug("Not enough foreground pixels to estimate skew; skipping.")
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angle in (-90, 0]; normalize to a signed
    # rotation in (-45, 45] representing how far off-vertical/horizontal we are.
    if angle < -45:
        angle = 90 + angle
    angle = -angle

    if abs(angle) > angle_limit:
        logger.debug("Estimated skew angle %.2f exceeds limit %.2f; skipping.", angle, angle_limit)
        return image

    if abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed image by %.2f degrees.", angle)
    return rotated


def remove_shadows(image: np.ndarray, blur_kernel: int = 35) -> np.ndarray:
    """
    Remove shadows / uneven illumination by estimating the background via
    heavy morphological dilation + blur, then dividing it out.

    Works per-channel for BGR images, or directly for grayscale.
    """
    def _correct_channel(channel: np.ndarray) -> np.ndarray:
        dilated = cv2.dilate(channel, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, blur_kernel)
        diff = 255 - cv2.absdiff(channel, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm

    if image.ndim == 2:
        return _correct_channel(image)

    channels = cv2.split(image)
    corrected = [_correct_channel(c) for c in channels]
    return cv2.merge(corrected)


def denoise(image: np.ndarray, strength: int = 7, method: str = "nlmeans") -> np.ndarray:
    """
    Denoise while preserving edges/strokes.

    method:
        "nlmeans"   - Non-Local Means. Best quality, SLOW (can dominate
                      total pipeline time on large images/batches).
        "bilateral" - Edge-preserving, much faster than NLMeans, good
                      middle ground for handwriting.
        "median"    - Fastest, simplest. Fine once binarization follows.
    """
    if method == "median":
        k = 3 if strength <= 5 else 5
        return cv2.medianBlur(image, k)

    if method == "bilateral":
        return cv2.bilateralFilter(image, d=7, sigmaColor=strength * 8, sigmaSpace=7)

    # default: nlmeans (slow path)
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=strength)
    return cv2.fastNlMeansDenoisingColored(image, h=strength, hColor=strength)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask style sharpening to make handwriting strokes crisper."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def binarize(image: np.ndarray, block_size: int = 25, C: int = 15) -> np.ndarray:
    """
    Adaptive thresholding to produce a clean black-on-white handwriting image.
    Returns a single-channel (grayscale) image.
    """
    gray = to_grayscale(image)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C,
    )
    return thresh


def remove_ruling_lines(binary_image: np.ndarray, line_kernel_length: int = 40) -> np.ndarray:
    """
    Remove horizontal notebook ruling lines from a binarized (black-on-white)
    image while preserving handwriting strokes.

    Expects a single-channel image where text is black (0) on white (255)
    background (i.e. the output of `binarize`).
    """
    inverted = cv2.bitwise_not(binary_image)  # text becomes white on black

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (line_kernel_length, 1)
    )
    detected_lines = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # Dilate slightly so we fully cover anti-aliased line edges.
    detected_lines = cv2.dilate(detected_lines, np.ones((2, 2), np.uint8))

    # Remove detected line pixels from the inverted (text=white) image.
    cleaned_inverted = cv2.subtract(inverted, detected_lines)

    # Close small gaps left in strokes where they crossed a ruling line.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_inverted = cv2.morphologyEx(cleaned_inverted, cv2.MORPH_CLOSE, close_kernel)

    return cv2.bitwise_not(cleaned_inverted)  # back to black-on-white


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def preprocess_page(image: np.ndarray, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single page image.

    Parameters
    ----------
    image : np.ndarray
        BGR image as loaded by cv2.imread.
    config : PreprocessConfig, optional
        Pipeline configuration. Uses defaults if not provided.

    Returns
    -------
    np.ndarray
        Cleaned, single-channel (or BGR, if binarize disabled) image ready
        for segmentation.
    """
    config = config or PreprocessConfig()
    config.validate()

    result = resize_long_edge(image, config.max_long_edge)

    if config.deskew_enabled:
        result = deskew(result, angle_limit=config.deskew_angle_limit)

    if config.shadow_removal_enabled:
        result = remove_shadows(result, blur_kernel=config.shadow_blur_kernel)

    if config.denoise_enabled:
        result = denoise(result, strength=config.denoise_strength, method=config.denoise_method)

    if config.sharpen_enabled:
        result = sharpen(result)

    if config.binarize_enabled:
        result = binarize(result, block_size=config.adaptive_block_size, C=config.adaptive_C)

        if config.remove_lines_enabled:
            result = remove_ruling_lines(result, line_kernel_length=config.line_kernel_length)

    return result


def preprocess_folder(
    input_dir: Path,
    output_dir: Path,
    config: Optional[PreprocessConfig] = None,
    pattern: str = "*.png",
) -> list[Path]:
    """
    Batch-process all page images in `input_dir` and write cleaned images
    to `output_dir`, preserving filenames.

    Parameters
    ----------
    input_dir : Path
        Directory containing extracted page images (e.g. dataset/pages/).
    output_dir : Path
        Directory to write processed images to. Created if missing.
    config : PreprocessConfig, optional
        Pipeline configuration.
    pattern : str
        Glob pattern to select input files.

    Returns
    -------
    list[Path]
        Paths of the written output images, in sorted order.
    """
    config = config or PreprocessConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob(pattern))
    if not input_paths:
        logger.warning("No files matching '%s' found in %s", pattern, input_dir)
        return []

    written: list[Path] = []
    for i, path in enumerate(input_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to read image: %s", path)
            continue

        try:
            processed = preprocess_page(image, config)
        except Exception:
            logger.exception("Preprocessing failed for %s; skipping.", path)
            continue

        out_path = output_dir / f"{path.stem}.{config.output_format}"
        cv2.imwrite(str(out_path), processed)
        written.append(out_path)
        logger.info("[%d/%d] Processed %s -> %s", i, len(input_paths), path.name, out_path.name)

    logger.info("Preprocessing complete: %d/%d pages written to %s", len(written), len(input_paths), output_dir)
    return written


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ScriptEase image preprocessing stage.")
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/pages"),
        help="Folder containing extracted page images (default: dataset/pages)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/processed"),
        help="Folder to write cleaned images to (default: dataset/processed)",
    )
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskewing.")
    parser.add_argument("--no-shadow-removal", action="store_true", help="Disable shadow removal.")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising.")
    parser.add_argument("--no-binarize", action="store_true", help="Disable binarization.")
    parser.add_argument(
        "--remove-lines", action="store_true",
        help="Enable notebook ruling-line removal (requires binarization).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Speed preset: resize to 1600px long edge + bilateral denoise "
             "instead of NLMeans. ~5-10x faster with a small quality tradeoff.",
    )
    parser.add_argument(
        "--max-long-edge", type=int, default=None,
        help="Resize long edge to this many pixels before processing (speed lever).",
    )
    args = parser.parse_args()

    cfg = PreprocessConfig(
        deskew_enabled=not args.no_deskew,
        shadow_removal_enabled=not args.no_shadow_removal,
        denoise_enabled=not args.no_denoise,
        binarize_enabled=not args.no_binarize,
        remove_lines_enabled=args.remove_lines,
    )

    if args.fast:
        cfg.max_long_edge = 1600
        cfg.denoise_method = "bilateral"
    if args.max_long_edge:
        cfg.max_long_edge = args.max_long_edge

    preprocess_folder(args.input, args.output, cfg)
