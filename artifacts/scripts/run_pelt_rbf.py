"""PELT-rbf changepoint gutter detector for manhwa chapter scrolls.

Builds the full stitched scroll for the test chapter once, computes a 5-D
per-row feature vector (mean luminance, std luminance, Canny edge density,
hue entropy, 1D black-hat valley response), z-scores each column, then runs
ruptures.Pelt(model="rbf") on the full scroll to detect changepoints. Cuts
are converted to segments via the same `_segment_slices_for_range` global_y
bookkeeping used by compare_gutters.build_segments_with_detector.

Run:
    cd /Users/guidotrevisan/Desktop/drawtoon && \
        uv run --active --with opencv-python-headless --with numpy \
        --with pillow --with boto3 --with ruptures --with scipy \
        python artifacts/scripts/run_pelt_rbf.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "artifacts" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "workflows" / "manga_filter"))

from compare_gutters import (  # noqa: E402
    BUCKET,
    CHAPTER_RELATIVE,
    INPUT_PREFIX,
    LOCAL_OUTPUT_ROOT,
    OUTPUT_BASE,
    REGION,
    _segment_slices_for_range,
    build_pages_for_chapter,
    list_chapter_pages,
)
from src.manhwa_raw_sheets import (  # noqa: E402
    MANHWA_RAW_GUTTER_COLOR_TOLERANCE,
    MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
    MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
    MANHWA_RAW_SHEET_PADDING_PX,
    MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
    write_sheets,
)

try:
    import ruptures as rpt
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "ruptures is required for run_pelt_rbf.py. Install with --with ruptures, e.g.:\n"
        "  uv run --active --with opencv-python-headless --with numpy --with pillow "
        "--with boto3 --with ruptures --with scipy python artifacts/scripts/run_pelt_rbf.py\n"
        f"Underlying import error: {exc}"
    ) from exc

try:
    from scipy.ndimage import grey_closing
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "scipy is required for run_pelt_rbf.py. Install with --with scipy.\n"
        f"Underlying import error: {exc}"
    ) from exc


LABEL = "pelt_rbf"
OUTPUT_PREFIX = f"{OUTPUT_BASE}/{LABEL}"
PEN_CANDIDATES: tuple[float, ...] = (6.0, 10.0, 16.0, 25.0, 40.0, 70.0, 120.0, 200.0, 320.0, 500.0)
TARGET_CUT_RANGE: tuple[int, int] = (25, 50)
MIN_PELT_SIZE = 200
PELT_JUMP = 5

THUMB_WIDTH = 360
PAGE_BOUNDARY_COLOR = (60, 140, 240)
CUT_COLOR = (240, 60, 60)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)

s3 = boto3.client("s3", region_name=REGION)


def get_bytes(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    local_path = LOCAL_OUTPUT_ROOT / Path(key).relative_to(OUTPUT_BASE)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


def make_uri(key: str) -> str:
    local_path = LOCAL_OUTPUT_ROOT / Path(key).relative_to(OUTPUT_BASE)
    return f"file://{local_path}"


def _load_page(page: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(get_bytes(str(page["source_key"])))).convert("RGB")


def stitch_chapter(pages: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Load all pages, fix `pages` bookkeeping in place, return RGB scroll array."""
    rgb_rows: list[np.ndarray] = []
    target_width = 0
    images: list[Image.Image] = []
    for page in pages:
        im = _load_page(page)
        target_width = max(target_width, im.width)
        images.append(im)
    if target_width <= 0:
        target_width = 720
    global_y = 0
    for accepted_index, (page, im) in enumerate(zip(pages, images)):
        if im.width != target_width:
            new_h = max(1, int(round(im.height * target_width / im.width)))
            resized = im.resize((target_width, new_h), Image.Resampling.LANCZOS)
            im.close()
            im = resized
        width, height = im.size
        page["accepted_page_index"] = accepted_index
        page["width"] = int(width)
        page["height"] = int(height)
        page["global_y_start"] = int(global_y)
        page["global_y_end"] = int(global_y + height)
        rgb_rows.append(np.asarray(im, dtype=np.uint8))
        im.close()
        global_y += height
    scroll = np.concatenate(rgb_rows, axis=0)
    return scroll, int(target_width)


def _shannon_entropy_hue(rows_hue: np.ndarray, bins: int = 16) -> np.ndarray:
    """Per-row Shannon entropy of hue histogram, vectorized via bincount loop.

    rows_hue: (H, W) uint8 array of OpenCV hue (0..179).
    """
    h, w = rows_hue.shape
    if w == 0 or h == 0:
        return np.zeros(h, dtype=np.float32)
    # Quantize hue into `bins` bins.
    edges = np.linspace(0, 180, bins + 1)
    quantized = np.clip(np.digitize(rows_hue, edges[1:-1], right=False), 0, bins - 1).astype(np.int32)
    # Build histogram per row by accumulating ones with np.add.at.
    hist = np.zeros((h, bins), dtype=np.float32)
    rows = np.repeat(np.arange(h, dtype=np.int32), w)
    np.add.at(hist, (rows, quantized.reshape(-1)), 1.0)
    hist /= max(1, w)
    safe = np.clip(hist, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    return entropy.astype(np.float32)


def build_row_features(scroll_rgb: np.ndarray) -> np.ndarray:
    """Return (H, 5) standardized per-row feature matrix."""
    gray = cv2.cvtColor(scroll_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(scroll_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]

    mean_lum = gray.mean(axis=1).astype(np.float32)
    std_lum = gray.std(axis=1).astype(np.float32)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean(axis=1).astype(np.float32)
    hue_entropy = _shannon_entropy_hue(hue, bins=16)
    closed = grey_closing(mean_lum, size=151)
    blackhat = (closed - mean_lum).astype(np.float32)

    features = np.stack(
        [mean_lum, std_lum, edge_density, hue_entropy, blackhat], axis=1
    )
    # Z-score each column over the full scroll.
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    standardized = (features - mu) / sigma
    return standardized.astype(np.float64)


def run_pelt(features: np.ndarray, pen: float) -> list[int]:
    """Run PELT-rbf, return interior changepoints (exclude final endpoint)."""
    algo = rpt.Pelt(model="rbf", min_size=MIN_PELT_SIZE, jump=PELT_JUMP).fit(features)
    raw = algo.predict(pen=float(pen))
    n_rows = int(features.shape[0])
    interior: list[int] = []
    for cp in raw:
        cp_int = int(cp)
        if 0 < cp_int < n_rows:
            interior.append(cp_int)
    interior.sort()
    return interior


def select_pen(features: np.ndarray) -> tuple[float, list[int], dict[str, list[int]]]:
    """Sweep PEN_CANDIDATES, pick the one inside TARGET_CUT_RANGE (prefer mid)."""
    history: dict[str, list[int]] = {}
    candidates: list[tuple[float, list[int]]] = []
    for pen in PEN_CANDIDATES:
        cuts = run_pelt(features, pen)
        history[str(pen)] = cuts
        candidates.append((pen, cuts))
        print(f"  pen={pen}: {len(cuts)} cuts")
    lo, hi = TARGET_CUT_RANGE
    in_range = [(pen, cuts) for pen, cuts in candidates if lo <= len(cuts) <= hi]
    if in_range:
        # Prefer the candidate closest to the middle of the range.
        target_mid = (lo + hi) / 2.0
        chosen = min(in_range, key=lambda item: abs(len(item[1]) - target_mid))
    else:
        # Fall back: any with > 8 cuts, prefer the one with cut count closest to mid.
        above_baseline = [(pen, cuts) for pen, cuts in candidates if len(cuts) > 8]
        if above_baseline:
            target_mid = (lo + hi) / 2.0
            chosen = min(above_baseline, key=lambda item: abs(len(item[1]) - target_mid))
        else:
            chosen = min(candidates, key=lambda item: item[0])  # lowest pen
    return float(chosen[0]), list(chosen[1]), history


def build_segments_from_cuts(
    *,
    pages: list[dict[str, Any]],
    cuts: list[int],
    target_width: int,
    total_height: int,
    min_segment_height: int,
) -> list[dict[str, Any]]:
    boundary_set = {0, int(total_height)}
    for cut in cuts:
        boundary_set.add(int(cut))
    boundaries = sorted(boundary_set)
    segments: list[dict[str, Any]] = []
    for start_y, end_y in zip(boundaries, boundaries[1:]):
        source_height = int(end_y - start_y)
        if source_height < min_segment_height:
            continue
        source_slices, rendered_height = _segment_slices_for_range(
            pages,
            start_y=int(start_y),
            end_y=int(end_y),
            target_width=int(target_width),
        )
        if not source_slices:
            continue
        seg_index = len(segments)
        segments.append(
            {
                "segment_id": f"segment_{seg_index:04d}",
                "segment_index": seg_index,
                "global_y_start": int(start_y),
                "global_y_end": int(end_y),
                "source_height": source_height,
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_height),
                "source_slices": source_slices,
                "page_indices": sorted({int(item["page_index"]) for item in source_slices}),
                "page_span": [int(source_slices[0]["page_index"]), int(source_slices[-1]["page_index"])],
            }
        )
    return segments


def render_overlay(
    scroll_rgb: np.ndarray,
    spans: list[tuple[int, int]],
    cuts: list[int],
    label: str,
) -> Image.Image:
    scroll = Image.fromarray(scroll_rgb)
    scale = THUMB_WIDTH / scroll.width
    new_size = (THUMB_WIDTH, max(1, int(round(scroll.height * scale))))
    overlay = scroll.resize(new_size, Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=12)
        big_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
        big_font = font
    # Page boundaries: blue dashed tick line + p<n> label at the START of each page.
    for idx, (y_start, _) in enumerate(spans):
        y = int(round(y_start * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=PAGE_BOUNDARY_COLOR, width=1)
        draw.text((2, y + 1), f"p{idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    # Cuts: solid red.
    for idx, cut in enumerate(cuts):
        y = int(round(cut * scale))
        draw.line([(0, y), (overlay.width, y)], fill=CUT_COLOR, width=2)
        draw.text((overlay.width - 26, y + 2), str(idx), fill=CUT_COLOR, font=font)
    header_h = 36
    bar = Image.new("RGB", (overlay.width, overlay.height + header_h), LABEL_BG)
    bar.paste(overlay, (0, header_h))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text((6, 4), f"{label}  cuts={len(cuts)}", fill=LABEL_FG, font=big_font)
    return bar


def encode_jpeg(image: Image.Image, quality: int = 86) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def main() -> None:
    print(f"== Running detector: {LABEL} ==")
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"pages found: {len(keys)}")
    if not keys:
        raise SystemExit(f"No pages found at s3://{BUCKET}/{INPUT_PREFIX}/{CHAPTER_RELATIVE}__page-*.jpg")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)
    print("stitching chapter scroll...")
    scroll_rgb, target_width = stitch_chapter(pages)
    total_height = int(scroll_rgb.shape[0])
    spans: list[tuple[int, int]] = [
        (int(p["global_y_start"]), int(p["global_y_end"])) for p in pages
    ]
    print(f"scroll: {scroll_rgb.shape[1]}x{scroll_rgb.shape[0]} (W x H)")

    print("computing per-row features...")
    features = build_row_features(scroll_rgb)
    print(f"feature matrix: {features.shape}")

    print("sweeping PELT pen values...")
    chosen_pen, cuts, pen_history = select_pen(features)
    print(f"chosen pen: {chosen_pen}  cuts={len(cuts)}")

    config = {
        "manhwa": {
            "raw_gutter_min_height_px": MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
            "raw_gutter_color_tolerance": MANHWA_RAW_GUTTER_COLOR_TOLERANCE,
            "raw_min_segment_height_px": MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
            "raw_sheet_padding_px": MANHWA_RAW_SHEET_PADDING_PX,
            "raw_sheet_span_threshold_px": MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
            "strip_jpeg_quality": 92,
        }
    }
    min_segment_height = max(16, MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX)
    segments = build_segments_from_cuts(
        pages=pages,
        cuts=cuts,
        target_width=target_width,
        total_height=total_height,
        min_segment_height=min_segment_height,
    )
    if segments:
        heights = [int(s["rendered_height"]) for s in segments]
        median = int(sorted(heights)[len(heights) // 2])
        print(f"segments: {len(segments)}  heights min={min(heights)} median={median} max={max(heights)}")
    else:
        heights = []
        median = 0
        print("segments: 0")

    sheet_result = write_sheets(
        output_prefix=OUTPUT_PREFIX,
        chapter_key=CHAPTER_RELATIVE,
        segments=segments,
        target_width=target_width,
        config=config,
        get_bytes=get_bytes,
        put_bytes=put_bytes,
        make_uri=make_uri,
    )

    overlay = render_overlay(scroll_rgb, spans, cuts, LABEL)
    overlay_key = f"{OUTPUT_PREFIX}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
    put_bytes(overlay_key, encode_jpeg(overlay), "image/jpeg")
    overlay_path = LOCAL_OUTPUT_ROOT / LABEL / CHAPTER_RELATIVE / "cuts_overlay.jpg"
    print(f"overlay -> {overlay_path}")

    segmentation_stats = {
        "total_virtual_height": int(total_height),
        "target_width": int(target_width),
        "detected_gutter_cuts": int(len(cuts)),
        "cut_count": int(len(cuts) + 2),  # interior cuts plus 0 + end boundary
        "pen": float(chosen_pen),
        "pen_candidates": [float(p) for p in PEN_CANDIDATES],
        "pen_history_counts": {str(p): len(v) for p, v in pen_history.items()},
        "min_pelt_size": int(MIN_PELT_SIZE),
        "pelt_jump": int(PELT_JUMP),
        "feature_names": ["mean_lum", "std_lum", "edge_density", "hue_entropy", "blackhat"],
        "feature_dim": int(features.shape[1]),
    }
    manifest = {
        "schema_version": 1,
        "detector": LABEL,
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": OUTPUT_PREFIX,
        "target_width": int(target_width),
        "segmentation": segmentation_stats,
        "segment_count": len(segments),
        "sheet_count": int(sheet_result.get("sheet_count") or 0),
        "segments": segments,
        "sheets": sheet_result.get("sheets") or [],
        "page_count": len(pages),
        "cuts_global_y": [int(c) for c in cuts],
        "page_spans": [
            {"page_index": idx, "global_y_start": int(s), "global_y_end": int(e)}
            for idx, (s, e) in enumerate(spans)
        ],
        "cuts_overlay_key": overlay_key,
        "cuts_overlay_uri": make_uri(overlay_key),
    }
    manifest_key = f"{OUTPUT_PREFIX}/{CHAPTER_RELATIVE}/manifest.json"
    put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")
    manifest_path = LOCAL_OUTPUT_ROOT / LABEL / CHAPTER_RELATIVE / "manifest.json"
    print(f"manifest -> {manifest_path}")
    print(f"sheets   -> {sheet_result.get('sheet_count')} written")

    print("\n=== Summary ===")
    print(f"chosen pen     : {chosen_pen}")
    print(f"cut count      : {len(cuts)}")
    if heights:
        print(f"segment heights: min={min(heights)} median={median} max={max(heights)}")
    else:
        print("segment heights: (no segments)")


if __name__ == "__main__":
    main()
