"""Offline gutter-detection bake-off.

Pulls the raw pages for a manhwa chapter from S3, runs two alternative gutter
detectors (KCC FIND_EDGES port and a Canny + Laplacian hybrid), builds the same
2x2 sheets via the existing manhwa_raw_sheets packer, and uploads the resulting
sheets + manifest under s3://<bucket>/datasets/pages/artifacts/gutter_comparison/.

Run:
    uv run --active --with opencv-python-headless --with scikit-image \
        --with numpy --with pillow --with boto3 \
        python workflows/manga_filter/scripts/compare_gutters.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import boto3
import cv2
import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workflows" / "manga_filter"))
from src.manhwa_raw_sheets import (  # noqa: E402
    MANHWA_RAW_GUTTER_COLOR_TOLERANCE,
    MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
    MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
    MANHWA_RAW_SHEET_PADDING_PX,
    MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
    pack_segments_into_sheets,
    write_sheets,
)

BUCKET = "drawtoon"
CHAPTER_RELATIVE = "the-mafia-nanny_manwa/episode-000025-episode-25"
INPUT_PREFIX = "datasets/pages/single"
OUTPUT_BASE = "datasets/pages/artifacts/gutter_comparison"  # unused; kept for back-compat
LOCAL_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "gutter_comparison"
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)


def list_chapter_pages(bucket: str, prefix: str, chapter_relative: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    page_prefix = f"{prefix.rstrip('/')}/{chapter_relative.split('/')[0]}/"
    pat = re.compile(rf"^{re.escape(prefix.rstrip('/'))}/{re.escape(chapter_relative)}__page-\d+\.jpg$")
    for page in paginator.paginate(Bucket=bucket, Prefix=page_prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if pat.match(key):
                keys.append(key)
    keys.sort()
    return keys


def get_bytes(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    """Write bytes to the local artifacts tree.

    `key` is interpreted as a relative path under the local artifacts root, so
    callers can keep passing the S3-style key and we still mirror the layout on
    disk for visual review in the editor.
    """
    local_path = LOCAL_OUTPUT_ROOT / Path(key).relative_to(OUTPUT_BASE)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


def make_uri(key: str) -> str:
    local_path = LOCAL_OUTPUT_ROOT / Path(key).relative_to(OUTPUT_BASE)
    return f"file://{local_path}"


# ---------------------------------------------------------------------------
# Detector A: KCC FIND_EDGES port (kindlecomicconverter/comic2panel.py:108-147)
#
# - Convert page to grayscale + Image.filter(FIND_EDGES).
# - Threshold edges at > 6 -> binary "content" mask.
# - Slide a window of height v_pad = width/80 down the image; a window is a
#   gutter when zero pixels survive the threshold inside the window.
# - We translate the KCC strip-loop into per-row "is_gutter" flags so the
#   existing build_segments cut-merging logic can reuse them downstream.
# ---------------------------------------------------------------------------


def detect_cuts_kcc(image: Image.Image, *, config: dict[str, Any]) -> list[int]:
    manhwa_cfg = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    min_band_height = max(4, int(manhwa_cfg.get("raw_gutter_min_height_px") or MANHWA_RAW_GUTTER_MIN_HEIGHT_PX))
    min_segment_height = max(16, int(manhwa_cfg.get("raw_min_segment_height_px") or MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX))
    width, height = image.size
    if width <= 0 or height <= min_segment_height * 2:
        return []
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges)
    # KCC's threshold > 6; ignore a 5% horizontal margin to skip page-border edges.
    h_pad = max(1, width // 20)
    inner = edges_arr[:, h_pad : width - h_pad]
    row_max = inner.max(axis=1) if inner.size else np.zeros(height, dtype=np.uint8)
    is_gutter = row_max <= 6
    return _runs_to_cuts(is_gutter, min_band_height=min_band_height, min_segment_height=min_segment_height, height=height)


# ---------------------------------------------------------------------------
# Detector B: Canny + Laplacian-std hybrid
#
# - Canny edge map, sum per row -> activity profile E[y].
# - Smooth profile with a small Gaussian kernel; estimate panel-row energy as
#   the median of rows above the 30th percentile (robust to gutter zeros).
# - Combine with per-row std and per-row Laplacian variance, AND together so
#   colored flat bands (low std, low laplacian) trigger only when edge density
#   is also low.
# ---------------------------------------------------------------------------


def detect_cuts_canny_lap(image: Image.Image, *, config: dict[str, Any]) -> list[int]:
    manhwa_cfg = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    min_band_height = max(4, int(manhwa_cfg.get("raw_gutter_min_height_px") or MANHWA_RAW_GUTTER_MIN_HEIGHT_PX))
    min_segment_height = max(16, int(manhwa_cfg.get("raw_min_segment_height_px") or MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX))
    width, height = image.size
    if width <= 0 or height <= min_segment_height * 2:
        return []
    gray = np.array(image.convert("L"))
    h_pad = max(1, width // 20)
    inner = gray[:, h_pad : width - h_pad]
    edges = cv2.Canny(inner, 50, 150)
    row_edge = edges.sum(axis=1).astype(np.float32)
    smoothed = cv2.GaussianBlur(row_edge.reshape(-1, 1), (1, 9), 0).ravel()
    nonzero = smoothed[smoothed > np.percentile(smoothed, 30)]
    if nonzero.size == 0:
        med_E = float(np.median(smoothed) or 1.0)
    else:
        med_E = float(np.median(nonzero) or 1.0)
    edge_thresh = max(1.0, 0.05 * med_E)
    row_std = inner.std(axis=1)
    lap = cv2.Laplacian(inner, cv2.CV_32F, ksize=3)
    lap_var = lap.var(axis=1)
    is_gutter = (smoothed < edge_thresh) & (row_std < 6.0) & (lap_var < 12.0)
    return _runs_to_cuts(is_gutter, min_band_height=min_band_height, min_segment_height=min_segment_height, height=height)


def _runs_to_cuts(is_gutter: np.ndarray, *, min_band_height: int, min_segment_height: int, height: int) -> list[int]:
    cuts: list[int] = []
    y = 0
    n = int(height)
    flags = is_gutter.astype(bool)
    while y < n:
        if not flags[y]:
            y += 1
            continue
        start = y
        while y < n and flags[y]:
            y += 1
        end = y
        if end - start < min_band_height:
            continue
        midpoint = (start + end) // 2
        if midpoint < min_segment_height or n - midpoint < min_segment_height:
            continue
        cuts.append(midpoint)
    deduped: list[int] = []
    for cut in cuts:
        if not deduped or cut - deduped[-1] >= min_segment_height:
            deduped.append(cut)
    return deduped


# ---------------------------------------------------------------------------
# build_segments mirrors src.manhwa_raw_sheets but lets us inject the detector.
# ---------------------------------------------------------------------------


def _segment_slices_for_range(
    pages: list[dict[str, Any]],
    *,
    start_y: int,
    end_y: int,
    target_width: int,
) -> tuple[list[dict[str, Any]], int]:
    slices: list[dict[str, Any]] = []
    rendered_height = 0
    for page in pages:
        page_start = int(page.get("global_y_start") or 0)
        page_end = int(page.get("global_y_end") or 0)
        overlap_start = max(int(start_y), page_start)
        overlap_end = min(int(end_y), page_end)
        if overlap_end <= overlap_start:
            continue
        source_y_start = overlap_start - page_start
        source_y_end = overlap_end - page_start
        source_width = max(1, int(page.get("width") or target_width))
        source_height = max(1, int(source_y_end - source_y_start))
        rendered_slice_height = max(1, int(round(source_height * float(target_width) / float(source_width))))
        slices.append(
            {
                "page_index": int(page.get("page_index") or 0),
                "accepted_page_index": int(page.get("accepted_page_index") or 0),
                "relative_path": str(page.get("relative_path") or ""),
                "source_key": str(page.get("source_key") or ""),
                "source_x_start": 0,
                "source_x_end": source_width,
                "source_y_start": int(source_y_start),
                "source_y_end": int(source_y_end),
                "source_width": source_width,
                "source_height": source_height,
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_slice_height),
            }
        )
        rendered_height += rendered_slice_height
    return slices, rendered_height


def build_segments_with_detector(
    *,
    pages: list[dict[str, Any]],
    config: dict[str, Any],
    get_bytes_fn: Callable[[str], bytes],
    detect_cuts: Callable[[Image.Image, dict[str, Any]], list[int]],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    if not pages:
        return [], 0, {"total_virtual_height": 0, "target_width": 0, "detected_gutter_cuts": 0}
    target_width = max(1, max(int(page.get("width") or 0) for page in pages))
    if target_width <= 1:
        target_width = 720
    cuts: set[int] = {0}
    global_y = 0
    detected_cuts = 0
    for accepted_index, page in enumerate(pages):
        source_key = str(page.get("source_key") or "")
        data = get_bytes_fn(source_key)
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            page["accepted_page_index"] = accepted_index
            page["width"] = int(width)
            page["height"] = int(height)
            page["global_y_start"] = int(global_y)
            page["global_y_end"] = int(global_y + height)
            for local_cut in detect_cuts(image, config=config):
                cuts.add(int(global_y + local_cut))
                detected_cuts += 1
            global_y += height
    cuts.add(int(global_y))
    sorted_cuts = sorted(cuts)
    min_segment_height = max(
        16,
        int(
            (config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}).get("raw_min_segment_height_px")
            or MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX
        ),
    )
    segments: list[dict[str, Any]] = []
    for start_y, end_y in zip(sorted_cuts, sorted_cuts[1:]):
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
        segment_index = len(segments)
        segments.append(
            {
                "segment_id": f"segment_{segment_index:04d}",
                "segment_index": segment_index,
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
    stats = {
        "total_virtual_height": int(global_y),
        "target_width": int(target_width),
        "detected_gutter_cuts": int(detected_cuts),
        "cut_count": len(sorted_cuts),
    }
    return segments, int(target_width), stats


def build_pages_for_chapter(keys: list[str], input_prefix: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for idx, key in enumerate(keys):
        relative_path = key[len(input_prefix) + 1 :] if key.startswith(input_prefix + "/") else key
        pages.append(
            {
                "page_index": idx,
                "relative_path": relative_path,
                "source_key": key,
            }
        )
    return pages


def run_detector(label: str, detector: Callable[..., list[int]]) -> dict[str, Any]:
    print(f"\n=== Running detector: {label} ===")
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"  pages found: {len(keys)}")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)
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
    segments, target_width, segmentation = build_segments_with_detector(
        pages=pages,
        config=config,
        get_bytes_fn=get_bytes,
        detect_cuts=detector,
    )
    print(f"  segments: {len(segments)}, detected_gutter_cuts: {segmentation['detected_gutter_cuts']}")
    if segments:
        heights = [int(s.get("rendered_height") or 0) for s in segments]
        print(f"  segment heights: min={min(heights)} max={max(heights)} avg={sum(heights)//len(heights)}")
    output_prefix = f"{OUTPUT_BASE}/{label}"
    sheet_result = write_sheets(
        output_prefix=output_prefix,
        chapter_key=CHAPTER_RELATIVE,
        segments=segments,
        target_width=target_width,
        config=config,
        get_bytes=get_bytes,
        put_bytes=put_bytes,
        make_uri=make_uri,
    )
    manifest = {
        "schema_version": 1,
        "detector": label,
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": output_prefix,
        "target_width": int(target_width),
        "segmentation": segmentation,
        "segment_count": len(segments),
        "sheet_count": int(sheet_result.get("sheet_count") or 0),
        "segments": segments,
        "sheets": sheet_result.get("sheets") or [],
        "page_count": len(pages),
    }
    manifest_key = f"{output_prefix}/{CHAPTER_RELATIVE}/manifest.json"
    put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")
    print(f"  wrote {manifest['sheet_count']} sheets -> s3://{BUCKET}/{output_prefix}/{CHAPTER_RELATIVE}/")
    print(f"  manifest: s3://{BUCKET}/{manifest_key}")
    return manifest


def main() -> None:
    summary = {}
    for label, detector in (
        ("kcc_find_edges", detect_cuts_kcc),
        ("canny_laplacian", detect_cuts_canny_lap),
    ):
        manifest = run_detector(label, detector)
        summary[label] = {
            "segment_count": manifest["segment_count"],
            "sheet_count": manifest["sheet_count"],
            "detected_gutter_cuts": manifest["segmentation"]["detected_gutter_cuts"],
            "target_width": manifest["target_width"],
        }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
