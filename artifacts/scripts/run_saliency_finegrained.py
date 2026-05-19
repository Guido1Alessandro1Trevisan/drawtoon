"""StaticSaliencyFineGrained gutter detector for ep25 of the-mafia-nanny_manwa.

Algorithm:
  1. Stitch all chapter pages into a single tall scroll (target_width wide).
  2. Compute per-pixel saliency with cv2.saliency.StaticSaliencyFineGrained.
  3. Row-mean -> 1-D saliency profile, smooth with sigma=12 Gaussian.
  4. Sweep low percentile thresholds {12, 18, 25, 35}; pick lowest p that
     yields 25-50 candidate cuts after run-length + min-segment filters.
  5. Build segments via _segment_slices_for_range against the global cuts,
     pack into sheets with write_sheets, and render a red/blue cuts overlay.

Run (the contrib build is REQUIRED for cv2.saliency):
    uv run --active --with opencv-contrib-python --with numpy --with pillow \
        --with boto3 --with scipy \
        python artifacts/scripts/run_saliency_finegrained.py
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
from scipy.ndimage import gaussian_filter1d

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

DETECTOR_LABEL = "saliency_finegrained"
PERCENTILES = (12, 18, 25, 35)
SIGMA = 12
MIN_BAND_HEIGHT = 12  # contiguous gutter-row run must be at least this many px
MIN_SEGMENT_HEIGHT = max(96, MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX)
DARK_STRETCH_PAGES = (7, 23)  # 1-indexed pages 7..23 are the dark-fire stretch
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


def stitch_pages(keys: list[str]) -> tuple[Image.Image, list[tuple[int, int]], int]:
    images: list[Image.Image] = []
    target_width = 0
    for key in keys:
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        images.append(im)
        target_width = max(target_width, im.width)
    normalized: list[Image.Image] = []
    for im in images:
        if im.width != target_width:
            scaled_h = int(round(im.height * target_width / im.width))
            resized = im.resize((target_width, scaled_h), Image.Resampling.LANCZOS)
            im.close()
            normalized.append(resized)
        else:
            normalized.append(im)
    spans: list[tuple[int, int]] = []
    canvas_height = 0
    for im in normalized:
        spans.append((canvas_height, canvas_height + im.height))
        canvas_height += im.height
    scroll = Image.new("RGB", (target_width, canvas_height), "white")
    y = 0
    for im in normalized:
        scroll.paste(im, (0, y))
        y += im.height
        im.close()
    return scroll, spans, target_width


def compute_smoothed_saliency(scroll: Image.Image) -> np.ndarray:
    bgr = cv2.cvtColor(np.array(scroll), cv2.COLOR_RGB2BGR)
    sal = cv2.saliency.StaticSaliencyFineGrained_create()
    ok, S = sal.computeSaliency(bgr)
    if not ok or S is None:
        raise RuntimeError("StaticSaliencyFineGrained.computeSaliency failed")
    S = S.astype(np.float32)
    row_sal = S.mean(axis=1)
    return gaussian_filter1d(row_sal, sigma=SIGMA).astype(np.float32)


def runs_to_cuts(
    is_gutter: np.ndarray,
    *,
    height: int,
    min_band_height: int,
    min_segment_height: int,
) -> list[int]:
    cuts: list[int] = []
    flags = is_gutter.astype(bool)
    y = 0
    while y < height:
        if not flags[y]:
            y += 1
            continue
        start = y
        while y < height and flags[y]:
            y += 1
        end = y
        if end - start < min_band_height:
            continue
        midpoint = (start + end) // 2
        if midpoint < min_segment_height or height - midpoint < min_segment_height:
            continue
        cuts.append(midpoint)
    deduped: list[int] = []
    for cut in cuts:
        if not deduped or cut - deduped[-1] >= min_segment_height:
            deduped.append(cut)
    return deduped


def choose_cuts(sm: np.ndarray, height: int) -> tuple[int, list[int], dict[int, int]]:
    counts: dict[int, int] = {}
    chosen_p: int | None = None
    chosen_cuts: list[int] = []
    for p in PERCENTILES:
        thr = float(np.percentile(sm, p))
        is_gutter = sm < thr
        cuts = runs_to_cuts(
            is_gutter,
            height=height,
            min_band_height=MIN_BAND_HEIGHT,
            min_segment_height=MIN_SEGMENT_HEIGHT,
        )
        counts[p] = len(cuts)
        if chosen_p is None and 25 <= len(cuts) <= 50:
            chosen_p = p
            chosen_cuts = cuts
    if chosen_p is None:
        # Fall back: pick the percentile whose count is closest to the [25, 50] band.
        def distance(p: int) -> int:
            c = counts[p]
            if c < 25:
                return 25 - c
            if c > 50:
                return c - 50
            return 0
        chosen_p = min(PERCENTILES, key=distance)
        thr = float(np.percentile(sm, chosen_p))
        chosen_cuts = runs_to_cuts(
            sm < thr,
            height=height,
            min_band_height=MIN_BAND_HEIGHT,
            min_segment_height=MIN_SEGMENT_HEIGHT,
        )
    return chosen_p, chosen_cuts, counts


def assign_global_y(pages: list[dict[str, Any]], spans: list[tuple[int, int]], target_width: int) -> int:
    global_height = 0
    for page, (y0, y1) in zip(pages, spans):
        # Original width unknown without re-loading; recover height via span and assume
        # rendered width == target_width (already normalized during stitch).
        page["accepted_page_index"] = int(page.get("page_index") or 0)
        page["width"] = int(target_width)
        page["height"] = int(y1 - y0)
        page["global_y_start"] = int(y0)
        page["global_y_end"] = int(y1)
        global_height = max(global_height, int(y1))
    return global_height


def build_segments_from_global_cuts(
    pages: list[dict[str, Any]],
    *,
    global_cuts: list[int],
    target_width: int,
    total_height: int,
) -> list[dict[str, Any]]:
    cuts = sorted({0, int(total_height), *(int(c) for c in global_cuts)})
    segments: list[dict[str, Any]] = []
    for start_y, end_y in zip(cuts, cuts[1:]):
        source_height = int(end_y - start_y)
        if source_height < MIN_SEGMENT_HEIGHT:
            continue
        slices, rendered_height = _segment_slices_for_range(
            pages,
            start_y=int(start_y),
            end_y=int(end_y),
            target_width=int(target_width),
        )
        if not slices:
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
                "source_slices": slices,
                "page_indices": sorted({int(item["page_index"]) for item in slices}),
                "page_span": [int(slices[0]["page_index"]), int(slices[-1]["page_index"])],
            }
        )
    return segments


def render_overlay(
    scroll: Image.Image,
    spans: list[tuple[int, int]],
    cuts: list[int],
    label: str,
    percentile: int,
) -> Image.Image:
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
    for idx, (_, y_end) in enumerate(spans[:-1]):
        y = int(round(y_end * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=PAGE_BOUNDARY_COLOR, width=1)
        draw.text((2, y - 13), f"p{idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    for idx, cut in enumerate(cuts):
        y = int(round(cut * scale))
        draw.line([(0, y), (overlay.width, y)], fill=CUT_COLOR, width=2)
        draw.text((overlay.width - 26, y + 2), str(idx), fill=CUT_COLOR, font=font)
    header_h = 36
    bar = Image.new("RGB", (overlay.width, overlay.height + header_h), LABEL_BG)
    bar.paste(overlay, (0, header_h))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text(
        (6, 4),
        f"{label}  cuts={len(cuts)}  p={percentile}",
        fill=LABEL_FG,
        font=big_font,
    )
    return bar


def encode_jpeg(image: Image.Image, quality: int = 86) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def dark_stretch_split(
    cuts: list[int], spans: list[tuple[int, int]]
) -> dict[str, Any]:
    if len(spans) < DARK_STRETCH_PAGES[1]:
        return {"checked": False}
    start_y = spans[DARK_STRETCH_PAGES[0] - 1][0]
    end_y = spans[DARK_STRETCH_PAGES[1] - 1][1]
    inside = [c for c in cuts if start_y <= c <= end_y]
    return {
        "checked": True,
        "page_start": DARK_STRETCH_PAGES[0],
        "page_end": DARK_STRETCH_PAGES[1],
        "global_y_start": int(start_y),
        "global_y_end": int(end_y),
        "cut_count_inside": int(len(inside)),
        "split": bool(inside),
    }


def main() -> None:
    if not hasattr(cv2, "saliency"):
        raise SystemExit(
            "cv2.saliency missing. Reinstall via:\n"
            "  uv run --active --with opencv-contrib-python --with numpy --with pillow "
            "--with boto3 --with scipy python artifacts/scripts/run_saliency_finegrained.py"
        )

    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"pages: {len(keys)}")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)

    scroll, spans, target_width = stitch_pages(keys)
    total_height = scroll.height
    print(f"scroll: {scroll.size} (W x H)")

    assign_global_y(pages, spans, target_width)

    sm = compute_smoothed_saliency(scroll)
    chosen_p, cuts, counts = choose_cuts(sm, total_height)
    print(f"threshold sweep counts: {counts}")
    print(f"chosen percentile: {chosen_p}  cuts: {len(cuts)}")

    stretch_info = dark_stretch_split(cuts, spans)
    print(f"dark stretch (pages 7-23) split: {stretch_info}")

    segments = build_segments_from_global_cuts(
        pages,
        global_cuts=cuts,
        target_width=target_width,
        total_height=total_height,
    )
    if segments:
        heights = [int(s["rendered_height"]) for s in segments]
        print(
            f"segments: {len(segments)}  height min={min(heights)} max={max(heights)} avg={sum(heights)//len(heights)}"
        )
    else:
        print("segments: 0")

    config = {
        "manhwa": {
            "raw_gutter_min_height_px": MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
            "raw_gutter_color_tolerance": MANHWA_RAW_GUTTER_COLOR_TOLERANCE,
            "raw_min_segment_height_px": MIN_SEGMENT_HEIGHT,
            "raw_sheet_padding_px": MANHWA_RAW_SHEET_PADDING_PX,
            "raw_sheet_span_threshold_px": MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
            "strip_jpeg_quality": 92,
        }
    }
    output_prefix = f"{OUTPUT_BASE}/{DETECTOR_LABEL}"
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
    sheet_count = int(sheet_result.get("sheet_count") or 0)
    print(f"sheets written: {sheet_count}")

    manifest = {
        "schema_version": 1,
        "detector": DETECTOR_LABEL,
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": output_prefix,
        "target_width": int(target_width),
        "total_virtual_height": int(total_height),
        "page_count": len(pages),
        "params": {
            "sigma": SIGMA,
            "percentile_sweep": list(PERCENTILES),
            "chosen_percentile": int(chosen_p),
            "percentile_counts": {str(k): int(v) for k, v in counts.items()},
            "min_band_height": MIN_BAND_HEIGHT,
            "min_segment_height": MIN_SEGMENT_HEIGHT,
        },
        "segmentation": {
            "cut_count": len(cuts),
            "global_cuts": [int(c) for c in cuts],
            "segment_count": len(segments),
        },
        "dark_stretch": stretch_info,
        "segment_count": len(segments),
        "sheet_count": sheet_count,
        "segments": segments,
        "sheets": sheet_result.get("sheets") or [],
    }
    manifest_key = f"{output_prefix}/{CHAPTER_RELATIVE}/manifest.json"
    put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")
    print(f"manifest -> {LOCAL_OUTPUT_ROOT / DETECTOR_LABEL / CHAPTER_RELATIVE / 'manifest.json'}")

    overlay_image = render_overlay(scroll, spans, cuts, DETECTOR_LABEL, chosen_p)
    overlay_key = f"{output_prefix}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
    put_bytes(overlay_key, encode_jpeg(overlay_image), "image/jpeg")
    print(f"overlay -> {LOCAL_OUTPUT_ROOT / DETECTOR_LABEL / CHAPTER_RELATIVE / 'cuts_overlay.jpg'}")

    # Verification: every sheet manifest entry must exist on disk.
    missing: list[str] = []
    for sheet in manifest["sheets"]:
        sheet_path = LOCAL_OUTPUT_ROOT / Path(sheet["sheet_key"]).relative_to(OUTPUT_BASE)
        if not sheet_path.exists() or sheet_path.stat().st_size == 0:
            missing.append(str(sheet_path))
    if missing:
        raise SystemExit(f"Missing or empty sheets: {missing}")

    overlay_path = LOCAL_OUTPUT_ROOT / DETECTOR_LABEL / CHAPTER_RELATIVE / "cuts_overlay.jpg"
    if not overlay_path.exists() or overlay_path.stat().st_size == 0:
        raise SystemExit(f"Empty overlay: {overlay_path}")

    if len(cuts) <= 8:
        raise SystemExit(
            f"Cut count {len(cuts)} is not above baseline threshold of 8; tune percentiles."
        )

    max_height = max((int(s["rendered_height"]) for s in segments), default=0)
    print(
        "\n=== FINAL ===\n"
        f"percentile={chosen_p}  cuts={len(cuts)}  segments={len(segments)}  "
        f"max_segment_height={max_height}  dark_stretch_split={stretch_info.get('split')}  "
        f"(cuts_inside={stretch_info.get('cut_count_inside')})"
    )


if __name__ == "__main__":
    main()
