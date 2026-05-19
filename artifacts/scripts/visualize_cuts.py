"""Render the full stitched ep25 scroll with cut overlays from each detector.

Outputs:
  s3://drawtoon/datasets/pages/artifacts/gutter_comparison/<detector>/cuts_overlay.jpg
  s3://drawtoon/datasets/pages/artifacts/gutter_comparison/cuts_overlay_side_by_side.jpg

Run:
    uv run --active --with opencv-python-headless --with numpy --with pillow \
        --with boto3 python artifacts/scripts/visualize_cuts.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Callable

import boto3
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
    build_pages_for_chapter,
    detect_cuts_canny_lap,
    detect_cuts_kcc,
    list_chapter_pages,
)
from src.manhwa_raw_sheets import (  # noqa: E402
    MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
    MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
    detect_gutter_cuts as baseline_detect_cuts,
)

s3 = boto3.client("s3", region_name=REGION)

THUMB_WIDTH = 360  # pixels for the rendered overlay column
PAGE_BOUNDARY_COLOR = (60, 140, 240)  # blue
CUT_COLOR = (240, 60, 60)  # red
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)


def get_bytes(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    local_path = LOCAL_OUTPUT_ROOT / Path(key).relative_to(OUTPUT_BASE)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


def stitch_pages(keys: list[str]) -> tuple[Image.Image, list[tuple[int, int]]]:
    """Return the full-size stitched scroll and per-page (y_start, y_end) spans."""
    images: list[Image.Image] = []
    spans: list[tuple[int, int]] = []
    target_width = 0
    y = 0
    for key in keys:
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        images.append(im)
        target_width = max(target_width, im.width)
    canvas_height = 0
    for im in images:
        if im.width != target_width:
            scaled_h = int(round(im.height * target_width / im.width))
            im_resized = im.resize((target_width, scaled_h), Image.Resampling.LANCZOS)
            im.close()
            images[images.index(im)] = im_resized
            im = im_resized
        spans.append((canvas_height, canvas_height + im.height))
        canvas_height += im.height
    scroll = Image.new("RGB", (target_width, canvas_height), "white")
    y = 0
    for im in images:
        scroll.paste(im, (0, y))
        y += im.height
        im.close()
    return scroll, spans


def collect_cuts(detector: Callable[[Image.Image, dict[str, Any]], list[int]], pages: list[dict[str, Any]]) -> list[int]:
    config = {
        "manhwa": {
            "raw_gutter_min_height_px": MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
            "raw_min_segment_height_px": MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
        }
    }
    cuts: list[int] = []
    global_y = 0
    for page in pages:
        data = get_bytes(str(page["source_key"]))
        with Image.open(io.BytesIO(data)) as im:
            image = im.convert("RGB")
            for local_cut in detector(image, config=config):
                cuts.append(global_y + local_cut)
            global_y += image.height
    return cuts


def render_overlay(scroll: Image.Image, spans: list[tuple[int, int]], cuts: list[int], label: str) -> Image.Image:
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
    # Page boundaries (blue, dashed-ish: short tick marks at the edges).
    for idx, (_, y_end) in enumerate(spans[:-1]):
        y = int(round(y_end * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=PAGE_BOUNDARY_COLOR, width=1)
        draw.text((2, y - 13), f"p{idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    # Detected cuts (red, solid).
    for idx, cut in enumerate(cuts):
        y = int(round(cut * scale))
        draw.line([(0, y), (overlay.width, y)], fill=CUT_COLOR, width=2)
        draw.text((overlay.width - 26, y + 2), str(idx), fill=CUT_COLOR, font=font)
    # Header bar with label + stats.
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
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"pages: {len(keys)}")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)
    scroll, spans = stitch_pages(keys)
    print(f"scroll: {scroll.size}, height={scroll.height}px")
    detectors = [
        ("baseline_white_black", baseline_detect_cuts),
        ("kcc_find_edges", detect_cuts_kcc),
        ("canny_laplacian", detect_cuts_canny_lap),
    ]
    panels: list[Image.Image] = []
    for label, fn in detectors:
        cuts = collect_cuts(fn, pages)
        print(f"  {label}: {len(cuts)} cuts")
        panel = render_overlay(scroll, spans, cuts, label)
        key = f"{OUTPUT_BASE}/{label}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
        put_bytes(key, encode_jpeg(panel), "image/jpeg")
        print(f"    -> {LOCAL_OUTPUT_ROOT / label / CHAPTER_RELATIVE / 'cuts_overlay.jpg'}")
        panels.append(panel)
    # Side-by-side combined panel for easy comparison.
    combined_width = sum(p.width for p in panels) + 8 * (len(panels) - 1)
    combined_height = max(p.height for p in panels)
    combined = Image.new("RGB", (combined_width, combined_height), (20, 20, 20))
    x = 0
    for panel in panels:
        combined.paste(panel, (x, 0))
        x += panel.width + 8
    side_by_side_key = f"{OUTPUT_BASE}/cuts_overlay_side_by_side.jpg"
    put_bytes(side_by_side_key, encode_jpeg(combined, quality=82), "image/jpeg")
    print(f"side-by-side -> {LOCAL_OUTPUT_ROOT / 'cuts_overlay_side_by_side.jpg'}")


if __name__ == "__main__":
    main()
