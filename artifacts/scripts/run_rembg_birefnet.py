"""Webtoon gutter detector using rembg foreground mask -> row-wise background fraction.

Premise: a webtoon gutter (white, color, gradient, smoke, fire, dark, anything)
contains *no foreground subject*. Run an alpha-matting / salient-object model
(`birefnet-general`) to get a per-pixel foreground mask. Rows with
overwhelmingly background alpha (>92% of pixels < 20) form gutter bands;
contiguous runs >= 12 rows are cut at their midpoint.

Pages are stitched end-to-end into a single global bg-fraction vector of length
~33,280 (26 pages * ~1280 px each), and `_segment_slices_for_range` from
`compare_gutters` builds the segments.

Run:
    uv run --active --with rembg[cli] --with onnxruntime \\
        --with numpy --with pillow --with boto3 --with scipy \\
        python artifacts/scripts/run_rembg_birefnet.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    _segment_slices_for_range,
    build_pages_for_chapter,
    get_bytes,
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

# --- Detector hyperparameters --------------------------------------------------
BG_FRAC_THRESHOLD = 0.92  # row is gutter if (alpha<20) covers more than this fraction
BG_FRAC_FALLBACK = 0.80   # used if BG_FRAC_THRESHOLD yields zero cuts
ALPHA_BG_LEVEL = 20       # alpha values below this are "background"
MIN_RUN_PX = 12           # contiguous gutter rows required to form a band
MIN_SEGMENT_HEIGHT_PX = 96
HORIZONTAL_MARGIN_PCT = 0.05  # ignore 5% on each side to avoid bleed at borders

# Output locations -- this detector writes to a *parallel* root so we don't trip
# over the `OUTPUT_BASE`-relative slicing inside compare_gutters.put_bytes.
LABEL = "rembg_birefnet"
LOCAL_DETECTOR_ROOT = REPO_ROOT / "artifacts" / "gutter_comparison" / LABEL

THUMB_WIDTH = 360
PAGE_BOUNDARY_COLOR = (60, 140, 240)
CUT_COLOR = (240, 60, 60)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)

# --- rembg session bootstrap ---------------------------------------------------


def make_session(preferred: str = "birefnet-general", fallback: str = "isnet-general-use") -> tuple[Any, str]:
    """Return (session, model_name). Try `preferred`; fall back if the model
    weights cannot be downloaded within ~2 minutes or the session import fails."""
    from rembg import new_session  # local import so file scans don't need rembg

    for name in (preferred, fallback):
        print(f"  rembg: instantiating session '{name}' ...", flush=True)
        t0 = time.time()
        try:
            sess = new_session(name)
        except Exception as exc:
            print(f"    -> failed to load '{name}': {exc!r}", flush=True)
            continue
        elapsed = time.time() - t0
        print(f"    -> session '{name}' ready in {elapsed:.1f}s", flush=True)
        return sess, name
    raise RuntimeError(f"could not load rembg session ({preferred} or {fallback})")


def alpha_for_page(page_bytes: bytes, session: Any) -> np.ndarray:
    """Run rembg and return the alpha channel as H x W uint8."""
    from rembg import remove

    with Image.open(io.BytesIO(page_bytes)) as opened:
        rgb = opened.convert("RGB")
        result = remove(rgb, session=session)
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    alpha = np.array(result.split()[-1], dtype=np.uint8)
    return alpha


# --- Cut extraction ------------------------------------------------------------


def runs_to_cuts(is_gutter: np.ndarray, *, min_run: int, min_segment: int, height: int) -> list[int]:
    cuts: list[int] = []
    flags = is_gutter.astype(bool)
    y = 0
    n = int(height)
    while y < n:
        if not flags[y]:
            y += 1
            continue
        start = y
        while y < n and flags[y]:
            y += 1
        end = y
        if end - start < min_run:
            continue
        mid = (start + end) // 2
        if mid < min_segment or n - mid < min_segment:
            continue
        cuts.append(mid)
    deduped: list[int] = []
    for cut in cuts:
        if not deduped or cut - deduped[-1] >= min_segment:
            deduped.append(cut)
    return deduped


def detect_global_cuts(
    bg_frac_global: np.ndarray,
    *,
    threshold: float,
    min_run: int,
    min_segment: int,
) -> list[int]:
    is_gutter = bg_frac_global > threshold
    return runs_to_cuts(is_gutter, min_run=min_run, min_segment=min_segment, height=bg_frac_global.shape[0])


# --- Segment build -------------------------------------------------------------


def build_segments_from_global_cuts(
    pages: list[dict[str, Any]],
    *,
    global_cuts: list[int],
    target_width: int,
    total_height: int,
    min_segment_height: int,
) -> list[dict[str, Any]]:
    boundaries = sorted({0, int(total_height), *(int(c) for c in global_cuts)})
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
        segment_index = len(segments)
        segments.append(
            {
                "segment_id": f"segment_{segment_index:04d}",
                "segment_index": segment_index,
                "global_y_start": int(start_y),
                "global_y_end": int(end_y),
                "source_height": int(source_height),
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_height),
                "source_slices": source_slices,
                "page_indices": sorted({int(item["page_index"]) for item in source_slices}),
                "page_span": [int(source_slices[0]["page_index"]), int(source_slices[-1]["page_index"])],
            }
        )
    return segments


# --- Output writers ------------------------------------------------------------


def local_put_bytes(key: str, data: bytes, content_type: str) -> None:
    """Write to LOCAL_DETECTOR_ROOT, stripping the leading output_prefix."""
    rel_key = key
    prefix = f"{OUTPUT_BASE}/{LABEL}/"
    if rel_key.startswith(prefix):
        rel_key = rel_key[len(prefix):]
    local_path = LOCAL_DETECTOR_ROOT / rel_key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


def local_make_uri(key: str) -> str:
    rel_key = key
    prefix = f"{OUTPUT_BASE}/{LABEL}/"
    if rel_key.startswith(prefix):
        rel_key = rel_key[len(prefix):]
    return f"file://{LOCAL_DETECTOR_ROOT / rel_key}"


# --- Overlay rendering ---------------------------------------------------------


def stitch_pages(keys: list[str]) -> tuple[Image.Image, list[tuple[int, int]]]:
    images: list[Image.Image] = []
    target_width = 0
    for key in keys:
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        images.append(im)
        target_width = max(target_width, im.width)
    normalized: list[Image.Image] = []
    spans: list[tuple[int, int]] = []
    canvas_height = 0
    for im in images:
        if im.width != target_width:
            new_h = int(round(im.height * target_width / im.width))
            resized = im.resize((target_width, new_h), Image.Resampling.LANCZOS)
            im.close()
            im = resized
        spans.append((canvas_height, canvas_height + im.height))
        canvas_height += im.height
        normalized.append(im)
    scroll = Image.new("RGB", (target_width, canvas_height), "white")
    y = 0
    for im in normalized:
        scroll.paste(im, (0, y))
        y += im.height
        im.close()
    return scroll, spans


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
    for idx, (_, y_end) in enumerate(spans[:-1]):
        y = int(round(y_end * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=PAGE_BOUNDARY_COLOR, width=1)
        draw.text((2, y - 13), f"p{idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    # Final page label, anchored just above the bottom edge.
    last_y = int(round(spans[-1][1] * scale))
    draw.text((2, max(0, last_y - 13)), f"p{len(spans)}", fill=PAGE_BOUNDARY_COLOR, font=font)
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


# --- Main pipeline -------------------------------------------------------------


def main() -> None:
    print(f"=== rembg gutter detector ({LABEL}) ===", flush=True)
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"pages found: {len(keys)}", flush=True)
    if not keys:
        raise RuntimeError("no pages found on S3")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)

    session, model_name = make_session()

    # Load every page once: keep both the raw bytes (for stitching/sheets via
    # write_sheets) and the per-page alpha bg fraction. Pages are normalized to
    # a common target_width before computing bg_frac so the row indices line up
    # exactly with the segment slice math in `_segment_slices_for_range`.
    page_bytes_cache: dict[str, bytes] = {}
    page_metas: list[dict[str, Any]] = []
    target_width = 0

    for page in pages:
        key = str(page["source_key"])
        data = get_bytes(key)
        page_bytes_cache[key] = data
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
        page_metas.append({"key": key, "width": int(width), "height": int(height)})
        target_width = max(target_width, int(width))
    if target_width <= 1:
        target_width = 720

    print(f"target_width={target_width}px; total raw height={sum(m['height'] for m in page_metas)}", flush=True)

    bg_frac_per_page: list[np.ndarray] = []
    global_y = 0
    for idx, (page, meta) in enumerate(zip(pages, page_metas)):
        t0 = time.time()
        alpha = alpha_for_page(page_bytes_cache[meta["key"]], session=session)
        h, w = alpha.shape
        # Trim a 5% horizontal margin (mirrors the other detectors) before
        # computing bg fraction so dark borders don't pretend to be content.
        h_pad = max(1, int(round(w * HORIZONTAL_MARGIN_PCT)))
        inner = alpha[:, h_pad : w - h_pad]
        is_bg = inner < ALPHA_BG_LEVEL
        bg_frac_row = is_bg.mean(axis=1)
        bg_frac_per_page.append(bg_frac_row.astype(np.float32))

        rendered_height = max(1, int(round(h * target_width / max(1, w))))
        page["accepted_page_index"] = idx
        page["width"] = int(target_width)
        page["height"] = int(rendered_height)
        page["global_y_start"] = int(global_y)
        page["global_y_end"] = int(global_y + rendered_height)
        global_y += rendered_height
        print(
            f"  p{idx + 1:02d}: alpha {h}x{w}, mean_bg={bg_frac_row.mean():.3f}, "
            f"max_bg={bg_frac_row.max():.3f}, t={time.time() - t0:.1f}s",
            flush=True,
        )

    # Stitch the per-page bg vectors into the global one. Each per-page vector
    # is at the page's native height; we resample to `rendered_height` so it
    # aligns with the global coordinate system used by `_segment_slices_for_range`.
    global_bg = np.empty(global_y, dtype=np.float32)
    cursor = 0
    for page, meta, row in zip(pages, page_metas, bg_frac_per_page):
        rendered_h = int(page["height"])
        if rendered_h <= 0:
            continue
        if row.shape[0] == rendered_h:
            resampled = row
        else:
            x_old = np.linspace(0.0, 1.0, num=row.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, num=rendered_h, dtype=np.float32)
            resampled = np.interp(x_new, x_old, row).astype(np.float32)
        global_bg[cursor : cursor + rendered_h] = resampled
        cursor += rendered_h
    assert cursor == global_y

    print(f"global vector length: {global_bg.shape[0]}", flush=True)

    threshold_used = BG_FRAC_THRESHOLD
    cuts = detect_global_cuts(
        global_bg,
        threshold=threshold_used,
        min_run=MIN_RUN_PX,
        min_segment=MIN_SEGMENT_HEIGHT_PX,
    )
    if not cuts:
        threshold_used = BG_FRAC_FALLBACK
        print(f"  no cuts at {BG_FRAC_THRESHOLD}; retrying at {threshold_used}", flush=True)
        cuts = detect_global_cuts(
            global_bg,
            threshold=threshold_used,
            min_run=MIN_RUN_PX,
            min_segment=MIN_SEGMENT_HEIGHT_PX,
        )

    print(f"cuts detected: {len(cuts)} (threshold={threshold_used})", flush=True)

    segments = build_segments_from_global_cuts(
        pages,
        global_cuts=cuts,
        target_width=int(target_width),
        total_height=int(global_y),
        min_segment_height=MIN_SEGMENT_HEIGHT_PX,
    )
    if segments:
        heights = [int(s.get("rendered_height") or 0) for s in segments]
        max_segment_h = max(heights)
        min_segment_h = min(heights)
        avg_segment_h = sum(heights) // len(heights)
        print(
            f"segments: {len(segments)}, heights min={min_segment_h} avg={avg_segment_h} max={max_segment_h}",
            flush=True,
        )
    else:
        max_segment_h = 0
        print("WARNING: 0 segments produced", flush=True)

    # Detect whether the dark-fire stretch (pages 7-23, 0-indexed 6..22) got
    # split: count cuts whose y falls inside that page span.
    dark_start_idx, dark_end_idx = 6, 22  # inclusive, 0-indexed
    dark_y_start = int(pages[dark_start_idx]["global_y_start"])
    dark_y_end = int(pages[dark_end_idx]["global_y_end"])
    dark_cuts = [c for c in cuts if dark_y_start < c < dark_y_end]
    dark_stretch_split = bool(dark_cuts)
    print(
        f"dark-fire stretch (p{dark_start_idx + 1}..p{dark_end_idx + 1}, y=[{dark_y_start},{dark_y_end}]): "
        f"{len(dark_cuts)} cuts -> split={'YES' if dark_stretch_split else 'NO'}",
        flush=True,
    )

    output_prefix = f"{OUTPUT_BASE}/{LABEL}"
    config = {
        "manhwa": {
            "raw_gutter_min_height_px": MANHWA_RAW_GUTTER_MIN_HEIGHT_PX,
            "raw_gutter_color_tolerance": MANHWA_RAW_GUTTER_COLOR_TOLERANCE,
            "raw_min_segment_height_px": MIN_SEGMENT_HEIGHT_PX,
            "raw_sheet_padding_px": MANHWA_RAW_SHEET_PADDING_PX,
            "raw_sheet_span_threshold_px": MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
            "strip_jpeg_quality": 92,
        }
    }
    sheet_result = write_sheets(
        output_prefix=output_prefix,
        chapter_key=CHAPTER_RELATIVE,
        segments=segments,
        target_width=int(target_width),
        config=config,
        get_bytes=lambda k: page_bytes_cache.get(k) or get_bytes(k),
        put_bytes=local_put_bytes,
        make_uri=local_make_uri,
    )

    # Overlay (full stitched scroll w/ red cuts + blue page ticks).
    print("rendering overlay ...", flush=True)
    scroll, spans = stitch_pages(keys)
    overlay = render_overlay(scroll, spans, cuts, LABEL)
    overlay_key = f"{output_prefix}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
    local_put_bytes(overlay_key, encode_jpeg(overlay), "image/jpeg")
    scroll.close()
    overlay.close()

    manifest = {
        "schema_version": 1,
        "detector": LABEL,
        "model": model_name,
        "threshold_used": float(threshold_used),
        "min_run_px": int(MIN_RUN_PX),
        "min_segment_height_px": int(MIN_SEGMENT_HEIGHT_PX),
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": output_prefix,
        "target_width": int(target_width),
        "page_count": len(pages),
        "global_height": int(global_y),
        "cuts": [int(c) for c in cuts],
        "cut_count": len(cuts),
        "max_segment_height_px": int(max_segment_h),
        "dark_stretch_split": dark_stretch_split,
        "dark_stretch_cuts": len(dark_cuts),
        "segmentation": {
            "total_virtual_height": int(global_y),
            "target_width": int(target_width),
            "detected_gutter_cuts": int(len(cuts)),
        },
        "segment_count": len(segments),
        "sheet_count": int(sheet_result.get("sheet_count") or 0),
        "segments": segments,
        "sheets": sheet_result.get("sheets") or [],
    }
    manifest_key = f"{output_prefix}/{CHAPTER_RELATIVE}/manifest.json"
    local_put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    # Verify that every sheet referenced by the manifest exists on disk.
    missing: list[str] = []
    for sheet in manifest["sheets"]:
        rel = sheet.get("sheet_key", "")
        prefix = f"{OUTPUT_BASE}/{LABEL}/"
        rel = rel[len(prefix):] if rel.startswith(prefix) else rel
        local_path = LOCAL_DETECTOR_ROOT / rel
        if not local_path.exists() or local_path.stat().st_size == 0:
            missing.append(str(local_path))
    if missing:
        raise RuntimeError(f"sheets missing or empty on disk: {missing}")

    overlay_path = LOCAL_DETECTOR_ROOT / overlay_key[len(f"{OUTPUT_BASE}/{LABEL}/"):]
    if not overlay_path.exists() or overlay_path.stat().st_size == 0:
        raise RuntimeError(f"overlay missing or empty on disk: {overlay_path}")

    print("\n=== summary ===", flush=True)
    print(f"  model:                  {model_name}")
    print(f"  threshold:              {threshold_used}")
    print(f"  cuts:                   {len(cuts)}")
    print(f"  segments:               {len(segments)}")
    print(f"  sheets:                 {manifest['sheet_count']}")
    print(f"  max_segment_height_px:  {max_segment_h}")
    print(f"  dark_stretch_split:     {'YES' if dark_stretch_split else 'NO'} ({len(dark_cuts)} cuts)")
    print(f"  manifest:               {LOCAL_DETECTOR_ROOT / CHAPTER_RELATIVE / 'manifest.json'}")
    print(f"  overlay:                {overlay_path}")


if __name__ == "__main__":
    main()
