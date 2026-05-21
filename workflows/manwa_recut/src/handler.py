"""Lambda handler for stitch + gutter-slice of one or more manwa chapters.

State machine sends items in batches; each item is a chapter row from the
manifest. We process every item in the batch sequentially (each is fast and
the slowest call is bounded by the largest chapter).

Manifest row shape:
  {
    "series": "<series>_manwa",
    "chapter": "chapter-000001"
  }

State-machine ItemSelector wraps each row as:
  {"row_index": int, "item": {...}, "config": {bucket, output_prefix, jpeg_quality}}

Output (per row): {status, series, chapter, slices, elapsed, error?}
"""
from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
import numpy as np
from botocore.config import Config
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Gutter slicer (inlined; was artifacts/gemini_panel_slicing/gutter_slicer.py) ---

UNIFORM_RANGE = 6
MIN_BAND_ROWS = 18
MIN_GAP_PX = 320
AREA_BUDGET = 1024 * 1024
SCAN_WINDOW_PX = 5000
MAX_OUTPUT_HEIGHT = 60000
MAX_OUTPUT_AREA = 2048 * 2048


class _Source:
    __slots__ = ("page_key", "page_idx", "page_y_start", "page_y_end",
                 "slice_y_start", "slice_y_end", "scale_x")

    def __init__(self, page_key, page_idx, page_y_start, page_y_end,
                 slice_y_start, slice_y_end, scale_x=1.0):
        self.page_key = page_key
        self.page_idx = page_idx
        self.page_y_start = page_y_start
        self.page_y_end = page_y_end
        self.slice_y_start = slice_y_start
        self.slice_y_end = slice_y_end
        self.scale_x = scale_x

    def to_json(self) -> dict[str, Any]:
        return {
            "page_key": self.page_key,
            "page_idx": int(self.page_idx),
            "page_y_start": int(self.page_y_start),
            "page_y_end": int(self.page_y_end),
            "slice_y_start": int(self.slice_y_start),
            "slice_y_end": int(self.slice_y_end),
            "scale_x": round(self.scale_x, 4),
        }


def _stitch_chapter(pages: list[dict[str, Any]]) -> tuple[Image.Image, list[_Source]]:
    if not pages:
        return Image.new("RGB", (1, 1), "white"), []
    target_w = max(p["image"].width for p in pages)
    normalized = []
    for i, p in enumerate(pages):
        im = p["image"]
        if im.width != target_w:
            ratio = target_w / im.width
            new_h = max(1, int(round(im.height * ratio)))
            im = im.resize((target_w, new_h), Image.Resampling.LANCZOS)
            scale = ratio
        else:
            scale = 1.0
        normalized.append((im, scale, p["key"], i))
    total_h = sum(im.height for im, *_ in normalized)
    strip = Image.new("RGB", (target_w, total_h), "white")
    sources: list[_Source] = []
    y = 0
    for im, scale_x, key, page_idx in normalized:
        strip.paste(im, (0, y))
        orig_h = int(round(im.height / scale_x))
        sources.append(_Source(key, page_idx, 0, orig_h, y, y + im.height, scale_x))
        y += im.height
    return strip, sources


def _find_gutter_cuts(strip: Image.Image) -> list[int]:
    arr = np.asarray(strip, dtype=np.int16)
    H = arr.shape[0]
    row_min = arr.min(axis=1)
    row_max = arr.max(axis=1)
    row_range = (row_max - row_min).max(axis=1)
    row_is_gutter = row_range <= UNIFORM_RANGE
    bands: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for y in range(H):
        if row_is_gutter[y]:
            if not in_run:
                run_start = y
                in_run = True
        else:
            if in_run:
                if y - run_start >= MIN_BAND_ROWS:
                    bands.append((run_start, y))
                in_run = False
    if in_run and H - run_start >= MIN_BAND_ROWS:
        bands.append((run_start, H))
    raw_cuts = [(b0 + b1) // 2 for b0, b1 in bands]
    cuts: list[int] = []
    for c in raw_cuts:
        if not cuts or c - cuts[-1] >= MIN_GAP_PX:
            cuts.append(c)
    return cuts


def _slice_at_cuts(strip: Image.Image, cuts: list[int], sources: list[_Source]):
    H = strip.height
    boundaries = [0] + cuts + [H]
    slices = []
    for i in range(len(boundaries) - 1):
        y0, y1 = boundaries[i], boundaries[i + 1]
        if y1 - y0 <= 0:
            continue
        seg = strip.crop((0, y0, strip.width, y1))
        seg_sources: list[_Source] = []
        for s in sources:
            ov0 = max(y0, s.slice_y_start)
            ov1 = min(y1, s.slice_y_end)
            if ov1 <= ov0:
                continue
            page_span = s.page_y_end - s.page_y_start
            strip_span = max(1, s.slice_y_end - s.slice_y_start)
            page_y0 = s.page_y_start + int(round((ov0 - s.slice_y_start) * page_span / strip_span))
            page_y1 = s.page_y_start + int(round((ov1 - s.slice_y_start) * page_span / strip_span))
            seg_sources.append(_Source(s.page_key, s.page_idx, page_y0, page_y1,
                                       ov0 - y0, ov1 - y0, s.scale_x))
        slices.append((seg, seg_sources))
    return slices


# --- Lambda glue ---

# One S3 client per Lambda container; the connection pool is configured for
# the per-chapter sub-parallelism we do for downloads/uploads.
_S3 = boto3.client(
    "s3",
    config=Config(max_pool_connections=64, retries={"mode": "adaptive", "max_attempts": 6}),
)
BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_INPUT_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/single_recut"


def _list_chapter_pages(bucket: str, input_prefix: str, series: str, chapter: str) -> list[str]:
    """Resolve a chapter key to its page list. Tries subdir layout first
    (<input_prefix>/<series>/<chapter>/page-NNNN.jpg), then flat layout
    (<input_prefix>/<series>/<chapter>__page-NNNN.jpg)."""
    out = []
    # (a) subdir layout
    for page in _S3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{input_prefix}/{series}/{chapter}/"
    ):
        for o in page.get("Contents") or []:
            if o["Key"].lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                out.append(o["Key"])
    if out:
        return sorted(out)
    # (b) flat layout — chapter key is the filename prefix before __page-
    for page in _S3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"{input_prefix}/{series}/{chapter}__page-"
    ):
        for o in page.get("Contents") or []:
            if o["Key"].lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                out.append(o["Key"])
    return sorted(out)


def _wipe_prefix(bucket: str, prefix: str) -> int:
    items: list[dict[str, str]] = []
    for page in _S3.get_paginator("list_object_versions").paginate(Bucket=bucket, Prefix=prefix):
        for v in page.get("Versions") or []:
            items.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        for m in page.get("DeleteMarkers") or []:
            items.append({"Key": m["Key"], "VersionId": m["VersionId"]})
    for i in range(0, len(items), 1000):
        _S3.delete_objects(Bucket=bucket, Delete={"Objects": items[i : i + 1000], "Quiet": True})
    return len(items)


def _load_pages(bucket: str, keys: list[str]) -> list[dict[str, Any]]:
    def _download(k: str) -> dict[str, Any]:
        body = _S3.get_object(Bucket=bucket, Key=k)["Body"].read()
        return {"key": k, "image": Image.open(io.BytesIO(body)).convert("RGB")}

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(_download, keys))


def _load_image(bucket: str, key: str) -> Image.Image:
    body = _S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return Image.open(io.BytesIO(body)).convert("RGB")


def _image_size(bucket: str, key: str) -> tuple[int, int]:
    body = _S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with Image.open(io.BytesIO(body)) as im:
        return im.size


def _build_page_sources(pages: list[dict[str, Any]]) -> tuple[int, int, list[_Source]]:
    target_w = max(p["image"].width for p in pages)
    sources: list[_Source] = []
    y = 0
    for idx, page in enumerate(pages):
        im = page["image"]
        scale_x = target_w / im.width
        norm_h = max(1, int(round(im.height * scale_x)))
        sources.append(_Source(page["key"], idx, 0, im.height, y, y + norm_h, scale_x))
        y += norm_h
    return target_w, y, sources


def _build_page_sources_from_keys(bucket: str, keys: list[str]) -> tuple[list[dict[str, Any]], int, int, list[_Source]]:
    pages: list[dict[str, Any]] = []
    for key in keys:
        width, height = _image_size(bucket, key)
        pages.append({"key": key, "width": width, "height": height})

    target_w = max(int(p["width"]) for p in pages)
    sources: list[_Source] = []
    y = 0
    for idx, page in enumerate(pages):
        width = int(page["width"])
        height = int(page["height"])
        scale_x = target_w / width
        norm_h = max(1, int(round(height * scale_x)))
        sources.append(_Source(str(page["key"]), idx, 0, height, y, y + norm_h, scale_x))
        y += norm_h
    return pages, target_w, y, sources


def _page_y_from_strip(source: _Source, y: int) -> int:
    page_span = source.page_y_end - source.page_y_start
    strip_span = max(1, source.slice_y_end - source.slice_y_start)
    local_y = min(max(0, y - source.slice_y_start), strip_span)
    return source.page_y_start + int(round(local_y * page_span / strip_span))


def _render_virtual_segment(
    pages: list[dict[str, Any]],
    sources: list[_Source],
    target_w: int,
    y0: int,
    y1: int,
) -> Image.Image:
    seg = Image.new("RGB", (target_w, max(1, y1 - y0)), "white")
    for source in sources:
        ov0 = max(y0, source.slice_y_start)
        ov1 = min(y1, source.slice_y_end)
        if ov1 <= ov0:
            continue
        page_y0 = _page_y_from_strip(source, ov0)
        page_y1 = _page_y_from_strip(source, ov1)
        page_y1 = max(page_y0 + 1, page_y1)
        im = pages[source.page_idx]["image"]
        crop = im.crop((0, page_y0, im.width, min(page_y1, im.height)))
        out_h = ov1 - ov0
        if crop.width != target_w or crop.height != out_h:
            crop = crop.resize((target_w, out_h), Image.Resampling.LANCZOS)
        seg.paste(crop, (0, ov0 - y0))
        try:
            crop.close()
        except Exception:
            pass
    return seg


def _normalized_page_image(bucket: str, page: dict[str, Any], source: _Source, target_w: int) -> Image.Image:
    im = _load_image(bucket, str(page["key"]))
    norm_h = max(1, source.slice_y_end - source.slice_y_start)
    if im.width == target_w and im.height == norm_h:
        return im
    norm = im.resize((target_w, norm_h), Image.Resampling.LANCZOS)
    try:
        im.close()
    except Exception:
        pass
    return norm


def _get_cached_normalized_page(
    bucket: str,
    pages: list[dict[str, Any]],
    sources: list[_Source],
    target_w: int,
    page_idx: int,
    cache: dict[int, Image.Image],
) -> Image.Image:
    cached = cache.get(page_idx)
    if cached is not None:
        return cached
    while len(cache) >= 1:
        old_key = next(iter(cache))
        old = cache.pop(old_key)
        try:
            old.close()
        except Exception:
            pass
    norm = _normalized_page_image(bucket, pages[page_idx], sources[page_idx], target_w)
    cache[page_idx] = norm
    return norm


def _render_virtual_segment_exact(
    bucket: str,
    pages: list[dict[str, Any]],
    sources: list[_Source],
    target_w: int,
    y0: int,
    y1: int,
    norm_cache: dict[int, Image.Image],
) -> Image.Image:
    seg = Image.new("RGB", (target_w, max(1, y1 - y0)), "white")
    for source in sources:
        ov0 = max(y0, source.slice_y_start)
        ov1 = min(y1, source.slice_y_end)
        if ov1 <= ov0:
            continue
        norm = _get_cached_normalized_page(bucket, pages, sources, target_w, source.page_idx, norm_cache)
        crop = norm.crop((
            0,
            ov0 - source.slice_y_start,
            target_w,
            ov1 - source.slice_y_start,
        ))
        seg.paste(crop, (0, ov0 - y0))
        try:
            crop.close()
        except Exception:
            pass
    return seg


def _feed_gutter_rows(
    row_is_gutter: np.ndarray,
    y_offset: int,
    state: dict[str, Any],
    cuts: list[int],
) -> None:
    for local_y, is_gutter in enumerate(row_is_gutter):
        y = y_offset + local_y
        if bool(is_gutter):
            if not state["in_run"]:
                state["run_start"] = y
                state["in_run"] = True
        elif state["in_run"]:
            run_start = int(state["run_start"])
            if y - run_start >= MIN_BAND_ROWS:
                cut = (run_start + y) // 2
                if not cuts or cut - cuts[-1] >= MIN_GAP_PX:
                    cuts.append(cut)
            state["in_run"] = False


def _find_gutter_cuts_windowed(
    bucket: str,
    pages: list[dict[str, Any]],
    sources: list[_Source],
    target_w: int,
    total_h: int,
) -> list[int]:
    cuts: list[int] = []
    state: dict[str, Any] = {"in_run": False, "run_start": 0}
    for source in sources:
        norm = _normalized_page_image(bucket, pages[source.page_idx], source, target_w)
        for local_y0 in range(0, norm.height, SCAN_WINDOW_PX):
            local_y1 = min(norm.height, local_y0 + SCAN_WINDOW_PX)
            window = norm.crop((0, local_y0, target_w, local_y1))
            arr = np.asarray(window, dtype=np.int16)
            row_min = arr.min(axis=1)
            row_max = arr.max(axis=1)
            row_range = (row_max - row_min).max(axis=1)
            _feed_gutter_rows(row_range <= UNIFORM_RANGE, source.slice_y_start + local_y0, state, cuts)
            try:
                window.close()
            except Exception:
                pass
        try:
            norm.close()
        except Exception:
            pass
    if state["in_run"]:
        run_start = int(state["run_start"])
        if total_h - run_start >= MIN_BAND_ROWS:
            cut = (run_start + total_h) // 2
            if not cuts or cut - cuts[-1] >= MIN_GAP_PX:
                cuts.append(cut)
    return cuts


def _add_codec_safety_cuts(cuts: list[int], total_h: int, sources: list[_Source]) -> tuple[list[int], list[int]]:
    base_cuts = sorted(set(c for c in cuts if 0 < c < total_h))
    page_boundaries = sorted(
        set(s.slice_y_start for s in sources[1:]) |
        set(s.slice_y_end for s in sources[:-1])
    )
    forced: list[int] = []
    for y0, y1 in zip([0] + base_cuts, base_cuts + [total_h]):
        cur = y0
        while y1 - cur > MAX_OUTPUT_HEIGHT:
            limit = cur + MAX_OUTPUT_HEIGHT
            candidates = [b for b in page_boundaries if cur + MIN_GAP_PX <= b <= limit]
            cut = candidates[-1] if candidates else limit
            if cut <= cur or cut >= y1:
                cut = limit
            forced.append(cut)
            cur = cut
    all_cuts = sorted(set(base_cuts + forced))
    return all_cuts, sorted(set(forced))


def _slice_sources(y0: int, y1: int, sources: list[_Source]) -> list[_Source]:
    seg_sources: list[_Source] = []
    for source in sources:
        ov0 = max(y0, source.slice_y_start)
        ov1 = min(y1, source.slice_y_end)
        if ov1 <= ov0:
            continue
        seg_sources.append(_Source(
            source.page_key,
            source.page_idx,
            _page_y_from_strip(source, ov0),
            _page_y_from_strip(source, ov1),
            ov0 - y0,
            ov1 - y0,
            source.scale_x,
        ))
    return seg_sources


def _process_one(item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    series = str(item["series"])
    chapter = str(item["chapter"])
    bucket = str(config.get("bucket") or BUCKET)
    in_prefix_root = str(config.get("input_prefix") or DEFAULT_INPUT_PREFIX)
    out_prefix_root = str(config.get("output_prefix") or DEFAULT_OUTPUT_PREFIX)
    jpeg_quality = int(config.get("jpeg_quality") or 90)
    out_prefix = f"{out_prefix_root}/{series}/{chapter}/"

    t0 = time.perf_counter()

    keys = _list_chapter_pages(bucket, in_prefix_root, series, chapter)
    if not keys:
        return {"status": "empty", "series": series, "chapter": chapter, "elapsed": round(time.perf_counter() - t0, 1)}

    pages, target_w, total_h, sources = _build_page_sources_from_keys(bucket, keys)
    gutter_cuts = _find_gutter_cuts_windowed(bucket, pages, sources, target_w, total_h)
    cuts, forced_codec_cuts = _add_codec_safety_cuts(gutter_cuts, total_h, sources)

    _wipe_prefix(bucket, out_prefix)

    sizes: list[int] = []
    slice_records: list[dict[str, Any]] = []
    dropped_area_slices: list[dict[str, Any]] = []
    norm_cache: dict[int, Image.Image] = {}
    boundaries = [0] + cuts + [total_h]
    try:
        for idx, (y0, y1) in enumerate(zip(boundaries, boundaries[1:])):
            if y1 <= y0:
                continue
            seg = _render_virtual_segment_exact(bucket, pages, sources, target_w, y0, y1, norm_cache)
            seg_sources = _slice_sources(y0, y1, sources)
            area = seg.width * seg.height
            if area > MAX_OUTPUT_AREA:
                dropped_area_slices.append({
                    "slice_index": idx,
                    "width": seg.width,
                    "height": seg.height,
                    "area": area,
                    "reason": "area_gt_max_output_area",
                    "sources": [src.to_json() for src in seg_sources],
                })
                try:
                    seg.close()
                except Exception:
                    pass
                continue
            buf = io.BytesIO()
            seg.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            body = buf.getvalue()
            key = f"{out_prefix}slice-{idx:04d}.jpg"
            _S3.put_object(
                Bucket=bucket, Key=key, Body=body, ContentType="image/jpeg",
                Metadata={"width": str(seg.width), "height": str(seg.height)},
            )
            sizes.append(len(body))
            slice_records.append({
                "key": key,
                "width": seg.width,
                "height": seg.height,
                "area": area,
                "sources": [src.to_json() for src in seg_sources],
            })
            try:
                seg.close()
            except Exception:
                pass
    finally:
        for norm in norm_cache.values():
            try:
                norm.close()
            except Exception:
                pass

    prov = {
        "series": series,
        "chapter": chapter,
        "input_prefix": f"s3://{bucket}/{in_prefix_root}/{series}/{chapter}/",
        "output_prefix": f"s3://{bucket}/{out_prefix}",
        "input_pages": keys,
        "strip_width": target_w,
        "strip_height": total_h,
        "gutter_cuts": gutter_cuts,
        "forced_codec_cuts": forced_codec_cuts,
        "max_output_height": MAX_OUTPUT_HEIGHT,
        "max_output_area": MAX_OUTPUT_AREA,
        "slices": slice_records,
        "dropped_area_slices": dropped_area_slices,
        "algorithm": {
            "uniform_range": UNIFORM_RANGE,
            "min_band_rows": MIN_BAND_ROWS,
            "min_gap_px": MIN_GAP_PX,
            "scan_window_px": SCAN_WINDOW_PX,
            "strict_full_row_uniformity": True,
        },
    }
    _S3.put_object(
        Bucket=bucket, Key=f"{out_prefix}_provenance.json",
        Body=json.dumps(prov, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "status": "ok",
        "series": series,
        "chapter": chapter,
        "input_pages": len(keys),
        "slices": len(slice_records),
        "dropped_area_slices": len(dropped_area_slices),
        "gutter_cuts": len(gutter_cuts),
        "forced_codec_cuts": len(forced_codec_cuts),
        "total_bytes": sum(sizes),
        "strip_h": total_h,
        "elapsed": round(time.perf_counter() - t0, 1),
    }


def recut_chapter_batch(event, _context):
    """Distributed Map ItemBatcher target. ``event`` is one of:
      - a single ItemSelector dict {"row_index", "item", "config"}, or
      - a batched form {"Items": [...]} where Items is a list of ItemSelector dicts.
    """
    if isinstance(event, list):
        items = event
    elif isinstance(event, dict) and isinstance(event.get("Items"), list):
        items = event["Items"]
    else:
        items = [event]
    results = []
    config = None
    for entry in items:
        if not isinstance(entry, dict):
            continue
        config = entry.get("config") or config or {}
        item = entry.get("item") or entry
        try:
            results.append(_process_one(item, config))
        except Exception as exc:
            if len(items) == 1:
                raise
            results.append({
                "status": "error",
                "series": item.get("series", "?"),
                "chapter": item.get("chapter", "?"),
                "error": repr(exc)[:240],
            })
    return {"results": results}
