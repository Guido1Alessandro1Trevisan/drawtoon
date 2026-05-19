"""Gemini-driven page filter + negative-space cut detection.

For every raw page in the chapter, asks Gemini 3 Flash:
  - is_title_page : keep this page or filter it?
  - cuts          : y-coordinates (0..page_height) of horizontal negative-space
                    boundaries where the page can be safely split between panels

Cuts are aggregated into the global scroll y-space and run through the same
sheet packer used by the production pipeline. Output lives under
artifacts/gutter_comparison/gemini_cuts/ for side-by-side comparison.

Run:
    cd /Users/guidotrevisan/Desktop/drawtoon && \
        uv run --active --with numpy --with pillow --with boto3 \
            --with google-genai --with scipy \
            python artifacts/scripts/run_gemini_cuts.py
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import sys
from pathlib import Path
from typing import Any

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
    _segment_slices_for_range,
    build_pages_for_chapter,
    get_bytes,
    list_chapter_pages,
    make_uri,
    put_bytes,
)
from src.manhwa_raw_filter import _resolve_gemini_api_key  # noqa: E402
from src.manhwa_raw_sheets import (  # noqa: E402
    MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
    MANHWA_RAW_SHEET_PADDING_PX,
    MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
    write_sheets,
)

DETECTOR = "gemini_cuts"
MODEL = "gemini-3-flash-preview"
MAX_PARALLEL = 10
MAX_OUTPUT_TOKENS = 1024
# Number of consecutive pages stitched into one image before sending to Gemini.
# 1 = legacy per-page mode. 2 = pairs (so the page-seam gutter is visible).
PAGES_PER_CHUNK = 2

PROMPT_TEMPLATE = """\
You are filtering one raw vertical manhwa/webtoon page (or several pages stitched into one tall image).
Return JSON only through the requested schema.

Set is_title_page to true ONLY for title pages, chapter/episode cards, covers, promo art, credits, translator notes, ads, blank/loading pages, reader UI/screenshots, and other non-story material.
Set is_title_page to false for real sequential comic story content -- panels, splash panels, action, characters, dialogue, narration, and normal story art.
If uncertain, set is_title_page to false.

If is_title_page is true, return cuts: [] and stop.

Otherwise, for each gutter between two panels on the image, return the y-pixel coordinate that splits that gutter. A gutter can be black, white, or a slightly shaded band -- if it separates two story beats, cut there.
Never return a y that crosses a speech bubble, narration box, SFX text, character, or panel artwork -- just don't include it in cuts (return an empty array if no safe cut exists).
"""

PROMPT_MIN_SEGMENT_HEIGHT = 96

SCHEMA = {
    "type": "object",
    "properties": {
        "is_title_page": {"type": "boolean"},
        "cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            },
        },
    },
    "required": ["is_title_page", "cuts"],
}


s3 = boto3.client("s3", region_name=REGION)
_GENAI_CLIENT: Any = None


def gemini_client():
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    from google import genai
    from google.genai import types

    _GENAI_CLIENT = genai.Client(
        api_key=_resolve_gemini_api_key(),
        http_options=types.HttpOptions(timeout=180000),
    )
    return _GENAI_CLIENT


def classify_and_cut(image_bytes: bytes, page_index: int, page_width: int, page_height: int) -> dict[str, Any]:
    from google.genai import types

    prompt = PROMPT_TEMPLATE
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = gemini_client().models.generate_content(
        model=MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=True,
            ),
        ),
    )
    # Split response parts into "thought" (reasoning summary) and answer.
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            txt = getattr(part, "text", None)
            if not txt:
                continue
            if getattr(part, "thought", False):
                thought_chunks.append(txt)
            else:
                answer_chunks.append(txt)
    answer_text = "".join(answer_chunks) or str(response.text or "{}")
    try:
        parsed = json.loads(answer_text)
    except json.JSONDecodeError:
        parsed = {}
    usage = getattr(response, "usage_metadata", None)
    raw_cuts = [c for c in (parsed.get("cuts") or []) if isinstance(c, dict)]

    def _rescale(y: int) -> int:
        # Gemini returns y normalized to 0..1000 (per the image-understanding docs).
        # Treat values >1000 as already-pixel (defensive), otherwise rescale.
        y = int(y)
        if y > 1000:
            return y
        return int(round(y * page_height / 1000.0))

    pixel_cuts = [_rescale(c.get("y") or 0) for c in raw_cuts]
    return {
        "page_index": page_index,
        "is_title_page": bool(parsed.get("is_title_page", False)),
        "cuts": pixel_cuts,
        "cuts_raw_normalized": [int(c.get("y") or 0) for c in raw_cuts],
        "cut_reasons": [str(c.get("reason") or "") for c in raw_cuts],
        "thought_summary": "\n".join(thought_chunks),
        "raw_cuts": raw_cuts,
        "usage": {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0,
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0) if usage else 0,
        },
    }


def build_kept_strip(
    keys: list[str],
    pages: list[dict[str, Any]],
    target_width: int,
    rejected_indices: set[int],
) -> Image.Image:
    """Stitch every kept page into a single full-resolution vertical strip."""
    parts: list[Image.Image] = []
    total_h = 0
    for idx, key in enumerate(keys):
        if idx in rejected_indices:
            continue
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        if im.width != target_width:
            new_h = int(round(im.height * target_width / im.width))
            im = im.resize((target_width, new_h), Image.Resampling.LANCZOS)
        parts.append(im)
        total_h += im.height
    strip = Image.new("RGB", (target_width, total_h), "white")
    y = 0
    for im in parts:
        strip.paste(im, (0, y))
        y += im.height
        im.close()
    return strip


def remap_cuts_to_kept_strip(
    global_cuts: list[int],
    pages: list[dict[str, Any]],
    page_heights: list[int],
    rejected_indices: set[int],
) -> list[int]:
    """Translate cuts that live in the all-pages scroll y-space into the
    kept-only strip y-space (which is shorter when some pages are filtered)."""
    # Precompute per-page (rejected, start_in_kept) info.
    kept_starts: list[int | None] = []
    running = 0
    for idx, h in enumerate(page_heights):
        if idx in rejected_indices:
            kept_starts.append(None)
        else:
            kept_starts.append(running)
            running += h
    remapped: list[int] = []
    for c in global_cuts:
        # Find which page the cut sits in.
        for idx, page in enumerate(pages):
            ps = int(page.get("global_y_start") or 0)
            pe = int(page.get("global_y_end") or 0)
            if ps <= c < pe:
                if idx in rejected_indices:
                    break  # cut falls in a dropped page; skip
                offset_in_page = c - ps
                kept_start = kept_starts[idx] or 0
                remapped.append(kept_start + offset_in_page)
                break
    remapped.sort()
    return remapped


def write_cut_strips(
    full_strip: Image.Image,
    cuts: list[int],
    output_prefix: str,
    chapter_key: str,
    jpeg_quality: int = 92,
) -> list[dict[str, Any]]:
    """Slice the full kept-only strip at the given y-cuts and write each piece
    as its own JPEG. Returns one record per output strip."""
    boundaries = [0] + sorted(set(cuts)) + [full_strip.height]
    records: list[dict[str, Any]] = []
    for i, (y0, y1) in enumerate(zip(boundaries, boundaries[1:])):
        if y1 <= y0:
            continue
        crop = full_strip.crop((0, y0, full_strip.width, y1))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        crop.close()
        data = buf.getvalue()
        key = (
            f"{output_prefix.rstrip('/')}/{chapter_key.strip('/')}/"
            f"strip_{i:04d}_y{y0:06d}-y{y1:06d}.jpg"
        )
        put_bytes(key, data, "image/jpeg")
        records.append(
            {
                "strip_id": f"strip_{i:04d}",
                "strip_index": i,
                "y_start": int(y0),
                "y_end": int(y1),
                "height": int(y1 - y0),
                "width": int(full_strip.width),
                "image_bytes": len(data),
                "key": key,
                "uri": make_uri(key),
            }
        )
    return records


def build_page_chunks(
    page_bytes: list[bytes],
    page_heights: list[int],
    target_width: int,
    pages_per_chunk: int,
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stitch consecutive pages together; each chunk is just one tall image.

    Each returned dict has:
      - chunk_index, page_indices, chunk_bytes, chunk_height, chunk_global_offset
    """
    chunks: list[dict[str, Any]] = []
    for chunk_idx, start in enumerate(range(0, len(page_bytes), pages_per_chunk)):
        end = min(start + pages_per_chunk, len(page_bytes))
        ims: list[Image.Image] = []
        chunk_h = 0
        for i in range(start, end):
            im = Image.open(io.BytesIO(page_bytes[i])).convert("RGB")
            if im.width != target_width:
                new_h = int(round(im.height * target_width / im.width))
                im = im.resize((target_width, new_h), Image.Resampling.LANCZOS)
            ims.append(im)
            chunk_h += im.height
        canvas = Image.new("RGB", (target_width, chunk_h), "white")
        y = 0
        for im in ims:
            canvas.paste(im, (0, y))
            y += im.height
            im.close()
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=92, optimize=True)
        canvas.close()
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "page_indices": list(range(start, end)),
                "chunk_bytes": buf.getvalue(),
                "chunk_height": int(chunk_h),
                "chunk_global_offset": int(pages[start]["global_y_start"]),
            }
        )
    return chunks


def classify_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Send one stitched chunk to Gemini, return is_title_page + global cuts."""
    from google.genai import types

    image_part = types.Part.from_bytes(data=chunk["chunk_bytes"], mime_type="image/jpeg")
    response = gemini_client().models.generate_content(
        model=MODEL,
        contents=[PROMPT_TEMPLATE, image_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=True,
            ),
        ),
    )
    thought_chunks: list[str] = []
    answer_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            txt = getattr(part, "text", None)
            if not txt:
                continue
            if getattr(part, "thought", False):
                thought_chunks.append(txt)
            else:
                answer_chunks.append(txt)
    answer_text = "".join(answer_chunks) or str(response.text or "{}")
    try:
        parsed = json.loads(answer_text)
    except json.JSONDecodeError:
        parsed = {}
    usage = getattr(response, "usage_metadata", None)
    raw_cuts = [c for c in (parsed.get("cuts") or []) if isinstance(c, dict)]
    chunk_h = int(chunk["chunk_height"])
    offset = int(chunk["chunk_global_offset"])

    def _rescale_to_global(y_norm: int) -> int:
        y = int(y_norm)
        if y > 1000:  # defensive: already pixel
            return offset + y
        return offset + int(round(y * chunk_h / 1000.0))

    global_cuts = [_rescale_to_global(c.get("y") or 0) for c in raw_cuts]
    return {
        "chunk_index": int(chunk["chunk_index"]),
        "page_indices": list(chunk["page_indices"]),
        "is_title_page": bool(parsed.get("is_title_page", False)),
        "global_cuts": global_cuts,
        "cuts_raw_normalized": [int(c.get("y") or 0) for c in raw_cuts],
        "cut_reasons": [str(c.get("reason") or "") for c in raw_cuts],
        "thought_summary": "\n".join(thought_chunks),
        "usage": {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0,
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0) if usage else 0,
        },
    }


def stitch_thumbnail(keys: list[str], target_width: int) -> Image.Image:
    pages = []
    total_h = 0
    for key in keys:
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        if im.width != target_width:
            new_h = int(round(im.height * target_width / im.width))
            im = im.resize((target_width, new_h), Image.Resampling.LANCZOS)
        pages.append(im)
        total_h += im.height
    scroll = Image.new("RGB", (target_width, total_h), "white")
    y = 0
    for im in pages:
        scroll.paste(im, (0, y))
        y += im.height
        im.close()
    return scroll


def render_overlay(scroll: Image.Image, page_spans: list[tuple[int, int]], cuts: list[int], label: str) -> Image.Image:
    THUMB_WIDTH = 360
    scale = THUMB_WIDTH / scroll.width
    overlay = scroll.resize((THUMB_WIDTH, int(round(scroll.height * scale))), Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=12)
        big_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
        big_font = font
    for idx, (_, y_end) in enumerate(page_spans[:-1]):
        y = int(round(y_end * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=(60, 140, 240), width=1)
        draw.text((2, y - 13), f"p{idx + 1}", fill=(60, 140, 240), font=font)
    for idx, cut in enumerate(cuts):
        y = int(round(cut * scale))
        draw.line([(0, y), (overlay.width, y)], fill=(240, 60, 60), width=2)
        draw.text((overlay.width - 26, y + 2), str(idx), fill=(240, 60, 60), font=font)
    header_h = 36
    bar = Image.new("RGB", (overlay.width, overlay.height + header_h), (0, 0, 0))
    bar.paste(overlay, (0, header_h))
    ImageDraw.Draw(bar).text((6, 4), f"{label}  cuts={len(cuts)}", fill=(255, 255, 255), font=big_font)
    return bar


def main() -> None:
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    print(f"pages: {len(keys)}")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)
    page_bytes: list[bytes] = []
    target_width = 0
    page_heights: list[int] = []
    for key in keys:
        data = get_bytes(key)
        page_bytes.append(data)
        with Image.open(io.BytesIO(data)) as im:
            target_width = max(target_width, im.width)
            page_heights.append(im.height)

    # Page bookkeeping (matches build_segments_with_detector).
    global_y = 0
    page_spans: list[tuple[int, int]] = []
    for idx, page in enumerate(pages):
        h = page_heights[idx]
        page["accepted_page_index"] = idx
        page["width"] = target_width
        page["height"] = h
        page["global_y_start"] = global_y
        page["global_y_end"] = global_y + h
        page_spans.append((global_y, global_y + h))
        global_y += h
    total_height = global_y

    # Build chunks (pairs of pages, stitched into one tall image each) and ask
    # Gemini for cuts on each chunk. The cuts come back as global y-pixels.
    chunks = build_page_chunks(
        page_bytes, page_heights, target_width, PAGES_PER_CHUNK, pages
    )
    print(
        f"querying Gemini ({MODEL}) on {len(chunks)} chunks of up to "
        f"{PAGES_PER_CHUNK} page(s) each, max_parallel={MAX_PARALLEL}"
    )
    chunk_results: list[dict[str, Any]] = [None] * len(chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        future_to_idx = {ex.submit(classify_chunk, ch): i for i, ch in enumerate(chunks)}
        for fut in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                chunk_results[i] = fut.result()
            except Exception as exc:
                ch = chunks[i]
                chunk_results[i] = {
                    "chunk_index": ch["chunk_index"],
                    "page_indices": list(ch["page_indices"]),
                    "is_title_page": False,
                    "global_cuts": [],
                    "cuts_raw_normalized": [],
                    "cut_reasons": [],
                    "thought_summary": "",
                    "error": str(exc),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0},
                }

    total_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in chunk_results)
    total_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in chunk_results)
    title_chunks = [r for r in chunk_results if r.get("is_title_page")]
    story_chunk_count = len(chunk_results) - len(title_chunks)
    print(f"  chunks: {len(chunks)}  story_chunks={story_chunk_count}  title_chunks={len(title_chunks)}")
    print(f"  usage: input_tokens={total_in} output_tokens={total_out}")

    # A page is dropped (rejected) when Gemini marks its containing chunk as a
    # title/cover/credits page. Otherwise it's a kept story page.
    FORCE_KEEP_ALL = True
    story_pages: set[int] = set()
    if FORCE_KEEP_ALL:
        story_pages = set(range(len(pages)))
    else:
        for r in chunk_results:
            if not r.get("is_title_page"):
                story_pages.update(r.get("page_indices") or [])
    kept_pages_indices = sorted(story_pages)
    rejected_pages_indices = [i for i in range(len(pages)) if i not in story_pages]

    # Aggregate global cuts: union of every chunk's reported cuts.
    min_segment = max(96, int(MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX))
    raw_global_cuts: list[int] = []
    for r in chunk_results:
        raw_global_cuts.extend(int(c) for c in (r.get("global_cuts") or []))
    # Add hard cuts at rejected page seams so dropped pages don't sit inside a strip.
    for idx in rejected_pages_indices:
        raw_global_cuts.append(int(pages[idx]["global_y_start"]))
        raw_global_cuts.append(int(pages[idx]["global_y_end"]))
    # Keep cuts strictly inside the scroll.
    global_cuts = sorted({c for c in raw_global_cuts if 0 < c < total_height})
    # Enforce min_segment spacing between consecutive cuts.
    pruned: list[int] = []
    for c in global_cuts:
        if not pruned or c - pruned[-1] >= min_segment:
            pruned.append(c)
    global_cuts = pruned

    print(f"  global cuts (after min-segment prune): {len(global_cuts)}")

    # Build segments from global cuts (used only for the manifest).
    if not kept_pages_indices:
        print("no kept pages; aborting")
        return
    cuts_for_segments: list[int] = [0] + global_cuts + [total_height]
    rejected_idx = set(rejected_pages_indices)
    segments: list[dict[str, Any]] = []
    for start_y, end_y in zip(cuts_for_segments, cuts_for_segments[1:]):
        if end_y - start_y < min_segment:
            continue
        slices, _ = _segment_slices_for_range(
            pages, start_y=start_y, end_y=end_y, target_width=target_width
        )
        slices = [s for s in slices if int(s["page_index"]) not in rejected_idx]
        if not slices:
            continue
        rendered_h = sum(int(s.get("rendered_height") or 0) for s in slices)
        seg_idx = len(segments)
        segments.append(
            {
                "segment_id": f"segment_{seg_idx:04d}",
                "segment_index": seg_idx,
                "global_y_start": int(start_y),
                "global_y_end": int(end_y),
                "source_height": int(end_y - start_y),
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_h),
                "source_slices": slices,
                "page_indices": sorted({int(s["page_index"]) for s in slices}),
                "page_span": [int(slices[0]["page_index"]), int(slices[-1]["page_index"])],
            }
        )

    heights = [s["rendered_height"] for s in segments]
    print(f"  segments: {len(segments)} (min={min(heights, default=0)}, "
          f"median={sorted(heights)[len(heights)//2] if heights else 0}, "
          f"max={max(heights, default=0)})")

    # Stitch ALL kept pages into one tall strip, then cut at the global gutter
    # y-positions Gemini chose. Each resulting piece is saved as its own JPEG.
    full_strip = build_kept_strip(
        keys=keys,
        pages=pages,
        target_width=target_width,
        rejected_indices=rejected_idx,
    )
    # Remap global cuts into the kept-strip coordinate space (drop any cut
    # falling inside a rejected page; shift later cuts down by removed heights).
    kept_strip_cuts = remap_cuts_to_kept_strip(
        global_cuts=global_cuts,
        pages=pages,
        page_heights=page_heights,
        rejected_indices=rejected_idx,
    )
    output_prefix = f"{OUTPUT_BASE}/{DETECTOR}"
    cut_strip_records = write_cut_strips(
        full_strip=full_strip,
        cuts=kept_strip_cuts,
        output_prefix=output_prefix,
        chapter_key=CHAPTER_RELATIVE,
        jpeg_quality=92,
    )
    print(f"  cut_strips: {len(cut_strip_records)}")
    full_strip.close()

    # Manifest.
    manifest = {
        "schema_version": 2,
        "detector": DETECTOR,
        "model": MODEL,
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": output_prefix,
        "target_width": int(target_width),
        "segmentation": {
            "total_virtual_height": int(total_height),
            "target_width": int(target_width),
            "detected_gutter_cuts": int(len(global_cuts)),
            "cut_count": int(len(cuts_for_segments)),
        },
        "segment_count": len(segments),
        "segments": segments,
        "cut_strip_count": len(cut_strip_records),
        "cut_strips": cut_strip_records,
        "page_count": len(pages),
        "kept_page_count": len(kept_pages_indices),
        "rejected_page_count": len(rejected_pages_indices),
        "pages_per_chunk": PAGES_PER_CHUNK,
        "per_chunk_results": [
            {
                "chunk_index": r.get("chunk_index"),
                "page_indices": r.get("page_indices") or [],
                "is_title_page": bool(r.get("is_title_page")),
                "global_cuts": list(r.get("global_cuts") or []),
                "cuts_raw_normalized": list(r.get("cuts_raw_normalized") or []),
                "cut_reasons": list(r.get("cut_reasons") or []),
                "usage": r.get("usage") or {},
                "error": r.get("error"),
            }
            for r in chunk_results
        ],
        "usage_totals": {
            "input_tokens": total_in,
            "output_tokens": total_out,
        },
    }
    manifest_key = f"{output_prefix}/{CHAPTER_RELATIVE}/manifest.json"
    put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")

    # Overlay.
    scroll = stitch_thumbnail(keys, target_width)
    overlay = render_overlay(scroll, page_spans, global_cuts, DETECTOR)
    overlay_buf = io.BytesIO()
    overlay.save(overlay_buf, format="JPEG", quality=86, optimize=True)
    overlay_key = f"{output_prefix}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
    put_bytes(overlay_key, overlay_buf.getvalue(), "image/jpeg")
    print(f"manifest -> {LOCAL_OUTPUT_ROOT / DETECTOR / CHAPTER_RELATIVE / 'manifest.json'}")
    print(f"overlay  -> {LOCAL_OUTPUT_ROOT / DETECTOR / CHAPTER_RELATIVE / 'cuts_overlay.jpg'}")
    print(f"strips   -> {LOCAL_OUTPUT_ROOT / DETECTOR / CHAPTER_RELATIVE}/strip_*.jpg")

    # Per-chunk thoughts dump.
    thoughts_lines: list[str] = [
        f"# Gemini cut-detection thoughts ({MODEL}, thinking_level=HIGH, "
        f"pages_per_chunk={PAGES_PER_CHUNK})\n"
    ]
    thoughts_lines.append(f"Chapter: {CHAPTER_RELATIVE}")
    thoughts_lines.append(f"Chunks: {len(chunk_results)} · Pages: {len(pages)}\n")
    for r in chunk_results:
        page_indices = r.get("page_indices") or []
        label = "p" + "+p".join(f"{i + 1:02d}" for i in page_indices)
        thoughts_lines.append(f"---\n\n## chunk {r.get('chunk_index')}  ({label})\n")
        cuts = r.get("global_cuts") or []
        raw_norm = r.get("cuts_raw_normalized") or []
        reasons = r.get("cut_reasons") or []
        usage = r.get("usage") or {}
        thoughts_lines.append(
            f"is_title_page = {r.get('is_title_page')}  ·  cuts={len(cuts)}  "
            f"thoughts_tokens={usage.get('thoughts_tokens', 0)}  "
            f"output_tokens={usage.get('output_tokens', 0)}\n"
        )
        if cuts:
            thoughts_lines.append("**Cuts returned (global y in scroll, rescaled):**")
            for g, raw, why in zip(cuts, raw_norm, reasons):
                why_str = f" :: {why}" if why else ""
                thoughts_lines.append(f"- global_y={g}px (raw_norm={raw}){why_str}")
            thoughts_lines.append("")
        else:
            thoughts_lines.append("_(no cuts returned)_\n")
        thought_text = r.get("thought_summary") or ""
        if thought_text.strip():
            thoughts_lines.append("**Reasoning trace:**\n")
            thoughts_lines.append("```")
            thoughts_lines.append(thought_text.rstrip())
            thoughts_lines.append("```\n")
        else:
            thoughts_lines.append("_(no thought summary returned)_\n")
    thoughts_path = LOCAL_OUTPUT_ROOT / DETECTOR / CHAPTER_RELATIVE / "thoughts.md"
    thoughts_path.parent.mkdir(parents=True, exist_ok=True)
    thoughts_path.write_text("\n".join(thoughts_lines))
    print(f"thoughts -> {thoughts_path}")


if __name__ == "__main__":
    main()
