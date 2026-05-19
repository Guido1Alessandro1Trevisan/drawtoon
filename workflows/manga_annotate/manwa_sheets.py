"""Gemini-driven manwa sheet construction.

Given a list of raw manhwa/webtoon page S3 keys for one chapter, this module:

  1. Stitches consecutive pages into 2-page chunks (so page-seam gutters are
     visible).
  2. Calls Gemini 3 Flash on each chunk: filter out title/cover/credits pages
     and return y-pixel coordinates of inter-panel gutters that the chapter
     can be safely split at.
  3. Builds the full kept-strip in memory and slices it at the global gutter
     y-positions into per-strip "sheets" that magi v3 then annotates.

Sheets are returned in memory (no S3 storage) so the magi worker can ingest
them directly.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import os
from typing import Any, Callable

import boto3
from PIL import Image

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_MODEL = os.environ.get("MANWA_SHEET_GEMINI_MODEL", "gemini-3-flash-preview")
DEFAULT_PAGES_PER_CHUNK = int(os.environ.get("MANWA_SHEET_PAGES_PER_CHUNK", "2"))
DEFAULT_MAX_PARALLEL = int(os.environ.get("MANWA_SHEET_MAX_PARALLEL", "10"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("MANWA_SHEET_MAX_OUTPUT_TOKENS", "1024"))
DEFAULT_MIN_SEGMENT_HEIGHT_PX = int(os.environ.get("MANWA_SHEET_MIN_SEGMENT_HEIGHT_PX", "96"))
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"

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

RESPONSE_SCHEMA = {
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


# ---------------------------------------------------------------------------
# Gemini client and image-bytes helpers
# ---------------------------------------------------------------------------


_GEMINI_CLIENT: Any = None
_GEMINI_API_KEY: str | None = None


def _resolve_gemini_api_key() -> str:
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY:
        return _GEMINI_API_KEY
    env_value = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if env_value:
        _GEMINI_API_KEY = env_value
        return _GEMINI_API_KEY
    if GEMINI_API_KEY_SECRET_NAME:
        client = boto3.client("secretsmanager", region_name=DEFAULT_REGION)
        secret = client.get_secret_value(SecretId=GEMINI_API_KEY_SECRET_NAME)
        secret_string = secret.get("SecretString")
        if secret_string:
            try:
                parsed = json.loads(secret_string)
                if isinstance(parsed, dict) and parsed.get(GEMINI_API_KEY_ENV):
                    _GEMINI_API_KEY = str(parsed[GEMINI_API_KEY_ENV]).strip()
                    return _GEMINI_API_KEY
            except json.JSONDecodeError:
                pass
            _GEMINI_API_KEY = secret_string.strip()
            return _GEMINI_API_KEY
    raise RuntimeError(
        f"Gemini API key not found. Set {GEMINI_API_KEY_ENV} env or secret {GEMINI_API_KEY_SECRET_NAME!r}."
    )


def _gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    from google import genai
    from google.genai import types

    _GEMINI_CLIENT = genai.Client(
        api_key=_resolve_gemini_api_key(),
        http_options=types.HttpOptions(timeout=180000),
    )
    return _GEMINI_CLIENT


# ---------------------------------------------------------------------------
# Chunk construction + classification
# ---------------------------------------------------------------------------


def _build_chunks(
    page_bytes: list[bytes],
    target_width: int,
    pages_per_chunk: int,
    page_global_offsets: list[int],
) -> list[dict[str, Any]]:
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
                "chunk_global_offset": int(page_global_offsets[start]),
            }
        )
    return chunks


def _classify_chunk(chunk: dict[str, Any], *, model: str) -> dict[str, Any]:
    from google.genai import types

    image_part = types.Part.from_bytes(data=chunk["chunk_bytes"], mime_type="image/jpeg")
    response = _gemini_client().models.generate_content(
        model=model,
        contents=[PROMPT_TEMPLATE, image_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            temperature=0,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
            ),
        ),
    )
    answer_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            txt = getattr(part, "text", None)
            if txt and not getattr(part, "thought", False):
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

    def _to_global(y: int) -> int:
        y = int(y)
        if y > 1000:
            return offset + y
        return offset + int(round(y * chunk_h / 1000.0))

    return {
        "chunk_index": int(chunk["chunk_index"]),
        "page_indices": list(chunk["page_indices"]),
        "is_title_page": bool(parsed.get("is_title_page", False)),
        "global_cuts": [_to_global(c.get("y") or 0) for c in raw_cuts],
        "usage": {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0,
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0) if usage else 0,
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_chapter_sheets(
    *,
    page_bytes: list[bytes],
    page_keys: list[str] | None = None,
    page_etags: list[str] | None = None,
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    min_segment_height_px: int = DEFAULT_MIN_SEGMENT_HEIGHT_PX,
    gemini_model: str = DEFAULT_MODEL,
    jpeg_quality: int = 92,
) -> dict[str, Any]:
    """Build in-memory manwa sheets for one chapter from raw page bytes.

    Returns dict with:
      - sheets: list of {sheet_index, sheet_id, image_bytes (JPEG), width,
                height, y_start_in_scroll, y_end_in_scroll, source_page_indices}
      - dropped_page_indices: pages flagged as title/cover and excluded
      - chunk_results: per-chunk diagnostic info (cuts, usage, is_title_page)
      - target_width, total_scroll_height
    """
    if not page_bytes:
        return {
            "sheets": [],
            "dropped_page_indices": [],
            "chunk_results": [],
            "target_width": 0,
            "total_scroll_height": 0,
        }
    page_keys = page_keys or [""] * len(page_bytes)
    page_etags = page_etags or [""] * len(page_bytes)
    # Decode page dimensions to compute global offsets. Width/height are
    # recorded per-slice so the trainer can detect bit-drift between
    # annotation-time and train-time reconstructions.
    target_width = 0
    page_heights: list[int] = []
    page_widths: list[int] = []
    for data in page_bytes:
        with Image.open(io.BytesIO(data)) as im:
            target_width = max(target_width, im.width)
            page_widths.append(im.width)
            page_heights.append(im.height)
    page_global_offsets: list[int] = []
    running = 0
    for h in page_heights:
        page_global_offsets.append(running)
        running += h
    total_scroll_height = running

    chunks = _build_chunks(page_bytes, target_width, pages_per_chunk, page_global_offsets)
    chunk_results: list[dict[str, Any]] = [None] * len(chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        future_to_idx = {
            ex.submit(_classify_chunk, ch, model=gemini_model): i
            for i, ch in enumerate(chunks)
        }
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
                    "error": str(exc),
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0,
                    },
                }

    dropped_page_indices: list[int] = []
    for r in chunk_results:
        if r and r.get("is_title_page"):
            dropped_page_indices.extend(r.get("page_indices") or [])

    # Aggregate global cuts; add hard cuts at every dropped page seam so dropped
    # pages never sit inside a sheet.
    raw_global_cuts: list[int] = []
    for r in chunk_results:
        if r:
            raw_global_cuts.extend(int(c) for c in (r.get("global_cuts") or []))
    for idx in dropped_page_indices:
        raw_global_cuts.append(page_global_offsets[idx])
        raw_global_cuts.append(page_global_offsets[idx] + page_heights[idx])
    global_cuts = sorted({c for c in raw_global_cuts if 0 < c < total_scroll_height})
    pruned: list[int] = []
    min_seg = max(16, int(min_segment_height_px))
    for c in global_cuts:
        if not pruned or c - pruned[-1] >= min_seg:
            pruned.append(c)
    global_cuts = pruned

    # Build the kept-pages strip (drop title-flagged pages entirely).
    dropped_set = set(dropped_page_indices)
    kept_indices = [i for i in range(len(page_bytes)) if i not in dropped_set]
    if not kept_indices:
        return {
            "sheets": [],
            "dropped_page_indices": dropped_page_indices,
            "chunk_results": chunk_results,
            "target_width": int(target_width),
            "total_scroll_height": int(total_scroll_height),
        }
    strip_h = sum(page_heights[i] for i in kept_indices)
    strip = Image.new("RGB", (int(target_width), int(strip_h)), "white")
    y = 0
    kept_starts: dict[int, int] = {}
    for idx in kept_indices:
        kept_starts[idx] = y
        im = Image.open(io.BytesIO(page_bytes[idx])).convert("RGB")
        if im.width != target_width:
            new_h = int(round(im.height * target_width / im.width))
            im = im.resize((int(target_width), new_h), Image.Resampling.LANCZOS)
        strip.paste(im, (0, y))
        y += im.height
        im.close()

    # Translate global_cuts into strip-local coords (skip cuts inside dropped pages).
    strip_cuts: list[int] = []
    for c in global_cuts:
        for idx in range(len(page_bytes)):
            p_start = page_global_offsets[idx]
            p_end = p_start + page_heights[idx]
            if p_start <= c < p_end:
                if idx in dropped_set:
                    break
                strip_cuts.append(kept_starts[idx] + (c - p_start))
                break
    strip_cuts = sorted(set(strip_cuts))

    boundaries = [0] + strip_cuts + [strip_h]
    sheets: list[dict[str, Any]] = []
    for i, (y0, y1) in enumerate(zip(boundaries, boundaries[1:])):
        if y1 <= y0:
            continue
        sheet = strip.crop((0, y0, strip.width, y1))
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        sheet.close()
        # Per-sheet provenance: which source page contributes which y-band of
        # the sheet image. The trainer uses these slices to reconstruct the
        # sheet on-the-fly from the always-available raw source pages — no
        # sheet JPEG is stored on S3.
        slices: list[dict[str, Any]] = []
        for idx in kept_indices:
            kept_start = kept_starts[idx]
            kept_end = kept_start + page_heights[idx]
            overlap_start = max(y0, kept_start)
            overlap_end = min(y1, kept_end)
            if overlap_end <= overlap_start:
                continue
            slices.append(
                {
                    "source_page_index": int(idx),
                    "source_page_key": str(page_keys[idx]),
                    "source_page_width": int(page_widths[idx]),
                    "source_page_height": int(page_heights[idx]),
                    "source_page_etag": str(page_etags[idx] or ""),
                    "source_y_start": int(overlap_start - kept_start),
                    "source_y_end": int(overlap_end - kept_start),
                    "sheet_y_start": int(overlap_start - y0),
                    "sheet_y_end": int(overlap_end - y0),
                }
            )
        sheets.append(
            {
                "sheet_index": i,
                "sheet_id": f"sheet_{i:04d}",
                "image_bytes": buf.getvalue(),
                "width": int(strip.width),
                "height": int(y1 - y0),
                "y_start_in_scroll": int(y0),
                "y_end_in_scroll": int(y1),
                "source_page_indices": [s["source_page_index"] for s in slices],
                "slices": slices,
            }
        )
    strip.close()

    return {
        "sheets": sheets,
        "dropped_page_indices": dropped_page_indices,
        "chunk_results": chunk_results,
        "target_width": int(target_width),
        "total_scroll_height": int(total_scroll_height),
        "global_cuts": global_cuts,
    }
