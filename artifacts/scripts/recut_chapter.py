"""Recut a manwa series into uniformly-sized "singles" by stitching each
chapter/episode and cutting at row-RGB-uniform gutters.

Pipeline per episode:
  1. List all page S3 keys for the episode (grouped by subdir or filename prefix).
  2. Download each page, decode, normalize widths.
  3. Stitch vertically into one tall PIL image (the strip).
  4. Detect gutter cuts using the baseline row-uniformity test:
       - per row: sample 96 horizontal positions
       - require horizontal-range <= TOLERANCE on every R/G/B channel
       - require luma >= 238, luma <= 35, or saturation <= 18
       - run-length >= MIN_BAND_HEIGHT_PX to count as a gutter
  5. Cut the strip at every gutter midpoint; require each output segment
     to be at least MIN_SEGMENT_HEIGHT_PX tall.
  6. Save each segment as a new JPEG single, locally for smoke + optionally
     uploaded to S3 (parallel prefix until verified).

Smoke run (default):
    uv run --active --with boto3 --with pillow --with numpy \
        python artifacts/scripts/recut_chapter.py
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
from botocore.config import Config
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET = "drawtoon"
SOURCE_PREFIX = "datasets/pages/single"
STAGING_PREFIX = "datasets/pages/single_recut"

# Detector knobs (matched to manhwa_raw_sheets.py defaults)
MIN_BAND_HEIGHT_PX = 18
COLOR_TOLERANCE = 10
SAT_TOL = 18
LUMA_BRIGHT = 238.0
LUMA_DARK = 35.0
SAMPLE_COLUMNS = 96

# Output-shape knobs (recut singles)
MIN_SEGMENT_HEIGHT_PX = 320   # don't emit slivers
JPEG_QUALITY = 92

s3 = boto3.Session().client(
    "s3",
    config=Config(max_pool_connections=128, retries={"mode": "adaptive", "max_attempts": 4}),
)


# ---------------------------------------------------------------------------
# S3 listing + episode grouping
# ---------------------------------------------------------------------------


def list_keys(prefix: str) -> list[dict]:
    """Return list of {key, size} for image objects under prefix."""
    out = []
    p = s3.get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj["Key"]
            if k.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                out.append({"key": k, "size": int(obj.get("Size") or 0)})
    return out


PAGE_NUMBER_RE = re.compile(r"page[-_]?(\d+)", re.IGNORECASE)


def group_by_episode(series: str, keys: list[dict]) -> dict[str, list[dict]]:
    """Group page keys into episode buckets. Returns {episode_id: [pages]}.

    The episode is detected as:
      - the sub-directory of the series prefix, if any (e.g. chapter-000001/)
      - else everything before "__page-" in the filename
      - else the literal "_root" (flat series, the whole series is one group)
    Pages within a group are sorted by trailing page number.
    """
    series_prefix = f"{SOURCE_PREFIX}/{series}/"
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in keys:
        rel = entry["key"][len(series_prefix):]
        parts = rel.split("/")
        if len(parts) >= 2:
            episode_id = parts[0]
        else:
            stem = Path(rel).stem
            if "__page-" in stem:
                episode_id = stem.split("__page-")[0]
            elif "__page_" in stem:
                episode_id = stem.split("__page_")[0]
            else:
                episode_id = "_root"
        groups[episode_id].append(entry)
    for v in groups.values():
        def page_num(e):
            m = PAGE_NUMBER_RE.search(e["key"])
            return int(m.group(1)) if m else 0
        v.sort(key=lambda e: (page_num(e), e["key"]))
    return dict(groups)


# ---------------------------------------------------------------------------
# Stitch + detect + cut (pure NumPy)
# ---------------------------------------------------------------------------


def stitch_pages(page_bytes_list: list[bytes]) -> tuple[Image.Image, list[int]]:
    """Decode + normalize widths + vertical stack. Returns (strip, page_offsets)."""
    decoded = []
    target_width = 0
    for data in page_bytes_list:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        decoded.append(im)
        target_width = max(target_width, im.width)
    resized = []
    for im in decoded:
        if im.width == target_width:
            resized.append(im)
        else:
            new_h = int(round(im.height * target_width / im.width))
            resized.append(im.resize((target_width, new_h), Image.Resampling.LANCZOS))
            im.close()
    total_h = sum(im.height for im in resized)
    strip = Image.new("RGB", (target_width, total_h), "white")
    offsets = []
    y = 0
    for im in resized:
        offsets.append(y)
        strip.paste(im, (0, y))
        y += im.height
        im.close()
    return strip, offsets


def detect_row_uniform_cuts(strip: Image.Image) -> tuple[list[int], list[bool]]:
    """Row-uniformity detector, vectorized over numpy. Returns (cuts, flat_rows)."""
    arr = np.asarray(strip)  # H x W x 3
    h, w = arr.shape[:2]
    step = max(1, w // SAMPLE_COLUMNS)
    xs = np.arange(0, w, step)
    if xs.size < 4:
        xs = np.linspace(0, w - 1, num=min(w, 4), dtype=int)
    sampled = arr[:, xs, :]  # H x S x 3

    horizontal_range = (sampled.max(axis=1) - sampled.min(axis=1)).max(axis=1)  # H
    means = sampled.mean(axis=1)                                                # H x 3
    luma = 0.2126 * means[:, 0] + 0.7152 * means[:, 1] + 0.0722 * means[:, 2]
    saturation_proxy = means.max(axis=1) - means.min(axis=1)
    is_flat_color = (
        (luma >= LUMA_BRIGHT)
        | (luma <= LUMA_DARK)
        | (saturation_proxy <= SAT_TOL)
    )
    flat_rows = (horizontal_range <= COLOR_TOLERANCE) & is_flat_color

    cuts: list[int] = []
    y = 0
    n = int(h)
    while y < n:
        if not flat_rows[y]:
            y += 1
            continue
        start = y
        while y < n and flat_rows[y]:
            y += 1
        end = y
        if end - start < MIN_BAND_HEIGHT_PX:
            continue
        midpoint = (start + end) // 2
        if midpoint < MIN_SEGMENT_HEIGHT_PX or n - midpoint < MIN_SEGMENT_HEIGHT_PX:
            continue
        cuts.append(midpoint)
    deduped: list[int] = []
    for c in cuts:
        if not deduped or c - deduped[-1] >= MIN_SEGMENT_HEIGHT_PX:
            deduped.append(c)
    return deduped, flat_rows.tolist()


def cut_to_segments(strip: Image.Image, cuts: list[int]) -> list[Image.Image]:
    boundaries = [0] + cuts + [strip.height]
    segs = []
    for y0, y1 in zip(boundaries, boundaries[1:]):
        if y1 - y0 < MIN_SEGMENT_HEIGHT_PX:
            continue
        segs.append(strip.crop((0, y0, strip.width, y1)))
    return segs


def render_overlay(strip: Image.Image, cuts: list[int], page_offsets: list[int], label: str) -> Image.Image:
    THUMB_W = 360
    scale = THUMB_W / strip.width
    thumb = strip.resize((THUMB_W, int(round(strip.height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(thumb)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=10)
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=15)
    except Exception:
        font = ImageFont.load_default()
        big = font
    for i, off in enumerate(page_offsets):
        y = int(round(off * scale))
        for x in range(0, thumb.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=(60, 140, 240), width=1)
        if i > 0:
            draw.text((2, y - 12), f"p{i + 1}", fill=(60, 140, 240), font=font)
    for idx, c in enumerate(cuts):
        y = int(round(c * scale))
        draw.line([(0, y), (thumb.width, y)], fill=(240, 60, 60), width=2)
        draw.text((thumb.width - 28, y + 2), f"{idx}", fill=(240, 60, 60), font=font)
    header_h = 32
    out = Image.new("RGB", (thumb.width, thumb.height + header_h), (0, 0, 0))
    out.paste(thumb, (0, header_h))
    ImageDraw.Draw(out).text((6, 6), label, fill=(255, 255, 255), font=big)
    return out


# ---------------------------------------------------------------------------
# Per-episode pipeline
# ---------------------------------------------------------------------------


def process_episode(
    series: str,
    episode_id: str,
    page_entries: list[dict],
    *,
    local_out: Path,
    upload_to_s3: bool,
) -> dict:
    t0 = time.perf_counter()
    page_bytes = []
    for e in page_entries:
        page_bytes.append(s3.get_object(Bucket=BUCKET, Key=e["key"])["Body"].read())
    strip, page_offsets = stitch_pages(page_bytes)
    print(f"  {episode_id}: stitched {len(page_bytes)} pages -> {strip.size} in {time.perf_counter()-t0:.1f}s", flush=True)
    cuts, _ = detect_row_uniform_cuts(strip)
    segments = cut_to_segments(strip, cuts)
    print(f"    cuts={len(cuts)} segments={len(segments)} min={min((s.height for s in segments), default=0)} max={max((s.height for s in segments), default=0)}", flush=True)

    local_out.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(strip, cuts, page_offsets, f"{series}/{episode_id}  cuts={len(cuts)}")
    overlay_path = local_out / f"{episode_id}_overlay.jpg"
    overlay.save(overlay_path, format="JPEG", quality=82, optimize=True)
    saved_keys: list[str] = []
    for idx, seg in enumerate(segments):
        out_name = f"{episode_id}__page-{idx + 1:04d}.jpg"
        buf = io.BytesIO()
        seg.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        seg.close()
        data = buf.getvalue()
        (local_out / out_name).write_bytes(data)
        if upload_to_s3:
            s3_key = f"{STAGING_PREFIX}/{series}/{out_name}"
            s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data, ContentType="image/jpeg")
            saved_keys.append(s3_key)
    strip.close()
    overlay.close()
    return {
        "series": series,
        "episode_id": episode_id,
        "source_pages": len(page_entries),
        "strip_height": strip.height,
        "cuts": cuts,
        "segments_written": len(segments),
        "overlay": str(overlay_path.relative_to(REPO_ROOT)),
        "s3_keys": saved_keys,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="the-mafia-nanny_manwa")
    ap.add_argument("--episode", default="", help="Specific episode id; default = first episode found")
    ap.add_argument(
        "--all-episodes",
        action="store_true",
        help="Process every episode in the series instead of just one",
    )
    ap.add_argument("--upload", action="store_true", help="Upload to S3 staging path")
    ap.add_argument(
        "--local-root",
        default=str(REPO_ROOT / "artifacts" / "recut_smoke"),
    )
    args = ap.parse_args()

    print(f"series: {args.series}")
    keys = list_keys(f"{SOURCE_PREFIX}/{args.series}/")
    if not keys:
        print(f"  no keys under {SOURCE_PREFIX}/{args.series}/", file=sys.stderr)
        sys.exit(1)
    groups = group_by_episode(args.series, keys)
    print(f"  episodes detected: {len(groups)}")
    for eid, pages in list(groups.items())[:5]:
        print(f"    {eid}: {len(pages)} pages (e.g. {pages[0]['key'].split('/')[-1]})")
    if len(groups) > 5:
        print(f"    ... and {len(groups) - 5} more")

    local_root = Path(args.local_root) / args.series
    targets: list[str]
    if args.all_episodes:
        targets = sorted(groups.keys())
    else:
        if args.episode:
            if args.episode not in groups:
                print(f"  episode {args.episode!r} not found; available: {sorted(groups)[:5]}", file=sys.stderr)
                sys.exit(1)
            targets = [args.episode]
        else:
            targets = [sorted(groups.keys())[0]]

    summary = []
    for eid in targets:
        summary.append(
            process_episode(
                args.series,
                eid,
                groups[eid],
                local_out=local_root / eid,
                upload_to_s3=args.upload,
            )
        )

    (local_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {local_root / 'summary.json'}")


if __name__ == "__main__":
    main()
