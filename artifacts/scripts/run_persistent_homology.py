"""Persistent-homology valley detector for manhwa gutters.

Stitches the full chapter into a single column-averaged luma profile B[y],
detrends it with a moving median (window 801 px) and computes a 1D
black-hat morphological signal. Persistent homology is then run on the
detrended profile (sublevel-set filtration on -B_d via `cripser`), and the
births of long-lived 0-dimensional features become the cut candidates.

The chosen `tau` is the smallest value in {5, 10, 20, 40, 80} that yields
between 25 and 50 cuts; ties are broken by preferring more cuts. Each
candidate is required to be at least `MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX`
away from the chapter start / end and from any neighbouring candidate.

Run:
    uv run --active --with opencv-python-headless --with numpy --with pillow \\
        --with boto3 --with cripser --with scipy \\
        python artifacts/scripts/run_persistent_homology.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import cripser
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import grey_closing, median_filter

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
    MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
    MANHWA_RAW_SHEET_PADDING_PX,
    MANHWA_RAW_SHEET_SPAN_THRESHOLD_PX,
    write_sheets,
)

DETECTOR_LABEL = "persistent_homology"
DETREND_WINDOW = 801
BLACKHAT_WINDOW = 151
TAU_SWEEP = (5, 10, 20, 40, 80)
TARGET_CUT_RANGE = (25, 50)
DARK_FIRE_PAGES = (7, 23)  # 1-indexed page range that should get split

# Overlay rendering constants (mirrors visualize_cuts.py).
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


# ---------------------------------------------------------------------------
# Stitch the chapter into a single column-averaged luma profile and keep a
# resized RGB scroll for the overlay rendering.
# ---------------------------------------------------------------------------


def build_scroll_and_profile(
    keys: list[str],
) -> tuple[Image.Image, list[tuple[int, int]], np.ndarray, int]:
    images: list[Image.Image] = []
    target_width = 0
    for key in keys:
        im = Image.open(io.BytesIO(get_bytes(key))).convert("RGB")
        images.append(im)
        target_width = max(target_width, im.width)
    # Resize every page to a common width so column averages stack cleanly.
    spans: list[tuple[int, int]] = []
    canvas_height = 0
    rescaled: list[Image.Image] = []
    for im in images:
        if im.width != target_width:
            scaled_h = int(round(im.height * target_width / im.width))
            new_im = im.resize((target_width, scaled_h), Image.Resampling.LANCZOS)
            im.close()
        else:
            new_im = im
        rescaled.append(new_im)
        spans.append((canvas_height, canvas_height + new_im.height))
        canvas_height += new_im.height

    scroll = Image.new("RGB", (target_width, canvas_height), "white")
    profile = np.empty(canvas_height, dtype=np.float64)
    y = 0
    for im in rescaled:
        scroll.paste(im, (0, y))
        arr = np.asarray(im, dtype=np.float32)
        # Rec.709 luma
        luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        profile[y : y + im.height] = luma.mean(axis=1)
        y += im.height
        im.close()
    return scroll, spans, profile, target_width


# ---------------------------------------------------------------------------
# Persistent-homology valley detection.
# ---------------------------------------------------------------------------


def compute_persistence_pairs(profile: np.ndarray) -> np.ndarray:
    """Run sublevel-set persistent homology on -B_d via cripser.

    Pass the signal as a 1D `(N,)` array — cripser treats a `(1, N)` input as a
    2D image of height 1, which collapses the entire sequence into a single
    connected component and yields only one (inf-persistence) pair.
    """
    signal = (-profile).astype(np.float64)
    pd = cripser.computePH(signal, maxdim=0)
    # cripser columns: [dim, birth, death, x0, y0, z0, x1, y1, z1].
    # For a 1D input, x0 is the index of the birth (local minimum) point.
    return pd


def detrend_profile(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = median_filter(profile, size=DETREND_WINDOW, mode="reflect")
    detrended = profile - base
    blackhat = grey_closing(profile, size=BLACKHAT_WINDOW, mode="reflect") - profile
    return detrended, base, blackhat


def cuts_from_persistence(
    pd: np.ndarray,
    *,
    tau: float,
    profile_len: int,
    min_segment_height: int,
) -> list[int]:
    if pd.size == 0:
        return []
    births = pd[:, 1]
    deaths = pd[:, 2]
    # Bound infinite-lifetime features by the maximum finite filtration value so
    # they cannot dominate the persistence ranking.
    finite_mask = np.isfinite(deaths)
    if finite_mask.any():
        max_finite = float(deaths[finite_mask].max())
    else:
        max_finite = float(births.max()) + 1.0
    deaths = np.where(finite_mask, deaths, max_finite)
    persistence = deaths - births
    locations = pd[:, 3].astype(np.int64)
    survivors = np.where(persistence >= float(tau))[0]
    if survivors.size == 0:
        return []
    # Sort by descending persistence so the strongest valleys win the spacing
    # competition first.
    survivors = survivors[np.argsort(-persistence[survivors])]
    accepted_sorted: list[int] = []
    for idx in survivors:
        loc = int(locations[idx])
        if loc < min_segment_height or profile_len - loc < min_segment_height:
            continue
        if any(abs(loc - other) < min_segment_height for other in accepted_sorted):
            continue
        accepted_sorted.append(loc)
    return sorted(accepted_sorted)


def pick_tau(
    pd: np.ndarray, *, profile_len: int, min_segment_height: int
) -> tuple[float, list[int], list[dict[str, Any]]]:
    sweep_results: list[dict[str, Any]] = []
    chosen: tuple[float, list[int]] | None = None
    for tau in TAU_SWEEP:
        cuts = cuts_from_persistence(
            pd,
            tau=float(tau),
            profile_len=profile_len,
            min_segment_height=min_segment_height,
        )
        sweep_results.append({"tau": int(tau), "cut_count": len(cuts)})
        if TARGET_CUT_RANGE[0] <= len(cuts) <= TARGET_CUT_RANGE[1] and chosen is None:
            chosen = (float(tau), cuts)
    if chosen is None:
        # Fall back to the tau whose count is closest to the target band, with a
        # bias towards more cuts (we'd rather over-split than miss a valley).
        def score(item: dict[str, Any]) -> tuple[int, int]:
            count = int(item["cut_count"])
            if count < TARGET_CUT_RANGE[0]:
                return (TARGET_CUT_RANGE[0] - count, -count)
            if count > TARGET_CUT_RANGE[1]:
                return (count - TARGET_CUT_RANGE[1], count)
            return (0, -count)

        best = min(sweep_results, key=score)
        tau = float(best["tau"])
        cuts = cuts_from_persistence(
            pd,
            tau=tau,
            profile_len=profile_len,
            min_segment_height=min_segment_height,
        )
        chosen = (tau, cuts)
    return chosen[0], chosen[1], sweep_results


# ---------------------------------------------------------------------------
# Segment / sheet construction.
# ---------------------------------------------------------------------------


def build_pages_with_dimensions(
    pages: list[dict[str, Any]], spans: list[tuple[int, int]], target_width: int
) -> None:
    for accepted_index, (page, (y_start, y_end)) in enumerate(zip(pages, spans)):
        page["accepted_page_index"] = accepted_index
        page["width"] = int(target_width)
        page["height"] = int(y_end - y_start)
        page["global_y_start"] = int(y_start)
        page["global_y_end"] = int(y_end)


def build_segments_from_global_cuts(
    *,
    pages: list[dict[str, Any]],
    cuts: list[int],
    total_height: int,
    target_width: int,
    min_segment_height: int,
) -> list[dict[str, Any]]:
    boundaries = sorted({0, total_height, *cuts})
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
                "source_height": source_height,
                "rendered_width": int(target_width),
                "rendered_height": int(rendered_height),
                "source_slices": source_slices,
                "page_indices": sorted({int(item["page_index"]) for item in source_slices}),
                "page_span": [int(source_slices[0]["page_index"]), int(source_slices[-1]["page_index"])],
            }
        )
    return segments


# ---------------------------------------------------------------------------
# Overlay rendering (mirrors visualize_cuts.py).
# ---------------------------------------------------------------------------


def render_overlay(
    scroll: Image.Image,
    spans: list[tuple[int, int]],
    cuts: list[int],
    label: str,
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
    # Page boundaries (blue dashes + p<n> labels).
    for idx, (_, y_end) in enumerate(spans[:-1]):
        y = int(round(y_end * scale))
        for x in range(0, overlay.width, 8):
            draw.line([(x, y), (x + 4, y)], fill=PAGE_BOUNDARY_COLOR, width=1)
        draw.text((2, y - 13), f"p{idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    # Final page label.
    if spans:
        last_idx = len(spans) - 1
        y_end_last = int(round(spans[last_idx][1] * scale)) - 14
        draw.text((2, max(0, y_end_last)), f"p{last_idx + 1}", fill=PAGE_BOUNDARY_COLOR, font=font)
    # Detected cuts (red solid lines).
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


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def main() -> None:
    keys = list_chapter_pages(BUCKET, INPUT_PREFIX, CHAPTER_RELATIVE)
    if not keys:
        raise SystemExit(f"No pages found at s3://{BUCKET}/{INPUT_PREFIX}/{CHAPTER_RELATIVE}")
    print(f"pages: {len(keys)}")
    pages = build_pages_for_chapter(keys, INPUT_PREFIX)
    scroll, spans, profile, target_width = build_scroll_and_profile(keys)
    total_height = int(profile.shape[0])
    print(f"scroll: {scroll.size}, profile length: {total_height}")
    build_pages_with_dimensions(pages, spans, target_width)

    detrended, _base, blackhat = detrend_profile(profile)
    # The detrended profile drives PH directly; the black-hat signal is a
    # secondary valley emphasiser we save for diagnostics / inspection.
    pd = compute_persistence_pairs(detrended)
    min_segment_height = max(16, int(MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX))
    tau, cuts, sweep = pick_tau(
        pd, profile_len=total_height, min_segment_height=min_segment_height
    )

    print(f"persistence pairs: {pd.shape[0]}")
    print(f"tau sweep: {sweep}")
    print(f"chosen tau={tau} -> {len(cuts)} cuts (min_segment={min_segment_height}px)")

    segments = build_segments_from_global_cuts(
        pages=pages,
        cuts=cuts,
        total_height=total_height,
        target_width=target_width,
        min_segment_height=min_segment_height,
    )
    if segments:
        heights = [int(seg["rendered_height"] or 0) for seg in segments]
        print(
            f"segments: {len(segments)} | height min={min(heights)} median={int(np.median(heights))} max={max(heights)}"
        )
    else:
        print("segments: 0")

    # Dark-fire stretch check (pages 7-23 inclusive, 1-indexed).
    if spans:
        dark_start = spans[DARK_FIRE_PAGES[0] - 1][0]
        dark_end = spans[DARK_FIRE_PAGES[1] - 1][1]
    else:
        dark_start = dark_end = 0
    cuts_inside_dark = [c for c in cuts if dark_start <= c <= dark_end]
    dark_fire_split = bool(cuts_inside_dark)
    print(
        f"dark-fire stretch pages {DARK_FIRE_PAGES[0]}-{DARK_FIRE_PAGES[1]} "
        f"(y={dark_start}..{dark_end}): {len(cuts_inside_dark)} cuts inside -> "
        f"{'SPLIT' if dark_fire_split else 'NOT split'}"
    )

    config = {
        "manhwa": {
            "raw_min_segment_height_px": MANHWA_RAW_MIN_SEGMENT_HEIGHT_PX,
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
        target_width=int(target_width),
        config=config,
        get_bytes=get_bytes,
        put_bytes=put_bytes,
        make_uri=make_uri,
    )

    overlay = render_overlay(scroll, spans, cuts, DETECTOR_LABEL)
    overlay_key = f"{output_prefix}/{CHAPTER_RELATIVE}/cuts_overlay.jpg"
    put_bytes(overlay_key, encode_jpeg(overlay), "image/jpeg")

    manifest = {
        "schema_version": 1,
        "detector": DETECTOR_LABEL,
        "chapter_key": CHAPTER_RELATIVE,
        "input_prefix": INPUT_PREFIX,
        "output_prefix": output_prefix,
        "target_width": int(target_width),
        "total_virtual_height": total_height,
        "page_count": len(pages),
        "page_spans": [[int(s), int(e)] for s, e in spans],
        "persistent_homology": {
            "detrend_window": DETREND_WINDOW,
            "blackhat_window": BLACKHAT_WINDOW,
            "blackhat_max": float(np.max(blackhat)) if blackhat.size else 0.0,
            "tau_sweep": sweep,
            "chosen_tau": float(tau),
            "persistence_pair_count": int(pd.shape[0]),
            "cut_count": len(cuts),
            "cuts": [int(c) for c in cuts],
            "dark_fire_pages": list(DARK_FIRE_PAGES),
            "dark_fire_window": [int(dark_start), int(dark_end)],
            "dark_fire_cuts": [int(c) for c in cuts_inside_dark],
            "dark_fire_split": dark_fire_split,
        },
        "segmentation": {
            "total_virtual_height": total_height,
            "target_width": int(target_width),
            "detected_gutter_cuts": len(cuts),
            "cut_count": len(cuts) + 2,
        },
        "segment_count": len(segments),
        "sheet_count": int(sheet_result.get("sheet_count") or 0),
        "segments": segments,
        "sheets": sheet_result.get("sheets") or [],
        "cuts_overlay_key": overlay_key,
        "cuts_overlay_uri": make_uri(overlay_key),
    }
    manifest_key = f"{output_prefix}/{CHAPTER_RELATIVE}/manifest.json"
    put_bytes(manifest_key, json.dumps(manifest, indent=2).encode("utf-8"), "application/json")
    local_manifest = LOCAL_OUTPUT_ROOT / Path(manifest_key).relative_to(OUTPUT_BASE)
    local_overlay = LOCAL_OUTPUT_ROOT / Path(overlay_key).relative_to(OUTPUT_BASE)
    print(f"manifest -> {local_manifest}")
    print(f"overlay  -> {local_overlay}")
    print(f"sheets   -> {manifest['sheet_count']} files under {local_manifest.parent}")


if __name__ == "__main__":
    main()
