"""Per-page row construction for the LayouSyn panel-layout JSONL.

Takes one source caption JSON (geometry from manga_caption) + one
compressed-caption JSON (short_caption per panel from compress_captions
vision_v1) and produces the LayouSyn training rows for that page.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .attribution import attribute_bubbles, label_for_bubble


ASPECT_MIN = 0.2
ASPECT_MAX = 5.0
MIN_PANEL_AREA_FRAC = 0.005
MAX_CHARACTERS = 5
MAX_BUBBLES = 7


def _bbox_norm_to_pm1(bbox01: list[float]) -> list[float]:
    return [round(float(v) * 2.0 - 1.0, 6) for v in bbox01]


def _reading_order_key(entry: dict[str, Any], *, rtl: bool) -> tuple[int, float]:
    bbox = entry.get("panel_bbox_norm") or entry.get("bbox_norm") or [0, 0, 0, 0]
    y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
    x_center = (float(bbox[0]) + float(bbox[2])) / 2.0
    row_bucket = int(y_center * 10)
    return (row_bucket, -x_center if rtl else x_center)


def chapter_is_rtl(chapter: str) -> bool:
    """drawtoon convention: `_manga`-suffixed slugs are RTL, `_manwa` are LTR."""
    c = chapter.lower()
    if c.endswith("_manwa"):
        return False
    if c.endswith("_manga") or c.endswith("_manga109") or c.endswith("_mangazero"):
        return True
    if "_manga109_" in c or "_mangazero_" in c:
        return True
    return True


def chapter_is_val(chapter: str, val_frac: float) -> bool:
    if val_frac <= 0:
        return False
    digest = hashlib.sha256(chapter.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_frac


def panel_to_row(
    *,
    source_panel: dict[str, Any],
    short_caption: str,
    page_width_px: int,
    page_height_px: int,
    chapter: str,
) -> dict[str, Any] | None:
    panel_bbox = source_panel.get("bbox")
    if not isinstance(panel_bbox, (list, tuple)) or len(panel_bbox) != 4:
        return None
    px0, py0, px1, py1 = (float(v) for v in panel_bbox)
    panel_w_px = max(1.0, px1 - px0)
    panel_h_px = max(1.0, py1 - py0)
    if page_width_px <= 0 or page_height_px <= 0:
        return None

    width_norm = round(panel_w_px / page_width_px, 6)
    height_norm = round(panel_h_px / page_height_px, 6)
    aspect = width_norm / max(1e-3, height_norm)
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        return {"_dropped": "aspect_oob"}
    if (width_norm * height_norm) < MIN_PANEL_AREA_FRAC:
        return {"_dropped": "tiny_panel"}

    characters = source_panel.get("characters") or []
    text_bubbles = source_panel.get("text_bubbles") or []
    tails = source_panel.get("tails") or []

    rtl = chapter_is_rtl(chapter)
    characters_sorted = sorted(
        [c for c in characters if isinstance(c, dict) and c.get("panel_bbox_norm")],
        key=lambda c: _reading_order_key(c, rtl=rtl),
    )[:MAX_CHARACTERS]
    bubbles_sorted = sorted(
        [b for b in text_bubbles if isinstance(b, dict) and b.get("panel_bbox_norm")],
        key=lambda b: _reading_order_key(b, rtl=rtl),
    )[:MAX_BUBBLES]

    attributed = attribute_bubbles(
        characters=characters_sorted,
        text_bubbles=bubbles_sorted,
        tails=[t for t in tails if isinstance(t, dict) and t.get("panel_bbox_norm")],
    )

    items: list[dict[str, Any]] = []
    for ordinal, ch in enumerate(characters_sorted, start=1):
        items.append(
            {
                "label": f"character {ordinal}",
                "bbox": _bbox_norm_to_pm1(ch["panel_bbox_norm"]),
            }
        )
    for bubble, record in zip(bubbles_sorted, attributed):
        items.append(
            {
                "label": label_for_bubble(record),
                "bbox": _bbox_norm_to_pm1(bubble["panel_bbox_norm"]),
            }
        )

    if not items:
        return {"_dropped": "no_items"}

    return {
        "prompt": short_caption.strip(),
        "width": width_norm,
        "height": height_norm,
        "items": items,
        "_meta": {
            "chapter": chapter,
            "panel_id": str(source_panel.get("panel_id") or ""),
            "panel_bbox_px": [round(v, 2) for v in panel_bbox],
            "attribution_methods": [r.get("method") for r in attributed],
        },
    }


def build_rows_for_page(
    *,
    source_payload: dict[str, Any],
    short_payload: dict[str, Any],
    chapter: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (rows, stats) for the page."""
    stats = {
        "panels_total": 0,
        "panels_emitted": 0,
        "panels_dropped_aspect": 0,
        "panels_dropped_tiny": 0,
        "panels_dropped_no_items": 0,
        "panels_missing_short": 0,
        "attribution_tail": 0,
        "attribution_centroid": 0,
        "attribution_none": 0,
        "attribution_narration": 0,
        "attribution_off_panel": 0,
        "bubble_speech": 0,
        "bubble_shout": 0,
        "bubble_narration": 0,
    }
    page_size = source_payload.get("page_size") or {}
    page_w = int(page_size.get("width_px") or 0)
    page_h = int(page_size.get("height_px") or 0)
    if page_w <= 0 or page_h <= 0:
        return [], stats

    short_by_idx: dict[int, str] = {}
    for p in short_payload.get("panels") or []:
        try:
            idx = int(p.get("panel_index"))
        except (TypeError, ValueError):
            continue
        cap = str(p.get("short_caption") or "").strip()
        if cap:
            short_by_idx[idx] = cap

    rows: list[dict[str, Any]] = []
    for panel in source_payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        stats["panels_total"] += 1
        try:
            idx = int(panel.get("panel_index"))
        except (TypeError, ValueError):
            continue
        prompt = short_by_idx.get(idx)
        if not prompt:
            stats["panels_missing_short"] += 1
            continue
        row = panel_to_row(
            source_panel=panel,
            short_caption=prompt,
            page_width_px=page_w,
            page_height_px=page_h,
            chapter=chapter,
        )
        if row is None or "_dropped" in row:
            reason = (row or {}).get("_dropped") or "no_items"
            if reason == "aspect_oob":
                stats["panels_dropped_aspect"] += 1
            elif reason == "tiny_panel":
                stats["panels_dropped_tiny"] += 1
            else:
                stats["panels_dropped_no_items"] += 1
            continue
        for method in row["_meta"]["attribution_methods"]:
            key = f"attribution_{method or 'none'}"
            stats[key] = stats.get(key, 0) + 1
        for item in row["items"]:
            label = item["label"]
            if label.startswith("speech bubble"):
                stats["bubble_speech"] += 1
            elif label.startswith("shout bubble"):
                stats["bubble_shout"] += 1
            elif label.startswith("narration bubble"):
                stats["bubble_narration"] += 1
        rows.append(row)
        stats["panels_emitted"] += 1
    return rows, stats
