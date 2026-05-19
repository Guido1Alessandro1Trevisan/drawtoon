"""Tail-aware speaker attribution for manga speech/shout bubbles.

Vendored copy of `lora-klein/creatilayout/data_prep/attribution.py`. Keep in
sync with that file (same algorithm; this copy lives in the Lambda's `src/`
so the bundle is self-contained).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional


BUBBLE_TYPE_TOKENS: dict[str, str] = {
    "Speech Bubble": "speech bubble",
    "Shout Bubble": "shout bubble",
    "Narration Bubble": "narration bubble",
}

NARRATION_TOKEN = "narration bubble"

DEFAULT_TAIL_OVERLAP_MIN = 0.2
DEFAULT_MAX_TIP_DISTANCE_FRAC = 0.30


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _overlap_area(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_point_to_bbox(point: tuple[float, float], bbox: list[float]) -> float:
    px, py = point
    dx = max(bbox[0] - px, 0.0, px - bbox[2])
    dy = max(bbox[1] - py, 0.0, py - bbox[3])
    return math.hypot(dx, dy)


def _panel_diag(panel_w_norm: float, panel_h_norm: float) -> float:
    return math.hypot(max(1e-6, panel_w_norm), max(1e-6, panel_h_norm))


def _panel_bbox(entry: dict[str, Any]) -> Optional[list[float]]:
    bbox = entry.get("panel_bbox_norm") or entry.get("bbox_norm")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [float(v) for v in bbox]


def _tail_tip(tail_bbox: list[float], bubble_bbox: list[float]) -> tuple[float, float]:
    bubble_c = _centroid(bubble_bbox)
    corners = [
        (tail_bbox[0], tail_bbox[1]),
        (tail_bbox[0], tail_bbox[3]),
        (tail_bbox[2], tail_bbox[1]),
        (tail_bbox[2], tail_bbox[3]),
    ]
    return max(corners, key=lambda c: _dist(c, bubble_c))


def _best_tail_for_bubble(
    bubble_bbox: list[float],
    tails: Iterable[dict[str, Any]],
    overlap_min: float,
) -> Optional[list[float]]:
    best_ratio = 0.0
    best_bbox: Optional[list[float]] = None
    for tail in tails:
        tbox = _panel_bbox(tail)
        if tbox is None:
            continue
        tarea = _bbox_area(tbox)
        if tarea <= 0:
            continue
        ratio = _overlap_area(tbox, bubble_bbox) / tarea
        if ratio > best_ratio:
            best_ratio = ratio
            best_bbox = tbox
    if best_bbox is None or best_ratio < overlap_min:
        return None
    return best_bbox


def _nearest_speaker(
    point: tuple[float, float],
    character_bboxes: list[tuple[int, list[float]]],
    *,
    max_dist: float,
) -> Optional[int]:
    if not character_bboxes:
        return None
    ranked = sorted(
        character_bboxes,
        key=lambda pair: _dist_point_to_bbox(point, pair[1]),
    )
    ordinal, bbox = ranked[0]
    if _dist_point_to_bbox(point, bbox) > max_dist:
        return None
    return ordinal


def attribute_bubbles(
    *,
    characters: list[dict[str, Any]],
    text_bubbles: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    panel_w_norm: float = 1.0,
    panel_h_norm: float = 1.0,
    tail_overlap_min: float = DEFAULT_TAIL_OVERLAP_MIN,
    max_tip_distance_frac: float = DEFAULT_MAX_TIP_DISTANCE_FRAC,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    panel_diag = _panel_diag(panel_w_norm, panel_h_norm)
    max_dist = panel_diag * max_tip_distance_frac

    char_bboxes: list[tuple[int, list[float]]] = []
    for ordinal, ch in enumerate(characters, start=1):
        bbox = _panel_bbox(ch)
        if bbox is not None:
            char_bboxes.append((ordinal, bbox))

    for bubble in text_bubbles:
        raw_type = str(bubble.get("type") or "Speech Bubble")
        token = BUBBLE_TYPE_TOKENS.get(raw_type, "speech bubble")
        if token == NARRATION_TOKEN:
            out.append({"type": NARRATION_TOKEN, "speaker_ordinal": None, "method": "narration"})
            continue
        if not char_bboxes:
            out.append({"type": NARRATION_TOKEN, "speaker_ordinal": None, "method": "off_panel"})
            continue
        bubble_bbox = _panel_bbox(bubble)
        if bubble_bbox is None:
            out.append({"type": token, "speaker_ordinal": None, "method": "none"})
            continue
        tail_bbox = _best_tail_for_bubble(bubble_bbox, tails, tail_overlap_min)
        if tail_bbox is not None:
            tip = _tail_tip(tail_bbox, bubble_bbox)
            speaker = _nearest_speaker(tip, char_bboxes, max_dist=max_dist)
            if speaker is not None:
                out.append({"type": token, "speaker_ordinal": speaker, "method": "tail"})
                continue
        bubble_c = _centroid(bubble_bbox)
        speaker = _nearest_speaker(bubble_c, char_bboxes, max_dist=max_dist)
        if speaker is not None:
            out.append({"type": token, "speaker_ordinal": speaker, "method": "centroid"})
            continue
        out.append({"type": token, "speaker_ordinal": None, "method": "none"})

    return out


def label_for_bubble(record: dict[str, Any]) -> str:
    token = record["type"]
    speaker = record.get("speaker_ordinal")
    if token == NARRATION_TOKEN or speaker is None:
        return token
    return f"{token} from character {speaker}"
