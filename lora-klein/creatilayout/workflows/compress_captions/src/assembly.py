"""Pure assembly logic — converts LLM extract + source-panel structure
into the final short_caption string. No AWS, no IO. Importable from the
Lambda handler AND from the local test runner."""

from __future__ import annotations

from typing import Any


COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

# Map source caption JSON bubble types -> compact prompt tokens.
BUBBLE_TYPE_TOKENS = {
    "Speech Bubble": "speech",
    "Shout Bubble": "shout",
    "Narration Bubble": "narration",
}

# Bubble types that get attributed to a speaker. Narration is speakerless.
ATTRIBUTABLE_BUBBLE_TYPES = ("speech", "shout", "thought")

# Order of bubble types inside a per-character group (frequency-descending).
BUBBLE_TYPE_ORDER = ("speech", "shout", "thought")


def word_for_count(n: int) -> str:
    """Return an English word for small counts; fall back to digit string."""
    if 0 <= n < len(COUNT_WORDS):
        return COUNT_WORDS[n]
    return str(n)


def pluralize_character(n: int) -> str:
    return "character" if n == 1 else "characters"


def _bbox_centroid(bbox: list[float] | tuple[float, ...] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _euclid_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def attribute_bubbles_to_characters(
    *,
    characters: list[dict[str, Any]],
    text_bubbles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute speaker ordinal (1-indexed) for each attributable bubble using
    nearest-character-by-centroid on `panel_bbox_norm` (panel-local coords).

    Returns a list parallel to `text_bubbles` of `{type_token, speaker_ordinal}`.
    Narration always has `speaker_ordinal=None`. If there are no characters at
    all, speech/shout/thought bubbles are relabeled as narration (off-panel
    dialogue convention).
    """
    char_centroids = []
    for index, character in enumerate(characters):
        centroid = _bbox_centroid(character.get("panel_bbox_norm") or character.get("bbox_norm"))
        if centroid is not None:
            char_centroids.append((index + 1, centroid))

    out: list[dict[str, Any]] = []
    for bubble in text_bubbles:
        raw_type = str(bubble.get("type") or "Speech Bubble")
        type_token = BUBBLE_TYPE_TOKENS.get(raw_type, "speech")
        if type_token == "narration":
            out.append({"type": "narration", "speaker_ordinal": None})
            continue
        if not char_centroids:
            out.append({"type": "narration", "speaker_ordinal": None})
            continue
        bubble_centroid = _bbox_centroid(bubble.get("panel_bbox_norm") or bubble.get("bbox_norm"))
        if bubble_centroid is None:
            out.append({"type": "narration", "speaker_ordinal": None})
            continue
        ordinal, _ = min(char_centroids, key=lambda pair: _euclid_sq(pair[1], bubble_centroid))
        out.append({"type": type_token, "speaker_ordinal": ordinal})
    return out


def assemble_short_caption(
    *,
    shot_size: str,
    camera_angle: str,
    action_phrase: str,
    character_count: int,
    attributed_bubbles: list[dict[str, Any]],
) -> str:
    """Build the final short_caption per the locked-in panel template:

        <shot_size>, <camera_angle>[, <N> character[s] <action>]
        [, <per-character bubble groups>][, <N> narration]
    """
    parts: list[str] = [str(shot_size).strip(), str(camera_angle).strip()]

    action_phrase = str(action_phrase or "").strip()
    if character_count > 0:
        char_clause = f"{word_for_count(character_count)} {pluralize_character(character_count)}"
        if action_phrase:
            char_clause = f"{char_clause} {action_phrase}"
        parts.append(char_clause)
    else:
        if action_phrase:
            parts.append(action_phrase)

    # Per-character bubble groups.
    per_char_counts: dict[int, dict[str, int]] = {}
    narration_count = 0
    for entry in attributed_bubbles:
        type_token = entry.get("type")
        ordinal = entry.get("speaker_ordinal")
        if type_token == "narration" or ordinal is None:
            narration_count += 1
            continue
        per_char_counts.setdefault(ordinal, {}).setdefault(type_token, 0)
        per_char_counts[ordinal][type_token] += 1

    for ordinal in sorted(per_char_counts):
        type_counts = per_char_counts[ordinal]
        type_parts = [
            f"{type_counts[t]} {t}"
            for t in BUBBLE_TYPE_ORDER
            if type_counts.get(t)
        ]
        if not type_parts:
            continue
        parts.append(f"{' '.join(type_parts)} from character {ordinal}")

    if narration_count > 0:
        parts.append(f"{narration_count} narration")

    return ", ".join(parts)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def strip_caption_prefix(caption: str, prefix: str) -> str:
    """Remove the 'Black and White Manga. <Title> by <Author>.' lead-in."""
    import re

    body = str(caption or "").strip()
    head = str(prefix or "").strip()
    if head and body.startswith(head):
        return body[len(head):].strip()
    return re.sub(
        r"^(?:Black and White Manga|Colored Manga)\.\s*(?:[^.]*\sby\s[^.]*\.\s*)?",
        "",
        body,
    ).strip()
