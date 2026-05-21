"""Claude Haiku 4.5 (Bedrock) text-only prompts for per-page manga annotation.

NO image is sent to Haiku. Per panel we pass:
  - Gemini's long caption (the v1 page-panel annotation)
  - normalized bbox + width/height (so Haiku knows shape + layout position)
  - MAGI counts (characters, speech/shout/narration bubbles)

Haiku outputs (forced via Bedrock tool-calling — see schema.py):
  {
    "page_caption": "1 sentence 15-25 words",
    "panels": [
      {"index": 0, "caption": "10-30 words", "shot_size": "MS", "panel_type": "beat"}
    ]
  }

The Lambda joins Haiku's output with MAGI counts (passed through) and bucket
(computed from bbox).
"""

from __future__ import annotations

from typing import Iterable


SHOT_SIZES = ("ECU", "CU", "MCU", "MS", "MLS", "LS", "ELS", "AMB")

PANEL_TYPES = (
    "establishing",
    "beat",
    "reaction",
    "action",
    "climax",
    "cinematic",
    "transition",
    "atmospheric",
    "splash",
    "amb",
)


SYSTEM_PROMPT = """You are a manga page annotator. You are given the contents of a single manga page as STRUCTURED TEXT — no image. Each panel arrives with: its index (in reading order), its normalized bbox on the page, its pixel width/height, MAGI bubble/character counts, and a long Gemini description.

Your job is to call the annotate_page tool with:
- page_caption: one sentence, 15-25 words, summarizing the whole page in present tense.
- panels[]: one object per input panel, in ascending index order. For each panel:
  - caption: 10-30 words, present tense, distilled from the Gemini description. Visible content only — no plot speculation, no dialogue transcription, no proper names unless they appear in the Gemini text.
  - shot_size: closed enum below. Use the bbox area + the Gemini description as your evidence.
  - panel_type: closed enum below. Determined by the panel's narrative role within the page (and panel size — splash panels span the page, establishing panels show setting, reaction panels are close on characters, etc.).

CLOSED VOCAB:
shot_size: ECU (eyes/detail) | CU (head) | MCU (head+shoulders) | MS (waist up) | MLS (knees up) | LS (full body) | ELS (tiny figure in environment) | AMB
panel_type: establishing | beat | reaction | action | climax | cinematic | transition | atmospheric | splash | amb

RULES:
- Return EXACTLY one panel object per input panel, in ascending index order. Never add, drop, merge, or split.
- Caption length is HARD: 10-30 words.
- All enum values MUST come from the lists above. Use AMB / amb if genuinely unsure.
- No prose, no markdown — just the tool call."""


def build_user_prompt(
    *,
    page_w: int,
    page_h: int,
    panels: Iterable[dict],
) -> str:
    """Per-page user message. `panels` is a list of dicts produced by
    `handlers._read_caption_panels`, each carrying:
      index, bbox [x1,y1,x2,y2], width_px, height_px, area_ratio,
      character_count, speech_bubble_count, shout_bubble_count,
      narration_bubble_count, gemini_caption.
    """
    lines: list[str] = [
        f"PAGE: {page_w} x {page_h} px, {sum(1 for _ in panels)} panels (regenerated below)."
    ]
    blocks: list[str] = []
    n = 0
    for p in panels:
        n += 1
        x1, y1, x2, y2 = p["bbox"]
        nx1 = round(x1 / max(1, page_w), 3)
        ny1 = round(y1 / max(1, page_h), 3)
        nx2 = round(x2 / max(1, page_w), 3)
        ny2 = round(y2 / max(1, page_h), 3)
        gem = (p.get("gemini_caption") or "").strip().replace("\n", " ")
        gem = gem[:600]  # cap to keep input token budget tight
        blocks.append(
            "  PANEL {idx}\n"
            "    page_norm_bbox: [{nx1}, {ny1}, {nx2}, {ny2}]\n"
            "    size_px: {w} x {h} (area_ratio {ar:.3f})\n"
            "    chars: {c}, speech: {s}, shout: {sh}, narration: {nb}\n"
            "    gemini_description: {gem}".format(
                idx=p["index"],
                nx1=nx1, ny1=ny1, nx2=nx2, ny2=ny2,
                w=p["width_px"], h=p["height_px"],
                ar=p["area_ratio"],
                c=p["character_count"],
                s=p["speech_bubble_count"],
                sh=p["shout_bubble_count"],
                nb=p["narration_bubble_count"],
                gem=gem if gem else "(none)",
            )
        )
    # Re-emit the page line with the correct N now that we've iterated.
    lines = [
        f"PAGE: {page_w} x {page_h} px, {n} panels listed below in reading order.",
        "",
    ]
    lines.extend(blocks)
    lines.append("")
    lines.append(
        f"Call the annotate_page tool with exactly {n} panel objects in "
        f"ascending index order."
    )
    return "\n".join(lines)
