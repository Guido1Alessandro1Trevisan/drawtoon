from __future__ import annotations

from typing import Any


ALLOWED_PAGE_TAGS = (
    "decompressed",
    "compressed",
    "fast_sequence",
    "beat_pause",
    "montage",
    "establishing",
    "initiating",
    "climactic",
    "release",
    "transition",
    "action",
    "atmosphere",
    "flashback",
    "monologue",
    "conversation",
    "confrontation",
    "exposition",
    "group_chat",
    "narration_voiceover",
    "silent",
    "sfx_heavy",
)


PAGE_CAPTION_SCHEMA_GEMINI: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": (
                "10-20 word caption describing the page as a whole, for "
                "page-layout supervision. Start with the panel count + 'panel "
                "page' (e.g., '5-panel page'). Then one descriptor for the "
                "page's primary content (action, dialogue, establishing shot, "
                "etc.). Optionally mention dominant panel size if striking "
                "(splash, hero, vertical strip). No proper names, no dialogue "
                "text, no mood adjectives, no manga-style mention."
            ),
        },
        "page_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "format": "enum",
                "enum": list(ALLOWED_PAGE_TAGS),
            },
            "description": (
                "Pick 1-4 page tags from the controlled vocabulary. Cover at "
                "most: one rhythm/pacing, one page-function, one mood, one "
                "dialogue-type. Never invent new tags."
            ),
        },
    },
    "required": ["caption", "page_tags"],
}


SYSTEM_INSTRUCTION = (
    "You write ultra-short layout captions for manga pages.\n"
    "\n"
    "A page-layout caption describes WHAT the page contains and WHAT KIND OF "
    "PAGE it is — not what it means. The caption is used as the global prompt "
    "for a page-layout model that predicts where each panel sits on the page; "
    "it is NOT a story summary.\n"
    "\n"
    "Rules:\n"
    "- 10 to 20 words. Hard cap 20.\n"
    "- Start with the panel count + 'panel page' (e.g., '5-panel page', "
    "'1-panel splash page').\n"
    "- Then add one descriptor for the page's primary content: action scene, "
    "dialogue scene, establishing shot, reaction beats, montage, transition, "
    "etc.\n"
    "- Optionally mention a striking dominant panel size: splash, hero, "
    "panoramic strip, vertical column. Skip if the page is a uniform grid.\n"
    "- Mention 'speech bubbles' or 'no dialogue' only if it is visually "
    "obvious and helps describe the page.\n"
    "- No proper names. No dialogue text. No narration text.\n"
    "- No mood, emotion, lighting, color, or style adjectives.\n"
    "- Do not mention black-and-white, ink, manga style, title, or author.\n"
    "\n"
    "Also return 1-4 page_tags from the controlled vocabulary "
    "(rhythm + function + mood + dialogue). Pick at most one tag per axis. "
    "If you cannot pick a tag for an axis with confidence, omit it.\n"
)


def build_user_text(*, panel_count: int, page_size: dict[str, int] | None) -> str:
    page_w = int((page_size or {}).get("width_px") or 0)
    page_h = int((page_size or {}).get("height_px") or 0)
    aspect = round(page_w / page_h, 3) if page_w and page_h else None
    lines = [
        f"This is one manga page. Pre-detected panel count: {panel_count}.",
    ]
    if aspect is not None:
        lines.append(f"Page aspect (w/h): {aspect}.")
    lines.append("")
    lines.append("Write the page-layout caption and pick page_tags.")
    return "\n".join(lines)
