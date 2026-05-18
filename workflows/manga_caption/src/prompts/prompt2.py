from __future__ import annotations

import json
from typing import Any


DEFAULT_PROMPT_BLUEPRINT: dict[str, Any] = {
    "scene": {
        "summary": "",
        "background": "",
        "composition": "",
        "style": "Black and White Manga. [manga name] style by [manga author].",
    },
    "subjects": [
        {"id": "Character 1", "description": ""},
        {"id": "Character 2", "description": ""},
    ],
    "notes": [
        "Include shot size, camera angle, gaze, facing direction, pose, background, composition, and subject relationships when available.",
        "scene.summary must describe what is happening in plain literal terms.",
        "scene.background must describe the visible setting and spatial structure.",
        "scene.composition must reference named characters directly and describe framing or relationship.",
        "style is injected and stays separate from the rest of the prompt.",
        "Return exactly one subject for each listed Character N and no other subjects.",
        "Each subject description should absorb position, pose, gaze, facing direction, crop, and relation to the other characters.",
    ],
}


STRUCTURED_CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Plain literal action or event happening in the panel.",
                },
                "background": {
                    "type": "string",
                    "description": "Visible setting and spatial structure only.",
                },
                "composition": {
                    "type": "string",
                    "description": "Framing, shot, and character relationship description using named characters directly.",
                },
                "style": {
                    "type": "string",
                    "description": "Injected style string. Return it unchanged.",
                },
            },
            "required": ["summary", "background", "composition", "style"],
            "additionalProperties": False,
        },
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Visible facts about the character, including crop, position, pose, gaze, facing direction, and relation to other characters.",
                    },
                },
                "required": ["id", "description"],
                "additionalProperties": False,
            },
            "minItems": 0,
        },
    },
    "required": ["scene", "subjects"],
    "additionalProperties": False,
}


STRUCTURED_CAPTION_SYSTEM_PROMPT = """You write structured JSON prompts for single manga panels.

Return structured output only: scene and subjects.
scene.summary must describe what is happening in plain literal terms.
scene.background must describe the visible setting and spatial structure.
scene.composition must reference named Character N labels directly and describe framing or relationships.
scene.style is injected and should be returned unchanged.
Return exactly one subject for each listed Character N and no other subjects.
Copy subject ids exactly: Character 1, Character 2, etc.
If no characters are listed, return an empty subjects array.
Never invent subject ids such as Character, person, soldier, crowd, Mounted Soldiers, or group labels.
Unlisted background figures can be described in scene.summary, scene.background, or scene.composition, but not in subjects.
Each subject description should absorb position, pose, gaze, facing direction, crop, and relation to the other characters.
Use only visible facts. Do not infer motives, dialogue meaning, or hidden story context.
Include shot size, camera angle, gaze, facing direction, pose, background, composition, and subject relationships when visible.
Ignore readable dialogue text when deciding the scene summary.
Do not quote or transcribe text, and do not classify bubbles here."""


def normalize_prompt_blueprint(value: object) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prompt_blueprint must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("prompt_blueprint must be an object or JSON object string")
    return value


def merge_prompt_blueprints(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key not in merged:
            merged[key] = value
            continue
        existing = merged[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_prompt_blueprints(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            merged[key] = [*existing, *value]
        else:
            merged[key] = value
    return merged


def prompt_blueprint_lines(prompt_blueprint: object) -> list[str]:
    blueprint = normalize_prompt_blueprint(prompt_blueprint)
    if not blueprint:
        return []
    lines: list[str] = ["Prompt blueprint:"]
    scene = blueprint.get("scene") if isinstance(blueprint.get("scene"), dict) else {}
    if isinstance(scene, dict):
        summary = str(scene.get("summary") or "").strip()
        background = str(scene.get("background") or "").strip()
        composition = str(scene.get("composition") or "").strip()
        style = str(scene.get("style") or "").strip()
        if summary:
            lines.append(f"- scene.summary: {summary}")
        if background:
            lines.append(f"- scene.background: {background}")
        if composition:
            lines.append(f"- scene.composition: {composition}")
        if style:
            lines.append(f"- scene.style: {style}")
    lines.append("- required visual coverage: shot size, camera angle, gaze, facing direction, pose, background, composition, and subject relationships.")
    subjects = blueprint.get("subjects")
    if isinstance(subjects, list) and subjects:
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            subject_id = str(subject.get("id") or subject.get("name") or "subject").strip()
            description = str(subject.get("description") or "").strip()
            if description:
                lines.append(f"- subject {subject_id}: {description}")
    notes = blueprint.get("notes")
    if isinstance(notes, list) and notes:
        for note in notes:
            text = str(note or "").strip()
            if text:
                lines.append(f"- note: {text}")
    return lines


def build_structured_caption_user_prompt(
    *,
    caption_prefix: str,
    prompt_blueprint_lines: list[str],
    character_lines: list[str],
    text_lines: list[str],
) -> str:
    parts: list[str] = [
        "Create a structured JSON prompt for one manga panel.",
        "Code will inject the manga rendering, title, and author prefix; do not include it.",
        f"Injected style prefix to use as scene.style: {caption_prefix}",
        "",
    ]
    parts.extend(prompt_blueprint_lines or [])
    if prompt_blueprint_lines:
        parts.append("")
    parts.extend(
        [
            "Characters to reference:",
            *(character_lines or ["- none"]),
            "",
            "Use the boxes only to understand layout and relations. Do not mention coordinates.",
            "Return exactly one subject for each listed Character N and no other subjects.",
            "Copy each subject id exactly as listed. If no characters are listed, return an empty subjects array.",
            "Never invent subject ids such as Character, person, soldier, crowd, Mounted Soldiers, or group labels.",
            "Unlisted background figures can be described in scene.summary, scene.background, or scene.composition, but not in subjects.",
            "Ignore readable dialogue text when deciding the scene summary.",
            "Return valid JSON matching this shape:",
            "{",
            '  "scene": {',
            '    "summary": "",',
            '    "background": "",',
            '    "composition": "",',
            '    "style": "Black and White Manga. [manga name] style by [manga author]."',
            "  },",
            '  "subjects": [',
            '    {"id": "Character 1", "description": ""}',
            "  ]",
            "}",
            "",
            "scene.summary must describe what is happening in plain literal terms.",
            "scene.background must describe the visible setting and spatial structure.",
            "scene.composition must reference named Character N labels directly and describe framing or relationship.",
            "style is injected and stays separate from the rest of the prompt.",
            "Each subject description should absorb position, pose, gaze, facing direction, crop, and relation to the other characters.",
            "Include shot size, camera angle, gaze, facing direction, pose, background, composition, and subject relationships when available.",
            "Use only visible facts.",
            "Do not infer motives, dialogue meaning, or hidden story context.",
            "Keep the JSON compact and clean.",
        ]
    )
    return "\n".join(parts)


TOOL_NAME = "panel_structured_caption"
TOOL_SCHEMA = STRUCTURED_CAPTION_SCHEMA
SYSTEM_PROMPT = STRUCTURED_CAPTION_SYSTEM_PROMPT
build_user_prompt = build_structured_caption_user_prompt
