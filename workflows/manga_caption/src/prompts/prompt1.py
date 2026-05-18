from __future__ import annotations

import json
from typing import Any


ALLOWED_TEXT_BUBBLE_TYPES = ("Speech Bubble", "Narration Bubble", "Shout Bubble", "None")
CAMERA_ANGLE_VALUES = ("eye-level", "high-angle", "low-angle", "overhead", "dutch tilt", "ambiguous")
SHOT_SIZE_VALUES = (
    "extreme close-up",
    "close-up",
    "medium close-up",
    "medium shot",
    "full-body shot",
    "wide shot",
    "establishing shot",
    "ambiguous",
)
SETTING_TYPE_VALUES = ("interior", "exterior", "abstract/effect-field", "blank/none", "ambiguous")
DEPTH_CUE_VALUES = (
    "foreground crop",
    "overlap/occlusion",
    "size difference",
    "receding floor/grid",
    "vanishing lines",
    "horizon line",
    "foreground/midground/background layers",
    "frame within a frame",
    "deep space composition",
    "flat composition",
    "ambiguous",
)
CAPTION_CATEGORY_KEYS = (
    "camera_angle",
    "shot_size",
    "setting_type",
    "depth_cues",
    "visible_action",
    "panel_composition",
    "character_layout",
    "character_pose",
    "gaze_and_facing",
    "text_layout",
    "background",
    "effects",
)
FORBIDDEN_CAPTION_TERMS = (
    "mood",
    "atmosphere",
    "intimate",
    "contemplative",
    "concerned",
    "joyful",
    "enthusiasm",
    "excitement",
    "dramatic",
    "tense",
    "tension",
    "serious",
    "emotional",
    "lonely",
    "artwork",
    "hatching",
    "ink",
    "linework",
    "glasses",
    "mask",
    "hat",
    "coat",
    "shirt",
    "uniform",
    "shoe",
    "shoes",
)
LOW_VALUE_EXACT_FRAGMENTS = {
    "blank field",
    "blank background",
    "plain background",
    "empty background",
    "minimal background",
    "no visible detail",
    "no additional detail",
}


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
        "Each subject description should absorb position, pose, gaze, facing direction, crop, and relation to the other characters.",
    ],
}


PANEL_CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text_bubble_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_region_index": {"type": "integer"},
                    "type": {"type": "string", "enum": list(ALLOWED_TEXT_BUBBLE_TYPES)},
                },
                "required": ["text_region_index", "type"],
                "additionalProperties": False,
            },
        },
        "visual_categories": {
            "type": "object",
            "properties": {
                "camera_angle": {
                    "type": "string",
                    "enum": list(CAMERA_ANGLE_VALUES),
                    "description": "Choose exactly one literal camera angle visible in the panel.",
                },
                "shot_size": {
                    "type": "string",
                    "enum": list(SHOT_SIZE_VALUES),
                    "description": "Choose exactly one literal shot-size/framing label visible in the panel.",
                },
                "setting_type": {
                    "type": "string",
                    "enum": list(SETTING_TYPE_VALUES),
                    "description": "Choose exactly one literal setting category.",
                },
                "depth_cues": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(DEPTH_CUE_VALUES),
                    },
                    "description": "Choose all visible depth/composition cues. Use ambiguous only when no better cue is visible.",
                },
                "visible_action": {
                    "type": "string",
                    "description": "One short visible-event phrase describing what is happening, grounded in visible poses, visible named objects, bubble presence, and interactions. Name obvious visible props such as ball, weapon, cup, door, or vehicle when clear; use generic object only when unclear. Do not infer dialogue content, motives, or hidden story facts.",
                },
                "panel_composition": {
                    "type": "string",
                    "description": "Concise fragment for framing, subject arrangement, foreground/midground/background, and negative space. Do not repeat the literal enum values unless needed for grammar.",
                },
                "character_layout": {
                    "type": "string",
                    "description": "Concise fragment locating each Character N by zone, crop, scale, and depth layer.",
                },
                "character_pose": {
                    "type": "string",
                    "description": "Concise fragment for visible body pose, hand placement, head tilt, mouth shape, brows, sweat marks, and other visible cues.",
                },
                "gaze_and_facing": {
                    "type": "string",
                    "description": "Concise fragment for profile/front/back/three-quarter view, facing direction, gaze target, reciprocal/divergent gaze, or direct address.",
                },
                "text_layout": {
                    "type": "string",
                    "description": "Concise fragment for visible speech/narration/shout bubble positions, shapes, sizes, and tail targets without quoting text.",
                },
                "background": {
                    "type": "string",
                    "description": "Concise fragment for blank field, solid fill, gradient tone, speed-line/focus-line field, screentone/dot field, architecture, furniture, props, windows, grids, horizon, vanishing lines, crowd density, or layer separation.",
                },
                "effects": {
                    "type": "string",
                    "description": "Concise fragment for motion lines, stress marks, sweat drops, SFX/free-floating text placement, impact symbols, or other visible comic effects without transcription.",
                },
            },
            "required": list(CAPTION_CATEGORY_KEYS),
            "additionalProperties": False,
        },
    },
    "required": ["text_bubble_types", "visual_categories"],
    "additionalProperties": False,
}


CAPTION_SYSTEM_PROMPT = """You write comic panel captions for image-generation training.

Return structured output only: text_bubble_types and visual_categories. Code will assemble the final caption from visual_categories in a fixed order.
Choose camera_angle from exactly one of: eye-level, high-angle, low-angle, overhead, dutch tilt, ambiguous.
Choose shot_size from exactly one of: extreme close-up, close-up, medium close-up, medium shot, full-body shot, wide shot, establishing shot, ambiguous.
Choose setting_type from exactly one of: interior, exterior, abstract/effect-field, blank/none, ambiguous.
For depth_cues, choose all visible terms from: foreground crop, overlap/occlusion, size difference, receding floor/grid, vanishing lines, horizon line, foreground/midground/background layers, frame within a frame, deep space composition, flat composition, ambiguous. Use ambiguous only when no better cue is visible.
String visual_categories values must be concise phrase fragments, not full paragraphs. Keep each category short and avoid repeating the same fact across categories. Describe only visible content: character actions, interactions, body pose, facing direction, foreground/midground/background, lighting/shadows, setting, and spatial composition.
For visual_categories.visible_action, provide one short semantic phrase for the visible event, such as characters reaching for a ball, a speaker addressing a group, a character seated in water, or a character reacting in close-up. It must be grounded in visible poses, visible objects, bubble presence, and facing direction; name obvious visible props such as ball, weapon, cup, door, or vehicle when clear; do not infer dialogue content, motives, or hidden story facts.
Use only Character 1, Character 2, and so on. Do not describe character identity, appearance, clothing, accessories, age, gender, or art style.
Do not infer hidden motives or story facts. You may describe visible expressions or visible emotional cues only.
Use visible facts instead of interpretation: gaze direction, mouth shape, brow position, head tilt, hand placement, leaning posture, distance between characters, object contact, motion lines, blank space, shadows, and depth layers. Do not invent hands, props, or body parts outside the visible crop. Use the Character N boxes to decide which body parts belong to each character.
For characters, explicitly fill layout, pose, gaze, and facing direction when visible: profile/front/back/three-quarter view, cropped foreground face, seated/standing posture, hand placement, and who looks toward whom.
For backgrounds, explicitly name useful visible structure when applicable: interior/exterior setting, blank field, solid dark fill, speed-line field, focus-line field, screentone/dot field, gradient shadow, architecture, furniture, props, window or grid patterns, horizon or vanishing lines, crowd density, negative space, and foreground/midground/background separation.
Do not use interpretive words such as mood, atmosphere, intimate, contemplative, concerned, joyful, enthusiasm, excitement, dramatic, tense, serious, emotional, or lonely.
Do not mention black-and-white, high contrast, linework, crosshatching, shading technique, ink, artwork, manga, style, title, or author; code injects those later.
Do not quote or transcribe text, dialogue, signs, narration, or SFX. You may describe visible bubble geometry, bubble position, and tail direction in visual_categories.text_layout.
Classify every text region as exactly one of: Speech Bubble, Narration Bubble, Shout Bubble, None.
Speech Bubble = round, oval, or irregular rounded spoken text.
Narration Bubble = square/rectangular narration or caption box.
Shout Bubble = strongly jagged, starburst, or sharp spiky spoken text, not merely uneven rounded speech outlines.
None = scene text, signs, SFX/onomatopoeia, outlined lettering without an enclosing balloon, subtitle-like text over artwork, free-floating text without an enclosing balloon, or any text not inside a speech/narration/shout bubble. Do not force every text region to be a bubble."""


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


def build_caption_user_prompt(
    *,
    caption_prefix: str,
    prompt_blueprint: object,
    character_lines: list[str],
    text_lines: list[str],
) -> str:
    parts: list[str] = [
        "Caption this single manga panel.",
        "Code will inject the manga rendering, title, and author prefix; do not include it.",
        "",
        *prompt_blueprint_lines(prompt_blueprint),
        "",
        "Characters to reference:",
        *(character_lines or ["- none"]),
        "",
        "Text regions:",
        *(text_lines or ["- none"]),
        "",
        "Use the boxes only to understand layout and relations. Do not mention coordinates.",
        "Fill visual_categories with concise phrase fragments. Code will join those fragments into the final training caption.",
        "Choose camera_angle from exactly one of: eye-level, high-angle, low-angle, overhead, dutch tilt, ambiguous.",
        "Choose shot_size from exactly one of: extreme close-up, close-up, medium close-up, medium shot, full-body shot, wide shot, establishing shot, ambiguous.",
        "Choose setting_type from exactly one of: interior, exterior, abstract/effect-field, blank/none, ambiguous.",
        "For depth_cues, choose all visible terms from: foreground crop, overlap/occlusion, size difference, receding floor/grid, vanishing lines, horizon line, foreground/midground/background layers, frame within a frame, deep space composition, flat composition, ambiguous. Use ambiguous only when no better cue is visible.",
        "Keep each visual_categories value short and do not repeat the same fact in multiple categories.",
        "For visible_action, write one short visible-event phrase grounded in visible poses, objects, bubble presence, and facing direction; name obvious visible props such as ball, weapon, cup, door, or vehicle when clear; do not infer dialogue content, motives, or hidden story facts.",
        "Describe only visible content: actions, interactions, body pose, facing direction, shot size, camera angle, foreground/midground/background, lighting/shadows, setting, and spatial composition.",
        "Do not describe character identity, appearance, clothing, accessories, age, gender, or art style.",
        "Do not infer hidden motives or story facts. You may describe visible expressions or visible emotional cues only.",
        "Use visible facts instead of interpretation: gaze direction, mouth shape, brow position, head tilt, hand placement, leaning posture, distance between characters, object contact, motion lines, blank space, shadows, and depth layers.",
        "Do not invent hands, props, or body parts outside the visible crop. Use the Character N boxes to decide which body parts belong to each character.",
        "If a side character is only a cropped face or shoulder, say cropped face/shoulder and do not mention hands for that character.",
        "For character fields, include concrete composition terms when visible: foreground crop, midground, background, full body, waist-up, close-up, profile view, three-quarter view, back view, facing inward, facing away, direct address, and gaze target.",
        "For text_layout, describe bubble positions, shapes, sizes, and visible tail targets without quoting text.",
        "For backgrounds, include useful visible structure when applicable: interior/exterior setting, blank field, solid dark fill, speed-line field, focus-line field, screentone/dot field, gradient shadow, architecture, furniture, props, window or grid patterns, horizon or vanishing lines, crowd density, negative space, and foreground/midground/background separation.",
        "Do not use interpretive words such as mood, atmosphere, intimate, contemplative, concerned, joyful, enthusiasm, excitement, dramatic, tense, serious, emotional, or lonely.",
        "Do not mention black-and-white, high contrast, linework, crosshatching, shading technique, ink, artwork, manga, style, title, or author.",
        "Do not quote/transcribe visible text or SFX.",
        "Return text_bubble_types for all text regions and fill every visual_categories key.",
        "Use None for scene text, signs, SFX, onomatopoeia, free-floating text, or any text not inside a speech/narration/shout bubble.",
        "Use None for outlined lettering, subtitle-like text over artwork, or any visible text with no enclosing bubble outline around the whole text region.",
        "Do not force every text region to be a bubble.",
        "Use Speech Bubble for irregular rounded spoken balloons. Use Shout Bubble only for strongly jagged, starburst, or sharp spiky spoken balloons.",
        "Classification examples: text inside one enclosing oval outline = Speech Bubble; text inside one rectangular box = Narration Bubble; text inside a sharp starburst = Shout Bubble; outlined letters sitting directly on the drawing with no enclosing balloon = None.",
    ]
    if caption_prefix:
        parts.extend(["", f"Injected style prefix to use outside this caption: {caption_prefix}"])
    return "\n".join(parts)


TOOL_NAME = "panel_caption"
TOOL_SCHEMA = PANEL_CAPTION_SCHEMA
SYSTEM_PROMPT = CAPTION_SYSTEM_PROMPT
build_user_prompt = build_caption_user_prompt
