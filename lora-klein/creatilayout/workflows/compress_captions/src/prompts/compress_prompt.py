"""Vision-mode prompt for compress_captions.

One Gemini call per panel. The model sees the cropped panel image (not the
full page) and a high thinking budget. It returns three fields used by the
assembler in `src/assembly.py` to build the final short_caption.
"""

from __future__ import annotations

from typing import Any


SHOT_SIZE_VALUES = (
    "extreme close-up",
    "close-up",
    "medium close-up",
    "medium shot",
    "medium long shot",
    "long shot",
    "extreme long shot",
    "ambiguous",
)

CAMERA_ANGLE_VALUES = (
    "eye-level",
    "low-angle",
    "high-angle",
    "overhead",
    "dutch tilt",
    "over-the-shoulder",
    "POV",
    "ambiguous",
)


PANEL_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shot_size": {
            "type": "string",
            "format": "enum",
            "enum": list(SHOT_SIZE_VALUES),
        },
        "camera_angle": {
            "type": "string",
            "format": "enum",
            "enum": list(CAMERA_ANGLE_VALUES),
        },
        "action_phrase": {
            "type": "string",
            "description": (
                "3-6 words. Start with a verb in -ing form or a scene "
                "preposition (in/on/at). No counts, no shot/angle words, "
                "no proper names, no mood/clothing/lighting, no manga style."
            ),
        },
    },
    "required": ["shot_size", "camera_angle", "action_phrase"],
}


SYSTEM_INSTRUCTION = (
    "You look at a single manga panel image and extract three fields for a "
    "layout-supervision model. The image is the cropped panel itself (not the "
    "full page).\n"
    "\n"
    "Fields:\n"
    "1. shot_size: one value from the closed vocabulary.\n"
    "2. camera_angle: one value from the closed vocabulary.\n"
    "3. action_phrase: a 3-6 word phrase describing what is happening or what "
    "the scene shows. Start with a verb in -ing form or a scene preposition "
    "(in/on/at). NO shot size, NO camera angle, NO character count, NO bubble "
    "count, NO proper names, NO dialogue text, NO mood/emotion adjectives, "
    "NO clothing, NO lighting, NO mention of manga/black-and-white.\n"
    "\n"
    "Good action_phrase examples:\n"
    "  'fighting in a street'\n"
    "  'looking up at the sky'\n"
    "  'walking through a forest'\n"
    "  'on a rooftop'\n"
    "  'reaching for a button'\n"
    "  'sitting at a kitchen table'\n"
    "  'empty schoolyard at dusk'\n"
    "  'standing in a school hallway'\n"
    "\n"
    "Bad examples (do NOT do):\n"
    "  'two boys fighting in a street' (has count)\n"
    "  'wide shot of fighting' (has shot size)\n"
    "  'angry fighting' (mood)\n"
    "  'speaking into a speech bubble' (do not narrate bubbles)\n"
    "\n"
    "If the panel is abstract / a transition strip / pure SFX with no scene, "
    "use 'ambiguous' for shot_size and camera_angle, and a brief noun-phrase "
    "for action_phrase (e.g., 'checkered transition pattern').\n"
)


USER_INSTRUCTION = (
    "Extract shot_size, camera_angle, and action_phrase for this panel."
)
