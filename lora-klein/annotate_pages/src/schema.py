"""JSON Schema for the Bedrock tool the model is forced to call.

Bedrock's converse() API with toolConfig + toolChoice constrains the model to
emit exactly this structure. No free-form JSON parsing.
"""

from __future__ import annotations

from .prompt import SHOT_SIZES, PANEL_TYPES


TOOL_NAME = "annotate_page"
TOOL_DESCRIPTION = (
    "Emit the structured annotation for a manga page. Call this tool exactly "
    "once with one panel object per numbered red rectangle on the page, "
    "in ascending index order."
)

ANNOTATE_PAGE_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "page_caption": {
            "type": "string",
            "description": (
                "One sentence, 15-25 words, summarizing the page's gist. "
                "Present tense, describes visible content."
            ),
        },
        "panels": {
            "type": "array",
            "description": (
                "One object per numbered red rectangle, in ascending index "
                "order. Length must equal the page's panel count."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Zero-based index matching the red rectangle's number.",
                    },
                    "caption": {
                        "type": "string",
                        "description": (
                            "10-30 words, present tense, visible content only. "
                            "No dialogue transcription, no speculation."
                        ),
                    },
                    "shot_size": {
                        "type": "string",
                        "enum": list(SHOT_SIZES),
                        "description": (
                            "Cinema shot-size: ECU=eyes/detail, CU=head, "
                            "MCU=head+shoulders, MS=waist up, MLS=knees up, "
                            "LS=full body, ELS=tiny figure, AMB=fallback."
                        ),
                    },
                    "panel_type": {
                        "type": "string",
                        "enum": list(PANEL_TYPES),
                        "description": (
                            "Panel's narrative role within the page. Use 'amb' "
                            "if genuinely uncertain."
                        ),
                    },
                },
                "required": ["index", "caption", "shot_size", "panel_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_caption", "panels"],
    "additionalProperties": False,
}
