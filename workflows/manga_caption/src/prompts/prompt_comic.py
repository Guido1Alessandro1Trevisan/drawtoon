"""Comic-specific overrides for the Gemini per-page caption prompt.

The manga prompt lives inline in handlers._build_gemini_user_text. This file
holds only the substrings that need to change when the chapter is a Western
comic (publisher slug starting with marvel_/dc_ and slug ending with _comic):

- "manga page" → "comic page"
- The "do not mention that it is a black-and-white manga panel" guard is
  dropped since the comic prefix already states "Colored Comic." and the
  visuals are colored.
- Reading order flips from manga right-to-left to Western left-to-right.
"""

# Header lines for one full comic page. Mirrors the manga header in
# handlers._build_gemini_user_text but with comic-appropriate wording and
# reading order.
def build_comic_header_lines(panel_count: int) -> list[str]:
    return [
        f"This is one full comic page containing {panel_count} panels.",
        "For EACH panel: dense ~80-word caption that is descriptive not interpretive.",
        "For each character cover position, pose, facing direction, gaze target, mouth shape, visible action.",
        "Include shot size (close-up/medium/wide), camera angle (eye-level/high/low/three-quarter), "
        "and panel composition language (speed lines, screen tone, dark fill, tilted framing, overlapping figures, silhouettes).",
        "Do not mention speech/shout/narration bubbles.",
        "Do not mention characters by name.",
        "Do not describe the medium or its rendering (colored, black-and-white, etc.).",
        "Return panels in Western comic reading order (left-to-right, top-to-bottom).",
    ]
