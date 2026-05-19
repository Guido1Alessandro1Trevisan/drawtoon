# Gemini Board-Sheet Character Assignment Prompt

Use with Gemini Flash / Gemini 3 Flash, thinking level high.

Input:
- Image 1: clean board sheet.
- Image 2: same board sheet with Magi overlays.
- Text metadata: sheet id, panel boxes, and Magi character boxes with exact coordinates.

Prompt:

```text
You are assigning Magi character boxes to the correct visual characters within one manga/webtoon board sheet.

Important:
- This is a per-sheet character assignment task.
- Magi labels such as C0, C1, source_character_id, and character_cluster_labels are detector labels only.
- Do not treat Magi labels as final identities.
- Use the clean board sheet for visual identity.
- Use the Magi overlay only to identify the numbered boxes.
- Use the coordinate list as the authoritative list of boxes to assign.

Task:
Map every provided Magi character box to a local character cluster.

Definitions:
- A local character cluster is one visually consistent person within this board sheet.
- The same person may appear in multiple panels with different pose, crop, scale, expression, lighting, or partial visibility.
- Different people may look similar, so do not merge based on one feature alone.

Evidence to use:
- hairstyle shape, hair color/tone, bangs, hair length
- face shape, eyes, eyebrows, mouth, facial hair
- glasses, scars, accessories, distinctive marks
- clothing, collar, tie, uniform, color, silhouette
- body type, height, posture, recurring pose
- panel continuity and reading order
- speech-bubble tails only when the tail clearly points to a character

Rules:
1. Every provided Magi character box must appear exactly once in box_assignments.
2. Do not invent boxes that are not provided.
3. Do not merge two people only because they are close together, in the same scene, near the same text, or have the same Magi source_character_id.
4. Do not split one person only because their pose, crop, expression, scale, or lighting changes.
5. Partial bodies, backs of heads, hands, silhouettes, tiny figures, and occluded figures should be assigned only when visual/context evidence is strong.
6. If identity evidence is weak, still choose the best local_character_id, but lower confidence and add an ambiguity entry.
7. Use local ids only: local_character_1, local_character_2, etc.
8. Return JSON only.

Return this JSON shape:
{
  "sheet_id": "string",
  "local_characters": [
    {
      "local_character_id": "local_character_1",
      "description": "short visual description",
      "distinguishing_features": ["feature"],
      "representative_magi_box_ids": ["sheet_id::char_0"]
    }
  ],
  "box_assignments": [
    {
      "magi_box_id": "sheet_id::char_0",
      "local_character_id": "local_character_1",
      "confidence": 0.0,
      "visible_extent": "full_body|upper_body|face_or_head|partial_body|tiny_or_background|silhouette_or_occluded|unclear",
      "evidence": ["specific visual evidence"]
    }
  ],
  "ambiguous_assignments": [
    {
      "magi_box_id": "sheet_id::char_0",
      "candidate_local_character_ids": ["local_character_1", "local_character_2"],
      "chosen_local_character_id": "local_character_1",
      "reason": "why this is ambiguous"
    }
  ],
  "quality_checks": {
    "all_boxes_assigned_once": true,
    "no_magi_id_global_assumption": true,
    "uncertain_cases_marked": true,
    "notes": "short audit note"
  }
}
```

Per-request metadata template:

```json
{
  "sheet_id": "sheet_000_segments_000-000",
  "panels": [
    {
      "panel_index": 0,
      "panel_id": "string",
      "bbox": [0, 0, 0, 0]
    }
  ],
  "magi_character_boxes_to_assign_exactly_once": [
    {
      "magi_box_id": "sheet_000_segments_000-000::char_0",
      "character_index": 0,
      "source_character_id": "0",
      "bbox": [0, 0, 0, 0]
    }
  ]
}
```
