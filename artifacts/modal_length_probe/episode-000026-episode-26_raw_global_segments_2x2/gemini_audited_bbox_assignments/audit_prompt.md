# Gemini Audited BBox Assignment Prompt

```text
You are auditing Magi character boxes on one manga/webtoon review sheet.

Inputs:
- Image 1 is the clean sheet.
- Image 2 is the same sheet with Magi boxes drawn and numbered.
- Metadata lists every bbox id and coordinates. You must classify every listed bbox exactly once.

Use Magi as the default detector: keep a previous label if it matches the visible person, but correct it when it is visibly wrong.
Use source_character_id only as a weak hint. It is not a true identity.

For each bbox:
- Assign a HUMAN/PERSON character only. Do not assign animals, pets, objects, text, speech bubbles, effects, or background.
- Prefer an exact known chapter label when it fits visually. If none fits, use a short stable visual description.
- Return "NoCharacter" if the box is only background, text, an animal/pet/object, a speech bubble, a body fragment with no usable identity, or a duplicate/overlapping smaller box for a person already covered better.
- If two boxes overlap heavily on the same person, keep the larger/clearer box and set the duplicate to "NoCharacter".
- If one box mixes multiple people/entities and no single human is clearly the primary isolated subject, set it to "NoCharacter".
- If one box has a clear primary human and only minor edge overlap from another person, assign the primary human.
- Do not drop a real face/upper body just because it is a close-up, masked, back-facing, occluded, or partly cropped; assign it when the person is visually identifiable.
- Never output story names such as pet names; use visual descriptions only.

Return JSON only with this shape:
{
  "sheet_id": "...",
  "bbox_assignments": {"bbox1": "short character description or NoCharacter"},
  "bbox_audit": {
    "bbox1": {
      "final_label": "same value as bbox_assignments[bbox1]",
      "decision": "keep_prior|correct_label|drop_not_character|drop_duplicate|drop_multi_character|uncertain_keep",
      "reason": "short visible-evidence explanation"
    }
  }
}

```
