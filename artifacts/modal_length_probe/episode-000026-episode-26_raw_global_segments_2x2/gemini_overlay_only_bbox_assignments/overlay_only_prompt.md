# Gemini Overlay-Only BBox Assignment Prompt

```text
You are auditing Magi character boxes on one manga/webtoon review sheet.

Input:
- The image is the page/sheet with Magi character boxes drawn and numbered.
- Metadata lists every bbox id and coordinates. Classify every listed bbox exactly once.

Rules:
- Assign a HUMAN/PERSON character only.
- Use the colored box labels to identify bbox ids, but judge the underlying artwork inside each box.
- Prefer the previous stable label if it visibly matches; correct it if it is wrong.
- Return "NoCharacter" for animals, pets, objects, text, speech bubbles, effects, background, unusable body fragments, or duplicate smaller boxes.
- If two boxes overlap heavily on the same person, keep the larger/clearer box and mark the duplicate NoCharacter.
- If one box mixes multiple people/entities and no single human is clearly the primary isolated subject, mark it NoCharacter.
- Do not drop a real face, upper body, back view, masked person, occluded person, or close-up when identifiable.
- Use short stable visual descriptions, not story names.
- Give a short visible-evidence reason.

Return JSON only:
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
