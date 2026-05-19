# Gemini Clean Image Plus Coordinates Prompt

```text
You are auditing Magi character boxes on one manga/webtoon review sheet.

Input:
- Image 1 is the clean page/sheet with no boxes drawn.
- Metadata lists every bbox id and exact pixel coordinates in that image.
- Coordinates are [x0, y0, x1, y1] measured from the top-left of the image.
- Classify every listed bbox exactly once.

Task:
For each bbox id, inspect the region described by its coordinates in the image and assign it.

Rules:
- Use only bbox ids present in metadata. Do not invent extra bbox ids.
- Assign a HUMAN/PERSON character only.
- Prefer the previous stable label if it visibly matches; correct it if wrong.
- Return "NoCharacter" for animals, pets, objects, text, speech bubbles, effects, background, unusable body fragments, or duplicate smaller boxes.
- If two boxes overlap heavily on the same person, keep the larger/clearer box and mark the duplicate NoCharacter.
- Do not mark a visible secondary character NoCharacter just because another person is also nearby or partly inside the box.
- Mark silhouettes, tiny background bystanders, or crowd figures as NoCharacter only when they are not visually identifiable enough to track.
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
      "decision": "keep_prior|correct_label|drop_not_character|drop_duplicate|drop_bystander_or_silhouette|uncertain_keep",
      "reason": "short visible-evidence explanation"
    }
  }
}

```
