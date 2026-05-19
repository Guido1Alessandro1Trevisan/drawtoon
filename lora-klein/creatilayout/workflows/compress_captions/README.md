# Compress Panel Captions

Distributed Map that produces a **10–20 word panel-layout caption** per panel
for the LayouSyn panel-layout fine-tune.

**Mode: vision + thinking.** For each panel the worker downloads the page
image, crops the panel by its MAGI v3 bbox, and sends the cropped image to
Gemini 3 Flash Preview with `thinking_level=high`. Gemini returns three
fields (`shot_size`, `camera_angle`, `action_phrase`). Code computes the
deterministic counts (characters, bubbles by type) and the speaker
attribution (nearest-character-by-centroid on bubble bbox), then assembles
the final caption.

A/B against text-only baseline (vision+thinking won on every axis):

| | Text-only | Vision+thinking |
|---|---|---|
| Over-cap (>20 words) | 15% | **6%** |
| p50 word count | 16 | **14** |
| Abstract-panel handling | wordy | crisp |
| Action phrase quality | sometimes broken grammar | verb-led, clean |

## Caption format

```
<shot_size>, <camera_angle>[, <N> character[s] <action>][, <per-character bubble groups>][, <N> narration]
```

Rules:
- Skip the character clause if `chars=0`. Skip the bubble clause if no bubbles.
- Bubbles always attributed to a character ordinal (`from character N`).
- Narration is always speakerless. Off-panel speech (a speech bubble with no
  characters present) is relabeled as narration.
- Bubble types are broken out per character: `1 speech 2 shout from character 1`.

Examples:

```
medium shot, low-angle, two characters fighting in a street, 1 speech from character 1, 1 shout from character 2
close-up, eye-level, one character looking up, 2 speech from character 1
wide shot, high-angle, empty schoolyard at dusk
long shot, low-angle, one character walking
extreme close-up, dutch tilt, one character screaming, 1 shout from character 1
ambiguous, ambiguous, checkered transition pattern, 4 narration
```

## Closed vocabularies

- **Shot size:** `extreme close-up | close-up | medium close-up | medium shot | medium long shot | long shot | extreme long shot | ambiguous`
- **Camera angle:** `eye-level | low-angle | high-angle | overhead | dutch tilt | over-the-shoulder | POV | ambiguous`
- **Bubble types** (from MAGI v3 `text_region.type`): `Speech Bubble | Shout Bubble | Narration Bubble`

## Input

```text
# Manifest source (one JSON per page, contains panel bboxes + characters + bubbles)
s3://drawtoon/captions/<source_caption_run>/<chapter>/<page_id>.json

# Page image (looked up via sources.page_key inside the manifest JSON)
s3://drawtoon/datasets/pages/text_removed/<chapter>/<page_id>.{png,jpg}
```

Default `source_caption_run` = `gemini3_flash_page_panel_v1` (the existing
deployed run). Override with `--source-caption-run`.

## Output

```text
s3://drawtoon/captions_short/<output_run>/<chapter>/<page_id>.json
```

```jsonc
{
  "schema_version": 3,
  "caption_type": "gemini_panel_compress_vision_v1",
  "model": {"id": "gemini-3-flash-preview", "mode": "vision", "thinking_level": "high"},
  "chapter": "...",
  "page_id": "...",
  "panels": [
    {
      "panel_index": 0,
      "bbox": [...], "bbox_norm": [...],
      "character_count": 2,
      "text_bubble_count": 2,
      "shot_size": "medium shot",
      "camera_angle": "eye-level",
      "action_phrase": "fighting in a street",
      "attributed_bubbles": [
        {"type": "speech", "speaker_ordinal": 1},
        {"type": "shout",  "speaker_ordinal": 2}
      ],
      "short_caption": "medium shot, eye-level, two characters fighting in a street, 1 speech from character 1, 1 shout from character 2",
      "short_word_count": 18,
      "exceeded_word_cap": false
    }
  ]
}
```

## Deploy

```bash
cd /Users/guidotrevisan/Desktop/drawtoon/lora-klein/creatilayout/workflows/compress_captions
sam build
sam deploy \
  --stack-name drawtoon-compress-captions \
  --region us-east-1 \
  --profile lineart2-s3 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides DatasetBucketName=drawtoon
```

## Launch

```bash
# Dry run
python start.py \
  --stack-name drawtoon-compress-captions \
  --profile lineart2-s3 \
  --dry-run \
  --max-concurrency 200 \
  compress \
  --source-caption-run gemini3_flash_page_panel_v1 \
  --output-run vision_v1 \
  --include-chapter-regex '_mangazero_manga$'

# Live run — drop --dry-run
```

The default `--max-concurrency` is 200 (lower than the text-only workflow
because each per-page Lambda now runs ~6 Gemini calls in parallel internally,
multiplying effective concurrency by ~8). Gemini Flash 3 has a 1000 RPM quota;
200 × 8 = 1600 max — tune down if you see 429s.

## Cost

~$0.0015 per panel (vision input + thinking + output at Gemini 3 Flash pricing).
For 100k panels: **~$150 total**. Within the HANDOFF.md $90 budget for the
panel pipeline if you cap the corpus or share budget with `page_caption`.

## Knobs (env vars set on the worker Lambda)

| Var | Default | Effect |
|---|---|---|
| `COMPRESS_THINKING_LEVEL` | `high` | `high` / `medium` / `low` thinking budget |
| `COMPRESS_PANEL_PARALLELISM` | `8` | Gemini calls in parallel per page worker |
| `COMPRESS_MAX_PANEL_SIDE` | `1024` | Max panel image side before resize |
| `COMPRESS_WORD_HARD_CAP` | `20` | Marks `exceeded_word_cap` in output |

## Tests

```bash
# 10 real pages, end-to-end, parallel — used to validate any caption changes
uv run --quiet --with boto3 --with google-genai --with pillow \
  python tests/test_10_panels.py
```

## Where speaker-attribution lives

Nearest-character-by-centroid runs in `src/assembly.py` against
`panel_bbox_norm` of each text_bubble. The geometry is decent but not perfect
when tails matter — the proper tail-aware attribution is a later concern for
`build_panel_dataset.py` (HANDOFF.md §4, currently a TODO).
