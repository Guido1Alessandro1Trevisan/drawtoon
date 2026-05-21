# annotate_pages — Bedrock Haiku page-annotation Lambda

SAM Distributed Map that calls **Claude Haiku 4.5 via Bedrock** (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) per manga page. Each Lambda invocation:

1. Reads the filtered/ page JPEG + MAGI panel bboxes
2. Computes per-panel MAGI counts (characters, speech/shout/narration bubbles)
3. Sorts panels into manga reading order (right-to-left, top-to-bottom)
4. Draws red numbered rectangles on the page
5. Calls Bedrock `converse()` with a forced `annotate_page` tool-call
6. Joins Haiku output with MAGI counts + computed bucket (FLUX.2 9-bucket grid)
7. Writes one JSON per page to S3

## Output

```
s3://drawtoon/captions/haiku_page_panel_v1/<chapter>/<page_id>.json
```

Each file:

```json
{
  "page_id": "...", "chapter": "...",
  "page_w_px": 754, "page_h_px": 1088,
  "panel_count": 5, "bucket": "manga-page",
  "s3_page_uri": "s3://drawtoon/.../page.jpg",
  "s3_magi_uri": "s3://drawtoon/.../page.jsonl",
  "page_caption": "1 sentence 15-25 words",
  "panel_types": ["establishing", "beat", "beat", "reaction", "action"],
  "panel_types_str": "establishing beat beat reaction action",
  "panels": [
    {
      "index": 0, "panel_id": "...",
      "bbox": [x1,y1,x2,y2], "width_px": ..., "height_px": ..., "bucket": "landscape",
      "character_count": 1, "speech_bubble_count": 0,
      "shout_bubble_count": 0, "narration_bubble_count": 0,
      "caption": "10-30 words", "shot_size": "MS", "panel_type": "establishing"
    }
  ],
  "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  "usage": {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...},
  "duration_s": 1.42
}
```

## Deploy

```bash
cd /Users/guidotrevisan/Desktop/drawtoon/lora-klein/annotate_pages
sam build
sam deploy \
  --stack-name drawtoon-annotate-pages \
  --region us-east-1 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --no-confirm-changeset \
  --parameter-overrides DatasetBucketName=drawtoon
```

## Smoke-test on 20 pages

```bash
python3 start.py \
  --stack-name drawtoon-annotate-pages \
  --region us-east-1 \
  annotate \
  --max-pages 20 \
  --max-concurrency 20
```

Inspect output at `s3://drawtoon/captions/haiku_page_panel_v1/`.

## Full corpus (3000 concurrency)

```bash
python3 start.py \
  --stack-name drawtoon-annotate-pages \
  --region us-east-1 \
  annotate \
  --max-concurrency 3000
```

Cost estimate at ~30k pages: ~6-15M output tokens through Haiku 4.5 ≈ **$5-15 total**.

## Knobs (env vars on the Lambda)

| Var | Default | Effect |
|---|---|---|
| `DEFAULT_ANNOTATION_MODEL` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock cross-region inference profile |
| `ANNOTATE_TIMEOUT_S` | `120` | Per-page Bedrock call timeout |
| `ANNOTATE_MAX_OUTPUT_TOKENS` | `2048` | Tool-call output budget |
| `ANNOTATE_OVERLAY_JPEG_QUALITY` | `85` | JPEG quality for the numbered overlay sent to Haiku |
| `BEDROCK_MAX_IMAGE_BYTES` | `3600000` | Bedrock per-image cap; oversized images are re-encoded |
| `BEDROCK_MAX_IMAGE_SIDE` | `8000` | Bedrock per-image side cap |

## Files

```
annotate_pages/
├── template.yaml                  ← SAM (2 Lambdas + Step Function)
├── statemachines/
│   └── annotate_pages.asl.json    ← Distributed Map definition
├── src/
│   ├── handlers.py                ← prepare_annotate_config + annotate_page
│   ├── prompt.py                  ← SYSTEM_PROMPT + build_user_prompt
│   ├── schema.py                  ← annotate_page tool JSON Schema
│   ├── bedrock_client.py          ← converse_tool() + image prep + retry
│   ├── overlay.py                 ← red-numbered-bbox drawing (Pillow)
│   └── buckets.py                 ← 9-bucket grid (must match generate.py)
├── start.py                       ← StartExecution CLI
└── requirements.txt
```
