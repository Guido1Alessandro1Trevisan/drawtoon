# Drawtoon Manga Change-of-Angle

Identifies consecutive panels on a manga page that show the **same scene from a
different camera shot** — change of angle, change of shot size (zoom in / out),
or both — where the **background is visibly preserved** across the cut.

Pattern: AWS Step Functions **Distributed Map** → Lambda (mirrors
`workflows/manga_caption`). Each Lambda worker invokes **Kimi K2.6** (Moonshot
AI, OpenAI-SDK-compatible) with reasoning toggleable per-execution.

**Trigger word**: `TRAIN_ANGLE`

## Layout

```
s3://drawtoon/datasets/pages/filtered/<chapter>/<page_id>.jpg          (input page)
s3://drawtoon/datasets/annotations/magi_v3/<chapter>/<page_id>.jsonl    (panel bboxes)
   →   s3://drawtoon/datasets/pages/change_angle/<run>/<chapter>/<page_id>.json
```

Each output JSON carries the panels in **manga reading order** (right-to-left,
top-to-bottom) and zero or more `angle_groups` — groups of two or more
consecutive reading-order indices that share a scene:

```jsonc
{
  "schema_name": "manga_change_of_angle_v1",
  "trigger": "TRAIN_ANGLE",
  "change_angle_run": "kimi_k26_v1",
  "panels_in_reading_order": [
    {"bbox": [x0,y0,x1,y1], "panel_id": "..."},
    ...
  ],
  "angle_groups": [
    {"panel_indices": [0, 1], "reason": "same desk and window in background; camera tilts up"}
  ],
  "summary": {"n_panels": 7, "n_panels_in_groups": 2, "n_groups": 1},
  "verification": {"status": "ok", "model": "kimi-k2.6", "thinking_enabled": true, ...}
}
```

## Pipeline (per execution)

```
SFN execution input
    │
    ▼
PrepareConfig (Lambda)
    │ lists filtered pages, skips ones with no magi_v3 annotation or with an
    │ existing output JSON, writes page_manifest.jsonl + worker_config.json
    │ to S3 under _jobs/<run_id>/, returns the SFN input shape (source +
    │ worker_config + batch + audit).
    ▼
DetectAnglePages (Map, Mode: DISTRIBUTED)
    │ ItemReader streams the manifest JSONL
    │ ItemSelector hands every row to the worker Lambda with config_ref
    │ MaxConcurrency from PrepareConfig output (default 100)
    │ ResultWriter dumps per-item audit JSONL under _audit/<run_id>/
    ▼
DetectAnglePage (Lambda, ×N items)
    │ downloads page + annotation, computes manga reading order, draws
    │ numbered overlays, calls Kimi K2.6 with extra_body={"thinking":
    │ {"type": "enabled"|"disabled"}}, validates groups (≥2 consecutive
    │ in-range indices), writes JSON to S3.
```

No GPU is used; the worker is pure I/O + Kimi API. Reasoning toggle (`thinking_enabled`) is propagated via the worker config so the worker pays one S3 read per cold start and zero per-page overhead.

## Deploy

```bash
cd workflows/manga_change_of_angle
sam build
sam deploy \
  --stack-name drawtoon-manga-change-of-angle \
  --region us-east-1 \
  --profile lineart2-s3 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides DatasetBucketName=drawtoon
```

Kimi API key must live in AWS Secrets Manager (default secret name
`drawtoon/kimi-api-key`), either as a plain string (the raw `sk-...` key) or
as JSON like `{"MOONSHOT_API_KEY": "sk-..."}`. Create it once:

```bash
aws secretsmanager create-secret \
  --name drawtoon/kimi-api-key \
  --secret-string '{"MOONSHOT_API_KEY":"sk-..."}'
```

## Run

`start.py` loads `<repo>/.env` automatically. Defaults match the Drawtoon
production layout. Override anything via the corresponding flag.

```bash
# Detect on all _mangazero_manga chapters with reasoning ON
python3 start.py \
  --stack-name drawtoon-manga-change-of-angle \
  --max-concurrency 100 \
  detect-pages \
    --change-angle-run kimi_k26_on_v1 \
    --include-chapter-regex '_mangazero_manga$'

# Same dataset with reasoning OFF (different output sub-prefix)
python3 start.py \
  --stack-name drawtoon-manga-change-of-angle \
  --max-concurrency 100 \
  detect-pages \
    --change-angle-run kimi_k26_off_v1 \
    --include-chapter-regex '_mangazero_manga$' \
    --no-thinking

# Smoke (small max-pages, dry-run prints SFN input only)
python3 start.py \
  --stack-name drawtoon-manga-change-of-angle \
  --dry-run \
  detect-pages \
    --change-angle-run smoke_v1 \
    --include-chapter-regex '_mangazero_manga$' \
    --max-pages 100
```

Each execution is independent. To compare reasoning ON vs OFF on the same
sample of pages, run two executions with different `--change-angle-run` (so
the output prefixes don't collide) and the same other args.

## Files

```
workflows/manga_change_of_angle/
├── README.md
├── requirements.txt
├── template.yaml                                        # SAM stack
├── start.py                                             # local CLI → start SFN execution
├── src/
│   ├── __init__.py
│   └── handlers.py                                      # prepare + detect-page Lambdas
└── statemachines/
    └── detect_change_of_angle_pages.asl.json            # Distributed Map
```
