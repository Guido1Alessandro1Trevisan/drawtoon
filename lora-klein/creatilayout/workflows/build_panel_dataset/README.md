# Build Panel Dataset

Distributed-Map workflow that joins the per-page source captions
(`manga_caption` output — geometry + bubble types) with the per-page short
captions (`compress_captions` vision_v1 output) and emits the LayouSyn
panel-layout training JSONL.

Same shape as `workflows/compress_captions/`: a `PrepareConfig` → Distributed
`Map` → `Finalize` state machine. Each Map worker handles one page and writes
a small fragment JSONL to S3. The final Lambda concatenates all fragments
into `train.jsonl` + `val.jsonl` (split by chapter hash) and writes audit
stats.

## I/O

| Source | Bucket | Prefix |
|---|---|---|
| Source captions (geometry + bubble TYPE) | `drawtoon` (read-only) | `captions/gemini3_flash_page_panel_v1/` |
| Short captions (prompt) | `drawtoon-layousyn` (read-only) | `captions_short/vision_v1/` |
| **Outputs** | **`drawtoon-layousyn`** | `datasets/panel_layout/` |

Output layout:

```text
s3://drawtoon-layousyn/datasets/panel_layout/
    train.jsonl
    val.jsonl
    _fragments/<run_id>/<chapter>__<page>.jsonl   # per-page intermediate
    _jobs/<run_id>/page_manifest.jsonl
    _jobs/<run_id>/worker_config.json
    _audit/<run_id>/...                            # Map ResultWriter dump
    _audit/<run_id>/stats.json                     # Finalize summary
```

JSONL row schema (matches `MangaLayout`):

```jsonc
{
  "prompt": "medium shot, eye-level, two characters fighting in a street, 1 speech from character 1",
  "width":  0.683,
  "height": 0.412,
  "items": [
    {"label": "character 1",                    "bbox": [-0.85, -0.81, -0.29, 0.56]},
    {"label": "speech bubble from character 1", "bbox": [-0.80, -0.94, -0.44, -0.63]}
  ],
  "_meta": {"chapter": "...", "panel_id": "panel_000", "panel_bbox_px": [...], "attribution_methods": ["tail", "tail"]}
}
```

## Deploy

```bash
cd /Users/guidotrevisan/Desktop/drawtoon/lora-klein/creatilayout/workflows/build_panel_dataset
sam build
sam deploy \
  --stack-name drawtoon-build-panel-dataset \
  --region us-east-1 --profile default --capabilities CAPABILITY_IAM --resolve-s3 \
  --no-confirm-changeset --no-fail-on-empty-changeset \
  --parameter-overrides SourceBucketName=drawtoon OutputBucketName=drawtoon-layousyn
```

## Run

```bash
python3 start.py \
  --stack-name drawtoon-build-panel-dataset \
  --profile default \
  --max-concurrency 3000 \
  build \
  --source-caption-run gemini3_flash_page_panel_v1 \
  --short-caption-run vision_v1 \
  --output-prefix datasets/panel_layout \
  --split-val-frac 0.05
```

Add `--max-pages 500` to run a smoke pass first.

## Notes

- Speaker attribution uses the **tail-aware** algorithm in `src/attribution.py`
  (same logic as `data_prep/attribution.py`).
- Aspect clamp: panels with `width/height < 0.2 or > 5.0` are dropped.
- Tiny-panel filter: `width × height < 0.005` of page area → dropped.
- Reading order: RTL for `_manga`-suffix chapters, LTR for `_manwa`.
- Train/val split: 95/5 by `sha256(chapter)[:8]` (whole series stays in one
  split — no leakage).
- Fragments left under `_fragments/<run_id>/` after the final concat; delete
  when you're done debugging a run.
