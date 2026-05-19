# creatilayout — drawtoon's LayouSyn fine-tune

Layout-generation pipeline. Trains two **18M-param LayouSyn DiT** models
(arXiv 2505.04718): a **panel model** (places character + speech-bubble
bboxes inside one panel) and a **page model** (places panel rects on one
page). Both are fine-tunes from the released HF checkpoint
`dsrivastavv/Lay-Your-Scene/grit/model.pt`.

## Data flow

```
s3://drawtoon/captions/gemini3_flash_page_panel_v1/<chapter>/<page>.json
  (manga_caption — MAGI v3 geometry + Haiku bubble TYPE classifications)
                            │
                            ▼
workflows/compress_captions/   one Gemini-3-flash-preview vision call per
                               panel; emits {shot_size, camera_angle,
                               action_phrase} + assembled short_caption
                            │
                            ▼
s3://drawtoon-layousyn/captions_short/vision_v1/<chapter>/<page>.json
                            │
                            ▼
workflows/build_panel_dataset/  joins source + short captions; tail-aware
                                speaker attribution; emits JSONL rows
                            │
                            ▼
s3://drawtoon-layousyn/datasets/panel_layout/{train,val}.jsonl
                            │
                            ▼
training/modal_train_layousyn.py   torchrun on a single H200; warm-starts
                                   from HF checkpoint
                            │
                            ▼
s3://drawtoon-layousyn/models/<run_name>/{checkpoints,final.pt,config.json}
```

A separate `workflows/page_caption/` produces one 10–15 word caption per
page for the page-layout model — the page-training data-prep step is
TODO (mirror `build_panel_dataset` but with one row per page, items = panel
rects with size_class labels).

## S3 buckets

| Bucket | Role | Where reads/writes happen |
|---|---|---|
| `drawtoon` | READ-ONLY upstream artifacts (pages, MAGI annotations, Gemini long captions) | Read by `compress_captions` + `build_panel_dataset` |
| `drawtoon-layousyn` | ALL LayouSyn outputs | Written by `compress_captions`, `build_panel_dataset`, `modal_train_layousyn`. Nukeable/rebuildable. |

## Repo layout

```text
lora-klein/creatilayout/
├── AGENTS.md                       # this file — only top-level doc
├── layousyn/                       # vendored mlpc-ucsd/Lay-Your-Scene @769e9bf
│   ├── layousyn/data/manga_layout.py        # drawtoon dataset adapter (NEW)
│   ├── layousyn/data/__init__.py            # patched: registers MangaLayout
│   ├── panel/config.json                    # drawtoon panel fine-tune config
│   ├── page/config.json                     # drawtoon page fine-tune config
│   └── …                                    # upstream pristine otherwise
├── training/
│   └── modal_train_layousyn.py     # Modal entrypoint (H200, single GPU)
└── workflows/                      # Step Functions Distributed Map workflows
    ├── compress_captions/          # see workflows/compress_captions/README.md
    ├── build_panel_dataset/        # see workflows/build_panel_dataset/README.md
    └── page_caption/               # see workflows/page_caption/README.md (TODO data-prep)
```

**Do not edit the vendored upstream code** under `layousyn/` except the two
files marked above (`data/manga_layout.py`, `data/__init__.py`). All other
upstream files are pristine `mlpc-ucsd/Lay-Your-Scene@769e9bf` and warmstart
loads cleanly because of that.

## Caption format (panel model)

```
<shot_size>, <camera_angle>[, <N> character[s] <action>][, <per-character bubble groups>][, <N> narration]
```

- `shot_size` ∈ closed enum: `extreme close-up | close-up | medium close-up | medium shot | medium long shot | long shot | extreme long shot | ambiguous`
- `camera_angle` ∈ closed enum: `eye-level | low-angle | high-angle | overhead | dutch tilt | over-the-shoulder | POV | ambiguous`
- Bubbles **always attributed** to a character ordinal — `1 speech from character 1, 1 shout from character 2`.
- Narration is always speaker-less.
- If `N == 0`: skip that clause entirely.
- 10–20 words total.

Examples:

```
wide shot, low-angle, two characters fighting in a street, 1 speech from character 1
close-up, eye-level, one character looking up, 2 speech from character 1
medium shot, over-the-shoulder, three characters talking in a classroom, 1 speech from character 1, 1 speech from character 2, 1 narration
wide shot, high-angle, empty schoolyard at dusk
```

## Per-item label format

| Item kind | Label |
|---|---|
| Character (training & inference) | `"character 1"`, `"character 2"`, … (1-indexed reading order) |
| Speech bubble, tail-attributed | `"speech bubble from character N"` |
| Shout bubble, tail-attributed | `"shout bubble from character N"` |
| Narration | `"narration bubble"` (always speakerless) |
| Unattributed speech/shout (no tail + no nearby char) | `"speech bubble"` / `"shout bubble"` |
| Off-panel speech (chars=0 + bubble present) | relabeled to `"narration bubble"` |

`bbox` per item is `[x0, y0, x1, y1]` in `[-1, 1]` panel-local coords. The
training row also carries `width` and `height` (panel_w_px / page_w_px and
panel_h_px / page_h_px, both in `[0, 1]`). Upstream LayouSyn consumes
`aspect_ratio = width / height` via its existing `ar_embedder` — no model
patch.

## Tail attribution algorithm

`workflows/build_panel_dataset/src/attribution.py`:

1. For each Speech/Shout bubble, find the tail with the highest
   `overlap_area / tail_area`. If ≥ `0.2`, that tail is "attached".
2. Tail tip = the bbox corner farthest from the bubble centroid.
3. Speaker = the character whose bbox is closest to the tip. If distance
   > `0.30 × panel_diagonal`, leave unattributed.
4. Fallback when no tail: nearest-character-by-centroid with same distance
   cap.
5. If no characters at all, relabel non-narration bubbles as narration.

## Reference: upstream paper config + recommended fine-tune setup

| Param | Paper (from-scratch, 2× A5000) | drawtoon fine-tune (single H200) |
|---|---|---|
| Model | DiT-S (8L × 8H × 256d, ~18M params) | unchanged |
| Diffusion steps | 100 train (DDPM), 10 eval (DDIM) | unchanged |
| Noise schedule | linear, α-scale `s = 2.0` | unchanged |
| CFG scale (inference) | 2.0 | 2.0 |
| Mixed precision | bf16 autocast + GradScaler | unchanged |
| Grad clip | 1.0 | unchanged |
| EMA decay | 0.9999 | unchanged |
| Optimizer | AdamW betas (0.9, 0.999), weight decay 0.0 | unchanged |
| Warmup / LR schedule | none (constant) | none |
| **Batch size** | 256 (128/GPU) | **1024** (1× H200, sqrt-scaled LR) |
| **LR** | 1e-4 | **2e-4** (= 1e-4 × sqrt(1024/256)) |
| **Epochs** | 400 (~800K steps) | **5–10** (warmstart converges fast) |
| **Hardware** | 2× A5000 (48 GB), ~20 h | 1× H200 (141 GB), ~20 min |
| Cost | n/a | ~$2/run |

The `LAYOUSYN_GPU` env var on the **local** side controls the GPU type baked
into the Modal function decorator at module load.

## Runbook

```bash
# 1. Deploy compress_captions  (one-shot)
cd lora-klein/creatilayout/workflows/compress_captions
sam build && sam deploy \
  --stack-name drawtoon-compress-captions \
  --region us-east-1 --profile default --capabilities CAPABILITY_IAM --resolve-s3 \
  --no-confirm-changeset --no-fail-on-empty-changeset \
  --parameter-overrides SourceBucketName=drawtoon OutputBucketName=drawtoon-layousyn \
                        CompressThinkingLevel=off

# 2. Run compress_captions  (~20 min, ~$10 with thinking=off)
python3 start.py --stack-name drawtoon-compress-captions --profile default \
  --max-concurrency 3000 compress \
  --source-caption-run gemini3_flash_page_panel_v1 --output-run vision_v1

# 3. Deploy build_panel_dataset  (one-shot)
cd ../build_panel_dataset
sam build && sam deploy \
  --stack-name drawtoon-build-panel-dataset \
  --region us-east-1 --profile default --capabilities CAPABILITY_IAM --resolve-s3 \
  --no-confirm-changeset --no-fail-on-empty-changeset \
  --parameter-overrides SourceBucketName=drawtoon OutputBucketName=drawtoon-layousyn

# 4. Run build_panel_dataset  (~10 min)
python3 start.py --stack-name drawtoon-build-panel-dataset --profile default \
  --max-concurrency 3000 build \
  --source-caption-run gemini3_flash_page_panel_v1 \
  --short-caption-run vision_v1 \
  --output-prefix datasets/panel_layout \
  --split-val-frac 0.05

# 5. Smoke train  (~10 min on H200)
modal run lora-klein/creatilayout/training/modal_train_layousyn.py \
  --variant panel --run-name smoke --epochs 1 --ckpt-every 100 \
  --batch-size 256 --lr 1e-4 --debug

# 6. Full panel fine-tune  (~20 min on H200, ~$2)
modal run lora-klein/creatilayout/training/modal_train_layousyn.py \
  --variant panel --run-name layousyn-panel-v1 \
  --epochs 10 --batch-size 1024 --lr 2e-4 --ckpt-every 500
```

## Modal secrets

- `lineart2-aws-s3` — AWS creds for both buckets.
- `hf-token` — HF token for warmstart download (public repo, but a token
  avoids anonymous rate limits).

## Per-workflow READMEs

Detailed CLI + per-knob docs at:

- `workflows/compress_captions/README.md`
- `workflows/build_panel_dataset/README.md`
- `workflows/page_caption/README.md`

## Sanity checks before a training run

1. `aws s3 ls s3://drawtoon-layousyn/datasets/panel_layout/train.jsonl` — non-empty.
2. `aws s3 cp s3://drawtoon-layousyn/datasets/panel_layout/_audit/<run>/stats.json -` — eyeball the attribution method histogram (tail vs centroid vs none) and the bubble-type counts.
3. `head -1` a few JSONL rows — confirm shape: `prompt`, `width`, `height`, `items[].label`, `items[].bbox` in `[-1, 1]`.

## Anti-patterns

- Don't edit upstream `layousyn/` files other than `data/__init__.py` +
  `data/manga_layout.py`. Warmstart breaks if you do.
- Don't add a `ThreadPoolExecutor` inside the compress Lambda — the
  google-genai SDK's httpx pool closes under threaded use ("Cannot send a
  request, as the client has been closed"). One Gemini call per Lambda,
  sequential per-panel, like `manga_caption`.
- Don't run at concurrency > ~1000 with `thinking_level=high` — Gemini
  throttles. `thinking_level=off` handles 3000 fine.
- Don't store anything in the `drawtoon` bucket from this pipeline. It's
  read-only for us.
