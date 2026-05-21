# Agent Handoff — Change-of-Angle Detection at Catalog Scale

You are picking up a deployed, tested change-of-angle detection workflow. Your job is to run it across the full manga + comic catalog and aggregate the output into final TRAIN_ANGLE training groups, staying within budget.

## Budget

| Resource | Amount |
|---|---|
| Kimi K2.6 balance | **$1,105.65 USD** |
| Already spent this session | ~$3.50 (the smoke + 2 jjk sweeps) |
| Effective budget for catalog | **~$1,100** |

**Per-page cost (measured on 100 jujutsu-kaisen pages with the v2 prompt)**:

| Mode | $/page | What 1,000 pages costs | What 100,000 pages costs |
|---|---|---|---|
| reasoning OFF | **$0.00236** | $2.36 | **$236** |
| reasoning ON  | $0.01305 | $13.05 | $1,305 |
| both passes   | $0.01541 | $15.41 | $1,541 |

Conclusion: **OFF easily fits the budget at any realistic catalog size.** ON is feasible only if the catalog is ≤80k pages (or as a QA pass on a subset).

## Goal

For every page in `s3://drawtoon/datasets/pages/filtered/<chapter>/`, detect groups of panels that:
1. Share the **same background scenery** visible behind the characters (dominant requirement)
2. Have at least **one shared character** in all panels of the group

Each group becomes one TRAIN_ANGLE training example downstream (one ctrl panel + one target panel + reference, derived from the same group; the consumer-side dataset builder handles that).

The detection workflow already exists and writes one JSON per page. **Your only job is to run it at scale and produce a clean aggregated dataset.**

## Workflow location and shape

```
workflows/manga_change_of_angle/
├── README.md                  # workflow architecture overview
├── template.yaml              # SAM stack (deployed)
├── src/handlers.py            # PrepareConfig + DetectChangeOfAnglePage Lambdas
├── statemachines/
│   └── detect_change_of_angle_pages.asl.json   # Distributed Map
└── start.py                   # local CLI → starts SFN execution
```

Pattern: SAM stack with two Lambda functions and a Step Functions Distributed Map (mirrors `workflows/manga_caption`). Calls Kimi K2.6 (Moonshot AI, OpenAI-SDK-compatible) per page.

- **Stack name**: `drawtoon-manga-change-of-angle` (already deployed in `us-east-1`)
- **State-machine ARN output**: `MangaChangeOfAngleStateMachineArn`
- **Kimi API key**: AWS Secrets Manager secret `drawtoon/kimi-api-key` (already populated)
- **Trigger word emitted in every output**: `TRAIN_ANGLE`

## How a single page becomes an output JSON

1. Lambda downloads page from `datasets/pages/filtered/<chapter>/<page>.jpg`
2. Downloads magi_v3 annotation from `datasets/annotations/magi_v3/<chapter>/<page>.jsonl`
3. Sorts panels into manga reading order (right-to-left, top-to-bottom)
4. Renders numbered colored overlay boxes on the page image
5. Sends image + simple text prompt to Kimi K2.6 (current prompt in `src/handlers.py` → `KIMI_SYSTEM_PROMPT`)
6. Validates groups (≥2 panel indices, in range, deduped — consecutive NOT required)
7. Writes JSON to `datasets/pages/change_angle/<change_angle_run>/<chapter>/<page>.json`

Output schema (one JSON per page):

```jsonc
{
  "schema_name": "manga_change_of_angle_v1",
  "trigger": "TRAIN_ANGLE",
  "chapter": "...",
  "page_id": "...",
  "panels_in_reading_order": [{"bbox": [x0,y0,x1,y1], "panel_id": "..."}, ...],
  "angle_groups": [
    {"panel_indices": [0, 1], "reason": "same desk and window in the background"},
    {"panel_indices": [3, 5], "reason": "same alley wall behind both characters"}
  ],
  "summary": {"n_panels": 7, "n_panels_in_groups": 4, "n_groups": 2},
  "verification": {
    "status": "ok",
    "model": "kimi-k2.6",
    "thinking_enabled": false,
    "usage": {"prompt_tokens": 2105, "completion_tokens": 110, "cached_tokens": 372}
  }
}
```

`verification.usage` is what lets you compute exact cost for any run; see `/tmp/aggregate_cost.py` for a working aggregator.

## Recommended plan

### Step 1: Inventory (no API spend)

Count pages on S3 to size the run:

```bash
aws s3 ls s3://drawtoon/datasets/pages/filtered/ --recursive | awk '
  { k=$NF
    if (k ~ /\.(jpg|jpeg|png|webp)$/) {
      split(k, p, "/"); ch=p[3]
      if (ch ~ /_manga$/) m++
      else if (ch ~ /_comic$/) c++
    }
  }
  END { print "manga:", m, "comic:", c, "total:", m+c }'
```

Also verify magi_v3 annotation coverage for `_comic` chapters — the worker skips pages without an annotation, so any `_comic` chapter that hasn't been MAGI-annotated is silently dropped. If coverage is bad, run `workflows/manga_annotate` on the missing chapters **before** the change-of-angle run.

### Step 2: Reasoning OFF on everything

OFF is the bulk run. Single SFN execution per content type:

```bash
cd workflows/manga_change_of_angle

# manga
python3 start.py \
  --stack-name drawtoon-manga-change-of-angle \
  --max-concurrency 200 \
  --tolerated-failure-count 500 \
  detect-pages \
    --change-angle-run kimi_k26_off_manga_v1 \
    --include-chapter-regex '_manga$' \
    --no-thinking

# comic
python3 start.py \
  --stack-name drawtoon-manga-change-of-angle \
  --max-concurrency 200 \
  --tolerated-failure-count 500 \
  detect-pages \
    --change-angle-run kimi_k26_off_comic_v1 \
    --include-chapter-regex '_comic$' \
    --no-thinking
```

`--tolerated-failure-count 500` lets transient Kimi/Lambda errors slide without killing the whole map. Re-run the same command without `--overwrite` to retry only the failures.

Expected cost at the measured rate:
- 100k pages: ~$236
- 200k pages: ~$471

Both fit budget comfortably.

### Step 3: Optional ON pass for QA only

If you want to spot-check whether OFF is over-grouping, run ON on a random sample of ~1,000 pages (cost ≈ $13). Use a **different** `--change-angle-run` (e.g. `kimi_k26_on_qa_v1`) so it doesn't collide with OFF.

Do **NOT** run ON on the whole catalog — it would cost ~$1,300 for 100k pages and based on the jjk smoke ON finds **fewer** groups than OFF (it's more conservative). The marginal value isn't worth the spend.

### Step 4: Aggregate into final training dataset

After both SFN executions reach `SUCCEEDED`:

```bash
# Compute exact cost (uses verification.usage in each output JSON)
python3 /tmp/aggregate_cost.py   # adjust RUNS dict to point at the new prefixes

# Produce the final TRAIN_ANGLE training manifest
#   - one JSONL row per detected group (not per page)
#   - each row carries: chapter, page_id, group_indices, group_panel_bboxes,
#     reason, source page S3 URI
#   - filter rows that you don't want (e.g. groups of 2 where reason mentions
#     "blank background" or "abstract")
```

There is no aggregator script yet — you write that. Suggested output:

```
s3://drawtoon/datasets/train_angle/manifests/<run>/groups.jsonl
```

Each line:
```jsonc
{
  "trigger": "TRAIN_ANGLE",
  "group_id": "<chapter>__<page_id>__g<i>",
  "chapter": "...",
  "page_id": "...",
  "page_key": "datasets/pages/filtered/...",
  "panel_bboxes": [[x0,y0,x1,y1], ...],   // in order from panels_in_reading_order
  "reason": "same desk and window in the background",
  "n_panels": 2
}
```

## Cost guardrails

Before you launch:
- Hard cap your execution concurrency at 200 (already in the recommended command). Avoid hammering Kimi above that — the rate-limit response surfaces as 5xx and chews your tolerated-failure budget.
- Tolerated failure budget: 500. If a run exceeds that, **stop and investigate** before retrying. A failure cliff means something broke (rate limit, Kimi outage, bad image), not noise.
- After each major run, immediately run the cost aggregator and reconcile against `Account Overview > Total Consumption`. Catch runaway spend within $10.

## What NOT to do

- **Do not** rerun completed pages. The worker already skips when an output exists unless `--overwrite` is passed. Don't pass `--overwrite` on a re-run.
- **Do not** try to "improve" the prompt mid-run. Iterating on prompts means re-running the whole batch. The v2 prompt was tuned on jjk; if it's clearly wrong on `_comic`, run a 100-page comic smoke first before launching the whole comic batch.
- **Do not** use reasoning ON for the bulk run. Measured 5–8× cost for fewer groups. OFF is the production setting.
- **Do not** modify `src/handlers.py` and forget to rebuild + redeploy. Always do `sam build && sam deploy --stack-name drawtoon-manga-change-of-angle --no-confirm-changeset` after a code change.
- **Do not** count `_manwa` (vertical webtoons) in this run. The reading-order logic and manga-page assumption don't apply. There's a separate `manga_annotate` manwa-sheet flow but no change-of-angle equivalent — defer.

## Quick reference

```bash
# Deploy after handler changes
cd workflows/manga_change_of_angle
sam build
sam deploy --stack-name drawtoon-manga-change-of-angle \
  --region us-east-1 --capabilities CAPABILITY_IAM --resolve-s3 \
  --no-confirm-changeset --parameter-overrides DatasetBucketName=drawtoon

# Check stack status
aws cloudformation describe-stacks --stack-name drawtoon-manga-change-of-angle \
  --query 'Stacks[0].StackStatus' --output text

# Check Kimi balance
# (visit https://platform.kimi.ai/console — no CLI as of writing)

# Tail Lambda logs during a run
aws logs tail "/aws/lambda/$(aws cloudformation describe-stack-resources \
  --stack-name drawtoon-manga-change-of-angle \
  --query \"StackResources[?LogicalResourceId=='DetectChangeOfAnglePageFunction'].PhysicalResourceId\" \
  --output text)" --follow

# Poll SFN execution
aws stepfunctions describe-execution --execution-arn <ARN> --query 'status'

# Count outputs landed
aws s3 ls s3://drawtoon/datasets/pages/change_angle/<run>/ --recursive | wc -l
```

## Earlier context (read these for the why)

- Workflow README: `workflows/manga_change_of_angle/README.md`
- v2 prompt (what's currently deployed): `src/handlers.py` → `KIMI_SYSTEM_PROMPT`
- Smoke comparison artifacts:
  - `artifacts/gemini_panel_slicing/` (v1 prompt — keep but reference, the v2 is below)
  - `artifacts/gemini_panel_slicing/jjk_on_vs_off_v2/` (v2 prompt, 65 sheets, jjk only)
- Cost aggregator: `/tmp/aggregate_cost.py`
- Sheet renderer: `/tmp/render_jjk_compare.py`

If you change the prompt, copy these two scripts into `tools/` under this workflow with the new prefixes — don't keep using `/tmp/` paths.
