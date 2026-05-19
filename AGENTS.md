# Repo Guide

## Scope

This file applies to the whole repository unless a more specific `AGENTS.md` exists in a child directory.

Known scoped guides:

- `dashboard/AGENTS.md`
- `lora-klein/training/AGENTS.md`

Read the scoped guide before editing files under those directories.

## Purpose

This repo contains Drawtoon manga captioning, dashboard, FLUX.2 Klein training, and validation tooling.

Primary active areas:

- `workflows/`: AWS/S3 workflow code for captioning and dataset preparation.
- `dashboard/`: Next.js dashboard plus Modal inference backend helpers.
- `lora-klein/training/`: Modal ai-toolkit training launcher and presets.
- `lora-klein/validation/`: fixed validation set and Modal validation/evaluation runner.
- `EVALUATION.md`: current validation report for the latest base vs fine-tuned comparison.

## Operational Rules

- Treat S3 as the durable source of truth for datasets, checkpoints, generated validation images, and metric summaries.
- Prefer `us-east-1` for AWS work unless a task explicitly uses another region.
- Do not hardcode Hugging Face, AWS, Modal, or Anthropic tokens. Use Modal Secrets and environment variables.
- Do not delete or rewrite existing generated artifacts unless the user explicitly asks for a regeneration or cleanup.
- When a run already has generated images, use evaluation-only mode instead of regenerating images.
- Keep Modal GPU work explicit about GPU, CPU, memory, timeout, and output S3 paths.
- Preserve user or prior-agent changes in the working tree. Do not revert unrelated files.

## AWS Lambda Concurrency Policy — DO NOT use Reserved Concurrency

**Never set `ReservedConcurrentExecutions` on any Lambda in this account.** Every function must share the account's unreserved pool.

Why: the account-level quota is `ConcurrentExecutions = 3000`. Each function that reserves N takes N out of the shared pool, even when idle. Tonight we found 8 functions had quietly reserved 2,870 of the 3,000 between them — leaving only **130 unreserved** for everything else. That cap silently throttled the manga-caption Distributed Map at ~130 concurrent and gave a 40% Map-item failure rate (Lambda `TooManyRequestsException` after exhausted retries), even though the Map asked for 500 and Gemini was at 17% of its quota.

Rules:

- **Never call `put_function_concurrency(ReservedConcurrentExecutions=…)`** in deploy scripts, CDK, SAM, Terraform, or by hand in the console. The two deploy scripts in `workflows/download_scraper/aws/scripts/deploy*.py` default `--reserved-concurrency 0` and actively `delete_function_concurrency` on every redeploy — keep that behaviour when copying them.
- **Never put `ReservedConcurrentExecutions:` in a SAM/CFN template.** Step Functions Distributed Maps drive every Lambda in this repo, and they assume the unreserved pool is the full 3,000.
- If a workload genuinely needs guaranteed capacity, raise the account quota (Service Quotas `L-B99A9384`, Concurrent executions) rather than carve it out of the shared pool. Talk to the user first.
- When auditing concurrency: `aws lambda get-account-settings --region us-east-1` must show `UnreservedConcurrentExecutions == ConcurrentExecutions` (i.e. nothing reserved). If it doesn't, run `aws lambda list-functions` and `aws lambda get-function-concurrency --function-name <fn>` per function, then `aws lambda delete-function-concurrency --function-name <fn>` on any with a reservation.
- All Step Functions Distributed Map launchers in this repo (e.g. `workflows/manga_caption/start.py --max-concurrency …`) should be safe to set up to ~2,500 concurrent; any throttling you see at lower numbers means somebody re-introduced reserved concurrency — find it and remove it.

## Suffix Distribution Reporting

When the user asks for "suffix distribution" (or anything equivalent like "show me the suffix split", "how many manga / manwa / manhua / comic per bucket"), always present the answer as this exact table, populated with the live counts you just measured. Do not summarise as bullets — render the table.

```text
Suffix distribution:

┌──────────────────────────────────────┬───────┬───────┬────────┬───────┐
│                Prefix                │ manga │ manwa │ manhua │ comic │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┤
│ pages/filtered                       │   ... │   ... │    ... │   ... │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┤
│ pages/text_removed                   │   ... │   ... │    ... │   ... │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┤
│ pages/single                         │   ... │   ... │    ... │   ... │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┤
│ annotations/magi_v3                  │   ... │   ... │    ... │   ... │
├──────────────────────────────────────┼───────┼───────┼────────┼───────┤
│ captions/gemini3_flash_page_panel_v1 │   ... │   ... │    ... │   ... │
└──────────────────────────────────────┴───────┴───────┴────────┴───────┘
```

Use `–` (en-dash) for any cell where the count is zero — it reads more clearly than `0`. After the table, add one short "key observation" paragraph that flags any pipeline gap (e.g. chapters that exist in `pages/single` but have never been filtered, annotated, or captioned). To measure: list `s3://drawtoon/<prefix>/` with `aws s3 ls`, group by suffix (`_manga | _manwa | _manhua | _manha | _comic`), and count distinct chapter dirs.

## WEBTOON / Manhwa Imports

Active import workspace:

```text
artifacts/webtoon_manga/
```

Current direct-to-dataset importer:

```text
artifacts/webtoon_manga/direct_single_worker.py
artifacts/webtoon_manga/deploy_direct_single.py
artifacts/webtoon_manga/manifest/direct_single_series_3000.jsonl
```

Use this flow for authorized WEBTOON/manhwa imports that should land directly in the training page dataset:

1. Put one title per line in `manifest/direct_single_series_3000.jsonl` as JSONL with `name`, `series_slug`, `title_no`, and `list_url`.
2. Keep `series_slug` without `_manwa`; the worker writes to:

```text
s3://drawtoon/datasets/pages/single/<series_slug>_manwa/
```

3. Launch from `us-east-1` with a hard per-title cap and distributed parallelism:

```bash
python3 artifacts/webtoon_manga/deploy_direct_single.py \
  --max-pages-per-series 3000 \
  --max-concurrency 15 \
  --image-workers 64 \
  --lambda-memory 4096 \
  --lambda-timeout 900 \
  --proxy-mode auto \
  --start
```

The deployer runs a dry smoke test before starting the Step Functions Distributed Map. The worker fetches direct first and falls back to Decodo proxies only for source HTML/image fetches when needed. S3 reads/writes must stay direct; do not route S3 through proxies and do not hardcode proxy credentials in repo files.

The direct worker discovers episode URLs, shuffles episodes deterministically by seed, downloads candidate images in memory, applies the dimension/story-run filter immediately, and uploads only kept pages. It does not write raw pages to `s3://drawtoon/datasets/pages/source/`.

Monitor active runs with:

```bash
aws stepfunctions describe-map-run --map-run-arn <map-run-arn> --region us-east-1
aws s3 ls s3://drawtoon/datasets/pages/manifests/webtoon_manga_direct_single/status/<run-id>/ --region us-east-1
```

`s3://drawtoon/datasets/pages/single/` is the canonical output. Do not delete from `single/` unless the user explicitly asks. `s3://drawtoon/datasets/pages/source/` was used by an older raw-download flow and is disposable once cleaned.

Recent completed direct import:

```text
run_id: 20260518T060621Z
execution: arn:aws:states:us-east-1:274213480586:execution:webtoon_manga_direct_single_map:webtoon-manga-direct-single-20260518T060621Z
output: s3://drawtoon/datasets/pages/single/*_manwa/
result: 15/15 title workers succeeded, 41,045 filtered pages written, 0 worker errors
```

## Validation

Active validation script:

```text
lora-klein/validation/validate_run.py
```

Current fixed validation set:

```text
lora-klein/validation/datasets/generalist
lora-klein/validation/datasets/attack_on_titan
```

Current eval ID:

```text
full200_step2000_20260515
```

Current internal metric suite:

- `CMMD`
- `SigLIP2-T`
- `DINOv3-I`
- `DINOv3-C`

Manga-specific metric:

- Haiku bubble-type judge for speech, narration, and shout bubble preservation.

Dialog/Magi F1 is intentionally not part of the active validation suite.

Use this mode to score existing generated images:

```bash
uv run --active --with modal python -m modal run lora-klein/validation/validate_run.py \
  --base \
  --eval-id full200_step2000_20260515 \
  --dataset generalist \
  --sample-count 200 \
  --shard-count 10 \
  --metric-batch-size 256 \
  --evaluate-existing \
  --wait
```

For the fine-tuned side, use `--no-base` and the same `eval-id`.

Use full validation only when regeneration is intended. Full validation launches generation shards and then evaluates:

```bash
uv run --active --with modal python -m modal run lora-klein/validation/validate_run.py \
  --no-base \
  --checkpoint-uri s3://.../checkpoint.safetensors \
  --eval-id <eval-id> \
  --dataset generalist \
  --sample-count 200 \
  --shard-count 10 \
  --metric-batch-size 256 \
  --regenerate-images \
  --wait
```

Use `--dataset attack_on_titan` to run against the fixed Attack on Titan-only validation set.

## Training

Training rules live in `lora-klein/training/AGENTS.md`.

Current training launcher:

```text
lora-klein/training/run_modal.py
```

Current Drawtoon preset family:

```text
lora-klein/training/configs/haiku-4.5/
```

Durable model artifacts are mirrored to:

```text
s3://drawtoon/models/<job_name>/
```

## Evaluation Report

After changing validation logic or rerunning key evaluations, update:

```text
EVALUATION.md
```

Current report artifacts live under:

```text
s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/
```

## Dashboard

Dashboard-specific guidance lives in `dashboard/AGENTS.md`.

For inference backends under `dashboard/backend/`, keep Modal secrets in Modal Secrets. Do not hardcode tokens into backend scripts.

## Verification

For Python-only validation changes, run at minimum:

```bash
uv run --active --with modal python -m py_compile lora-klein/validation/validate_run.py
```

For Modal changes, prefer a cheap evaluation-only or preflight command before launching GPU generation.
