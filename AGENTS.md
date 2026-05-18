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
