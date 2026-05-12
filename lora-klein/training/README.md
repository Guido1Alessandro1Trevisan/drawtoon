# Training

This directory owns Modal + ai-toolkit FLUX.2 Klein training for Drawtoon.

## Active Flow

1. Read canonical Drawtoon data from S3:
   - `datasets/pages/filtered/`
   - `datasets/annotations/magi_v3/`
   - `captions/<caption_run>/`
2. Build a temporary LAMIC manifest cache in the Modal dataset volume before GPU allocation.
3. Launch ai-toolkit training from that local manifest.
4. Mirror durable model artifacts to `s3://drawtoon/models/<job_name>/`.

The temporary cache is not dataset truth. It is rebuildable local training state:

```text
/root/training/datasets_cache/drawtoon_lamic/<fingerprint>/
  manifest.jsonl
  panels/
  refs/
  summary.json
  resolved_config.yaml
  _SUCCESS
```

## Storage

- S3 canonical inputs stay under the Drawtoon dataset layout.
- Modal volume `flux-dataset-cache` holds the temporary training cache.
- Modal volume `flux-lora-models` is older single-GPU resume state only.
- DDP checkpoints stage on local disk and are mirrored to S3 during training.
- Durable training outputs go under:

```text
s3://drawtoon/models/<job_name>/checkpoints/
s3://drawtoon/models/<job_name>/validate/
s3://drawtoon/models/<job_name>/final/
```

## Directory Layout

```text
training/
  run_modal.py        # active Modal launcher
  utils.py            # one Python helper entrypoint
  sync_validate_images.sh
  configs/           # training presets
```

## Launch

Use `--preset`, not direct generated config paths:

```bash
uv run --active --with modal --with pyyaml --with boto3 python -m modal run --detach lora-klein/training/run_modal.py \
  --preset "haiku-4.5/lamic_panel_prediction_same_page_not_target_native_pad16_lr28e7_ga8" \
  --target-epochs 4 \
  --ddp-world-size 8
```

For a small smoke cache and training parse check:

```bash
uv run --active --with modal --with pyyaml --with boto3 python -m modal run --detach lora-klein/training/run_modal.py \
  --preset "haiku-4.5/lamic_panel_prediction_same_page_not_target_native_pad16_lr28e7_ga8" \
  --drawtoon-max-pages 32 \
  --max-train-steps 2 \
  --ddp-world-size 1
```

Sync validation images:

```bash
bash lora-klein/training/sync_validate_images.sh \
  --job mangazero_flux2_klein9b_lamic_panel_prediction_same_page_not_target_native_pad16_haiku45_lr28e7_ga8
```
