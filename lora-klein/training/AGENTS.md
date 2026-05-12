# Training Guide

## Scope

This file applies to `lora-klein/training/`.

Use it for Modal training launches, config selection, validation sync, and training-side troubleshooting.

## Active Pipeline

The standard path is:

1. Use Drawtoon canonical S3 data:
   - `datasets/pages/filtered`
   - `datasets/annotations/magi_v3`
   - `captions/<caption_run>`
2. Build a temporary Modal Volume cache with CPU workers before GPU allocation.
3. Train ai-toolkit from the generated local `manifest.jsonl`.
4. Mirror durable artifacts to `s3://drawtoon/models/<job_name>/`.

Do not reintroduce S3 `sample_records`, S3 `caption_groups`, or S3 `training_views` as active training inputs.

## Launch Rule

Default practice is:

```text
--preset <config-relative-path-without-.yaml>
```

The current Drawtoon preset is:

```text
haiku-4.5/lamic_panel_prediction_same_page_not_target_native_pad16_lr28e7_ga8
```

## Directory Layout

- `run_modal.py`: active Modal launcher.
- `configs/`: active presets.
- `utils.py`: one Python helper entrypoint for launcher support and maintenance jobs.
- `sync_validate_images.sh`: local validation-image sync helper.

## Dataset Encoding

Keep ai-toolkit training on `type: manifest`. The launcher fills `manifest_path` after it builds the temporary Drawtoon cache.

Use upstream-style dataset controls:

- `buckets: false`
- `batch_size: 1`
- `cache_latents: false`
- `cache_latents_to_disk: false`

Target panels are padded to model divisibility by the cache builder before training.

## Storage Architecture

- S3 is the canonical source for pages, annotations, captions, and model artifacts.
- Modal volume `flux-dataset-cache` stores rebuildable training cache files.
- Modal volume `flux-lora-models` is older single-GPU resume state only.
- DDP checkpoints are written to local disk first, then mirrored to S3.

## Fork Differences

- `lora-klein/` and `lora-klein/ai-toolkit/` are vendored in this repo.
- `run_modal.py` adds the local ai-toolkit fork into the Modal image.
- Manifest training consumes local cache paths generated from Drawtoon captions.
- Do not reintroduce regional-control/RAG generation hooks into the FLUX.2 model or pipeline.
