# FLUX.2 Klein MangaZero Panel Validation

Date: 2026-05-15

Eval ID: `full200_step2000_20260515`

Validation set: fixed 200-sample generalist set under `lora-klein/validation/datasets/generalist`.

An Attack on Titan-only 200-sample fixed set is available under `lora-klein/validation/datasets/attack_on_titan` and can be selected with `--dataset attack_on_titan`.

## Latest Attack on Titan Text-Removed Full Fine-Tune

Eval ID: `aot_mangazero_text_removed_full_b300_1epoch_20260517`

Validation set: fixed 200-sample Attack on Titan set under `lora-klein/validation/datasets/attack_on_titan`.

Checkpoint:

`s3://drawtoon/models/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch/checkpoints/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch.safetensors`

Training setup: MangaZero text-removed full fine-tune, DDP `8`, per-rank batch `1`, gradient accumulation `8`, LR `2.8e-6`, one epoch, Haiku 4.5 captions, updated text-color bubble and character-ref border control encoding.

| Metric | Value |
|---|---:|
| CMMD ↓ | 0.4303 |
| SigLIP2-T ↑ | 0.1574 |
| DINOv3-I ↑ | 0.8663 |
| DINOv3-C ↑ | 0.7388 |
| Bubble overall ↑ | 90.2 |
| Speech Bubble ↑ | 94.7 |
| Shout Bubble ↑ | 41.2 |
| Narration Bubble ↑ | 50.0 |

Compared with the prior fixed-set standalone Attack on Titan `LoRA r64` result (`aot_lora_r64_20260515_170929`), this full fine-tune is worse on CMMD and DINOv3-C, essentially tied on DINOv3-I, and slightly better on overall bubble adherence. The r64 LoRA remains the better default for character/reference fidelity on this validation set.

Artifacts:

- Samples and controls: `s3://drawtoon/validation/flux2_klein_panel_eval/aot_mangazero_text_removed_full_b300_1epoch_20260517/finetuned/samples/`
- Metric summary: `s3://drawtoon/validation/flux2_klein_panel_eval/aot_mangazero_text_removed_full_b300_1epoch_20260517/finetuned/metrics/summary.json`
- Haiku bubble summary: `s3://drawtoon/validation/flux2_klein_panel_eval/aot_mangazero_text_removed_full_b300_1epoch_20260517/finetuned/metrics/haiku_bubble/summary.json`

Fine-tuned checkpoint:

`s3://drawtoon/models/mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lr28e7_ga8_1epoch/checkpoints/mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lr28e7_ga8_1epoch_000002000.safetensors`

The checkpoint load reported `192` loaded block tensors from `201` checkpoint tensors, with `0` unexpected tensors and `0` shape/key skips. This matches a partial transformer fine-tune over `transformer.double_blocks` and `transformer.single_blocks`, with the rest coming from the base model.

## Current Internal Metrics

These are the metrics currently used for internal model selection:

- `CMMD`: distribution distance between generated and target panels using normalized `openai/clip-vit-large-patch14-336` image embeddings and MMD. Lower is better.
- `SigLIP2-T`: image-caption cosine similarity using `google/siglip2-base-patch16-384`. Higher is better.
- `DINOv3-I`: full generated panel vs target panel cosine similarity using `timm/vit_base_patch16_dinov3.lvd1689m`. Higher is better.
- `DINOv3-C`: generated character crop vs reference character crop cosine similarity using the same DINOv3 model and layout boxes. Higher is better.

| Metric | Base FLUX | Fine-tuned | Delta | Better |
|---|---:|---:|---:|---|
| CMMD | 0.5652 | 0.6908 | +0.1256 | Base |
| SigLIP2-T | 0.1560 | 0.1453 | -0.0108 | Base |
| DINOv3-I | 0.7954 | 0.8712 | +0.0758 | Fine-tuned |
| DINOv3-C | 0.6913 | 0.7781 | +0.0868 | Fine-tuned |

Paired bootstrap 95% confidence intervals:

- `SigLIP2-T delta`: -0.0108, CI [-0.0138, -0.0076]
- `DINOv3-I delta`: +0.0758, CI [+0.0662, +0.0860]
- `DINOv3-C delta`: +0.0868, CI [+0.0736, +0.1011]

Interpretation: the fine-tuned checkpoint substantially improves target-panel structure and character-reference preservation under DINOv3. The base model remains better on CMMD and SigLIP2-T, so the fine-tune appears more faithful to the reference/target visual structure but slightly less aligned to the caption distribution under SigLIP2.

## Manga-Specific Bubble Evaluation

The Haiku bubble judge compares target and generated panels for whether speech, narration, and shout bubble types are respected. It judged 200 samples; 158 had expected bubbles and 42 had no expected bubbles.

| Bubble Metric | Base FLUX | Fine-tuned | Delta |
|---|---:|---:|---:|
| Overall respected | 53.39% | 76.84% | +22.67 pp |
| Speech Bubble | 61.37% | 85.56% | +23.47 pp |
| Narration Bubble | 24.24% | 36.36% | +16.30 pp |
| Shout Bubble | 25.00% | 52.27% | +33.33 pp |

Paired bootstrap 95% confidence intervals:

- `Overall delta`: +22.67 pp, CI [+15.64, +29.87]
- `Speech Bubble delta`: +23.47 pp, CI [+15.22, +31.91]
- `Narration Bubble delta`: +16.30 pp, CI [-1.09, +34.78]
- `Shout Bubble delta`: +33.33 pp, CI [+16.67, +50.00]

Interpretation: the fine-tuned checkpoint is clearly better at respecting bubble container types, especially speech and shout bubbles. Narration improves, but the confidence interval crosses zero.

## Legacy DiffSensei-Compatible Metrics

These were used for comparison with the DiffSensei paper-style metric family. They are retained for historical comparison only; current internal evaluation uses the CMMD/SigLIP2/DINOv3 suite above.

| Metric | Base FLUX | Fine-tuned | Delta | Better |
|---|---:|---:|---:|---|
| FID | 141.4092 | 103.0786 | -38.3306 | Fine-tuned |
| KID mean | 0.02287 | 0.00315 | -0.01972 | Fine-tuned |
| CLIP score | 0.3195 | 0.3103 | -0.0092 | Base |
| DINO-I | 0.5710 | 0.6982 | +0.1271 | Fine-tuned |
| DINO-C | 0.4914 | 0.6334 | +0.1420 | Fine-tuned |

Paired bootstrap 95% confidence intervals:

- `CLIP delta`: -0.0092, CI [-0.0132, -0.0052]
- `DINO-I delta`: +0.1271, CI [+0.1071, +0.1504]
- `DINO-C delta`: +0.1420, CI [+0.1205, +0.1668]

Dialog F1 was removed from the evaluation suite. It is not reported.

## DiffSensei Reference Numbers

DiffSensei reported:

| Dataset | DINO-I | DINO-C |
|---|---:|---:|
| MangaZero eval | 0.618 | 0.651 |
| Manga109 eval | 0.588 | 0.600 |

The fine-tuned checkpoint's legacy DINO-I was higher than both reported DiffSensei DINO-I values, and its legacy DINO-C was between the DiffSensei Manga109 and MangaZero values. This is not a formal apples-to-apples benchmark because the exact validation set, reference sampling, and evaluator implementation differ.

## Artifact Locations

Current internal metric summaries:

- Base: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/base/metrics/summary.json`
- Fine-tuned: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/finetuned/metrics/summary.json`
- Paired deltas: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/paired/paired_summary.json`

Haiku bubble summaries:

- Base: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/base/metrics/haiku_bubble/summary.json`
- Fine-tuned: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/finetuned/metrics/haiku_bubble/summary.json`
- Paired deltas: `s3://drawtoon/validation/flux2_klein_panel_eval/full200_step2000_20260515/paired/haiku_bubble_summary.json`

## Implementation Notes

Evaluation script: `lora-klein/validation/validate_run.py`

Relevant runtime options:

- `--evaluate-existing`: score already-generated S3 images without regenerating.
- `--regenerate-images`: explicit marker for the full generation path.
- `--dataset`: choose the fixed local validation dataset mounted into Modal. Current values are `generalist` and `attack_on_titan`.
- `--metric-batch-size`: controls GPU metric batch size. The tested default is `256`.
- `--haiku-bubble-judge`: optionally run the Haiku bubble-type evaluator.

The official `facebook/dinov3-*` Hugging Face repositories are gated for the current token, so the evaluator uses the public timm DINOv3 wrapper `timm/vit_base_patch16_dinov3.lvd1689m`.
