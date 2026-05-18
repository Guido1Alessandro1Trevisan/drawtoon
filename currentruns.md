# Current Runs

## B300 MangaZero Text-Removed Full Fine-Tune

Snapshot: 2026-05-17 12:14 CEST (+0200)

- Instance: `i-0fb5269c087508e18`
- Docker container: `drawtoon-b300-full-finetune`
- Job: `drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch`
- Dataset: all MangaZero titles from `s3://drawtoon/datasets/pages/text_removed/qwen2511_master_prompt_mangazero_v1/`
- Captions: `haiku45_mangazero_page_panel_v1`
- Training setup: full fine-tune, no LoRA/network block, DDP `8`, per-rank batch `1`, gradient accumulation `8`, LR `2.8e-6`, one epoch
- Validation generation: disabled during training (`--validation-samples 0`, `sample_every: 0`)
- Checkpoint policy: runner computes the DDP8 one-epoch step count, then sets `save_every = ceil(total_steps / 2)` for a half-epoch checkpoint; final save still happens at completion
- Local log: `/mnt/local/drawtoon-b300/full-finetune-text-removed-ddp8/train.log`
- Local status: `/mnt/local/drawtoon-b300/full-finetune-text-removed-ddp8/status.json`
- S3 model prefix: `s3://drawtoon/models/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch/`
- S3 run logs: `s3://drawtoon/ec2/b300-full-finetune/full-finetune-textremoved-ddp8-halfckpt-20260517T013850Z/`
- Latest observed status: completed cleanly at 2026-05-17 12:13 CEST. `status.json` is `completed`; the monitor reported `docker_state=exited` only because the training container exited after finishing. The half-epoch checkpoint at step `1943` and the final full checkpoint are both synced to S3:
  - Half checkpoint: `s3://drawtoon/models/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch/checkpoints/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch_000001943.safetensors`
  - Final checkpoint: `s3://drawtoon/models/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch/checkpoints/drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch.safetensors`
- Validation status: completed on the fixed Attack on Titan 200-sample set with eval ID `aot_mangazero_text_removed_full_b300_1epoch_20260517`. Modal generation was interrupted by one worker preemption after all 200 samples had uploaded, so the final scoring pass used `--evaluate-existing` against the completed S3 samples.
- Validation S3 prefix: `s3://drawtoon/validation/flux2_klein_panel_eval/aot_mangazero_text_removed_full_b300_1epoch_20260517/finetuned/`
- Validation result:
  - CMMD: `0.4303`
  - SigLIP2-T: `0.1574`
  - DINOv3-I: `0.8663`
  - DINOv3-C: `0.7388`
  - Bubble overall: `90.2`
  - Speech: `94.7`
  - Shout: `41.2`
  - Narration: `50.0`

## B300 AOT Text-Color Bubble Ablation

Snapshot: 2026-05-16 23:28:58 CEST (+0200)

- Instance: `i-0fb5269c087508e18`
- GPU: `1`
- Job: `drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1_textcolor_bubbles`
- Dataset: Attack on Titan only
- Training setup: LoRA rank `64`, LR `5e-5`, global batch `1`, `3500` steps, no in-training validation
- Control ablation: speech bubbles are solid blue rectangles, narration boxes are solid orange rectangles, shout bubbles are solid violet rectangles; character colors are unchanged
- Local log: `/mnt/local/drawtoon-aot-textcolor/train.log`
- S3 model prefix: `s3://drawtoon/models/drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1_textcolor_bubbles/`
- Validation eval ID: `aot_textcolor_bubbles_r64_b300_gb1_lr5e5_20260516`
- Comparison baseline: `aot_r64_b300_gb1_lr5e5_20260516`

Status: completed and validated. The existing normal AOT r64 adapter has been copied to `s3://drawtoon/models/old/drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1/final/peft_adapter/`.

Validation result on fixed Attack on Titan 200-sample set:

| Model | CMMD ↓ | SigLIP2-T ↑ | DINOv3-I ↑ | DINOv3-C ↑ | Bubble ↑ | Speech ↑ | Shout ↑ | Narration ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal AOT r64 gb1 lr5e-5 | 0.3676 | 0.1575 | 0.8595 | 0.7398 | 88.0 | 93.7 | 23.5 | 50.0 |
| Text-color bubble AOT r64 gb1 lr5e-5 | 0.4070 | 0.1607 | 0.8507 | 0.7144 | 88.9 | 93.7 | 41.2 | 0.0 |
| Charbox + 3px ref-border AOT r64 gb1 lr5e-5 | 0.3781 | 0.1589 | 0.8561 | 0.7392 | 86.7 | 93.7 | 11.8 | 0.0 |

Conclusion: text-color bubble rectangles improved `Shout Bubble` preservation from `4/17` to `7/17` and slightly improved SigLIP2-T, but it worsened CMMD, DINOv3-I, and especially DINOv3-C. The charbox + 3px ref-border variant recovered most of the core-image loss from text-color bubbles, but it still did not beat the normal AOT r64 baseline: CMMD, DINOv3-I, DINOv3-C, overall bubble, and shout bubble scores are all lower than normal r64. Normal AOT r64 remains the stronger default for character/layout fidelity.

Next control candidate started at 2026-05-16 23:52 CEST: text/bubble regions stay filled colored rectangles, character regions are colored outline boxes with per-character thickness, and each character reference crop gets a `3px` border in the same color as its target layout box. Training manifest rows, ai-toolkit training loader, dynamic validation, and fixed validation manifest preparation all apply the same bordered character refs. Example artifact: `artifacts/control_encoding_example/control_sheet.png`. Code was synced to the B300 instance and py_compile passed locally and remotely.

- Job: `drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1_charbox_refborder`
- Instance: `i-0fb5269c087508e18`
- GPU: `0`
- Dataset: Attack on Titan only
- Training setup: LoRA rank `64`, LR `5e-5`, global batch `1`, `3500` steps, no in-training validation
- Local log: `/mnt/local/drawtoon-aot-charbox-refborder/train.log`
- S3 model prefix: `s3://drawtoon/models/drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1_charbox_refborder/`
- Latest observed status: completed at 2026-05-17 00:37 CEST. Raw checkpoint and PEFT adapter synced to S3. Modal validation completed with eval ID `aot_charbox_refborder_r64_b300_gb1_lr5e5_20260517` against the fixed `attack_on_titan` set.

## Planned Text-Removed Per-Title LoRA Retrain

Snapshot: 2026-05-16 23:28:58 CEST (+0200)

- Remove-text run: `qwen2511_master_prompt_mangazero_v1`
- Text-removed pages prefix: `s3://drawtoon/datasets/pages/text_removed/qwen2511_master_prompt_mangazero_v1/`
- Step Functions map run: `arn:aws:states:us-east-1:274213480586:mapRun:RemoveTextPagesStateMachine-0yuiVvNPhoLQ/RemoveTextPages:8d1943cb-f8c1-4c1c-ad26-99347996f459`
- Latest observed remove-text progress: `21989 / 62008` succeeded, `39` running, `0` failed
- Current text-removal area: `fairy-tail_mangazero`, around `581 / 1808` pages by manifest position
- Training plan after AOT bubble-control ablation is evaluated: one LoRA per title, one B300 GPU per LoRA, rank `64`, LR `5e-5`, batch `1`, gradient accumulation `1`, `3500` steps, no in-training validation
- New model suffix for retrained adapters: `_text_removed`
- Archive destination for previous non-text-removed adapters: `s3://drawtoon/models/old/<legacy_job_name>/`

Status: prepared, not launched. The queue script now supports `--pages-prefix`, `--job-suffix`, and `--archive-legacy-adapters-to-old`; launch only after the current AOT bubble-control ablation decides which control encoding to use.

## AWS Manga Caption Structured Prompt Run

Snapshot: 2026-05-16 17:20 CEST (+0200)

Purpose: generate `structured_caption` outputs for all filtered MangaZero and Manga109 chapters using the distributed manga caption workflow.

Current clean structured-only run:

- Caption run: `haiku45_mangazero_manga109_prompt2_json_structured_only_v1`
- State machine: `arn:aws:states:us-east-1:274213480586:stateMachine:MangaCaptionStateMachine-PdgaIMTPfCtr`
- Execution: `arn:aws:states:us-east-1:274213480586:execution:MangaCaptionStateMachine-PdgaIMTPfCtr:haiku45-mangazero-manga109-prompt2-json-structured-only-20260516151945`
- Map run: `arn:aws:states:us-east-1:274213480586:mapRun:MangaCaptionStateMachine-PdgaIMTPfCtr/CaptionPages:c8a47c0a-e941-4192-9814-9f700bf93dbe`
- Source pages: `s3://drawtoon/datasets/pages/filtered/`
- Annotations: `s3://drawtoon/datasets/annotations/magi_v3/`
- Output: `s3://drawtoon/captions/haiku45_mangazero_manga109_prompt2_json_structured_only_v1/`
- Include regex: `_(mangazero|manga109)$`
- Max concurrency: `500`
- Tolerated failures: `0`
- Model: `global.anthropic.claude-haiku-4-5-20251001-v1:0`
- Mode: `structured_only=true`; the worker skips the flat prompt and writes only structured caption outputs.
- Latest observed progress: original execution stopped after `4931 / 81431` succeeded because 17 structured style-prefix validation mismatches exceeded the zero-failure threshold.

Retry after style-prefix canonicalization:

- Execution: `arn:aws:states:us-east-1:274213480586:execution:MangaCaptionStateMachine-PdgaIMTPfCtr:haiku45-mangazero-manga109-prompt2-json-structured-only-20260516152816`
- Map run: `arn:aws:states:us-east-1:274213480586:mapRun:MangaCaptionStateMachine-PdgaIMTPfCtr/CaptionPages:70602c54-9d27-44f6-81a7-24bd17c3071a`
- Manifest count: `76174` remaining pages; `overwrite=false` skipped already written JSON outputs.
- Latest observed retry progress: `14150 / 76174` succeeded, `497` running, `0` failed at 2026-05-16 17:40 CEST.
- S3 output count: `19507` JSON objects under the caption run prefix.

Resume with tolerated page failures:

- Execution: `arn:aws:states:us-east-1:274213480586:execution:MangaCaptionStateMachine-PdgaIMTPfCtr:haiku45-mangazero-manga109-prompt2-json-structured-only-20260516162546`
- Map run: `arn:aws:states:us-east-1:274213480586:mapRun:MangaCaptionStateMachine-PdgaIMTPfCtr/CaptionPages:02311398-5afc-4205-8eaa-e904182ab23d`
- Tolerated failures: `1000`
- Manifest count: `58426` remaining pages; `overwrite=false` skipped already written JSON outputs.
- Status: stopped by user request at 2026-05-16 18:31 CEST.
- Final observed progress: `5708 / 58426` succeeded, `0` running, `1` failed, `233` aborted after stop.
- S3 output count at stop: `28946` JSON objects under the caption run prefix.
- Guardrail after stop: Map Run `maxConcurrency` was set to `0` while AWS finalizes the aborted parent, so it cannot dispatch more child executions.

Previous stopped attempt:

- Caption run: `haiku45_mangazero_manga109_prompt2_json_structured_v1`
- Execution: `arn:aws:states:us-east-1:274213480586:execution:MangaCaptionStateMachine-PdgaIMTPfCtr:haiku45-mangazero-manga109-prompt2-json-structured-v1-20260516151309`
- Status: stopped after the old worker still ran the flat prompt path before structured output and hit prompt1 validation failures.

## AWS B300 Capacity Block Training

Snapshot: 2026-05-16 14:39 CEST (+0200)

This run is intentionally not using Modal. It is using the purchased AWS EC2 Capacity Block:

- Capacity reservation: `cr-08e55d1633fc6a797`
- Instance: `i-0fb5269c087508e18`
- Instance type: `p6-b300.48xlarge`
- Market: `capacity-block`
- AZ: `us-east-1a`
- Run ID: `b300-r64-aot-docker-20260516T122934Z`
- Host AMI: `ami-0b23f7b6d63e20932` (`Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04) 20260427`)
- Docker container: `drawtoon-b300-aot`
- Docker image: `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- Local repo copy: `/home/ubuntu/drawtoon`
- Work root: `/mnt/local/drawtoon-b300`
- Bootstrap logs: `s3://drawtoon/ec2/b300-r64-gb8/b300-r64-aot-docker-20260516T122934Z/logs/`

Status: running. The B300 instance launched from the DLAMI, SSH is available, Docker is active, and the container sees 8 `NVIDIA B300 SXM6 AC` GPUs. The Docker run is intentionally configured with `AUTO_SHUTDOWN_ON_EXIT=0`, so if training fails the instance remains up for a fixed restart.

Previous failed attempt:

- Instance: `i-0c4c53460752723bc`
- Run ID: `b300-r64-gb8-20260516T103214Z`
- Failure: Hugging Face token could not download `black-forest-labs/FLUX.2-klein-base-9B/model_index.json`.
- Resolution: updated the AWS Secrets Manager secret `lineart2-hf-token` and verified the exact gated `model_index.json` download locally before relaunching.
- Instance: `i-0227d8b143a6a305b`
- Run ID: `b300-r64-gb8-20260516T111150Z`
- Failure: EC2 cache finalization tried to read the Attack-on-Titan holdout list from the Modal path `/root/training/configs/...`.
- Resolution: patched `lora-klein/training/utils.py` so `build-ec2-cache` resolves Modal-style config paths to the local EC2 config directory.

Failure:

```text
403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings to view this repository.
Cannot access content at: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/main/model_index.json
```

No training steps were run and no LoRA checkpoint was produced from the previous failed B300 attempt.

The current Docker run is limited to two one-epoch DDP8/global-batch-8 LoRA r64 AOT jobs sequentially on the same B300 node:

| Order | Job | Dataset | Purpose | Expected one-epoch steps |
|---:|---|---|---|---:|
| 1 | `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r64_lr5e5_ddp8_gb8` | Attack on Titan only | Isolate global batch size 8 vs the existing single-GPU/global-batch-1 r64 baseline | about `ceil(3901 / 8) = 488` |
| 2 | `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r64_lr1e4_ddp8_gb8` | Attack on Titan only | Test LR scaling at global batch 8 | about `ceil(3901 / 8) = 488` |

The three-series job is deliberately not part of this Docker job list. Launch it only after the two AOT batch-size/LR ablations finish and the user confirms the next step.

All jobs use rank 64, per-rank batch size `1`, gradient accumulation `1`, world size `8`, and `target_epochs=1`. Jobs 1 and 2 use LR `5e-5`; job 3 uses LR `1e-4`. The EC2 config preparation prints `row_count`, `steps_per_epoch`, and `total_steps` for each run before training starts.

After these finish, compare against the existing `aot_lora_r64_20260515_170929` baseline on the fixed Attack on Titan 200-sample validation set.

Planned follow-up after the current B300 sequence completes:

| Job | Dataset | Purpose | Main change |
|---|---|---|---|
| `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r64_lr1e4_ddp8_gb8` | Attack on Titan only | Test LR scaling for global batch 8 | LR `1e-4` instead of `5e-5`, same rank 64 and global batch 8 |

## Latest Validation Report

Snapshot: 2026-05-15 20:53:56 CEST (+0200)

Dataset: fixed Attack on Titan validation set, 200 generated samples, 50 steps, guidance `4.0`, 10 H100 generation shards. Metrics: CMMD lower is better; SigLIP2-T, DINOv3-I, DINOv3-C, and bubble scores higher are better.

The newest combined run loaded the full fine-tune first, then applied the Attack-on-Titan r32 LoRA overlay at scale `1.0`.

| Model | Eval ID | CMMD ↓ | SigLIP2-T ↑ | DINOv3-I ↑ | DINOv3-C ↑ | Bubble ↑ | Speech ↑ | Shout ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | `aot_base_full_wait_20260515_165542` | 0.9623 | 0.1669 | 0.7583 | 0.6405 | 45.3 | 47.6 | 23.5 |
| Full fine-tune | `aot_base_full_wait_20260515_165542` | 0.5360 | 0.1581 | 0.8575 | 0.7401 | 84.0 | 87.4 | 47.1 |
| LoRA r32 | `aot_lora_r32_20260515_170929` | 0.3545 | 0.1596 | 0.8550 | 0.7369 | 85.3 | 90.8 | 29.4 |
| LoRA r64 | `aot_lora_r64_20260515_170929` | 0.3885 | 0.1586 | 0.8666 | 0.7653 | 89.3 | 94.7 | 35.3 |
| LoRA r128 | `aot_lora_r128_20260515_170929` | 0.3988 | 0.1596 | 0.8626 | 0.7459 | 86.2 | 92.7 | 17.6 |
| Full fine-tune + LoRA r32 | `aot_full_plus_lora_r32_20260515_193222` | 0.4658 | 0.1565 | 0.8603 | 0.7494 | 86.2 | 91.3 | 35.3 |

Load verification for `aot_full_plus_lora_r32_20260515_193222`:

- Full fine-tune checkpoint format: `full_finetune_blocks`
- Full fine-tune loaded block tensors: `192`
- Overlay checkpoint format: `ai_toolkit_lora`
- Overlay rank: `32`
- Overlay loaded LoRA modules: `112`
- Overlay missing targets: `0`
- Overlay shape skips: `0`
- Overlay scale: `1.0`

Conclusion: full fine-tune + r32 LoRA improves over the full fine-tune alone, but it does not beat the standalone r64 LoRA overall. The standalone `LoRA r64` remains the strongest model on this Attack on Titan validation set because it wins DINOv3-I, DINOv3-C, overall bubble adherence, and speech bubble adherence.

Narration bubble scoring is not reliable for ranking here because only two expected narration regions were present. Bubble scoring used 124 samples with expected bubbles and 225 expected bubble regions.

Snapshot: 2026-05-15 16:50:11 CEST (+0200)

Source: `modal app list`, `modal container list`, and short bounded reads from `modal app logs`. I did not stop, restart, or modify any Modal run.

## Active Training Apps

| App ID | Created | Tasks | Job | Dataset / purpose | Main settings | Latest observed progress |
|---|---:|---:|---|---|---|---|
| `ap-YFKnpoUX5rXj0NxH3KysHe` | 2026-05-15 14:26 CEST | 2 | `mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga8` | General MangaZero panel prediction with same-page character refs and Haiku 4.5 captions | FLUX.2 Klein 9B, LoRA rank 32, LR `5e-5`, gradient accumulation 8 | About `1137/7772` steps, `15%`; effective batch log `1+1+1+1+1+1+1+1=8`; recent step times mostly `5-9s` with occasional longer steps |
| `ap-f3ZTKaoK24dmhRqRBan4HJ` | 2026-05-15 15:58 CEST | 2 | `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r32_lr5e5_ga1_h200` | Attack-on-Titan-only panel prediction LoRA, rank 32 | FLUX.2 Klein 9B, LoRA rank 32, LR `5e-5`, gradient accumulation 1, H200 target | About `2425/3901` steps, `62%`; batch `1`; recent training steps often around `0.4-1.5s`; validation image generation was also active in the latest log window |
| `ap-cvB5hCMwV1Ab03pZa5zGZ8` | 2026-05-15 15:58 CEST | 2 | `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r64_lr5e5_ga1_h200` | Attack-on-Titan-only panel prediction LoRA, rank 64 | FLUX.2 Klein 9B, LoRA rank 64, LR `5e-5`, gradient accumulation 1, H200 target | About `2699/3901` steps, `69%`; batch `1`; recent training steps often around `0.4-1.5s`; validation image generation began in the latest log window |
| `ap-OZPkbodw0lCwirVkBJNep9` | 2026-05-15 15:58 CEST | 3 | `mangazero_flux2_klein9b_attack_on_titan_panel_prediction_same_page_refs_native_pad16_haiku45_lora_r128_lr5e5_ga1_h200` | Attack-on-Titan-only panel prediction LoRA, rank 128 | FLUX.2 Klein 9B, LoRA rank 128, LR `5e-5`, gradient accumulation 1, H200 target | Logs show two interleaved active progress streams for this same job: one around `2446/3901`, `63%`, and another around `1058/3901`, `27%`. Modal currently lists 3 tasks for this app, so this app likely has multiple active containers emitting logs under the same job name. |

## Active Containers

| App ID | Active containers observed |
|---|---|
| `ap-YFKnpoUX5rXj0NxH3KysHe` | `ta-01KRNSPHFEQ6G05PAFADAB1H26`, `ta-01KRNSMNEZ4SKHT2A023BZSFRH` |
| `ap-f3ZTKaoK24dmhRqRBan4HJ` | `ta-01KRNZ0A8XECJD6QF2CMGK2D1R`, `ta-01KRNYX6ZPAVXWTJ1TXWZZ9DSC` |
| `ap-cvB5hCMwV1Ab03pZa5zGZ8` | `ta-01KRNYZ98Q0BJRP9JMBD5B5DN6`, `ta-01KRNYX580FM8DCRZHP447HA4S` |
| `ap-OZPkbodw0lCwirVkBJNep9` | `ta-01KRP0HXWCHM09N7AHZ0YGPMD6`, `ta-01KRP0H1WJ7P638ZQYFG1X9HBH`, `ta-01KRNZ106MHWA1V4P2XHY81GZY` |

One previously listed container, `ta-01KRP1PHZ1S15MJVKPH2PA95RS`, had already finished by the time I attempted to inspect it.

## Stopped / Not Currently Training

The recent `drawtoon-flux2-klein-validation` Modal apps visible in the app list are stopped at this snapshot. I did not see active validation or inference-evaluation containers in the current `modal container list`; the active containers are under `flux-lora-training`.

## Notes

- The three Attack-on-Titan runs appear to be a rank sweep: `r32`, `r64`, and `r128`, all at LR `5e-5` and gradient accumulation 1.
- The general MangaZero run is a broader run with gradient accumulation 8 and a longer total step count.
- The `r128` app needs extra care when interpreting logs because Modal is interleaving multiple container streams for the same job name.
