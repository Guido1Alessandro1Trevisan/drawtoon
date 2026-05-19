# EC2 training launcher

One bash script, one config path, and a bootstrap recipe. Replaces the
older `ec2_b300_train.sh` plus the four `ec2_*.py` wrappers, which each
hardcoded a job and several long-deleted preset paths.

## What it does

`train.sh` runs on a fresh EC2 instance and:

1. Installs apt deps (skip with `INSTALL_SYSTEM_DEPS=0` if your AMI already has them).
2. Downloads the repo tarball from S3 (or uses `LOCAL_REPO_ROOT`), optionally verifying a SHA-256.
3. Creates a venv at `$WORK_ROOT/venv`, pins `torch==2.9.1 + torchvision==0.24.1 + torchaudio==2.9.1` on the cu128 wheel index, installs `ai-toolkit/requirements.txt`.
4. Resolves the HF token from AWS Secrets Manager.
5. Calls `lora-klein/training/utils.py build-ec2-cache` to build the panel-cache manifest.
6. Calls `lora-klein/training/utils.py prepare-ec2-config` to materialize the ai-toolkit YAML with `target_epochs / world_size / validation samples / max_train_steps` baked in.
7. Runs `torchrun --standalone --nnodes=1 --nproc-per-node=$WORLD_SIZE run.py <prepared_config>`.
8. Converts the LoRA checkpoint to a PEFT adapter (skipped if the config is a full fine-tune).
9. `aws s3 sync`s `checkpoints/`, `validate/`, `config.yaml`, `optimizer.pt` to `s3://$S3_BUCKET/models/<job_name>/`.
10. Uploads `train.log` + `status.json` to `s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/` and **shuts the box down** unless `AUTO_SHUTDOWN_ON_EXIT=0`.

## Required environment

| var | required | default | notes |
|-----|----------|---------|-------|
| `CONFIG_PATH` | **yes** | — | Path relative to `lora-klein/training/configs/`, e.g. `gemini-3-flash/panel_prediction_..._gemini3flash_..._1epoch.yaml` |
| `REPO_TARBALL_S3` | yes (or `LOCAL_REPO_ROOT`) | — | `s3://...repo.tar.gz` |
| `REPO_TARBALL_SHA256` | strongly recommended | — | If set, the tarball is verified before extract |
| `LOCAL_REPO_ROOT` | — | unset | When set, skips the tarball download (testing only) |
| `HF_SECRET_ID` | — | `lineart2-hf-token` | AWS Secrets Manager id |
| `AWS_REGION` | — | `us-east-1` | |
| `S3_BUCKET` | — | `drawtoon` | |
| `STAGE_PREFIX` | — | `ec2/runs` | Run log prefix |
| `WORK_ROOT` | — | `/mnt/local/drawtoon` | Local workspace (must have NVMe-class storage) |
| `WORLD_SIZE` | — | `8` | DDP world size (e.g. 8 for B300:8 / H200:8) |
| `TARGET_EPOCHS` | — | `1` | |
| `MAX_TRAIN_STEPS` | — | `0` | `0` = no cap |
| `VALIDATION_SAMPLES` | — | `16` | |
| `AUTO_SHUTDOWN_ON_EXIT` | — | `1` | Set `0` while debugging |
| `INSTALL_SYSTEM_DEPS` | — | `1` | Set `0` if your AMI already has libgl1 etc. |
| `RUN_ID` | — | `<config-basename>-<UTC-ts>` | Override for re-runs |

## Recommended AMI

A recent **Deep Learning AMI (Ubuntu 22.04) with CUDA 12.x** (any DLAMI
post mid-2024 ships CUDA 12 + nvidia drivers, which is what `torch==2.9.1
+ cu128` expects). Confirmed working: `ami-*` for DLAMI Base GPU CUDA 12.4
Ubuntu 22.04 — the apt-installs in the script add only the missing libs.

If you're using a vanilla Ubuntu AMI, ensure NVIDIA drivers are installed
ahead of time (the script does not install them).

## Bootstrap example (instance user-data)

Paste this into the EC2 **User data** field when launching the instance.
It downloads `train.sh` from S3 and runs it in the background so the
launch console does not block.

```bash
#!/usr/bin/env bash
set -euxo pipefail

export CONFIG_PATH="gemini-3-flash/panel_prediction_mangazero_text_removed_same_page_refs_native_pad16_gemini3flash_lr28e7_ga8_full_b300_1epoch.yaml"
export REPO_TARBALL_S3="s3://drawtoon/ec2/bootstrap/drawtoon-2026-05-19.tar.gz"
export REPO_TARBALL_SHA256="<sha256-of-the-tarball>"
export HF_SECRET_ID="lineart2-hf-token"
export AWS_REGION="us-east-1"
export WORLD_SIZE=8
export TARGET_EPOCHS=1
export AUTO_SHUTDOWN_ON_EXIT=1

mkdir -p /var/log/drawtoon
aws s3 cp s3://drawtoon/ec2/bootstrap/train.sh /usr/local/bin/drawtoon-train.sh \
  --region "$AWS_REGION"
chmod +x /usr/local/bin/drawtoon-train.sh

nohup /usr/local/bin/drawtoon-train.sh \
  >/var/log/drawtoon/bootstrap.log 2>&1 &
```

To upload `train.sh` to S3 once:

```bash
aws s3 cp lora-klein/training/ec2/train.sh \
  s3://drawtoon/ec2/bootstrap/train.sh \
  --region us-east-1
```

To bundle the repo:

```bash
git archive --format=tar.gz --output=/tmp/drawtoon.tar.gz HEAD
aws s3 cp /tmp/drawtoon.tar.gz s3://drawtoon/ec2/bootstrap/drawtoon-$(date -u +%Y-%m-%d).tar.gz \
  --region us-east-1
sha256sum /tmp/drawtoon.tar.gz   # use as REPO_TARBALL_SHA256
```

## Monitoring

While the instance is running:

```bash
# Latest live log (uploaded on every flush + on exit)
aws s3 cp s3://drawtoon/ec2/runs/$RUN_ID/logs/train.log - --region us-east-1 | tail -n 200

# Final status file (written on exit)
aws s3 cp s3://drawtoon/ec2/runs/$RUN_ID/logs/status.json -
```

Saved checkpoints land under:

```
s3://drawtoon/models/<job_name>/checkpoints/<job_name>_<step>.safetensors
s3://drawtoon/models/<job_name>/optimizer.pt
s3://drawtoon/models/<job_name>/config.yaml
s3://drawtoon/models/<job_name>/validate/...
s3://drawtoon/models/<job_name>/final/peft_adapter/  (LoRA only)
```

`<job_name>` is `config.name` (or `process[0].name`) inside the YAML.

## Local dry-run

```bash
WORLD_SIZE=1 \
LOCAL_REPO_ROOT="$PWD" \
CONFIG_PATH=gemini-3-flash/smoke_save_every_2.yaml \
AUTO_SHUTDOWN_ON_EXIT=0 \
INSTALL_SYSTEM_DEPS=0 \
WORK_ROOT=/tmp/drawtoon-dry \
lora-klein/training/ec2/train.sh
```

(Requires CUDA-capable host, otherwise torchrun will fail at startup.)

## Auto-shutdown

`AUTO_SHUTDOWN_ON_EXIT=1` (default) calls `shutdown -h now` in the EXIT
trap so an EC2 instance with `InstanceInitiatedShutdownBehavior=terminate`
self-terminates after a finished or failed run. Always set the
shutdown-behaviour explicitly when launching the instance; the script
relies on it.

## Why not the old `ec2_*.py` wrappers?

The old layout had four hardcoded launchers
(`ec2_mangazero_full_finetune.py`, `ec2_aot_textcolor_ablation.py`,
`ec2_aot_charbox_refborder_ablation.py`, `ec2_title_lora_queue.py`) and a
bash script (`ec2_b300_train.sh`) that contained a `case` statement over
named preset jobs pointing at `configs/haiku-4.5/*.yaml`. Every wrapper
and most of those preset YAMLs are no longer in the repo, so the old
launchers were silently bricked. Consolidated to one config-driven script
plus this README.
