#!/usr/bin/env bash
# Generic EC2 launcher for lora-klein FLUX.2 training jobs.
#
# Reads a single training-config YAML (path inside the repo, relative to
# `lora-klein/training/configs/`), hydrates the repo + venv + HF token, runs
# `torchrun --standalone --nproc-per-node=$WORLD_SIZE run.py <prepared_config>`
# and mirrors checkpoints + logs back to S3. Optionally auto-shuts down the
# EC2 instance on exit.
#
# Required env (with sensible defaults):
#
#   CONFIG_PATH        relative path under lora-klein/training/configs/, e.g.
#                      gemini-3-flash/panel_prediction_..._gemini3flash_..._1epoch.yaml
#   RUN_ID             defaults to <basename-of-config>-<UTC-timestamp>
#   S3_BUCKET          drawtoon
#   STAGE_PREFIX       ec2/runs                 (s3 path for logs + status)
#   REPO_TARBALL_S3    s3://.../repo.tar.gz     OR set LOCAL_REPO_ROOT instead
#   REPO_TARBALL_SHA256  optional; if set, the tarball is verified before extract
#   LOCAL_REPO_ROOT    /path/to/checked-out/repo (skip tarball download)
#   HF_SECRET_ID       lineart2-hf-token        (AWS Secrets Manager id)
#   AWS_REGION         us-east-1
#   WORK_ROOT          /mnt/local/drawtoon
#   WORLD_SIZE         8                        (B300:8 / H200:8 etc.)
#   TARGET_EPOCHS      1
#   MAX_TRAIN_STEPS    0                        (0 = no cap)
#   VALIDATION_SAMPLES 16
#   AUTO_SHUTDOWN_ON_EXIT  1                    (set to 0 to keep the box alive)
#   INSTALL_SYSTEM_DEPS    1                    (set to 0 if DLAMI already has them)
#   SYNC_INTERVAL_SEC      30                   (background S3 sync daemon poll interval; 0 disables)
#
# See README.md in this directory for a recommended user-data bootstrap.

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-}"
if [ -z "$CONFIG_PATH" ]; then
  echo "CONFIG_PATH is required (relative path under lora-klein/training/configs/)" >&2
  exit 2
fi

config_basename="$(basename "$CONFIG_PATH" .yaml)"
RUN_ID="${RUN_ID:-${config_basename}-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_BUCKET="${S3_BUCKET:-drawtoon}"
STAGE_PREFIX="${STAGE_PREFIX:-ec2/runs}"
REPO_TARBALL_S3="${REPO_TARBALL_S3:-}"
REPO_TARBALL_SHA256="${REPO_TARBALL_SHA256:-}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-}"
HF_SECRET_ID="${HF_SECRET_ID:-lineart2-hf-token}"
AWS_REGION="${AWS_REGION:-us-east-1}"
WORK_ROOT="${WORK_ROOT:-/mnt/local/drawtoon}"
WORLD_SIZE="${WORLD_SIZE:-8}"
TARGET_EPOCHS="${TARGET_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-16}"
AUTO_SHUTDOWN_ON_EXIT="${AUTO_SHUTDOWN_ON_EXIT:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-30}"
SYNC_DAEMON_PID=""

REPO_ROOT="$WORK_ROOT/repo"
VENV_DIR="$WORK_ROOT/venv"
OUTPUT_BASE="$WORK_ROOT/output"
CACHE_ROOT="$WORK_ROOT/datasets_cache"
HF_HOME_DIR="$WORK_ROOT/hf"
LOG_DIR="$WORK_ROOT/logs"
LOG_FILE="$LOG_DIR/train.log"
STATUS_FILE="$LOG_DIR/status.json"

mkdir -p "$WORK_ROOT" "$LOG_DIR" "$OUTPUT_BASE" "$CACHE_ROOT" "$HF_HOME_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

upload_logs() {
  aws s3 cp "$LOG_FILE" "s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/train.log" \
    --region "$AWS_REGION" --only-show-errors --no-progress || true
}

# Background daemon that mirrors $save_root -> S3 every $SYNC_INTERVAL_SEC.
# Started right before torchrun, stopped from finish(). Ensures checkpoints
# land on S3 within ~30s of being written, so the box dying never costs more
# than that.
start_output_sync_daemon() {
  if [ "$SYNC_INTERVAL_SEC" = "0" ]; then
    echo "[$(date -Is)] S3 sync daemon disabled (SYNC_INTERVAL_SEC=0)"
    return
  fi
  local save_root="$1"
  local s3_prefix="$2"
  mkdir -p "$save_root"
  (
    while true; do
      aws s3 sync "$save_root" "s3://$S3_BUCKET/$s3_prefix" \
        --region "$AWS_REGION" \
        --exclude "*.tmp" --exclude "*.partial" --exclude "*.lock" \
        --only-show-errors --no-progress \
        >>"$LOG_DIR/sync_daemon.log" 2>&1 || true
      sleep "$SYNC_INTERVAL_SEC"
    done
  ) &
  SYNC_DAEMON_PID=$!
  echo "[$(date -Is)] Started S3 sync daemon pid=$SYNC_DAEMON_PID interval=${SYNC_INTERVAL_SEC}s target=s3://$S3_BUCKET/$s3_prefix"
}

stop_output_sync_daemon() {
  if [ -n "$SYNC_DAEMON_PID" ] && kill -0 "$SYNC_DAEMON_PID" 2>/dev/null; then
    kill "$SYNC_DAEMON_PID" 2>/dev/null || true
    wait "$SYNC_DAEMON_PID" 2>/dev/null || true
    echo "[$(date -Is)] Stopped S3 sync daemon pid=$SYNC_DAEMON_PID"
  fi
}

finish() {
  local exit_code="$?"
  stop_output_sync_daemon
  python - "$STATUS_FILE" "$exit_code" <<'PY' || true
import json, sys, time
path, exit_code = sys.argv[1], int(sys.argv[2])
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "exit_code": exit_code,
            "ok": exit_code == 0,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
  upload_logs
  aws s3 cp "$STATUS_FILE" "s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/status.json" \
    --region "$AWS_REGION" --only-show-errors --no-progress || true
  if [ "$AUTO_SHUTDOWN_ON_EXIT" = "1" ]; then
    shutdown -h now || true
  fi
  exit "$exit_code"
}
trap finish EXIT

echo "[$(date -Is)] EC2 training bootstrap started run_id=$RUN_ID"
echo "[$(date -Is)] config=$CONFIG_PATH work_root=$WORK_ROOT world_size=$WORLD_SIZE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

# NVIDIA fabric-manager — REQUIRED on Blackwell SXM (B100/B200/B300) for CUDA to init.
# DLAMI ships the package but doesn't enable the service. Skip silently inside a
# Docker container (no systemd / no privilege to start host services).
if command -v systemctl >/dev/null 2>&1 && [ ! -f /.dockerenv ]; then
  if systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
    if ! systemctl is-active nvidia-fabricmanager >/dev/null 2>&1; then
      echo "[$(date -Is)] enabling + starting nvidia-fabricmanager (required on Blackwell SXM)"
      systemctl enable --now nvidia-fabricmanager || echo "[WARN] could not start fabric-manager"
      sleep 5
    fi
    echo "[$(date -Is)] fabric-manager: $(systemctl is-active nvidia-fabricmanager 2>&1)"
  fi
fi

if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git rsync python3-venv python3-pip libgl1 libglib2.0-0 nvidia-fabricmanager-580 || \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git rsync python3-venv python3-pip libgl1 libglib2.0-0
fi

if [ -n "$LOCAL_REPO_ROOT" ]; then
  REPO_ROOT="$LOCAL_REPO_ROOT"
  echo "[$(date -Is)] Using LOCAL_REPO_ROOT=$REPO_ROOT"
else
  if [ -z "$REPO_TARBALL_S3" ]; then
    echo "REPO_TARBALL_S3 is required when LOCAL_REPO_ROOT is not set" >&2
    exit 2
  fi
  echo "[$(date -Is)] Downloading repo tarball from $REPO_TARBALL_S3"
  aws s3 cp "$REPO_TARBALL_S3" "$WORK_ROOT/repo.tar.gz" \
    --region "$AWS_REGION" --only-show-errors --no-progress

  if [ -n "$REPO_TARBALL_SHA256" ]; then
    echo "[$(date -Is)] Verifying tarball sha256"
    actual_sha="$(sha256sum "$WORK_ROOT/repo.tar.gz" | awk '{print $1}')"
    if [ "$actual_sha" != "$REPO_TARBALL_SHA256" ]; then
      echo "Tarball checksum mismatch (expected=$REPO_TARBALL_SHA256 actual=$actual_sha)" >&2
      exit 3
    fi
  else
    echo "[$(date -Is)] WARNING: REPO_TARBALL_SHA256 not set — tarball not verified" >&2
  fi

  # Belt-and-braces: only wipe inside WORK_ROOT, never the root filesystem.
  if [[ -n "$WORK_ROOT" && "$REPO_ROOT" == "$WORK_ROOT/repo" ]]; then
    rm -rf "$REPO_ROOT"
  fi
  mkdir -p "$REPO_ROOT"
  tar -xzf "$WORK_ROOT/repo.tar.gz" -C "$REPO_ROOT"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
if [ ! -f "$VENV_DIR/.drawtoon_deps_installed" ]; then
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
  python -m pip install --no-cache-dir boto3 awscli pyyaml
  python -m pip install --no-cache-dir -r "$REPO_ROOT/lora-klein/ai-toolkit/requirements.txt"
  touch "$VENV_DIR/.drawtoon_deps_installed"
fi

echo "[$(date -Is)] Resolving HF token from Secrets Manager id=$HF_SECRET_ID"
HF_SECRET_VALUE="$(aws secretsmanager get-secret-value \
  --secret-id "$HF_SECRET_ID" --region "$AWS_REGION" --query SecretString --output text)"
HF_TOKEN="$(python - "$HF_SECRET_VALUE" <<'PY'
import json, sys
value = sys.argv[1]
try:
    parsed = json.loads(value)
    print(parsed.get("HF_TOKEN") or parsed.get("hf_token") or value)
except Exception:
    print(value)
PY
)"
export HF_TOKEN
export HF_HOME="$HF_HOME_DIR"

parse_final_json() {
  python - "$1" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read().rstrip()
depth, start = 0, None
for i, ch in enumerate(text):
    if ch == "{":
        if depth == 0:
            start = i
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0 and start is not None:
            json.loads(text[start:i + 1])
            print(text[start:i + 1])
            break
PY
}

label="${config_basename}"
config_path="$REPO_ROOT/lora-klein/training/configs/$CONFIG_PATH"
cache_log="$LOG_DIR/${label}_cache.log"
ec2_summary="$LOG_DIR/${label}_ec2_config.json"
prepared_config="$WORK_ROOT/configs/${label}_ai_toolkit.yaml"
output_root="$OUTPUT_BASE/$label"
mkdir -p "$WORK_ROOT/configs" "$output_root"

if [ ! -f "$config_path" ]; then
  echo "Config not found: $config_path" >&2
  exit 2
fi

echo "[$(date -Is)] Building Drawtoon panel cache for $label"
python "$REPO_ROOT/lora-klein/training/utils.py" build-ec2-cache \
  --config "$config_path" \
  --cache-root "$CACHE_ROOT" \
  --workers "${CACHE_WORKERS:-192}" \
  --overwrite | tee "$cache_log"

cache_json="$(parse_final_json "$cache_log")"
manifest_path="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_path"])' "$cache_json")"
resolved_config="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["resolved_config_path"])' "$cache_json")"

echo "[$(date -Is)] Preparing ai-toolkit EC2 config"
max_train_args=()
if [ "$MAX_TRAIN_STEPS" != "0" ]; then
  max_train_args=(--max-train-steps "$MAX_TRAIN_STEPS")
fi

python "$REPO_ROOT/lora-klein/training/utils.py" prepare-ec2-config \
  --config "$resolved_config" \
  --output "$prepared_config" \
  --manifest "$manifest_path" \
  --target-epochs "$TARGET_EPOCHS" \
  --world-size "$WORLD_SIZE" \
  --output-root "$output_root" \
  --validation-samples "$VALIDATION_SAMPLES" \
  "${max_train_args[@]}" | tee "$ec2_summary"

job_name="$(python - "$prepared_config" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["config"]["process"][0]["name"])
PY
)"
model_prefix="models/$job_name"
save_root="$output_root/$job_name"
validate_root="$output_root/validate/$job_name"

echo "[$(date -Is)] Starting torchrun for $job_name (world_size=$WORLD_SIZE)"
echo "[$(date -Is)] S3 model prefix: s3://$S3_BUCKET/$model_prefix"
export LINEART2_TRAINING_OUTPUT_ROOT="$output_root"
export S3_VALIDATION_UPLOAD_ROOT="s3://$S3_BUCKET/$model_prefix/validate"
start_output_sync_daemon "$save_root" "$model_prefix"
cd "$REPO_ROOT/lora-klein/ai-toolkit"
torchrun --standalone --nnodes=1 --nproc-per-node="$WORLD_SIZE" run.py "$prepared_config"
stop_output_sync_daemon  # final flush handled by the post-torchrun sync below

echo "[$(date -Is)] Converting LoRA checkpoint to PEFT adapter (if applicable) for $job_name"
export REPO_ROOT CACHE_ROOT
python - "$job_name" "$prepared_config" <<'PY'
import os, sys, yaml
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "lora-klein", "training"))
from utils import _load_training_module_for_ec2
mod = _load_training_module_for_ec2(os.environ["CACHE_ROOT"])
job_name = sys.argv[1]
config_path = sys.argv[2]
cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
process = cfg["config"]["process"][0]
network = process.get("network") or {}
model = process.get("model") or {}
if not network:
    print("No LoRA network config; skipping PEFT export")
    raise SystemExit(0)
raw_lora_path = mod.find_raw_lora_path(job_name)
adapter_dir = raw_lora_path.parent / "peft_adapter"
mod.build_peft_adapter(
    source_path=raw_lora_path,
    output_dir=adapter_dir,
    base_model=model.get("name_or_path", "black-forest-labs/FLUX.2-klein-base-9B"),
    rank=network.get("linear"),
    alpha=network.get("linear_alpha"),
)
mod.validate_peft_adapter_dir(adapter_dir)
print(f"PEFT adapter ready: {adapter_dir}")
fal_paths = mod.ensure_fal_lora_siblings(raw_lora_path.parent, job_name)
for fal_path in fal_paths:
    print(f"fal-compatible LoRA ready: {fal_path}")
PY

echo "[$(date -Is)] Syncing artifacts for $job_name"
if [ -f "$save_root/config.yaml" ]; then
  aws s3 cp "$save_root/config.yaml" "s3://$S3_BUCKET/$model_prefix/config.yaml" \
    --region "$AWS_REGION" --only-show-errors --no-progress
fi
if [ -f "$save_root/optimizer.pt" ]; then
  aws s3 cp "$save_root/optimizer.pt" "s3://$S3_BUCKET/$model_prefix/optimizer.pt" \
    --region "$AWS_REGION" --only-show-errors --no-progress
fi
if [ -d "$save_root" ]; then
  aws s3 sync "$save_root" "s3://$S3_BUCKET/$model_prefix/checkpoints" \
    --region "$AWS_REGION" --only-show-errors --no-progress
fi
if [ -d "$save_root/peft_adapter" ]; then
  aws s3 sync "$save_root/peft_adapter" "s3://$S3_BUCKET/$model_prefix/final/peft_adapter" \
    --region "$AWS_REGION" --only-show-errors --no-progress
fi
if [ -d "$validate_root" ]; then
  aws s3 sync "$validate_root" "s3://$S3_BUCKET/$model_prefix/validate" \
    --region "$AWS_REGION" --only-show-errors --no-progress
fi
aws s3 cp "$cache_log" "s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/${label}_cache.log" \
  --region "$AWS_REGION" --only-show-errors --no-progress || true
aws s3 cp "$ec2_summary" "s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/${label}_ec2_config.json" \
  --region "$AWS_REGION" --only-show-errors --no-progress || true

echo "[$(date -Is)] Completed $job_name"
