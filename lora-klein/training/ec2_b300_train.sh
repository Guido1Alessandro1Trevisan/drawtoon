#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-b300-r64-gb8-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_BUCKET="${S3_BUCKET:-drawtoon}"
STAGE_PREFIX="${STAGE_PREFIX:-ec2/b300-r64-gb8}"
REPO_TARBALL_S3="${REPO_TARBALL_S3:-}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-}"
HF_SECRET_ID="${HF_SECRET_ID:-lineart2-hf-token}"
AWS_REGION="${AWS_REGION:-us-east-1}"
WORK_ROOT="${WORK_ROOT:-/mnt/local/drawtoon-b300}"
REPO_ROOT="$WORK_ROOT/repo"
VENV_DIR="$WORK_ROOT/venv"
OUTPUT_BASE="$WORK_ROOT/output"
CACHE_ROOT="$WORK_ROOT/datasets_cache"
HF_HOME_DIR="$WORK_ROOT/hf"
LOG_DIR="$WORK_ROOT/logs"
LOG_FILE="$LOG_DIR/train.log"
STATUS_FILE="$LOG_DIR/status.json"
AUTO_SHUTDOWN_ON_EXIT="${AUTO_SHUTDOWN_ON_EXIT:-1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
B300_VALIDATION_SAMPLES="${B300_VALIDATION_SAMPLES:-16}"
B300_MAX_TRAIN_STEPS="${B300_MAX_TRAIN_STEPS:-0}"
B300_WORLD_SIZE="${B300_WORLD_SIZE:-1}"

mkdir -p "$WORK_ROOT" "$LOG_DIR" "$OUTPUT_BASE" "$CACHE_ROOT" "$HF_HOME_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

upload_logs() {
  aws s3 cp "$LOG_FILE" "s3://$S3_BUCKET/$STAGE_PREFIX/$RUN_ID/logs/train.log" \
    --region "$AWS_REGION" --only-show-errors --no-progress || true
}

finish() {
  local exit_code="$?"
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

echo "[$(date -Is)] B300 training bootstrap started run_id=$RUN_ID"
echo "[$(date -Is)] repo=$REPO_TARBALL_S3 work_root=$WORK_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git rsync python3-venv python3-pip libgl1 libglib2.0-0
fi

if [ -n "$LOCAL_REPO_ROOT" ]; then
  REPO_ROOT="$LOCAL_REPO_ROOT"
else
  if [ -z "$REPO_TARBALL_S3" ]; then
    echo "REPO_TARBALL_S3 is required when LOCAL_REPO_ROOT is not set" >&2
    exit 2
  fi
  aws s3 cp "$REPO_TARBALL_S3" "$WORK_ROOT/repo.tar.gz" \
    --region "$AWS_REGION" --only-show-errors --no-progress
  rm -rf "$REPO_ROOT"
  mkdir -p "$REPO_ROOT"
  tar -xzf "$WORK_ROOT/repo.tar.gz" -C "$REPO_ROOT"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
if [ ! -f "$VENV_DIR/.drawtoon_b300_deps_installed" ]; then
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
  python -m pip install --no-cache-dir boto3 awscli pyyaml
  python -m pip install --no-cache-dir -r "$REPO_ROOT/lora-klein/ai-toolkit/requirements.txt"
  touch "$VENV_DIR/.drawtoon_b300_deps_installed"
fi

export HF_TOKEN
HF_SECRET_VALUE="$(aws secretsmanager get-secret-value \
  --secret-id "$HF_SECRET_ID" \
  --region "$AWS_REGION" \
  --query SecretString \
  --output text)"
HF_TOKEN="$(python - "$HF_SECRET_VALUE" <<'PY'
import json
import sys

raw = sys.argv[1].strip()
if raw.startswith("{"):
    payload = json.loads(raw)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "token"):
        value = str(payload.get(key) or "").strip()
        if value:
            print(value)
            break
    else:
        raise SystemExit("Hugging Face secret JSON did not contain HF_TOKEN/HUGGING_FACE_HUB_TOKEN/token")
else:
    print(raw)
PY
)"
unset HF_SECRET_VALUE
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NO_ALBUMENTATIONS_UPDATE=1
export DISABLE_TELEMETRY=YES
export PYTHONPATH="$REPO_ROOT/lora-klein/ai-toolkit"
export QUANTO_BYPASS_OBJECT_COPY=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export AITK_DDP_STATIC_GRAPH=1
export AITK_DDP_FIND_UNUSED_PARAMETERS=0
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD=1
export S3_CHECKPOINT_KEEP_LATEST_LOCAL=1
export S3_CHECKPOINT_HYDRATE_ON_START=1
export LINEART2_TRAINING_OUTPUT_ROOT
export REPO_ROOT
export CACHE_ROOT

python - <<'PY'
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    "black-forest-labs/FLUX.2-klein-base-9B",
    "model_index.json",
    token=os.environ["HF_TOKEN"],
)
print("HF gated model preflight succeeded")
PY

parse_final_json() {
  python - "$1" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
starts = [i for i, ch in enumerate(text) if ch == "{"]
for start in reversed(starts):
    try:
        obj = json.loads(text[start:])
    except Exception:
        continue
    print(json.dumps(obj, sort_keys=True))
    break
else:
    raise SystemExit(f"No JSON object found in {sys.argv[1]}")
PY
}

run_training_job() {
  local preset_path="$1"
  local label="$2"
  local config_path="$REPO_ROOT/lora-klein/training/configs/$preset_path"
  local cache_log="$LOG_DIR/${label}_cache.log"
  local ec2_summary="$LOG_DIR/${label}_ec2_config.json"
  local prepared_config="$WORK_ROOT/configs/${label}_ai_toolkit.yaml"
  local output_root="$OUTPUT_BASE/$label"

  mkdir -p "$WORK_ROOT/configs" "$output_root"
  echo "[$(date -Is)] Building Drawtoon cache for $label from $preset_path"
  python "$REPO_ROOT/lora-klein/training/utils.py" build-ec2-cache \
    --config "$config_path" \
    --cache-root "$CACHE_ROOT" \
    --workers 96 \
    --overwrite | tee "$cache_log"

  local cache_json
  cache_json="$(parse_final_json "$cache_log")"
  local manifest_path
  manifest_path="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_path"])' "$cache_json")"
  local resolved_config
  resolved_config="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["resolved_config_path"])' "$cache_json")"

  echo "[$(date -Is)] Preparing ai-toolkit EC2 config for $label"
  local max_train_args=()
  if [ "$B300_MAX_TRAIN_STEPS" != "0" ]; then
    max_train_args=(--max-train-steps "$B300_MAX_TRAIN_STEPS")
  fi

  python "$REPO_ROOT/lora-klein/training/utils.py" prepare-ec2-config \
    --config "$resolved_config" \
    --output "$prepared_config" \
    --manifest "$manifest_path" \
    --target-epochs 1 \
    --world-size "$B300_WORLD_SIZE" \
    --output-root "$output_root" \
    --validation-samples "$B300_VALIDATION_SAMPLES" \
    "${max_train_args[@]}" | tee "$ec2_summary"

  local job_name
  job_name="$(python - "$prepared_config" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["config"]["process"][0]["name"])
PY
)"
  local model_prefix="models/$job_name"
  local save_root="$output_root/$job_name"
  local validate_root="$output_root/validate/$job_name"

  echo "[$(date -Is)] Starting torchrun for $job_name"
  echo "[$(date -Is)] S3 model prefix: s3://$S3_BUCKET/$model_prefix"
  export LINEART2_TRAINING_OUTPUT_ROOT="$output_root"
  export S3_VALIDATION_UPLOAD_ROOT="s3://$S3_BUCKET/$model_prefix/validate"
  cd "$REPO_ROOT/lora-klein/ai-toolkit"
  torchrun --standalone --nnodes=1 --nproc-per-node="$B300_WORLD_SIZE" run.py "$prepared_config"

  echo "[$(date -Is)] Converting LoRA checkpoint to PEFT adapter for $job_name"
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
}

export REPO_ROOT CACHE_ROOT

run_named_training_job() {
  local job_key="$1"
  case "$job_key" in
    aot_5e5)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr5e5_ddp8_gb8.yaml" \
        "aot_r64_ddp8_gb8"
      ;;
    aot_5e5_gb1)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr5e5_b300_gb1.yaml" \
        "aot_r64_lr5e5_b300_gb1"
      ;;
    aot_lr1e4)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr1e4_ddp8_gb8.yaml" \
        "aot_r64_lr1e4_ddp8_gb8"
      ;;
    aot_lr1e5)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr1e5_ddp8_gb8.yaml" \
        "aot_r64_lr1e5_ddp8_gb8"
      ;;
    aot_lr1e5_gb1)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr1e5_b300_gb1.yaml" \
        "aot_r64_lr1e5_b300_gb1"
      ;;
    aot_prompt2_r64_5e5)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_prompt2_json_structured_caption_native_pad16_lora_r64_lr5e5_ddp8_gb8.yaml" \
        "aot_prompt2_json_structured_r64_lr5e5_ddp8_gb8"
      ;;
    aot_prompt2_r64_5e5_gb1)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_prompt2_json_structured_caption_native_pad16_lora_r64_lr5e5_b300_gb1.yaml" \
        "aot_prompt2_json_structured_r64_lr5e5_b300_gb1"
      ;;
    three_series_5e5)
      run_training_job \
        "haiku-4.5/panel_prediction_attack_on_titan_demon_slayer_20thcb_native_pad16_lora_r64_lr5e5_ddp8_gb8.yaml" \
        "aot_demonslayer_20thcb_r64_ddp8_gb8"
      ;;
    *)
      echo "Unknown B300_JOB_LIST entry: $job_key" >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a b300_job_keys <<< "${B300_JOB_LIST:-aot_5e5,aot_lr1e4}"
for b300_job_key in "${b300_job_keys[@]}"; do
  run_named_training_job "$b300_job_key"
done

echo "[$(date -Is)] All B300 ablations completed"
