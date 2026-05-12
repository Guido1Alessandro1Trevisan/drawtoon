#!/usr/bin/env bash
set -euo pipefail

JOB="mangazero_flux2_klein9b_lamic_panel_prediction_same_page_not_target_native_pad16_haiku45_lr28e7_ga8"
CONTAINER="lineart2-b300-haiku45"
IMAGE="lineart2-training:b300"
REPO_ROOT="/home/ubuntu/lineart2"
OUTPUT_ROOT="/tmp/lineart2_training_output"
DATASET_CACHE="/tmp/lineart2_datasets_cache"
HF_CACHE="/home/ubuntu/.cache/huggingface"
CONFIG_IN="/root/training/configs/haiku-4.5/lamic_panel_prediction_same_page_not_target_native_pad16_lr5e6_ga8.yaml"
CONFIG_OUT="/root/ai-toolkit/config/ec2_${JOB}.yaml"
MANIFEST_PATH="${MANIFEST_PATH:-}"
MODEL_PREFIX="${MODEL_PREFIX:-s3://drawtoon/models/${JOB}}"

if [ -z "${MANIFEST_PATH}" ]; then
  echo "This EC2 helper expects MANIFEST_PATH to point at a local Drawtoon training cache manifest. Use the Modal launcher for the standard distributed cache build." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${DATASET_CACHE}" "${HF_CACHE}"
sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

sudo docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env-file "${REPO_ROOT}/ec2_training.env" \
  -e S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD=1 \
  -e S3_CHECKPOINT_KEEP_LATEST_LOCAL=1 \
  -e S3_CHECKPOINT_HYDRATE_ON_START=1 \
  -e S3_VALIDATION_UPLOAD_ROOT="${MODEL_PREFIX}/validate" \
  -e EC2_JOB="${JOB}" \
  -e EC2_MODEL_PREFIX="${MODEL_PREFIX}" \
  -e EC2_CONFIG_IN="${CONFIG_IN}" \
  -e EC2_CONFIG_OUT="${CONFIG_OUT}" \
  -e EC2_MANIFEST_PATH="${MANIFEST_PATH}" \
  -e EC2_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  -v "${REPO_ROOT}/lora-klein/ai-toolkit:/root/ai-toolkit" \
  -v "${REPO_ROOT}/lora-klein/training:/root/training" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${DATASET_CACHE}:/root/training/datasets_cache" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
JOB="${EC2_JOB:?}"
MODEL_PREFIX="${EC2_MODEL_PREFIX:?}"
CONFIG_IN="${EC2_CONFIG_IN:?}"
CONFIG_OUT="${EC2_CONFIG_OUT:?}"
MANIFEST_PATH="${EC2_MANIFEST_PATH:?}"
OUTPUT_ROOT="${EC2_OUTPUT_ROOT:?}"

mkdir -p "${OUTPUT_ROOT}" /root/ai-toolkit/config /root/training/datasets_cache

echo "[ec2] preparing ai-toolkit config at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python /root/training/prepare_ec2_ai_toolkit_config.py \
  --config "${CONFIG_IN}" \
  --output "${CONFIG_OUT}" \
  --manifest "${MANIFEST_PATH}" \
  --model-id "${JOB}" \
  --target-epochs 4 \
  --world-size 8 \
  --validation-samples 16

echo "[ec2] hydrating any existing S3 checkpoints at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
aws s3 sync "${MODEL_PREFIX}/checkpoints" "${OUTPUT_ROOT}/${JOB}" --only-show-errors --no-progress || true

echo "[ec2] starting checkpoint mirror loop at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
(
  while true; do
    if [ -d "${OUTPUT_ROOT}/${JOB}" ]; then
      aws s3 sync "${OUTPUT_ROOT}/${JOB}" "${MODEL_PREFIX}/checkpoints" --only-show-errors --no-progress || true
    fi
    sleep 60
  done
) &
SYNC_PID=$!

cleanup() {
  kill "${SYNC_PID}" >/dev/null 2>&1 || true
  wait "${SYNC_PID}" >/dev/null 2>&1 || true
  if [ -d "${OUTPUT_ROOT}/${JOB}" ]; then
    aws s3 sync "${OUTPUT_ROOT}/${JOB}" "${MODEL_PREFIX}/checkpoints" --only-show-errors --no-progress || true
  fi
  if [ -d "${OUTPUT_ROOT}/validate/${JOB}" ]; then
    aws s3 sync "${OUTPUT_ROOT}/validate/${JOB}" "${MODEL_PREFIX}/validate" --only-show-errors --no-progress || true
  fi
}
trap cleanup EXIT

cd /root/ai-toolkit
echo "[ec2] launching torchrun at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
torchrun --standalone --nnodes=1 --nproc-per-node=8 /root/ai-toolkit/run.py "${CONFIG_OUT}"
'

if [ "${FOLLOW_LOGS:-1}" = "1" ]; then
  sudo docker logs -f --tail=120 "${CONTAINER}"
else
  sudo docker logs --tail=40 "${CONTAINER}" || true
fi
