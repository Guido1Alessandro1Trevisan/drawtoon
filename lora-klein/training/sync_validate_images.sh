#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_ROOT="$SCRIPT_DIR/validate"
FORCE_ARGS=()
JOB_NAMES=()
S3_MODELS_ROOT="${S3_MODELS_ROOT:-s3://drawtoon/models}"
AWS_REGION="${AWS_REGION:-us-east-1}"

mkdir -p "$VALIDATE_ROOT"

modal_volume_get() {
  uv run --active --with modal python -m modal volume get "$@"
}

modal_volume_ls() {
  uv run --active --with modal python -m modal volume ls "$@"
}

sync_s3_validation() {
  local job_name="$1"
  local target_dir="$2"
  local s3_root="$S3_MODELS_ROOT/$job_name/validate"

  if aws s3 ls "$s3_root/" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Syncing S3 validation folder $s3_root into $target_dir"
    aws s3 sync "$s3_root" "$target_dir" --region "$AWS_REGION" --only-show-errors --no-progress
    normalize_job_dir "$target_dir" "$job_name"
    return 0
  fi

  return 1
}

merge_dir_contents() {
  local src_dir="$1"
  local dst_dir="$2"
  [ -d "$src_dir" ] || return 0
  mkdir -p "$dst_dir"

  find "$src_dir" -mindepth 1 -maxdepth 1 | while read -r item_path; do
    local item_name dst_path
    item_name="$(basename "$item_path")"
    dst_path="$dst_dir/$item_name"

    if [ -d "$item_path" ]; then
      if [ -d "$dst_path" ]; then
        merge_dir_contents "$item_path" "$dst_path"
        rmdir "$item_path" 2>/dev/null || true
      else
        mv "$item_path" "$dst_path"
      fi
    else
      if [ -e "$dst_path" ]; then
        rm -f "$dst_path"
      fi
      mv "$item_path" "$dst_path"
    fi
  done
}

normalize_job_dir() {
  local job_dir="$1"
  local job_name="$2"
  [ -d "$job_dir" ] || return 0

  local nested_root
  for nested_root in "$job_dir/validate/$job_name" "$job_dir/$job_name"; do
    if [ -d "$nested_root" ]; then
      merge_dir_contents "$nested_root" "$job_dir"
      rm -rf "$nested_root"
    fi
  done

  find "$job_dir" -mindepth 2 -maxdepth 2 -type d -name 'step_*' | while read -r nested_step_dir; do
    local parent_dir step_name canonical_step_dir
    parent_dir="$(dirname "$nested_step_dir")"
    step_name="$(basename "$nested_step_dir")"
    canonical_step_dir="$job_dir/$step_name"

    if [[ "$parent_dir" != "$job_dir" ]]; then
      merge_dir_contents "$nested_step_dir" "$canonical_step_dir"
      rm -rf "$nested_step_dir"
      rmdir "$parent_dir" 2>/dev/null || true
    fi
  done

  if [ -d "$job_dir/validate" ]; then
    rm -rf "$job_dir/validate"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --force)
      FORCE_ARGS+=(--force)
      shift
      ;;
    --job)
      if [ $# -lt 2 ]; then
        echo "Usage: $0 [--force] --job JOB_NAME [--job JOB_NAME ...]"
        exit 1
      fi
      JOB_NAMES+=("$2")
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--force] --job JOB_NAME [--job JOB_NAME ...]"
      exit 0
      ;;
    *)
      echo "Usage: $0 [--force] --job JOB_NAME [--job JOB_NAME ...]"
      exit 1
      ;;
  esac
done

if [ ${#JOB_NAMES[@]} -eq 0 ]; then
  echo "Usage: $0 [--force] --job JOB_NAME [--job JOB_NAME ...]"
  exit 1
fi

for JOB_NAME in "${JOB_NAMES[@]}"; do
  TARGET_DIR="$VALIDATE_ROOT/$JOB_NAME"
  mkdir -p "$TARGET_DIR"
  normalize_job_dir "$TARGET_DIR" "$JOB_NAME"

  echo "Syncing validation images for $JOB_NAME into $TARGET_DIR"
  REMOTE_ROOT="validate/$JOB_NAME"
  TMP_LIST="$(mktemp "$VALIDATE_ROOT/.list_${JOB_NAME}.XXXXXX")"

  if ! modal_volume_ls flux-lora-models "$REMOTE_ROOT" >"$TMP_LIST" 2>/dev/null; then
    rm -f "$TMP_LIST"
    if ! sync_s3_validation "$JOB_NAME" "$TARGET_DIR"; then
      echo "No remote validation folder found for $JOB_NAME"
    fi
    continue
  fi

  REMOTE_STEP_DIRS=()
  while read -r remote_entry; do
    [ -n "$remote_entry" ] || continue
    step_name="$(basename "$remote_entry")"
    if [[ "$step_name" == step_* ]]; then
      REMOTE_STEP_DIRS+=("$step_name")
    fi
  done <"$TMP_LIST"
  rm -f "$TMP_LIST"

  if [ ${#REMOTE_STEP_DIRS[@]} -eq 0 ]; then
    if ! sync_s3_validation "$JOB_NAME" "$TARGET_DIR"; then
      echo "No remote step folders found for $JOB_NAME"
    fi
    continue
  fi

  for STEP_NAME in "${REMOTE_STEP_DIRS[@]}"; do
    if [ ${#FORCE_ARGS[@]} -eq 0 ] && [ -d "$TARGET_DIR/$STEP_NAME" ]; then
      echo "Skipping existing $JOB_NAME/$STEP_NAME"
      continue
    fi

    TMP_DIR="$(mktemp -d "$VALIDATE_ROOT/.sync_${JOB_NAME}_${STEP_NAME}.XXXXXX")"
    echo "Downloading $REMOTE_ROOT/$STEP_NAME"
    if [ ${#FORCE_ARGS[@]} -gt 0 ]; then
      MODAL_GET_CMD=(modal_volume_get "${FORCE_ARGS[@]}" flux-lora-models "$REMOTE_ROOT/$STEP_NAME" "$TMP_DIR")
    else
      MODAL_GET_CMD=(modal_volume_get flux-lora-models "$REMOTE_ROOT/$STEP_NAME" "$TMP_DIR")
    fi

    if ! "${MODAL_GET_CMD[@]}"; then
      rm -rf "$TMP_DIR"
      echo "Failed to download $JOB_NAME/$STEP_NAME"
      continue
    fi

    normalize_job_dir "$TMP_DIR" "$JOB_NAME"
    merge_dir_contents "$TMP_DIR" "$TARGET_DIR"
    rm -rf "$TMP_DIR"
    normalize_job_dir "$TARGET_DIR" "$JOB_NAME"
  done

  sync_s3_validation "$JOB_NAME" "$TARGET_DIR" || true
done

echo "Done."
