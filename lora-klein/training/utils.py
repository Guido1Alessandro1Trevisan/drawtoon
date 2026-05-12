#!/usr/bin/env python3
"""Prepare an ai-toolkit DDP config for direct EC2 training.

This mirrors the small runtime rewrites that run_modal.py normally performs
before launching Modal, without importing Modal or changing the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image


S3_BUCKET = os.environ.get("DRAWTOON_S3_BUCKET") or os.environ.get("S3_BUCKET") or "drawtoon"
S3_MODELS_PREFIX = "models"
DEFAULT_OUTPUT_ROOT = "/tmp/lineart2_training_output"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket_and_key = uri[5:]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return bucket, key


def s3_uri(*parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    return f"s3://{S3_BUCKET}/{'/'.join(clean_parts)}" if clean_parts else f"s3://{S3_BUCKET}"


def s3_head(s3_client, uri: str) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    return s3_client.head_object(Bucket=bucket, Key=key)


def download_s3_uri(s3_client, uri: str, destination: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = int(s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if destination.exists() and destination.stat().st_size == remote_size:
        return destination
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    s3_client.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(destination)
    return destination


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def iter_manifest_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def cache_path_for_s3_asset(asset_s3: str, root: Path) -> Path:
    bucket, key = parse_s3_uri(asset_s3)
    digest = hashlib.sha1(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:16]
    suffix = Path(key).suffix or ".bin"
    name = Path(key).stem[:80]
    return root / digest[:2] / f"{name}_{digest}{suffix}"


def materialize_asset(
    s3_client,
    asset_ref: str,
    *,
    asset_root: Path,
    dataset_s3_path: str,
) -> Path:
    raw = str(asset_ref).strip()
    if not raw:
        raise ValueError("Empty validation asset reference")
    if raw.startswith("s3://"):
        return download_s3_uri(s3_client, raw, cache_path_for_s3_asset(raw, asset_root))
    if os.path.isabs(raw):
        return Path(raw)
    if not dataset_s3_path:
        raise ValueError(f"Relative asset reference needs dataset_s3_path: {raw!r}")
    return materialize_asset(
        s3_client,
        dataset_s3_path.rstrip("/") + "/" + raw.lstrip("/"),
        asset_root=asset_root,
        dataset_s3_path=dataset_s3_path,
    )


def build_static_validation_samples(
    s3_client,
    *,
    manifest_path: Path,
    dataset_s3_path: str,
    output_root: Path,
    sample_count: int,
    max_character_refs: int,
) -> list[dict[str, Any]]:
    asset_root = output_root / "_validation_assets"
    samples: list[dict[str, Any]] = []
    for row in iter_manifest_rows(manifest_path):
        if row.get("sample_type") not in {"character_ref_to_panel", "lamic_panel_prediction"}:
            continue
        target_path = materialize_asset(
            s3_client,
            str(row["target_panel"]),
            asset_root=asset_root,
            dataset_s3_path=dataset_s3_path,
        )
        with Image.open(target_path) as image:
            width, height = image.size
        sample: dict[str, Any] = {
            "prompt": str(row["caption"]),
            "width": int(width),
            "height": int(height),
            "validation_sample_id": str(row.get("sample_id", "")),
            "validation_target_path": str(target_path),
            "validation_caption": str(row["caption"]),
        }
        control_paths: list[str] = []
        controls = row.get("controls", {}) or {}
        for ref_path in controls.get("character_ref_paths", [])[:max_character_refs]:
            local_ref = materialize_asset(
                s3_client,
                str(ref_path),
                asset_root=asset_root,
                dataset_s3_path=dataset_s3_path,
            )
            control_paths.append(str(local_ref))
        for idx, control_path in enumerate(control_paths[:7], start=1):
            sample[f"ctrl_img_{idx}"] = control_path
        sample["validation_control_paths"] = control_paths[:7]
        samples.append(sample)
        if len(samples) >= sample_count:
            break
    if not samples:
        raise RuntimeError(f"No validation samples could be built from {manifest_path}")
    return samples


def compute_steps_per_epoch(row_count: int, *, batch_size: int, gradient_accumulation_steps: int, world_size: int) -> int:
    emitted_microbatches = max(1, math.ceil(row_count / max(1, batch_size)))
    per_rank_microbatches = max(1, math.ceil(emitted_microbatches / max(1, world_size)))
    return max(1, math.ceil(per_rank_microbatches / max(1, gradient_accumulation_steps)))


def prepare_ec2_ai_toolkit_config(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True, help="Local or s3:// manifest.jsonl prepared from Drawtoon canonical data.")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--target-epochs", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-samples", type=int, default=16)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_path = Path(args.output)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    manifest_source = args.manifest.strip()
    if manifest_source.startswith("s3://"):
        manifest_head = s3_head(s3_client, manifest_source)
        manifest_digest = hashlib.sha1(manifest_source.encode("utf-8")).hexdigest()[:16]
        local_manifest = output_root / "_resolved_manifests" / f"{manifest_digest}.jsonl"
        download_s3_uri(s3_client, manifest_source, local_manifest)
        manifest_size = int(manifest_head["ContentLength"])
    else:
        local_manifest = Path(manifest_source)
        if not local_manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {local_manifest}")
        manifest_size = local_manifest.stat().st_size

    row_count = count_jsonl_rows(local_manifest)

    parsed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    process_config = parsed_config.setdefault("config", {}).setdefault("process", [{}])[0]
    train_config = process_config.setdefault("train", {})
    datasets = process_config.setdefault("datasets", [])
    if not datasets:
        raise RuntimeError("Training config has no datasets")
    dataset = datasets[0]

    model_id = args.model_id.strip() or process_config.get("name") or parsed_config["config"].get("name")
    if not model_id:
        raise RuntimeError("Training config is missing a job name")
    parsed_config["config"]["name"] = model_id
    process_config["name"] = model_id
    parsed_config.setdefault("meta", {})["name"] = model_id

    process_config["training_folder"] = str(output_root)
    process_config["sample_copy_root"] = str(output_root / "validate")
    process_config["device"] = "cuda"

    dataset["manifest_path"] = str(local_manifest)
    dataset["_prepared_row_count"] = row_count
    dataset["_prepared_manifest_summary"] = {"source": manifest_source, "row_count": row_count}
    dataset.setdefault("dataset_s3_path", str(local_manifest.parent))
    dataset_s3_path = str(dataset.get("dataset_s3_path") or "").strip()
    max_character_refs = int(dataset.get("max_character_refs") or 6)

    batch_size = int(train_config.get("batch_size") or 1)
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps") or 1)
    steps_per_epoch = compute_steps_per_epoch(
        row_count,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=args.world_size,
    )
    total_steps = steps_per_epoch * max(1, args.target_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    train_config["steps"] = total_steps
    train_config["force_first_sample"] = False
    train_config["skip_first_sample"] = True

    save_config = process_config.setdefault("save", {})
    save_every = int(save_config.get("save_every", 0) or 0)
    if save_every <= 0:
        save_config["save_every"] = steps_per_epoch
    save_config.setdefault("max_step_saves_to_keep", 5)

    validation_samples = build_static_validation_samples(
        s3_client,
        manifest_path=local_manifest,
        dataset_s3_path=dataset_s3_path,
        output_root=output_root,
        sample_count=max(1, args.validation_samples),
        max_character_refs=max_character_refs,
    )
    sample_config = process_config.setdefault("sample", {})
    sample_config.pop("prompts", None)
    sample_config["samples"] = validation_samples
    if args.world_size > 1:
        sample_config["distributed_sampling"] = True
        sample_config["distributed_sample_total"] = len(validation_samples)
        sample_config["distributed_samples_per_rank"] = max(1, math.ceil(len(validation_samples) / args.world_size))
        sample_config["distributed_sample_strategy"] = "round_robin"
    sample_config["dynamic_validation_enabled"] = True
    sample_config["dynamic_validation_manifest_paths"] = [str(local_manifest)]
    sample_config["dynamic_validation_bucket_tolerance"] = int(dataset.get("bucket_tolerance") or 16)
    sample_config["dynamic_validation_character_panel_count"] = len(validation_samples)
    sample_config["dynamic_validation_character_panel_fixed_count"] = max(0, len(validation_samples) - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(parsed_config, sort_keys=False), encoding="utf-8")

    summary = {
        "config": str(output_path),
        "job_name": model_id,
        "manifest_source": manifest_source,
        "manifest_local": str(local_manifest),
        "manifest_size": manifest_size,
        "row_count": row_count,
        "steps_per_epoch": steps_per_epoch,
        "target_epochs": args.target_epochs,
        "total_steps": total_steps,
        "validation_samples": len(validation_samples),
        "dynamic_validation_enabled": bool(sample_config["dynamic_validation_enabled"]),
        "dynamic_validation_fixed_samples": int(sample_config["dynamic_validation_character_panel_fixed_count"]),
        "dynamic_validation_random_samples": int(
            len(validation_samples) - sample_config["dynamic_validation_character_panel_fixed_count"]
        ),
        "s3_model_prefix": s3_uri(S3_MODELS_PREFIX, model_id),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _define_checkpoint_sync_modal_app():
    import modal

    volume_mount = Path("/mnt/models")
    app = modal.App("lineart2-sync-existing-checkpoints-to-s3")
    model_volume = modal.Volume.from_name("flux-lora-models")
    aws_secret = modal.Secret.from_name("lineart2-aws-s3")
    image = modal.Image.debian_slim().pip_install("boto3")

    @app.function(
        image=image,
        volumes={str(volume_mount): model_volume},
        secrets=[aws_secret],
        timeout=6 * 60 * 60,
        cpu=4,
        memory=8192,
    )
    def sync_existing_checkpoints(job_names: list[str]) -> dict:
        import boto3
        from boto3.s3.transfer import TransferConfig

        client = boto3.client("s3")
        transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=128 * 1024 * 1024,
            max_concurrency=16,
            use_threads=True,
        )
        uploaded = []
        missing = []

        def upload_file(path: Path, key: str) -> None:
            print(f"upload {path} -> s3://{S3_BUCKET}/{key}", flush=True)
            client.upload_file(str(path), S3_BUCKET, key, Config=transfer_config)
            uploaded.append(
                {
                    "path": str(path),
                    "s3": f"s3://{S3_BUCKET}/{key}",
                    "bytes": path.stat().st_size,
                }
            )

        for job_name in job_names:
            save_root = volume_mount / job_name
            if not save_root.exists():
                missing.append({"job": job_name, "reason": "missing volume directory"})
                continue

            model_prefix = f"models/{job_name}"
            for name in ("config.yaml", "optimizer.pt"):
                path = save_root / name
                if path.is_file():
                    upload_file(path, f"{model_prefix}/{name}")

            checkpoint_paths = [path for path in sorted(save_root.glob(f"{job_name}*")) if path.is_file()]
            if not checkpoint_paths:
                missing.append({"job": job_name, "reason": "no checkpoint files"})
                continue

            for path in checkpoint_paths:
                upload_file(path, f"{model_prefix}/checkpoints/{path.name}")

        return {"uploaded": uploaded, "missing": missing}

    @app.local_entrypoint()
    def sync_existing_checkpoints_to_s3(jobs: str):
        job_names = [part.strip() for part in jobs.split(",") if part.strip()]
        if not job_names:
            raise ValueError("Pass --jobs with one job name or comma-separated job names")
        result = sync_existing_checkpoints.remote(job_names)
        print(result)

    return app, sync_existing_checkpoints


try:
    app, sync_existing_checkpoints = _define_checkpoint_sync_modal_app()
except ModuleNotFoundError:
    app = None
    sync_existing_checkpoints = None


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in {"prepare-ec2-ai-toolkit-config", "prepare-ec2-config"}:
        prepare_ec2_ai_toolkit_config(sys.argv[2:])
        return
    if command and not command.startswith("-"):
        raise SystemExit(f"Unknown utils command: {command}")
    prepare_ec2_ai_toolkit_config(sys.argv[1:])


if __name__ == "__main__":
    main()
