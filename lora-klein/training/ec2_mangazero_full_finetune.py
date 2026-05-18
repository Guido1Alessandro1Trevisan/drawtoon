#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml


S3_BUCKET = "drawtoon"
JOB_NAME = "drawtoon_flux2_klein9b_mangazero_text_removed_panel_prediction_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch"
BASE_CONFIG = (
    "lora-klein/training/configs/haiku-4.5/"
    "panel_prediction_mangazero_text_removed_same_page_refs_native_pad16_haiku45_lr28e7_ga8_full_b300_1epoch.yaml"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    text = " ".join(cmd)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] $ {text}\n")
            handle.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return subprocess.CompletedProcess(cmd, proc.returncode, "", "")

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def parse_last_json(text: str) -> dict[str, Any]:
    starts = [idx for idx, char in enumerate(text) if char == "{"]
    for start in reversed(starts):
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in command output")


def extract_latest_step(log_text: str) -> int | None:
    matches = re.findall(r"(?:step|steps?)\D+([0-9]{1,9})(?:\D+of\D+|/)([0-9]{1,9})", log_text, re.IGNORECASE)
    if matches:
        return int(matches[-1][0])
    saved = re.findall(r"Saved checkpoint to .*?_([0-9]{9})(?:\D|$)", log_text)
    if saved:
        return int(saved[-1])
    return None


def write_status(status_path: Path, data: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_hf_token(*, secret_id: str, region: str) -> str:
    if os.environ.get("HF_TOKEN"):
        return str(os.environ["HF_TOKEN"]).strip()
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return str(os.environ["HUGGING_FACE_HUB_TOKEN"]).strip()
    proc = run(
        [
            "aws",
            "secretsmanager",
            "get-secret-value",
            "--secret-id",
            secret_id,
            "--region",
            region,
            "--query",
            "SecretString",
            "--output",
            "text",
        ]
    )
    raw = proc.stdout.strip()
    if raw.startswith("{"):
        payload = json.loads(raw)
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "token"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        raise RuntimeError(f"Hugging Face secret JSON did not contain a token key: {secret_id}")
    return raw


def hf_preflight(env: dict[str, str], log_path: Path) -> None:
    code = """
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    "black-forest-labs/FLUX.2-klein-base-9B",
    "model_index.json",
    token=os.environ["HF_TOKEN"],
)
print("HF gated model preflight succeeded")
""".strip()
    run([sys.executable, "-c", code], env=env, log_path=log_path)


def sync_artifacts(
    *,
    output_root: Path,
    work_root: Path,
    log_path: Path,
    status_path: Path,
    prepared_config: Path,
    s3_prefix: str,
) -> None:
    model_uri = f"s3://{S3_BUCKET}/models/{JOB_NAME}"
    save_root = output_root / JOB_NAME
    commands: list[list[str]] = []
    if save_root.exists():
        commands.append(["aws", "s3", "sync", str(save_root), f"{model_uri}/checkpoints", "--only-show-errors"])
    config_file = save_root / "config.yaml"
    if config_file.exists():
        commands.append(["aws", "s3", "cp", str(config_file), f"{model_uri}/config.yaml", "--only-show-errors"])
    optimizer_file = save_root / "optimizer.pt"
    if optimizer_file.exists():
        commands.append(["aws", "s3", "cp", str(optimizer_file), f"{model_uri}/optimizer.pt", "--only-show-errors"])
    if prepared_config.exists():
        commands.append(
            ["aws", "s3", "cp", str(prepared_config), f"{model_uri}/prepared_ai_toolkit.yaml", "--only-show-errors"]
        )
    if status_path.exists():
        commands.append(["aws", "s3", "cp", str(status_path), f"{model_uri}/status.json", "--only-show-errors"])
    if log_path.exists():
        commands.append(["aws", "s3", "cp", str(log_path), f"{model_uri}/logs/train.log", "--only-show-errors"])
        commands.append(["aws", "s3", "cp", str(log_path), f"s3://{S3_BUCKET}/{s3_prefix}/train.log", "--only-show-errors"])
    commands.append(
        [
            "aws",
            "s3",
            "sync",
            str(work_root),
            f"s3://{S3_BUCKET}/{s3_prefix}/work",
            "--exclude",
            "output/*",
            "--exclude",
            "cache/*",
            "--only-show-errors",
        ]
    )

    for cmd in commands:
        try:
            run(cmd)
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{utc_now()}] artifact sync failed for {' '.join(cmd)}: {exc!r}\n")


def start_monitor_thread(
    *,
    output_root: Path,
    work_root: Path,
    log_path: Path,
    status_path: Path,
    prepared_config: Path,
    status: dict[str, Any],
    s3_prefix: str,
    interval_seconds: int,
    stop_event: threading.Event,
) -> threading.Thread:
    def worker() -> None:
        while not stop_event.wait(interval_seconds):
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            latest_step = extract_latest_step(log_text)
            status.update(
                {
                    "status": status.get("status", "running"),
                    "last_monitor_at": utc_now(),
                    "latest_step_seen": latest_step,
                }
            )
            write_status(status_path, status)
            sync_artifacts(
                output_root=output_root,
                work_root=work_root,
                log_path=log_path,
                status_path=status_path,
                prepared_config=prepared_config,
                s3_prefix=s3_prefix,
            )

    thread = threading.Thread(target=worker, name="artifact-monitor", daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MangaZero text-removed full fine-tune with DDP on B300 GPUs.")
    parser.add_argument("--repo-root", default="/workspace")
    parser.add_argument("--work-root", default="/mnt/local/drawtoon-b300/full-finetune-text-removed")
    parser.add_argument("--cache-root", default="/mnt/local/drawtoon-b300/full-finetune-text-removed/cache")
    parser.add_argument("--hf-home", default="/mnt/local/drawtoon-b300/hf")
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=96)
    parser.add_argument("--sync-seconds", type=int, default=1800)
    parser.add_argument("--s3-prefix", default=f"ec2/b300-full-finetune/{JOB_NAME}")
    parser.add_argument("--hf-secret-id", default=os.environ.get("HF_SECRET_ID", "lineart2-hf-token"))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_root = Path(args.work_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    output_root = work_root / "output"
    log_path = work_root / "train.log"
    status_path = work_root / "status.json"
    base_config = repo_root / BASE_CONFIG
    prepared_config = work_root / "prepared_ai_toolkit.yaml"

    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"[{utc_now()}] Starting {JOB_NAME} on GPUs {args.gpu}\n", encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(Path(args.hf_home).resolve()),
            "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_home).resolve() / "hub"),
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "DISABLE_TELEMETRY": "YES",
            "PYTHONPATH": str(repo_root / "lora-klein" / "ai-toolkit"),
            "QUANTO_BYPASS_OBJECT_COPY": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LINEART2_TRAINING_OUTPUT_ROOT": str(output_root),
            "S3_VALIDATION_UPLOAD_ROOT": f"s3://{S3_BUCKET}/models/{JOB_NAME}/validate",
            "S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD": "0",
            "S3_CHECKPOINT_KEEP_LATEST_LOCAL": "1",
            "S3_CHECKPOINT_HYDRATE_ON_START": "0",
            "REPO_ROOT": str(repo_root),
            "CACHE_ROOT": str(cache_root),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "TORCH_DISTRIBUTED_DEBUG": os.environ.get("TORCH_DISTRIBUTED_DEBUG", "OFF"),
            "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "1"),
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0"),
            "AITK_DDP_STATIC_GRAPH": os.environ.get("AITK_DDP_STATIC_GRAPH", "1"),
            "AITK_DDP_FIND_UNUSED_PARAMETERS": os.environ.get("AITK_DDP_FIND_UNUSED_PARAMETERS", "0"),
        }
    )

    status: dict[str, Any] = {
        "job_name": JOB_NAME,
        "status": "running",
        "started_at": utc_now(),
        "visible_devices": args.gpu,
        "world_size": args.world_size,
        "repo_root": str(repo_root),
        "work_root": str(work_root),
        "s3": f"s3://{S3_BUCKET}/models/{JOB_NAME}/",
        "caption_run": "haiku45_mangazero_page_panel_v1",
        "pages_prefix": "datasets/pages/text_removed/qwen2511_master_prompt_mangazero_v1",
        "learning_rate": 2.8e-6,
        "gradient_accumulation_steps": 8,
        "full_finetune": True,
    }
    write_status(status_path, status)

    hf_token = resolve_hf_token(secret_id=args.hf_secret_id, region=args.aws_region)
    if not hf_token:
        raise RuntimeError(f"Empty Hugging Face token from {args.hf_secret_id}")
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env.setdefault("AWS_REGION", args.aws_region)
    env.setdefault("AWS_DEFAULT_REGION", args.aws_region)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{utc_now()}] Loaded Hugging Face token from {shlex.quote(args.hf_secret_id)} "
            f"and AWS region {args.aws_region}\n"
        )

    stop_event = threading.Event()
    monitor_thread = start_monitor_thread(
        output_root=output_root,
        work_root=work_root,
        log_path=log_path,
        status_path=status_path,
        prepared_config=prepared_config,
        status=status,
        s3_prefix=args.s3_prefix,
        interval_seconds=max(60, int(args.sync_seconds)),
        stop_event=stop_event,
    )

    try:
        hf_preflight(env, log_path)
        cache_proc = run(
            [
                sys.executable,
                str(repo_root / "lora-klein" / "training" / "utils.py"),
                "build-ec2-cache",
                "--config",
                str(base_config),
                "--cache-root",
                str(cache_root),
                "--workers",
                str(args.cache_workers),
                "--overwrite",
            ],
            env=env,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(cache_proc.stdout)
        cache_summary = parse_last_json(cache_proc.stdout)
        manifest_path = str(cache_summary["manifest_path"])
        resolved_config = str(cache_summary["resolved_config_path"])

        prepare_proc = run(
            [
                sys.executable,
                str(repo_root / "lora-klein" / "training" / "utils.py"),
                "prepare-ec2-config",
                "--config",
                resolved_config,
                "--output",
                str(prepared_config),
                "--manifest",
                manifest_path,
                "--target-epochs",
                "1",
                "--world-size",
                str(args.world_size),
                "--output-root",
                str(output_root),
                "--validation-samples",
                "0",
                "--model-id",
                JOB_NAME,
            ],
            env=env,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(prepare_proc.stdout)
        prepare_summary = parse_last_json(prepare_proc.stdout)
        total_steps = int(prepare_summary.get("total_steps") or 1)
        half_epoch_step = max(1, int(math.ceil(total_steps / 2)))
        prepared = yaml.safe_load(prepared_config.read_text(encoding="utf-8"))
        process = prepared.setdefault("config", {}).setdefault("process", [{}])[0]
        process.setdefault("save", {})["save_every"] = half_epoch_step
        process.setdefault("save", {})["max_step_saves_to_keep"] = 3
        process.setdefault("sample", {})["sample_every"] = 0
        process.setdefault("sample", {})["samples"] = []
        process["sample"]["dynamic_validation_enabled"] = False
        prepared_config.write_text(yaml.safe_dump(prepared, sort_keys=False), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] Set half-epoch checkpoint save_every={half_epoch_step} for total_steps={total_steps}\n")
        status.update(
            {
                "manifest_path": manifest_path,
                "row_count": prepare_summary.get("row_count"),
                "steps_per_epoch": prepare_summary.get("steps_per_epoch"),
                "total_steps": prepare_summary.get("total_steps"),
                "save_every": half_epoch_step,
                "checkpoint_policy": "single intermediate checkpoint at half epoch plus final save",
                "validation_samples": 0,
                "prepared_config": str(prepared_config),
            }
        )
        write_status(status_path, status)
        sync_artifacts(
            output_root=output_root,
            work_root=work_root,
            log_path=log_path,
            status_path=status_path,
            prepared_config=prepared_config,
            s3_prefix=args.s3_prefix,
        )

        run(
            [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={args.world_size}",
                "run.py",
                str(prepared_config),
            ],
            cwd=repo_root / "lora-klein" / "ai-toolkit",
            env=env,
            log_path=log_path,
        )
        status.update({"status": "completed", "finished_at": utc_now()})
        write_status(status_path, status)
        sync_artifacts(
            output_root=output_root,
            work_root=work_root,
            log_path=log_path,
            status_path=status_path,
            prepared_config=prepared_config,
            s3_prefix=args.s3_prefix,
        )
    except Exception as exc:
        status.update({"status": "failed", "finished_at": utc_now(), "error": repr(exc)})
        write_status(status_path, status)
        sync_artifacts(
            output_root=output_root,
            work_root=work_root,
            log_path=log_path,
            status_path=status_path,
            prepared_config=prepared_config,
            s3_prefix=args.s3_prefix,
        )
        raise
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)


if __name__ == "__main__":
    main()
