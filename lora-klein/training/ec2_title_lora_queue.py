#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import boto3
import yaml


S3_BUCKET = "drawtoon"
PAGES_PREFIX = "datasets/pages/filtered"
ANNOTATIONS_PREFIX = "datasets/annotations/magi_v3"
CAPTIONS_PREFIX = "captions"
MANGAZERO_CAPTION_RUN = "haiku45_mangazero_page_panel_v1"
MANGA109_CAPTION_RUN = "haiku45_manga109_page_panel_v1"
DEFAULT_BASE_CONFIG = "lora-klein/training/configs/haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr5e5_b300_gb1.yaml"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
JOB_SUFFIX = ""


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


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._=-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:160] or hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def list_common_prefixes(s3: Any, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix.rstrip("/") + "/", Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            name = str(item.get("Prefix") or "").rstrip("/").split("/")[-1]
            if name:
                prefixes.append(name)
    return sorted(set(prefixes))


def count_objects(s3: Any, prefix: str, suffixes: str | tuple[str, ...]) -> int:
    if isinstance(suffixes, str):
        suffix_tuple = (suffixes,)
    else:
        suffix_tuple = tuple(suffixes)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.lower().endswith(suffix_tuple):
                count += 1
    return count


def caption_run_for_title(title: str) -> str:
    if title.endswith("_manga109"):
        return MANGA109_CAPTION_RUN
    if title.endswith("_mangazero"):
        return MANGAZERO_CAPTION_RUN
    raise ValueError(f"Unsupported title suffix: {title}")


def title_ready(s3: Any, title: str) -> tuple[bool, int, int]:
    page_count = count_objects(s3, f"{PAGES_PREFIX}/{title}", (".jpg", ".jpeg", ".png", ".webp"))
    caption_count = count_objects(s3, f"{CAPTIONS_PREFIX}/{caption_run_for_title(title)}/{title}", ".json")
    return page_count > 0 and caption_count >= page_count, page_count, caption_count


def title_sort_key(title: str) -> tuple[int, str]:
    dataset_rank = 1 if title.endswith("_manga109") else 0
    return dataset_rank, title


def job_name_for_title(title: str) -> str:
    suffix = f"_{sanitize_name(JOB_SUFFIX)}" if JOB_SUFFIX.strip() else ""
    return (
        "drawtoon_flux2_klein9b_"
        f"{sanitize_name(title)}"
        f"_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1{suffix}"
    )


def legacy_job_name_for_title(title: str) -> str:
    return (
        "drawtoon_flux2_klein9b_"
        f"{sanitize_name(title)}"
        "_panel_prediction_native_pad16_haiku45_lora_r64_lr5e5_3500_b300_gb1"
    )


def completed_adapter_exists(s3: Any, title: str) -> bool:
    key = f"models/{job_name_for_title(title)}/final/peft_adapter/adapter_model.safetensors"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def archive_legacy_adapter_to_old(s3: Any, title: str) -> str:
    legacy_job = legacy_job_name_for_title(title)
    source_prefix = f"models/{legacy_job}/final/peft_adapter/"
    target_prefix = f"models/old/{legacy_job}/final/peft_adapter/"
    copied = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            source_key = str(obj.get("Key") or "")
            if not source_key:
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": source_key},
                Key=target_key,
            )
            copied += 1
    if copied:
        return f"s3://{S3_BUCKET}/{target_prefix}"
    return ""


class QueueState:
    def __init__(self, path: Path, lora_path: Path, titles: list[str], *, pages_prefix: str, job_suffix: str) -> None:
        self.path = path
        self.lora_path = lora_path
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {
            title: {
                "title": title,
                "dataset": "manga109" if title.endswith("_manga109") else "mangazero",
                "caption_run": caption_run_for_title(title),
                "status": "pending",
                "job_name": job_name_for_title(title),
                "s3": f"s3://{S3_BUCKET}/models/{job_name_for_title(title)}/",
                "gpu": "",
                "page_count": "",
                "caption_count": "",
                "archive": "",
                "started_at": "",
                "finished_at": "",
                "error": "",
            }
            for title in titles
        }
        self.write()

    def update(self, title: str, **values: Any) -> None:
        with self.lock:
            self.rows[title].update({key: str(value) for key, value in values.items()})
            self.write_locked()

    def write(self) -> None:
        with self.lock:
            self.write_locked()

    def write_locked(self) -> None:
        lines = [
            "# Per-title LoRA Checklist",
            "",
            f"Last updated: `{utc_now()}`",
            "",
            "Setup:",
            "",
            "- One LoRA per title folder/franchise.",
            "- One B300 GPU per LoRA process; up to 8 jobs run in parallel.",
            "- Rank `64`, LR `5e-5`, batch size `1`, gradient accumulation `1`.",
            "- Fixed `3500` train steps.",
            "- Validation disabled during training.",
            f"- Pages prefix: `{pages_prefix}`.",
            f"- Job suffix: `{job_suffix or '(none)'}`.",
            "- MangaZero captions: `haiku45_mangazero_page_panel_v1`.",
            "- Manga109 captions: `haiku45_manga109_page_panel_v1`.",
            "",
            "| Status | Dataset | Title | GPU | Captions | Job | S3 | Archive | Started | Finished | Error |",
            "|---|---|---|---:|---:|---|---|---|---|---|---|",
        ]
        status_order = {
            "running": 0,
            "completed": 1,
            "failed": 2,
            "waiting_captions": 3,
            "pending": 4,
        }
        for row in sorted(
            self.rows.values(),
            key=lambda item: (status_order.get(str(item["status"]), 9), str(item["dataset"]), str(item["title"])),
        ):
            captions = f"{row.get('caption_count') or '?'}/{row.get('page_count') or '?'}"
            lines.append(
                "| {status} | {dataset} | `{title}` | {gpu} | {captions} | `{job_name}` | {s3} | {archive} | {started_at} | {finished_at} | {error} |".format(
                    status=row.get("status", ""),
                    dataset=row.get("dataset", ""),
                    title=row.get("title", ""),
                    gpu=row.get("gpu", ""),
                    captions=captions,
                    job_name=row.get("job_name", ""),
                    s3=row.get("s3", ""),
                    archive=row.get("archive", ""),
                    started_at=row.get("started_at", ""),
                    finished_at=row.get("finished_at", ""),
                    error=str(row.get("error", "")).replace("|", "/")[:180],
                )
            )
        body = "\n".join(lines) + "\n"
        self.path.write_text(body, encoding="utf-8")
        self.lora_path.write_text(body, encoding="utf-8")


def write_title_config(base_config: Path, output_config: Path, title: str) -> str:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    job_name = job_name_for_title(title)
    process = cfg["config"]["process"][0]
    cfg["config"]["name"] = job_name
    process["name"] = job_name
    cfg.setdefault("meta", {})["name"] = job_name
    cfg["meta"]["description"] = (
        f"Per-title LoRA for {title}, rank 64, LR 5e-5, single B300 GPU/global batch 1, "
        "3500 steps, validation disabled."
    )
    process.setdefault("network", {})["linear"] = 64
    process.setdefault("network", {})["linear_alpha"] = 64
    process.setdefault("save", {})["save_every"] = 0
    process.setdefault("save", {})["max_step_saves_to_keep"] = 1
    train = process.setdefault("train", {})
    train["batch_size"] = 1
    train["gradient_accumulation_steps"] = 1
    train["lr"] = 5.0e-5
    train["steps"] = 3500
    sample = process.setdefault("sample", {})
    sample["sample_every"] = 0
    sample["prompts"] = []
    drawtoon = cfg.setdefault("drawtoon", {})
    drawtoon["bucket"] = S3_BUCKET
    drawtoon["pages_prefix"] = PAGES_PREFIX
    drawtoon["annotations_prefix"] = ANNOTATIONS_PREFIX
    drawtoon["captions_prefix"] = CAPTIONS_PREFIX
    drawtoon["caption_run"] = caption_run_for_title(title)
    drawtoon["caption_field"] = "caption"
    drawtoon["caption_format"] = "text"
    drawtoon["include_chapter_regex"] = f"^{re.escape(title)}$"
    drawtoon.pop("exclude_sample_ids_path", None)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return job_name


def export_peft(
    *,
    repo_root: Path,
    cache_root: Path,
    output_root: Path,
    job_name: str,
    prepared_config: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    code = r"""
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
"""
    export_env = dict(env)
    export_env["REPO_ROOT"] = str(repo_root)
    export_env["CACHE_ROOT"] = str(cache_root)
    export_env["LINEART2_TRAINING_OUTPUT_ROOT"] = str(output_root)
    run([sys.executable, "-c", code, job_name, str(prepared_config)], env=export_env, log_path=log_path)


def sync_artifacts(job_name: str, output_root: Path, log_path: Path) -> None:
    save_root = output_root / job_name
    model_prefix = f"s3://{S3_BUCKET}/models/{job_name}"
    if save_root.exists():
        run(["aws", "s3", "sync", str(save_root), f"{model_prefix}/checkpoints", "--only-show-errors"], log_path=log_path)
    if (save_root / "peft_adapter").exists():
        run(["aws", "s3", "sync", str(save_root / "peft_adapter"), f"{model_prefix}/final/peft_adapter", "--only-show-errors"], log_path=log_path)
    if (save_root / "config.yaml").exists():
        run(["aws", "s3", "cp", str(save_root / "config.yaml"), f"{model_prefix}/config.yaml", "--only-show-errors"], log_path=log_path)
    run(["aws", "s3", "cp", str(log_path), f"{model_prefix}/logs/train.log", "--only-show-errors"], log_path=log_path)


def train_title(args: argparse.Namespace, state: QueueState, title: str, gpu: int) -> None:
    repo_root = Path(args.repo_root).resolve()
    work_root = Path(args.work_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    base_config = repo_root / args.base_config
    title_root = work_root / "title_loras" / sanitize_name(title)
    config_path = title_root / "config.yaml"
    prepared_config = title_root / "prepared_ai_toolkit.yaml"
    output_root = title_root / "output"
    log_path = title_root / "train.log"
    job_name = write_title_config(base_config, config_path, title)
    state.update(title, status="running", gpu=gpu, started_at=utc_now(), error="")

    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(Path(args.hf_home).resolve()),
            "HUGGINGFACE_HUB_CACHE": str(Path(args.hf_home).resolve() / "hub"),
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "DISABLE_TELEMETRY": "YES",
            "PYTHONPATH": str(repo_root / "lora-klein" / "ai-toolkit"),
            "QUANTO_BYPASS_OBJECT_COPY": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "LINEART2_TRAINING_OUTPUT_ROOT": str(output_root),
            "S3_VALIDATION_UPLOAD_ROOT": f"s3://{S3_BUCKET}/models/{job_name}/validate",
            "S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD": "0",
            "S3_CHECKPOINT_KEEP_LATEST_LOCAL": "1",
            "S3_CHECKPOINT_HYDRATE_ON_START": "0",
            "REPO_ROOT": str(repo_root),
            "CACHE_ROOT": str(cache_root),
        }
    )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"[{utc_now()}] Starting {title} on GPU {gpu}\n", encoding="utf-8")

        cache_cmd = [
            sys.executable,
            str(repo_root / "lora-klein" / "training" / "utils.py"),
            "build-ec2-cache",
            "--config",
            str(config_path),
            "--cache-root",
            str(cache_root),
            "--workers",
            str(args.cache_workers),
        ]
        cache_proc = run(cache_cmd, env=env)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(cache_proc.stdout)
        cache_summary = parse_last_json(cache_proc.stdout)
        manifest_path = str(cache_summary["manifest_path"])
        resolved_config = str(cache_summary["resolved_config_path"])

        prepare_cmd = [
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
            "999",
            "--world-size",
            "1",
            "--output-root",
            str(output_root),
            "--validation-samples",
            "0",
            "--max-train-steps",
            str(args.steps),
            "--model-id",
            job_name,
        ]
        prepare_proc = run(prepare_cmd, env=env)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(prepare_proc.stdout)

        run(
            [
                "torchrun",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--master-addr=127.0.0.1",
                f"--master-port={29600 + int(gpu)}",
                "run.py",
                str(prepared_config),
            ],
            cwd=repo_root / "lora-klein" / "ai-toolkit",
            env=env,
            log_path=log_path,
        )
        export_peft(
            repo_root=repo_root,
            cache_root=cache_root,
            output_root=output_root,
            job_name=job_name,
            prepared_config=prepared_config,
            env=env,
            log_path=log_path,
        )
        sync_artifacts(job_name, output_root, log_path)
        state.update(title, status="completed", gpu="", finished_at=utc_now())
    except Exception as exc:
        try:
            sync_artifacts(job_name, output_root, log_path)
        except Exception:
            pass
        state.update(title, status="failed", gpu="", finished_at=utc_now(), error=repr(exc))
        raise


def main() -> None:
    global PAGES_PREFIX, JOB_SUFFIX

    parser = argparse.ArgumentParser(description="Run one rank-64 LoRA per Drawtoon title on a B300 node.")
    parser.add_argument("--repo-root", default="/home/ubuntu/drawtoon")
    parser.add_argument("--work-root", default="/mnt/local/drawtoon-title-loras")
    parser.add_argument("--cache-root", default="/mnt/local/drawtoon-title-loras/cache")
    parser.add_argument("--hf-home", default="/mnt/local/drawtoon-b300/hf")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--checklist", default="/home/ubuntu/drawtoon/checklist.md")
    parser.add_argument("--lora-md", default="/home/ubuntu/drawtoon/lora.md")
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--dataset", choices=["all", "mangazero", "manga109"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pages-prefix", default=PAGES_PREFIX)
    parser.add_argument("--job-suffix", default="")
    parser.add_argument(
        "--archive-legacy-adapters-to-old",
        action="store_true",
        help="Copy existing non-suffixed final PEFT adapters to s3://drawtoon/models/old/<job>/ before retraining.",
    )
    args = parser.parse_args()

    PAGES_PREFIX = args.pages_prefix.strip().strip("/")
    JOB_SUFFIX = args.job_suffix.strip()

    s3 = boto3.client("s3")
    titles = [
        title
        for title in list_common_prefixes(s3, PAGES_PREFIX)
        if title.endswith(("_mangazero", "_manga109"))
    ]
    titles.sort(key=title_sort_key)
    if args.dataset != "all":
        suffix = f"_{args.dataset}"
        titles = [title for title in titles if title.endswith(suffix)]
    if args.limit > 0:
        titles = titles[: args.limit]
    if not titles:
        raise SystemExit("No title folders found")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    state = QueueState(Path(args.checklist), Path(args.lora_md), titles, pages_prefix=PAGES_PREFIX, job_suffix=JOB_SUFFIX)
    pending: list[str] = []
    for title in titles:
        ready, page_count, caption_count = title_ready(s3, title)
        archive_uri = ""
        if args.archive_legacy_adapters_to_old:
            archive_uri = archive_legacy_adapter_to_old(s3, title)
        if completed_adapter_exists(s3, title):
            state.update(
                title,
                status="completed",
                page_count=page_count,
                caption_count=caption_count,
                archive=archive_uri,
                finished_at="existing",
            )
        else:
            if archive_uri:
                state.update(title, archive=archive_uri)
            pending.append(title)
    running: dict[concurrent.futures.Future[None], tuple[str, int]] = {}
    gpus = [int(part) for part in args.gpus.split(",") if part.strip() != ""]
    failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        while pending or running:
            done = [future for future in running if future.done()]
            for future in done:
                title, gpu = running.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    failures += 1
                    print(f"[{utc_now()}] FAILED {title} on GPU {gpu}: {exc}", flush=True)

            busy_gpus = {gpu for _, gpu in running.values()}
            free_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]
            launched = False
            for gpu in list(free_gpus):
                ready_title = ""
                for title in list(pending):
                    ready, page_count, caption_count = title_ready(s3, title)
                    state.update(
                        title,
                        status="pending" if ready else "waiting_captions",
                        page_count=page_count,
                        caption_count=caption_count,
                    )
                    if ready:
                        ready_title = title
                        pending.remove(title)
                        break
                if not ready_title:
                    break
                future = pool.submit(train_title, args, state, ready_title, gpu)
                running[future] = (ready_title, gpu)
                print(f"[{utc_now()}] Launched {ready_title} on GPU {gpu}", flush=True)
                launched = True

            if pending and not launched and not running:
                print(f"[{utc_now()}] Waiting for captions for {len(pending)} titles", flush=True)
            if pending or running:
                time.sleep(max(10, int(args.poll_seconds)))

    print(f"[{utc_now()}] Queue finished with failures={failures}", flush=True)


if __name__ == "__main__":
    main()
