#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


S3_BUCKET = "drawtoon"
JOB_NAME = (
    "drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_"
    "haiku45_lora_r64_lr5e5_3500_b300_gb1_textcolor_bubbles"
)
BASE_CONFIG = "lora-klein/training/configs/haiku-4.5/panel_prediction_attack_on_titan_native_pad16_lora_r64_lr5e5_b300_gb1.yaml"
TITLE = "attack-on-titan_mangazero"
TEXT_COLORS = {
    "speech_blue": (0, 96, 255),
    "narration_orange": (255, 128, 0),
    "shout_violet": (128, 0, 255),
}


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


def write_config(repo_root: Path, output_config: Path) -> None:
    base_config = repo_root / BASE_CONFIG
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    process = cfg["config"]["process"][0]
    cfg["config"]["name"] = JOB_NAME
    process["name"] = JOB_NAME
    cfg.setdefault("meta", {})["name"] = JOB_NAME
    cfg["meta"]["description"] = (
        "Attack on Titan text-color layout ablation. Same r64/LR 5e-5/global-batch-1/3500-step setup "
        "as the baseline, but speech/narration/shout text layout masks use blue/orange/violet rectangles "
        "instead of shape-coded blue masks."
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
    drawtoon["pages_prefix"] = "datasets/pages/filtered"
    drawtoon["annotations_prefix"] = "datasets/annotations/magi_v3"
    drawtoon["captions_prefix"] = "captions"
    drawtoon["caption_run"] = "haiku45_mangazero_page_panel_v1"
    drawtoon["caption_field"] = "caption"
    drawtoon["caption_format"] = "text"
    drawtoon["include_chapter_regex"] = f"^{TITLE}$"
    drawtoon["exclude_sample_ids_path"] = str(
        repo_root / "lora-klein" / "training" / "configs" / "haiku-4.5" / "attack_on_titan_holdout_sample_ids.txt"
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def summarize_layout_colors(manifest_path: Path, *, max_rows: int = 256) -> dict[str, Any]:
    expected = set(TEXT_COLORS.values())
    observed: dict[tuple[int, int, int], int] = {}
    checked = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if checked >= max_rows:
                break
            row = json.loads(line)
            layout_ref = ((row.get("controls") or {}).get("layout_control_path") or "").strip()
            if not layout_ref or layout_ref.startswith("s3://"):
                continue
            path = Path(layout_ref)
            if not path.exists():
                continue
            with Image.open(path).convert("RGB") as image:
                for color, count in image.getcolors(maxcolors=4096) or []:
                    if count <= 0:
                        continue
                    observed[color] = observed.get(color, 0) + int(count)
            checked += 1
    present = {name: observed.get(rgb, 0) for name, rgb in TEXT_COLORS.items()}
    unexpected_text_shape_colors = {
        ",".join(str(value) for value in rgb): count
        for rgb, count in observed.items()
        if rgb not in expected
        and rgb
        not in {
            (255, 255, 255),
            (255, 0, 0),
            (0, 180, 0),
            (255, 220, 0),
            (255, 0, 255),
            (0, 220, 255),
        }
    }
    return {
        "checked_layout_images": checked,
        "expected_text_color_pixels": present,
        "unexpected_non_palette_colors": unexpected_text_shape_colors,
    }


def export_peft(
    *,
    repo_root: Path,
    cache_root: Path,
    output_root: Path,
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
    run([sys.executable, "-c", code, JOB_NAME, str(prepared_config)], env=export_env, log_path=log_path)


def sync_artifacts(output_root: Path, log_path: Path, status_path: Path) -> None:
    save_root = output_root / JOB_NAME
    model_prefix = f"s3://{S3_BUCKET}/models/{JOB_NAME}"
    if save_root.exists():
        run(["aws", "s3", "sync", str(save_root), f"{model_prefix}/checkpoints", "--only-show-errors"], log_path=log_path)
    if (save_root / "peft_adapter").exists():
        run(["aws", "s3", "sync", str(save_root / "peft_adapter"), f"{model_prefix}/final/peft_adapter", "--only-show-errors"], log_path=log_path)
    if (save_root / "config.yaml").exists():
        run(["aws", "s3", "cp", str(save_root / "config.yaml"), f"{model_prefix}/config.yaml", "--only-show-errors"], log_path=log_path)
    if status_path.exists():
        run(["aws", "s3", "cp", str(status_path), f"{model_prefix}/status.json", "--only-show-errors"], log_path=log_path)
    run(["aws", "s3", "cp", str(log_path), f"{model_prefix}/logs/train.log", "--only-show-errors"], log_path=log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/ubuntu/drawtoon")
    parser.add_argument("--work-root", default="/mnt/local/drawtoon-aot-textcolor")
    parser.add_argument("--cache-root", default="/mnt/local/training/datasets_cache")
    parser.add_argument("--hf-home", default="/mnt/local/drawtoon-b300/hf")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3500)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_root = Path(args.work_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    config_path = work_root / "config.yaml"
    prepared_config = work_root / "prepared_ai_toolkit.yaml"
    output_root = work_root / "output"
    log_path = work_root / "train.log"
    status_path = work_root / "status.json"

    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"[{utc_now()}] Starting {JOB_NAME} on GPU {args.gpu}\n", encoding="utf-8")
    status = {"job_name": JOB_NAME, "title": TITLE, "status": "running", "started_at": utc_now(), "gpu": args.gpu}
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    write_config(repo_root, config_path)

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
        }
    )

    try:
        cache_proc = run(
            [
                sys.executable,
                str(repo_root / "lora-klein" / "training" / "utils.py"),
                "build-ec2-cache",
                "--config",
                str(config_path),
                "--cache-root",
                str(cache_root),
                "--workers",
                str(args.cache_workers),
            ],
            env=env,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(cache_proc.stdout)
        cache_summary = parse_last_json(cache_proc.stdout)
        manifest_path = Path(cache_summary["manifest_path"])
        resolved_config = str(cache_summary["resolved_config_path"])

        color_summary = summarize_layout_colors(manifest_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[layout_color_summary]\n")
            handle.write(json.dumps(color_summary, indent=2, sort_keys=True))
            handle.write("\n")

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
                str(manifest_path),
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
                JOB_NAME,
            ],
            env=env,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(prepare_proc.stdout)

        run(
            [
                "torchrun",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--master-addr=127.0.0.1",
                f"--master-port={29800 + int(args.gpu)}",
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
            prepared_config=prepared_config,
            env=env,
            log_path=log_path,
        )
        status.update({"status": "completed", "finished_at": utc_now(), "s3": f"s3://{S3_BUCKET}/models/{JOB_NAME}/"})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        sync_artifacts(output_root, log_path, status_path)
    except Exception as exc:
        status.update({"status": "failed", "finished_at": utc_now(), "error": repr(exc)})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        sync_artifacts(output_root, log_path, status_path)
        raise


if __name__ == "__main__":
    main()
