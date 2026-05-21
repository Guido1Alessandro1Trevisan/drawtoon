#!/usr/bin/env python3
"""Launcher for the Drawtoon manga_change_of_angle workflow.

Mirrors workflows/manga_caption/start.py — looks up the deployed SAM stack's
state-machine ARN from CloudFormation outputs, then submits a SFN execution.
The Distributed Map inside the state machine fans page-level Kimi K2.6 calls
across Lambda. Reasoning is toggled per-execution via ``--thinking-enabled``
(default ON; pass ``--no-thinking`` for OFF).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import boto3


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path.cwd() / ".env")

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/change_angle"
DEFAULT_CHANGE_ANGLE_RUN = os.environ.get("DEFAULT_CHANGE_ANGLE_RUN", "kimi_k26_v1")
DEFAULT_KIMI_MODEL = os.environ.get("DEFAULT_KIMI_MODEL", "kimi-k2.6")


def boto3_session(profile: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=DEFAULT_REGION)
    return boto3.Session(region_name=DEFAULT_REGION)


def cloudformation_output_map(session: boto3.Session, stack_name: str) -> dict[str, str]:
    client = session.client("cloudformation", region_name=DEFAULT_REGION)
    stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]
    return {
        str(output.get("OutputKey")): str(output.get("OutputValue"))
        for output in stack.get("Outputs", [])
        if output.get("OutputKey") and output.get("OutputValue")
    }


def infer_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


def default_execution_name(prefix: str) -> str:
    normalized = prefix.replace("/", "-").replace("_", "-").replace(".", "-").strip("-")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{(normalized or 'change-angle')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    detect_cmd = subparsers.add_parser("detect-pages")
    detect_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    detect_cmd.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    detect_cmd.add_argument("--annotation-prefix", default=DEFAULT_ANNOTATION_PREFIX)
    detect_cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    detect_cmd.add_argument(
        "--input-manifest-key",
        default="",
        help=(
            "Optional existing JSONL page manifest key to retry. "
            "Existing outputs are still skipped unless --overwrite is passed."
        ),
    )
    detect_cmd.add_argument(
        "--change-angle-run",
        default=DEFAULT_CHANGE_ANGLE_RUN,
        help="Sub-prefix under --output-prefix where outputs land (one per page).",
    )
    detect_cmd.add_argument("--model", default=DEFAULT_KIMI_MODEL)
    detect_cmd.add_argument(
        "--thinking-enabled",
        dest="thinking_enabled",
        action="store_true",
        default=True,
        help="Run Kimi K2.6 with reasoning enabled (default).",
    )
    detect_cmd.add_argument(
        "--no-thinking",
        dest="thinking_enabled",
        action="store_false",
        help="Run Kimi K2.6 with reasoning disabled.",
    )
    detect_cmd.add_argument("--include-chapter-regex", default="")
    detect_cmd.add_argument("--max-pages", type=int, default=0)
    detect_cmd.add_argument(
        "--sampling-strategy",
        choices=("sequential", "round_robin_chapters"),
        default="sequential",
        help=(
            "How PrepareConfig selects pages when --max-pages is set. "
            "round_robin_chapters spreads the cap across chapters."
        ),
    )
    detect_cmd.add_argument(
        "--shuffle-seed",
        default="",
        help="Optional stable seed for round_robin_chapters ordering.",
    )
    detect_cmd.add_argument("--run-id", default="")
    detect_cmd.add_argument("--overwrite", action="store_true")
    detect_cmd.add_argument("--allow-missing-annotations", action="store_true")

    args = parser.parse_args()
    if args.command != "detect-pages":
        raise ValueError(f"Unsupported command={args.command!r}")

    session = boto3_session(str(args.profile or "").strip())
    outputs = cloudformation_output_map(session, args.stack_name)
    payload: dict[str, Any] = {
        "job_name": str(args.job_name or "").strip(),
        "max_concurrency": max(1, int(args.max_concurrency)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "git_sha": infer_git_sha(),
        "bucket": str(args.bucket).strip(),
        "source_prefix": str(args.source_prefix).strip().strip("/"),
        "annotation_prefix": str(args.annotation_prefix).strip().strip("/"),
        "output_prefix": str(args.output_prefix).strip().strip("/"),
        "input_manifest_key": str(args.input_manifest_key or "").strip().strip("/"),
        "change_angle_run": str(args.change_angle_run or "").strip(),
        "model": str(args.model).strip(),
        "thinking_enabled": bool(args.thinking_enabled),
        "include_chapter_regex": str(args.include_chapter_regex or "").strip(),
        "max_pages": max(0, int(args.max_pages)),
        "sampling_strategy": str(args.sampling_strategy or "sequential").strip(),
        "shuffle_seed": str(args.shuffle_seed or "").strip(),
        "require_annotations": not bool(args.allow_missing_annotations),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
    }
    arn = outputs["MangaChangeOfAngleStateMachineArn"]
    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    client = session.client("stepfunctions", region_name=DEFAULT_REGION)
    response = client.start_execution(
        stateMachineArn=arn,
        name=default_execution_name(
            args.job_name or args.change_angle_run or "change-of-angle"
        ),
        input=json.dumps(payload, ensure_ascii=False),
    )
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
