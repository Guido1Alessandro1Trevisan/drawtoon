#!/usr/bin/env python3
"""CLI launcher for the build_panel_dataset workflow."""

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


load_dotenv(Path(__file__).resolve().parents[4] / ".env")
load_dotenv(Path.cwd() / ".env")

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET_NAME", "drawtoon")
DEFAULT_OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET_NAME", "drawtoon-layousyn")
DEFAULT_SOURCE_PREFIX = "captions"
DEFAULT_SOURCE_RUN = "gemini3_flash_page_panel_v1"
DEFAULT_SHORT_PREFIX = "captions_short"
DEFAULT_SHORT_RUN = "vision_v1"
DEFAULT_OUTPUT_PREFIX = "datasets/panel_layout"


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
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def default_execution_name(prefix: str) -> str:
    normalized = prefix.replace("/", "-").replace("_", "-").replace(".", "-").strip("-")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{(normalized or 'build-panel-dataset')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=3000)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    cmd = subparsers.add_parser("build")
    cmd.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    cmd.add_argument("--short-bucket", default=DEFAULT_OUTPUT_BUCKET)
    cmd.add_argument("--output-bucket", default=DEFAULT_OUTPUT_BUCKET)
    cmd.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    cmd.add_argument("--source-caption-run", default=DEFAULT_SOURCE_RUN)
    cmd.add_argument("--short-prefix", default=DEFAULT_SHORT_PREFIX)
    cmd.add_argument("--short-caption-run", default=DEFAULT_SHORT_RUN)
    cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    cmd.add_argument("--include-chapter-regex", default="")
    cmd.add_argument("--max-pages", type=int, default=0)
    cmd.add_argument("--split-val-frac", type=float, default=0.05)
    cmd.add_argument("--run-id", default="")
    cmd.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command != "build":
        raise ValueError(f"Unsupported command={args.command!r}")

    session = boto3_session(str(args.profile or "").strip())
    outputs = cloudformation_output_map(session, args.stack_name)
    payload: dict[str, Any] = {
        "job_name": str(args.job_name or "").strip(),
        "max_concurrency": max(1, int(args.max_concurrency)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "git_sha": infer_git_sha(),
        "source_bucket": str(args.source_bucket).strip(),
        "short_bucket": str(args.short_bucket).strip(),
        "output_bucket": str(args.output_bucket).strip(),
        "source_prefix": str(args.source_prefix).strip().strip("/"),
        "source_caption_run": str(args.source_caption_run).strip().strip("/"),
        "short_prefix": str(args.short_prefix).strip().strip("/"),
        "short_caption_run": str(args.short_caption_run).strip().strip("/"),
        "output_prefix": str(args.output_prefix).strip().strip("/"),
        "include_chapter_regex": str(args.include_chapter_regex or "").strip(),
        "max_pages": max(0, int(args.max_pages)),
        "split_val_frac": float(args.split_val_frac),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
    }
    arn = outputs["BuildPanelDatasetStateMachineArn"]
    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    client = session.client("stepfunctions", region_name=DEFAULT_REGION)
    response = client.start_execution(
        stateMachineArn=arn,
        name=default_execution_name(args.job_name or "build-panel-dataset"),
        input=json.dumps(payload, ensure_ascii=False),
    )
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
