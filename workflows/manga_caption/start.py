#!/usr/bin/env python3
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
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_CAPTION_RUN = "haiku45_mangazero_panel_simple_v1"
DEFAULT_CAPTION_MODEL = os.environ.get(
    "DEFAULT_CAPTION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_MANGA_METADATA_JSON = os.environ.get(
    "MANGA_METADATA_JSON",
    "metadata/mangazero_manga_credits_20260511.json",
)


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
    return f"{(normalized or 'manga-caption')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=400)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    caption_cmd = subparsers.add_parser("caption-pages")
    caption_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    caption_cmd.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    caption_cmd.add_argument("--annotation-prefix", default=DEFAULT_ANNOTATION_PREFIX)
    caption_cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    caption_cmd.add_argument("--caption-run", default=DEFAULT_CAPTION_RUN)
    caption_cmd.add_argument("--model", default=DEFAULT_CAPTION_MODEL)
    caption_cmd.add_argument("--manga-metadata-json", default=DEFAULT_MANGA_METADATA_JSON)
    caption_cmd.add_argument("--timeout-seconds", type=float, default=180.0)
    caption_cmd.add_argument("--retries", type=int, default=3)
    caption_cmd.add_argument("--max-output-tokens", type=int, default=512)
    caption_cmd.add_argument("--include-chapter-regex", default="")
    caption_cmd.add_argument("--max-pages", type=int, default=0)
    caption_cmd.add_argument("--run-id", default="")
    caption_cmd.add_argument("--overwrite", action="store_true")
    caption_cmd.add_argument("--allow-missing-annotations", action="store_true")

    args = parser.parse_args()
    if args.command != "caption-pages":
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
        "caption_run": str(args.caption_run or "").strip(),
        "model": str(args.model).strip(),
        "manga_metadata_json": str(args.manga_metadata_json or "").strip(),
        "timeout_seconds": float(args.timeout_seconds),
        "retries": int(args.retries),
        "max_output_tokens": int(args.max_output_tokens),
        "include_chapter_regex": str(args.include_chapter_regex or "").strip(),
        "max_pages": max(0, int(args.max_pages)),
        "require_annotations": not bool(args.allow_missing_annotations),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
    }
    arn = outputs["MangaCaptionStateMachineArn"]
    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    client = session.client("stepfunctions", region_name=DEFAULT_REGION)
    response = client.start_execution(
        stateMachineArn=arn,
        name=default_execution_name(args.job_name or args.caption_run or "caption-manga-pages"),
        input=json.dumps(payload, ensure_ascii=False),
    )
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
