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
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path.cwd() / ".env")

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_CAPTION_MODEL = os.environ.get(
    "DEFAULT_CAPTION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_MAX_CONCURRENCY = 8


def boto3_session(profile: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=DEFAULT_REGION)
    return boto3.Session(region_name=DEFAULT_REGION)


def cloudformation_output_map(session: boto3.Session, stack_name: str) -> dict[str, str]:
    client = session.client("cloudformation", region_name=DEFAULT_REGION)
    stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs: dict[str, str] = {}
    for output in stack.get("Outputs", []):
        key = str(output.get("OutputKey") or "").strip()
        value = str(output.get("OutputValue") or "").strip()
        if key and value:
            outputs[key] = value
    return outputs


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


def start_execution(
    session: boto3.Session,
    state_machine_arn: str,
    payload: dict[str, Any],
    execution_name: str,
) -> dict[str, Any]:
    client = session.client("stepfunctions", region_name=DEFAULT_REGION)
    return client.start_execution(
        stateMachineArn=state_machine_arn,
        name=execution_name,
        input=json.dumps(payload, ensure_ascii=False),
    )


def default_execution_name(prefix: str) -> str:
    normalized = prefix.replace("/", "-").replace("_", "-").replace(".", "-").strip("-")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{(normalized or 'manga-caption')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    caption_cmd = subparsers.add_parser("caption-pages")
    caption_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    caption_cmd.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    caption_cmd.add_argument("--annotation-prefix", default=DEFAULT_ANNOTATION_PREFIX)
    caption_cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    caption_cmd.add_argument("--caption-run", default="")
    caption_cmd.add_argument("--model", default=DEFAULT_CAPTION_MODEL)
    caption_cmd.add_argument("--prompt-filename", default="caption_manga_page_memory.md")
    caption_cmd.add_argument(
        "--manga-metadata-json",
        default="",
        help="Optional manga title/mangaka JSON. Use s3://... or a workflow-relative file such as metadata/manga_metadata.json.",
    )
    caption_cmd.add_argument("--timeout-seconds", type=float, default=180.0)
    caption_cmd.add_argument("--retries", type=int, default=3)
    caption_cmd.add_argument("--max-output-tokens", type=int, default=4096)
    caption_cmd.add_argument("--include-chapter-regex", default="")
    caption_cmd.add_argument("--side-order", choices=["rtl", "ltr", "key"], default="rtl")
    caption_cmd.add_argument("--run-id", default="")
    caption_cmd.add_argument("--overwrite", action="store_true")
    caption_cmd.add_argument("--allow-missing-annotations", action="store_true")
    caption_cmd.add_argument(
        "--no-bedrock-preflight",
        action="store_true",
        help="Skip the one-call Bedrock model access probe before Distributed Map fanout.",
    )

    args = parser.parse_args()
    session = boto3_session(str(args.profile or "").strip())
    outputs = cloudformation_output_map(session, args.stack_name)
    common = {
        "job_name": str(args.job_name or "").strip(),
        "max_concurrency": max(1, int(args.max_concurrency)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "git_sha": infer_git_sha(),
    }

    if args.command != "caption-pages":
        raise ValueError(f"Unsupported command={args.command!r}")

    arn = outputs["MangaCaptionStateMachineArn"]
    payload = {
        **common,
        "bucket": str(args.bucket).strip(),
        "source_prefix": str(args.source_prefix).strip().strip("/"),
        "annotation_prefix": str(args.annotation_prefix).strip().strip("/"),
        "output_prefix": str(args.output_prefix).strip().strip("/"),
        "caption_run": str(args.caption_run or "").strip(),
        "model": str(args.model).strip(),
        "prompt_filename": str(args.prompt_filename).strip(),
        "manga_metadata_json": str(args.manga_metadata_json or "").strip(),
        "timeout_seconds": float(args.timeout_seconds),
        "retries": int(args.retries),
        "max_output_tokens": int(args.max_output_tokens),
        "include_chapter_regex": str(args.include_chapter_regex or "").strip(),
        "side_order": str(args.side_order or "rtl").strip(),
        "require_annotations": not bool(args.allow_missing_annotations),
        "bedrock_preflight": not bool(args.no_bedrock_preflight),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
    }
    execution_name = default_execution_name(common["job_name"] or "caption-manga-pages")

    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    response = start_execution(session, arn, payload, execution_name)
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
