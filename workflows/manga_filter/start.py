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
DEFAULT_INPUT_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/filtered"
DEFAULT_FILTER_MODE = "manga"
DEFAULT_CLASSIFICATION_MODEL = os.environ.get(
    "DEFAULT_CLASSIFICATION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_MAX_CONCURRENCY = 64


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
    return f"{(normalized or 'manga-filter')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-items-per-batch", type=int, default=4)
    parser.add_argument("--max-input-bytes-per-batch", type=int, default=131072)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_cmd = subparsers.add_parser("filter-pages")
    filter_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    filter_cmd.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    filter_cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    filter_cmd.add_argument("--mode", choices=["manga", "manhwa", "manwa", "webtoon"], default=DEFAULT_FILTER_MODE)
    filter_cmd.add_argument("--model", default=DEFAULT_CLASSIFICATION_MODEL)
    filter_cmd.add_argument("--prompt-filename", default="")
    filter_cmd.add_argument("--timeout-seconds", type=float, default=60.0)
    filter_cmd.add_argument("--retries", type=int, default=5)
    filter_cmd.add_argument("--max-output-tokens", type=int, default=0)
    filter_cmd.add_argument("--manhwa-diagnostic-width", type=int, default=896)
    filter_cmd.add_argument("--manhwa-full-thumb-height", type=int, default=640)
    filter_cmd.add_argument("--manhwa-edge-crop-source-px", type=int, default=1536)
    filter_cmd.add_argument("--manhwa-jpeg-quality", type=int, default=82)
    filter_cmd.add_argument("--manhwa-chain-confidence-threshold", type=float, default=0.75)
    filter_cmd.add_argument("--include-relative-path-regex", default="")
    filter_cmd.add_argument("--run-id", default="")
    filter_cmd.add_argument("--overwrite", action="store_true")
    filter_cmd.add_argument(
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
        "max_items_per_batch": max(1, int(args.max_items_per_batch)),
        "max_input_bytes_per_batch": max(1024, int(args.max_input_bytes_per_batch)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "git_sha": infer_git_sha(),
    }

    if args.command != "filter-pages":
        raise ValueError(f"Unsupported command={args.command!r}")

    arn = outputs["FilterMangaPagesStateMachineArn"]
    payload = {
        **common,
        "bucket": str(args.bucket).strip(),
        "input_prefix": str(args.input_prefix).strip().strip("/"),
        "output_prefix": str(args.output_prefix).strip().strip("/"),
        "mode": str(args.mode).strip(),
        "model": str(args.model).strip(),
        "prompt_filename": str(args.prompt_filename).strip(),
        "timeout_seconds": float(args.timeout_seconds),
        "retries": int(args.retries),
        "max_output_tokens": int(args.max_output_tokens),
        "manhwa_diagnostic_width": int(args.manhwa_diagnostic_width),
        "manhwa_full_thumb_height": int(args.manhwa_full_thumb_height),
        "manhwa_edge_crop_source_px": int(args.manhwa_edge_crop_source_px),
        "manhwa_jpeg_quality": int(args.manhwa_jpeg_quality),
        "manhwa_chain_confidence_threshold": float(args.manhwa_chain_confidence_threshold),
        "include_relative_path_regex": str(args.include_relative_path_regex or "").strip(),
        "bedrock_preflight": not bool(args.no_bedrock_preflight),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
    }
    execution_name = default_execution_name(common["job_name"] or "filter-manga-pages")

    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    response = start_execution(session, arn, payload, execution_name)
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
