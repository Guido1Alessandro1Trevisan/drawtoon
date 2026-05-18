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
DEFAULT_INPUT_PREFIX = "datasets/pages/filtered"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/text_removed"
DEFAULT_REMOVE_TEXT_RUN = "qwen2511_master_prompt_v1"
DEFAULT_FAL_ENDPOINT = os.environ.get("DEFAULT_FAL_ENDPOINT", "fal-ai/qwen-image-edit-2511")
DEFAULT_MAX_CONCURRENCY = 32


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


def default_execution_name(prefix: str) -> str:
    normalized = prefix.replace("/", "-").replace("_", "-").replace(".", "-").strip("-")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{(normalized or 'remove-text')[:55]}-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--tolerated-failure-count", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    remove_cmd = subparsers.add_parser("remove-pages")
    remove_cmd.add_argument("--bucket", default=DEFAULT_BUCKET)
    remove_cmd.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    remove_cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    remove_cmd.add_argument("--remove-text-run", default=DEFAULT_REMOVE_TEXT_RUN)
    remove_cmd.add_argument("--fal-endpoint", default=DEFAULT_FAL_ENDPOINT)
    remove_cmd.add_argument("--prompt-filename", default="master_prompt.md")
    remove_cmd.add_argument("--timeout-seconds", type=float, default=840.0)
    remove_cmd.add_argument("--request-timeout-seconds", type=float, default=60.0)
    remove_cmd.add_argument("--download-timeout-seconds", type=float, default=180.0)
    remove_cmd.add_argument("--poll-interval-seconds", type=float, default=8.0)
    remove_cmd.add_argument("--presign-expires-seconds", type=int, default=12 * 60 * 60)
    remove_cmd.add_argument("--num-inference-steps", type=int, default=40)
    remove_cmd.add_argument("--guidance-scale", type=float, default=4.5)
    remove_cmd.add_argument("--num-images", type=int, default=1)
    remove_cmd.add_argument("--output-format", default="png")
    remove_cmd.add_argument("--acceleration", default="regular")
    remove_cmd.add_argument("--include-relative-path-regex", default="")
    remove_cmd.add_argument("--max-pages", type=int, default=0)
    remove_cmd.add_argument("--run-id", default="")
    remove_cmd.add_argument("--overwrite", action="store_true")
    remove_cmd.add_argument(
        "--no-fal-preflight",
        action="store_true",
        help="Skip the FAL key presence probe before Distributed Map fanout.",
    )

    args = parser.parse_args()
    if args.command != "remove-pages":
        raise ValueError(f"Unsupported command={args.command!r}")

    session = boto3_session(str(args.profile or "").strip())
    outputs = cloudformation_output_map(session, args.stack_name)
    payload: dict[str, Any] = {
        "job_name": str(args.job_name or "").strip(),
        "max_concurrency": max(1, int(args.max_concurrency)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "git_sha": infer_git_sha(),
        "bucket": str(args.bucket).strip(),
        "input_prefix": str(args.input_prefix).strip().strip("/"),
        "output_prefix": str(args.output_prefix).strip().strip("/"),
        "remove_text_run": str(args.remove_text_run or "").strip().strip("/"),
        "fal_endpoint": str(args.fal_endpoint).strip().strip("/"),
        "prompt_filename": str(args.prompt_filename).strip(),
        "timeout_seconds": float(args.timeout_seconds),
        "request_timeout_seconds": float(args.request_timeout_seconds),
        "download_timeout_seconds": float(args.download_timeout_seconds),
        "poll_interval_seconds": float(args.poll_interval_seconds),
        "presign_expires_seconds": max(60, int(args.presign_expires_seconds)),
        "num_inference_steps": max(1, int(args.num_inference_steps)),
        "guidance_scale": float(args.guidance_scale),
        "num_images": max(1, int(args.num_images)),
        "output_format": str(args.output_format or "png").strip().lower(),
        "acceleration": str(args.acceleration or "regular").strip(),
        "include_relative_path_regex": str(args.include_relative_path_regex or "").strip(),
        "max_pages": max(0, int(args.max_pages)),
        "run_id": str(args.run_id or "").strip(),
        "overwrite": bool(args.overwrite),
        "fal_preflight": not bool(args.no_fal_preflight),
    }
    arn = outputs["RemoveTextPagesStateMachineArn"]
    if args.dry_run:
        print(json.dumps({"state_machine_arn": arn, "input": payload}, indent=2))
        return

    client = session.client("stepfunctions", region_name=DEFAULT_REGION)
    response = client.start_execution(
        stateMachineArn=arn,
        name=default_execution_name(args.job_name or args.remove_text_run or "remove-text-pages"),
        input=json.dumps(payload, ensure_ascii=False),
    )
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
