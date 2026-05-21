#!/usr/bin/env python3
"""Start a page-annotation Distributed Map execution.

Reads the deployed stack's state-machine ARN, builds an input payload, and
calls StartExecution.

Usage:
    python3 start.py --stack-name drawtoon-annotate-pages --profile default annotate
    python3 start.py --stack-name drawtoon-annotate-pages --profile default annotate \\
        --include-chapter-regex '_mangazero_manga$' --max-pages 20 --max-concurrency 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import boto3


DEFAULT_SOURCE_BUCKET = "drawtoon"
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_OUTPUT_RUN = "haiku_page_panel_v1"


def _resolve_state_machine_arn(cfn, stack_name: str) -> str:
    resp = cfn.describe_stack_resources(StackName=stack_name)
    for r in resp.get("StackResources", []):
        if r.get("LogicalResourceId") == "AnnotatePagesStateMachine":
            arn = r.get("PhysicalResourceId")
            if arn:
                return arn
    raise SystemExit(f"AnnotatePagesStateMachine not found in stack {stack_name!r}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Start AnnotatePages execution")
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--dry-run", action="store_true")

    sub = parser.add_subparsers(dest="cmd", required=True)
    cmd = sub.add_parser("annotate")
    cmd.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    cmd.add_argument("--input-prefix", default="captions/gemini3_flash_page_panel_v1",
                     help="Where to enumerate per-page caption JSONs from (Gemini v1).")
    cmd.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    cmd.add_argument("--output-run", default=DEFAULT_OUTPUT_RUN)
    cmd.add_argument("--model", default="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    cmd.add_argument("--include-chapter-regex", default="")
    cmd.add_argument("--max-pages", type=int, default=0)
    cmd.add_argument("--max-concurrency", type=int, default=1000)
    cmd.add_argument("--tolerated-failure-count", type=int, default=0)
    cmd.add_argument("--overwrite", action="store_true")
    cmd.add_argument("--run-id", default="")

    args = parser.parse_args(argv)
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    cfn = session.client("cloudformation", region_name=args.region)
    sf = session.client("stepfunctions", region_name=args.region)

    arn = _resolve_state_machine_arn(cfn, args.stack_name)
    payload = {
        "source_bucket": args.source_bucket,
        "input_prefix": args.input_prefix,
        "output_prefix": args.output_prefix,
        "output_run": args.output_run,
        "model": args.model,
        "include_chapter_regex": args.include_chapter_regex,
        "max_pages": args.max_pages,
        "max_concurrency": args.max_concurrency,
        "tolerated_failure_count": args.tolerated_failure_count,
        "overwrite": args.overwrite,
        "run_id": args.run_id,
    }
    job_name = args.job_name or (
        f"{args.output_run.replace('_', '-')}-{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    )
    body = {
        "stateMachineArn": arn,
        "name": job_name,
        "input": json.dumps(payload),
    }

    print(json.dumps({"input": payload, "execution_name": job_name}, indent=2))
    if args.dry_run:
        print("[dry-run] not starting execution.", file=sys.stderr)
        return 0
    resp = sf.start_execution(**body)
    print(json.dumps({k: str(v) for k, v in resp.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
