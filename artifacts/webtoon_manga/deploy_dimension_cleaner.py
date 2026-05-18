#!/usr/bin/env python3
"""Deploy and optionally start the webtoon_manga dimension cleanup workflow."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "dimension_clean_worker.py"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_SOURCE_PREFIX = "datasets/pages/source/webtoon"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/source/webtoon_cleaned"
DEFAULT_STATUS_PREFIX = "datasets/pages/source/webtoon_cleaned/_status"
DEFAULT_MANIFEST = ROOT / "manifest/webtoon_episodes.jsonl"
FUNCTION_NAME = "webtoon_manga_dimension_cleaner_v1"
STATE_MACHINE_NAME = "webtoon_manga_dimension_cleaner_map"
LAMBDA_ROLE_NAME = "webtoon_manga_dimension_cleaner_lambda_role"
SFN_ROLE_NAME = "webtoon_manga_dimension_cleaner_sfn_role"


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_role(iam, *, name: str, service: str, policy_name: str, policy_doc: dict[str, Any]) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
    }
    try:
        role = iam.get_role(RoleName=name)["Role"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]
        time.sleep(8)
    iam.put_role_policy(RoleName=name, PolicyName=policy_name, PolicyDocument=json.dumps(policy_doc))
    return role["Arn"]


def lambda_policy(bucket: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": "arn:aws:logs:*:*:*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
        ],
    }


def sfn_policy(bucket: str, lambda_arn: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": [lambda_arn, f"{lambda_arn}:*"]},
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
            {"Effect": "Allow", "Action": ["states:StartExecution", "states:DescribeExecution", "states:StopExecution"], "Resource": "*"},
        ],
    }


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(WORKER, "dimension_clean_worker.py")
    return buffer.getvalue()


def ensure_lambda(lambda_client, *, role_arn: str, args: argparse.Namespace) -> str:
    env = {
        "DEST_BUCKET": args.bucket,
        "SOURCE_PREFIX": args.source_prefix,
        "OUTPUT_PREFIX": args.output_prefix,
        "STATUS_PREFIX": args.status_prefix,
        "DIMENSION_WORKERS": str(args.dimension_workers),
        "COPY_WORKERS": str(args.copy_workers),
        "EDGE_HEAD": str(args.edge_head),
        "EDGE_TAIL": str(args.edge_tail),
        "MIN_WIDTH": str(args.min_width),
        "MIN_HEIGHT": str(args.min_height),
        "MIN_RATIO": str(args.min_ratio),
        "MAX_RATIO": str(args.max_ratio),
        "WIDTH_TOLERANCE_RATIO": str(args.width_tolerance_ratio),
        "WIDTH_TOLERANCE_PX": str(args.width_tolerance_px),
    }
    zip_bytes = build_zip()
    try:
        current = lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes, Publish=True)
        lambda_client.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Runtime=args.runtime,
            Handler="dimension_clean_worker.handler",
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment={"Variables": env},
        )
        function_arn = current["FunctionArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=args.runtime,
            Role=role_arn,
            Handler="dimension_clean_worker.handler",
            Code={"ZipFile": zip_bytes},
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment={"Variables": env},
            Publish=True,
        )
        function_arn = response["FunctionArn"]
    lambda_client.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
    if args.reserved_concurrency:
        lambda_client.put_function_concurrency(
            FunctionName=FUNCTION_NAME,
            ReservedConcurrentExecutions=args.reserved_concurrency,
        )
    return function_arn


def state_machine_definition(
    *,
    lambda_arn: str,
    bucket: str,
    manifest_key: str,
    source_prefix: str,
    output_prefix: str,
    status_prefix: str,
    audit_prefix: str,
    max_concurrency: int,
    tolerated_failure_percentage: float,
    dry_run: bool,
) -> str:
    definition = {
        "Comment": "Dimension-first cleanup of downloaded webtoon_manga episode images.",
        "StartAt": "CleanEpisodes",
        "States": {
            "CleanEpisodes": {
                "Type": "Map",
                "MaxConcurrency": max_concurrency,
                "ToleratedFailurePercentage": tolerated_failure_percentage,
                "ItemReader": {
                    "Resource": "arn:aws:states:::s3:getObject",
                    "ReaderConfig": {"InputType": "JSONL"},
                    "Parameters": {"Bucket": bucket, "Key": manifest_key},
                },
                "ItemSelector": {
                    "episode.$": "$$.Map.Item.Value.episode",
                    "bucket": bucket,
                    "source_prefix": source_prefix,
                    "output_prefix": output_prefix,
                    "status_prefix": status_prefix,
                    "dry_run": dry_run,
                },
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "EXPRESS"},
                    "StartAt": "CleanEpisode",
                    "States": {
                        "CleanEpisode": {
                            "Type": "Task",
                            "Resource": "arn:aws:states:::lambda:invoke",
                            "OutputPath": "$.Payload",
                            "Parameters": {"FunctionName": lambda_arn, "Payload.$": "$"},
                            "Retry": [
                                {
                                    "ErrorEquals": [
                                        "Lambda.TooManyRequestsException",
                                        "Lambda.ServiceException",
                                        "Lambda.AWSLambdaException",
                                        "Lambda.SdkClientException",
                                    ],
                                    "IntervalSeconds": 2,
                                    "MaxAttempts": 6,
                                    "BackoffRate": 2,
                                }
                            ],
                            "End": True,
                        }
                    },
                },
                "ResultWriter": {
                    "Resource": "arn:aws:states:::s3:putObject",
                    "Parameters": {"Bucket": bucket, "Prefix": audit_prefix},
                },
                "End": True,
            }
        },
        "TimeoutSeconds": 7200,
    }
    return json.dumps(definition, indent=2, sort_keys=True)


def ensure_state_machine(sfn, *, name: str, role_arn: str, definition: str) -> str:
    existing_arn = None
    for page in sfn.get_paginator("list_state_machines").paginate():
        for state_machine in page.get("stateMachines", []):
            if state_machine["name"] == name:
                existing_arn = state_machine["stateMachineArn"]
                break
        if existing_arn:
            break
    if existing_arn:
        sfn.update_state_machine(stateMachineArn=existing_arn, definition=definition, roleArn=role_arn)
        return existing_arn
    return sfn.create_state_machine(name=name, definition=definition, roleArn=role_arn, type="STANDARD")["stateMachineArn"]


def first_manifest_item(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty manifest: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--status-prefix", default=DEFAULT_STATUS_PREFIX)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--manifest-key")
    parser.add_argument("--skip-manifest-upload", action="store_true")
    parser.add_argument("--audit-prefix")
    parser.add_argument("--max-concurrency", type=int, default=1000)
    parser.add_argument("--tolerated-failure-percentage", type=float, default=1.0)
    parser.add_argument("--dimension-workers", type=int, default=24)
    parser.add_argument("--copy-workers", type=int, default=24)
    parser.add_argument("--edge-head", type=int, default=40)
    parser.add_argument("--edge-tail", type=int, default=80)
    parser.add_argument("--min-width", type=int, default=100)
    parser.add_argument("--min-height", type=int, default=80)
    parser.add_argument("--min-ratio", type=float, default=0.62)
    parser.add_argument("--max-ratio", type=float, default=7.5)
    parser.add_argument("--width-tolerance-ratio", type=float, default=0.08)
    parser.add_argument("--width-tolerance-px", type=int, default=28)
    parser.add_argument("--lambda-timeout", type=int, default=900)
    parser.add_argument("--lambda-memory", type=int, default=1536)
    parser.add_argument("--runtime", default="python3.11")
    parser.add_argument("--reserved-concurrency", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    run_id = now_id()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_key = args.manifest_key or f"datasets/pages/manifests/webtoon_manga_dimension_cleaner/{run_id}/episodes.jsonl"
    audit_prefix = args.audit_prefix or f"workflows/webtoon_manga/dimension_cleaner/audit/{run_id}/"

    session = boto3.Session(region_name=args.region)
    client_config = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 5, "mode": "adaptive"})
    iam = session.client("iam", config=client_config)
    lambda_client = session.client("lambda", config=client_config)
    sfn = session.client("stepfunctions", config=client_config)
    s3 = session.client("s3", config=client_config)

    print("ensuring lambda role", flush=True)
    lambda_role_arn = ensure_role(
        iam,
        name=LAMBDA_ROLE_NAME,
        service="lambda.amazonaws.com",
        policy_name="webtoon_manga_dimension_cleaner_lambda_policy",
        policy_doc=lambda_policy(args.bucket),
    )
    print("deploying lambda", flush=True)
    function_arn = ensure_lambda(lambda_client, role_arn=lambda_role_arn, args=args)
    print("ensuring state-machine role", flush=True)
    sfn_role_arn = ensure_role(
        iam,
        name=SFN_ROLE_NAME,
        service="states.amazonaws.com",
        policy_name="webtoon_manga_dimension_cleaner_sfn_policy",
        policy_doc=sfn_policy(args.bucket, function_arn),
    )

    if args.skip_manifest_upload:
        print(f"reusing manifest s3://{args.bucket}/{manifest_key}", flush=True)
    else:
        print(f"uploading manifest s3://{args.bucket}/{manifest_key}", flush=True)
        s3.put_object(
            Bucket=args.bucket,
            Key=manifest_key,
            Body=manifest_path.read_bytes(),
            ContentType="application/x-ndjson; charset=utf-8",
        )
    print("creating/updating state machine", flush=True)
    definition = state_machine_definition(
        lambda_arn=function_arn,
        bucket=args.bucket,
        manifest_key=manifest_key,
        source_prefix=args.source_prefix.strip("/"),
        output_prefix=args.output_prefix.strip("/"),
        status_prefix=args.status_prefix.strip("/"),
        audit_prefix=audit_prefix,
        max_concurrency=max(1, int(args.max_concurrency)),
        tolerated_failure_percentage=float(args.tolerated_failure_percentage),
        dry_run=bool(args.dry_run),
    )
    state_machine_arn = ensure_state_machine(sfn, name=STATE_MACHINE_NAME, role_arn=sfn_role_arn, definition=definition)

    print("running dry-run smoke invoke", flush=True)
    smoke_payload = {
        **first_manifest_item(manifest_path),
        "bucket": args.bucket,
        "source_prefix": args.source_prefix,
        "output_prefix": args.output_prefix,
        "status_prefix": args.status_prefix,
        "dry_run": True,
        "max_images": 80,
    }
    response = lambda_client.invoke(
        FunctionName=function_arn,
        InvocationType="RequestResponse",
        Payload=json.dumps(smoke_payload).encode("utf-8"),
    )
    body = response["Payload"].read().decode("utf-8", errors="replace")
    smoke = json.loads(body) if body else {}
    if response.get("FunctionError"):
        raise RuntimeError(json.dumps(smoke, sort_keys=True))
    print(f"smoke=dry-run ok {json.dumps(smoke, sort_keys=True)}", flush=True)
    print(f"manifest=s3://{args.bucket}/{manifest_key}", flush=True)
    print(f"audit=s3://{args.bucket}/{audit_prefix}", flush=True)
    print(f"output=s3://{args.bucket}/{args.output_prefix.strip('/')}/", flush=True)
    print(f"status=s3://{args.bucket}/{args.status_prefix.strip('/')}/", flush=True)
    print(f"lambda={function_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    if args.start:
        execution = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"webtoon-manga-clean-{run_id}",
            input=json.dumps({"run_id": run_id}),
        )
        print(f"execution={execution['executionArn']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
