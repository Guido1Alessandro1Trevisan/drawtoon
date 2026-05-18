#!/usr/bin/env python3
"""Deploy and optionally start direct-to-single WEBTOON/manhwa download."""

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
WORKER = ROOT / "direct_single_worker.py"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_SINGLE_PREFIX = "datasets/pages/single"
DEFAULT_STATUS_PREFIX = "datasets/pages/manifests/webtoon_manga_direct_single/status"
DEFAULT_MANIFEST = ROOT / "manifest/direct_single_series_3000.jsonl"
FUNCTION_NAME = "webtoon_manga_direct_single_worker_v1"
STATE_MACHINE_NAME = "webtoon_manga_direct_single_map"
LAMBDA_ROLE_NAME = "webtoon_manga_direct_single_lambda_role"
SFN_ROLE_NAME = "webtoon_manga_direct_single_sfn_role"
EXISTING_DECODO_FUNCTION = "webtoon_manga_episode_worker_v2"


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
        zf.write(WORKER, "direct_single_worker.py")
    return buffer.getvalue()


def decodo_env(lambda_client) -> dict[str, str]:
    keys = ["DECODO_PROXY_HOST", "DECODO_PROXY_PORTS", "DECODO_PROXY_USER", "DECODO_PROXY_PASS"]
    env: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value:
            env[key] = value
    missing = [key for key in keys if key not in env]
    if not missing:
        return env
    try:
        current = lambda_client.get_function_configuration(FunctionName=EXISTING_DECODO_FUNCTION)
        old_env = (current.get("Environment") or {}).get("Variables") or {}
    except ClientError:
        old_env = {}
    for key in missing:
        value = old_env.get(key)
        if value:
            env[key] = value
    return env


def ensure_lambda(lambda_client, *, role_arn: str, args: argparse.Namespace) -> str:
    env = {
        "DEST_BUCKET": args.bucket,
        "SINGLE_PREFIX": args.single_prefix,
        "STATUS_PREFIX": args.status_prefix,
        "PROXY_MODE": args.proxy_mode,
        "IMAGE_WORKERS": str(args.image_workers),
        "IMAGE_RETRIES": str(args.image_retries),
        "HTML_RETRIES": str(args.html_retries),
        "HTTP_TIMEOUT": str(args.http_timeout),
        "MAX_LIST_PAGES": str(args.max_list_pages),
        "MAX_PAGES_PER_SERIES": str(args.max_pages_per_series),
        "SEED": str(args.seed),
        "MIN_WIDTH": str(args.min_width),
        "MIN_HEIGHT": str(args.min_height),
        "MIN_RATIO": str(args.min_ratio),
        "MAX_RATIO": str(args.max_ratio),
        "WIDTH_TOLERANCE_RATIO": str(args.width_tolerance_ratio),
        "WIDTH_TOLERANCE_PX": str(args.width_tolerance_px),
        **decodo_env(lambda_client),
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
            Handler="direct_single_worker.handler",
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
            Handler="direct_single_worker.handler",
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
    single_prefix: str,
    status_prefix: str,
    audit_prefix: str,
    max_concurrency: int,
    proxy_mode: str,
    max_pages_per_series: int,
    max_list_pages: int,
    seed: int,
    tolerated_failure_percentage: float,
) -> str:
    definition = {
        "Comment": "Direct dimension-filtered WEBTOON/manhwa download into datasets/pages/single.",
        "StartAt": "DownloadSeries",
        "States": {
            "DownloadSeries": {
                "Type": "Map",
                "MaxConcurrency": max_concurrency,
                "ToleratedFailurePercentage": tolerated_failure_percentage,
                "ItemReader": {
                    "Resource": "arn:aws:states:::s3:getObject",
                    "ReaderConfig": {"InputType": "JSONL"},
                    "Parameters": {"Bucket": bucket, "Key": manifest_key},
                },
                "ItemSelector": {
                    "series.$": "$$.Map.Item.Value.series",
                    "bucket": bucket,
                    "single_prefix": single_prefix,
                    "status_prefix": status_prefix,
                    "proxy_mode": proxy_mode,
                    "max_pages": max_pages_per_series,
                    "max_list_pages": max_list_pages,
                    "seed": seed,
                    "run_id.$": "$.run_id",
                    "overwrite": False,
                },
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                    "StartAt": "DownloadSeriesItem",
                    "States": {
                        "DownloadSeriesItem": {
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
                                },
                                {
                                    "ErrorEquals": ["States.TaskFailed"],
                                    "IntervalSeconds": 8,
                                    "MaxAttempts": 2,
                                    "BackoffRate": 2,
                                },
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


def upload_manifest(s3, *, bucket: str, manifest_path: Path, manifest_key: str) -> None:
    s3.upload_file(str(manifest_path), bucket, manifest_key)


def first_manifest_series(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)["series"]
    raise RuntimeError(f"empty manifest: {path}")


def invoke_smoke(
    lambda_client,
    *,
    function_arn: str,
    series: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    payload = {
        "series": series,
        "bucket": args.bucket,
        "single_prefix": args.single_prefix,
        "status_prefix": args.status_prefix,
        "proxy_mode": args.proxy_mode,
        "max_pages": args.smoke_pages,
        "max_list_pages": min(args.max_list_pages, 3),
        "run_id": f"{run_id}-smoke",
        "seed": args.seed,
        "dry_run": True,
    }
    response = lambda_client.invoke(
        FunctionName=function_arn,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    body = response["Payload"].read().decode("utf-8", errors="replace")
    parsed = json.loads(body) if body else {}
    if response.get("FunctionError"):
        raise RuntimeError(json.dumps(parsed, sort_keys=True))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--single-prefix", default=DEFAULT_SINGLE_PREFIX)
    parser.add_argument("--status-prefix", default=DEFAULT_STATUS_PREFIX)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--manifest-key")
    parser.add_argument("--audit-prefix")
    parser.add_argument("--max-concurrency", type=int, default=15)
    parser.add_argument("--proxy-mode", choices=["direct", "proxy", "auto"], default="auto")
    parser.add_argument("--max-pages-per-series", type=int, default=3000)
    parser.add_argument("--max-list-pages", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--tolerated-failure-percentage", type=float, default=25.0)
    parser.add_argument("--image-workers", type=int, default=64)
    parser.add_argument("--image-retries", type=int, default=4)
    parser.add_argument("--html-retries", type=int, default=4)
    parser.add_argument("--http-timeout", type=float, default=45)
    parser.add_argument("--lambda-timeout", type=int, default=900)
    parser.add_argument("--lambda-memory", type=int, default=4096)
    parser.add_argument("--runtime", default="python3.11")
    parser.add_argument("--reserved-concurrency", type=int, default=30)
    parser.add_argument("--smoke-pages", type=int, default=5)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--min-width", type=int, default=100)
    parser.add_argument("--min-height", type=int, default=80)
    parser.add_argument("--min-ratio", type=float, default=0.62)
    parser.add_argument("--max-ratio", type=float, default=7.5)
    parser.add_argument("--width-tolerance-ratio", type=float, default=0.08)
    parser.add_argument("--width-tolerance-px", type=int, default=28)
    args = parser.parse_args()

    run_id = now_id()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_key = args.manifest_key or f"datasets/pages/manifests/webtoon_manga_direct_single/{run_id}/series.jsonl"
    audit_prefix = args.audit_prefix or f"workflows/webtoon_manga/direct_single_audit/{run_id}/"

    session = boto3.Session(region_name=args.region)
    iam = session.client("iam")
    lambda_client = session.client(
        "lambda",
        config=Config(
            connect_timeout=10,
            read_timeout=max(args.lambda_timeout + 90, 120),
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")

    lambda_role_arn = ensure_role(
        iam,
        name=LAMBDA_ROLE_NAME,
        service="lambda.amazonaws.com",
        policy_name="webtoon_manga_direct_single_lambda_policy",
        policy_doc=lambda_policy(args.bucket),
    )
    function_arn = ensure_lambda(lambda_client, role_arn=lambda_role_arn, args=args)
    sfn_role_arn = ensure_role(
        iam,
        name=SFN_ROLE_NAME,
        service="states.amazonaws.com",
        policy_name="webtoon_manga_direct_single_sfn_policy",
        policy_doc=sfn_policy(args.bucket, function_arn),
    )

    upload_manifest(s3, bucket=args.bucket, manifest_path=manifest_path, manifest_key=manifest_key)
    definition = state_machine_definition(
        lambda_arn=function_arn,
        bucket=args.bucket,
        manifest_key=manifest_key,
        single_prefix=args.single_prefix,
        status_prefix=args.status_prefix,
        audit_prefix=audit_prefix,
        max_concurrency=args.max_concurrency,
        proxy_mode=args.proxy_mode,
        max_pages_per_series=args.max_pages_per_series,
        max_list_pages=args.max_list_pages,
        seed=args.seed,
        tolerated_failure_percentage=args.tolerated_failure_percentage,
    )
    state_machine_arn = ensure_state_machine(sfn, name=STATE_MACHINE_NAME, role_arn=sfn_role_arn, definition=definition)

    if not args.skip_smoke:
        smoke = invoke_smoke(
            lambda_client,
            function_arn=function_arn,
            series=first_manifest_series(manifest_path),
            args=args,
            run_id=run_id,
        )
        print(f"smoke=dry-run ok {json.dumps(smoke, sort_keys=True)}", flush=True)

    print(f"manifest=s3://{args.bucket}/{manifest_key}", flush=True)
    print(f"audit=s3://{args.bucket}/{audit_prefix}", flush=True)
    print(f"status=s3://{args.bucket}/{args.status_prefix.strip('/')}/{run_id}/", flush=True)
    print(f"lambda={function_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    if args.start:
        execution = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"webtoon-manga-direct-single-{run_id}",
            input=json.dumps({"run_id": run_id}),
        )
        print(f"execution={execution['executionArn']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
