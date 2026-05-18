#!/usr/bin/env python3
"""Deploy and optionally start the webtoon_manga Distributed Map downloader."""

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
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "worker.py"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_SOURCE_PREFIX = "datasets/pages/source/webtoon"
DEFAULT_MANIFEST = ROOT / "manifest/webtoon_episodes.jsonl"
FUNCTION_NAME = "webtoon_manga_episode_worker_v2"
STATE_MACHINE_NAME = "webtoon_manga_distributed_map"
LAMBDA_ROLE_NAME = "webtoon_manga_lambda_role"
SFN_ROLE_NAME = "webtoon_manga_sfn_role"


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def ensure_role(iam, *, name: str, service: str, policy_name: str, policy_doc: dict[str, Any]) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": service},
                "Action": "sts:AssumeRole",
            }
        ],
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
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
        ],
    }


def sfn_policy(bucket: str, lambda_arn: str, account: str, region: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": [lambda_arn, f"{lambda_arn}:*"]},
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
            {"Effect": "Allow", "Action": ["states:StartExecution", "states:DescribeExecution", "states:StopExecution"], "Resource": "*"},
        ],
    }


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(WORKER, "worker.py")
    return buffer.getvalue()


def decodo_env_from_process() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ["DECODO_PROXY_HOST", "DECODO_PROXY_PORTS", "DECODO_PROXY_USER", "DECODO_PROXY_PASS"]:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def ensure_lambda(lambda_client, *, role_arn: str, args: argparse.Namespace) -> str:
    env = {
        "DEST_BUCKET": args.bucket,
        "SOURCE_PREFIX": args.source_prefix,
        "PROXY_MODE": args.proxy_mode,
        "IMAGE_WORKERS": str(args.lambda_image_workers),
        "IMAGE_RETRIES": str(args.image_retries),
        "HTML_RETRIES": str(args.html_retries),
        "HTTP_TIMEOUT": str(args.http_timeout),
        "CHECK_EXISTING": "1" if args.check_existing else "0",
        "FAIL_ON_PARTIAL_IMAGE_FAILURE": "1" if args.fail_on_partial_image_failure else "0",
        **decodo_env_from_process(),
    }
    zip_bytes = build_zip()
    try:
        current = lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes, Publish=True)
        waiter = lambda_client.get_waiter("function_updated")
        waiter.wait(FunctionName=FUNCTION_NAME)
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Runtime=args.runtime,
            Handler="worker.handler",
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
            Handler="worker.handler",
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
    audit_prefix: str,
    max_concurrency: int,
    proxy_mode: str,
    tolerated_failure_percentage: float,
) -> str:
    definition = {
        "Comment": "Distributed download of authorized webtoon_manga episodes.",
        "StartAt": "DownloadEpisodes",
        "States": {
            "DownloadEpisodes": {
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
                    "proxy_mode": proxy_mode,
                    "overwrite": False,
                },
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "EXPRESS"},
                    "StartAt": "DownloadEpisode",
                    "States": {
                        "DownloadEpisode": {
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
    paginator = sfn.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for state_machine in page.get("stateMachines", []):
            if state_machine["name"] == name:
                existing_arn = state_machine["stateMachineArn"]
                break
        if existing_arn:
            break
    if existing_arn:
        sfn.update_state_machine(stateMachineArn=existing_arn, definition=definition, roleArn=role_arn)
        return existing_arn
    return sfn.create_state_machine(
        name=name,
        definition=definition,
        roleArn=role_arn,
        type="STANDARD",
        loggingConfiguration={"level": "OFF"},
    )["stateMachineArn"]


def upload_manifest(s3, *, bucket: str, manifest_path: Path, manifest_key: str) -> None:
    s3.upload_file(str(manifest_path), bucket, manifest_key)


def first_manifest_item(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"empty manifest: {path}")


def invoke_smoke(lambda_client, *, function_arn: str, item: dict[str, Any], bucket: str, source_prefix: str, proxy_mode: str) -> dict[str, Any]:
    payload = {
        **item,
        "bucket": bucket,
        "source_prefix": source_prefix,
        "proxy_mode": proxy_mode,
        "max_images": 1,
    }
    response = lambda_client.invoke(FunctionName=function_arn, InvocationType="RequestResponse", Payload=json.dumps(payload).encode("utf-8"))
    body = response["Payload"].read().decode("utf-8", errors="replace")
    parsed = json.loads(body) if body else {}
    if response.get("FunctionError"):
        raise RuntimeError(json.dumps(parsed, sort_keys=True))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--manifest-key")
    parser.add_argument("--audit-prefix")
    parser.add_argument("--max-concurrency", type=int, default=1000)
    parser.add_argument("--proxy-mode", choices=["direct", "proxy", "auto"], default="auto")
    parser.add_argument("--tolerated-failure-percentage", type=float, default=5.0)
    parser.add_argument("--lambda-image-workers", type=int, default=4)
    parser.add_argument("--image-retries", type=int, default=4)
    parser.add_argument("--html-retries", type=int, default=3)
    parser.add_argument("--http-timeout", type=float, default=45)
    parser.add_argument("--lambda-timeout", type=int, default=300)
    parser.add_argument("--lambda-memory", type=int, default=1024)
    parser.add_argument("--runtime", default="python3.11")
    parser.add_argument("--reserved-concurrency", type=int, default=1200)
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--fail-on-partial-image-failure", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--smoke-direct-first", action="store_true", default=True)
    parser.add_argument("--no-smoke-direct-first", dest="smoke_direct_first", action="store_false")
    args = parser.parse_args()

    run_id = now_id()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_key = args.manifest_key or f"datasets/pages/manifests/webtoon_manga/{run_id}/episodes.jsonl"
    audit_prefix = args.audit_prefix or f"workflows/webtoon_manga/audit/{run_id}/"

    session = boto3.Session(region_name=args.region)
    sts = session.client("sts")
    iam = session.client("iam")
    lambda_client = session.client("lambda")
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")
    account = account_id(sts)

    lambda_role_arn = ensure_role(
        iam,
        name=LAMBDA_ROLE_NAME,
        service="lambda.amazonaws.com",
        policy_name="webtoon_manga_lambda_policy",
        policy_doc=lambda_policy(args.bucket),
    )
    function_arn = ensure_lambda(lambda_client, role_arn=lambda_role_arn, args=args)
    sfn_role_arn = ensure_role(
        iam,
        name=SFN_ROLE_NAME,
        service="states.amazonaws.com",
        policy_name="webtoon_manga_sfn_policy",
        policy_doc=sfn_policy(args.bucket, function_arn, account, args.region),
    )

    upload_manifest(s3, bucket=args.bucket, manifest_path=manifest_path, manifest_key=manifest_key)
    definition = state_machine_definition(
        lambda_arn=function_arn,
        bucket=args.bucket,
        manifest_key=manifest_key,
        source_prefix=args.source_prefix,
        audit_prefix=audit_prefix,
        max_concurrency=args.max_concurrency,
        proxy_mode=args.proxy_mode,
        tolerated_failure_percentage=args.tolerated_failure_percentage,
    )
    state_machine_arn = ensure_state_machine(sfn, name=STATE_MACHINE_NAME, role_arn=sfn_role_arn, definition=definition)

    item = first_manifest_item(manifest_path)
    smoke_mode = "direct" if args.smoke_direct_first else args.proxy_mode
    try:
        smoke = invoke_smoke(
            lambda_client,
            function_arn=function_arn,
            item=item,
            bucket=args.bucket,
            source_prefix=args.source_prefix,
            proxy_mode=smoke_mode,
        )
        print(f"smoke={smoke_mode} ok {json.dumps(smoke, sort_keys=True)}", flush=True)
    except Exception as exc:
        if args.proxy_mode in {"proxy", "auto"} and decodo_env_from_process():
            smoke = invoke_smoke(
                lambda_client,
                function_arn=function_arn,
                item=item,
                bucket=args.bucket,
                source_prefix=args.source_prefix,
                proxy_mode="proxy",
            )
            print(f"smoke=proxy ok {json.dumps(smoke, sort_keys=True)}", flush=True)
        else:
            raise RuntimeError(f"direct smoke failed and proxy env is not configured for deployment: {exc}") from exc

    print(f"manifest=s3://{args.bucket}/{manifest_key}", flush=True)
    print(f"audit=s3://{args.bucket}/{audit_prefix}", flush=True)
    print(f"lambda={function_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    if args.start:
        execution = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=f"webtoon-manga-{run_id}",
            input=json.dumps({"run_id": run_id}),
        )
        print(f"execution={execution['executionArn']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
