#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "aws" / "src" / "worker.py"
FUNCTION_NAME = "download-scraper-worker"
STATE_MACHINE_NAME = "download-scraper-map"
LAMBDA_ROLE_NAME = "download-scraper-lambda-role"
SFN_ROLE_NAME = "download-scraper-sfn-role"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_MANIFEST_PREFIX = "workflows/download_scraper/manifests"
DEFAULT_PROXY_ENV_SOURCE_FUNCTION = "webtoon_manga_episode_worker_v2"


def now_id() -> str:
    return "download_scraper_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(WORKER, "worker.py")
    return buffer.getvalue()


def ensure_role(iam, *, role_name: str, service: str) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
    }
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
    role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]
    time.sleep(8)
    return role["Arn"]


def put_lambda_policy(iam, *, bucket: str) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:*:*:*"},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
            {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": "*"},
        ],
    }
    iam.put_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyName="download-scraper-lambda", PolicyDocument=json.dumps(policy))


def put_sfn_policy(iam, *, bucket: str, region: str, account_id: str) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": f"arn:aws:lambda:{region}:{account_id}:function:{FUNCTION_NAME}*"},
            {"Effect": "Allow", "Action": ["states:StartExecution", "states:DescribeExecution", "states:StopExecution", "states:RedriveExecution"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"], "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        ],
    }
    iam.put_role_policy(RoleName=SFN_ROLE_NAME, PolicyName="download-scraper-sfn", PolicyDocument=json.dumps(policy))


def wait_for_sfn_permissions(iam, *, role_arn: str, bucket: str, audit_prefix: str, timeout_seconds: int = 90) -> None:
    """Wait until IAM propagation lets Step Functions write ResultWriter output."""
    deadline = time.time() + timeout_seconds
    resource = f"arn:aws:s3:::{bucket}/{audit_prefix.strip('/')}/permission-probe.json"
    while True:
        response = iam.simulate_principal_policy(
            PolicySourceArn=role_arn,
            ActionNames=["s3:PutObject"],
            ResourceArns=[resource],
        )
        decision = ((response.get("EvaluationResults") or [{}])[0]).get("EvalDecision")
        if decision == "allowed":
            return
        if time.time() >= deadline:
            raise RuntimeError(f"timed out waiting for {role_arn} to allow s3:PutObject on {resource}: {decision}")
        time.sleep(3)


def proxy_env(lambda_client, *, source_function_name: str) -> dict[str, str]:
    keys = ["DECODO_PROXY_HOST", "DECODO_PROXY_PORTS", "DECODO_PROXY_USER", "DECODO_PROXY_PASS"]
    env: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value:
            env[key] = value
    missing = [key for key in keys if key not in env]
    if missing and source_function_name:
        try:
            current = lambda_client.get_function_configuration(FunctionName=source_function_name)
            source_env = (current.get("Environment") or {}).get("Variables") or {}
        except ClientError:
            source_env = {}
        for key in missing:
            value = source_env.get(key)
            if value:
                env[key] = value
    return env


def ensure_lambda(client, *, role_arn: str, args: argparse.Namespace) -> str:
    env = {
        "DEST_BUCKET": args.bucket,
        "PROXY_MODE": args.proxy_mode,
        "SESSION_SECRET_NAME": args.session_secret_name or "",
        "HTTP_TIMEOUT": str(args.http_timeout),
        "IMAGE_RETRIES": str(args.image_retries),
        "MIN_WIDTH": str(args.min_width),
        "MIN_HEIGHT": str(args.min_height),
        "REQUEST_DELAY_SECONDS": str(args.request_delay_seconds),
        "SKIP_EXISTING": "1" if args.skip_existing else "0",
    }
    env.update(proxy_env(client, source_function_name=args.proxy_env_source_function))
    code = build_zip()
    try:
        current = client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=code)
        client.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
        response = client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Runtime=args.runtime,
            Role=role_arn,
            Handler="worker.handler",
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment={"Variables": env},
        )
        client.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
        arn = response.get("FunctionArn") or current["FunctionArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        response = client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=args.runtime,
            Role=role_arn,
            Handler="worker.handler",
            Code={"ZipFile": code},
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment={"Variables": env},
        )
        client.get_waiter("function_active").wait(FunctionName=FUNCTION_NAME)
        arn = response["FunctionArn"]
    if args.reserved_concurrency > 0:
        client.put_function_concurrency(FunctionName=FUNCTION_NAME, ReservedConcurrentExecutions=args.reserved_concurrency)
    return arn


def state_machine_definition(lambda_arn: str, *, task_timeout_seconds: int, parent_timeout_seconds: int) -> str:
    definition = {
        "Comment": "Generic manifest-first authorized image downloader.",
        "StartAt": "DownloadRows",
        "States": {
            "DownloadRows": {
                "Type": "Map",
                "Label": "DownloadScraperRows",
                "ItemReader": {
                    "Resource": "arn:aws:states:::s3:getObject",
                    "Parameters": {"Bucket.$": "$.manifest.bucket", "Key.$": "$.manifest.key"},
                    "ReaderConfig": {"InputType": "JSONL"},
                },
                "ItemSelector": {"row.$": "$$.Map.Item.Value", "config.$": "$.config"},
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "EXPRESS"},
                    "StartAt": "DownloadRow",
                    "States": {
                        "DownloadRow": {
                            "Type": "Task",
                            "Resource": "arn:aws:states:::lambda:invoke",
                            "Parameters": {"FunctionName": lambda_arn, "Payload.$": "$"},
                            "TimeoutSeconds": task_timeout_seconds,
                            "OutputPath": "$.Payload",
                            "Retry": [
                                {
                                    "ErrorEquals": ["Lambda.TooManyRequestsException", "Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException"],
                                    "IntervalSeconds": 2,
                                    "MaxAttempts": 6,
                                    "BackoffRate": 2,
                                },
                                {"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 5, "MaxAttempts": 2, "BackoffRate": 2},
                            ],
                            "End": True,
                        }
                    },
                },
                "MaxConcurrencyPath": "$.max_concurrency",
                "ToleratedFailurePercentagePath": "$.tolerated_failure_percentage",
                "ResultWriter": {
                    "Resource": "arn:aws:states:::s3:putObject",
                    "Parameters": {"Bucket.$": "$.audit.bucket", "Prefix.$": "$.audit.prefix"},
                    "WriterConfig": {"OutputType": "JSONL", "Transformation": "COMPACT"},
                },
                "End": True,
            }
        },
        "TimeoutSeconds": parent_timeout_seconds,
    }
    return json.dumps(definition, indent=2)


def ensure_state_machine(client, *, definition: str, role_arn: str) -> str:
    existing = None
    for page in client.get_paginator("list_state_machines").paginate():
        for item in page.get("stateMachines", []):
            if item["name"] == STATE_MACHINE_NAME:
                existing = item["stateMachineArn"]
                break
        if existing:
            break
    if existing:
        client.update_state_machine(stateMachineArn=existing, definition=definition, roleArn=role_arn)
        return existing
    return client.create_state_machine(name=STATE_MACHINE_NAME, definition=definition, roleArn=role_arn, type="STANDARD")["stateMachineArn"]


def first_manifest_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty manifest: {path}")


def execution_name(run_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--manifest-key", default="")
    parser.add_argument("--audit-prefix", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--session-secret-name", default="")
    parser.add_argument("--proxy-mode", choices=["direct", "proxy", "auto"], default="auto")
    parser.add_argument("--proxy-env-source-function", default=DEFAULT_PROXY_ENV_SOURCE_FUNCTION)
    parser.add_argument("--max-concurrency", type=int, default=300)
    parser.add_argument("--reserved-concurrency", type=int, default=350)
    parser.add_argument("--tolerated-failure-percentage", type=float, default=2.0)
    parser.add_argument("--http-timeout", type=float, default=45.0)
    parser.add_argument("--image-retries", type=int, default=4)
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--min-height", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--runtime", default="python3.12")
    parser.add_argument("--lambda-timeout", type=int, default=120)
    parser.add_argument("--lambda-memory", type=int, default=1024)
    parser.add_argument("--task-timeout-seconds", type=int, default=120)
    parser.add_argument("--parent-timeout-seconds", type=int, default=7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or now_id()
    manifest_path = Path(args.manifest_path)
    manifest_key = args.manifest_key or f"{args.manifest_prefix.strip('/')}/{run_id}/manifest.jsonl"
    audit_prefix = args.audit_prefix or f"{args.manifest_prefix.strip('/')}/{run_id}/audit/"

    session = boto3.Session(region_name=args.region)
    iam = session.client("iam")
    lambda_client = session.client("lambda")
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")
    account_id = session.client("sts").get_caller_identity()["Account"]

    lambda_role = ensure_role(iam, role_name=LAMBDA_ROLE_NAME, service="lambda.amazonaws.com")
    put_lambda_policy(iam, bucket=args.bucket)
    lambda_arn = ensure_lambda(lambda_client, role_arn=lambda_role, args=args)
    sfn_role = ensure_role(iam, role_name=SFN_ROLE_NAME, service="states.amazonaws.com")
    put_sfn_policy(iam, bucket=args.bucket, region=args.region, account_id=account_id)
    wait_for_sfn_permissions(iam, role_arn=sfn_role, bucket=args.bucket, audit_prefix=audit_prefix)
    state_machine_arn = ensure_state_machine(
        sfn,
        definition=state_machine_definition(
            lambda_arn,
            task_timeout_seconds=args.task_timeout_seconds,
            parent_timeout_seconds=args.parent_timeout_seconds,
        ),
        role_arn=sfn_role,
    )

    s3.upload_file(str(manifest_path), args.bucket, manifest_key)
    config = {
        "proxy_mode": args.proxy_mode,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "request_delay_seconds": args.request_delay_seconds,
    }
    payload = {
        "run_id": run_id,
        "manifest": {"bucket": args.bucket, "key": manifest_key},
        "audit": {"bucket": args.bucket, "prefix": audit_prefix},
        "config": config,
        "max_concurrency": args.max_concurrency,
        "tolerated_failure_percentage": args.tolerated_failure_percentage,
    }

    print(f"run_id={run_id}", flush=True)
    print(f"manifest=s3://{args.bucket}/{manifest_key}", flush=True)
    print(f"audit=s3://{args.bucket}/{audit_prefix}", flush=True)
    print(f"lambda={lambda_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    if args.smoke:
        response = lambda_client.invoke(
            FunctionName=lambda_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps({"row": first_manifest_row(manifest_path), "config": {**config, "dry_run": True}}).encode("utf-8"),
        )
        body = response["Payload"].read().decode("utf-8", errors="replace")
        print(f"smoke={body}", flush=True)
        if response.get("FunctionError"):
            raise RuntimeError(body)

    if args.start:
        execution = sfn.start_execution(stateMachineArn=state_machine_arn, name=execution_name(run_id), input=json.dumps(payload))
        print(f"execution={execution['executionArn']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
