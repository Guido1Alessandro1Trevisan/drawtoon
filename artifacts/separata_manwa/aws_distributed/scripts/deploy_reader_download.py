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


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "reader_handlers.py"
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_PREFIX = "datasets/pages/single"
DEFAULT_MANIFEST_PREFIX = "datasets/pages/single/_distributed_runs"
FUNCTION_NAME = "separata-manwa-reader-download-worker"
STATE_MACHINE_NAME = "separata-manwa-reader-download"
LAMBDA_ROLE_NAME = "separata-manwa-reader-download-lambda-role"
SFN_ROLE_NAME = "separata-manwa-reader-download-sfn-role"


def now_id() -> str:
    return "reader_download_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def zip_worker() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(WORKER, "reader_handlers.py")
    return buffer.getvalue()


def get_account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


def ensure_role(iam, *, role_name: str, service: str) -> str:
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
    }
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
    response = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_policy))
    time.sleep(8)
    return response["Role"]["Arn"]


def put_lambda_policy(iam, *, bucket: str, proxy_secret_name: str) -> None:
    secret_resource = "*"
    if proxy_secret_name:
        secret_resource = f"arn:aws:secretsmanager:*:*:secret:{proxy_secret_name}-*"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:*:*:*"},
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
            {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": secret_resource},
        ],
    }
    iam.put_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyName="separata-manwa-reader-download-lambda", PolicyDocument=json.dumps(policy))


def put_sfn_policy(iam, *, bucket: str, region: str, account_id: str) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": f"arn:aws:lambda:{region}:{account_id}:function:{FUNCTION_NAME}*"},
            {"Effect": "Allow", "Action": ["states:StartExecution"], "Resource": f"arn:aws:states:{region}:{account_id}:stateMachine:*"},
            {"Effect": "Allow", "Action": ["states:DescribeExecution", "states:StopExecution", "states:RedriveExecution"], "Resource": f"arn:aws:states:{region}:{account_id}:execution:*:*"},
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        ],
    }
    iam.put_role_policy(RoleName=SFN_ROLE_NAME, PolicyName="separata-manwa-reader-download-sfn", PolicyDocument=json.dumps(policy))


def wait_function(client, name: str) -> None:
    client.get_waiter("function_updated").wait(FunctionName=name, WaiterConfig={"Delay": 2, "MaxAttempts": 60})


def ensure_lambda(client, *, role_arn: str, args: argparse.Namespace) -> str:
    code = zip_worker()
    env = {"Variables": {"DATASET_BUCKET_NAME": args.bucket}}
    try:
        current = client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=code)
        wait_function(client, FUNCTION_NAME)
        response = client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Runtime=args.runtime,
            Role=role_arn,
            Handler="reader_handlers.handler",
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment=env,
        )
        wait_function(client, FUNCTION_NAME)
        arn = response.get("FunctionArn") or current["FunctionArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        response = client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=args.runtime,
            Role=role_arn,
            Handler="reader_handlers.handler",
            Code={"ZipFile": code},
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment=env,
        )
        wait_function(client, FUNCTION_NAME)
        arn = response["FunctionArn"]
    if args.reserved_concurrency > 0:
        client.put_function_concurrency(FunctionName=FUNCTION_NAME, ReservedConcurrentExecutions=args.reserved_concurrency)
    return arn


def state_machine_definition(lambda_arn: str) -> str:
    definition = {
        "Comment": "Distributed downloader for user-authorized reader-site manhwa chapters.",
        "StartAt": "DownloadChapters",
        "States": {
            "DownloadChapters": {
                "Type": "Map",
                "Label": "ReaderChapters",
                "ItemReader": {
                    "Resource": "arn:aws:states:::s3:getObject",
                    "Parameters": {"Bucket.$": "$.manifest.bucket", "Key.$": "$.manifest.key"},
                    "ReaderConfig": {"InputType": "JSONL"},
                },
                "ItemSelector": {"chapter.$": "$$.Map.Item.Value", "config.$": "$.config"},
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                    "StartAt": "DownloadChapter",
                    "States": {
                        "DownloadChapter": {
                            "Type": "Task",
                            "Resource": "arn:aws:states:::lambda:invoke",
                            "Parameters": {"FunctionName": lambda_arn, "Payload.$": "$"},
                            "TimeoutSeconds": 900,
                            "OutputPath": "$.Payload",
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
                                {"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 8, "MaxAttempts": 2, "BackoffRate": 2},
                            ],
                            "End": True,
                        }
                    },
                },
                "MaxConcurrencyPath": "$.max_concurrency",
                "ToleratedFailureCountPath": "$.tolerated_failure_count",
                "ResultWriter": {
                    "Resource": "arn:aws:states:::s3:putObject",
                    "Parameters": {"Bucket.$": "$.audit.bucket", "Prefix.$": "$.audit.prefix"},
                    "WriterConfig": {"OutputType": "JSONL", "Transformation": "COMPACT"},
                },
                "ResultPath": "$.map_result",
                "End": True,
            }
        },
        "TimeoutSeconds": 14400,
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


def read_manifest_rows(paths: list[str], series_filter: set[str], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if series_filter and row.get("output_series") not in series_filter and row.get("site_key") not in series_filter:
                    continue
                identity = (str(row.get("output_series")), str(row.get("chapter_slug")))
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(row)
                if max_rows > 0 and len(rows) >= max_rows:
                    return rows
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def execution_name(run_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)
    return safe[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--manifest-path", action="append", default=[], required=True)
    parser.add_argument("--series", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=250)
    parser.add_argument("--reserved-concurrency", type=int, default=300)
    parser.add_argument("--image-concurrency", type=int, default=8)
    parser.add_argument("--tolerated-failure-count", type=int, default=100)
    parser.add_argument("--network-mode", choices=["direct", "proxy"], default="direct")
    parser.add_argument("--proxy-secret-name", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--runtime", default="python3.12")
    parser.add_argument("--architecture", default="arm64")
    parser.add_argument("--lambda-timeout", type=int, default=900)
    parser.add_argument("--lambda-memory", type=int, default=1536)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or now_id()
    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    iam = session.client("iam")
    lambdas = session.client("lambda")
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")
    account_id = get_account_id(session)

    rows = read_manifest_rows(args.manifest_path, set(args.series), args.max_rows)
    if not rows:
        raise RuntimeError("no manifest rows selected")
    local_manifest = Path("artifacts/separata_manwa/manifests") / f"{run_id}_distributed_chapters.jsonl"
    write_jsonl(local_manifest, rows)
    manifest_key = f"{args.manifest_prefix.strip('/')}/{run_id}/chapters.jsonl"
    audit_prefix = f"{args.manifest_prefix.strip('/')}/{run_id}/audit/"
    s3.upload_file(str(local_manifest), args.bucket, manifest_key)

    lambda_role_arn = ensure_role(iam, role_name=LAMBDA_ROLE_NAME, service="lambda.amazonaws.com")
    put_lambda_policy(iam, bucket=args.bucket, proxy_secret_name=args.proxy_secret_name)
    worker_arn = ensure_lambda(lambdas, role_arn=lambda_role_arn, args=args)
    sfn_role_arn = ensure_role(iam, role_name=SFN_ROLE_NAME, service="states.amazonaws.com")
    put_sfn_policy(iam, bucket=args.bucket, region=args.region, account_id=account_id)
    state_machine_arn = ensure_state_machine(sfn, definition=state_machine_definition(worker_arn), role_arn=sfn_role_arn)

    config = {
        "bucket": args.bucket,
        "prefix": args.prefix.strip("/"),
        "image_concurrency": args.image_concurrency,
        "overwrite": bool(args.overwrite),
        "network_mode": args.network_mode,
        "proxy_secret_name": args.proxy_secret_name,
    }
    payload = {
        "run_id": run_id,
        "manifest": {"bucket": args.bucket, "key": manifest_key},
        "audit": {"bucket": args.bucket, "prefix": audit_prefix},
        "config": config,
        "max_concurrency": args.max_concurrency,
        "tolerated_failure_count": args.tolerated_failure_count,
    }
    print(f"rows={len(rows)}", flush=True)
    print(f"manifest=s3://{args.bucket}/{manifest_key}", flush=True)
    print(f"audit=s3://{args.bucket}/{audit_prefix}", flush=True)
    print(f"output=s3://{args.bucket}/{args.prefix.strip('/')}/", flush=True)
    print(f"lambda={worker_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    smoke_payload = {"chapter": rows[0], "config": {**config, "image_concurrency": min(2, args.image_concurrency)}}
    response = lambdas.invoke(FunctionName=worker_arn, InvocationType="RequestResponse", Payload=json.dumps(smoke_payload).encode("utf-8"))
    body = response["Payload"].read().decode("utf-8", errors="replace")
    if response.get("FunctionError"):
        raise RuntimeError(body)
    print(f"smoke={body}", flush=True)

    if args.start:
        execution = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name(run_id),
            input=json.dumps(payload),
        )
        print(f"execution={execution['executionArn']}", flush=True)


if __name__ == "__main__":
    main()
