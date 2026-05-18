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
WORKER = ROOT / "clean_single_pages.py"
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_SOURCE_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/single_relevant"
DEFAULT_STATUS_PREFIX = "datasets/pages/single_relevant/_status"
DEFAULT_MANIFEST_PREFIX = "datasets/pages/single_relevant/_distributed_runs"
DEFAULT_SERIES = (
    "solo-leveling_manwa",
    "sss-class-suicide-hunter_manwa",
    "second-life-ranker_manwa",
    "a-returners-magic-should-be-special_manwa",
    "the-great-mage-returns-after-4000-years_manwa",
    "lout-of-counts-family_manwa",
)
FUNCTION_NAME = "separata-manwa-reader-clean-worker"
STATE_MACHINE_NAME = "separata-manwa-reader-clean"
LAMBDA_ROLE_NAME = "separata-manwa-reader-clean-lambda-role"
SFN_ROLE_NAME = "separata-manwa-reader-clean-sfn-role"


def now_id() -> str:
    return "reader_clean_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def zip_worker() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(WORKER, "clean_single_pages.py")
    return buffer.getvalue()


def get_account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


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
    response = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust))
    time.sleep(8)
    return response["Role"]["Arn"]


def put_lambda_policy(iam, *, bucket: str) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:*:*:*"},
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
        ],
    }
    iam.put_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyName="separata-manwa-reader-clean-lambda", PolicyDocument=json.dumps(policy))


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
    iam.put_role_policy(RoleName=SFN_ROLE_NAME, PolicyName="separata-manwa-reader-clean-sfn", PolicyDocument=json.dumps(policy))


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
            Handler="clean_single_pages.lambda_handler",
            Timeout=args.lambda_timeout,
            MemorySize=args.lambda_memory,
            Environment=env,
        )
        wait_function(client, FUNCTION_NAME)
        return response.get("FunctionArn") or current["FunctionArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
    response = client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime=args.runtime,
        Role=role_arn,
        Handler="clean_single_pages.lambda_handler",
        Code={"ZipFile": code},
        Timeout=args.lambda_timeout,
        MemorySize=args.lambda_memory,
        Environment=env,
    )
    wait_function(client, FUNCTION_NAME)
    return response["FunctionArn"]


def state_machine_definition(lambda_arn: str) -> str:
    definition = {
        "Comment": "Distributed relevant-page cleanup for authorized reader-site manhwa pages.",
        "StartAt": "CleanChapters",
        "States": {
            "CleanChapters": {
                "Type": "Map",
                "Label": "ReaderCleanChapters",
                "ItemReader": {
                    "Resource": "arn:aws:states:::s3:getObject",
                    "Parameters": {"Bucket.$": "$.manifest.bucket", "Key.$": "$.manifest.key"},
                    "ReaderConfig": {"InputType": "JSONL"},
                },
                "ItemSelector": {"chapter.$": "$$.Map.Item.Value", "config.$": "$.config"},
                "ItemProcessor": {
                    "ProcessorConfig": {"Mode": "DISTRIBUTED", "ExecutionType": "STANDARD"},
                    "StartAt": "CleanChapter",
                    "States": {
                        "CleanChapter": {
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


def list_chapter_rows(s3, *, bucket: str, source_prefix: str, series_list: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for series in series_list:
        root = f"{source_prefix.strip('/')}/{series.strip('/')}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
            for item in page.get("CommonPrefixes", []):
                prefix = str(item.get("Prefix") or "")
                if not prefix:
                    continue
                chapter = prefix.rstrip("/").rsplit("/", 1)[-1]
                rows.append({"series": series, "chapter": chapter, "prefix": prefix})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--status-prefix", default=DEFAULT_STATUS_PREFIX)
    parser.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--series", action="append", default=[])
    parser.add_argument("--max-concurrency", type=int, default=250)
    parser.add_argument("--reserved-concurrency", type=int, default=350)
    parser.add_argument("--dimension-workers", type=int, default=12)
    parser.add_argument("--copy-workers", type=int, default=12)
    parser.add_argument("--min-width", type=int, default=220)
    parser.add_argument("--min-height", type=int, default=120)
    parser.add_argument("--min-ratio", type=float, default=0.45)
    parser.add_argument("--max-ratio", type=float, default=60.0)
    parser.add_argument("--width-tolerance-ratio", type=float, default=0.10)
    parser.add_argument("--width-tolerance-px", type=int, default=40)
    parser.add_argument("--story-window", type=int, default=10)
    parser.add_argument("--story-window-min-good", type=int, default=5)
    parser.add_argument("--tolerated-failure-count", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--runtime", default="python3.12")
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
    series_list = args.series or list(DEFAULT_SERIES)

    rows = list_chapter_rows(s3, bucket=args.bucket, source_prefix=args.source_prefix, series_list=series_list)
    if not rows:
        raise RuntimeError("no chapter prefixes found")
    local_manifest = Path("artifacts/separata_manwa/manifests") / f"{run_id}_clean_chapters.jsonl"
    write_jsonl(local_manifest, rows)
    manifest_key = f"{args.manifest_prefix.strip('/')}/{run_id}/chapters.jsonl"
    audit_prefix = f"{args.manifest_prefix.strip('/')}/{run_id}/audit/"
    s3.upload_file(str(local_manifest), args.bucket, manifest_key)

    lambda_role_arn = ensure_role(iam, role_name=LAMBDA_ROLE_NAME, service="lambda.amazonaws.com")
    put_lambda_policy(iam, bucket=args.bucket)
    worker_arn = ensure_lambda(lambdas, role_arn=lambda_role_arn, args=args)
    if args.reserved_concurrency > 0:
        lambdas.put_function_concurrency(FunctionName=FUNCTION_NAME, ReservedConcurrentExecutions=args.reserved_concurrency)
    sfn_role_arn = ensure_role(iam, role_name=SFN_ROLE_NAME, service="states.amazonaws.com")
    put_sfn_policy(iam, bucket=args.bucket, region=args.region, account_id=account_id)
    state_machine_arn = ensure_state_machine(sfn, definition=state_machine_definition(worker_arn), role_arn=sfn_role_arn)

    config = {
        "bucket": args.bucket,
        "source_prefix": args.source_prefix.strip("/"),
        "output_prefix": args.output_prefix.strip("/"),
        "status_prefix": args.status_prefix.strip("/"),
        "dimension_workers": args.dimension_workers,
        "copy_workers": args.copy_workers,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "min_ratio": args.min_ratio,
        "max_ratio": args.max_ratio,
        "width_tolerance_ratio": args.width_tolerance_ratio,
        "width_tolerance_px": args.width_tolerance_px,
        "story_window": args.story_window,
        "story_window_min_good": args.story_window_min_good,
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
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
    print(f"output=s3://{args.bucket}/{args.output_prefix.strip('/')}/", flush=True)
    print(f"lambda={worker_arn}", flush=True)
    print(f"state_machine={state_machine_arn}", flush=True)

    smoke_payload = {"chapter": rows[0], "config": {**config, "dry_run": True, "dimension_workers": 4, "copy_workers": 1}}
    response = lambdas.invoke(FunctionName=worker_arn, InvocationType="RequestResponse", Payload=json.dumps(smoke_payload).encode("utf-8"))
    body = response["Payload"].read().decode("utf-8", errors="replace")
    if response.get("FunctionError"):
        raise RuntimeError(body)
    print(f"smoke={body}", flush=True)

    if args.start:
        execution = sfn.start_execution(stateMachineArn=state_machine_arn, name=execution_name(run_id), input=json.dumps(payload))
        print(f"execution={execution['executionArn']}", flush=True)


if __name__ == "__main__":
    main()
