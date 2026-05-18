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
SRC_DIR = ROOT / "src"
STATE_MACHINE_PATH = ROOT / "statemachines" / "download_episodes.asl.json"
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/source/webtoon"
DEFAULT_MANIFEST_PREFIX = "datasets/pages/source/webtoon/_distributed_runs"


def zip_source() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SRC_DIR.rglob("*.py")):
            archive.write(path, path.relative_to(SRC_DIR).as_posix())
    return buffer.getvalue()


def get_account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


def ensure_role(iam, *, role_name: str, service: str) -> str:
    trust_policy = {
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
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Drawtoon separata_manwa distributed downloader role",
    )
    return response["Role"]["Arn"]


def put_lambda_policy(iam, *, role_name: str, bucket: str, proxy_secret_name: str) -> None:
    secret_resource = "*"
    if proxy_secret_name:
        secret_resource = f"arn:aws:secretsmanager:*:*:secret:{proxy_secret_name}-*"
    policy = {
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
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": secret_resource,
            },
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName="separata-manwa-download-lambda", PolicyDocument=json.dumps(policy))


def put_state_machine_policy(iam, *, role_name: str, bucket: str, region: str, account_id: str) -> None:
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": f"arn:aws:lambda:{region}:{account_id}:function:separata-manwa-*",
            },
            {
                "Effect": "Allow",
                "Action": ["states:StartExecution"],
                "Resource": f"arn:aws:states:{region}:{account_id}:stateMachine:*",
            },
            {
                "Effect": "Allow",
                "Action": ["states:DescribeExecution", "states:StopExecution", "states:RedriveExecution"],
                "Resource": f"arn:aws:states:{region}:{account_id}:execution:*:*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName="separata-manwa-download-states", PolicyDocument=json.dumps(policy))


def wait_function(client, function_name: str) -> None:
    waiter = client.get_waiter("function_updated")
    waiter.wait(FunctionName=function_name, WaiterConfig={"Delay": 2, "MaxAttempts": 60})


def ensure_function(
    client,
    *,
    function_name: str,
    handler: str,
    role_arn: str,
    code_zip: bytes,
    timeout: int,
    memory: int,
    bucket: str,
) -> str:
    env = {"Variables": {"DATASET_BUCKET_NAME": bucket}}
    try:
        client.get_function(FunctionName=function_name)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        response = client.create_function(
            FunctionName=function_name,
            Runtime="python3.12",
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": code_zip},
            Timeout=timeout,
            MemorySize=memory,
            Architectures=["arm64"],
            Environment=env,
        )
        wait_function(client, function_name)
        return response["FunctionArn"]

    client.update_function_code(FunctionName=function_name, ZipFile=code_zip)
    wait_function(client, function_name)
    response = client.update_function_configuration(
        FunctionName=function_name,
        Runtime="python3.12",
        Role=role_arn,
        Handler=handler,
        Timeout=timeout,
        MemorySize=memory,
        Environment=env,
    )
    wait_function(client, function_name)
    return response["FunctionArn"]


def build_state_machine_definition(*, prepare_arn: str, worker_arn: str) -> str:
    raw = STATE_MACHINE_PATH.read_text(encoding="utf-8")
    return raw.replace("${PrepareManifestArn}", prepare_arn).replace("${DownloadEpisodeArn}", worker_arn)


def ensure_state_machine(client, *, name: str, definition: str, role_arn: str) -> str:
    try:
        response = client.create_state_machine(
            name=name,
            definition=definition,
            roleArn=role_arn,
            type="STANDARD",
        )
        return response["stateMachineArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "StateMachineAlreadyExists":
            raise
    arn = client.list_state_machines()["stateMachines"]
    state_machine_arn = next(item["stateMachineArn"] for item in arn if item["name"] == name)
    client.update_state_machine(stateMachineArn=state_machine_arn, definition=definition, roleArn=role_arn)
    return state_machine_arn


def default_run_id() -> str:
    return "separata_manwa_" + dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def execution_name(run_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)
    return safe[:80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--state-machine-name", default="separata-manwa-download")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-concurrency", type=int, default=300)
    parser.add_argument("--worker-image-concurrency", type=int, default=8)
    parser.add_argument("--tolerated-failure-count", type=int, default=250)
    parser.add_argument("--proxy-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--proxy-secret-name", default="")
    parser.add_argument("--max-list-pages", type=int, default=200)
    parser.add_argument("--max-episodes-per-series", type=int, default=0)
    parser.add_argument("--max-images-per-episode", type=int, default=0)
    parser.add_argument("--series", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--deploy-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    account_id = get_account_id(session)
    iam = session.client("iam")
    lambda_client = session.client("lambda")
    states_client = session.client("stepfunctions")

    lambda_role_name = "separata-manwa-download-lambda-role"
    states_role_name = "separata-manwa-download-states-role"
    lambda_role_arn = ensure_role(iam, role_name=lambda_role_name, service="lambda.amazonaws.com")
    states_role_arn = ensure_role(iam, role_name=states_role_name, service="states.amazonaws.com")
    put_lambda_policy(iam, role_name=lambda_role_name, bucket=args.bucket, proxy_secret_name=args.proxy_secret_name)
    put_state_machine_policy(iam, role_name=states_role_name, bucket=args.bucket, region=args.region, account_id=account_id)
    time.sleep(8)

    code_zip = zip_source()
    prepare_arn = ensure_function(
        lambda_client,
        function_name="separata-manwa-prepare-manifest",
        handler="handlers.prepare_manifest",
        role_arn=lambda_role_arn,
        code_zip=code_zip,
        timeout=900,
        memory=1024,
        bucket=args.bucket,
    )
    worker_arn = ensure_function(
        lambda_client,
        function_name="separata-manwa-download-episode",
        handler="handlers.download_episode",
        role_arn=lambda_role_arn,
        code_zip=code_zip,
        timeout=900,
        memory=2048,
        bucket=args.bucket,
    )
    definition = build_state_machine_definition(prepare_arn=prepare_arn, worker_arn=worker_arn)
    state_machine_arn = ensure_state_machine(
        states_client,
        name=args.state_machine_name,
        definition=definition,
        role_arn=states_role_arn,
    )

    run_id = args.run_id or default_run_id()
    payload: dict[str, Any] = {
        "bucket": args.bucket,
        "output_prefix": args.output_prefix.strip("/"),
        "manifest_prefix": args.manifest_prefix.strip("/"),
        "run_id": run_id,
        "max_concurrency": max(1, int(args.max_concurrency)),
        "worker_image_concurrency": max(1, int(args.worker_image_concurrency)),
        "tolerated_failure_count": max(0, int(args.tolerated_failure_count)),
        "proxy_mode": args.proxy_mode,
        "proxy_secret_name": args.proxy_secret_name,
        "max_list_pages": max(1, int(args.max_list_pages)),
        "max_episodes_per_series": max(0, int(args.max_episodes_per_series)),
        "max_images_per_episode": max(0, int(args.max_images_per_episode)),
        "series": args.series,
        "overwrite": bool(args.overwrite),
    }
    summary = {
        "state_machine_arn": state_machine_arn,
        "prepare_function_arn": prepare_arn,
        "worker_function_arn": worker_arn,
        "input": payload,
    }
    if args.dry_run or args.deploy_only:
        print(json.dumps(summary, indent=2))
        return
    response = states_client.start_execution(
        stateMachineArn=state_machine_arn,
        name=execution_name(run_id),
        input=json.dumps(payload, ensure_ascii=False),
    )
    summary["execution_arn"] = response["executionArn"]
    summary["start_date"] = str(response["startDate"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
