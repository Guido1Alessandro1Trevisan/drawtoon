#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import boto3


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("execution_arn")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default="")
    args = parser.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else boto3.Session(region_name=args.region)
    client = session.client("stepfunctions")
    execution = client.describe_execution(executionArn=args.execution_arn)
    output = {
        "execution_arn": args.execution_arn,
        "status": execution["status"],
        "start_date": str(execution["startDate"]),
        "stop_date": str(execution.get("stopDate", "")),
    }
    if "output" in execution:
        try:
            output["output"] = json.loads(execution["output"])
        except Exception:
            output["output_raw"] = execution["output"]
    try:
        map_runs = client.list_map_runs(executionArn=args.execution_arn).get("mapRuns", [])
    except Exception as exc:
        map_runs = []
        output["map_run_error"] = str(exc)
    output["map_runs"] = []
    for item in map_runs:
        detail = client.describe_map_run(mapRunArn=item["mapRunArn"])
        output["map_runs"].append(
            {
                "map_run_arn": item["mapRunArn"],
                "status": detail["status"],
                "start_date": str(detail["startDate"]),
                "stop_date": str(detail.get("stopDate", "")),
                "item_counts": detail.get("itemCounts", {}),
                "execution_counts": detail.get("executionCounts", {}),
            }
        )
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
