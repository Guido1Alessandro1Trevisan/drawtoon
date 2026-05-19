#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import boto3


def main() -> int:
    parser = argparse.ArgumentParser(description="Show download_scraper Step Functions status.")
    parser.add_argument("--execution-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    sfn = boto3.client("stepfunctions", region_name=args.region)
    execution = sfn.describe_execution(executionArn=args.execution_arn)
    summary = {key: execution.get(key) for key in ["name", "status", "startDate", "stopDate"]}
    print(json.dumps(summary, default=str, indent=2), flush=True)

    runs = sfn.list_map_runs(executionArn=args.execution_arn).get("mapRuns", [])
    for run in runs:
        detail = sfn.describe_map_run(mapRunArn=run["mapRunArn"])
        print(json.dumps({key: detail.get(key) for key in ["mapRunArn", "status", "itemCounts", "executionCounts"]}, default=str, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

