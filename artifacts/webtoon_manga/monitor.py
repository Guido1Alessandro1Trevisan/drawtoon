#!/usr/bin/env python3
"""Monitor a webtoon_manga Step Functions execution and S3 output count."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import boto3


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def find_map_run_arn(sfn, execution_arn: str) -> str | None:
    paginator = sfn.get_paginator("get_execution_history")
    for page in paginator.paginate(executionArn=execution_arn, reverseOrder=True, PaginationConfig={"PageSize": 100}):
        for event in page.get("events", []):
            if event.get("type") == "MapRunStarted":
                return event.get("mapRunStartedEventDetails", {}).get("mapRunArn")
    return None


def count_s3_objects(s3, *, bucket: str, prefix: str) -> tuple[int, int]:
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    size = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            count += 1
            size += int(item.get("Size", 0))
    return count, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_arn")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", default="drawtoon")
    parser.add_argument("--source-prefix", default="datasets/pages/source/webtoon")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")
    map_run_arn = None
    started = time.time()

    while True:
        execution = sfn.describe_execution(executionArn=args.execution_arn)
        if map_run_arn is None:
            map_run_arn = find_map_run_arn(sfn, args.execution_arn)

        map_text = "map=starting"
        eta = None
        if map_run_arn:
            run = sfn.describe_map_run(mapRunArn=map_run_arn)
            counts = run.get("itemCounts", {})
            total = int(counts.get("total", 0) or 0)
            succeeded = int(counts.get("succeeded", 0) or 0)
            failed = int(counts.get("failed", 0) or 0)
            running = int(counts.get("running", 0) or 0)
            pending = int(counts.get("pending", 0) or 0)
            elapsed = max(0.001, time.time() - started)
            rate = succeeded / elapsed if succeeded else 0
            if rate and total:
                eta = max(0, total - succeeded - failed) / rate
            map_text = (
                f"map={run.get('status')} items={succeeded}/{total} "
                f"running={running} pending={pending} failed={failed} item_rate={rate:.2f}/s"
            )

        object_count, total_size = count_s3_objects(s3, bucket=args.bucket, prefix=args.source_prefix)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(
            f"{timestamp} execution={execution['status']} {map_text} "
            f"s3_objects={object_count} s3_gb={total_size / 1_000_000_000:.2f} eta={format_eta(eta)}",
            flush=True,
        )
        if execution["status"] not in {"RUNNING"}:
            return 0 if execution["status"] == "SUCCEEDED" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
