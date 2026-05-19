#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def iter_jsonl_objects(s3, *, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key.endswith(".jsonl"):
                yield key


def read_object_text(s3, *, bucket: str, key: str, attempts: int = 5) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", errors="replace")
        except (BotoCoreError, ClientError, TimeoutError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(10.0, 0.75 * attempt))
    raise RuntimeError(f"failed to read s3://{bucket}/{key}: {last_error}")


def read_jsonl_s3(s3, *, bucket: str, key: str):
    body = read_object_text(s3, bucket=bucket, key=key)
    for line in body.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def dedupe_key(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        current: Any = row
        for part in field.split("."):
            if not isinstance(current, dict):
                current = ""
                break
            current = current.get(part)
        values.append(str(current or ""))
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect RFCO catalog JSONL shards from S3 into one local manifest.")
    parser.add_argument("--bucket", default="drawtoon")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dedupe-field", action="append", default=[])
    args = parser.parse_args()

    fields = list(args.dedupe_field or []) or ["url"]
    s3 = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(connect_timeout=10, read_timeout=180, retries={"max_attempts": 8, "mode": "adaptive"}),
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    shard_count = 0
    raw_count = 0
    for key in sorted(iter_jsonl_objects(s3, bucket=args.bucket, prefix=args.prefix)):
        shard_count += 1
        for row in read_jsonl_s3(s3, bucket=args.bucket, key=key):
            raw_count += 1
            marker = dedupe_key(row, fields)
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "bucket": args.bucket,
        "prefix": args.prefix,
        "shard_count": shard_count,
        "raw_count": raw_count,
        "dedupe_fields": fields,
        "row_count": len(rows),
        "output": str(output_path),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
