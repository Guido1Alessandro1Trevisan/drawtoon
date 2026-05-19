#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".avif")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def output_prefix(row: dict[str, Any], default_bucket: str) -> tuple[str, str]:
    bucket = str(row.get("bucket") or default_bucket)
    prefix = str(row.get("output_prefix") or "").strip("/")
    if prefix.startswith("s3://"):
        parsed = urlparse(prefix)
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
    return bucket, prefix


def count_prefix(s3, bucket: str, prefix: str) -> int:
    total = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "").lower()
            if key.endswith(IMAGE_SUFFIXES):
                total += 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize S3 object counts for a download_scraper manifest.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--bucket", default="drawtoon")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    rows = read_rows(Path(args.manifest_path))
    prefixes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        bucket, prefix = output_prefix(row, args.bucket)
        group = "/".join(part for part in [str(row.get("platform") or ""), str(row.get("series_slug") or ""), str(row.get("issue_slug") or "")] if part)
        prefixes[(bucket, prefix)].add(group)

    s3 = boto3.client("s3", region_name=args.region)
    total = 0
    for bucket, prefix in sorted(prefixes):
        count = count_prefix(s3, bucket, prefix)
        total += count
        print(f"s3://{bucket}/{prefix}/ {count}", flush=True)
    print(f"total {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

