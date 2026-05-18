#!/usr/bin/env python3
"""Promote filtered reader-site pages back into datasets/pages/single.

This makes datasets/pages/single/<series>_manwa contain only the pages that are
currently present in datasets/pages/single_relevant/<series>_manwa, then can
delete the single_relevant prefix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from typing import Any

import boto3
from botocore.config import Config


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_RAW_PREFIX = "datasets/pages/single"
DEFAULT_RELEVANT_PREFIX = "datasets/pages/single_relevant"
DEFAULT_MANIFEST_PREFIX = "datasets/pages/single/_manifests"
DEFAULT_SERIES = (
    "solo-leveling_manwa",
    "sss-class-suicide-hunter_manwa",
    "second-life-ranker_manwa",
    "a-returners-magic-should-be-special_manwa",
    "the-great-mage-returns-after-4000-years_manwa",
    "lout-of-counts-family_manwa",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=128,
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )


def is_page_key(key: str) -> bool:
    lower = key.lower()
    return any(lower.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def list_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key and is_page_key(key):
                keys.append(key)
    return keys


def list_all_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key:
                keys.append(key)
    return keys


def batch_delete(client: Any, bucket: str, keys: list[str]) -> int:
    deleted = 0
    for index in range(0, len(keys), 1000):
        chunk = keys[index : index + 1000]
        if not chunk:
            continue
        response = client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True})
        errors = response.get("Errors") or []
        if errors:
            raise RuntimeError(f"delete errors: {errors[:5]}")
        deleted += len(chunk)
    return deleted


def copy_missing(client: Any, bucket: str, missing_raw_keys: list[str], raw_prefix: str, relevant_prefix: str) -> int:
    copied = 0
    raw_root = raw_prefix.rstrip("/") + "/"
    relevant_root = relevant_prefix.rstrip("/") + "/"
    for raw_key in missing_raw_keys:
        source_key = relevant_root + raw_key[len(raw_root) :]
        client.copy_object(
            Bucket=bucket,
            Key=raw_key,
            CopySource={"Bucket": bucket, "Key": source_key},
            MetadataDirective="COPY",
        )
        copied += 1
    return copied


def put_manifest(client: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--relevant-prefix", default=DEFAULT_RELEVANT_PREFIX)
    parser.add_argument("--manifest-prefix", default=DEFAULT_MANIFEST_PREFIX)
    parser.add_argument("--series", action="append", default=[])
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--delete-relevant", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or "promote_single_relevant_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    series_list = args.series or list(DEFAULT_SERIES)
    raw_root = args.raw_prefix.rstrip("/") + "/"
    relevant_root = args.relevant_prefix.rstrip("/") + "/"
    client = s3_client()

    per_series: dict[str, dict[str, int]] = {}
    extra_raw_keys: list[str] = []
    missing_raw_keys: list[str] = []
    started = time.monotonic()
    for series in series_list:
        raw_prefix = raw_root + series
        relevant_prefix = relevant_root + series
        raw_keys = list_keys(client, args.bucket, raw_prefix)
        relevant_keys = list_keys(client, args.bucket, relevant_prefix)
        raw_rel = {key[len(raw_root) :] for key in raw_keys}
        relevant_rel = {key[len(relevant_root) :] for key in relevant_keys}
        extras = sorted(raw_rel - relevant_rel)
        missing = sorted(relevant_rel - raw_rel)
        extra_raw_keys.extend(raw_root + key for key in extras)
        missing_raw_keys.extend(raw_root + key for key in missing)
        per_series[series] = {
            "raw_pages_before": len(raw_keys),
            "relevant_pages": len(relevant_keys),
            "raw_extra_pages_to_delete": len(extras),
            "relevant_pages_missing_from_raw": len(missing),
            "expected_raw_pages_after": len(relevant_keys),
        }

    relevant_all_keys = list_all_keys(client, args.bucket, args.relevant_prefix)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bucket": args.bucket,
        "raw_prefix": args.raw_prefix.rstrip("/"),
        "relevant_prefix": args.relevant_prefix.rstrip("/"),
        "series": series_list,
        "execute": bool(args.execute),
        "delete_relevant": bool(args.delete_relevant),
        "per_series": per_series,
        "totals": {
            "raw_extra_pages_to_delete": len(extra_raw_keys),
            "relevant_pages_missing_from_raw": len(missing_raw_keys),
            "relevant_prefix_objects_to_delete": len(relevant_all_keys) if args.delete_relevant else 0,
        },
        "elapsed_seconds_before_mutation": round(time.monotonic() - started, 3),
    }

    if args.execute:
        copied = copy_missing(client, args.bucket, missing_raw_keys, raw_root, relevant_root)
        deleted_extras = batch_delete(client, args.bucket, extra_raw_keys)
        deleted_relevant = batch_delete(client, args.bucket, relevant_all_keys) if args.delete_relevant else 0
        summary["mutation"] = {
            "copied_missing_relevant_pages_to_raw": copied,
            "deleted_extra_raw_pages": deleted_extras,
            "deleted_relevant_prefix_objects": deleted_relevant,
        }
    else:
        summary["mutation"] = {"dry_run": True}

    manifest_key = f"{args.manifest_prefix.rstrip('/')}/{run_id}/summary.json"
    put_manifest(client, args.bucket, manifest_key, summary)
    summary["manifest"] = f"s3://{args.bucket}/{manifest_key}"
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
