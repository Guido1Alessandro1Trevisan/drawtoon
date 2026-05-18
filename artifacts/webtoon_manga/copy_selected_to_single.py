#!/usr/bin/env python3
"""Copy selected cleaned WEBTOON episodes into the canonical pages/single layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config


ROOT = Path(__file__).resolve().parent
DEFAULT_SELECTED = ROOT / "selected_5k_per_series_cleaned_dim_v2.jsonl"
DEFAULT_REPORT = ROOT / "selected_5k_per_series_single_copy_report.csv"
DEFAULT_SUMMARY = ROOT / "selected_5k_per_series_single_copy_summary.json"
DEFAULT_BUCKET = "drawtoon"
DEFAULT_SOURCE_PREFIX = "datasets/pages/source/webtoon_cleaned_dim_v2"
DEFAULT_DEST_PREFIX = "datasets/pages/single"
PAGE_RE = re.compile(r"/(page-\d+\.[^.]+)$", re.IGNORECASE)


def load_selected(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def list_episode_objects(s3: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/") or "/_status/" in key:
                continue
            if not key.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            objects.append({"key": key, "size": int(obj.get("Size") or 0)})
    objects.sort(key=lambda item: item["key"])
    return objects


def dest_key_for(*, dest_prefix: str, series_slug: str, episode_slug: str, source_key: str) -> str:
    match = PAGE_RE.search(source_key)
    if not match:
        raise ValueError(f"cannot infer page filename from {source_key}")
    filename = match.group(1)
    return f"{dest_prefix.strip('/')}/{series_slug}_manwa/{episode_slug}__{filename}"


def copy_one(s3: Any, *, bucket: str, source_key: str, dest_key: str, dry_run: bool) -> dict[str, Any]:
    if not dry_run:
        s3.copy_object(
            Bucket=bucket,
            Key=dest_key,
            CopySource={"Bucket": bucket, "Key": source_key},
            MetadataDirective="COPY",
        )
    return {"source_key": source_key, "dest_key": dest_key}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", default=str(DEFAULT_SELECTED))
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    started = time.time()
    session = boto3.Session(region_name="us-east-1")
    s3 = session.client(
        "s3",
        config=Config(
            max_pool_connections=max(128, int(args.workers) + 16),
            connect_timeout=5,
            read_timeout=60,
            retries={"mode": "adaptive", "max_attempts": 8},
        ),
    )
    selected = load_selected(Path(args.selected_jsonl))
    copy_jobs: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for item in selected:
        episode = item["episode"]
        selection = item.get("selection") or {}
        series_slug = episode["series_slug"]
        episode_slug = episode["slug"]
        source_prefix = f"{args.source_prefix.strip('/')}/{series_slug}/{episode_slug}"
        objects = list_episode_objects(s3, bucket=args.bucket, prefix=source_prefix)
        for obj in objects:
            copy_jobs.append(
                {
                    "series_slug": series_slug,
                    "series_name": episode.get("series_name", ""),
                    "episode_no": int(episode["episode_no"]),
                    "episode_slug": episode_slug,
                    "source_key": obj["key"],
                    "dest_key": dest_key_for(
                        dest_prefix=args.dest_prefix,
                        series_slug=series_slug,
                        episode_slug=episode_slug,
                        source_key=obj["key"],
                    ),
                    "size": obj["size"],
                }
            )
        episode_rows.append(
            {
                "series_slug": series_slug,
                "episode_no": int(episode["episode_no"]),
                "episode_slug": episode_slug,
                "expected_kept_images": int(selection.get("kept_images") or 0),
                "listed_source_images": len(objects),
            }
        )

    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(
                copy_one,
                s3,
                bucket=args.bucket,
                source_key=job["source_key"],
                dest_key=job["dest_key"],
                dry_run=bool(args.dry_run),
            ): job
            for job in copy_jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                future.result()
                completed.append(job)
            except Exception as exc:
                errors.append({**job, "error": str(exc)[:500]})
            if index % 2500 == 0:
                print(
                    json.dumps(
                        {
                            "copied": len(completed),
                            "errors": len(errors),
                            "total": len(copy_jobs),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["series_slug", "series_name", "episode_no", "episode_slug", "source_key", "dest_key", "size"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed)

    by_series: dict[str, dict[str, Any]] = {}
    for row in completed:
        summary = by_series.setdefault(
            row["series_slug"],
            {"series_slug": row["series_slug"], "series_name": row["series_name"], "episodes": set(), "images": 0, "bytes": 0},
        )
        summary["episodes"].add(int(row["episode_no"]))
        summary["images"] += 1
        summary["bytes"] += int(row["size"])
    series_summaries = []
    for summary in by_series.values():
        episodes = sorted(summary.pop("episodes"))
        summary["episode_count"] = len(episodes)
        summary["episode_numbers"] = episodes
        series_summaries.append(summary)
    series_summaries.sort(key=lambda row: row["series_slug"])

    output = {
        "dry_run": bool(args.dry_run),
        "bucket": args.bucket,
        "source_prefix": args.source_prefix.strip("/"),
        "dest_prefix": args.dest_prefix.strip("/"),
        "selected_episodes": len(selected),
        "episode_rows": episode_rows,
        "copy_jobs": len(copy_jobs),
        "copied": len(completed),
        "errors": len(errors),
        "error_examples": errors[:20],
        "series": series_summaries,
        "elapsed_seconds": round(time.time() - started, 3),
        "report": str(report_path),
    }
    Path(args.summary).write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
