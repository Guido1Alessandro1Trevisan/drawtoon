#!/usr/bin/env python3
"""Copy likely story pages from raw authorized-reader S3 prefixes.

Raw reader-site downloads intentionally keep every page-like image they can see.
This pass leaves that raw data intact and creates a filtered copy under
datasets/pages/single_relevant/<series>_manwa/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/single_relevant"
DEFAULT_STATUS_PREFIX = "datasets/pages/single_relevant/_status"
DEFAULT_MANIFEST_DIR = Path("artifacts/separata_manwa/manifests")
DEFAULT_SERIES = (
    "solo-leveling_manwa",
    "sss-class-suicide-hunter_manwa",
    "second-life-ranker_manwa",
    "a-returners-magic-should-be-special_manwa",
    "the-great-mage-returns-after-4000-years_manwa",
    "lout-of-counts-family_manwa",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PAGE_RE = re.compile(r"/page-(\d+)\.[^.]+$", re.IGNORECASE)
URL_REJECT_RE = re.compile(
    r"(?:language[-_]?warning|age[-_]?warning|warning|notice|recommend|"
    r"banner|download[-_]?app|app[-_]?download|subscribe|promotion|promo|"
    r"advert|credit|credits|author[-_]?note|creator[-_]?note|thanks|"
    r"instagram|facebook|twitter|youtube|wallpaper|logo|cover)",
    re.IGNORECASE,
)


@dataclasses.dataclass
class ImageRow:
    index: int
    key: str
    size: int
    width: int = 0
    height: int = 0
    ratio: float = 0.0
    source_url: str = ""
    content_type: str = ""
    status: str = "undecided"
    reason: str = ""
    output_key: str = ""
    error: str = ""


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=256,
            connect_timeout=8,
            read_timeout=60,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )


def s3_call(operation: str, func, *args, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= 6:
                break
            time.sleep(min(20.0, 0.7 * attempt + random.random()))
    raise RuntimeError(f"S3 {operation} failed after retries: {last_error}")


def object_exists(client: Any, bucket: str, key: str) -> bool:
    for attempt in range(1, 5):
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            if attempt >= 4:
                raise
        time.sleep(min(10.0, 0.6 * attempt + random.random()))
    return False


def page_index(key: str) -> int:
    match = PAGE_RE.search(key)
    return int(match.group(1)) if match else 10**12


def has_image_suffix(key: str) -> bool:
    lower = key.lower()
    return any(lower.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def parse_image_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if len(data) >= 30 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if chunk == b"VP8 " and len(data) >= 30:
            start = 20
            return (
                int.from_bytes(data[start + 6 : start + 8], "little") & 0x3FFF,
                int.from_bytes(data[start + 8 : start + 10], "little") & 0x3FFF,
            )
        if chunk == b"VP8L" and len(data) >= 25:
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            return 1 + (((b1 & 0x3F) << 8) | b0), 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
    if len(data) >= 4 and data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            while index < len(data) and data[index] == 0xFF:
                index += 1
            if index >= len(data):
                break
            marker = data[index]
            index += 1
            if marker in {0xD8, 0xD9, 0x01}:
                continue
            if index + 2 > len(data):
                break
            segment_length = int.from_bytes(data[index : index + 2], "big")
            if segment_length < 2:
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                return int.from_bytes(data[index + 5 : index + 7], "big"), int.from_bytes(data[index + 3 : index + 5], "big")
            index += segment_length
    raise ValueError("unsupported image header or missing dimensions")


def list_chapter_prefixes(client: Any, bucket: str, source_prefix: str, series: str) -> list[str]:
    root = f"{source_prefix.strip('/')}/{series.strip('/')}/"
    prefixes: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            prefix = str(item.get("Prefix") or "")
            if prefix and not prefix.rstrip("/").endswith("_manifests"):
                prefixes.append(prefix)
    return sorted(set(prefixes))


def list_images(client: Any, bucket: str, chapter_prefix: str) -> list[ImageRow]:
    rows: list[ImageRow] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=chapter_prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key and has_image_suffix(key):
                rows.append(ImageRow(index=page_index(key), key=key, size=int(obj.get("Size") or 0)))
    return sorted(rows, key=lambda row: (row.index, row.key))


def read_dimensions(client: Any, bucket: str, row: ImageRow) -> ImageRow:
    try:
        response = s3_call("get_object", client.get_object, Bucket=bucket, Key=row.key, Range="bytes=0-131071")
        data = response["Body"].read()
        try:
            width, height = parse_image_dimensions(data)
        except Exception:
            response = s3_call("get_object", client.get_object, Bucket=bucket, Key=row.key, Range="bytes=0-1048575")
            data = response["Body"].read()
            width, height = parse_image_dimensions(data)
        row.width = int(width)
        row.height = int(height)
        row.ratio = float(height) / max(1.0, float(width))
        metadata = response.get("Metadata") or {}
        row.source_url = str(metadata.get("source-url") or "")
        row.content_type = str(response.get("ContentType") or "")
    except Exception as exc:
        row.status = "drop"
        row.reason = "dimension_error"
        row.error = str(exc)[:500]
    return row


def dominant_width(rows: list[ImageRow], min_width: int, min_height: int) -> int:
    counts: Counter[int] = Counter()
    for row in rows:
        if row.width >= min_width and row.height >= min_height:
            counts[int(round(row.width / 10.0) * 10)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    widths = sorted(row.width for row in rows if row.width > 0)
    return widths[len(widths) // 2] if widths else 0


def base_candidate(row: ImageRow, dom_width: int, args: argparse.Namespace) -> tuple[bool, str]:
    if row.reason == "dimension_error":
        return False, row.reason
    if row.width < args.min_width or row.height < args.min_height:
        return False, "too_small"
    if row.ratio < args.min_ratio:
        return False, "ratio_low"
    if row.ratio > args.max_ratio:
        return False, "ratio_high"
    tolerance = max(args.width_tolerance_px, int(dom_width * args.width_tolerance_ratio))
    if dom_width and abs(row.width - dom_width) > tolerance:
        return False, "width_outlier"
    if row.source_url and URL_REJECT_RE.search(row.source_url):
        return False, "url_reject"
    return True, "story_dimension_match"


def story_bounds(flags: list[bool], window: int, min_good: int) -> tuple[int, int]:
    if not flags:
        return 0, -1
    if sum(flags) < max(3, min_good):
        good = [i for i, flag in enumerate(flags) if flag]
        return (good[0], good[-1]) if good else (0, -1)
    start = 0
    for index in range(len(flags)):
        current = flags[index : index + window]
        if sum(current) >= min(min_good, len(current)):
            start = index
            break
    end = len(flags) - 1
    for index in range(len(flags) - 1, -1, -1):
        current = flags[max(0, index - window + 1) : index + 1]
        if sum(current) >= min(min_good, len(current)):
            end = index
            break
    return start, end


def output_key_for(source_prefix: str, output_prefix: str, key: str) -> str:
    source = source_prefix.strip("/") + "/"
    if key.startswith(source):
        return f"{output_prefix.strip('/')}/{key[len(source):]}"
    return f"{output_prefix.strip('/')}/{key.rsplit('/', 1)[-1]}"


def copy_row(client: Any, bucket: str, row: ImageRow, overwrite: bool) -> ImageRow:
    try:
        if not overwrite and object_exists(client, bucket, row.output_key):
            row.status = "already_clean"
            return row
        s3_call(
            "copy_object",
            client.copy_object,
            Bucket=bucket,
            Key=row.output_key,
            CopySource={"Bucket": bucket, "Key": row.key},
            MetadataDirective="COPY",
        )
    except Exception as exc:
        row.status = "copy_error"
        row.reason = "copy_error"
        row.error = str(exc)[:500]
    return row


def clean_chapter(client: Any, bucket: str, chapter_prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    rows = list_images(client, bucket, chapter_prefix)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.dimension_workers)) as pool:
        rows = list(pool.map(lambda row: read_dimensions(client, bucket, row), rows))
    rows.sort(key=lambda row: (row.index, row.key))

    dom_width = dominant_width(rows, args.min_width, args.min_height)
    candidate_flags: list[bool] = []
    candidate_reasons: Counter[str] = Counter()
    for row in rows:
        ok, reason = base_candidate(row, dom_width, args)
        row.reason = reason
        candidate_flags.append(ok)
        candidate_reasons[reason] += 1

    start, end = story_bounds(candidate_flags, args.story_window, args.story_window_min_good)
    kept: list[ImageRow] = []
    dropped: list[ImageRow] = []
    for position, row in enumerate(rows):
        if candidate_flags[position] and start <= position <= end:
            row.status = "keep"
            row.output_key = output_key_for(args.source_prefix, args.output_prefix, row.key)
            kept.append(row)
        else:
            row.status = "drop"
            if candidate_flags[position] and not (start <= position <= end):
                row.reason = "outside_story_run"
            dropped.append(row)

    copied = 0
    already_clean = 0
    copy_errors = 0
    if kept and not args.dry_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.copy_workers)) as pool:
            copied_rows = list(pool.map(lambda row: copy_row(client, bucket, row, args.overwrite), kept))
        copied = sum(1 for row in copied_rows if row.status == "keep")
        already_clean = sum(1 for row in copied_rows if row.status == "already_clean")
        copy_errors = sum(1 for row in copied_rows if row.status == "copy_error")
        kept = copied_rows

    parts = chapter_prefix.strip("/").split("/")
    series = parts[-2] if len(parts) >= 2 else ""
    chapter = parts[-1] if parts else ""
    drop_reasons = Counter(row.reason for row in dropped)
    return {
        "status": "ok" if copy_errors == 0 else "copy_error",
        "series": series,
        "chapter": chapter,
        "source_prefix": chapter_prefix,
        "output_prefix": f"{args.output_prefix.strip('/')}/{series}/{chapter}/",
        "input_images": len(rows),
        "kept": sum(1 for row in kept if row.status in {"keep", "already_clean"}),
        "copied": copied,
        "already_clean": already_clean,
        "dropped": len(dropped),
        "copy_errors": copy_errors,
        "dominant_width": dom_width,
        "story_start_position": start + 1 if end >= start else None,
        "story_end_position": end + 1 if end >= start else None,
        "candidate_reasons": dict(sorted(candidate_reasons.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "dropped_examples": [dataclasses.asdict(row) for row in dropped[:12]],
        "kept_examples": [dataclasses.asdict(row) for row in kept[:5]],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_text(client: Any, bucket: str, key: str, text: str, content_type: str) -> None:
    s3_call("put_object", client.put_object, Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("input_images", "kept", "copied", "already_clean", "dropped", "copy_errors")
    result = {"chapters": len(rows)}
    for key in keys:
        result[key] = sum(int(row.get(key) or 0) for row in rows)
    result["ok"] = sum(1 for row in rows if row.get("status") == "ok")
    return result


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config = event.get("config") or {}
    chapter = event.get("chapter") or event.get("item") or event
    source_prefix = str(config.get("source_prefix") or DEFAULT_SOURCE_PREFIX)
    output_prefix = str(config.get("output_prefix") or DEFAULT_OUTPUT_PREFIX)
    status_prefix = str(config.get("status_prefix") or DEFAULT_STATUS_PREFIX)
    args = argparse.Namespace(
        bucket=str(config.get("bucket") or DEFAULT_BUCKET),
        source_prefix=source_prefix,
        output_prefix=output_prefix,
        status_prefix=status_prefix,
        dimension_workers=int(config.get("dimension_workers") or 12),
        copy_workers=int(config.get("copy_workers") or 12),
        min_width=int(config.get("min_width") or 220),
        min_height=int(config.get("min_height") or 120),
        min_ratio=float(config.get("min_ratio") or 0.45),
        max_ratio=float(config.get("max_ratio") or 12.0),
        width_tolerance_ratio=float(config.get("width_tolerance_ratio") or 0.10),
        width_tolerance_px=int(config.get("width_tolerance_px") or 40),
        story_window=int(config.get("story_window") or 10),
        story_window_min_good=int(config.get("story_window_min_good") or 5),
        dry_run=bool(config.get("dry_run") or False),
        overwrite=bool(config.get("overwrite") or False),
    )
    prefix = str(chapter.get("prefix") or chapter.get("source_prefix") or "")
    if not prefix:
        series = str(chapter["series"])
        chapter_slug = str(chapter["chapter"])
        prefix = f"{source_prefix.strip('/')}/{series}/{chapter_slug}/"
    return clean_chapter(s3_client(), args.bucket, prefix, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--status-prefix", default=DEFAULT_STATUS_PREFIX)
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--series", action="append", default=[], help="Series slug under datasets/pages/single. Repeatable.")
    parser.add_argument("--chapter-workers", type=int, default=16)
    parser.add_argument("--dimension-workers", type=int, default=12)
    parser.add_argument("--copy-workers", type=int, default=12)
    parser.add_argument("--min-width", type=int, default=220)
    parser.add_argument("--min-height", type=int, default=120)
    parser.add_argument("--min-ratio", type=float, default=0.45)
    parser.add_argument("--max-ratio", type=float, default=12.0)
    parser.add_argument("--width-tolerance-ratio", type=float, default=0.10)
    parser.add_argument("--width-tolerance-px", type=int, default=40)
    parser.add_argument("--story-window", type=int, default=10)
    parser.add_argument("--story-window-min-good", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload-manifest", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=20.0)
    args = parser.parse_args()

    run_id = args.run_id or "single_relevant_clean_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    series_list = args.series or list(DEFAULT_SERIES)
    manifest_dir = Path(args.manifest_dir)
    status_path = manifest_dir / f"{run_id}_status.jsonl"
    summary_path = manifest_dir / f"{run_id}_summary.json"
    client = s3_client()

    chapter_prefixes: list[str] = []
    for series in series_list:
        found = list_chapter_prefixes(client, args.bucket, args.source_prefix, series)
        print(f"discover: series={series} chapters={len(found)}", flush=True)
        chapter_prefixes.extend(found)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    last = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.chapter_workers)) as pool:
        futures = {pool.submit(clean_chapter, client, args.bucket, prefix, args): prefix for prefix in chapter_prefixes}
        for future in concurrent.futures.as_completed(futures):
            prefix = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                parts = prefix.strip("/").split("/")
                row = {
                    "status": "failed",
                    "series": parts[-2] if len(parts) >= 2 else "",
                    "chapter": parts[-1] if parts else "",
                    "source_prefix": prefix,
                    "input_images": 0,
                    "kept": 0,
                    "copied": 0,
                    "already_clean": 0,
                    "dropped": 0,
                    "copy_errors": 1,
                    "error": str(exc)[:500],
                }
            rows.append(row)
            now = time.monotonic()
            if now - last >= args.progress_interval or len(rows) == len(chapter_prefixes):
                last = now
                current = counts(rows)
                elapsed = max(0.1, now - started)
                rate = len(rows) / elapsed
                eta = max(0, len(chapter_prefixes) - len(rows)) / rate if rate > 0 else 0
                print(
                    "progress: "
                    f"chapters={len(rows)}/{len(chapter_prefixes)} input={current['input_images']} "
                    f"kept={current['kept']} copied={current['copied']} already_clean={current['already_clean']} "
                    f"dropped={current['dropped']} copy_errors={current['copy_errors']} "
                    f"chapter_rate={rate:.2f}/s eta={eta/60:.1f}m",
                    flush=True,
                )

    write_jsonl(status_path, rows)
    summary = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bucket": args.bucket,
        "source_prefix": args.source_prefix.strip("/"),
        "output_prefix": args.output_prefix.strip("/"),
        "series": series_list,
        "dry_run": bool(args.dry_run),
        "counts": counts(rows),
        "status_path": str(status_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary: " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.upload_manifest:
        manifest_prefix = f"{args.output_prefix.strip('/')}/_manifests/{run_id}"
        upload_text(client, args.bucket, f"{manifest_prefix}/status.jsonl", status_path.read_text(encoding="utf-8"), "application/x-jsonlines")
        upload_text(client, args.bucket, f"{manifest_prefix}/summary.json", summary_path.read_text(encoding="utf-8"), "application/json")
        print(f"uploaded_manifest: s3://{args.bucket}/{manifest_prefix}/", flush=True)


if __name__ == "__main__":
    main()
