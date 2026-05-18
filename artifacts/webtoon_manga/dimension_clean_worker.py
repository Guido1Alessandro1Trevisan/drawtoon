#!/usr/bin/env python3
"""Dimension-first cleaner for downloaded WEBTOON/manhwa episode images.

The raw downloader intentionally preserves every viewer image. This worker
creates a cleaned copy prefix by keeping the dominant contiguous story-image
run for each episode and dropping small, landscape, width-outlier, warning, and
promo-like edge images.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


BUCKET = os.environ.get("DEST_BUCKET", "drawtoon")
SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "datasets/pages/source/webtoon")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "datasets/pages/source/webtoon_cleaned")
STATUS_PREFIX = os.environ.get("STATUS_PREFIX", "datasets/pages/source/webtoon_cleaned/_status")
DIMENSION_WORKERS = int(os.environ.get("DIMENSION_WORKERS", "24"))
COPY_WORKERS = int(os.environ.get("COPY_WORKERS", "24"))
EDGE_HEAD = int(os.environ.get("EDGE_HEAD", "40"))
EDGE_TAIL = int(os.environ.get("EDGE_TAIL", "80"))
WINDOW = int(os.environ.get("STORY_WINDOW", "12"))
WINDOW_MIN_GOOD = int(os.environ.get("STORY_WINDOW_MIN_GOOD", "7"))
MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "100"))
MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "80"))
MIN_RATIO = float(os.environ.get("MIN_RATIO", "0.62"))
MAX_RATIO = float(os.environ.get("MAX_RATIO", "7.5"))
WIDTH_TOLERANCE_RATIO = float(os.environ.get("WIDTH_TOLERANCE_RATIO", "0.08"))
WIDTH_TOLERANCE_PX = int(os.environ.get("WIDTH_TOLERANCE_PX", "28"))
S3 = boto3.client(
    "s3",
    config=Config(
        max_pool_connections=max(64, DIMENSION_WORKERS + COPY_WORKERS + 8),
        connect_timeout=5,
        read_timeout=25,
        retries={"max_attempts": 5, "mode": "adaptive"},
    ),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PAGE_RE = re.compile(r"/page-(\d+)\.[^.]+$", re.IGNORECASE)
URL_REJECT_RE = re.compile(
    r"(?:thumb|thumbnail|language[-_]?warning|age[-_]?warning|warning|notice|recommend|"
    r"banner|download[-_]?app|app[-_]?download|subscribe|promotion|promo|"
    r"advert|credit|credits|author[-_]?note|creator[-_]?note|thanks|"
    r"instagram|facebook|twitter|youtube|wallpaper)",
    re.IGNORECASE,
)


@dataclass
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


def page_index(key: str) -> int:
    match = PAGE_RE.search(key)
    return int(match.group(1)) if match else 10**12


def has_image_suffix(key: str) -> bool:
    lower = key.lower()
    return any(lower.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def episode_prefix(source_prefix: str, episode: dict[str, Any]) -> str:
    return f"{source_prefix.strip('/')}/{episode['series_slug']}/{episode['slug']}/"


def list_episode_objects(bucket: str, prefix: str) -> list[ImageRow]:
    rows: list[ImageRow] = []
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/") or not has_image_suffix(key):
                continue
            rows.append(ImageRow(index=page_index(key), key=key, size=int(obj.get("Size") or 0)))
    rows.sort(key=lambda row: (row.index, row.key))
    return rows


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
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                return int.from_bytes(data[index + 5 : index + 7], "big"), int.from_bytes(data[index + 3 : index + 5], "big")
            index += segment_length
    raise ValueError("unsupported image header or missing dimensions")


def read_dimensions(bucket: str, row: ImageRow) -> ImageRow:
    try:
        response = S3.get_object(Bucket=bucket, Key=row.key, Range="bytes=0-131071")
        data = response["Body"].read()
        try:
            width, height = parse_image_dimensions(data)
        except Exception:
            response = S3.get_object(Bucket=bucket, Key=row.key, Range="bytes=0-1048575")
            data = response["Body"].read()
            width, height = parse_image_dimensions(data)
        row.width = int(width)
        row.height = int(height)
        row.ratio = float(height) / max(1, float(width))
        metadata = response.get("Metadata") or {}
        row.source_url = str(metadata.get("source-url") or "")
        row.content_type = str(response.get("ContentType") or "")
    except Exception as exc:
        row.status = "drop"
        row.reason = "dimension_error"
        row.error = str(exc)[:500]
    return row


def dominant_width(rows: list[ImageRow]) -> int:
    counts: Counter[int] = Counter()
    for row in rows:
        if row.source_url and URL_REJECT_RE.search(row.source_url):
            continue
        if row.width >= MIN_WIDTH and row.height >= MIN_HEIGHT:
            counts[int(round(row.width / 10.0) * 10)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    fallback = sorted(row.width for row in rows if row.width > 0)
    return fallback[len(fallback) // 2] if fallback else 0


def base_candidate(row: ImageRow, dom_width: int) -> tuple[bool, str]:
    if row.reason == "dimension_error":
        return False, row.reason
    if row.width < MIN_WIDTH or row.height < MIN_HEIGHT:
        return False, "too_small"
    if row.ratio < MIN_RATIO:
        return False, "ratio_low"
    if row.ratio > MAX_RATIO:
        return False, "ratio_high"
    if dom_width and abs(row.width - dom_width) > max(WIDTH_TOLERANCE_PX, int(dom_width * WIDTH_TOLERANCE_RATIO)):
        return False, "width_outlier"
    if row.source_url and URL_REJECT_RE.search(row.source_url):
        return False, "url_reject"
    return True, "story_dimension_match"


def find_story_bounds(flags: list[bool]) -> tuple[int, int]:
    if not flags:
        return 0, -1
    if sum(flags) < max(3, WINDOW_MIN_GOOD):
        good = [i for i, flag in enumerate(flags) if flag]
        return (good[0], good[-1]) if good else (0, -1)

    start = 0
    for i in range(len(flags)):
        window = flags[i : i + WINDOW]
        if sum(window) >= min(WINDOW_MIN_GOOD, len(window)):
            start = i
            break

    end = len(flags) - 1
    for i in range(len(flags) - 1, -1, -1):
        window = flags[max(0, i - WINDOW + 1) : i + 1]
        if sum(window) >= min(WINDOW_MIN_GOOD, len(window)):
            end = i
            break
    return start, end


def output_key_for(row: ImageRow, source_prefix: str, output_prefix: str) -> str:
    relative = row.key[len(source_prefix.strip("/") + "/") :] if row.key.startswith(source_prefix.strip("/") + "/") else row.key.rsplit("/", 1)[-1]
    return f"{output_prefix.strip('/')}/{relative}"


def copy_row(bucket: str, row: ImageRow) -> ImageRow:
    try:
        S3.copy_object(
            Bucket=bucket,
            Key=row.output_key,
            CopySource={"Bucket": bucket, "Key": row.key},
            MetadataDirective="COPY",
        )
    except ClientError as exc:
        row.status = "copy_error"
        row.reason = "copy_error"
        row.error = str(exc)[:500]
    return row


def write_status(bucket: str, status_prefix: str, episode: dict[str, Any], payload: dict[str, Any]) -> None:
    key = f"{status_prefix.strip('/')}/{episode['series_slug']}/{episode['slug']}.json"
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    started = time.time()
    episode = event.get("episode") or event.get("item") or event
    bucket = str(event.get("bucket") or BUCKET)
    source_prefix = str(event.get("source_prefix") or SOURCE_PREFIX).strip("/")
    output_prefix = str(event.get("output_prefix") or OUTPUT_PREFIX).strip("/")
    status_prefix = str(event.get("status_prefix") or STATUS_PREFIX).strip("/")
    dry_run = bool(event.get("dry_run", False))
    max_images = int(event.get("max_images") or 0)

    rows = list_episode_objects(bucket, episode_prefix(source_prefix, episode))
    if max_images > 0:
        rows = rows[:max_images]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, DIMENSION_WORKERS)) as pool:
        rows = list(pool.map(lambda row: read_dimensions(bucket, row), rows))
    rows.sort(key=lambda row: (row.index, row.key))

    dom_width = dominant_width(rows)
    candidate_flags: list[bool] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        ok, reason = base_candidate(row, dom_width)
        candidate_flags.append(ok)
        row.reason = reason
        reasons[reason] += 1

    start_idx, end_idx = find_story_bounds(candidate_flags)
    kept: list[ImageRow] = []
    dropped: list[ImageRow] = []
    for pos, row in enumerate(rows):
        ok = candidate_flags[pos] and start_idx <= pos <= end_idx
        if ok:
            row.status = "keep"
            row.output_key = output_key_for(row, source_prefix, output_prefix)
            kept.append(row)
        else:
            row.status = "drop"
            if candidate_flags[pos] and not (start_idx <= pos <= end_idx):
                row.reason = "outside_story_run"
            dropped.append(row)

    copy_errors = 0
    if not dry_run and kept:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, COPY_WORKERS)) as pool:
            copied = list(pool.map(lambda row: copy_row(bucket, row), kept))
        copy_errors = sum(1 for row in copied if row.status == "copy_error")
        kept = [row for row in copied if row.status == "keep"]
        dropped.extend(row for row in copied if row.status == "copy_error")

    drop_reasons = Counter(row.reason for row in dropped)
    payload = {
        "schema_version": 1,
        "series_slug": episode["series_slug"],
        "episode_no": episode["episode_no"],
        "slug": episode["slug"],
        "source_prefix": source_prefix,
        "output_prefix": output_prefix,
        "dominant_width": dom_width,
        "story_start_position": start_idx + 1 if end_idx >= start_idx else None,
        "story_end_position": end_idx + 1 if end_idx >= start_idx else None,
        "input_images": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "copy_errors": copy_errors,
        "dry_run": dry_run,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "candidate_reasons": dict(sorted(reasons.items())),
        "dropped_examples": [asdict(row) for row in dropped[:20]],
        "kept_examples": [asdict(row) for row in kept[:5]],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_status(bucket, status_prefix, episode, payload)
    if copy_errors:
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return {
        "series_slug": episode["series_slug"],
        "episode_no": episode["episode_no"],
        "input_images": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "dominant_width": dom_width,
        "elapsed_seconds": payload["elapsed_seconds"],
    }
