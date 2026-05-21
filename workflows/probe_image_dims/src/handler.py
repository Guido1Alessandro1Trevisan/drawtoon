"""Lambda handler: probe a single S3 image's dimensions via range-bytes + PIL.

State machine ItemSelector wraps each row as:
  {"row_index": int, "item": {"key": "..."}, "config": {"bucket": "..."}}

Output (per row):
  {"key": "...", "width": int, "height": int, "bytes_read": int, "status": "ok"}
  {"key": "...", "status": "error", "error": "..."}
"""
from __future__ import annotations

import io
import os
from typing import Any

import boto3
from botocore.config import Config
from PIL import Image

_BUCKET_DEFAULT = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")

# Reusable client; Lambda reuses the same warm container for many invocations.
_S3 = boto3.client(
    "s3",
    config=Config(
        retries={"max_attempts": 4, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=15,
        max_pool_connections=8,
    ),
)

# Most JPEGs/PNGs have SOF in the first ~16 KB; if not, we grow once.
_FIRST_RANGE = 16 * 1024
_SECOND_RANGE = 128 * 1024


def _try_open(buf: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(buf)) as im:
            # PIL is lazy; force header decode.
            w, h = im.size
            if w and h:
                return int(w), int(h)
    except Exception:
        return None
    return None


def _get_range(bucket: str, key: str, n: int) -> bytes:
    resp = _S3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{n - 1}")
    return resp["Body"].read()


def probe_image_dims(event: dict[str, Any], _context) -> dict[str, Any]:
    item = event.get("item") or {}
    config = event.get("config") or {}
    bucket = config.get("bucket") or _BUCKET_DEFAULT
    key = item.get("key")
    if not key:
        return {"key": "", "status": "error", "error": "missing key"}

    try:
        buf = _get_range(bucket, key, _FIRST_RANGE)
        dims = _try_open(buf)
        bytes_read = len(buf)
        if dims is None:
            # Fallback: larger range.
            buf = _get_range(bucket, key, _SECOND_RANGE)
            dims = _try_open(buf)
            bytes_read = len(buf)
        if dims is None:
            return {"key": key, "status": "error", "error": "could not parse header", "bytes_read": bytes_read}
        w, h = dims
        return {
            "key": key,
            "status": "ok",
            "width": w,
            "height": h,
            "bytes_read": bytes_read,
        }
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:512]}
