"""Lambda handlers for the build_panel_dataset Distributed Map workflow.

prepare_build_config  — enumerate pages with BOTH source caption + short
                        caption available, write manifest + worker_config.
build_page_panels     — per-page worker: read both JSONs, build rows, write
                        a fragment JSONL to s3://<output_bucket>/<output_root>/
                        _fragments/<run_id>/<chapter>__<page>.jsonl, return
                        compact stats so the Map's ResultWriter stays small.
finalize_dataset      — list all fragments, stream them into train.jsonl +
                        val.jsonl (split by chapter hash), write audit stats.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .assembly import build_rows_for_page, chapter_is_val


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET_NAME", "drawtoon")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET_NAME", "drawtoon-layousyn")

DEFAULT_SOURCE_PREFIX = "captions"
DEFAULT_SOURCE_RUN = "gemini3_flash_page_panel_v1"
DEFAULT_SHORT_PREFIX = "captions_short"
DEFAULT_SHORT_RUN = "vision_v1"
DEFAULT_OUTPUT_PREFIX = "datasets/panel_layout"
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_BUILD_MAX_CONCURRENCY", "1000"))
DEFAULT_SPLIT_VAL_FRAC = float(os.environ.get("DEFAULT_SPLIT_VAL_FRAC", "0.05"))
FINALIZE_PARALLELISM = int(os.environ.get("FINALIZE_PARALLELISM", "256"))

_S3_CLIENT = None


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "adaptive", "max_attempts": 10},
                max_pool_connections=512,
                connect_timeout=10,
                read_timeout=300,
            ),
        )
    return _S3_CLIENT


def _json_dumps(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None) + "\n"


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_token() -> str:
    return f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000, 9999)}"


def _normalize_prefix(value: object, *, default: str) -> str:
    prefix = str(value or default).strip().strip("/")
    if not prefix:
        raise ValueError("S3 prefix must not be empty")
    return prefix


def _join_key(*parts: object) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _get_s3_bytes(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _put_s3_bytes(bucket: str, key: str, body: bytes, *, content_type: str) -> None:
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def _object_exists(bucket: str, key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _get_s3_json(bucket: str, key: str) -> dict[str, Any]:
    text = _get_s3_bytes(bucket, key).decode("utf-8").strip()
    if not text:
        raise ValueError(f"Empty JSON at {_join_s3_uri(bucket, key)}")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {_join_s3_uri(bucket, key)}")
    return payload


def _list_relative_jsons(*, bucket: str, root_prefix: str) -> set[str]:
    prefix = root_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    out: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if "/_jobs/" in key or "/_audit/" in key or "/_fragments/" in key:
                continue
            if not key.endswith(".json"):
                continue
            out.add(key[len(prefix):])
    return out


# =========================================================================
# prepare_build_config
# =========================================================================


def prepare_build_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    source_bucket = str(event.get("source_bucket") or SOURCE_BUCKET).strip()
    short_bucket = str(event.get("short_bucket") or OUTPUT_BUCKET).strip()
    output_bucket = str(event.get("output_bucket") or OUTPUT_BUCKET).strip()
    source_prefix = _normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_PREFIX)
    source_caption_run = str(event.get("source_caption_run") or DEFAULT_SOURCE_RUN).strip().strip("/")
    short_prefix = _normalize_prefix(event.get("short_prefix"), default=DEFAULT_SHORT_PREFIX)
    short_caption_run = str(event.get("short_caption_run") or DEFAULT_SHORT_RUN).strip().strip("/")
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))
    split_val_frac = float(event.get("split_val_frac") if event.get("split_val_frac") is not None else DEFAULT_SPLIT_VAL_FRAC)
    overwrite = bool(event.get("overwrite", False))

    source_root = _join_key(source_prefix, source_caption_run)
    short_root = _join_key(short_prefix, short_caption_run)
    fragments_root = _join_key(output_prefix, "_fragments", run_id)

    source_rels = _list_relative_jsons(bucket=source_bucket, root_prefix=source_root)
    short_rels = _list_relative_jsons(bucket=short_bucket, root_prefix=short_root)
    intersection = sorted(source_rels & short_rels)

    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    if include_re:
        intersection = [r for r in intersection if include_re.search(r.split("/", 1)[0])]
    if max_pages > 0:
        intersection = intersection[:max_pages]

    rows: list[dict[str, Any]] = []
    for relative in intersection:
        parts = relative.split("/")
        if len(parts) != 2:
            continue
        chapter, filename = parts
        page_id = Path(filename).stem
        rows.append(
            {
                "chapter": chapter,
                "page_id": page_id,
                "relative_path": relative,
                "source_bucket": source_bucket,
                "short_bucket": short_bucket,
                "output_bucket": output_bucket,
                "source_caption_key": _join_key(source_root, relative),
                "short_caption_key": _join_key(short_root, relative),
                "fragment_key": _join_key(fragments_root, f"{chapter}__{page_id}.jsonl"),
            }
        )

    manifest_key = _join_key(output_prefix, "_jobs", run_id, "page_manifest.jsonl")
    manifest_body = "".join(_json_dumps(row) for row in rows).encode("utf-8")
    _put_s3_bytes(output_bucket, manifest_key, manifest_body, content_type="application/x-ndjson")

    worker_config = {
        "source_bucket": source_bucket,
        "short_bucket": short_bucket,
        "output_bucket": output_bucket,
        "source_root": source_root,
        "short_root": short_root,
        "output_root": output_prefix,
        "fragments_root": fragments_root,
        "run_id": run_id,
        "split_val_frac": split_val_frac,
        "overwrite": overwrite,
        "created_at": _now_utc_iso(),
        "git_sha": str(event.get("git_sha") or ""),
    }
    config_key = _join_key(output_prefix, "_jobs", run_id, "worker_config.json")
    _put_s3_bytes(output_bucket, config_key, _json_dumps(worker_config, pretty=True).encode("utf-8"), content_type="application/json")

    return {
        "schema_version": 1,
        "source": {"bucket": output_bucket, "page_manifest_key": manifest_key},
        "worker_config": {"bucket": output_bucket, "key": config_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": output_bucket, "prefix": _join_key(output_prefix, "_audit", run_id) + "/"},
        "output": {
            "bucket": output_bucket,
            "prefix": output_prefix,
            "run_id": run_id,
            "fragments_root": fragments_root,
            "split_val_frac": split_val_frac,
        },
        "stats": {
            "source_caption_count": len(source_rels),
            "short_caption_count": len(short_rels),
            "intersection_count": len(intersection),
            "manifest_count": len(rows),
        },
    }


# =========================================================================
# build_page_panels  (per-page worker)
# =========================================================================


def build_page_panels(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    page = event.get("page") if isinstance(event.get("page"), dict) else event

    source_bucket = str(page.get("source_bucket") or config["source_bucket"])
    short_bucket = str(page.get("short_bucket") or config["short_bucket"])
    output_bucket = str(page.get("output_bucket") or config["output_bucket"])
    fragment_key = str(page["fragment_key"])
    chapter = str(page.get("chapter") or "")
    page_id = str(page.get("page_id") or "")

    if not bool(config.get("overwrite")) and _object_exists(output_bucket, fragment_key):
        return {"status": "skipped_existing", "chapter": chapter, "page_id": page_id, "fragment_key": fragment_key}

    try:
        source_payload = _get_s3_json(source_bucket, str(page["source_caption_key"]))
    except Exception as exc:  # noqa: BLE001
        return {"status": "source_missing", "chapter": chapter, "page_id": page_id, "error": str(exc)}
    try:
        short_payload = _get_s3_json(short_bucket, str(page["short_caption_key"]))
    except Exception as exc:  # noqa: BLE001
        return {"status": "short_missing", "chapter": chapter, "page_id": page_id, "error": str(exc)}

    rows, stats = build_rows_for_page(
        source_payload=source_payload,
        short_payload=short_payload,
        chapter=chapter,
    )

    if rows:
        body = b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
        )
        _put_s3_bytes(output_bucket, fragment_key, body, content_type="application/x-ndjson")

    return {
        "status": "ok",
        "chapter": chapter,
        "page_id": page_id,
        "fragment_key": fragment_key,
        "row_count": len(rows),
        "stats": stats,
    }


# =========================================================================
# finalize_dataset
# =========================================================================


def _list_fragment_keys(bucket: str, fragments_root: str) -> list[str]:
    prefix = fragments_root.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.endswith(".jsonl"):
                keys.append(key)
    return keys


def _chapter_from_fragment_key(fragments_root: str, key: str) -> str:
    name = key[len(fragments_root) + 1:]  # strip "<fragments_root>/"
    stem = name.split("/")[-1]  # "<chapter>__<page>.jsonl"
    # Recover the chapter prefix (everything before the last "__").
    if "__" in stem:
        return stem.rsplit("__", 1)[0]
    return stem


def _fetch_fragment(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def finalize_dataset(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("worker_config") if isinstance(event.get("worker_config"), dict) else {}
    if not config_ref:
        config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))

    output_bucket = str(config["output_bucket"])
    output_root = str(config["output_root"])
    fragments_root = str(config["fragments_root"])
    run_id = str(config["run_id"])
    split_val_frac = float(config.get("split_val_frac") or 0.0)

    fragment_keys = _list_fragment_keys(output_bucket, fragments_root)

    # Stream fragments through ThreadPoolExecutor; split rows into two buckets
    # by chapter hash; serialize each bucket to bytes; write to S3.
    train_buf = io.BytesIO()
    val_buf = io.BytesIO()
    train_rows = 0
    val_rows = 0
    fragment_count_with_rows = 0

    def _work(key: str) -> tuple[str, bytes]:
        return key, _fetch_fragment(output_bucket, key)

    with ThreadPoolExecutor(max_workers=FINALIZE_PARALLELISM) as pool:
        futures = [pool.submit(_work, k) for k in fragment_keys]
        for fut in as_completed(futures):
            key, body = fut.result()
            chapter = _chapter_from_fragment_key(fragments_root, key)
            is_val = chapter_is_val(chapter, split_val_frac)
            buf = val_buf if is_val else train_buf
            buf.write(body)
            lines = body.count(b"\n")
            if lines:
                fragment_count_with_rows += 1
                if is_val:
                    val_rows += lines
                else:
                    train_rows += lines

    train_key = _join_key(output_root, "train.jsonl")
    val_key = _join_key(output_root, "val.jsonl")
    audit_key = _join_key(output_root, "_audit", run_id, "stats.json")

    _put_s3_bytes(output_bucket, train_key, train_buf.getvalue(), content_type="application/x-ndjson")
    _put_s3_bytes(output_bucket, val_key, val_buf.getvalue(), content_type="application/x-ndjson")

    stats = {
        "run_id": run_id,
        "created_at": _now_utc_iso(),
        "fragments_total": len(fragment_keys),
        "fragments_with_rows": fragment_count_with_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "total_rows": train_rows + val_rows,
        "split_val_frac": split_val_frac,
        "output": {
            "bucket": output_bucket,
            "train_key": train_key,
            "val_key": val_key,
        },
    }
    _put_s3_bytes(output_bucket, audit_key, (json.dumps(stats, indent=2) + "\n").encode("utf-8"), content_type="application/json")

    return {
        "status": "ok",
        "train_uri": _join_s3_uri(output_bucket, train_key),
        "val_uri": _join_s3_uri(output_bucket, val_key),
        "audit_uri": _join_s3_uri(output_bucket, audit_key),
        "train_rows": train_rows,
        "val_rows": val_rows,
    }
