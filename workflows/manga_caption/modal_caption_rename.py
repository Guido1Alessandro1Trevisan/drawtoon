"""One-shot Modal app: rename every S3 caption file from the unsuffixed
chapter convention (`<title>_<source>/<page>.json`) to the suffixed
convention (`<title>_<source>_manga/<page>.json`) that matches the
durable pages / annotations layout. Also updates the in-file `chapter`
field and any stale `sources.page_key` / `sources.annotation_key`
strings, then deletes the old object.

Idempotent: if the chapter folder already ends in `_manga`, the file is
skipped. Skips admin folders (`_jobs`, `_audit`, `_failed`, `_smoke`,
`_status`, `_outliers`, `_probe`, `_artifact_*`).

Usage::

    modal deploy workflows/manga_caption/modal_caption_rename.py
    modal run workflows/manga_caption/modal_caption_rename.py::rename_all
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import modal


BUCKET = os.environ.get("DRAWTOON_S3_BUCKET", "drawtoon")
CAPTIONS_PREFIX = "captions"
AWS_SECRET_NAME = os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3")

MAX_CONTAINERS = int(os.environ.get("CAPTION_RENAME_MAX_CONTAINERS", "80"))
INPUTS_PER_CONTAINER = int(os.environ.get("CAPTION_RENAME_INPUTS_PER_CONTAINER", "30"))

EXCLUDED_PREFIXES = ("_",)
EXCLUDED_NAMES = frozenset({"artifacts"})


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3==1.35.99", "botocore==1.35.99")
)

app = modal.App("drawtoon-caption-rename", image=image)
aws_secret = modal.Secret.from_name(AWS_SECRET_NAME)


def _s3():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            retries={"mode": "adaptive", "max_attempts": 10},
            connect_timeout=10,
            read_timeout=60,
            max_pool_connections=64,
        ),
    )


def _is_admin_chapter(chapter: str) -> bool:
    if not chapter:
        return True
    if chapter in EXCLUDED_NAMES:
        return True
    if chapter.startswith(EXCLUDED_PREFIXES):
        return True
    return False


@app.function(
    cpu=2.0,
    memory=4096,
    secrets=[aws_secret],
    timeout=1800,
)
def list_caption_jsons() -> list[dict[str, str]]:
    """List every caption JSON under captions/*/<chapter>/<page>.json where
    chapter does NOT end in `_manga`. Returns rows of {run, chapter, key}."""
    client = _s3()
    paginator = client.get_paginator("list_objects_v2")
    rows: list[dict[str, str]] = []
    seen = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{CAPTIONS_PREFIX}/"):
        for obj in page.get("Contents") or []:
            key = obj.get("Key") or ""
            if not key.endswith(".json"):
                continue
            seen += 1
            # captions/<run>/<chapter>/<page>.json
            parts = key.split("/")
            if len(parts) != 4:
                continue
            _, run, chapter, _ = parts
            if _is_admin_chapter(run) or _is_admin_chapter(chapter):
                continue
            if chapter.endswith("_manga"):
                continue
            rows.append({"key": key, "run": run, "chapter": chapter})
    return rows


def _rewrite_chapter_in_payload(payload: dict[str, Any], new_chapter: str, old_chapter: str) -> dict[str, Any]:
    payload = dict(payload)
    payload["chapter"] = new_chapter
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else None
    if isinstance(sources, dict):
        sources = dict(sources)
        for k, v in list(sources.items()):
            if not isinstance(v, str) or not v:
                continue
            # replace `/<old_chapter>/` only when it appears as a path segment
            sources[k] = re.sub(rf"/{re.escape(old_chapter)}/", f"/{new_chapter}/", v)
        payload["sources"] = sources
    return payload


@app.function(
    cpu=0.5,
    memory=512,
    secrets=[aws_secret],
    max_containers=MAX_CONTAINERS,
    timeout=300,
)
@modal.concurrent(max_inputs=INPUTS_PER_CONTAINER)
def rename_one(record: dict[str, str]) -> dict[str, Any]:
    """For one caption file: read, rewrite chapter+sources, write to the
    `_manga`-suffixed path, delete the original."""
    client = _s3()
    src_key = record["key"]
    chapter = record["chapter"]
    new_chapter = f"{chapter}_manga"
    parts = src_key.split("/")
    parts[2] = new_chapter
    dst_key = "/".join(parts)

    if dst_key == src_key:
        return {"key": src_key, "status": "noop"}

    try:
        obj = client.get_object(Bucket=BUCKET, Key=src_key)
        body = obj["Body"].read().decode("utf-8")
        payload = json.loads(body)
    except Exception as exc:
        return {"key": src_key, "status": "read_error", "error": repr(exc)[:200]}

    if not isinstance(payload, dict):
        return {"key": src_key, "status": "skip_not_dict"}

    payload = _rewrite_chapter_in_payload(payload, new_chapter=new_chapter, old_chapter=chapter)

    try:
        body_out = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        client.put_object(
            Bucket=BUCKET,
            Key=dst_key,
            Body=body_out,
            ContentType="application/json; charset=utf-8",
        )
    except Exception as exc:
        return {"key": src_key, "status": "write_error", "error": repr(exc)[:200]}

    try:
        client.delete_object(Bucket=BUCKET, Key=src_key)
    except Exception as exc:
        return {
            "key": src_key,
            "dst_key": dst_key,
            "status": "delete_error",
            "error": repr(exc)[:200],
        }

    return {"key": src_key, "dst_key": dst_key, "status": "ok"}


@app.local_entrypoint()
def rename_all(progress_every: int = 5000):
    print(json.dumps({"event": "listing"}), flush=True)
    rows = list_caption_jsons.remote()
    total = len(rows)
    print(
        json.dumps(
            {
                "event": "listed",
                "total_keys": total,
                "max_containers": MAX_CONTAINERS,
                "inputs_per_container": INPUTS_PER_CONTAINER,
                "target_concurrency": MAX_CONTAINERS * INPUTS_PER_CONTAINER,
            }
        ),
        flush=True,
    )
    if not rows:
        print(json.dumps({"event": "done", "ok": 0, "errors": 0, "wall_sec": 0.0}), flush=True)
        return

    ok = 0
    errors = 0
    error_samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_logged = 0
    for result in rename_one.map(rows, order_outputs=False):
        if result.get("status") == "ok":
            ok += 1
        elif result.get("status") == "noop":
            pass
        else:
            errors += 1
            if len(error_samples) < 10:
                error_samples.append(result)
        processed = ok + errors
        if processed - last_logged >= progress_every or processed == total:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": total,
                        "ok": ok,
                        "errors": errors,
                        "elapsed_sec": round(elapsed, 1),
                        "files_per_sec": round(processed / elapsed, 1) if elapsed > 0 else None,
                    }
                ),
                flush=True,
            )
            last_logged = processed

    print(
        json.dumps(
            {
                "event": "done",
                "ok": ok,
                "errors": errors,
                "wall_sec": round(time.perf_counter() - started, 1),
                "error_samples": error_samples,
            }
        ),
        flush=True,
    )
