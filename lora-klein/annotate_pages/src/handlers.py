"""Lambda entry points for the page-annotation Distributed Map.

Text-only Haiku 4.5 (Bedrock). For each page we read the Gemini v1 caption
JSON — which already has panels[] with bbox + characters + text_bubbles + the
long Gemini description — and ask Haiku to:
  - rewrite each panel's caption to 10-30 words
  - tag shot_size + panel_type (closed enum)
  - emit a 1-sentence page caption

NO image is sent. Output is enriched with MAGI counts (passed through) and a
FLUX.2 bucket (computed from each panel's bbox dimensions).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .bedrock_client import DEFAULT_MODEL, converse_tool
from .buckets import closest_bucket
from .overlay import manga_reading_order
from .prompt import (
    PANEL_TYPES,
    SHOT_SIZES,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .schema import (
    ANNOTATE_PAGE_TOOL_SCHEMA,
    TOOL_DESCRIPTION,
    TOOL_NAME,
)


# ----- Config -----

DEFAULT_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)
DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")

DEFAULT_INPUT_PREFIX = "captions/gemini3_flash_page_panel_v1"
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_OUTPUT_RUN = "haiku_page_panel_v1"

DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_ANNOTATE_MAX_CONCURRENCY", "1000"))
ANNOTATE_TIMEOUT_S = float(os.environ.get("ANNOTATE_TIMEOUT_S", "60"))
ANNOTATE_MAX_OUTPUT_TOKENS = int(os.environ.get("ANNOTATE_MAX_OUTPUT_TOKENS", "2048"))
MAX_PANELS_PER_PAGE = int(os.environ.get("ANNOTATE_MAX_PANELS_PER_PAGE", "20"))

VALID_SHOT_SIZES = set(SHOT_SIZES)
VALID_PANEL_TYPES = set(PANEL_TYPES)


# ----- S3 helpers -----

_S3_CLIENT = None


def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "adaptive", "max_attempts": 10},
                max_pool_connections=256,
                connect_timeout=10,
                read_timeout=120,
            ),
        )
    return _S3_CLIENT


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_token() -> str:
    return f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000, 9999)}"


def _normalize_prefix(value: object, *, default: str) -> str:
    p = str(value or default).strip().strip("/")
    if not p:
        raise ValueError("S3 prefix must not be empty")
    return p


def _join_key(*parts: object) -> str:
    return "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))


def _join_s3(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _get_bytes(bucket: str, key: str) -> bytes:
    return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()


def _put_json(bucket: str, key: str, payload: object) -> None:
    _s3().put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def _exists(bucket: str, key: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


# ----- Caption JSON parsing -----


def _parse_caption_json(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Return (panels, page_w, page_h). Each panel dict carries:
       index, bbox [x1,y1,x2,y2], width_px, height_px, area_ratio,
       character_count, speech_bubble_count, shout_bubble_count,
       narration_bubble_count, gemini_caption, panel_id.
    """
    page_size = doc.get("page_size") or {}
    page_w = int(page_size.get("width_px") or 0)
    page_h = int(page_size.get("height_px") or 0)
    raw_panels = doc.get("panels") or []
    page_area = max(1.0, float(page_w * page_h)) if (page_w and page_h) else 1.0
    out: list[dict[str, Any]] = []
    for p in raw_panels:
        bbox = list(p.get("bbox") or [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        w_px = max(1, x2 - x1)
        h_px = max(1, y2 - y1)
        area_ratio = (w_px * h_px) / page_area
        chars = p.get("characters") or []
        text_bubbles = p.get("text_bubbles") or []
        speech = sum(1 for t in text_bubbles if str(t.get("type") or "") == "Speech Bubble")
        shout = sum(1 for t in text_bubbles if str(t.get("type") or "") == "Shout Bubble")
        narr = sum(1 for t in text_bubbles if str(t.get("type") or "") == "Narration Bubble")
        gem = str(p.get("caption") or "").strip()
        out.append({
            "panel_id": str(p.get("panel_id") or ""),
            "bbox": [x1, y1, x2, y2],
            "width_px": w_px,
            "height_px": h_px,
            "area_ratio": area_ratio,
            "character_count": len(chars),
            "speech_bubble_count": speech,
            "shout_bubble_count": shout,
            "narration_bubble_count": narr,
            "gemini_caption": gem,
        })
    return out, page_w, page_h


# ----- prepare step -----


def prepare_annotate_config(event: dict[str, Any], _ctx: Any) -> dict[str, Any]:
    """Enumerate Gemini v1 caption JSONs as the source-of-truth dataset
    (manga + manwa + manhua + dc/marvel comics). Write a manifest.
    """
    source_bucket = str(event.get("source_bucket") or DATASET_BUCKET).strip()
    input_prefix = _normalize_prefix(event.get("input_prefix"), default=DEFAULT_INPUT_PREFIX)
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    output_run = str(event.get("output_run") or DEFAULT_OUTPUT_RUN).strip().strip("/")
    output_root = _join_key(output_prefix, output_run)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    max_pages = max(0, int(event.get("max_pages") or 0))
    overwrite = bool(event.get("overwrite", False))
    model = str(event.get("model") or DEFAULT_MODEL).strip()

    # Existing output keys for skip-on-overwrite=False
    existing: set[str] = set()
    if not overwrite:
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=source_bucket, Prefix=output_root.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                key = str(obj.get("Key") or "")
                if key.endswith(".json") and "/_jobs/" not in key and "/_audit/" not in key:
                    existing.add(key[len(output_root) + 1:])

    # List caption JSONs
    rows: list[dict[str, Any]] = []
    stats = {"caption_listed": 0, "skipped_regex": 0, "skipped_existing": 0}
    paginator = _s3().get_paginator("list_objects_v2")
    in_pfx = input_prefix.rstrip("/") + "/"
    for page in paginator.paginate(Bucket=source_bucket, Prefix=in_pfx):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            if "/_jobs/" in key or "/_audit/" in key:
                continue
            relative = key[len(in_pfx):]
            parts = relative.split("/")
            if len(parts) != 2:
                continue
            chapter, filename = parts
            page_id = Path(filename).stem
            stats["caption_listed"] += 1
            if include_re and not include_re.search(chapter):
                stats["skipped_regex"] += 1
                continue
            rel_out = f"{chapter}/{page_id}.json"
            if not overwrite and rel_out in existing:
                stats["skipped_existing"] += 1
                continue
            rows.append(
                {
                    "chapter": chapter,
                    "page_id": page_id,
                    "caption_key": key,
                    "output_key": f"{output_root}/{rel_out}",
                }
            )
            if max_pages > 0 and len(rows) >= max_pages:
                break
        if max_pages > 0 and len(rows) >= max_pages:
            break

    # Write manifest
    manifest_key = f"{output_root}/_jobs/{run_id}/page_manifest.jsonl"
    body = "".join((json.dumps(r) + "\n") for r in rows).encode("utf-8")
    _s3().put_object(Bucket=source_bucket, Key=manifest_key, Body=body,
                     ContentType="application/x-ndjson; charset=utf-8")

    worker_cfg = {
        "source_bucket": source_bucket,
        "input_prefix": input_prefix,
        "output_root": output_root,
        "output_run": output_run,
        "run_id": run_id,
        "model": model,
        "overwrite": overwrite,
        "created_at": _now_iso(),
    }
    cfg_key = f"{output_root}/_jobs/{run_id}/worker_config.json"
    _put_json(source_bucket, cfg_key, worker_cfg)

    return {
        "schema_version": 2,
        "source": {"bucket": source_bucket, "page_manifest_key": manifest_key},
        "worker_config": {"bucket": source_bucket, "key": cfg_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": source_bucket, "prefix": f"{output_root}/_audit/{run_id}/"},
        "output": {"bucket": source_bucket, "prefix": output_root, "output_run": output_run, "run_id": run_id},
        "stats": {**stats, "manifest_count": len(rows)},
    }


# ----- per-page worker -----


def _coerce_panels(raw_panels: list[Any], expected_n: int) -> list[dict[str, Any]]:
    """Enforce length, sort by index, coerce enums to AMB if invalid."""
    out: list[dict[str, Any] | None] = [None] * expected_n
    for p in raw_panels or []:
        if not isinstance(p, dict):
            continue
        try:
            idx = int(p.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= expected_n:
            continue
        cap = str(p.get("caption") or "").strip()
        ss = str(p.get("shot_size") or "AMB").upper()
        if ss not in VALID_SHOT_SIZES:
            ss = "AMB"
        pt = str(p.get("panel_type") or "amb").lower()
        if pt not in VALID_PANEL_TYPES:
            pt = "amb"
        out[idx] = {"index": idx, "caption": cap, "shot_size": ss, "panel_type": pt}
    for i, slot in enumerate(out):
        if slot is None:
            out[i] = {"index": i, "caption": "", "shot_size": "AMB", "panel_type": "amb"}
    return out  # type: ignore[return-value]


def annotate_page(event: dict[str, Any], _ctx: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    cfg_bucket = str(config_ref["bucket"])
    cfg_key = str(config_ref["key"])
    cfg = json.loads(_get_bytes(cfg_bucket, cfg_key).decode("utf-8"))
    source_bucket = str(cfg["source_bucket"])
    output_root = str(cfg["output_root"])
    model = str(cfg["model"])
    overwrite = bool(cfg.get("overwrite"))

    page = event.get("page") if isinstance(event.get("page"), dict) else event
    chapter = str(page["chapter"])
    page_id = str(page["page_id"])
    caption_key = str(page["caption_key"])
    output_key = str(page["output_key"])

    if not overwrite and _exists(source_bucket, output_key):
        return {"status": "skipped_existing", "chapter": chapter, "page_id": page_id, "output_key": output_key}

    # Read caption JSON (the source of truth)
    try:
        caption_doc = json.loads(_get_bytes(source_bucket, caption_key).decode("utf-8"))
    except ClientError as exc:
        return {"status": "caption_missing", "chapter": chapter, "page_id": page_id, "error": str(exc)}

    panels, page_w, page_h = _parse_caption_json(caption_doc)
    if not panels:
        return {"status": "no_panels", "chapter": chapter, "page_id": page_id}
    if len(panels) > MAX_PANELS_PER_PAGE:
        return {
            "status": "too_many_panels",
            "chapter": chapter, "page_id": page_id,
            "panel_count": len(panels),
            "limit": MAX_PANELS_PER_PAGE,
        }

    # Sort into manga reading order
    ordering = manga_reading_order([p["bbox"] for p in panels])
    ordered = []
    for read_pos, orig_idx in enumerate(ordering):
        p = dict(panels[orig_idx])
        p["index"] = read_pos
        ordered.append(p)
    n = len(ordered)

    user_text = build_user_prompt(page_w=page_w, page_h=page_h, panels=ordered)

    t0 = time.time()
    try:
        tool_input, usage = converse_tool(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            image_block=None,                       # text-only
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            tool_schema=ANNOTATE_PAGE_TOOL_SCHEMA,
            max_output_tokens=ANNOTATE_MAX_OUTPUT_TOKENS,
            timeout_seconds=ANNOTATE_TIMEOUT_S,
            client_request_id=f"{chapter}__{page_id}",
        )
    except Exception as exc:
        return {
            "status": "model_error",
            "chapter": chapter, "page_id": page_id,
            "error": str(exc)[:500],
            "duration_s": round(time.time() - t0, 2),
        }
    duration_s = round(time.time() - t0, 2)

    page_caption = str(tool_input.get("page_caption") or "").strip()
    haiku_panels = _coerce_panels(tool_input.get("panels") or [], expected_n=n)

    # Enrich + merge
    enriched: list[dict[str, Any]] = []
    panel_types: list[str] = []
    for op, hp in zip(ordered, haiku_panels):
        bbox = op["bbox"]
        bw = op["width_px"]
        bh = op["height_px"]
        bkt = closest_bucket(bw, bh)
        enriched.append({
            "index": op["index"],
            "panel_id": op["panel_id"] or f"{chapter}__{page_id}__p{op['index']:03d}",
            "bbox": bbox,
            "width_px": bw,
            "height_px": bh,
            "area_ratio": round(float(op["area_ratio"]), 6),
            "bucket": bkt,
            # MAGI ground truth (carried through)
            "character_count":        int(op["character_count"]),
            "speech_bubble_count":    int(op["speech_bubble_count"]),
            "shout_bubble_count":     int(op["shout_bubble_count"]),
            "narration_bubble_count": int(op["narration_bubble_count"]),
            # Haiku
            "caption": hp["caption"],
            "shot_size": hp["shot_size"],
            "panel_type": hp["panel_type"],
        })
        panel_types.append(hp["panel_type"])

    page_bucket = closest_bucket(max(1, page_w), max(1, page_h))
    panel_types_str = " ".join(panel_types)

    output = {
        "schema_version": 2,
        "annotation_run": str(cfg.get("output_run")),
        "model": model,
        "created_at": _now_iso(),
        "page_id": page_id,
        "chapter": chapter,
        "page_w_px": int(page_w),
        "page_h_px": int(page_h),
        "panel_count": int(n),
        "bucket": page_bucket,
        "source_caption_key": caption_key,
        "page_caption": page_caption,
        "panel_types": panel_types,
        "panel_types_str": panel_types_str,
        "panels": enriched,
        "usage": usage,
        "duration_s": duration_s,
    }
    _put_json(source_bucket, output_key, output)
    return {
        "status": "ok",
        "chapter": chapter, "page_id": page_id,
        "output_key": output_key,
        "panel_count": n,
        "usage": usage,
        "duration_s": duration_s,
    }
