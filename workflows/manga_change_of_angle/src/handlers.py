"""Lambda handlers for the manga_change_of_angle Step Functions Distributed Map.

Two functions:
  prepare_detect_change_of_angle_config — lists pages, writes manifest + worker
    config to S3, returns the JSON input for the Distributed Map.
  detect_change_of_angle_page — per-page worker invoked by every Map iteration:
    downloads the page + magi_v3 annotation, computes manga reading order,
    overlays numbered boxes, asks Kimi K2.6 to group consecutive panels that
    share a scene with the background visibly preserved, writes the output
    JSON to S3.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"

KIMI_API_KEY_SECRET_NAME = os.environ.get("KIMI_API_KEY_SECRET_NAME", "drawtoon/kimi-api-key")
KIMI_API_KEY_ENV = "MOONSHOT_API_KEY"
KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
DEFAULT_KIMI_MODEL = os.environ.get("DEFAULT_KIMI_MODEL", "kimi-k2.6")

DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/change_angle"
DEFAULT_CHANGE_ANGLE_RUN = os.environ.get("DEFAULT_CHANGE_ANGLE_RUN", "kimi_k26_v1")
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_CHANGE_ANGLE_MAX_CONCURRENCY", "100"))

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SCHEMA_NAME = "manga_change_of_angle_v1"
TRIGGER_WORD = "TRAIN_ANGLE"
PROMPT_VERSION = "manga_change_of_angle_v1_reading_order_numbered_kimi"

KIMI_MAX_IMAGE_SIDE = int(os.environ.get("KIMI_MAX_IMAGE_SIDE", "2048"))
KIMI_MAX_IMAGE_BYTES = int(os.environ.get("KIMI_MAX_IMAGE_BYTES", "8000000"))
KIMI_TIMEOUT_SECONDS = float(os.environ.get("KIMI_TIMEOUT_SECONDS", "240"))
# Per Kimi JSON-mode docs: a too-small max_tokens causes truncation even in
# JSON mode (finish_reason == "length"). With reasoning enabled the thinking
# tokens eat into this budget, so leave plenty of headroom.
KIMI_MAX_TOKENS = int(os.environ.get("KIMI_MAX_TOKENS", "16384"))

# Manga reading order: panels whose top edges are within this many pixels are
# treated as the same row.
READING_ROW_TOLERANCE = int(os.environ.get("READING_ROW_TOLERANCE", "60"))

KIMI_SYSTEM_PROMPT = (
    "You are inspecting a single manga page with numbered panel boxes "
    "(numbered in manga reading order: right-to-left, top-to-bottom).\n"
    "\n"
    "Group panels that share the SAME BACKGROUND SCENERY visible behind the "
    "characters. Background match is required and dominant — same character "
    "but different background is NOT a group.\n"
    "\n"
    "Each group needs:\n"
    "  1. Same background scenery (name the shared element in `reason`).\n"
    "  2. At least one shared character (others may differ).\n"
    "\n"
    "Panels do not need to be adjacent. When in doubt, do not group.\n"
    "\n"
    "Respond with one JSON object only, no markdown:\n"
    "{\"groups\": [{\"panel_indices\": [<int>, ...], \"reason\": \"<shared "
    "background element>\"}], \"notes\": \"<optional>\"}"
)

KIMI_USER_TAIL = ""  # the system prompt carries the full spec; nothing to add


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            config=Config(
                retries={"mode": "adaptive", "max_attempts": 10},
                connect_timeout=10,
                read_timeout=120,
                max_pool_connections=64,
            ),
        )
    return _s3


def _join_key(*parts: object) -> str:
    return "/".join(str(p).strip("/") for p in parts if p is not None and str(p).strip("/") != "")


def _join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _normalize_prefix(value: object, *, default: str) -> str:
    text = str(value or default).strip().strip("/")
    return text or default


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_token() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def _object_exists(bucket: str, key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _is_missing_s3_error(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code") or "")
    return code in {"404", "NoSuchKey", "NotFound"}


def _get_s3_bytes(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _get_s3_json_or_jsonl(bucket: str, key: str) -> dict[str, Any]:
    """Handle both pretty-printed JSON and JSONL (we use both formats)."""
    text = _get_s3_bytes(bucket, key).decode("utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.splitlines()[0])


def _put_s3_json(bucket: str, key: str, payload: object, *, pretty: bool = False) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None).encode("utf-8")
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Reading order + overlay drawing
# ---------------------------------------------------------------------------


def _reading_order(panels: list[dict[str, Any]], *, row_tolerance: int = READING_ROW_TOLERANCE) -> list[int]:
    """Return original panel indices in manga reading order (right→left, top→bottom)."""
    indexed: list[tuple[int, int, int, int, int]] = []
    for i, panel in enumerate(panels):
        bbox = panel.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue
        indexed.append((i, x0, y0, x1, y1))

    indexed.sort(key=lambda row: row[2])
    rows: list[list[tuple[int, int, int, int, int]]] = []
    for entry in indexed:
        _, _, y0, _, _ = entry
        if rows and y0 - rows[-1][0][2] <= row_tolerance:
            rows[-1].append(entry)
        else:
            rows.append([entry])

    order: list[int] = []
    for row in rows:
        row.sort(key=lambda r: r[3], reverse=True)  # rightmost first
        order.extend(r[0] for r in row)
    return order


def _draw_overlay(image_obj: Image.Image, panels: list[dict[str, Any]], reading_order: list[int]) -> Image.Image:
    """Draw the reading-order index inside each panel bbox."""
    canvas = image_obj.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    side = max(canvas.size)
    stroke = max(2, side // 400)
    pad = max(4, side // 200)
    font_size = max(20, side // 36)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for reading_index, panel_idx in enumerate(reading_order):
        panel = panels[panel_idx]
        bbox = panel.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = (int(v) for v in bbox[:4])
        draw.rectangle((x0, y0, x1, y1), outline=(0, 132, 255, 255), width=stroke)

        label = str(reading_index)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        bx0 = x0 + pad
        by0 = y0 + pad
        bx1 = bx0 + tw + 2 * pad
        by1 = by0 + th + 2 * pad
        draw.rectangle((bx0, by0, bx1, by1), fill=(0, 0, 0, 220), outline=(255, 255, 255, 255), width=2)
        draw.text((bx0 + pad, by0 + pad - text_bbox[1]), label, fill=(255, 255, 255, 255), font=font)
    return canvas


def _encode_image_for_kimi(image_obj: Image.Image) -> tuple[str, str]:
    """Return (data_url, mime). Always JPEG so we can stay under the byte cap."""
    img = image_obj
    if max(img.size) > KIMI_MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((KIMI_MAX_IMAGE_SIDE, KIMI_MAX_IMAGE_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    data = buf.getvalue()
    if len(data) > KIMI_MAX_IMAGE_BYTES:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80, optimize=True)
        data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", "image/jpeg"


# ---------------------------------------------------------------------------
# Kimi client + call
# ---------------------------------------------------------------------------


_kimi_client = None
_kimi_api_key: str | None = None


def _resolve_kimi_api_key() -> str:
    global _kimi_api_key
    if _kimi_api_key:
        return _kimi_api_key
    for env_name in (KIMI_API_KEY_ENV, "KIMI_API_KEY"):
        env_value = (os.environ.get(env_name) or "").strip()
        if env_value:
            _kimi_api_key = env_value
            return env_value
    response = boto3.client("secretsmanager", region_name=AWS_REGION).get_secret_value(
        SecretId=KIMI_API_KEY_SECRET_NAME
    )
    value = str(response.get("SecretString") or "").strip()
    if not value:
        raise RuntimeError(f"empty Kimi API secret {KIMI_API_KEY_SECRET_NAME!r}")
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            for key in (KIMI_API_KEY_ENV, "KIMI_API_KEY"):
                if parsed.get(key):
                    _kimi_api_key = str(parsed[key]).strip()
                    return _kimi_api_key
    except json.JSONDecodeError:
        pass
    _kimi_api_key = value
    return value


def _get_kimi_client():
    global _kimi_client
    if _kimi_client is None:
        from openai import OpenAI

        _kimi_client = OpenAI(
            api_key=_resolve_kimi_api_key(),
            base_url=KIMI_BASE_URL,
            timeout=KIMI_TIMEOUT_SECONDS,
        )
    return _kimi_client


def _call_kimi_for_groups(
    *,
    image_obj: Image.Image,
    n_panels: int,
    model: str,
    thinking_enabled: bool,
) -> dict[str, Any]:
    client = _get_kimi_client()
    data_url, _ = _encode_image_for_kimi(image_obj)
    user_text = f"This page has {n_panels} panels numbered 0..{n_panels - 1}. List the groups."
    messages = [
        {"role": "system", "content": KIMI_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=KIMI_MAX_TOKENS,
            extra_body={"thinking": {"type": "enabled" if thinking_enabled else "disabled"}},
        )
    except Exception as exc:
        if exc.__class__.__name__ == "APITimeoutError":
            return {
                "groups": [],
                "notes": "kimi_api_timeout: request timed out",
                "usage": {},
                "finish_reason": "api_timeout",
                "status": "kimi_api_timeout",
            }
        raise
    choice = response.choices[0]
    finish = getattr(choice, "finish_reason", None)
    text = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    usage_payload: dict[str, Any] = {}
    if usage is not None:
        usage_payload = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", None)
            if cached is not None:
                usage_payload["cached_tokens"] = int(cached or 0)
    if finish == "length":
        return {
            "groups": [],
            "notes": (
                "kimi_truncated: finish_reason=length; "
                f"max_tokens={KIMI_MAX_TOKENS}; first_200={text[:200]!r}"
            ),
            "usage": usage_payload,
            "finish_reason": finish,
            "status": "kimi_truncated",
        }
    if not text:
        return {
            "groups": [],
            "notes": f"kimi_empty_response: finish_reason={finish!r}",
            "usage": usage_payload,
            "finish_reason": finish,
            "status": "kimi_empty_response",
        }
    # Kimi (especially with reasoning on) occasionally wraps JSON in
    # ```json ... ``` markdown fences even when response_format is set and
    # the prompt forbids fences. Strip them defensively before parsing.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "groups": [],
            "notes": f"kimi_invalid_json: {text[:200]!r}",
            "usage": usage_payload,
            "finish_reason": finish,
            "status": "kimi_invalid_json",
        }
    groups = parsed.get("groups") or []
    if not isinstance(groups, list):
        return {
            "groups": [],
            "notes": f"kimi_groups_not_list: {type(groups).__name__}",
            "usage": usage_payload,
            "finish_reason": finish,
            "status": "kimi_invalid_groups",
        }
    return {
        "groups": groups,
        "notes": parsed.get("notes", ""),
        "usage": usage_payload,
        "finish_reason": finish,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_group(raw_indices: Any, *, n_panels: int) -> list[int] | None:
    """2+ sorted unique in-range panel indices, else None. Indices do not need
    to be consecutive — a group can be any subset of the page's panels."""
    if not isinstance(raw_indices, list) or len(raw_indices) < 2:
        return None
    try:
        sorted_idx = sorted({int(x) for x in raw_indices})
    except (TypeError, ValueError):
        return None
    if len(sorted_idx) < 2:
        return None
    if any(x < 0 or x >= n_panels for x in sorted_idx):
        return None
    return sorted_idx


# ---------------------------------------------------------------------------
# Page listing (for the prepare Lambda)
# ---------------------------------------------------------------------------


def _list_set_under_prefix(*, bucket: str, prefix: str, suffix: str) -> set[str]:
    root = prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=root):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if not key.endswith(suffix):
                continue
            seen.add(key[len(root):])
    return seen


def _list_page_rows(
    *,
    bucket: str,
    source_prefix: str,
    annotation_prefix: str,
    output_root: str,
    include_chapter_regex: str,
    overwrite: bool,
    require_annotations: bool,
    max_pages: int,
    sampling_strategy: str,
    shuffle_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotation_relatives = _list_set_under_prefix(
        bucket=bucket, prefix=annotation_prefix, suffix=".jsonl"
    )
    existing_output_relatives = (
        set() if overwrite else _list_set_under_prefix(bucket=bucket, prefix=output_root, suffix=".json")
    )
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    source_root = source_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    rows: list[dict[str, Any]] = []
    stats = {
        "source_image_count": 0,
        "skipped_non_image_count": 0,
        "skipped_chapter_regex_count": 0,
        "skipped_missing_annotation_count": 0,
        "skipped_existing_count": 0,
        "eligible_count": 0,
        "eligible_chapter_count": 0,
    }
    strategy = sampling_strategy.strip() or "sequential"
    if strategy not in {"sequential", "round_robin_chapters"}:
        raise ValueError(f"unsupported sampling_strategy={sampling_strategy!r}")
    collect_all = strategy == "round_robin_chapters"

    for page in paginator.paginate(Bucket=bucket, Prefix=source_root):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            suffix = Path(key).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                stats["skipped_non_image_count"] += 1
                continue
            stats["source_image_count"] += 1
            relative = key[len(source_root):]
            parts = relative.split("/")
            if len(parts) != 2:
                continue
            chapter, filename = parts
            if include_re and not include_re.search(chapter):
                stats["skipped_chapter_regex_count"] += 1
                continue
            page_id = Path(filename).stem
            annotation_relative = f"{chapter}/{page_id}.jsonl"
            if require_annotations and annotation_relative not in annotation_relatives:
                stats["skipped_missing_annotation_count"] += 1
                continue
            output_relative = f"{chapter}/{page_id}.json"
            output_key = _join_key(output_root, output_relative)
            if not overwrite and output_relative in existing_output_relatives:
                stats["skipped_existing_count"] += 1
                continue
            rows.append(
                {
                    "chapter": chapter,
                    "page_id": page_id,
                    "relative_path": relative,
                    "page_key": key,
                    "annotation_key": _join_key(annotation_prefix, annotation_relative),
                    "output_key": output_key,
                }
            )
            if max_pages > 0 and len(rows) >= max_pages and not collect_all:
                stats["eligible_count"] = len(rows)
                stats["eligible_chapter_count"] = len({row["chapter"] for row in rows})
                return rows, stats
    stats["eligible_count"] = len(rows)
    stats["eligible_chapter_count"] = len({row["chapter"] for row in rows})
    if collect_all and max_pages > 0 and len(rows) > max_pages:
        rows = _round_robin_chapter_sample(rows, max_pages=max_pages, shuffle_seed=shuffle_seed)
    return rows, stats


def _list_page_rows_from_manifest(
    *,
    bucket: str,
    input_manifest_key: str,
    output_root: str,
    overwrite: bool,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    existing_output_relatives = (
        set() if overwrite else _list_set_under_prefix(bucket=bucket, prefix=output_root, suffix=".json")
    )
    text = _get_s3_bytes(bucket, input_manifest_key).decode("utf-8")
    rows: list[dict[str, Any]] = []
    stats = {
        "source_image_count": 0,
        "skipped_non_image_count": 0,
        "skipped_chapter_regex_count": 0,
        "skipped_missing_annotation_count": 0,
        "skipped_existing_count": 0,
        "eligible_count": 0,
        "eligible_chapter_count": 0,
        "input_manifest_count": 0,
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stats["input_manifest_count"] += 1
        row = json.loads(line)
        chapter = str(row["chapter"])
        page_id = str(row["page_id"])
        output_relative = f"{chapter}/{page_id}.json"
        if not overwrite and output_relative in existing_output_relatives:
            stats["skipped_existing_count"] += 1
            continue
        row["output_key"] = _join_key(output_root, output_relative)
        rows.append(row)
        if max_pages > 0 and len(rows) >= max_pages:
            break
    stats["source_image_count"] = stats["input_manifest_count"]
    stats["eligible_count"] = len(rows)
    stats["eligible_chapter_count"] = len({str(row["chapter"]) for row in rows})
    return rows, stats


def _stable_seed(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _round_robin_chapter_sample(
    rows: list[dict[str, Any]],
    *,
    max_pages: int,
    shuffle_seed: str,
) -> list[dict[str, Any]]:
    """Select up to max_pages while spreading coverage across chapters."""
    by_chapter: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_chapter.setdefault(str(row["chapter"]), []).append(row)

    seed = shuffle_seed.strip() or "drawtoon-change-angle"
    chapters = sorted(by_chapter)
    random.Random(_stable_seed(f"{seed}:chapters")).shuffle(chapters)
    for chapter, chapter_rows in by_chapter.items():
        chapter_rows.sort(key=lambda r: str(r.get("page_id") or ""))
        random.Random(_stable_seed(f"{seed}:pages:{chapter}")).shuffle(chapter_rows)

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < max_pages:
        added = False
        for chapter in chapters:
            chapter_rows = by_chapter[chapter]
            if round_index >= len(chapter_rows):
                continue
            selected.append(chapter_rows[round_index])
            added = True
            if len(selected) >= max_pages:
                break
        if not added:
            break
        round_index += 1
    return selected


# ---------------------------------------------------------------------------
# Lambda 1: prepare config
# ---------------------------------------------------------------------------


def prepare_detect_change_of_angle_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or DATASET_BUCKET).strip()
    source_prefix = _normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_PREFIX)
    annotation_prefix = _normalize_prefix(event.get("annotation_prefix"), default=DEFAULT_ANNOTATION_PREFIX)
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    input_manifest_key = str(event.get("input_manifest_key") or "").strip().strip("/")
    change_angle_run = str(event.get("change_angle_run") or DEFAULT_CHANGE_ANGLE_RUN).strip().strip("/")
    if not change_angle_run:
        raise ValueError("change_angle_run must not be empty")
    output_root = _join_key(output_prefix, change_angle_run)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))
    sampling_strategy = str(event.get("sampling_strategy") or "sequential").strip()
    shuffle_seed = str(event.get("shuffle_seed") or "").strip()
    overwrite = bool(event.get("overwrite", False))
    require_annotations = bool(event.get("require_annotations", True))
    thinking_enabled = bool(event.get("thinking_enabled", True))
    model = str(event.get("model") or DEFAULT_KIMI_MODEL).strip()

    if input_manifest_key:
        rows, page_stats = _list_page_rows_from_manifest(
            bucket=bucket,
            input_manifest_key=input_manifest_key,
            output_root=output_root,
            overwrite=overwrite,
            max_pages=max_pages,
        )
    else:
        rows, page_stats = _list_page_rows(
            bucket=bucket,
            source_prefix=source_prefix,
            annotation_prefix=annotation_prefix,
            output_root=output_root,
            include_chapter_regex=include_chapter_regex,
            overwrite=overwrite,
            require_annotations=require_annotations,
            max_pages=max_pages,
            sampling_strategy=sampling_strategy,
            shuffle_seed=shuffle_seed,
        )
    manifest_key = _join_key(output_root, "_jobs", run_id, "page_manifest.jsonl")
    manifest_body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _s3_client().put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )

    worker_config = {
        "bucket": bucket,
        "source_prefix": source_prefix,
        "annotation_prefix": annotation_prefix,
        "output_prefix": output_prefix,
        "change_angle_run": change_angle_run,
        "output_root": output_root,
        "run_id": run_id,
        "model": model,
        "thinking_enabled": thinking_enabled,
        "overwrite": overwrite,
        "created_at": _now_iso(),
        "git_sha": str(event.get("git_sha") or ""),
        "sampling_strategy": sampling_strategy,
        "shuffle_seed": shuffle_seed,
        "input_manifest_key": input_manifest_key,
    }
    config_key = _join_key(output_root, "_jobs", run_id, "worker_config.json")
    _put_s3_json(bucket, config_key, worker_config, pretty=True)

    return {
        "schema_version": 1,
        "source": {"bucket": bucket, "page_manifest_key": manifest_key},
        "worker_config": {"bucket": bucket, "key": config_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": bucket, "prefix": _join_key(output_root, "_audit", run_id) + "/"},
        "output": {
            "bucket": bucket,
            "prefix": output_root,
            "change_angle_run": change_angle_run,
            "run_id": run_id,
        },
        "stats": {**page_stats, "manifest_count": len(rows)},
    }


# ---------------------------------------------------------------------------
# Lambda 2: per-page worker
# ---------------------------------------------------------------------------


def detect_change_of_angle_page(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json_or_jsonl(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    page = event.get("page") if isinstance(event.get("page"), dict) else event
    chapter = str(page["chapter"])
    page_id = str(page["page_id"])
    page_key = str(page["page_key"])
    annotation_key = str(page["annotation_key"])
    output_key = str(page["output_key"])
    sample_id = f"{chapter}__{page_id}"

    if not bool(config.get("overwrite")) and _object_exists(bucket, output_key):
        return {"status": "skipped_existing", "output_key": output_key, "sample_id": sample_id}

    started = time.perf_counter()

    try:
        annotation = _get_s3_json_or_jsonl(bucket, annotation_key)
    except ClientError as exc:
        if not _is_missing_s3_error(exc):
            raise
        payload = {
            "schema_name": SCHEMA_NAME,
            "trigger": TRIGGER_WORD,
            "change_angle_run": str(config.get("change_angle_run") or DEFAULT_CHANGE_ANGLE_RUN),
            "run_id": str(config.get("run_id") or ""),
            "sample_id": sample_id,
            "chapter": chapter,
            "page_id": page_id,
            "page_key": page_key,
            "annotation_key": annotation_key,
            "image_size": {"width": 0, "height": 0},
            "panels_in_reading_order": [],
            "angle_groups": [],
            "summary": {"n_panels": 0, "n_panels_in_groups": 0, "n_groups": 0},
            "verification": {
                "status": "skipped_missing_annotation",
                "provider": "kimi",
                "model": str(config.get("model") or DEFAULT_KIMI_MODEL).strip(),
                "thinking_enabled": bool(config.get("thinking_enabled", True)),
                "prompt_version": PROMPT_VERSION,
            },
            "sources": {
                "page": _join_s3_uri(bucket, page_key),
                "annotation": _join_s3_uri(bucket, annotation_key),
            },
            "created_at": _now_iso(),
        }
        _put_s3_json(bucket, output_key, payload, pretty=False)
        return {
            "status": "skipped_missing_annotation",
            "sample_id": sample_id,
            "output_key": output_key,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
    detections = annotation.get("detections") or {}
    panels = detections.get("panels") or []
    image_size = annotation.get("image_size") or {}

    reading_order = _reading_order(panels)
    n_panels = len(reading_order)
    panels_payload = [
        {"bbox": panels[pi].get("bbox"), "panel_id": panels[pi].get("panel_id")}
        for pi in reading_order
    ]

    model = str(config.get("model") or DEFAULT_KIMI_MODEL).strip()
    thinking_enabled = bool(config.get("thinking_enabled", True))

    verification: dict[str, Any] = {
        "status": "ok",
        "provider": "kimi",
        "model": model,
        "thinking_enabled": thinking_enabled,
        "prompt_version": PROMPT_VERSION,
    }
    angle_groups: list[dict[str, Any]] = []

    if n_panels < 2:
        verification["status"] = "skipped_too_few_panels"
    else:
        try:
            page_bytes = _get_s3_bytes(bucket, page_key)
        except ClientError as exc:
            if not _is_missing_s3_error(exc):
                raise
            verification["status"] = "skipped_missing_source"
            page_bytes = b""
        if page_bytes:
            with Image.open(io.BytesIO(page_bytes)) as raw:
                raw.load()
                rotated = ImageOps.exif_transpose(raw)
                page_image = rotated.convert("RGB")
                if rotated is not raw:
                    rotated.close()
            overlay = _draw_overlay(page_image, panels, reading_order)
            kimi_result = _call_kimi_for_groups(
                image_obj=overlay,
                n_panels=n_panels,
                model=model,
                thinking_enabled=thinking_enabled,
            )
            if kimi_result.get("status") and kimi_result.get("status") != "ok":
                verification["status"] = str(kimi_result["status"])
            for raw_group in kimi_result["groups"]:
                if not isinstance(raw_group, dict):
                    continue
                indices = _validate_group(raw_group.get("panel_indices"), n_panels=n_panels)
                if not indices:
                    continue
                angle_groups.append(
                    {
                        "panel_indices": indices,
                        "reason": str(raw_group.get("reason") or "").strip(),
                    }
                )
            if kimi_result.get("notes"):
                verification["notes"] = str(kimi_result["notes"])[:1000]
            if kimi_result.get("usage"):
                verification["usage"] = kimi_result["usage"]
            if kimi_result.get("finish_reason"):
                verification["finish_reason"] = kimi_result["finish_reason"]

    n_in_groups = sum(len(g["panel_indices"]) for g in angle_groups)
    payload = {
        "schema_name": SCHEMA_NAME,
        "trigger": TRIGGER_WORD,
        "change_angle_run": str(config.get("change_angle_run") or DEFAULT_CHANGE_ANGLE_RUN),
        "run_id": str(config.get("run_id") or ""),
        "sample_id": sample_id,
        "chapter": chapter,
        "page_id": page_id,
        "page_key": page_key,
        "annotation_key": annotation_key,
        "image_size": {
            "width": int(image_size.get("width") or 0),
            "height": int(image_size.get("height") or 0),
        },
        "panels_in_reading_order": panels_payload,
        "angle_groups": angle_groups,
        "summary": {
            "n_panels": n_panels,
            "n_panels_in_groups": n_in_groups,
            "n_groups": len(angle_groups),
        },
        "verification": verification,
        "sources": {
            "page": _join_s3_uri(bucket, page_key),
            "annotation": _join_s3_uri(bucket, annotation_key),
        },
        "created_at": _now_iso(),
    }
    _put_s3_json(bucket, output_key, payload, pretty=False)
    return {
        "status": "ok",
        "sample_id": sample_id,
        "output_key": output_key,
        "n_panels": n_panels,
        "n_groups": len(angle_groups),
        "n_panels_in_groups": n_in_groups,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
