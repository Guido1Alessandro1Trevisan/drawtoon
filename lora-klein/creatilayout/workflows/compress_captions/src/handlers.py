from __future__ import annotations

import datetime as dt
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

from .assembly import (
    assemble_short_caption,
    attribute_bubbles_to_characters,
    strip_caption_prefix,
    word_count,
)
from .prompts.compress_prompt import (
    CAMERA_ANGLE_VALUES,
    PANEL_EXTRACT_SCHEMA,
    SHOT_SIZE_VALUES,
    SYSTEM_INSTRUCTION,
    USER_INSTRUCTION,
)


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET_NAME", "drawtoon")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET_NAME", "drawtoon-layousyn")
DEFAULT_GEMINI_MODEL = os.environ.get("DEFAULT_GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"

DEFAULT_SOURCE_CAPTION_PREFIX = "captions"
DEFAULT_SOURCE_CAPTION_RUN = "gemini3_flash_page_panel_v1"
DEFAULT_OUTPUT_PREFIX = "captions_short"
DEFAULT_OUTPUT_RUN = "vision_v1"

DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_COMPRESS_MAX_CONCURRENCY", "200"))
WORD_HARD_CAP = int(os.environ.get("COMPRESS_WORD_HARD_CAP", "20"))
MAX_PANEL_SIDE = int(os.environ.get("COMPRESS_MAX_PANEL_SIDE", "1024"))
THINKING_LEVEL = os.environ.get("COMPRESS_THINKING_LEVEL", "high")

_S3_CLIENT = None
_GENAI_CLIENT: Any = None
_GENAI_API_KEY: str | None = None
_GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "180000"))
_GEMINI_RETRY_ATTEMPTS = int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "8"))
_GEMINI_RETRY_MAX_DELAY_S = float(os.environ.get("GEMINI_RETRY_MAX_DELAY_S", "120.0"))


# =========================================================================
# S3 + utility helpers
# =========================================================================


def _s3_client():
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


def _put_s3_json(bucket: str, key: str, payload: object, *, pretty: bool = False) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=_json_dumps(payload, pretty=pretty).encode("utf-8"),
        ContentType="application/json",
    )


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


# =========================================================================
# Gemini client (vision + thinking)
# =========================================================================


def _resolve_gemini_api_key() -> str:
    global _GENAI_API_KEY
    if _GENAI_API_KEY:
        return _GENAI_API_KEY
    env_value = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if env_value:
        _GENAI_API_KEY = env_value
        return _GENAI_API_KEY
    if GEMINI_API_KEY_SECRET_NAME:
        try:
            client = boto3.client("secretsmanager", region_name=DEFAULT_REGION)
            resp = client.get_secret_value(SecretId=GEMINI_API_KEY_SECRET_NAME)
            secret_string = resp.get("SecretString")
            if secret_string:
                try:
                    parsed = json.loads(secret_string)
                    if isinstance(parsed, dict) and parsed.get(GEMINI_API_KEY_ENV):
                        _GENAI_API_KEY = str(parsed[GEMINI_API_KEY_ENV])
                        return _GENAI_API_KEY
                except json.JSONDecodeError:
                    pass
                _GENAI_API_KEY = secret_string.strip()
                return _GENAI_API_KEY
        except ClientError as exc:
            raise RuntimeError(
                f"Could not load Gemini API key from secret {GEMINI_API_KEY_SECRET_NAME!r}: {exc}"
            ) from exc
    raise RuntimeError(
        f"Gemini API key not found. Set {GEMINI_API_KEY_ENV} env var or "
        f"secret {GEMINI_API_KEY_SECRET_NAME!r}."
    )


def _genai_client():
    """Module-level singleton — same pattern as workflows/manga_caption.

    Used single-threaded inside a Lambda invocation (no in-process fan-out).
    The cached client survives across warm-container invocations.
    """
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package not installed") from exc
    _GENAI_CLIENT = genai.Client(
        api_key=_resolve_gemini_api_key(),
        http_options=types.HttpOptions(
            timeout=_GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=_GEMINI_RETRY_ATTEMPTS,
                max_delay=_GEMINI_RETRY_MAX_DELAY_S,
            ),
        ),
    )
    return _GENAI_CLIENT


# =========================================================================
# Panel crop + Gemini extraction
# =========================================================================


def _crop_panel_png(page_image: Image.Image, panel_bbox: list[int]) -> bytes:
    x0, y0, x1, y1 = [int(round(v)) for v in panel_bbox]
    crop = page_image.crop((x0, y0, x1, y1))
    if max(crop.size) > MAX_PANEL_SIDE:
        crop = crop.copy()
        crop.thumbnail((MAX_PANEL_SIDE, MAX_PANEL_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _extract_one_panel(
    *, model: str, panel_png: bytes
) -> tuple[dict[str, str], dict[str, int]]:
    from google.genai import types  # type: ignore

    # COMPRESS_THINKING_LEVEL knob:
    #   "off" / "none" / "0" -> thinking disabled (thinking_budget=0)
    #   "low" / "medium" / "high" -> Gemini 3 reasoning tiers
    level = (THINKING_LEVEL or "high").strip().lower()
    if level in {"off", "none", "0", "disabled", "false"}:
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    else:
        thinking_config = types.ThinkingConfig(thinking_level=level)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PANEL_EXTRACT_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION,
        thinking_config=thinking_config,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    resp = _genai_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=panel_png, mime_type="image/png"),
            USER_INSTRUCTION,
        ],
        config=config,
    )
    raw = json.loads(resp.text or "{}")
    shot = str(raw.get("shot_size") or "ambiguous")
    angle = str(raw.get("camera_angle") or "ambiguous")
    action = str(raw.get("action_phrase") or "").strip()
    if shot not in SHOT_SIZE_VALUES:
        shot = "ambiguous"
    if angle not in CAMERA_ANGLE_VALUES:
        angle = "ambiguous"
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
        "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
        "thoughts_tokens": int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
        "total_tokens": int(getattr(um, "total_token_count", 0) or 0) if um else 0,
    }
    return {"shot_size": shot, "camera_angle": angle, "action_phrase": action}, usage


# =========================================================================
# Manifest listing
# =========================================================================


def _list_relative_jsons(*, bucket: str, root_prefix: str) -> list[str]:
    prefix = root_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if "/_jobs/" in key or "/_audit/" in key:
                continue
            if not key.endswith(".json"):
                continue
            relatives.append(key[len(prefix):])
    return relatives


def _existing_output_relatives(*, bucket: str, root_prefix: str) -> set[str]:
    return set(_list_relative_jsons(bucket=bucket, root_prefix=root_prefix))


def _list_page_rows(
    *,
    source_bucket: str,
    output_bucket: str,
    source_root: str,
    output_root: str,
    include_chapter_regex: str,
    overwrite: bool,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    existing = (
        set()
        if overwrite
        else _existing_output_relatives(bucket=output_bucket, root_prefix=output_root)
    )
    rows: list[dict[str, Any]] = []
    stats = {
        "source_caption_count": 0,
        "skipped_chapter_regex_count": 0,
        "skipped_existing_count": 0,
    }
    for relative in _list_relative_jsons(bucket=source_bucket, root_prefix=source_root):
        stats["source_caption_count"] += 1
        parts = relative.split("/")
        if len(parts) != 2:
            continue
        chapter, filename = parts
        if include_re and not include_re.search(chapter):
            stats["skipped_chapter_regex_count"] += 1
            continue
        if not overwrite and relative in existing:
            stats["skipped_existing_count"] += 1
            continue
        page_id = Path(filename).stem
        source_key = _join_key(source_root, relative)
        output_key = _join_key(output_root, relative)
        rows.append(
            {
                "chapter": chapter,
                "page_id": page_id,
                "relative_path": relative,
                "source_bucket": source_bucket,
                "output_bucket": output_bucket,
                "source_caption_key": source_key,
                "output_key": output_key,
                "source_caption_s3_uri": _join_s3_uri(source_bucket, source_key),
            }
        )
        if max_pages > 0 and len(rows) >= max_pages:
            break
    return rows, stats


# =========================================================================
# Lambda entry points
# =========================================================================


def prepare_compress_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    source_bucket = str(event.get("source_bucket") or SOURCE_BUCKET).strip()
    output_bucket = str(event.get("output_bucket") or OUTPUT_BUCKET).strip()
    source_prefix = _normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_CAPTION_PREFIX)
    source_caption_run = str(event.get("source_caption_run") or DEFAULT_SOURCE_CAPTION_RUN).strip().strip("/")
    if not source_caption_run:
        raise ValueError("source_caption_run must not be empty")
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    output_run = str(event.get("output_run") or DEFAULT_OUTPUT_RUN).strip().strip("/")
    if not output_run:
        raise ValueError("output_run must not be empty")
    source_root = _join_key(source_prefix, source_caption_run)
    output_root = _join_key(output_prefix, output_run)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))
    overwrite = bool(event.get("overwrite", False))
    model = str(event.get("model") or DEFAULT_GEMINI_MODEL).strip()

    rows, page_stats = _list_page_rows(
        source_bucket=source_bucket,
        output_bucket=output_bucket,
        source_root=source_root,
        output_root=output_root,
        include_chapter_regex=include_chapter_regex,
        overwrite=overwrite,
        max_pages=max_pages,
    )
    manifest_key = _join_key(output_root, "_jobs", run_id, "page_manifest.jsonl")
    manifest_body = "".join(_json_dumps(row) for row in rows)
    _s3_client().put_object(
        Bucket=output_bucket,
        Key=manifest_key,
        Body=manifest_body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )

    worker_config = {
        "source_bucket": source_bucket,
        "output_bucket": output_bucket,
        "source_prefix": source_prefix,
        "source_caption_run": source_caption_run,
        "source_root": source_root,
        "output_prefix": output_prefix,
        "output_run": output_run,
        "output_root": output_root,
        "run_id": run_id,
        "model": model,
        "overwrite": overwrite,
        "created_at": _now_utc_iso(),
        "git_sha": str(event.get("git_sha") or ""),
    }
    config_key = _join_key(output_root, "_jobs", run_id, "worker_config.json")
    _put_s3_json(output_bucket, config_key, worker_config, pretty=True)

    return {
        "schema_version": 2,
        "source": {"bucket": output_bucket, "page_manifest_key": manifest_key},
        "worker_config": {"bucket": output_bucket, "key": config_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": output_bucket, "prefix": _join_key(output_root, "_audit", run_id) + "/"},
        "output": {"bucket": output_bucket, "prefix": output_root, "output_run": output_run, "run_id": run_id},
        "stats": {**page_stats, "manifest_count": len(rows)},
    }


def compress_caption_page(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    source_bucket = str(config["source_bucket"])
    output_bucket = str(config["output_bucket"])
    page = event.get("page") if isinstance(event.get("page"), dict) else event
    output_key = str(page["output_key"])
    if not bool(config.get("overwrite")) and _object_exists(output_bucket, output_key):
        return {"status": "skipped_existing", "output_key": output_key, "page_id": str(page.get("page_id") or "")}

    source = _get_s3_json(source_bucket, str(page["source_caption_key"]))
    caption_prefix = str(source.get("caption_prefix") or "")
    panels_in = source.get("panels") or []
    page_key = str(source.get("sources", {}).get("page_key") or "")

    if not panels_in:
        return {
            "status": "skipped_empty",
            "chapter": str(page.get("chapter") or ""),
            "page_id": str(page.get("page_id") or ""),
            "output_key": output_key,
        }
    if not page_key:
        return {
            "status": "skipped_missing_page_key",
            "chapter": str(page.get("chapter") or ""),
            "page_id": str(page.get("page_id") or ""),
            "output_key": output_key,
        }

    page_bytes = _get_s3_bytes(source_bucket, page_key)
    with Image.open(io.BytesIO(page_bytes)) as image:
        page_image = image.convert("RGB")

    crops: dict[int, bytes] = {}
    panel_meta: list[dict[str, Any]] = []
    for panel in panels_in:
        if not isinstance(panel, dict):
            continue
        idx = int(panel.get("panel_index") or 0)
        panel_bbox = panel.get("bbox")
        if not panel_bbox:
            continue
        try:
            crops[idx] = _crop_panel_png(page_image, panel_bbox)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "crop_failed",
                "chapter": str(page.get("chapter") or ""),
                "page_id": str(page.get("page_id") or ""),
                "panel_index": idx,
                "error": str(exc),
            }
        panel_meta.append(
            {
                "panel_index": idx,
                "panel_id": str(panel.get("panel_id") or f"panel_{idx:03d}"),
                "bbox": panel.get("bbox"),
                "bbox_norm": panel.get("bbox_norm"),
                "source_caption": strip_caption_prefix(str(panel.get("caption") or ""), caption_prefix),
                "characters": panel.get("characters") or [],
                "text_bubbles": panel.get("text_bubbles") or [],
            }
        )

    if not panel_meta:
        return {
            "status": "skipped_no_panels_with_bbox",
            "chapter": str(page.get("chapter") or ""),
            "page_id": str(page.get("page_id") or ""),
            "output_key": output_key,
        }

    # Sequential per-panel Gemini calls — same single-threaded pattern as
    # workflows/manga_caption. Threaded fan-out inside the Lambda triggers
    # google-genai's shared httpx pool to close mid-flight under load
    # ("RuntimeError: Cannot send a request, as the client has been closed").
    extracts: dict[int, dict[str, str]] = {}
    usages: dict[int, dict[str, int]] = {}
    model = str(config["model"])

    for meta in panel_meta:
        idx = int(meta["panel_index"])
        try:
            extract, usage = _extract_one_panel(model=model, panel_png=crops[idx])
        except Exception as exc:  # noqa: BLE001
            # Record the panel as ambiguous + empty so the page output still
            # writes successfully. A separate audit pass can re-run only the
            # ambiguous panels later if desired.
            print(f"panel {idx} extract failed: {exc!r}; using ambiguous fallback")
            extract = {"shot_size": "ambiguous", "camera_angle": "ambiguous", "action_phrase": ""}
            usage = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
        extracts[idx] = extract
        usages[idx] = usage

    out_panels: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    for meta in panel_meta:
        idx = int(meta["panel_index"])
        extract = extracts.get(idx) or {"shot_size": "ambiguous", "camera_angle": "ambiguous", "action_phrase": ""}
        usage = usages.get(idx) or {}
        for key in usage_total:
            usage_total[key] += int(usage.get(key) or 0)

        attributed = attribute_bubbles_to_characters(
            characters=meta["characters"],
            text_bubbles=meta["text_bubbles"],
        )
        short_caption = assemble_short_caption(
            shot_size=extract["shot_size"],
            camera_angle=extract["camera_angle"],
            action_phrase=extract["action_phrase"],
            character_count=len(meta["characters"]),
            attributed_bubbles=attributed,
        )
        out_panels.append(
            {
                "panel_index": idx,
                "panel_id": meta["panel_id"],
                "bbox": meta["bbox"],
                "bbox_norm": meta["bbox_norm"],
                "source_caption": meta["source_caption"],
                "source_word_count": word_count(meta["source_caption"]),
                "character_count": len(meta["characters"]),
                "text_bubble_count": len(meta["text_bubbles"]),
                "shot_size": extract["shot_size"],
                "camera_angle": extract["camera_angle"],
                "action_phrase": extract["action_phrase"],
                "attributed_bubbles": attributed,
                "short_caption": short_caption,
                "short_word_count": word_count(short_caption),
                "exceeded_word_cap": word_count(short_caption) > WORD_HARD_CAP,
            }
        )

    output = {
        "schema_version": 3,
        "caption_type": "gemini_panel_compress_vision_v1",
        "status": "ok",
        "source_caption_run": str(config["source_caption_run"]),
        "source_caption_uri": str(
            page.get("source_caption_s3_uri")
            or _join_s3_uri(source_bucket, str(page["source_caption_key"]))
        ),
        "source_page_key": page_key,
        "source_bucket": source_bucket,
        "output_bucket": output_bucket,
        "output_run": str(config["output_run"]),
        "run_id": str(config["run_id"]),
        "created_at": _now_utc_iso(),
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "page_size": source.get("page_size") or {},
        "manga_credit": source.get("manga_credit") or {},
        "model": {"id": model, "provider": "gemini", "mode": "vision", "thinking_level": THINKING_LEVEL},
        "panels": out_panels,
        "usage": usage_total,
    }
    _put_s3_json(output_bucket, output_key, output, pretty=False)
    return {
        "status": "ok",
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "output_key": output_key,
        "panel_count": len(out_panels),
        "usage": usage_total,
    }
