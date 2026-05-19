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

from .prompts.page_caption_prompt import (
    ALLOWED_PAGE_TAGS,
    PAGE_CAPTION_SCHEMA_GEMINI,
    SYSTEM_INSTRUCTION,
    build_user_text,
)


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_GEMINI_MODEL = os.environ.get("DEFAULT_GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
DEFAULT_SOURCE_PREFIX = "datasets/pages/text_removed"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "page_captions"
DEFAULT_OUTPUT_RUN = "page_v1"
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_PAGE_CAPTION_MAX_CONCURRENCY", "300"))
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
WORD_HARD_CAP = int(os.environ.get("PAGE_CAPTION_WORD_HARD_CAP", "20"))

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


def _get_s3_json_or_jsonl(bucket: str, key: str) -> dict[str, Any]:
    text = _get_s3_bytes(bucket, key).decode("utf-8").strip()
    if not text:
        raise ValueError(f"Empty JSON at {_join_s3_uri(bucket, key)}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for line in text.splitlines():
            line = line.strip()
            if line:
                payload = json.loads(line)
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object/JSONL row at {_join_s3_uri(bucket, key)}")
    return payload


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _panel_count_from_annotation(annotation: dict[str, Any]) -> int:
    detections = annotation.get("detections") if isinstance(annotation.get("detections"), dict) else {}
    panels = detections.get("panels") if isinstance(detections.get("panels"), list) else []
    return sum(1 for panel in panels if isinstance(panel, dict))


# =========================================================================
# Gemini client (vision)
# =========================================================================

_GENAI_CLIENT: Any = None
_GENAI_API_KEY: str | None = None
_GEMINI_MAX_IMAGE_SIDE = int(os.environ.get("GEMINI_MAX_IMAGE_SIDE", "2048"))
_GEMINI_MAX_IMAGE_BYTES = int(os.environ.get("GEMINI_MAX_IMAGE_BYTES", "8000000"))
_GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "180000"))
_GEMINI_RETRY_ATTEMPTS = int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "8"))
_GEMINI_RETRY_MAX_DELAY_S = float(os.environ.get("GEMINI_RETRY_MAX_DELAY_S", "120.0"))


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


def _encode_page_for_gemini(page_image: Image.Image) -> tuple[bytes, str, dict[str, int]]:
    img = page_image
    if max(img.size) > _GEMINI_MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((_GEMINI_MAX_IMAGE_SIDE, _GEMINI_MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    encoded = buf.getvalue()
    if len(encoded) > _GEMINI_MAX_IMAGE_BYTES:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
        encoded = buf.getvalue()
        mime = "image/jpeg"
    else:
        mime = "image/png"
    width, height = img.size
    meta = {
        "image_format": mime.split("/")[-1],
        "image_bytes": len(encoded),
        "image_width": int(width),
        "image_height": int(height),
    }
    return encoded, mime, meta


def _caption_one_page_gemini(
    *,
    model: str,
    page_image: Image.Image,
    panel_count: int,
    page_size: dict[str, int] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    from google.genai import types  # type: ignore

    client = _genai_client()
    page_bytes, mime, image_meta = _encode_page_for_gemini(page_image)
    user_text = build_user_text(panel_count=panel_count, page_size=page_size)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PAGE_CAPTION_SCHEMA_GEMINI,
        system_instruction=SYSTEM_INSTRUCTION,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=page_bytes, mime_type=mime), user_text],
        config=config,
    )
    payload = json.loads(resp.text or "{}")
    out = {
        "caption": str(payload.get("caption") or "").strip(),
        "page_tags": [tag for tag in payload.get("page_tags") or [] if tag in ALLOWED_PAGE_TAGS],
    }
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
        "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
        "reasoning_tokens": int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
        "total_tokens": int(getattr(um, "total_token_count", 0) or 0) if um else 0,
    }
    return out, image_meta, usage


# =========================================================================
# Manifest listing
# =========================================================================


def _list_annotation_relatives(*, bucket: str, annotation_prefix: str) -> set[str]:
    prefix = annotation_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.endswith(".jsonl") or key.endswith(".json"):
                relatives.add(key[len(prefix):])
    return relatives


def _existing_output_relatives(*, bucket: str, output_root: str) -> set[str]:
    prefix = output_root.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if "/_jobs/" in key or "/_audit/" in key:
                continue
            if key.endswith(".json"):
                relatives.add(key[len(prefix):])
    return relatives


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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotation_relatives = (
        _list_annotation_relatives(bucket=bucket, annotation_prefix=annotation_prefix)
        if annotation_prefix
        else set()
    )
    existing = set() if overwrite else _existing_output_relatives(bucket=bucket, output_root=output_root)
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
    }

    for page in paginator.paginate(Bucket=bucket, Prefix=source_root):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            suffix = Path(key).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
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
            has_annotation = annotation_relative in annotation_relatives
            if require_annotations and not has_annotation:
                stats["skipped_missing_annotation_count"] += 1
                continue
            output_relative = f"{chapter}/{page_id}.json"
            if not overwrite and output_relative in existing:
                stats["skipped_existing_count"] += 1
                continue
            row = {
                "chapter": chapter,
                "page_id": page_id,
                "relative_path": relative,
                "page_key": key,
                "output_key": _join_key(output_root, output_relative),
                "page_s3_uri": _join_s3_uri(bucket, key),
                "annotation_key": _join_key(annotation_prefix, annotation_relative) if has_annotation else "",
                "annotation_s3_uri": (
                    _join_s3_uri(bucket, _join_key(annotation_prefix, annotation_relative)) if has_annotation else ""
                ),
            }
            rows.append(row)
            if max_pages > 0 and len(rows) >= max_pages:
                return rows, stats
    return rows, stats


# =========================================================================
# Lambda entry points
# =========================================================================


def prepare_page_caption_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or DATASET_BUCKET).strip()
    source_prefix = _normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_PREFIX)
    annotation_prefix = str(event.get("annotation_prefix") or DEFAULT_ANNOTATION_PREFIX).strip().strip("/")
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    output_run = str(event.get("output_run") or DEFAULT_OUTPUT_RUN).strip().strip("/")
    if not output_run:
        raise ValueError("output_run must not be empty")
    output_root = _join_key(output_prefix, output_run)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))
    overwrite = bool(event.get("overwrite", False))
    require_annotations = bool(event.get("require_annotations", False))
    model = str(event.get("model") or DEFAULT_GEMINI_MODEL).strip()

    rows, page_stats = _list_page_rows(
        bucket=bucket,
        source_prefix=source_prefix,
        annotation_prefix=annotation_prefix,
        output_root=output_root,
        include_chapter_regex=include_chapter_regex,
        overwrite=overwrite,
        require_annotations=require_annotations,
        max_pages=max_pages,
    )
    manifest_key = _join_key(output_root, "_jobs", run_id, "page_manifest.jsonl")
    manifest_body = "".join(_json_dumps(row) for row in rows)
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
        "output_run": output_run,
        "output_root": output_root,
        "run_id": run_id,
        "model": model,
        "overwrite": overwrite,
        "created_at": _now_utc_iso(),
        "git_sha": str(event.get("git_sha") or ""),
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
        "output": {"bucket": bucket, "prefix": output_root, "output_run": output_run, "run_id": run_id},
        "stats": {**page_stats, "manifest_count": len(rows)},
    }


def caption_full_page(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json_or_jsonl(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    page = event.get("page") if isinstance(event.get("page"), dict) else event
    output_key = str(page["output_key"])
    if not bool(config.get("overwrite")) and _object_exists(bucket, output_key):
        return {"status": "skipped_existing", "output_key": output_key, "page_id": str(page.get("page_id") or "")}

    page_bytes = _get_s3_bytes(bucket, str(page["page_key"]))
    with Image.open(io.BytesIO(page_bytes)) as image:
        width, height = image.size
        page_image = image.convert("RGB")
        annotation_key = str(page.get("annotation_key") or "")
        panel_count = 0
        if annotation_key:
            try:
                annotation = _get_s3_json_or_jsonl(bucket, annotation_key)
                panel_count = _panel_count_from_annotation(annotation)
            except Exception:
                panel_count = 0
        page_size = {"width_px": int(width), "height_px": int(height)}
        caption_payload, image_meta, usage = _caption_one_page_gemini(
            model=str(config["model"]),
            page_image=page_image,
            panel_count=panel_count,
            page_size=page_size,
        )

    caption = str(caption_payload.get("caption") or "").strip()
    page_tags = list(caption_payload.get("page_tags") or [])
    output = {
        "schema_version": 1,
        "caption_type": "gemini_page_caption_v1",
        "status": "ok",
        "output_run": str(config["output_run"]),
        "run_id": str(config["run_id"]),
        "created_at": _now_utc_iso(),
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "page_size": page_size,
        "panel_count": panel_count,
        "caption": caption,
        "word_count": _word_count(caption),
        "exceeded_word_cap": _word_count(caption) > WORD_HARD_CAP,
        "page_tags": page_tags,
        "sources": {
            "page": str(page.get("page_s3_uri") or _join_s3_uri(bucket, str(page["page_key"]))),
            "annotation": str(page.get("annotation_s3_uri") or ""),
            "page_key": str(page["page_key"]),
            "annotation_key": annotation_key,
        },
        "image": image_meta,
        "model": {"id": str(config["model"]), "provider": "gemini"},
        "usage": usage,
    }
    _put_s3_json(bucket, output_key, output, pretty=False)
    return {
        "status": "ok",
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "output_key": output_key,
        "panel_count": panel_count,
        "usage": usage,
    }
