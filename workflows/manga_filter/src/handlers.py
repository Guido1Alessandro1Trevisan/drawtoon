from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_CLASSIFICATION_MODEL = os.environ.get(
    "DEFAULT_CLASSIFICATION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_INPUT_PREFIX = "datasets/pages/single"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/filtered"
DEFAULT_PROMPT_FILENAME = "classify_manga_pages.md"
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_MANGA_FILTER_MAX_CONCURRENCY", "64"))
MANGA_FILTER_BATCH_PARALLEL = max(1, int(os.environ.get("MANGA_FILTER_BATCH_PARALLEL", "8")))
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "8000"))
DEFAULT_FILTER_MODE = "manga"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NON_MANGA_SUFFIXES = ("_manwa", "_manhwa", "_manha", "_manhua", "_comic")
KNOWN_PLAIN_MANGA_CHAPTERS = {
    "jujutsu-kaisen",
    "monster",
    "my-hero-academia",
    "the-fragrant-flower-blooms-with-dignity",
    "vagabond",
    "vinland-saga",
}
S3_MAX_ATTEMPTS = 10

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_manga_panel_page": {
            "type": "boolean",
            "description": "True only for black-and-white manga story page, panel, or crop content.",
        },
        "page_type": {
            "type": "string",
            "enum": [
                "manga_panel_page",
                "title_or_chapter_page",
                "cover_or_illustration",
                "credits_or_text_page",
                "blank_or_low_content",
                "screenshot_or_ui",
                "color_page",
                "other_non_manga",
                "uncertain",
            ],
        },
        "reason": {
            "type": "string",
            "description": "Short visual reason for the decision.",
        },
    },
    "required": ["is_manga_panel_page", "page_type", "reason"],
    "additionalProperties": False,
}

_S3_CLIENT = None
_BEDROCK_RUNTIME_CLIENTS: dict[int, Any] = {}


def _json_dumps(payload: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "adaptive", "max_attempts": S3_MAX_ATTEMPTS},
                max_pool_connections=128,
                connect_timeout=10,
                read_timeout=300,
            ),
        )
    return _S3_CLIENT


def _bedrock_runtime_client(timeout_seconds: float = 900.0):
    read_timeout = max(1, min(900, int(math.ceil(float(timeout_seconds or 900.0)))))
    client = _BEDROCK_RUNTIME_CLIENTS.get(read_timeout)
    if client is None:
        client = boto3.client(
            "bedrock-runtime",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "standard", "total_max_attempts": 1},
                connect_timeout=10,
                read_timeout=read_timeout,
            ),
        )
        _BEDROCK_RUNTIME_CLIENTS[read_timeout] = client
    return client


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def make_run_token() -> str:
    return f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000, 9999)}"


def sanitize_s3_key_component(value: object, *, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value or "").strip()).strip("_")
    return normalized[:240] or fallback


def normalize_prefix(value: object, *, default: str) -> str:
    prefix = str(value or default).strip().strip("/")
    if not prefix:
        raise ValueError("S3 prefix must not be empty")
    return f"{prefix}/"


def join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def load_prompt_text(prompt_filename: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / str(prompt_filename).strip()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def get_s3_bytes(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def get_s3_json(bucket: str, key: str) -> dict[str, Any]:
    payload = json.loads(get_s3_bytes(bucket, key).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {join_s3_uri(bucket, key)}")
    return payload


def put_s3_json(bucket: str, key: str, payload: object) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=_json_dumps(payload, pretty=True).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def put_s3_bytes(bucket: str, key: str, body: bytes, *, content_type: str) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def put_s3_jsonl(bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    body = "".join(_json_dumps(row) + "\n" for row in rows).encode("utf-8")
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/x-ndjson; charset=utf-8",
    )


def head_s3_object(bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return _s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def delete_s3_prefix(bucket: str, prefix: str) -> dict[str, Any]:
    normalized_prefix = str(prefix or "").strip().lstrip("/")
    if not normalized_prefix:
        raise ValueError("Refusing to delete empty S3 prefix")
    paginator = _s3_client().get_paginator("list_objects_v2")
    deleted = 0
    batches = 0
    batch: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key:
                continue
            batch.append({"Key": key})
            if len(batch) >= 1000:
                _s3_client().delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
                deleted += len(batch)
                batches += 1
                batch = []
    if batch:
        _s3_client().delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
        deleted += len(batch)
        batches += 1
    return {"prefix": normalized_prefix, "deleted_objects": deleted, "delete_batches": batches}


def copy_s3_object(source_bucket: str, source_key: str, output_bucket: str, output_key: str) -> None:
    _s3_client().copy_object(
        Bucket=output_bucket,
        Key=output_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
        MetadataDirective="COPY",
    )


def delete_s3_object(bucket: str, key: str) -> None:
    _s3_client().delete_object(Bucket=bucket, Key=key)


def _image_format_from_key(key: str) -> str:
    mime, _ = mimetypes.guess_type(key)
    if mime == "image/png":
        return "png"
    if mime == "image/gif":
        return "gif"
    if mime == "image/webp":
        return "webp"
    return "jpeg"


def _page_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    data = bytes(image_bytes)
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if len(data) >= 4 and data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            while marker == 0xFF and index < len(data):
                marker = data[index]
                index += 1
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(data):
                break
            segment_length = int.from_bytes(data[index : index + 2], "big")
            if segment_length < 2 or index + segment_length > len(data):
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
                height = int.from_bytes(data[index + 3 : index + 5], "big")
                width = int.from_bytes(data[index + 5 : index + 7], "big")
                if width > 0 and height > 0:
                    return width, height
                break
            index += segment_length
    raise ValueError("Unsupported image format or missing image dimensions")


def _prepare_bedrock_image_block(image_bytes: bytes, image_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fmt = _image_format_from_key(image_key)
    width = height = 0
    try:
        width, height = _page_image_dimensions(image_bytes)
    except Exception:
        pass
    if (
        fmt in {"jpeg", "png", "gif", "webp"}
        and len(image_bytes) <= BEDROCK_MAX_IMAGE_BYTES
        and (width <= 0 or width <= BEDROCK_MAX_IMAGE_SIDE)
        and (height <= 0 or height <= BEDROCK_MAX_IMAGE_SIDE)
    ):
        return (
            {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
            {
                "image_format": fmt,
                "image_bytes": len(image_bytes),
                "image_width": width,
                "image_height": height,
                "reencoded": False,
            },
        )

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        scale = min(
            1.0,
            BEDROCK_MAX_IMAGE_SIDE / max(1, original_width),
            BEDROCK_MAX_IMAGE_SIDE / max(1, original_height),
        )
        if scale < 1.0:
            image = image.resize(
                (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in (92, 85, 78, 70, 62, 54, 46):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= BEDROCK_MAX_IMAGE_BYTES:
                sent_width, sent_height = image.size
                return (
                    {"image": {"format": "jpeg", "source": {"bytes": encoded}}},
                    {
                        "image_format": "jpeg",
                        "image_bytes": len(encoded),
                        "image_width": sent_width,
                        "image_height": sent_height,
                        "source_image_bytes": len(image_bytes),
                        "source_image_width": original_width,
                        "source_image_height": original_height,
                        "jpeg_quality": quality,
                        "reencoded": True,
                    },
                )
    raise RuntimeError(
        f"Image {image_key} could not be encoded under Bedrock image limit "
        f"({BEDROCK_MAX_IMAGE_BYTES} bytes)"
    )


def _bedrock_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_bedrock_tool_input(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    content = ((response.get("output") or {}).get("message") or {}).get("content") or []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and str(tool_use.get("name") or "") == tool_name:
            value = tool_use.get("input")
            if isinstance(value, dict):
                return value
            raise ValueError(f"Bedrock tool {tool_name!r} returned non-object input")
        text = block.get("text")
        if text:
            texts.append(str(text))
    for text in texts:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Bedrock response did not contain tool input for {tool_name!r}")


def _bedrock_error_is_retryable(
    exc: Exception,
    *,
    include_marketplace_access_denied: bool = False,
) -> bool:
    if isinstance(exc, BotoCoreError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = str((exc.response.get("Error") or {}).get("Code") or "")
    message = str((exc.response.get("Error") or {}).get("Message") or "")
    http_status = int((exc.response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    if include_marketplace_access_denied and code == "AccessDeniedException" and (
        "aws-marketplace" in message
        or "Marketplace subscription" in message
        or "try again after 2 minutes" in message
    ):
        return True
    return code in {
        "ThrottlingException",
        "TooManyRequestsException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",
        "ServiceUnavailableException",
    } or http_status in {408, 429, 500, 502, 503, 504}


def _bedrock_retry_delay_seconds(exc: Exception, default_delay: float) -> float:
    if isinstance(exc, ClientError):
        code = str((exc.response.get("Error") or {}).get("Code") or "")
        message = str((exc.response.get("Error") or {}).get("Message") or "")
        if code in {"ThrottlingException", "TooManyRequestsException"} and "too many tokens" in message.lower():
            return max(float(default_delay), 12.0)
        if code == "AccessDeniedException" and (
            "aws-marketplace" in message
            or "Marketplace subscription" in message
            or "try again after 2 minutes" in message
        ):
            return max(float(default_delay), 130.0)
    return float(default_delay)


def _bedrock_model_access_probe(
    *,
    model: str,
    timeout_seconds: float = 60.0,
    retries: int = 1,
) -> dict[str, Any]:
    started_at = time.time()
    delay = 2.0
    attempts: list[dict[str, Any]] = []
    for attempt in range(max(0, int(retries)) + 1):
        try:
            response = _bedrock_runtime_client(timeout_seconds).converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": "Reply with ok."}]}],
                inferenceConfig={"maxTokens": 4, "temperature": 0},
                requestMetadata={
                    "client_request_id": sanitize_s3_key_component(
                        f"drawtoon-manga-filter-preflight-{model}-{attempt}",
                        fallback="bedrock-preflight",
                    )[:256],
                },
            )
            return {
                "status": "ok",
                "model": model,
                "attempts": attempt + 1,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "usage": _bedrock_usage(response),
            }
        except Exception as exc:
            retryable = _bedrock_error_is_retryable(exc, include_marketplace_access_denied=True)
            attempts.append({"attempt": attempt + 1, "retryable": retryable, "error": str(exc)})
            if retryable and attempt < retries:
                sleep_seconds = _bedrock_retry_delay_seconds(exc, delay)
                time.sleep(sleep_seconds + random.uniform(0.0, 0.75))
                delay = min(delay * 2.0, 20.0)
                continue
            raise RuntimeError(
                f"Bedrock model access preflight failed for {model!r} "
                f"after {attempt + 1} attempt(s): {exc}"
            ) from exc


def _bedrock_preflight_enabled(event: dict[str, Any]) -> bool:
    value = event.get("bedrock_preflight")
    if value is None:
        value = event.get("bedrock_model_access_preflight")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _bool_event(event: dict[str, Any], key: str, *, default: bool) -> bool:
    value = event.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def bedrock_converse_tool(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    image_block: dict[str, Any],
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    max_output_tokens: int,
    timeout_seconds: float,
    client_request_id: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    response = _bedrock_runtime_client(timeout_seconds).converse(
        modelId=model,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": user_text},
                    image_block,
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": max(1, int(max_output_tokens)),
            "temperature": 0,
        },
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": tool_description,
                        "inputSchema": {"json": tool_schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        },
        requestMetadata={
            "client_request_id": sanitize_s3_key_component(client_request_id, fallback="request")[:256],
        },
    )
    return _extract_bedrock_tool_input(response, tool_name=tool_name), _bedrock_usage(response)


def infer_audit_prefix(run_id: str) -> tuple[str, str]:
    return DATASET_BUCKET, f"datasets/_stepfunctions_audit/filter-manga-pages/{run_id}"


def build_source_manifest(
    *,
    bucket: str,
    input_prefix: str,
    include_relative_path_regex: str,
) -> list[dict[str, Any]]:
    include_re = re.compile(include_relative_path_regex) if include_relative_path_regex else None
    rows: list[dict[str, Any]] = []
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            source_key = str(obj.get("Key") or "")
            if not source_key or source_key.endswith("/"):
                continue
            if Path(source_key).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative_path = source_key[len(input_prefix) :].lstrip("/")
            if not relative_path or relative_path.startswith("_"):
                continue
            if include_re and not include_re.search(relative_path):
                continue
            rows.append(
                {
                    "Key": source_key,
                    "Size": int(obj.get("Size") or 0),
                    "ETag": str(obj.get("ETag") or "").strip('"'),
                    "LastModified": str(obj.get("LastModified") or ""),
                }
            )
    return rows


def _normalize_filter_mode(value: object) -> str:
    # Only manga single-page mode is supported. All legacy aliases collapse to "manga".
    return "manga"


def _natural_sort_key(value: object) -> list[object]:
    parts = re.split(r"(\d+)", str(value or ""))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def normalize_manga_chapter_name(chapter: str) -> str:
    if (
        not chapter
        or chapter.startswith("_")
        or chapter.endswith("_manga")
        or chapter.endswith(NON_MANGA_SUFFIXES)
    ):
        return chapter
    if chapter.endswith(("_mangazero", "_manga109")) or chapter in KNOWN_PLAIN_MANGA_CHAPTERS:
        return f"{chapter}_manga"
    return chapter


def normalize_manga_relative_path(relative_path: str) -> str:
    parts = [part for part in Path(relative_path).as_posix().split("/") if part]
    if not parts:
        return relative_path
    parts[0] = normalize_manga_chapter_name(parts[0])
    return "/".join(parts)


def _source_row_from_s3_obj(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "Key": str(obj.get("Key") or ""),
        "Size": int(obj.get("Size") or 0),
        "ETag": str(obj.get("ETag") or "").strip('"'),
        "LastModified": str(obj.get("LastModified") or ""),
    }


def prepare_manga_filter_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or event.get("dataset_bucket") or DATASET_BUCKET).strip()
    if bucket != DATASET_BUCKET:
        raise ValueError(f"This workflow is configured for bucket={DATASET_BUCKET!r}, got {bucket!r}")
    input_prefix = normalize_prefix(event.get("input_prefix"), default=DEFAULT_INPUT_PREFIX)
    output_prefix = normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    if input_prefix == output_prefix:
        raise ValueError("input_prefix and output_prefix must be different")

    mode = _normalize_filter_mode(event.get("mode"))
    run_id = str(event.get("run_id") or "").strip() or make_run_token()
    requested_model = str(event.get("model") or "").strip()
    model = requested_model or DEFAULT_CLASSIFICATION_MODEL
    prompt_filename = str(event.get("prompt_filename") or DEFAULT_PROMPT_FILENAME).strip()
    prompt = load_prompt_text(prompt_filename)

    preflight = (
        _bedrock_model_access_probe(model=model)
        if _bedrock_preflight_enabled(event)
        else {"status": "disabled", "model": model}
    )
    status_prefix = f"{output_prefix.rstrip('/')}/_status"
    job_prefix = f"{output_prefix.rstrip('/')}/_jobs/{sanitize_s3_key_component(run_id)}"
    artifact_prefix = str(event.get("artifact_prefix") or f"{output_prefix.rstrip('/')}/_artifacts/{sanitize_s3_key_component(run_id)}").strip().strip("/")
    include_relative_path_regex = str(event.get("include_relative_path_regex") or "").strip()
    worker_config_key = f"{job_prefix}/worker_config.json"
    manifest_key = f"{job_prefix}/source_manifest.jsonl"
    manifest_rows = build_source_manifest(
        bucket=bucket,
        input_prefix=input_prefix,
        include_relative_path_regex=include_relative_path_regex,
    )
    put_s3_jsonl(bucket, manifest_key, manifest_rows)
    worker_config = {
        "mode": mode,
        "bucket": bucket,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "status_prefix": status_prefix,
        "artifact_prefix": artifact_prefix,
        "include_relative_path_regex": include_relative_path_regex,
        "model": model,
        "prompt_filename": prompt_filename,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "timeout_seconds": float(event.get("timeout_seconds") or 60.0),
        "retries": int(event["retries"]) if event.get("retries") is not None else 5,
        "max_output_tokens": max(1, int(event.get("max_output_tokens") or 160)),
        "overwrite": bool(event.get("overwrite", False)),
        "git_sha": str(event.get("git_sha") or "").strip(),
        "run_id": run_id,
        "prepared_at": now_utc_iso(),
    }
    put_s3_json(bucket, worker_config_key, worker_config)
    audit_bucket, audit_prefix = infer_audit_prefix(run_id)
    return {
        "source": {
            "bucket": bucket,
            "prefix": input_prefix,
            "input_prefix": input_prefix,
            "manifest_key": manifest_key,
            "manifest_count": len(manifest_rows),
            "mode": mode,
        },
        "output": {
            "bucket": bucket,
            "output_prefix": output_prefix,
            "status_prefix": status_prefix,
            "artifact_prefix": artifact_prefix,
        },
        "worker_config": {"bucket": bucket, "key": worker_config_key},
        "audit": {"bucket": audit_bucket, "prefix": audit_prefix},
        "batch": {
            "max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY)),
            "max_items_per_batch": max(1, int(event.get("max_items_per_batch") or 4)),
            "max_input_bytes_per_batch": max(1024, int(event.get("max_input_bytes_per_batch") or 131072)),
        },
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "run": {
            "run_id": run_id,
            "job_name": str(event.get("job_name") or "filter-manga-pages").strip(),
            "git_sha": str(event.get("git_sha") or "").strip(),
            "prepared_at": now_utc_iso(),
            "bedrock_preflight": preflight,
            "manifest_count": len(manifest_rows),
            "mode": mode,
        },
    }


def _object_key(item: dict[str, Any]) -> str:
    return str(item.get("Key") or item.get("key") or "").strip()


def _object_size(item: dict[str, Any]) -> int:
    try:
        return int(item.get("Size") or item.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def _object_etag(item: dict[str, Any]) -> str:
    return str(item.get("ETag") or item.get("etag") or "").strip().strip('"')


def _object_last_modified(item: dict[str, Any]) -> str:
    value = item.get("LastModified") or item.get("lastModified") or item.get("last_modified") or ""
    return str(value)


def _status_key_for_relative_path(status_prefix: str, relative_path: str) -> str:
    return f"{status_prefix.rstrip('/')}/{Path(relative_path).with_suffix('.json').as_posix()}"


def _classify_page(
    *,
    row_index: int,
    source_key: str,
    relative_path: str,
    image_bytes: bytes,
    image_meta: dict[str, Any],
    image_block: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt = load_prompt_text(str(config["prompt_filename"]))
    request_text = (
        "Classify this image for the Drawtoon manga-page filtering stage. "
        "The source object metadata is provided for traceability only; make the decision from the visible image.\n\n"
        f"{_json_dumps({'source_key': source_key, 'relative_path': relative_path, 'image': image_meta}, pretty=True)}"
    )
    retries = int(config["retries"]) if config.get("retries") is not None else 5
    delay = 4.0
    last_error = ""
    for attempt in range(retries + 1):
        try:
            parsed, usage = bedrock_converse_tool(
                model=str(config["model"]),
                system_prompt=prompt,
                user_text=request_text,
                image_block=image_block,
                tool_name="manga_page_classification",
                tool_description=(
                    "Classify whether the supplied image is usable black-and-white manga page/panel content "
                    "or a non-manga/title/credits/cover/text page that should be filtered out."
                ),
                tool_schema=CLASSIFICATION_SCHEMA,
                max_output_tokens=max(1, int(config.get("max_output_tokens") or 160)),
                timeout_seconds=float(config.get("timeout_seconds") or 60.0),
                client_request_id=f"drawtoon-manga-filter-{relative_path}-{int(row_index)}-{attempt}",
            )
            if not isinstance(parsed.get("is_manga_panel_page"), bool):
                raise ValueError("Bedrock classification omitted boolean is_manga_panel_page")
            page_type = str(parsed.get("page_type") or "").strip()
            if not page_type:
                page_type = "manga_panel_page" if parsed["is_manga_panel_page"] else "uncertain"
            reason = " ".join(str(parsed.get("reason") or "").split()).strip()
            if not reason:
                reason = "No reason returned by model."
            parsed["page_type"] = page_type
            parsed["reason"] = reason[:500]
            return parsed, usage
        except Exception as exc:
            last_error = str(exc)
            retryable = _bedrock_error_is_retryable(exc) or isinstance(exc, json.JSONDecodeError)
            if retryable and attempt < retries:
                sleep_seconds = _bedrock_retry_delay_seconds(exc, delay)
                time.sleep(sleep_seconds + random.uniform(0.0, 2.5))
                delay = min(delay * 2.0, 45.0)
                continue
            break
    raise RuntimeError(f"classification failed for {source_key}: {last_error or 'unknown error'}")



def filter_single_page(row_index: int, obj: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    bucket = str(config["bucket"])
    input_prefix = str(config["input_prefix"])
    output_prefix = str(config["output_prefix"])
    status_prefix = str(config["status_prefix"])
    source_key = _object_key(obj)
    if not source_key or source_key.endswith("/"):
        return {"row_index": int(row_index), "status": "skipped_non_object", "source_key": source_key}
    if not source_key.startswith(input_prefix):
        return {"row_index": int(row_index), "status": "skipped_outside_prefix", "source_key": source_key}
    if Path(source_key).suffix.lower() not in SUPPORTED_SUFFIXES:
        return {"row_index": int(row_index), "status": "skipped_non_image", "source_key": source_key}

    relative_path = source_key[len(input_prefix) :].lstrip("/")
    if not relative_path or relative_path.startswith("_"):
        return {"row_index": int(row_index), "status": "skipped_internal", "source_key": source_key}
    include_relative_path_regex = str(config.get("include_relative_path_regex") or "").strip()
    if include_relative_path_regex and not re.search(include_relative_path_regex, relative_path):
        return {
            "row_index": int(row_index),
            "status": "skipped_key_filter",
            "source_key": source_key,
        }
    output_relative_path = normalize_manga_relative_path(relative_path)
    output_key = f"{output_prefix.rstrip('/')}/{output_relative_path}"
    status_key = _status_key_for_relative_path(status_prefix, output_relative_path)

    if not bool(config.get("overwrite", False)) and head_s3_object(bucket, status_key) is not None:
        existing = get_s3_json(bucket, status_key)
        if str(existing.get("status") or "") == "ok":
            return {
                "row_index": int(row_index),
                "status": "skipped_existing",
                "source_key": source_key,
                "status_key": status_key,
                "is_manga_panel_page": bool(existing.get("is_manga_panel_page", False)),
                "copied": bool((existing.get("output") or {}).get("copied", False)),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

    image_meta: dict[str, Any] = {}
    try:
        image_bytes = get_s3_bytes(bucket, source_key)
        image_block, image_meta = _prepare_bedrock_image_block(image_bytes, source_key)
        parsed, usage = _classify_page(
            row_index=row_index,
            source_key=source_key,
            relative_path=relative_path,
            image_bytes=image_bytes,
            image_meta=image_meta,
            image_block=image_block,
            config=config,
        )
        is_manga = bool(parsed["is_manga_panel_page"])
        copied = False
        if is_manga:
            if bool(config.get("overwrite", False)) or head_s3_object(bucket, output_key) is None:
                copy_s3_object(bucket, source_key, bucket, output_key)
            copied = True
        elif bool(config.get("overwrite", False)):
            delete_s3_object(bucket, output_key)
        payload = {
            "schema_version": 1,
            "row_index": int(row_index),
            "status": "ok",
            "is_manga_panel_page": is_manga,
            "page_type": str(parsed.get("page_type") or ""),
            "reason": str(parsed.get("reason") or ""),
            "source": {
                "bucket": bucket,
                "key": source_key,
                "s3_uri": join_s3_uri(bucket, source_key),
                "relative_path": relative_path,
                "output_relative_path": output_relative_path,
                "size": _object_size(obj),
                "etag": _object_etag(obj),
                "last_modified": _object_last_modified(obj),
            },
            "output": {
                "bucket": bucket,
                "key": output_key if copied else "",
                "s3_uri": join_s3_uri(bucket, output_key) if copied else "",
                "copied": copied,
            },
            "model": str(config["model"]),
            "prompt_filename": str(config["prompt_filename"]),
            "prompt_sha256": str(config.get("prompt_sha256") or ""),
            "image": image_meta,
            "usage": usage,
            "run_id": str(config.get("run_id") or ""),
            "created_at": now_utc_iso(),
        }
        put_s3_json(bucket, status_key, payload)
        return payload
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "row_index": int(row_index),
            "status": "error",
            "is_manga_panel_page": False,
            "page_type": "uncertain",
            "reason": "Operational failure during classification.",
            "error": str(exc),
            "source": {
                "bucket": bucket,
                "key": source_key,
                "s3_uri": join_s3_uri(bucket, source_key),
                "relative_path": relative_path,
                "size": _object_size(obj),
                "etag": _object_etag(obj),
                "last_modified": _object_last_modified(obj),
            },
            "output": {"bucket": bucket, "key": "", "s3_uri": "", "copied": False},
            "model": str(config.get("model") or ""),
            "prompt_filename": str(config.get("prompt_filename") or ""),
            "prompt_sha256": str(config.get("prompt_sha256") or ""),
            "image": image_meta,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "run_id": str(config.get("run_id") or ""),
            "created_at": now_utc_iso(),
        }
        put_s3_json(bucket, status_key, payload)
        return payload


def filter_manga_page_batch(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    items = list(event.get("Items") or [])
    if not items:
        return {
            "input_count": 0,
            "success_count": 0,
            "kept_count": 0,
            "filtered_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "total_tokens": 0,
        }
    batch_input = dict(event.get("BatchInput") or {})
    config_ref = dict(batch_input.get("config_ref") or {})
    if not config_ref:
        config_ref = dict(items[0].get("config_ref") or {})
    if not config_ref:
        raise ValueError("Missing config_ref")
    config = get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))

    success_count = 0
    kept_count = 0
    filtered_count = 0
    error_count = 0
    skipped_count = 0
    total_tokens = 0
    error_examples: list[dict[str, Any]] = []
    _normalize_filter_mode(config.get("mode"))

    def _process_item(item: dict[str, Any]) -> dict[str, Any] | None:
        row_index_local = int(item.get("row_index") or 0)
        obj_local = item.get("object")
        if not isinstance(obj_local, dict):
            return None
        return filter_single_page(row_index_local, obj_local, config)

    workers = max(1, min(int(MANGA_FILTER_BATCH_PARALLEL), len(items)))
    if workers <= 1:
        payloads_ordered: list[dict[str, Any] | None] = [_process_item(item) for item in items]
    else:
        payloads_ordered = [None] * len(items)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_process_item, item): index for index, item in enumerate(items)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    payloads_ordered[index] = future.result()
                except Exception as exc:
                    payloads_ordered[index] = {"status": "error", "error": str(exc)}

    for payload in payloads_ordered:
        if payload is None:
            error_count += 1
            continue
        status = str(payload.get("status") or "")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        total_tokens += int(usage.get("total_tokens") or 0)
        if status in {"ok", "skipped_existing"}:
            success_count += 1
            if bool(payload.get("is_manga_panel_page", False)):
                kept_count += 1
            else:
                filtered_count += 1
        elif status.startswith("skipped_"):
            skipped_count += 1
        else:
            error_count += 1
            if len(error_examples) < 10:
                source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
                error_examples.append(
                    {
                        "source_key": source.get("key") or payload.get("source_key"),
                        "error": payload.get("error"),
                    }
                )
    return {
        "input_count": len(items),
        "success_count": success_count,
        "kept_count": kept_count,
        "filtered_count": filtered_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "total_tokens": total_tokens,
    }


def finalize_manga_filter_run(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("worker_config") if isinstance(event.get("worker_config"), dict) else {}
    if not config_ref:
        config_ref = dict((event.get("prepared") or {}).get("worker_config") or {})
    if not config_ref:
        raise ValueError("Missing worker_config")
    config = get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    _normalize_filter_mode(config.get("mode"))
    bucket = str(config["bucket"])
    output_prefix = str(config["output_prefix"])
    summary = {
        "schema_version": 1,
        "mode": "manga",
        "status": "ok",
        "run_id": str(config.get("run_id") or ""),
        "created_at": now_utc_iso(),
    }
    summary_key = f"{output_prefix.rstrip('/')}/_jobs/{sanitize_s3_key_component(config['run_id'])}/manga_filter_summary.json"
    put_s3_json(bucket, summary_key, summary)
    return {**summary, "summary_key": summary_key, "bucket": bucket}
