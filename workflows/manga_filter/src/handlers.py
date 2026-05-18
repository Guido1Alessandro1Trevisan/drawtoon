from __future__ import annotations

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
DEFAULT_MANHWA_PROMPT_FILENAME = "classify_manhwa_page_pairs.md"
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_MANGA_FILTER_MAX_CONCURRENCY", "64"))
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "8000"))
DEFAULT_FILTER_MODE = "manga"
MANHWA_DIAGNOSTIC_WIDTH = int(os.environ.get("MANHWA_DIAGNOSTIC_WIDTH", "896"))
MANHWA_FULL_THUMB_HEIGHT = int(os.environ.get("MANHWA_FULL_THUMB_HEIGHT", "640"))
MANHWA_EDGE_CROP_SOURCE_PX = int(os.environ.get("MANHWA_EDGE_CROP_SOURCE_PX", "1536"))
MANHWA_DIAGNOSTIC_JPEG_QUALITY = int(os.environ.get("MANHWA_DIAGNOSTIC_JPEG_QUALITY", "82"))
MANHWA_CHAIN_CONFIDENCE_THRESHOLD = float(os.environ.get("MANHWA_CHAIN_CONFIDENCE_THRESHOLD", "0.75"))
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
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

MANHWA_PAGE_TYPES = [
    "story",
    "title_or_chapter",
    "cover_or_illustration",
    "credits_or_text",
    "blank_or_low_content",
    "screenshot_or_ui",
    "other_non_story",
    "uncertain",
]

MANHWA_PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "left_page_type": {"type": "string", "enum": MANHWA_PAGE_TYPES},
        "right_page_type": {"type": "string", "enum": MANHWA_PAGE_TYPES},
        "left_is_story_page": {"type": "boolean"},
        "right_is_story_page": {"type": "boolean"},
        "left_page_confidence": {"type": "number", "description": "0.0 to 1.0 confidence for the left page type."},
        "right_page_confidence": {"type": "number", "description": "0.0 to 1.0 confidence for the right page type."},
        "continues_same_panel": {
            "type": "boolean",
            "description": "True when the bottom of the left page and top of the right page are one continuous panel/artwork.",
        },
        "chain_break": {
            "type": "boolean",
            "description": "True when this pair should break hard same-panel chains.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 confidence for the same-panel continuation decision.",
        },
        "reason": {
            "type": "string",
            "description": "Short visual reason for the page-type and continuation decisions.",
        },
    },
    "required": [
        "left_page_type",
        "right_page_type",
        "left_is_story_page",
        "right_is_story_page",
        "left_page_confidence",
        "right_page_confidence",
        "continues_same_panel",
        "chain_break",
        "confidence",
        "reason",
    ],
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


def copy_s3_object(source_bucket: str, source_key: str, output_bucket: str, output_key: str) -> None:
    _s3_client().copy_object(
        Bucket=output_bucket,
        Key=output_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
        MetadataDirective="COPY",
    )


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
    mode = str(value or DEFAULT_FILTER_MODE).strip().lower()
    aliases = {"manwa": "manhwa", "webtoon": "manhwa"}
    mode = aliases.get(mode, mode)
    if mode not in {"manga", "manhwa"}:
        raise ValueError(f"Unsupported filter mode={mode!r}; expected manga or manhwa")
    return mode


def _natural_sort_key(value: object) -> list[object]:
    parts = re.split(r"(\d+)", str(value or ""))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _source_row_from_s3_obj(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "Key": str(obj.get("Key") or ""),
        "Size": int(obj.get("Size") or 0),
        "ETag": str(obj.get("ETag") or "").strip('"'),
        "LastModified": str(obj.get("LastModified") or ""),
    }


def _manhwa_chapter_key(relative_path: str) -> str:
    parts = [part for part in Path(relative_path).as_posix().split("/") if part]
    if len(parts) >= 3:
        return "/".join(parts[:-1])
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def _manhwa_pair_status_key(status_prefix: str, chapter_key: str, pair_index: int) -> str:
    return f"{status_prefix.rstrip('/')}/{chapter_key}/_pairs/pair_{int(pair_index):05d}.json"


def build_manhwa_pair_manifest(
    *,
    bucket: str,
    input_prefix: str,
    status_prefix: str,
    include_relative_path_regex: str,
) -> list[dict[str, Any]]:
    include_re = re.compile(include_relative_path_regex) if include_relative_path_regex else None
    chapter_pages: dict[str, list[dict[str, Any]]] = {}
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
            default_source_root = DEFAULT_INPUT_PREFIX.rstrip("/") + "/"
            chapter_relative_path = (
                source_key[len(default_source_root) :]
                if source_key.startswith(default_source_root)
                else relative_path
            )
            chapter_key = _manhwa_chapter_key(chapter_relative_path)
            row = _source_row_from_s3_obj(obj)
            row.update({"relative_path": chapter_relative_path, "chapter_key": chapter_key})
            chapter_pages.setdefault(chapter_key, []).append(row)

    manifest_rows: list[dict[str, Any]] = []
    for chapter_key in sorted(chapter_pages, key=_natural_sort_key):
        pages = sorted(chapter_pages[chapter_key], key=lambda item: _natural_sort_key(item["relative_path"]))
        if len(pages) < 2:
            continue
        for pair_index, (left, right) in enumerate(zip(pages, pages[1:], strict=False)):
            manifest_rows.append(
                {
                    "row_type": "manhwa_page_pair",
                    "chapter_key": chapter_key,
                    "pair_index": pair_index,
                    "left_page_index": pair_index,
                    "right_page_index": pair_index + 1,
                    "left": left,
                    "right": right,
                    "status_key": _manhwa_pair_status_key(status_prefix, chapter_key, pair_index),
                }
            )
    return manifest_rows


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
    model = str(event.get("model") or DEFAULT_CLASSIFICATION_MODEL).strip()
    default_prompt = DEFAULT_MANHWA_PROMPT_FILENAME if mode == "manhwa" else DEFAULT_PROMPT_FILENAME
    prompt_filename = str(event.get("prompt_filename") or default_prompt).strip()
    prompt = load_prompt_text(prompt_filename)

    preflight = (
        _bedrock_model_access_probe(model=model)
        if _bedrock_preflight_enabled(event)
        else {"status": "disabled", "model": model}
    )
    status_prefix = f"{output_prefix.rstrip('/')}/_status"
    job_prefix = f"{output_prefix.rstrip('/')}/_jobs/{sanitize_s3_key_component(run_id)}"
    include_relative_path_regex = str(event.get("include_relative_path_regex") or "").strip()
    worker_config_key = f"{job_prefix}/worker_config.json"
    manifest_key = f"{job_prefix}/source_manifest.jsonl"
    if mode == "manhwa":
        manifest_rows = build_manhwa_pair_manifest(
            bucket=bucket,
            input_prefix=input_prefix,
            status_prefix=status_prefix,
            include_relative_path_regex=include_relative_path_regex,
        )
    else:
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
        "include_relative_path_regex": include_relative_path_regex,
        "model": model,
        "prompt_filename": prompt_filename,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "timeout_seconds": float(event.get("timeout_seconds") or 60.0),
        "retries": int(event["retries"]) if event.get("retries") is not None else 5,
        "max_output_tokens": max(1, int(event.get("max_output_tokens") or (220 if mode == "manhwa" else 160))),
        "overwrite": bool(event.get("overwrite", False)),
        "git_sha": str(event.get("git_sha") or "").strip(),
        "run_id": run_id,
        "prepared_at": now_utc_iso(),
        "manhwa": {
            "diagnostic_width": max(256, int(event.get("manhwa_diagnostic_width") or MANHWA_DIAGNOSTIC_WIDTH)),
            "full_thumb_height": max(128, int(event.get("manhwa_full_thumb_height") or MANHWA_FULL_THUMB_HEIGHT)),
            "edge_crop_source_px": max(256, int(event.get("manhwa_edge_crop_source_px") or MANHWA_EDGE_CROP_SOURCE_PX)),
            "jpeg_quality": max(40, min(95, int(event.get("manhwa_jpeg_quality") or MANHWA_DIAGNOSTIC_JPEG_QUALITY))),
            "chain_confidence_threshold": max(
                0.0,
                min(
                    1.0,
                    float(event.get("manhwa_chain_confidence_threshold") or MANHWA_CHAIN_CONFIDENCE_THRESHOLD),
                ),
            ),
        },
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


def _fit_rgb_image(image: Any, *, max_width: int, max_height: int) -> Any:
    from PIL import Image

    width, height = image.size
    scale = min(float(max_width) / max(1, width), float(max_height) / max(1, height))
    scale = max(0.01, scale)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    if target == image.size:
        return image.copy()
    return image.resize(target, Image.Resampling.LANCZOS)


def _edge_crop(image: Any, *, edge: str, source_px: int) -> Any:
    width, height = image.size
    crop_height = min(height, max(1, min(int(source_px), max(256, int(height * 0.35)))))
    if edge == "bottom":
        return image.crop((0, height - crop_height, width, height))
    return image.crop((0, 0, width, crop_height))


def _build_manhwa_pair_diagnostic_image(
    *,
    left_bytes: bytes,
    right_bytes: bytes,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image, ImageDraw

    manhwa_config = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    diagnostic_width = max(256, int(manhwa_config.get("diagnostic_width") or MANHWA_DIAGNOSTIC_WIDTH))
    full_thumb_height = max(128, int(manhwa_config.get("full_thumb_height") or MANHWA_FULL_THUMB_HEIGHT))
    edge_crop_source_px = max(256, int(manhwa_config.get("edge_crop_source_px") or MANHWA_EDGE_CROP_SOURCE_PX))
    jpeg_quality = max(40, min(95, int(manhwa_config.get("jpeg_quality") or MANHWA_DIAGNOSTIC_JPEG_QUALITY)))

    with Image.open(io.BytesIO(left_bytes)) as left_image, Image.open(io.BytesIO(right_bytes)) as right_image:
        left = left_image.convert("RGB")
        right = right_image.convert("RGB")
        left_size = left.size
        right_size = right.size

        gutter = max(8, diagnostic_width // 56)
        label_height = 24
        content_width = max(96, diagnostic_width - gutter * 2)
        # Full-page views are intentionally non-uniform overview thumbnails.
        # Tall webtoon strips become unreadably narrow if resized proportionally
        # to a fixed height; the boundary crops below preserve true geometry.
        left_full = left.resize((content_width, full_thumb_height), Image.Resampling.LANCZOS)
        right_full = right.resize((content_width, full_thumb_height), Image.Resampling.LANCZOS)
        left_bottom = _fit_rgb_image(
            _edge_crop(left, edge="bottom", source_px=edge_crop_source_px),
            max_width=content_width,
            max_height=max(256, full_thumb_height * 2),
        )
        right_top = _fit_rgb_image(
            _edge_crop(right, edge="top", source_px=edge_crop_source_px),
            max_width=content_width,
            max_height=max(256, full_thumb_height * 2),
        )
        guide_width = max(4, content_width // 160)
        ImageDraw.Draw(left_bottom).line(
            [(0, left_bottom.height - 1), (left_bottom.width, left_bottom.height - 1)],
            fill=(220, 0, 0),
            width=guide_width,
        )
        ImageDraw.Draw(right_top).line(
            [(0, 0), (right_top.width, 0)],
            fill=(220, 0, 0),
            width=guide_width,
        )

        sections = [
            ("left full page", left_full),
            ("right full page", right_full),
            ("left bottom crop", left_bottom),
            ("right top crop", right_top),
        ]
        canvas_height = gutter + sum(label_height + image.height + gutter for _, image in sections)
        canvas = Image.new("RGB", (diagnostic_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        y = gutter
        for label, image in sections:
            draw.text((gutter, y), label, fill=(0, 0, 0))
            y += label_height
            x = max(0, (diagnostic_width - image.width) // 2)
            canvas.paste(image, (x, y))
            y += image.height + gutter

    encoded = b""
    used_quality = jpeg_quality
    for quality in (jpeg_quality, 76, 70, 64, 58, 52):
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = output.getvalue()
        used_quality = quality
        if len(encoded) <= BEDROCK_MAX_IMAGE_BYTES:
            break
    if len(encoded) > BEDROCK_MAX_IMAGE_BYTES:
        raise RuntimeError(
            f"Manhwa diagnostic image could not be encoded under Bedrock image limit "
            f"({len(encoded)} bytes > {BEDROCK_MAX_IMAGE_BYTES} bytes)"
        )
    meta = {
        "image_format": "jpeg",
        "image_bytes": len(encoded),
        "image_width": canvas.width,
        "image_height": canvas.height,
        "jpeg_quality": used_quality,
        "diagnostic_width": diagnostic_width,
        "full_thumb_height": full_thumb_height,
        "edge_crop_source_px": edge_crop_source_px,
        "boundary_guide": "red line marks the bottom edge of the left page and top edge of the right page",
        "left_source_width": left_size[0],
        "left_source_height": left_size[1],
        "right_source_width": right_size[0],
        "right_source_height": right_size[1],
    }
    return {"image": {"format": "jpeg", "source": {"bytes": encoded}}}, meta


def _classify_manhwa_pair(
    *,
    row_index: int,
    pair: dict[str, Any],
    image_block: dict[str, Any],
    image_meta: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt = load_prompt_text(str(config["prompt_filename"]))
    chapter_key = str(pair.get("chapter_key") or "")
    pair_index = int(pair.get("pair_index") or 0)
    request_text = (
        "Classify this adjacent manhwa page pair. The diagnostic image is a vertical composite with four labeled sections: "
        "left full page, right full page, the bottom crop of the left page, and the top crop of the right page. "
        "Use the full-page thumbnails for title/non-story detection and the boundary crops for chain detection.\n\n"
        f"{_json_dumps({'chapter_key': chapter_key, 'pair_index': pair_index, 'image': image_meta}, pretty=True)}"
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
                tool_name="manhwa_page_pair_classification",
                tool_description=(
                    "Classify adjacent manhwa pages as story/non-story and decide whether the pages "
                    "are connected parts of the same continuous panel."
                ),
                tool_schema=MANHWA_PAIR_SCHEMA,
                max_output_tokens=max(1, int(config.get("max_output_tokens") or 220)),
                timeout_seconds=float(config.get("timeout_seconds") or 60.0),
                client_request_id=f"drawtoon-manhwa-filter-{chapter_key}-{pair_index}-{int(row_index)}-{attempt}",
            )
            for key in ("left_page_type", "right_page_type"):
                value = str(parsed.get(key) or "").strip()
                if value not in MANHWA_PAGE_TYPES:
                    raise ValueError(f"Bedrock classification returned invalid {key}={value!r}")
                parsed[key] = value
            for key in ("left_is_story_page", "right_is_story_page", "continues_same_panel", "chain_break"):
                if not isinstance(parsed.get(key), bool):
                    raise ValueError(f"Bedrock classification omitted boolean {key}")
            for key in ("left_page_confidence", "right_page_confidence", "confidence"):
                try:
                    parsed[key] = max(0.0, min(1.0, float(parsed.get(key) or 0.0)))
                except (TypeError, ValueError):
                    parsed[key] = 0.0
            parsed["left_is_story_page"] = bool(parsed["left_is_story_page"]) and parsed["left_page_type"] == "story"
            parsed["right_is_story_page"] = bool(parsed["right_is_story_page"]) and parsed["right_page_type"] == "story"
            if not bool(parsed["left_is_story_page"]) or not bool(parsed["right_is_story_page"]):
                parsed["continues_same_panel"] = False
                parsed["chain_break"] = True
            else:
                parsed["continues_same_panel"] = bool(parsed["continues_same_panel"])
                parsed["chain_break"] = not parsed["continues_same_panel"]
            reason = " ".join(str(parsed.get("reason") or "").split()).strip()
            parsed["reason"] = (reason or "No reason returned by model.")[:500]
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
    raise RuntimeError(
        f"manhwa pair classification failed for {pair.get('chapter_key')} pair {pair.get('pair_index')}: "
        f"{last_error or 'unknown error'}"
    )


def _copy_filtered_page_if_story(
    *,
    bucket: str,
    source_page: dict[str, Any],
    output_prefix: str,
    is_story_page: bool,
    overwrite: bool,
) -> dict[str, Any]:
    relative_path = str(source_page.get("relative_path") or "").strip()
    output_key = f"{output_prefix.rstrip('/')}/{relative_path}" if relative_path else ""
    copied = False
    if is_story_page and output_key:
        if overwrite or head_s3_object(bucket, output_key) is None:
            copy_s3_object(bucket, str(source_page["Key"]), bucket, output_key)
        copied = True
    return {
        "bucket": bucket,
        "key": output_key if copied else "",
        "s3_uri": join_s3_uri(bucket, output_key) if copied else "",
        "copied": copied,
    }


def _manhwa_page_payload(
    *,
    bucket: str,
    page: dict[str, Any],
    page_index: int,
    page_type: str,
    is_story_page: bool,
    confidence: float,
    output: dict[str, Any],
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    return {
        "page_index": int(page_index),
        "page_type": page_type,
        "is_story_page": bool(is_story_page),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "width": int(width or 0),
        "height": int(height or 0),
        "source": {
            "bucket": bucket,
            "key": str(page.get("Key") or ""),
            "s3_uri": join_s3_uri(bucket, str(page.get("Key") or "")),
            "relative_path": str(page.get("relative_path") or ""),
            "size": _object_size(page),
            "etag": _object_etag(page),
            "last_modified": _object_last_modified(page),
        },
        "output": output,
    }


def filter_manhwa_page_pair(row_index: int, pair: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    bucket = str(config["bucket"])
    output_prefix = str(config["output_prefix"])
    status_key = str(pair.get("status_key") or "")
    if not status_key:
        status_key = _manhwa_pair_status_key(
            str(config["status_prefix"]),
            str(pair.get("chapter_key") or "unknown"),
            int(pair.get("pair_index") or 0),
        )
    if not bool(config.get("overwrite", False)) and head_s3_object(bucket, status_key) is not None:
        existing = get_s3_json(bucket, status_key)
        if str(existing.get("status") or "") == "ok":
            return {
                "row_index": int(row_index),
                "status": "skipped_existing",
                "row_type": "manhwa_page_pair",
                "status_key": status_key,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "kept_page_count": int(existing.get("kept_page_count") or 0),
            }

    left = pair.get("left") if isinstance(pair.get("left"), dict) else {}
    right = pair.get("right") if isinstance(pair.get("right"), dict) else {}
    image_meta: dict[str, Any] = {}
    try:
        left_bytes = get_s3_bytes(bucket, str(left["Key"]))
        right_bytes = get_s3_bytes(bucket, str(right["Key"]))
        image_block, image_meta = _build_manhwa_pair_diagnostic_image(
            left_bytes=left_bytes,
            right_bytes=right_bytes,
            config=config,
        )
        parsed, usage = _classify_manhwa_pair(
            row_index=row_index,
            pair=pair,
            image_block=image_block,
            image_meta=image_meta,
            config=config,
        )
        left_output = _copy_filtered_page_if_story(
            bucket=bucket,
            source_page=left,
            output_prefix=output_prefix,
            is_story_page=bool(parsed["left_is_story_page"]),
            overwrite=bool(config.get("overwrite", False)),
        )
        right_output = _copy_filtered_page_if_story(
            bucket=bucket,
            source_page=right,
            output_prefix=output_prefix,
            is_story_page=bool(parsed["right_is_story_page"]),
            overwrite=bool(config.get("overwrite", False)),
        )
        kept_page_count = int(bool(left_output["copied"])) + int(bool(right_output["copied"]))
        payload = {
            "schema_version": 1,
            "mode": "manhwa",
            "row_type": "manhwa_page_pair",
            "row_index": int(row_index),
            "status": "ok",
            "chapter_key": str(pair.get("chapter_key") or ""),
            "pair_index": int(pair.get("pair_index") or 0),
            "left": _manhwa_page_payload(
                bucket=bucket,
                page=left,
                page_index=int(pair.get("left_page_index") or 0),
                page_type=str(parsed["left_page_type"]),
                is_story_page=bool(parsed["left_is_story_page"]),
                confidence=float(parsed["left_page_confidence"]),
                output=left_output,
                width=int(image_meta.get("left_source_width") or 0),
                height=int(image_meta.get("left_source_height") or 0),
            ),
            "right": _manhwa_page_payload(
                bucket=bucket,
                page=right,
                page_index=int(pair.get("right_page_index") or 0),
                page_type=str(parsed["right_page_type"]),
                is_story_page=bool(parsed["right_is_story_page"]),
                confidence=float(parsed["right_page_confidence"]),
                output=right_output,
                width=int(image_meta.get("right_source_width") or 0),
                height=int(image_meta.get("right_source_height") or 0),
            ),
            "edge": {
                "continues_same_panel": bool(parsed["continues_same_panel"]),
                "chain_break": bool(parsed["chain_break"]),
                "confidence": float(parsed["confidence"]),
                "reason": str(parsed["reason"]),
            },
            "kept_page_count": kept_page_count,
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
            "mode": "manhwa",
            "row_type": "manhwa_page_pair",
            "row_index": int(row_index),
            "status": "error",
            "chapter_key": str(pair.get("chapter_key") or ""),
            "pair_index": int(pair.get("pair_index") or 0),
            "status_key": status_key,
            "error": str(exc),
            "image": image_meta,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "run_id": str(config.get("run_id") or ""),
            "created_at": now_utc_iso(),
        }
        put_s3_json(bucket, status_key, payload)
        return payload


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
    output_key = f"{output_prefix.rstrip('/')}/{relative_path}"
    status_key = _status_key_for_relative_path(status_prefix, relative_path)

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
    mode = _normalize_filter_mode(config.get("mode"))
    for item in items:
        row_index = int(item.get("row_index") or 0)
        obj = item.get("object")
        if not isinstance(obj, dict):
            error_count += 1
            continue
        row_type = str(obj.get("row_type") or "").strip()
        if mode == "manhwa" or row_type == "manhwa_page_pair":
            payload = filter_manhwa_page_pair(row_index, obj, config)
        else:
            payload = filter_single_page(row_index, obj, config)
        status = str(payload.get("status") or "")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        total_tokens += int(usage.get("total_tokens") or 0)
        if status in {"ok", "skipped_existing"}:
            success_count += 1
            if str(payload.get("row_type") or "") == "manhwa_page_pair":
                pair_kept_count = int(payload.get("kept_page_count") or 0)
                kept_count += pair_kept_count
                filtered_count += max(0, 2 - pair_kept_count)
            elif bool(payload.get("is_manga_panel_page", False)):
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


def _load_jsonl_from_s3(bucket: str, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = get_s3_bytes(bucket, key).decode("utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _chapter_manifest_key(output_prefix: str, chapter_key: str) -> str:
    return f"{output_prefix.rstrip('/')}/{chapter_key.strip('/')}/manifest.json"


def _select_page_vote(votes: list[dict[str, Any]]) -> tuple[str, float]:
    clean_votes = [
        vote
        for vote in votes
        if isinstance(vote, dict) and str(vote.get("page_type") or "") in MANHWA_PAGE_TYPES
    ]
    if not clean_votes:
        return "uncertain", 0.0
    story_votes = [vote for vote in clean_votes if str(vote.get("page_type") or "") == "story"]
    if story_votes:
        confidence = max(float(vote.get("confidence") or 0.0) for vote in story_votes)
        return "story", max(0.0, min(1.0, confidence))
    type_votes = [str(vote.get("page_type") or "") for vote in clean_votes]
    ranked = sorted(
        ((type_votes.count(vote), vote) for vote in set(type_votes)),
        key=lambda item: (-item[0], MANHWA_PAGE_TYPES.index(item[1])),
    )
    page_type = ranked[0][1]
    confidence = max(
        float(vote.get("confidence") or 0.0)
        for vote in clean_votes
        if str(vote.get("page_type") or "") == page_type
    )
    return page_type, max(0.0, min(1.0, confidence))


def _build_chain_components(story_page_indices: set[int], connected_edges: set[int]) -> list[list[int]]:
    if not story_page_indices:
        return []
    parent = {index: index for index in story_page_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge_index in connected_edges:
        if edge_index in story_page_indices and edge_index + 1 in story_page_indices:
            union(edge_index, edge_index + 1)
    groups: dict[int, list[int]] = {}
    for index in sorted(story_page_indices):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def finalize_manga_filter_run(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("worker_config") if isinstance(event.get("worker_config"), dict) else {}
    if not config_ref:
        config_ref = dict((event.get("prepared") or {}).get("worker_config") or {})
    if not config_ref:
        raise ValueError("Missing worker_config")
    config = get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    mode = _normalize_filter_mode(config.get("mode"))
    if mode != "manhwa":
        return {"status": "skipped", "mode": mode, "reason": "chapter manifests are only built for manhwa mode"}

    bucket = str(config["bucket"])
    output_prefix = str(config["output_prefix"])
    manifest_key = (
        str((event.get("source") or {}).get("manifest_key") or "")
        if isinstance(event.get("source"), dict)
        else ""
    )
    if not manifest_key:
        manifest_key = f"{output_prefix.rstrip('/')}/_jobs/{sanitize_s3_key_component(config['run_id'])}/source_manifest.jsonl"
    rows = _load_jsonl_from_s3(bucket, manifest_key)
    chapters: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("row_type") or "") != "manhwa_page_pair":
            continue
        chapter_key = str(row.get("chapter_key") or "unknown")
        chapter = chapters.setdefault(chapter_key, {"rows": [], "pages": {}})
        chapter["rows"].append(row)
        for side in ("left", "right"):
            page = row.get(side) if isinstance(row.get(side), dict) else {}
            relative_path = str(page.get("relative_path") or "")
            if relative_path:
                chapter["pages"].setdefault(
                    relative_path,
                    {
                        "relative_path": relative_path,
                        "source_key": str(page.get("Key") or ""),
                        "size": _object_size(page),
                        "etag": _object_etag(page),
                        "last_modified": _object_last_modified(page),
                    },
                )

    written: list[dict[str, Any]] = []
    manhwa_config = config.get("manhwa") if isinstance(config.get("manhwa"), dict) else {}
    threshold = float(manhwa_config.get("chain_confidence_threshold") or MANHWA_CHAIN_CONFIDENCE_THRESHOLD)
    for chapter_key in sorted(chapters, key=_natural_sort_key):
        chapter = chapters[chapter_key]
        pages = sorted(chapter["pages"].values(), key=lambda item: _natural_sort_key(item["relative_path"]))
        page_index_by_relative = {str(page["relative_path"]): index for index, page in enumerate(pages)}
        page_votes: dict[str, list[dict[str, Any]]] = {str(page["relative_path"]): [] for page in pages}
        page_dimensions: dict[str, dict[str, int]] = {str(page["relative_path"]): {} for page in pages}
        edges: list[dict[str, Any]] = []
        connected_edges: set[int] = set()
        errors = 0
        missing = 0
        candidate_continuation_edges = 0
        low_confidence_continuation_edges = 0
        for row in sorted(chapter["rows"], key=lambda item: int(item.get("pair_index") or 0)):
            pair_index = int(row.get("pair_index") or 0)
            status_key = str(row.get("status_key") or _manhwa_pair_status_key(str(config["status_prefix"]), chapter_key, pair_index))
            try:
                status_payload = get_s3_json(bucket, status_key)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code") or "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    missing += 1
                    edges.append({"left": pair_index, "right": pair_index + 1, "status": "missing", "status_key": status_key})
                    continue
                raise
            if str(status_payload.get("status") or "") != "ok":
                errors += 1
                edges.append(
                    {
                        "left": pair_index,
                        "right": pair_index + 1,
                        "status": str(status_payload.get("status") or "error"),
                        "status_key": status_key,
                        "error": status_payload.get("error"),
                    }
                )
                continue
            left = status_payload.get("left") if isinstance(status_payload.get("left"), dict) else {}
            right = status_payload.get("right") if isinstance(status_payload.get("right"), dict) else {}
            left_source = left.get("source") if isinstance(left.get("source"), dict) else {}
            right_source = right.get("source") if isinstance(right.get("source"), dict) else {}
            left_rel = str(left_source.get("relative_path") or "")
            right_rel = str(right_source.get("relative_path") or "")
            if left_rel in page_votes:
                page_votes[left_rel].append(
                    {
                        "page_type": str(left.get("page_type") or "uncertain"),
                        "confidence": float(left.get("confidence") or 0.0),
                        "pair_index": pair_index,
                        "side": "left",
                    }
                )
                page_dimensions[left_rel] = {
                    "width": int(left.get("width") or 0),
                    "height": int(left.get("height") or 0),
                }
            if right_rel in page_votes:
                page_votes[right_rel].append(
                    {
                        "page_type": str(right.get("page_type") or "uncertain"),
                        "confidence": float(right.get("confidence") or 0.0),
                        "pair_index": pair_index,
                        "side": "right",
                    }
                )
                page_dimensions[right_rel] = {
                    "width": int(right.get("width") or 0),
                    "height": int(right.get("height") or 0),
                }
            edge = status_payload.get("edge") if isinstance(status_payload.get("edge"), dict) else {}
            confidence = float(edge.get("confidence") or 0.0)
            left_index = page_index_by_relative.get(left_rel, int(status_payload.get("pair_index") or 0))
            right_index = page_index_by_relative.get(right_rel, int(left_index) + 1)
            left_is_story = bool(left.get("is_story_page")) and str(left.get("page_type") or "") == "story"
            right_is_story = bool(right.get("is_story_page")) and str(right.get("page_type") or "") == "story"
            raw_continues_same_panel = bool(edge.get("continues_same_panel"))
            if raw_continues_same_panel and left_is_story and right_is_story:
                candidate_continuation_edges += 1
            is_chain_edge = raw_continues_same_panel and confidence >= threshold and left_is_story and right_is_story
            if raw_continues_same_panel and left_is_story and right_is_story and not is_chain_edge:
                low_confidence_continuation_edges += 1
            if is_chain_edge:
                connected_edges.add(int(left_index))
            edges.append(
                {
                    "left": int(left_index),
                    "right": int(right_index),
                    "status": "ok",
                    "left_relative_path": left_rel,
                    "right_relative_path": right_rel,
                    "left_page_type": str(left.get("page_type") or "uncertain"),
                    "right_page_type": str(right.get("page_type") or "uncertain"),
                    "left_is_story_page": left_is_story,
                    "right_is_story_page": right_is_story,
                    "continues_same_panel": is_chain_edge,
                    "chain_break": not is_chain_edge,
                    "raw_continues_same_panel": raw_continues_same_panel,
                    "raw_chain_break": bool(edge.get("chain_break", not raw_continues_same_panel)),
                    "confidence": confidence,
                    "reason": str(edge.get("reason") or ""),
                    "status_key": status_key,
                }
            )

        manifest_pages: list[dict[str, Any]] = []
        story_count = 0
        for index, page in enumerate(pages):
            relative_path = str(page["relative_path"])
            page_type, page_confidence = _select_page_vote(page_votes.get(relative_path, []))
            is_story = page_type == "story"
            story_count += int(is_story)
            output_key = f"{output_prefix.rstrip('/')}/{relative_path}" if is_story else ""
            dimensions = page_dimensions.get(relative_path) or {}
            manifest_pages.append(
                {
                    "page_index": index,
                    "page_key": join_s3_uri(bucket, str(page["source_key"])),
                    "relative_path": relative_path,
                    "source_key": page["source_key"],
                    "width": int(dimensions.get("width") or 0),
                    "height": int(dimensions.get("height") or 0),
                    "page_type": page_type,
                    "is_story_page": is_story,
                    "confidence": page_confidence,
                    "output_key": output_key,
                    "votes": page_votes.get(str(page["relative_path"]), []),
                }
            )
        story_page_indices = {int(page["page_index"]) for page in manifest_pages if bool(page["is_story_page"])}
        chains = [
            {
                "chain_id": f"chain_{chain_index:04d}",
                "page_indices": component,
                "relative_paths": [str(manifest_pages[index]["relative_path"]) for index in component],
                "length": len(component),
                "is_multi_page_chain": len(component) > 1,
            }
            for chain_index, component in enumerate(_build_chain_components(story_page_indices, connected_edges))
        ]
        chain_lengths = [int(chain["length"]) for chain in chains]
        chain_length_distribution: dict[str, int] = {}
        for length in chain_lengths:
            key = str(length)
            chain_length_distribution[key] = chain_length_distribution.get(key, 0) + 1
        summary_counts = {
            "total_pages": len(manifest_pages),
            "story_pages": story_count,
            "non_story_pages": max(0, len(manifest_pages) - story_count),
            "chain_count": len(chains),
            "chain_length_distribution": chain_length_distribution,
            "single_story_pages": sum(1 for length in chain_lengths if length == 1),
            "multi_page_chains": sum(1 for length in chain_lengths if length > 1),
            "two_page_chains": sum(1 for length in chain_lengths if length == 2),
            "three_page_chains": sum(1 for length in chain_lengths if length == 3),
            "four_plus_page_chains": sum(1 for length in chain_lengths if length >= 4),
            "chained_pages": sum(length for length in chain_lengths if length > 1),
            "accepted_chain_edges": len(connected_edges),
            "candidate_continuation_edges": candidate_continuation_edges,
            "low_confidence_continuation_edges": low_confidence_continuation_edges,
            "missing_pair_statuses": missing,
            "error_pair_statuses": errors,
        }
        chapter_manifest = {
            "schema_version": 1,
            "filter_mode": "manhwa",
            "status": "ok" if errors == 0 and missing == 0 else "incomplete",
            "run_id": str(config.get("run_id") or ""),
            "created_at": now_utc_iso(),
            "chapter_key": chapter_key,
            "source_prefix": str(config.get("input_prefix") or ""),
            "output_prefix": output_prefix,
            "chain_confidence_threshold": threshold,
            "pages": manifest_pages,
            "edges": edges,
            "chains": chains,
            "summary": summary_counts,
            "counts": summary_counts,
        }
        manifest_output_key = _chapter_manifest_key(output_prefix, chapter_key)
        put_s3_json(bucket, manifest_output_key, chapter_manifest)
        written.append(
            {
                "chapter_key": chapter_key,
                "manifest_key": manifest_output_key,
                "page_count": len(manifest_pages),
                "chain_count": len(chains),
                "status": chapter_manifest["status"],
            }
        )

    summary = {
        "schema_version": 1,
        "mode": "manhwa",
        "status": "ok",
        "run_id": str(config.get("run_id") or ""),
        "created_at": now_utc_iso(),
        "chapter_manifest_count": len(written),
        "chapter_manifests": written,
    }
    summary_key = f"{output_prefix.rstrip('/')}/_jobs/{sanitize_s3_key_component(config['run_id'])}/manhwa_manifest_summary.json"
    put_s3_json(bucket, summary_key, summary)
    return {**summary, "summary_key": summary_key}
