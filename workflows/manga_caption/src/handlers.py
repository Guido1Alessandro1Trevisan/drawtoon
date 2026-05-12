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
DEFAULT_CAPTION_MODEL = os.environ.get(
    "DEFAULT_CAPTION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_PROMPT_FILENAME = "caption_manga_page_memory.md"
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANGA_METADATA_CANDIDATES = (
    "metadata/manga_metadata.json",
    "metadata/manga_credits.json",
    "manga_metadata.json",
    "manga_credits.json",
)
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_MANGA_CAPTION_MAX_CONCURRENCY", "8"))
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "8000"))
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
S3_MAX_ATTEMPTS = 10
MAX_LAMIC_CHARACTERS_PER_PANEL = 5
MAX_LAMIC_TEXT_BUBBLES_PER_PANEL = 7
ALLOWED_TEXT_BUBBLE_TYPES = {
    "Speech Bubble",
    "Thought Bubble",
    "Narration Bubble",
    "Shout Bubble",
    "Text Bubble",
    "Black Bubble",
    "Whisper Bubble",
}
MIN_PANEL_ENTITY_OVERLAP_RATIO = float(os.environ.get("MIN_PANEL_ENTITY_OVERLAP_RATIO", "0.05"))

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "chapter_summary": {"type": "string"},
        "recurring_characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "recent_state": {"type": "string"},
                },
                "required": ["memory_id", "visual_description", "recent_state"],
                "additionalProperties": False,
            },
        },
        "recurring_settings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "description": {"type": "string"},
                    "recent_state": {"type": "string"},
                },
                "required": ["memory_id", "description", "recent_state"],
                "additionalProperties": False,
            },
        },
        "important_props": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "description": {"type": "string"},
                    "recent_state": {"type": "string"},
                },
                "required": ["memory_id", "description", "recent_state"],
                "additionalProperties": False,
            },
        },
        "open_threads": {
            "type": "array",
            "items": {"type": "string"},
        },
        "last_pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["page_id", "caption"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "chapter_summary",
        "recurring_characters",
        "recurring_settings",
        "important_props",
        "open_threads",
        "last_pages",
    ],
    "additionalProperties": False,
}

LAMIC_PANEL_SCHEMA = {
    "type": "object",
    "properties": {
        "panel_index": {"type": "integer"},
        "CEI": {"type": "string"},
        "character_SADs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "character_index": {"type": "integer"},
                    "SAD": {"type": "string"},
                },
                "required": ["character_index", "SAD"],
                "additionalProperties": False,
            },
        },
        "text_bubble_SADs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_region_index": {"type": "integer"},
                    "SAD": {"type": "string"},
                },
                "required": ["text_region_index", "SAD"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["panel_index", "CEI", "character_SADs", "text_bubble_SADs"],
    "additionalProperties": False,
}

PAGE_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "page_caption": {"type": "string"},
        "lamic_panels": {
            "type": "array",
            "items": LAMIC_PANEL_SCHEMA,
        },
        "memory_update": {
            "type": "object",
            "properties": {
                "visual_events": {"type": "array", "items": {"type": "string"}},
                "character_observations": {"type": "array", "items": {"type": "string"}},
                "setting_observations": {"type": "array", "items": {"type": "string"}},
                "prop_observations": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "visual_events",
                "character_observations",
                "setting_observations",
                "prop_observations",
            ],
            "additionalProperties": False,
        },
        "updated_memory": MEMORY_SCHEMA,
    },
    "required": ["page_caption", "lamic_panels", "memory_update", "updated_memory"],
    "additionalProperties": False,
}

_S3_CLIENT = None
_BEDROCK_RUNTIME_CLIENTS: dict[int, Any] = {}
_MANGA_METADATA_CACHE: dict[str, dict[str, Any]] = {}
_LEADING_CAPTION_STYLE_RE = re.compile(
    r"^(?:black\s+and\s+white|colored)\s+manga"
    r"(?:\.\s*(?:manga|title)\s*:\s*[^.]+\.?)?"
    r"(?:\s*(?:mangaka|author)\s*:\s*[^.]+\.?)?"
    r"\s*",
    re.IGNORECASE,
)
_LEADING_MANGA_CREDIT_RE = re.compile(
    r"^(?:(?:manga|title)\s*:\s*[^.]+\.?\s*)?"
    r"(?:(?:mangaka|author)\s*:\s*[^.]+\.?\s*)?",
    re.IGNORECASE,
)


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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    text = str(uri or "").strip()
    if not text.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = text[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def load_prompt_text(prompt_filename: str) -> str:
    prompt_path = WORKFLOW_ROOT / "prompts" / str(prompt_filename).strip()
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


def get_s3_json_or_jsonl(bucket: str, key: str) -> dict[str, Any]:
    text = get_s3_bytes(bucket, key).decode("utf-8").strip()
    if not text:
        raise ValueError(f"Empty JSON object at {join_s3_uri(bucket, key)}")
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
        raise ValueError(f"Expected JSON object/JSONL row at {join_s3_uri(bucket, key)}")
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
    for attempt in range(max(0, int(retries)) + 1):
        try:
            response = _bedrock_runtime_client(timeout_seconds).converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": "Reply with ok."}]}],
                inferenceConfig={"maxTokens": 4, "temperature": 0},
                requestMetadata={
                    "client_request_id": sanitize_s3_key_component(
                        f"drawtoon-manga-caption-preflight-{model}-{attempt}",
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


def _clean_text(value: object, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:max_chars]


def _default_manga_metadata_ref() -> str:
    explicit = str(
        os.environ.get("MANGA_METADATA_JSON")
        or os.environ.get("MANGA_CREDITS_JSON")
        or os.environ.get("MANGA_CREDIT_MAP_JSON")
        or ""
    ).strip()
    if explicit:
        return explicit
    for candidate in DEFAULT_MANGA_METADATA_CANDIDATES:
        if (WORKFLOW_ROOT / candidate).exists():
            return candidate
    return ""


def _metadata_local_path(metadata_ref: str) -> Path:
    path = Path(str(metadata_ref or "").strip())
    if path.is_absolute():
        return path
    return WORKFLOW_ROOT / path


def _strip_dataset_suffix(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"_(?:mangazero|manga109)$", "", text, flags=re.IGNORECASE)


def _metadata_lookup_key(value: object) -> str:
    text = _strip_dataset_suffix(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _metadata_key_candidates(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    candidates = [
        raw,
        _strip_dataset_suffix(raw),
        raw.replace("_", "-"),
        _strip_dataset_suffix(raw).replace("_", "-"),
        raw.replace("-", " "),
        _strip_dataset_suffix(raw).replace("-", " "),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _metadata_lookup_key(candidate)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _title_from_chapter(chapter: object) -> str:
    text = _strip_dataset_suffix(chapter)
    text = re.sub(r"[-_]+", " ", text).strip()
    if not text:
        return ""
    words = []
    for word in text.split():
        if word.isupper() or any(ch.isdigit() for ch in word):
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def _iter_manga_metadata_records(payload: object):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield "", item
        return
    if not isinstance(payload, dict):
        return
    for list_key in ("items", "mangas", "metadata", "chapters", "credits", "series"):
        rows = payload.get(list_key)
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    yield "", item
            return
    for key, value in payload.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict):
            yield str(key), value
        elif isinstance(value, str):
            yield str(key), {"mangaka": value}


def _metadata_record_credit(record_key: str, record: dict[str, Any], metadata_ref: str) -> dict[str, Any] | None:
    title = _clean_text(
        record.get("title")
        or record.get("manga_title")
        or record.get("manga_name")
        or record.get("display_title")
        or record.get("series_title")
        or record.get("canonical_title")
        or record.get("english_title")
        or record.get("original_title")
        or record.get("name")
        or record.get("manga")
        or record.get("series")
        or record_key,
        max_chars=180,
    )
    mangaka = _clean_text(
        record.get("mangaka")
        or record.get("mangaka_name")
        or record.get("credit_name")
        or record.get("author")
        or record.get("author_name")
        or record.get("artist")
        or record.get("creator")
        or record.get("manga_author"),
        max_chars=180,
    )
    if not title and not mangaka:
        return None
    return {
        "title": title,
        "mangaka": mangaka,
        "metadata_key": record_key,
        "metadata_source": metadata_ref,
    }


def load_manga_metadata_index(metadata_ref: object, *, strict: bool = False) -> dict[str, Any]:
    ref = str(metadata_ref or "").strip()
    if not ref:
        return {}
    cached = _MANGA_METADATA_CACHE.get(ref)
    if cached is not None:
        return cached
    try:
        if ref.startswith("s3://"):
            bucket, key = parse_s3_uri(ref)
            payload = json.loads(get_s3_bytes(bucket, key).decode("utf-8"))
        else:
            path = _metadata_local_path(ref)
            if not path.exists():
                if strict:
                    raise FileNotFoundError(f"Manga metadata JSON not found: {path}")
                _MANGA_METADATA_CACHE[ref] = {}
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if strict:
            raise
        _MANGA_METADATA_CACHE[ref] = {}
        return {}

    index: dict[str, Any] = {}
    for record_key, record in _iter_manga_metadata_records(payload):
        credit = _metadata_record_credit(record_key, record, ref)
        if not credit:
            continue
        keys: list[object] = [
            record_key,
            record.get("key"),
            record.get("chapter"),
            record.get("chapter_name"),
            record.get("folder"),
            record.get("target_folder"),
            record.get("slug"),
            record.get("manga_slug"),
            record.get("source_book"),
            record.get("title"),
            record.get("manga_title"),
            record.get("manga_name"),
            record.get("display_title"),
            record.get("series_title"),
            record.get("canonical_title"),
            record.get("english_title"),
            record.get("original_title"),
            record.get("name"),
            record.get("manga"),
            record.get("series"),
        ]
        for key_value in keys:
            for lookup_key in _metadata_key_candidates(key_value):
                index.setdefault(lookup_key, credit)
    _MANGA_METADATA_CACHE[ref] = index
    return index


def lookup_manga_credit(page: dict[str, Any], annotation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metadata_ref = str(config.get("manga_metadata_json") or "").strip()
    index = load_manga_metadata_index(metadata_ref)
    candidates: list[object] = [
        page.get("chapter"),
        Path(str(page.get("relative_path") or "")).parent.as_posix(),
    ]
    source = annotation.get("source") if isinstance(annotation.get("source"), dict) else {}
    manga109 = annotation.get("manga109") if isinstance(annotation.get("manga109"), dict) else {}
    candidates.extend(
        [
            annotation.get("chapter"),
            annotation.get("sample_id"),
            source.get("chapter"),
            source.get("target_folder"),
            source.get("source_book"),
            manga109.get("target_folder"),
            manga109.get("source_book"),
        ]
    )
    for candidate in candidates:
        for lookup_key in _metadata_key_candidates(candidate):
            credit = index.get(lookup_key)
            if credit:
                return {
                    "title": str(credit.get("title") or _title_from_chapter(page.get("chapter"))),
                    "mangaka": str(credit.get("mangaka") or ""),
                    "metadata_key": str(credit.get("metadata_key") or ""),
                    "metadata_source": str(credit.get("metadata_source") or ""),
                    "fallback": False,
                }
    return {
        "title": _title_from_chapter(page.get("chapter")),
        "mangaka": "",
        "metadata_key": "",
        "metadata_source": metadata_ref,
        "fallback": True,
    }


def infer_manga_rendering_label(image_bytes: bytes) -> str:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256), Image.Resampling.BILINEAR)
            pixels = list(image.getdata())
        if not pixels:
            return "Black and White Manga"
        step = max(1, len(pixels) // 20000)
        sampled = pixels[::step]
        chroma_total = 0.0
        colorful = 0
        for red, green, blue in sampled:
            max_channel = max(red, green, blue)
            min_channel = min(red, green, blue)
            chroma = max_channel - min_channel
            chroma_total += chroma
            if chroma >= 24 and max_channel >= 40:
                colorful += 1
        colorful_ratio = colorful / max(1, len(sampled))
        average_chroma = chroma_total / max(1, len(sampled))
        if colorful_ratio >= 0.03 and average_chroma >= 8.0:
            return "Colored Manga"
    except Exception:
        pass
    return "Black and White Manga"


def build_caption_prefix(rendering_label: str, manga_credit: dict[str, Any]) -> str:
    style = "Colored Manga" if str(rendering_label or "").strip() == "Colored Manga" else "Black and White Manga"
    title = _clean_text(manga_credit.get("title"), max_chars=180)
    mangaka = _clean_text(manga_credit.get("mangaka"), max_chars=180)
    parts = [f"{style}."]
    if title:
        parts.append(f"{title}.")
    if mangaka:
        parts.append(f"by {mangaka}.")
    return " ".join(parts)


def enforce_caption_prefix(caption: object, caption_prefix: str) -> str:
    prefix = _clean_text(caption_prefix, max_chars=420).rstrip()
    text = _clean_text(caption, max_chars=1200).strip()
    if not prefix:
        return text
    if text.lower().startswith(prefix.lower()):
        rest = text[len(prefix) :].strip(" -:")
        return f"{prefix} {rest}".strip() if rest else prefix
    rest = _LEADING_CAPTION_STYLE_RE.sub("", text, count=1).strip(" -:")
    rest = _LEADING_MANGA_CREDIT_RE.sub("", rest, count=1).strip(" -:")
    return f"{prefix} {rest}".strip() if rest else prefix


def empty_memory() -> dict[str, Any]:
    return {
        "chapter_summary": "",
        "recurring_characters": [],
        "recurring_settings": [],
        "important_props": [],
        "open_threads": [],
        "last_pages": [],
    }


def compact_memory(value: object) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    memory = empty_memory()
    memory["chapter_summary"] = _clean_text(source.get("chapter_summary"), max_chars=1600)

    for field, desc_key in (
        ("recurring_characters", "visual_description"),
        ("recurring_settings", "description"),
        ("important_props", "description"),
    ):
        rows = source.get(field)
        if not isinstance(rows, list):
            continue
        compacted: list[dict[str, str]] = []
        for idx, item in enumerate(rows[:12]):
            item_obj = item if isinstance(item, dict) else {}
            compacted.append(
                {
                    "memory_id": sanitize_s3_key_component(item_obj.get("memory_id") or f"{field}_{idx + 1}"),
                    desc_key: _clean_text(item_obj.get(desc_key), max_chars=240),
                    "recent_state": _clean_text(item_obj.get("recent_state"), max_chars=220),
                }
            )
        memory[field] = [item for item in compacted if item[desc_key] or item["recent_state"]]

    open_threads = source.get("open_threads")
    if isinstance(open_threads, list):
        memory["open_threads"] = [_clean_text(item, max_chars=180) for item in open_threads[:8] if _clean_text(item, max_chars=180)]

    last_pages = source.get("last_pages")
    if isinstance(last_pages, list):
        rows = []
        for item in last_pages[-4:]:
            item_obj = item if isinstance(item, dict) else {}
            page_id = sanitize_s3_key_component(item_obj.get("page_id") or "page", fallback="page")
            caption = _clean_text(item_obj.get("caption"), max_chars=360)
            if caption:
                rows.append({"page_id": page_id, "caption": caption})
        memory["last_pages"] = rows
    return memory


def compact_memory_update(value: object) -> dict[str, list[str]]:
    source = value if isinstance(value, dict) else {}
    payload: dict[str, list[str]] = {}
    for field in ("visual_events", "character_observations", "setting_observations", "prop_observations"):
        values = source.get(field)
        if not isinstance(values, list):
            values = []
        payload[field] = [_clean_text(item, max_chars=220) for item in values[:8] if _clean_text(item, max_chars=220)]
    return payload


def memory_with_last_page(memory: dict[str, Any], *, page_id: str, caption: str) -> dict[str, Any]:
    updated = compact_memory(memory)
    rows = [
        item
        for item in updated.get("last_pages", [])
        if isinstance(item, dict) and str(item.get("page_id") or "") != page_id
    ]
    rows.append({"page_id": page_id, "caption": _clean_text(caption, max_chars=360)})
    updated["last_pages"] = rows[-4:]
    return updated


def infer_audit_prefix(caption_root_prefix: str, run_id: str) -> tuple[str, str]:
    safe_run_id = sanitize_s3_key_component(run_id, fallback="run")
    return DATASET_BUCKET, f"{caption_root_prefix.rstrip('/')}/_audit/{safe_run_id}"


def page_sort_key(relative_path: str, *, side_order: str) -> tuple[Any, ...]:
    path = Path(relative_path)
    stem = path.stem
    match = re.match(r"^(\d+)(?:__side_(\d+))?$", stem)
    if not match:
        return (path.parent.as_posix(), stem)
    page_num = int(match.group(1))
    side = int(match.group(2) or 0)
    if side_order == "rtl":
        side_value = -side
    elif side_order == "ltr":
        side_value = side
    else:
        side_value = stem
    return (path.parent.as_posix(), page_num, side_value, stem)


def list_annotation_relatives(*, bucket: str, annotation_prefix: str) -> set[str]:
    relatives: set[str] = set()
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=annotation_prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key.endswith(".jsonl"):
                continue
            relative = key[len(annotation_prefix) :].lstrip("/")
            if not relative or relative == "pages.jsonl" or relative.startswith("_"):
                continue
            relatives.add(relative)
    return relatives


def build_chapter_pages(
    *,
    bucket: str,
    source_prefix: str,
    annotation_prefix: str,
    caption_root_prefix: str,
    include_chapter_regex: str,
    require_annotations: bool,
    side_order: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    annotation_relatives = (
        list_annotation_relatives(bucket=bucket, annotation_prefix=annotation_prefix)
        if require_annotations
        else set()
    )
    chapters: dict[str, list[dict[str, Any]]] = {}
    stats = {
        "source_image_count": 0,
        "included_page_count": 0,
        "skipped_missing_annotation_count": 0,
        "skipped_chapter_filter_count": 0,
    }
    paginator = _s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            source_key = str(obj.get("Key") or "")
            if not source_key or source_key.endswith("/"):
                continue
            if Path(source_key).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative_path = source_key[len(source_prefix) :].lstrip("/")
            if not relative_path or relative_path.startswith("_"):
                continue
            parts = relative_path.split("/", 1)
            if len(parts) != 2:
                continue
            chapter, page_name = parts
            stats["source_image_count"] += 1
            if include_re and not include_re.search(chapter):
                stats["skipped_chapter_filter_count"] += 1
                continue
            annotation_relative = Path(relative_path).with_suffix(".jsonl").as_posix()
            if require_annotations and annotation_relative not in annotation_relatives:
                stats["skipped_missing_annotation_count"] += 1
                continue
            page_id = Path(page_name).stem
            output_key = f"{caption_root_prefix.rstrip('/')}/{chapter}/{page_id}.json"
            chapters.setdefault(chapter, []).append(
                {
                    "chapter": chapter,
                    "page_id": page_id,
                    "relative_path": relative_path,
                    "page_key": source_key,
                    "page_s3_uri": join_s3_uri(bucket, source_key),
                    "annotation_key": f"{annotation_prefix.rstrip('/')}/{annotation_relative}",
                    "annotation_s3_uri": join_s3_uri(bucket, f"{annotation_prefix.rstrip('/')}/{annotation_relative}"),
                    "output_key": output_key,
                    "sort_key": list(page_sort_key(relative_path, side_order=side_order)),
                    "source_size": int(obj.get("Size") or 0),
                    "source_etag": str(obj.get("ETag") or "").strip('"'),
                    "source_last_modified": str(obj.get("LastModified") or ""),
                }
            )
            stats["included_page_count"] += 1

    for rows in chapters.values():
        rows.sort(key=lambda item: tuple(item.get("sort_key") or []))
        for index, row in enumerate(rows):
            row["chapter_page_index"] = index
    return chapters, stats


def prepare_manga_caption_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or event.get("dataset_bucket") or DATASET_BUCKET).strip()
    if bucket != DATASET_BUCKET:
        raise ValueError(f"This workflow is configured for bucket={DATASET_BUCKET!r}, got {bucket!r}")

    source_prefix = normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_PREFIX)
    annotation_prefix = normalize_prefix(event.get("annotation_prefix"), default=DEFAULT_ANNOTATION_PREFIX)
    output_prefix = normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    run_id = str(event.get("run_id") or "").strip() or make_run_token()
    caption_run = sanitize_s3_key_component(
        event.get("caption_run") or f"haiku45_page_memory_v1_{run_id}",
        fallback="caption-run",
    )
    caption_root_prefix = f"{output_prefix.rstrip('/')}/{caption_run}"
    job_prefix = f"{caption_root_prefix}/_jobs/{sanitize_s3_key_component(run_id)}"

    model = str(event.get("model") or DEFAULT_CAPTION_MODEL).strip()
    prompt_filename = str(event.get("prompt_filename") or DEFAULT_PROMPT_FILENAME).strip()
    prompt = load_prompt_text(prompt_filename)
    manga_metadata_json = str(
        event.get("manga_metadata_json")
        or event.get("manga_metadata_path")
        or event.get("manga_credit_map_json")
        or _default_manga_metadata_ref()
    ).strip()
    manga_metadata_index = load_manga_metadata_index(manga_metadata_json, strict=bool(manga_metadata_json))
    preflight = (
        _bedrock_model_access_probe(model=model)
        if _bedrock_preflight_enabled(event)
        else {"status": "disabled", "model": model}
    )

    side_order = str(event.get("side_order") or "rtl").strip().lower()
    if side_order not in {"rtl", "ltr", "key"}:
        raise ValueError("side_order must be one of: rtl, ltr, key")

    chapters, manifest_stats = build_chapter_pages(
        bucket=bucket,
        source_prefix=source_prefix,
        annotation_prefix=annotation_prefix,
        caption_root_prefix=caption_root_prefix,
        include_chapter_regex=str(event.get("include_chapter_regex") or "").strip(),
        require_annotations=bool(event.get("require_annotations", True)),
        side_order=side_order,
    )
    chapter_manifest_rows: list[dict[str, Any]] = []
    for chapter in sorted(chapters):
        pages = chapters[chapter]
        pages_key = f"{job_prefix}/chapters/{sanitize_s3_key_component(chapter)}/pages.json"
        put_s3_json(bucket, pages_key, {"chapter": chapter, "page_count": len(pages), "pages": pages})
        chapter_manifest_rows.append(
            {
                "chapter": chapter,
                "pages_key": pages_key,
                "page_count": len(pages),
                "first_page_id": pages[0]["page_id"] if pages else "",
                "last_page_id": pages[-1]["page_id"] if pages else "",
            }
        )

    chapter_manifest_key = f"{job_prefix}/chapter_manifest.jsonl"
    put_s3_jsonl(bucket, chapter_manifest_key, chapter_manifest_rows)
    worker_config_key = f"{job_prefix}/worker_config.json"
    worker_config = {
        "bucket": bucket,
        "source_prefix": source_prefix,
        "annotation_prefix": annotation_prefix,
        "output_prefix": output_prefix,
        "caption_run": caption_run,
        "caption_root_prefix": caption_root_prefix,
        "job_prefix": job_prefix,
        "model": model,
        "prompt_filename": prompt_filename,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "manga_metadata_json": manga_metadata_json,
        "manga_metadata_entry_count": len(manga_metadata_index),
        "timeout_seconds": float(event.get("timeout_seconds") or 180.0),
        "retries": int(event["retries"]) if event.get("retries") is not None else 3,
        "max_output_tokens": max(1, int(event.get("max_output_tokens") or 4096)),
        "overwrite": bool(event.get("overwrite", False)),
        "side_order": side_order,
        "require_annotations": bool(event.get("require_annotations", True)),
        "git_sha": str(event.get("git_sha") or "").strip(),
        "run_id": run_id,
        "prepared_at": now_utc_iso(),
    }
    put_s3_json(bucket, worker_config_key, worker_config)
    audit_bucket, audit_prefix = infer_audit_prefix(caption_root_prefix, run_id)
    return {
        "source": {
            "bucket": bucket,
            "source_prefix": source_prefix,
            "annotation_prefix": annotation_prefix,
            "chapter_manifest_key": chapter_manifest_key,
            "chapter_count": len(chapter_manifest_rows),
            **manifest_stats,
        },
        "output": {
            "bucket": bucket,
            "output_prefix": output_prefix,
            "caption_run": caption_run,
            "caption_root_prefix": caption_root_prefix,
        },
        "worker_config": {"bucket": bucket, "key": worker_config_key},
        "audit": {"bucket": audit_bucket, "prefix": audit_prefix},
        "batch": {
            "max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY)),
        },
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "run": {
            "run_id": run_id,
            "job_name": str(event.get("job_name") or "caption-manga-pages").strip(),
            "git_sha": str(event.get("git_sha") or "").strip(),
            "prepared_at": now_utc_iso(),
            "bedrock_preflight": preflight,
            "caption_run": caption_run,
            "chapter_count": len(chapter_manifest_rows),
            **manifest_stats,
        },
    }


def _memory_checkpoint_key(config: dict[str, Any], chapter: str) -> str:
    return (
        f"{str(config['caption_root_prefix']).rstrip('/')}/"
        f"_state/{sanitize_s3_key_component(chapter, fallback='chapter')}/memory.json"
    )


def _load_pages(bucket: str, pages_key: str) -> list[dict[str, Any]]:
    payload = get_s3_json(bucket, pages_key)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Chapter pages file is missing pages[]: {join_s3_uri(bucket, pages_key)}")
    return [item for item in pages if isinstance(item, dict)]


def _write_memory_checkpoint(
    *,
    bucket: str,
    config: dict[str, Any],
    chapter: str,
    state: dict[str, Any],
    memory: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "caption_run": str(config.get("caption_run") or ""),
        "run_id": str(config.get("run_id") or ""),
        "chapter": chapter,
        "page_count": int(state.get("page_count") or 0),
        "next_page_index": int(state.get("next_page_index") or 0),
        "done": bool(state.get("done", False)),
        "counts": state.get("counts") if isinstance(state.get("counts"), dict) else {},
        "memory": compact_memory(memory),
        "updated_at": now_utc_iso(),
    }
    put_s3_json(bucket, _memory_checkpoint_key(config, chapter), payload)


def initialize_manga_caption_chapter(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = dict(event.get("config_ref") or {})
    if not config_ref:
        raise ValueError("Missing config_ref")
    config = get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    chapter = str(event.get("chapter") or "").strip()
    pages_key = str(event.get("pages_key") or "").strip()
    pages = _load_pages(bucket, pages_key)
    memory = empty_memory()
    counts = {"ok": 0, "skipped_existing": 0, "error": 0}
    next_page_index = 0

    if not bool(config.get("overwrite", False)):
        while next_page_index < len(pages):
            output_key = str(pages[next_page_index].get("output_key") or "")
            if not output_key or head_s3_object(bucket, output_key) is None:
                break
            existing = get_s3_json(bucket, output_key)
            if str(existing.get("status") or "") != "ok":
                break
            memory = compact_memory(existing.get("memory_after") or memory)
            counts["skipped_existing"] += 1
            next_page_index += 1

    state = {
        "schema_version": 1,
        "row_index": int(event.get("row_index") or 0),
        "config_ref": config_ref,
        "chapter": chapter,
        "pages_key": pages_key,
        "page_count": len(pages),
        "next_page_index": next_page_index,
        "memory": memory,
        "counts": counts,
        "done": next_page_index >= len(pages),
        "started_at": now_utc_iso(),
    }
    _write_memory_checkpoint(bucket=bucket, config=config, chapter=chapter, state=state, memory=memory)
    return state


def _normalize_bbox(value: object) -> list[float] | None:
    if isinstance(value, dict):
        values = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
        if any(item is None for item in values):
            values = [value.get("left"), value.get("top"), value.get("right"), value.get("bottom")]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        values = list(value[:4])
    else:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _bbox_to_norm(bbox: list[float] | None, *, width: int, height: int) -> list[float]:
    if not bbox or width <= 0 or height <= 0:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def _bbox_size_payload(bbox: object, bbox_norm: object = None) -> dict[str, float]:
    box = _normalize_bbox(bbox)
    norm = _normalize_bbox(bbox_norm)
    width_px = height_px = area_px = 0.0
    if box:
        width_px = max(0.0, float(box[2]) - float(box[0]))
        height_px = max(0.0, float(box[3]) - float(box[1]))
        area_px = width_px * height_px
    width_norm = height_norm = area_norm = 0.0
    if norm:
        width_norm = max(0.0, float(norm[2]) - float(norm[0]))
        height_norm = max(0.0, float(norm[3]) - float(norm[1]))
        area_norm = width_norm * height_norm
    return {
        "width_px": round(width_px, 3),
        "height_px": round(height_px, 3),
        "area_px": round(area_px, 3),
        "width_norm": round(width_norm, 6),
        "height_norm": round(height_norm, 6),
        "area_norm": round(area_norm, 6),
    }


def _bbox_area(box: list[float] | None) -> float:
    if not box:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _entity_overlaps_panel(
    entity_bbox: list[float] | None,
    panel_bbox: list[float] | None,
    *,
    min_ratio: float = MIN_PANEL_ENTITY_OVERLAP_RATIO,
) -> bool:
    entity_area = _bbox_area(entity_bbox)
    if entity_area <= 0:
        return False
    return (_bbox_intersection_area(entity_bbox, panel_bbox) / entity_area) >= min_ratio


def annotation_prompt_payload(annotation: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    detections = annotation.get("detections") if isinstance(annotation.get("detections"), dict) else {}
    raw_panels = detections.get("panels") if isinstance(detections.get("panels"), list) else []
    raw_characters = detections.get("characters") if isinstance(detections.get("characters"), list) else []
    raw_texts = detections.get("texts") if isinstance(detections.get("texts"), list) else []

    character_boxes = [_normalize_bbox(item.get("bbox") if isinstance(item, dict) else None) for item in raw_characters]
    text_boxes = [_normalize_bbox(item.get("bbox") if isinstance(item, dict) else None) for item in raw_texts]
    panels: list[dict[str, Any]] = []
    for index, item in enumerate(raw_panels):
        item_obj = item if isinstance(item, dict) else {}
        bbox = _normalize_bbox(item_obj.get("bbox"))
        if not bbox:
            continue
        panels.append(
            {
                "panel_index": index,
                "panel_id": str(item_obj.get("panel_id") or f"panel_{index:03d}"),
                "bbox": [round(v, 3) for v in bbox],
                "bbox_norm": [round(v, 6) for v in _bbox_to_norm(bbox, width=width, height=height)],
                "character_count": sum(1 for box in character_boxes if _entity_overlaps_panel(box, bbox)),
                "text_region_count": sum(1 for box in text_boxes if _entity_overlaps_panel(box, bbox)),
            }
        )
        panels[-1]["size"] = _bbox_size_payload(panels[-1]["bbox"], panels[-1]["bbox_norm"])

    if not panels:
        panels.append(
            {
                "panel_index": 0,
                "panel_id": "full_page",
                "bbox": [0, 0, width, height],
                "bbox_norm": [0, 0, 1, 1],
                "size": _bbox_size_payload([0, 0, width, height], [0, 0, 1, 1]),
                "character_count": len([box for box in character_boxes if box]),
                "text_region_count": len([box for box in text_boxes if box]),
            }
        )

    characters: list[dict[str, Any]] = []
    for index, item in enumerate(raw_characters[:80]):
        item_obj = item if isinstance(item, dict) else {}
        bbox = _normalize_bbox(item_obj.get("bbox"))
        if not bbox:
            continue
        characters.append(
            {
                "character_index": index,
                "entity_id": f"character_{index:03d}",
                "source_character_id": str(item_obj.get("source_character_id") or ""),
                "bbox": [round(v, 3) for v in bbox],
                "bbox_norm": [round(v, 6) for v in _bbox_to_norm(bbox, width=width, height=height)],
            }
        )
        characters[-1]["size"] = _bbox_size_payload(characters[-1]["bbox"], characters[-1]["bbox_norm"])

    text_regions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_texts[:80]):
        item_obj = item if isinstance(item, dict) else {}
        bbox = _normalize_bbox(item_obj.get("bbox"))
        if not bbox:
            continue
        text_regions.append(
            {
                "text_region_index": index,
                "entity_id": f"text_region_{index:03d}",
                "type": simple_text_region_type(item_obj.get("box_semantics") or item_obj.get("type") or "text_region"),
                "bbox": [round(v, 3) for v in bbox],
                "bbox_norm": [round(v, 6) for v in _bbox_to_norm(bbox, width=width, height=height)],
            }
        )
        text_regions[-1]["size"] = _bbox_size_payload(text_regions[-1]["bbox"], text_regions[-1]["bbox_norm"])

    assigned_character_indexes: set[int] = set()
    assigned_text_region_indexes: set[int] = set()

    for panel in panels:
        panel_bbox = _normalize_bbox(panel.get("bbox"))
        panel_characters = [
            character
            for character in characters
            if _entity_overlaps_panel(_normalize_bbox(character.get("bbox")), panel_bbox)
        ][:MAX_LAMIC_CHARACTERS_PER_PANEL]
        panel_text_regions = [
            text_region
            for text_region in text_regions
            if _entity_overlaps_panel(_normalize_bbox(text_region.get("bbox")), panel_bbox)
        ][:MAX_LAMIC_TEXT_BUBBLES_PER_PANEL]
        assigned_character_indexes.update(int(character["character_index"]) for character in panel_characters)
        assigned_text_region_indexes.update(int(text_region["text_region_index"]) for text_region in panel_text_regions)
        panel["lamic_entities"] = {
            "characters": [
                {
                    "id": f"Character {local_index + 1}",
                    "character_index": int(character["character_index"]),
                    "source_character_id": str(character.get("source_character_id") or ""),
                    "bbox": character.get("bbox"),
                    "bbox_norm": character.get("bbox_norm"),
                    "size": character.get("size"),
                }
                for local_index, character in enumerate(panel_characters)
            ],
            "text_bubbles": [
                {
                    "id": f"Speech Bubble {local_index + 1}",
                    "text_region_index": int(text_region["text_region_index"]),
                    "type": str(text_region.get("type") or "Text Bubble"),
                    "bbox": text_region.get("bbox"),
                    "bbox_norm": text_region.get("bbox_norm"),
                    "size": text_region.get("size"),
                }
                for local_index, text_region in enumerate(panel_text_regions)
            ],
        }

    summary = annotation.get("summary") if isinstance(annotation.get("summary"), dict) else {}
    return {
        "annotation_source": str(annotation.get("annotation_source") or ""),
        "schema_name": str(annotation.get("schema_name") or ""),
        "sample_id": str(annotation.get("sample_id") or ""),
        "summary": {
            "panel_count": int(summary.get("panel_count") or len(panels)),
            "character_count": int(summary.get("character_count") or len(characters)),
            "text_count": int(summary.get("text_count") or len(text_regions)),
        },
        "page_size": {"width_px": int(width), "height_px": int(height)},
        "validation": {
            "panel_entity_min_overlap_ratio": MIN_PANEL_ENTITY_OVERLAP_RATIO,
            "unassigned_character_count": max(0, len(characters) - len(assigned_character_indexes)),
            "unassigned_text_region_count": max(0, len(text_regions) - len(assigned_text_region_indexes)),
        },
        "panels": panels,
        "characters": characters,
        "text_regions": text_regions,
    }


def normalize_page_caption(value: object) -> str:
    text = _clean_text(value, max_chars=1200)
    text = re.sub(r"^(?:black[- ]and[- ]white\s+)?manga page\.?\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip(" -:")


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fallback_text_bubble_sad(entity_id: str, box_semantics: str) -> str:
    kind = simple_text_region_type(box_semantics)
    return f"{entity_id}, {kind}."


def simple_text_region_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if "black" in text or "inverted" in text or "white text" in text:
        value = "Black Bubble"
    elif "whisper" in text or "small voice" in text:
        value = "Whisper Bubble"
    elif "narration" in text or "caption" in text:
        value = "Narration Bubble"
    elif "thought" in text:
        value = "Thought Bubble"
    elif "shout" in text or "jagged" in text:
        value = "Shout Bubble"
    elif "written" in text or "text" in text or "sfx" in text or "sound" in text:
        value = "Text Bubble"
    else:
        value = "Speech Bubble"
    if value not in ALLOWED_TEXT_BUBBLE_TYPES:
        return "Speech Bubble"
    return value


def _normalize_character_sad(value: object, entity_id: str) -> str:
    text = _clean_text(value, max_chars=300).strip(" ;")
    text = re.sub(rf"^\s*{re.escape(entity_id)}\s*,?\s*", "", text, flags=re.IGNORECASE).strip(" ;")
    if not text:
        text = "preserve appearance; visible pose and expression"
    if not text.lower().startswith("preserve appearance"):
        text = f"preserve appearance; {text[0].lower() + text[1:] if text else text}"
    return text.rstrip(".") + "."


def _normalize_text_bubble_sad(value: object, entity_id: str, box_semantics: str) -> str:
    return f"{simple_text_region_type(box_semantics)}."


def _model_lamic_by_panel(parsed: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    rows = parsed.get("lamic_panels")
    if not isinstance(rows, list):
        return result
    for item in rows:
        if not isinstance(item, dict):
            continue
        panel_index = _safe_int(item.get("panel_index"))
        if panel_index >= 0:
            result[panel_index] = item
    return result


def normalize_lamic_payload(parsed: dict[str, Any], annotation_payload: dict[str, Any]) -> dict[str, Any]:
    model_by_panel = _model_lamic_by_panel(parsed)
    panels: list[dict[str, Any]] = []
    for panel in annotation_payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        panel_index = _safe_int(panel.get("panel_index"))
        model_panel = model_by_panel.get(panel_index, {})
        entities = panel.get("lamic_entities") if isinstance(panel.get("lamic_entities"), dict) else {}

        raw_character_sads = model_panel.get("character_SADs") if isinstance(model_panel, dict) else []
        if not isinstance(raw_character_sads, list):
            raw_character_sads = []
        raw_text_sads = model_panel.get("text_bubble_SADs") if isinstance(model_panel, dict) else []
        if not isinstance(raw_text_sads, list):
            raw_text_sads = []

        character_sad_by_index = {
            _safe_int(item.get("character_index")): item
            for item in raw_character_sads
            if isinstance(item, dict) and _safe_int(item.get("character_index")) >= 0
        }
        text_sad_by_index = {
            _safe_int(item.get("text_region_index")): item
            for item in raw_text_sads
            if isinstance(item, dict) and _safe_int(item.get("text_region_index")) >= 0
        }

        characters: list[dict[str, Any]] = []
        for entity in list(entities.get("characters") or [])[:MAX_LAMIC_CHARACTERS_PER_PANEL]:
            if not isinstance(entity, dict):
                continue
            character_index = _safe_int(entity.get("character_index"))
            entity_id = str(entity.get("id") or f"Character {len(characters) + 1}")
            raw_sad = character_sad_by_index.get(character_index, {}).get("SAD")
            characters.append(
                {
                    "id": entity_id,
                    "character_index": character_index,
                    "source_character_id": str(entity.get("source_character_id") or ""),
                    "SAD": _normalize_character_sad(raw_sad, entity_id),
                    "bbox": entity.get("bbox"),
                    "bbox_norm": entity.get("bbox_norm"),
                    "size": entity.get("size") or _bbox_size_payload(entity.get("bbox"), entity.get("bbox_norm")),
                }
            )

        text_bubbles: list[dict[str, Any]] = []
        for entity in list(entities.get("text_bubbles") or [])[:MAX_LAMIC_TEXT_BUBBLES_PER_PANEL]:
            if not isinstance(entity, dict):
                continue
            text_region_index = _safe_int(entity.get("text_region_index"))
            entity_id = str(entity.get("id") or f"Speech Bubble {len(text_bubbles) + 1}")
            bubble_type = str(entity.get("type") or "Text Bubble")
            raw_sad = text_sad_by_index.get(text_region_index, {}).get("SAD")
            text_bubbles.append(
                {
                    "id": entity_id,
                    "text_region_index": text_region_index,
                    "type": simple_text_region_type(bubble_type),
                    "SAD": _normalize_text_bubble_sad(raw_sad, entity_id, bubble_type),
                    "bbox": entity.get("bbox"),
                    "bbox_norm": entity.get("bbox_norm"),
                    "size": entity.get("size") or _bbox_size_payload(entity.get("bbox"), entity.get("bbox_norm")),
                }
            )

        cei = _clean_text(model_panel.get("CEI") if isinstance(model_panel, dict) else "", max_chars=900)
        if not cei:
            cei = "Create one manga panel using the supplied character and speech-bubble entities."
        panels.append(
            {
                "panel_index": panel_index,
                "panel_id": str(panel.get("panel_id") or f"panel_{panel_index:03d}"),
                "bbox": panel.get("bbox"),
                "bbox_norm": panel.get("bbox_norm"),
                "size": panel.get("size") or _bbox_size_payload(panel.get("bbox"), panel.get("bbox_norm")),
                "CEI": cei.rstrip(".") + ".",
                "characters": characters,
                "text_bubbles": text_bubbles,
            }
        )

    return {
        "schema_version": 1,
        "coord_space": "page_pixel_bbox_and_bbox_norm_0_1",
        "max_characters_per_panel": MAX_LAMIC_CHARACTERS_PER_PANEL,
        "max_text_bubbles_per_panel": MAX_LAMIC_TEXT_BUBBLES_PER_PANEL,
        "allowed_text_bubble_types": sorted(ALLOWED_TEXT_BUBBLE_TYPES),
        "validation": annotation_payload.get("validation") if isinstance(annotation_payload.get("validation"), dict) else {},
        "panels": panels,
    }


def run_page_caption_model(
    *,
    row_index: int,
    page: dict[str, Any],
    image_block: dict[str, Any],
    image_meta: dict[str, Any],
    annotation_payload: dict[str, Any],
    memory: dict[str, Any],
    caption_prefix: str,
    manga_credit: dict[str, Any],
    rendering_label: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt = load_prompt_text(str(config["prompt_filename"]))
    request_payload = {
        "chapter": str(page.get("chapter") or ""),
        "page_id": str(page.get("page_id") or ""),
        "page_index_in_chapter": int(page.get("chapter_page_index") or 0),
        "source_stage": "filtered",
        "caption_prefix": caption_prefix,
        "manga_rendering": rendering_label,
        "manga_credit": manga_credit,
        "image": {
            "width": int(image_meta.get("image_width") or 0),
            "height": int(image_meta.get("image_height") or 0),
            "format": str(image_meta.get("image_format") or ""),
        },
        "annotation": annotation_payload,
        "prior_memory": compact_memory(memory),
    }
    request_text = (
        "Caption this full manga page as one page-level training caption. "
        "Start page_caption exactly with caption_prefix. "
        "Also return LAMIC panel data: one CEI per panel plus SADs for the supplied panel-local "
        "characters and speech/text bubbles. Use the prior_memory only for conservative visual "
        "continuity, then return updated_memory for the next page in the same chapter.\n\n"
        f"{_json_dumps(request_payload, pretty=True)}"
    )
    retries = int(config["retries"]) if config.get("retries") is not None else 3
    delay = 2.0
    last_error = ""
    for attempt in range(retries + 1):
        try:
            parsed, usage = bedrock_converse_tool(
                model=str(config["model"]),
                system_prompt=prompt,
                user_text=request_text,
                image_block=image_block,
                tool_name="manga_page_memory_caption",
                tool_description=(
                    "Return one page-level manga training caption, panel-local LAMIC CEI/SAD fields, "
                    "and a compact updated chapter memory for sequential captioning."
                ),
                tool_schema=PAGE_CAPTION_SCHEMA,
                max_output_tokens=max(1, int(config.get("max_output_tokens") or 4096)),
                timeout_seconds=float(config.get("timeout_seconds") or 180.0),
                client_request_id=(
                    f"drawtoon-manga-caption-{page.get('chapter')}-"
                    f"{page.get('page_id')}-{int(row_index)}-{attempt}"
                ),
            )
            if not isinstance(parsed, dict):
                raise ValueError("Bedrock structured output was not a JSON object")
            caption = enforce_caption_prefix(normalize_page_caption(parsed.get("page_caption")), caption_prefix)
            if not caption:
                raise ValueError("Bedrock caption output was empty")
            updated_memory = memory_with_last_page(
                compact_memory(parsed.get("updated_memory")),
                page_id=str(page.get("page_id") or ""),
                caption=caption,
            )
            return (
                {
                    "caption": caption,
                    "lamic": normalize_lamic_payload(parsed, annotation_payload),
                    "memory_update": compact_memory_update(parsed.get("memory_update")),
                    "memory_after": updated_memory,
                },
                usage,
            )
        except Exception as exc:
            last_error = str(exc)
            retryable = _bedrock_error_is_retryable(exc) or isinstance(exc, json.JSONDecodeError)
            if retryable and attempt < retries:
                sleep_seconds = _bedrock_retry_delay_seconds(exc, delay)
                time.sleep(sleep_seconds + random.uniform(0.0, 1.0))
                delay = min(delay * 2.0, 30.0)
                continue
            break
    raise RuntimeError(f"caption failed for {page.get('page_key')}: {last_error or 'unknown error'}")


def caption_manga_page(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = dict(event.get("config_ref") or {})
    if not config_ref:
        raise ValueError("Missing config_ref")
    config = get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    chapter = str(event.get("chapter") or "").strip()
    pages_key = str(event.get("pages_key") or "").strip()
    pages = _load_pages(bucket, pages_key)
    next_page_index = int(event.get("next_page_index") or 0)
    counts = event.get("counts") if isinstance(event.get("counts"), dict) else {}
    counts = {
        "ok": int(counts.get("ok") or 0),
        "skipped_existing": int(counts.get("skipped_existing") or 0),
        "error": int(counts.get("error") or 0),
    }
    memory_before = compact_memory(event.get("memory"))

    if next_page_index >= len(pages):
        state = {
            **event,
            "page_count": len(pages),
            "next_page_index": next_page_index,
            "counts": counts,
            "memory": memory_before,
            "done": True,
            "finished_at": now_utc_iso(),
        }
        _write_memory_checkpoint(bucket=bucket, config=config, chapter=chapter, state=state, memory=memory_before)
        return state

    page = pages[next_page_index]
    output_key = str(page.get("output_key") or "")
    if not output_key:
        raise ValueError(f"Page index={next_page_index} is missing output_key")

    if not bool(config.get("overwrite", False)) and head_s3_object(bucket, output_key) is not None:
        existing = get_s3_json(bucket, output_key)
        if str(existing.get("status") or "") == "ok":
            memory_after = compact_memory(existing.get("memory_after") or memory_before)
            counts["skipped_existing"] += 1
            state = {
                **event,
                "page_count": len(pages),
                "next_page_index": next_page_index + 1,
                "counts": counts,
                "memory": memory_after,
                "done": next_page_index + 1 >= len(pages),
                "updated_at": now_utc_iso(),
            }
            _write_memory_checkpoint(bucket=bucket, config=config, chapter=chapter, state=state, memory=memory_after)
            return state

    image_meta: dict[str, Any] = {}
    manga_credit: dict[str, Any] = {}
    rendering_label = ""
    caption_prefix = ""
    try:
        image_bytes = get_s3_bytes(bucket, str(page["page_key"]))
        width, height = _page_image_dimensions(image_bytes)
        image_block, image_meta = _prepare_bedrock_image_block(image_bytes, str(page["page_key"]))
        if not int(image_meta.get("image_width") or 0):
            image_meta["image_width"] = width
        if not int(image_meta.get("image_height") or 0):
            image_meta["image_height"] = height
        annotation = get_s3_json_or_jsonl(bucket, str(page["annotation_key"]))
        annotation_payload = annotation_prompt_payload(annotation, width=width, height=height)
        rendering_label = infer_manga_rendering_label(image_bytes)
        manga_credit = lookup_manga_credit(page, annotation, config)
        caption_prefix = build_caption_prefix(rendering_label, manga_credit)
        model_payload, usage = run_page_caption_model(
            row_index=next_page_index,
            page=page,
            image_block=image_block,
            image_meta=image_meta,
            annotation_payload=annotation_payload,
            memory=memory_before,
            caption_prefix=caption_prefix,
            manga_credit=manga_credit,
            rendering_label=rendering_label,
            config=config,
        )
        memory_after = compact_memory(model_payload["memory_after"])
        payload = {
            "schema_version": 1,
            "caption_type": "page_memory_caption",
            "caption_run": str(config.get("caption_run") or ""),
            "run_id": str(config.get("run_id") or ""),
            "status": "ok",
            "chapter": chapter,
            "page_id": str(page.get("page_id") or ""),
            "page_index_in_chapter": next_page_index,
            "page_count": len(pages),
            "model": str(config["model"]),
            "prompt_filename": str(config["prompt_filename"]),
            "prompt_sha256": str(config.get("prompt_sha256") or ""),
            "source_stage": "filtered",
            "caption_prefix": caption_prefix,
            "manga_rendering": rendering_label,
            "manga_credit": manga_credit,
            "page_size": {"width_px": int(width), "height_px": int(height)},
            "sources": {
                "page": str(page.get("page_s3_uri") or ""),
                "annotation": str(page.get("annotation_s3_uri") or ""),
                "annotation_type": "magi_v3",
                "bucket": bucket,
                "page_key": str(page.get("page_key") or ""),
                "annotation_key": str(page.get("annotation_key") or ""),
                "relative_path": str(page.get("relative_path") or ""),
                "source_size": int(page.get("source_size") or 0),
                "source_etag": str(page.get("source_etag") or ""),
                "source_last_modified": str(page.get("source_last_modified") or ""),
            },
            "image": image_meta,
            "annotation_summary": annotation_payload.get("summary") or {},
            "caption": str(model_payload["caption"]),
            "lamic": model_payload["lamic"],
            "memory_before": memory_before,
            "memory_update": model_payload["memory_update"],
            "memory_after": memory_after,
            "usage": usage,
            "created_at": now_utc_iso(),
        }
        put_s3_json(bucket, output_key, payload)
        counts["ok"] += 1
        state = {
            **event,
            "page_count": len(pages),
            "next_page_index": next_page_index + 1,
            "counts": counts,
            "memory": memory_after,
            "done": next_page_index + 1 >= len(pages),
            "updated_at": now_utc_iso(),
        }
        _write_memory_checkpoint(bucket=bucket, config=config, chapter=chapter, state=state, memory=memory_after)
        return state
    except Exception as exc:
        counts["error"] += 1
        error_payload = {
            "schema_version": 1,
            "caption_type": "page_memory_caption",
            "caption_run": str(config.get("caption_run") or ""),
            "run_id": str(config.get("run_id") or ""),
            "status": "error",
            "chapter": chapter,
            "page_id": str(page.get("page_id") or ""),
            "page_index_in_chapter": next_page_index,
            "page_count": len(pages),
            "model": str(config.get("model") or ""),
            "prompt_filename": str(config.get("prompt_filename") or ""),
            "prompt_sha256": str(config.get("prompt_sha256") or ""),
            "source_stage": "filtered",
            "caption_prefix": caption_prefix,
            "manga_rendering": rendering_label,
            "manga_credit": manga_credit,
            "page_size": {
                "width_px": int(image_meta.get("source_image_width") or image_meta.get("image_width") or 0),
                "height_px": int(image_meta.get("source_image_height") or image_meta.get("image_height") or 0),
            },
            "sources": {
                "page": str(page.get("page_s3_uri") or ""),
                "annotation": str(page.get("annotation_s3_uri") or ""),
                "annotation_type": "magi_v3",
                "bucket": bucket,
                "page_key": str(page.get("page_key") or ""),
                "annotation_key": str(page.get("annotation_key") or ""),
                "relative_path": str(page.get("relative_path") or ""),
            },
            "image": image_meta,
            "caption": "",
            "memory_before": memory_before,
            "memory_after": memory_before,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "error": str(exc),
            "created_at": now_utc_iso(),
        }
        put_s3_json(bucket, output_key, error_payload)
        raise
