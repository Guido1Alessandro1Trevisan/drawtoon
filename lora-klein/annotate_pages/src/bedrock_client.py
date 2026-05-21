"""Bedrock Runtime client + converse-with-tool helper.

Lifted from workflows/manga_filter/src/handlers.py with the unused bits trimmed.
Uses cross-region inference profile `global.anthropic.claude-haiku-4-5-20251001-v1:0`
by default; can be overridden via the DEFAULT_ANNOTATION_MODEL env var.
"""

from __future__ import annotations

import io
import math
import os
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_ANNOTATION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "8000"))


_RUNTIME_CLIENTS: dict[int, Any] = {}


def bedrock_runtime(timeout_seconds: float = 900.0):
    read_timeout = max(1, min(900, int(math.ceil(float(timeout_seconds or 900.0)))))
    client = _RUNTIME_CLIENTS.get(read_timeout)
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
        _RUNTIME_CLIENTS[read_timeout] = client
    return client


def prepare_image_block(image_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wrap image bytes into a Bedrock content block, re-encoding if needed.

    Bedrock has a hard 3.5 MB / 8000-px-side limit per image. Re-encode to
    JPEG at decreasing quality until we fit.
    """
    fmt = "jpeg"
    try:
        if len(image_bytes) <= BEDROCK_MAX_IMAGE_BYTES:
            # Try the raw bytes first; PIL probe to confirm dims
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as probe:
                w, h = probe.size
            if w <= BEDROCK_MAX_IMAGE_SIDE and h <= BEDROCK_MAX_IMAGE_SIDE:
                return (
                    {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                    {"image_format": fmt, "image_bytes": len(image_bytes),
                     "image_width": w, "image_height": h, "reencoded": False},
                )
    except Exception:
        pass

    from PIL import Image
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        ow, oh = image.size
        scale = min(
            1.0,
            BEDROCK_MAX_IMAGE_SIDE / max(1, ow),
            BEDROCK_MAX_IMAGE_SIDE / max(1, oh),
        )
        if scale < 1.0:
            image = image.resize(
                (max(1, int(ow * scale)), max(1, int(oh * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in (92, 85, 78, 70, 62, 54, 46):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= BEDROCK_MAX_IMAGE_BYTES:
                sw, sh = image.size
                return (
                    {"image": {"format": "jpeg", "source": {"bytes": encoded}}},
                    {
                        "image_format": "jpeg",
                        "image_bytes": len(encoded),
                        "image_width": sw, "image_height": sh,
                        "source_image_bytes": len(image_bytes),
                        "source_image_width": ow, "source_image_height": oh,
                        "jpeg_quality": quality, "reencoded": True,
                    },
                )
    raise RuntimeError(
        f"image could not be encoded under {BEDROCK_MAX_IMAGE_BYTES} bytes"
    )


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, BotoCoreError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = str((exc.response.get("Error") or {}).get("Code") or "")
    http_status = int((exc.response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    return code in {
        "ThrottlingException", "TooManyRequestsException", "ModelNotReadyException",
        "ModelTimeoutException", "InternalServerException", "ServiceUnavailableException",
    } or http_status in {408, 429, 500, 502, 503, 504}


def retry_delay_seconds(exc: Exception, default_delay: float) -> float:
    if isinstance(exc, ClientError):
        code = str((exc.response.get("Error") or {}).get("Code") or "")
        message = str((exc.response.get("Error") or {}).get("Message") or "").lower()
        if code in {"ThrottlingException", "TooManyRequestsException"} and "too many tokens" in message:
            return max(float(default_delay), 12.0)
    return float(default_delay)


def extract_tool_input(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    content = ((response.get("output") or {}).get("message") or {}).get("content") or []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and str(tool_use.get("name") or "") == tool_name:
            value = tool_use.get("input")
            if isinstance(value, dict):
                return value
            raise ValueError(f"tool {tool_name!r} returned non-object input")
    raise ValueError(f"response did not contain tool input for {tool_name!r}")


def usage_of(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    inp = int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)
    out = int(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0)
    tot = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0) or (inp + out)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def converse_tool(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    image_block: dict[str, Any] | None = None,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
    max_output_tokens: int,
    timeout_seconds: float,
    client_request_id: str,
    max_retries: int = 5,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Single converse() call with tool-choice forcing. Pass image_block=None
    for text-only requests.

    Retries on throttling / 5xx with exponential backoff. Raises on
    non-retryable errors.
    """
    content: list[dict[str, Any]] = [{"text": user_text}]
    if image_block is not None:
        content.append(image_block)
    delay = 1.5
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = bedrock_runtime(timeout_seconds).converse(
                modelId=model,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": content}],
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
                    "client_request_id": (client_request_id or "annotate")[:256],
                },
            )
            return extract_tool_input(response, tool_name=tool_name), usage_of(response)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not is_retryable(exc):
                raise
            time.sleep(retry_delay_seconds(exc, delay))
            delay = min(delay * 2.0, 30.0)
    assert last_exc is not None
    raise last_exc
