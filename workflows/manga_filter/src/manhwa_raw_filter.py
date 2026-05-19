from __future__ import annotations

import io
import json
import os
import random
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DEFAULT_GEMINI_FILTER_MODEL = os.environ.get("DEFAULT_GEMINI_FILTER_MODEL", "gemini-3-flash-preview")
DEFAULT_MANHWA_RAW_PROMPT_FILENAME = "classify_manhwa_raw_page.md"
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
GEMINI_FILTER_MAX_IMAGE_SIDE = int(os.environ.get("GEMINI_FILTER_MAX_IMAGE_SIDE", "4096"))
GEMINI_FILTER_MAX_IMAGE_BYTES = int(os.environ.get("GEMINI_FILTER_MAX_IMAGE_BYTES", "8000000"))
GEMINI_FILTER_TIMEOUT_MS = int(os.environ.get("GEMINI_FILTER_TIMEOUT_MS", "180000"))
GEMINI_FILTER_RETRY_ATTEMPTS = int(os.environ.get("GEMINI_FILTER_RETRY_ATTEMPTS", "8"))
GEMINI_FILTER_RETRY_MAX_DELAY_S = float(os.environ.get("GEMINI_FILTER_RETRY_MAX_DELAY_S", "120.0"))

MANHWA_RAW_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_story_page": {
            "type": "boolean",
            "description": "True only for real sequential manhwa/webtoon/comic story content.",
        },
    },
    "required": ["is_story_page"],
    "additionalProperties": False,
}

_GENAI_CLIENT: Any = None
_GENAI_API_KEY: str | None = None


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
            response = client.get_secret_value(SecretId=GEMINI_API_KEY_SECRET_NAME)
            secret_string = response.get("SecretString")
            if secret_string:
                try:
                    parsed = json.loads(secret_string)
                    if isinstance(parsed, dict) and parsed.get(GEMINI_API_KEY_ENV):
                        _GENAI_API_KEY = str(parsed[GEMINI_API_KEY_ENV]).strip()
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
        f"Gemini API key not found. Set {GEMINI_API_KEY_ENV} env var or secret "
        f"{GEMINI_API_KEY_SECRET_NAME!r}."
    )


def _gemini_filter_client():
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package not installed; add it to the Lambda dependencies.") from exc
    _GENAI_CLIENT = genai.Client(
        api_key=_resolve_gemini_api_key(),
        http_options=types.HttpOptions(
            timeout=GEMINI_FILTER_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=GEMINI_FILTER_RETRY_ATTEMPTS,
                max_delay=GEMINI_FILTER_RETRY_MAX_DELAY_S,
            ),
        ),
    )
    return _GENAI_CLIENT


def _prepare_image_part(image_bytes: bytes, image_key: str) -> tuple[Any, dict[str, Any]]:
    from google.genai import types  # type: ignore
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        scale = min(
            1.0,
            float(GEMINI_FILTER_MAX_IMAGE_SIDE) / max(1, original_width),
            float(GEMINI_FILTER_MAX_IMAGE_SIDE) / max(1, original_height),
        )
        if scale < 1.0:
            image = image.resize(
                (max(1, int(round(original_width * scale))), max(1, int(round(original_height * scale)))),
                Image.Resampling.LANCZOS,
            )
        encoded = b""
        used_quality = 90
        for quality in (90, 84, 78, 72, 66, 60):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            used_quality = quality
            if len(encoded) <= GEMINI_FILTER_MAX_IMAGE_BYTES:
                break
        if len(encoded) > GEMINI_FILTER_MAX_IMAGE_BYTES:
            raise RuntimeError(
                f"Image {image_key} could not be encoded under Gemini image limit "
                f"({GEMINI_FILTER_MAX_IMAGE_BYTES} bytes)"
            )
        meta = {
            "image_format": "jpeg",
            "image_bytes": len(encoded),
            "image_width": int(image.width),
            "image_height": int(image.height),
            "source_image_bytes": len(image_bytes),
            "source_image_width": int(original_width),
            "source_image_height": int(original_height),
            "jpeg_quality": int(used_quality),
            "reencoded": True,
            "resized": scale < 1.0,
            "max_image_side": int(GEMINI_FILTER_MAX_IMAGE_SIDE),
        }
    return types.Part.from_bytes(data=encoded, mime_type="image/jpeg"), meta


def _usage(response: Any) -> dict[str, int]:
    usage_meta = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": int(getattr(usage_meta, "prompt_token_count", 0) or 0) if usage_meta else 0,
        "output_tokens": int(getattr(usage_meta, "candidates_token_count", 0) or 0) if usage_meta else 0,
        "reasoning_tokens": int(getattr(usage_meta, "thoughts_token_count", 0) or 0) if usage_meta else 0,
        "total_tokens": int(getattr(usage_meta, "total_token_count", 0) or 0) if usage_meta else 0,
    }


def classify_raw_page(
    *,
    image_bytes: bytes,
    image_key: str,
    prompt: str,
    model: str,
    request_metadata: dict[str, Any],
    max_output_tokens: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    from google.genai import types  # type: ignore

    image_part, image_meta = _prepare_image_part(image_bytes, image_key)
    request_text = (
        "Classify this single raw manhwa/webtoon page for the filtering stage. "
        "Return only the requested boolean JSON object.\n\n"
        f"{json.dumps(request_metadata, ensure_ascii=False, sort_keys=True, indent=2)}"
    )
    delay = 2.0
    last_error = ""
    for attempt in range(max(0, int(retries)) + 1):
        try:
            # Deliberately no thinking_config: this is the cheap keep/drop raw-page pass.
            response = _gemini_filter_client().models.generate_content(
                model=model,
                contents=[prompt, request_text, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MANHWA_RAW_PAGE_SCHEMA,
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                    temperature=0,
                    max_output_tokens=max(1, int(max_output_tokens)),
                ),
            )
            parsed = json.loads(str(response.text or "{}"))
            if not isinstance(parsed, dict) or not isinstance(parsed.get("is_story_page"), bool):
                raise ValueError("Gemini classification omitted boolean is_story_page")
            return {"is_story_page": bool(parsed["is_story_page"])}, _usage(response), image_meta
        except Exception as exc:
            last_error = str(exc)
            if attempt < max(0, int(retries)):
                time.sleep(delay + random.uniform(0.0, 1.0))
                delay = min(delay * 2.0, 30.0)
                continue
            break
    raise RuntimeError(f"raw manhwa page classification failed for {image_key}: {last_error or 'unknown error'}")
