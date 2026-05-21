"""Shared Gemini helpers.

Holds the Gemini API key resolver and related constants that multiple
workflows + scripts need. Previously colocated in manhwa_raw_filter.py.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import boto3
from botocore.exceptions import ClientError

DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"

_GENAI_API_KEY: str | None = None
_GENAI_API_KEY_LOCK = threading.Lock()


def _resolve_gemini_api_key() -> str:
    global _GENAI_API_KEY
    if _GENAI_API_KEY:
        return _GENAI_API_KEY
    with _GENAI_API_KEY_LOCK:
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
