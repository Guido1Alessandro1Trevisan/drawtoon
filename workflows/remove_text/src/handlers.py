from __future__ import annotations

import datetime as dt
import hashlib
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_INPUT_PREFIX = "datasets/pages/filtered"
DEFAULT_OUTPUT_PREFIX = "datasets/pages/text_removed"
DEFAULT_REMOVE_TEXT_RUN = "qwen2511_master_prompt_v1"
DEFAULT_FAL_ENDPOINT = os.environ.get("DEFAULT_FAL_ENDPOINT", "fal-ai/qwen-image-edit-2511")
DEFAULT_PROMPT_FILENAME = "master_prompt.md"
QUEUE_ROOT = "https://queue.fal.run"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
S3_MAX_ATTEMPTS = 10

_S3_CLIENT = None
_SECRETS_CLIENT = None
_FAL_KEY_CACHE = None


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


def _secrets_client():
    global _SECRETS_CLIENT
    if _SECRETS_CLIENT is None:
        _SECRETS_CLIENT = boto3.client(
            "secretsmanager",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "standard", "max_attempts": 4},
                connect_timeout=10,
                read_timeout=30,
            ),
        )
    return _SECRETS_CLIENT


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_token() -> str:
    return f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000, 9999)}"


def _sanitize_key_component(value: object, *, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value or "").strip()).strip("_")
    return normalized[:240] or fallback


def _normalize_prefix(value: object, *, default: str) -> str:
    prefix = str(value or default).strip().strip("/")
    if not prefix:
        raise ValueError("S3 prefix must not be empty")
    return prefix


def _join_key(*parts: object) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _load_prompt_text(prompt_filename: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / str(prompt_filename).strip()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def _get_s3_json(bucket: str, key: str) -> dict[str, Any]:
    payload = json.loads(_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {_join_s3_uri(bucket, key)}")
    return payload


def _put_s3_json(bucket: str, key: str, payload: object, *, pretty: bool = True) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=_json_dumps(payload, pretty=pretty).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def _put_s3_jsonl(bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    body = "".join(_json_dumps(row) + "\n" for row in rows).encode("utf-8")
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/x-ndjson; charset=utf-8",
    )


def _put_s3_bytes(bucket: str, key: str, body: bytes, *, content_type: str, metadata: dict[str, str]) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={k: v[:2048] for k, v in metadata.items() if v},
    )


def _head_s3_object(bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return _s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _content_type_for_key(key: str, *, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(key)
    return guessed or fallback


def _output_relative_path(relative_path: str, output_format: str) -> str:
    suffix = "." + str(output_format or "png").strip().lower().lstrip(".")
    return Path(relative_path).with_suffix(suffix).as_posix()


def _list_existing_output_relatives(*, bucket: str, output_root: str, output_format: str) -> set[str]:
    prefix = output_root.rstrip("/") + "/"
    suffix = "." + output_format.strip(".").lower()
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if "/_jobs/" in key or "/_audit/" in key or "/_status/" in key:
                continue
            if key.lower().endswith(suffix):
                relatives.add(key[len(prefix) :])
    return relatives


def _fal_key() -> str:
    global _FAL_KEY_CACHE
    if _FAL_KEY_CACHE:
        return str(_FAL_KEY_CACHE)
    env_key = os.environ.get("FAL_KEY", "").strip()
    if env_key:
        _FAL_KEY_CACHE = env_key
        return env_key
    secret_name = os.environ.get("FAL_SECRET_NAME", "").strip()
    if not secret_name:
        raise RuntimeError("FAL_KEY or FAL_SECRET_NAME is required")
    response = _secrets_client().get_secret_value(SecretId=secret_name)
    secret_string = str(response.get("SecretString") or "").strip()
    if not secret_string:
        raise RuntimeError(f"Secret {secret_name!r} has no SecretString")
    try:
        parsed = json.loads(secret_string)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("FAL_KEY", "fal_key", "key", "token"):
            value = str(parsed.get(key) or "").strip()
            if value:
                _FAL_KEY_CACHE = value
                return value
    _FAL_KEY_CACHE = secret_string
    return secret_string


def _fal_headers() -> dict[str, str]:
    return {
        "Authorization": f"Key {_fal_key()}",
        "Content-Type": "application/json",
    }


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method.upper())
    for key, value in _fal_headers().items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"FAL HTTP {exc.code} for {method.upper()} {url}: {error_body}") from exc
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"FAL returned non-object JSON for {method.upper()} {url}")
    return parsed


def _http_get_bytes(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Output download HTTP {exc.code} for {url}: {error_body}") from exc


def _fal_endpoint_url(endpoint: str) -> str:
    endpoint = str(endpoint or DEFAULT_FAL_ENDPOINT).strip().strip("/")
    if not endpoint:
        raise ValueError("fal_endpoint must not be empty")
    if endpoint.endswith("/lora") or "/lora/" in endpoint:
        raise ValueError("remove_text uses stock Qwen Image Edit 2511; do not use the /lora endpoint")
    return f"{QUEUE_ROOT}/{endpoint}"


def _presign_source_image(bucket: str, key: str, expires_seconds: int) -> str:
    return _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=max(60, int(expires_seconds)),
    )


def _build_source_manifest(
    *,
    bucket: str,
    input_prefix: str,
    output_root: str,
    output_format: str,
    include_relative_path_regex: str,
    overwrite: bool,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    include_re = re.compile(include_relative_path_regex) if include_relative_path_regex else None
    source_root = input_prefix.rstrip("/") + "/"
    existing_outputs = set() if overwrite else _list_existing_output_relatives(
        bucket=bucket,
        output_root=output_root,
        output_format=output_format,
    )
    paginator = _s3_client().get_paginator("list_objects_v2")
    rows: list[dict[str, Any]] = []
    stats = {
        "source_image_count": 0,
        "skipped_non_image_count": 0,
        "skipped_key_filter_count": 0,
        "skipped_existing_count": 0,
    }
    for page in paginator.paginate(Bucket=bucket, Prefix=source_root):
        for obj in page.get("Contents", []):
            source_key = str(obj.get("Key") or "")
            if not source_key or source_key.endswith("/"):
                continue
            if Path(source_key).suffix.lower() not in SUPPORTED_SUFFIXES:
                stats["skipped_non_image_count"] += 1
                continue
            stats["source_image_count"] += 1
            relative_path = source_key[len(source_root) :]
            if not relative_path or relative_path.startswith("_"):
                continue
            if include_re and not include_re.search(relative_path):
                stats["skipped_key_filter_count"] += 1
                continue
            output_relative = _output_relative_path(relative_path, output_format)
            if not overwrite and output_relative in existing_outputs:
                stats["skipped_existing_count"] += 1
                continue
            rows.append(
                {
                    "source_key": source_key,
                    "source_s3_uri": _join_s3_uri(bucket, source_key),
                    "relative_path": relative_path,
                    "output_relative_path": output_relative,
                    "output_key": _join_key(output_root, output_relative),
                    "status_key": _join_key(output_root, "_status", Path(relative_path).with_suffix(".json").as_posix()),
                    "source_size": int(obj.get("Size") or 0),
                    "source_etag": str(obj.get("ETag") or "").strip('"'),
                    "source_last_modified": str(obj.get("LastModified") or ""),
                }
            )
            if max_pages > 0 and len(rows) >= max_pages:
                return rows, stats
    return rows, stats


def prepare_remove_text_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or DATASET_BUCKET).strip()
    if bucket != DATASET_BUCKET:
        raise ValueError(f"This workflow is configured for bucket={DATASET_BUCKET!r}, got {bucket!r}")
    if bool(event.get("fal_preflight", True)):
        _fal_key()
    input_prefix = _normalize_prefix(event.get("input_prefix"), default=DEFAULT_INPUT_PREFIX)
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    remove_text_run = str(event.get("remove_text_run") or DEFAULT_REMOVE_TEXT_RUN).strip().strip("/")
    if not remove_text_run:
        raise ValueError("remove_text_run must not be empty")
    output_root = _join_key(output_prefix, remove_text_run)
    output_format = str(event.get("output_format") or "png").strip().lower().lstrip(".")
    if output_format != "png":
        raise ValueError("Qwen text removal workflow currently writes png output only")
    prompt_filename = str(event.get("prompt_filename") or DEFAULT_PROMPT_FILENAME).strip()
    prompt = _load_prompt_text(prompt_filename)
    fal_endpoint = str(event.get("fal_endpoint") or DEFAULT_FAL_ENDPOINT).strip().strip("/")
    _fal_endpoint_url(fal_endpoint)
    run_id = str(event.get("run_id") or _run_token()).strip()
    overwrite = bool(event.get("overwrite", False))
    include_relative_path_regex = str(event.get("include_relative_path_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))

    rows, manifest_stats = _build_source_manifest(
        bucket=bucket,
        input_prefix=input_prefix,
        output_root=output_root,
        output_format=output_format,
        include_relative_path_regex=include_relative_path_regex,
        overwrite=overwrite,
        max_pages=max_pages,
    )
    job_root = _join_key(output_root, "_jobs", run_id)
    manifest_key = _join_key(job_root, "page_manifest.jsonl")
    worker_config_key = _join_key(job_root, "worker_config.json")
    _put_s3_jsonl(bucket, manifest_key, rows)

    worker_config = {
        "bucket": bucket,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "remove_text_run": remove_text_run,
        "output_root": output_root,
        "run_id": run_id,
        "fal_endpoint": fal_endpoint,
        "prompt_filename": prompt_filename,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "timeout_seconds": float(event.get("timeout_seconds") or 840.0),
        "request_timeout_seconds": float(event.get("request_timeout_seconds") or 60.0),
        "download_timeout_seconds": float(event.get("download_timeout_seconds") or 180.0),
        "poll_interval_seconds": float(event.get("poll_interval_seconds") or 8.0),
        "presign_expires_seconds": max(60, int(event.get("presign_expires_seconds") or 12 * 60 * 60)),
        "num_inference_steps": max(1, int(event.get("num_inference_steps") or 40)),
        "guidance_scale": float(event.get("guidance_scale") or 4.5),
        "num_images": max(1, int(event.get("num_images") or 1)),
        "output_format": output_format,
        "acceleration": str(event.get("acceleration") or "regular").strip(),
        "enable_safety_checker": False,
        "use_lora": False,
        "overwrite": overwrite,
        "created_at": _now_utc_iso(),
        "git_sha": str(event.get("git_sha") or ""),
    }
    _put_s3_json(bucket, worker_config_key, worker_config, pretty=True)

    return {
        "schema_version": 1,
        "source": {"bucket": bucket, "prefix": input_prefix, "manifest_key": manifest_key},
        "worker_config": {"bucket": bucket, "key": worker_config_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or 32))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": bucket, "prefix": _join_key(output_root, "_audit", run_id) + "/"},
        "output": {"bucket": bucket, "prefix": output_root, "remove_text_run": remove_text_run, "run_id": run_id},
        "stats": {**manifest_stats, "manifest_count": len(rows)},
    }


def _remaining_seconds(context: Any, default: float) -> float:
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if callable(getter):
        return max(0.0, float(getter()) / 1000.0)
    return float(default)


def _write_status(config: dict[str, Any], row: dict[str, Any], payload: dict[str, Any]) -> None:
    bucket = str(config["bucket"])
    status_key = str(row["status_key"])
    _put_s3_json(bucket, status_key, payload, pretty=True)


def _completed_payload(
    *,
    config: dict[str, Any],
    row_index: int,
    row: dict[str, Any],
    request_payload: dict[str, Any],
    submit_response: dict[str, Any],
    result: dict[str, Any],
    output_image: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    bucket = str(config["bucket"])
    return {
        "schema_version": 1,
        "row_index": int(row_index),
        "status": "completed",
        "source": {
            "bucket": bucket,
            "key": row["source_key"],
            "s3_uri": _join_s3_uri(bucket, str(row["source_key"])),
            "relative_path": row["relative_path"],
            "size": int(row.get("source_size") or 0),
            "etag": str(row.get("source_etag") or ""),
            "last_modified": str(row.get("source_last_modified") or ""),
        },
        "output": {
            "bucket": bucket,
            "key": row["output_key"],
            "s3_uri": _join_s3_uri(bucket, str(row["output_key"])),
            "relative_path": row["output_relative_path"],
            "content_type": _content_type_for_key(str(row["output_key"]), fallback="image/png"),
            "image": output_image,
        },
        "fal": {
            "endpoint": str(config["fal_endpoint"]),
            "request_id": submit_response.get("request_id"),
            "status_url": submit_response.get("status_url"),
            "response_url": submit_response.get("response_url"),
            "result": result,
        },
        "request": {
            key: value
            for key, value in request_payload.items()
            if key not in {"image_urls"}
        },
        "prompt_filename": str(config["prompt_filename"]),
        "prompt_sha256": str(config.get("prompt_sha256") or ""),
        "use_lora": False,
        "run_id": str(config.get("run_id") or ""),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "created_at": _now_utc_iso(),
    }


def remove_text_page(event: dict[str, Any], context: Any) -> dict[str, Any]:
    started_at = time.time()
    row_index = int(event.get("row_index") or 0)
    row = event.get("page")
    if not isinstance(row, dict):
        raise ValueError("Missing page row")
    config_ref = dict(event.get("config_ref") or {})
    if not config_ref:
        raise ValueError("Missing config_ref")
    config = _get_s3_json(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    output_key = str(row["output_key"])

    if not bool(config.get("overwrite", False)) and _head_s3_object(bucket, output_key) is not None:
        return {
            "row_index": row_index,
            "status": "skipped_existing",
            "source_key": row.get("source_key"),
            "output_key": output_key,
        }

    prompt = _load_prompt_text(str(config["prompt_filename"]))
    endpoint_url = _fal_endpoint_url(str(config["fal_endpoint"]))
    request_payload = {
        "prompt": prompt,
        "image_urls": [
            _presign_source_image(
                bucket,
                str(row["source_key"]),
                int(config.get("presign_expires_seconds") or 12 * 60 * 60),
            )
        ],
        "num_inference_steps": max(1, int(config.get("num_inference_steps") or 40)),
        "guidance_scale": float(config.get("guidance_scale") or 4.5),
        "num_images": max(1, int(config.get("num_images") or 1)),
        "enable_safety_checker": False,
        "output_format": "png",
        "acceleration": str(config.get("acceleration") or "regular"),
    }
    request_timeout = float(config.get("request_timeout_seconds") or 60.0)
    download_timeout = float(config.get("download_timeout_seconds") or 180.0)
    poll_interval = max(1.0, float(config.get("poll_interval_seconds") or 8.0))
    max_runtime = max(30.0, float(config.get("timeout_seconds") or 840.0))

    submit_response: dict[str, Any] = {}
    try:
        submit_response = _http_json("POST", endpoint_url, payload=request_payload, timeout_seconds=request_timeout)
        request_id = str(submit_response.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError(f"FAL submit response omitted request_id: {submit_response}")
        status_url = str(submit_response.get("status_url") or "").strip()
        if not status_url:
            status_url = f"{endpoint_url}/requests/{urllib.parse.quote(request_id)}/status"
        response_url = str(submit_response.get("response_url") or "").strip()
        if not response_url:
            response_url = f"{endpoint_url}/requests/{urllib.parse.quote(request_id)}"

        last_status: dict[str, Any] = {}
        while True:
            remaining = min(_remaining_seconds(context, max_runtime), max_runtime - (time.time() - started_at))
            if remaining < max(20.0, request_timeout + 5.0):
                payload = {
                    "schema_version": 1,
                    "row_index": row_index,
                    "status": "pending_timeout",
                    "source": {"bucket": bucket, "key": row["source_key"], "relative_path": row["relative_path"]},
                    "output": {"bucket": bucket, "key": output_key},
                    "fal": {
                        "endpoint": str(config["fal_endpoint"]),
                        "request_id": request_id,
                        "status_url": status_url,
                        "response_url": response_url,
                        "last_status": last_status,
                    },
                    "run_id": str(config.get("run_id") or ""),
                    "created_at": _now_utc_iso(),
                }
                _write_status(config, row, payload)
                return payload

            query_url = status_url
            if "logs=" not in query_url:
                separator = "&" if "?" in query_url else "?"
                query_url = f"{query_url}{separator}logs=1"
            last_status = _http_json("GET", query_url, timeout_seconds=request_timeout)
            state = str(last_status.get("status") or "").strip()
            if state == "COMPLETED":
                result = _http_json("GET", response_url, timeout_seconds=request_timeout)
                images = result.get("images") if isinstance(result.get("images"), list) else []
                if not images:
                    raise RuntimeError(f"FAL result completed without images: {result}")
                image_info = images[0] if isinstance(images[0], dict) else {}
                image_url = str(image_info.get("url") or "").strip()
                if not image_url:
                    raise RuntimeError(f"FAL image result omitted url: {image_info}")
                image_bytes = _http_get_bytes(image_url, timeout_seconds=download_timeout)
                content_type = _content_type_for_key(output_key, fallback="image/png")
                _put_s3_bytes(
                    bucket,
                    output_key,
                    image_bytes,
                    content_type=content_type,
                    metadata={
                        "source-key": str(row["source_key"]),
                        "fal-endpoint": str(config["fal_endpoint"]),
                        "fal-request-id": request_id,
                        "remove-text-run": str(config.get("remove_text_run") or ""),
                    },
                )
                payload = _completed_payload(
                    config=config,
                    row_index=row_index,
                    row=row,
                    request_payload=request_payload,
                    submit_response=submit_response,
                    result=result,
                    output_image={
                        "bytes": len(image_bytes),
                        "fal_image": image_info,
                    },
                    elapsed_seconds=time.time() - started_at,
                )
                _write_status(config, row, payload)
                return payload
            if state in {"FAILED", "TIMED_OUT", "ABORTED"}:
                raise RuntimeError(f"FAL request {request_id} ended with status {state}: {last_status}")
            time.sleep(poll_interval)
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "row_index": row_index,
            "status": "error",
            "error": str(exc),
            "source": {
                "bucket": bucket,
                "key": row.get("source_key"),
                "s3_uri": _join_s3_uri(bucket, str(row.get("source_key") or "")),
                "relative_path": row.get("relative_path"),
            },
            "output": {"bucket": bucket, "key": output_key, "s3_uri": _join_s3_uri(bucket, output_key)},
            "fal": {
                "endpoint": str(config.get("fal_endpoint") or ""),
                "request_id": submit_response.get("request_id") if isinstance(submit_response, dict) else "",
                "status_url": submit_response.get("status_url") if isinstance(submit_response, dict) else "",
                "response_url": submit_response.get("response_url") if isinstance(submit_response, dict) else "",
            },
            "prompt_filename": str(config.get("prompt_filename") or ""),
            "prompt_sha256": str(config.get("prompt_sha256") or ""),
            "use_lora": False,
            "run_id": str(config.get("run_id") or ""),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "created_at": _now_utc_iso(),
        }
        _write_status(config, row, payload)
        return payload
