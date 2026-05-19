from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEST_BUCKET = os.environ.get("DEST_BUCKET", "drawtoon")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "45"))
IMAGE_RETRIES = int(os.environ.get("IMAGE_RETRIES", "4"))
PROXY_MODE = os.environ.get("PROXY_MODE", "auto").strip().lower()
SESSION_SECRET_NAME = os.environ.get("SESSION_SECRET_NAME", "").strip()
MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "1"))
MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "1"))
REQUEST_DELAY_SECONDS = float(os.environ.get("REQUEST_DELAY_SECONDS", "0"))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1").strip().lower() not in {"0", "false", "no"}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

S3 = boto3.client(
    "s3",
    config=Config(max_pool_connections=64, connect_timeout=5, read_timeout=60, retries={"max_attempts": 5, "mode": "adaptive"}),
)
SECRETS = boto3.client("secretsmanager", config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 3}))
_SESSION_HEADERS: dict[str, str] | None = None
_OPENERS: dict[str, Any] = {}


def decodo_proxy_urls() -> list[str]:
    host = os.environ.get("DECODO_PROXY_HOST", "").strip()
    user = os.environ.get("DECODO_PROXY_USER", "").strip()
    password = os.environ.get("DECODO_PROXY_PASS", "")
    ports = [part.strip() for part in os.environ.get("DECODO_PROXY_PORTS", "").split(",") if part.strip()]
    if not (host and user and password and ports):
        return []
    safe_user = quote(user, safe="")
    safe_password = quote(password, safe="")
    return [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]


PROXIES = decodo_proxy_urls()


def opener_for(proxy_url: str | None):
    key = proxy_url or "direct"
    opener = _OPENERS.get(key)
    if opener is not None:
        return opener
    if proxy_url:
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        opener = build_opener()
    _OPENERS[key] = opener
    return opener


def proxy_for_attempt(attempt: int, url: str) -> str | None:
    if not PROXIES:
        return None
    offset = abs(hash(url)) % len(PROXIES)
    return PROXIES[(offset + attempt - 1) % len(PROXIES)]


def session_headers() -> dict[str, str]:
    global _SESSION_HEADERS
    if _SESSION_HEADERS is not None:
        return _SESSION_HEADERS
    headers: dict[str, str] = {}
    inline = os.environ.get("SESSION_HEADERS_JSON", "").strip()
    if inline:
        payload = json.loads(inline)
        headers.update({str(k): str(v) for k, v in (payload.get("headers") or payload).items()})
    if SESSION_SECRET_NAME:
        secret = SECRETS.get_secret_value(SecretId=SESSION_SECRET_NAME)
        payload = json.loads(secret.get("SecretString") or "{}")
        headers.update({str(k): str(v) for k, v in (payload.get("headers") or {}).items()})
        if payload.get("cookies") and "Cookie" not in headers:
            headers["Cookie"] = str(payload["cookies"])
    _SESSION_HEADERS = headers
    return headers


def parse_image_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if len(data) >= 30 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if chunk == b"VP8 " and len(data) >= 30:
            start = 20
            return int.from_bytes(data[start + 6 : start + 8], "little") & 0x3FFF, int.from_bytes(data[start + 8 : start + 10], "little") & 0x3FFF
        if chunk == b"VP8L" and len(data) >= 25:
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            return 1 + (((b1 & 0x3F) << 8) | b0), 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
    if len(data) >= 4 and data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            while index < len(data) and data[index] == 0xFF:
                index += 1
            if index >= len(data):
                break
            marker = data[index]
            index += 1
            if marker in {0xD8, 0xD9, 0x01}:
                continue
            if index + 2 > len(data):
                break
            segment_length = int.from_bytes(data[index : index + 2], "big")
            if segment_length < 2:
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                return int.from_bytes(data[index + 5 : index + 7], "big"), int.from_bytes(data[index + 3 : index + 5], "big")
            index += segment_length
    raise ValueError("unsupported image header or missing dimensions")


def extension_for(url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed == ".jpeg":
            return ".jpg"
        if guessed in IMAGE_EXTENSIONS:
            return guessed
    return ".jpg"


def output_location(row: dict[str, Any], ext: str) -> tuple[str, str]:
    bucket = str(row.get("bucket") or DEST_BUCKET)
    if row.get("output_key"):
        return bucket, str(row["output_key"]).lstrip("/")
    prefix = str(row.get("output_prefix") or "").strip("/")
    if prefix.startswith("s3://"):
        parsed = urlparse(prefix)
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
    page_no = int(row.get("page_no") or 1)
    return bucket, f"{prefix}/page-{page_no:04d}{ext}"


def object_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def open_url(url: str, headers: dict[str, str], proxy_url: str | None) -> tuple[bytes, str | None, str]:
    request = Request(url, headers=headers)
    with opener_for(proxy_url).open(request, timeout=HTTP_TIMEOUT) as response:
        return response.read(), response.headers.get("Content-Type"), "proxy" if proxy_url else "direct"


def retry_after_seconds(value: str | None, attempt: int) -> float:
    if value:
        try:
            return min(30.0, max(1.0, float(value)))
        except ValueError:
            pass
    return min(30.0, 1.5 * attempt)


def fetch_image(row: dict[str, Any], proxy_mode: str) -> tuple[bytes, str | None, str]:
    url = str(row["image_url"])
    headers = dict(DEFAULT_HEADERS)
    headers.update(session_headers())
    if row.get("referer"):
        headers["Referer"] = str(row["referer"])
    mode = (proxy_mode or PROXY_MODE or "auto").lower()
    attempts: list[str | None] = []
    if mode in {"direct", "auto"}:
        attempts.extend([None] * max(1, IMAGE_RETRIES))
    if mode in {"proxy", "auto"}:
        attempts.extend(proxy_for_attempt(i, url) for i in range(1, min(IMAGE_RETRIES, max(1, len(PROXIES))) + 1))
    if not attempts:
        attempts.append(None)
    last_error: Exception | None = None
    for attempt, proxy_url in enumerate(attempts, start=1):
        try:
            return open_url(url, headers, proxy_url)
        except HTTPError as exc:
            last_error = exc
            if exc.code in {404, 410}:
                break
            if attempt < len(attempts):
                time.sleep(retry_after_seconds(exc.headers.get("Retry-After"), attempt))
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < len(attempts):
                time.sleep(min(10.0, 0.75 * attempt))
    raise RuntimeError(str(last_error or "fetch failed")[:500])


def process_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    dry_run = bool(config.get("dry_run") or False)
    proxy_mode = str(config.get("proxy_mode") or PROXY_MODE or "auto")
    try:
        prefetch_ext = extension_for(str(row.get("image_url") or ""), None)
        prefetch_bucket, prefetch_key = output_location(row, prefetch_ext)
        if SKIP_EXISTING and not dry_run and object_exists(prefetch_bucket, prefetch_key):
            return {"status": "skipped_existing", "bucket": prefetch_bucket, "key": prefetch_key}
        delay = float(config.get("request_delay_seconds") or REQUEST_DELAY_SECONDS or 0)
        if delay > 0 and not dry_run:
            time.sleep(delay)
        body, content_type, transport = fetch_image(row, proxy_mode)
        if content_type and not content_type.lower().split(";", 1)[0].startswith("image/"):
            return {"status": "dropped", "reason": f"non_image:{content_type}", "transport": transport}
        width, height = parse_image_dimensions(body[: min(len(body), 2_000_000)])
        if width < int(config.get("min_width") or MIN_WIDTH) or height < int(config.get("min_height") or MIN_HEIGHT):
            return {"status": "dropped", "reason": "too_small", "width": width, "height": height, "transport": transport}
        ext = extension_for(str(row["image_url"]), content_type)
        bucket, key = output_location(row, ext)
        if SKIP_EXISTING and object_exists(bucket, key):
            return {"status": "skipped_existing", "bucket": bucket, "key": key, "width": width, "height": height}
        if dry_run:
            return {"status": "would_write", "bucket": bucket, "key": key, "width": width, "height": height, "transport": transport}
        metadata = {
            "source-type": str(row.get("source_type") or ""),
            "platform": str(row.get("platform") or ""),
            "series-slug": str(row.get("series_slug") or ""),
            "issue-slug": str(row.get("issue_slug") or ""),
            "source-url": str(row.get("image_url") or ""),
            "referer": str(row.get("referer") or ""),
            "width": str(width),
            "height": str(height),
        }
        S3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type or mimetypes.guess_type(key)[0] or "image/jpeg", Metadata=metadata)
        return {"status": "written", "bucket": bucket, "key": key, "width": width, "height": height, "bytes": len(body), "transport": transport, "elapsed_seconds": round(time.time() - started, 3)}
    except Exception as exc:
        result = {
            "status": "failed",
            "error": str(exc)[:500],
            "image_url": str(row.get("image_url") or ""),
            "series_slug": str(row.get("series_slug") or ""),
            "issue_slug": str(row.get("issue_slug") or ""),
            "page_no": row.get("page_no"),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        try:
            bucket, key = output_location(row, extension_for(str(row.get("image_url") or ""), None))
            result.update({"bucket": bucket, "key": key})
        except Exception:
            pass
        return result


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config = event.get("config") or {}
    row = event.get("row") or event.get("item") or event
    if isinstance(row, list):
        results = [process_row(item, config) for item in row]
        counts: dict[str, int] = {}
        for result in results:
            status = str(result.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {"status": "batch_done", "counts": counts, "results": results[:20]}
    return process_row(row, config)
