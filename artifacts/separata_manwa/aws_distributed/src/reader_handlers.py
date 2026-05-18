from __future__ import annotations

import concurrent.futures
import html
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_PREFIX = "datasets/pages/single"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_PATTERNS = {
    "solo": (
        r"eu2\.contabostorage\.com/.*[:/]manga/Solo%20Manwha/Chapter%20",
        r"eu2\.contabostorage\.com/.*[:/]manga/Solo Manwha/Chapter ",
    ),
    "sss": (
        r"eu2\.contabostorage\.com/.*[:/]manga/SSS-Class%20Revival%20Hunter/Chapter%20",
        r"eu2\.contabostorage\.com/.*[:/]manga/SSS-Class Revival Hunter/Chapter ",
    ),
    "second": (r"i\.imgur\.com/[^/]+\.(?:jpg|jpeg|png|webp)(?:\?.*)?$",),
    "returner": (r"cdn\.mangavf\.fr/cdn/Book-en/A%20Returner%27s%20Magic%20Should%20Be%20Special%20ENGLISH/Chapter%20",),
    "great": (r"abyssrift\.com/GMR/[^/]+\.(?:jpg|jpeg|png|webp)(?:\?.*)?$",),
    "lout": (r"scans\.lastation\.us/manga/Trash-of-the-Counts-Family/",),
}

S3 = boto3.client(
    "s3",
    config=Config(
        max_pool_connections=128,
        connect_timeout=8,
        read_timeout=120,
        retries={"max_attempts": 8, "mode": "adaptive"},
    ),
)


def extension_for_url(url: str, content_type: str | None = None) -> str:
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


def s3_call(operation: str, func, *args, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= 6:
                break
            time.sleep(min(15.0, 0.7 * attempt + random.random()))
    raise RuntimeError(f"S3 {operation} failed after retries: {last_error}")


def object_exists(bucket: str, key: str) -> bool:
    for attempt in range(1, 5):
        try:
            S3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            if attempt >= 4:
                raise
        time.sleep(min(8.0, 0.5 * attempt + random.random()))
    return False


def load_proxy_secret(secret_name: str) -> dict[str, Any]:
    if not secret_name:
        return {}
    raw = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)["SecretString"]
    data = json.loads(raw)
    host = str(data.get("host") or data.get("DECODO_PROXY_HOST") or "").strip()
    ports_value = data.get("ports") or data.get("DECODO_PROXY_PORTS") or []
    if isinstance(ports_value, str):
        ports = [part.strip() for part in ports_value.split(",") if part.strip()]
    else:
        ports = [str(part).strip() for part in ports_value if str(part).strip()]
    user = str(data.get("user") or data.get("username") or data.get("DECODO_PROXY_USER") or "").strip()
    password = str(data.get("password") or data.get("pass") or data.get("DECODO_PROXY_PASS") or "")
    if not (host and ports and user and password):
        raise ValueError("proxy secret must contain host, ports, user, and password")
    safe_user = quote(user, safe="")
    safe_password = quote(password, safe="")
    return {"urls": [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]}


def pick_proxy(proxy_config: dict[str, Any]) -> str | None:
    urls = proxy_config.get("urls") or []
    return str(random.choice(urls)) if urls else None


def request_bytes(
    url: str,
    *,
    referer: str | None,
    accept: str,
    timeout: float,
    retries: int,
    network_mode: str,
    proxy_config: dict[str, Any],
) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": USER_AGENT, "Accept": accept, "Accept-Language": "en-US,en;q=0.9"}
            if referer:
                headers["Referer"] = referer
            request = Request(url, headers=headers)
            if network_mode == "proxy":
                proxy = pick_proxy(proxy_config)
                if not proxy:
                    raise RuntimeError("proxy mode selected but no proxy URL is configured")
                opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
                response = opener.open(request, timeout=timeout)
            else:
                response = urlopen(request, timeout=timeout)
            with response:
                return response.read(), {key.lower(): value for key, value in response.headers.items()}
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(10.0, 0.7 * attempt + random.random()))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def request_text(url: str, *, referer: str | None, network_mode: str, proxy_config: dict[str, Any]) -> str:
    body, headers = request_bytes(
        url,
        referer=referer,
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        timeout=45.0,
        retries=4,
        network_mode=network_mode,
        proxy_config=proxy_config,
    )
    content_type = headers.get("content-type", "")
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, re.I)
    if match:
        encoding = match.group(1).strip()
    return body.decode(encoding, errors="replace")


def image_allowed(site_key: str, url: str) -> bool:
    if not url or url.startswith("data:"):
        return False
    cleaned = url.replace(" ", "%20")
    for pattern in IMAGE_PATTERNS.get(site_key, ()):
        if re.search(pattern, cleaned, re.I) or re.search(pattern, url, re.I):
            return True
    return False


def srcset_urls(value: str) -> list[str]:
    urls: list[str] = []
    for part in value.split(","):
        token = part.strip().split(" ")[0].strip()
        if token:
            urls.append(token)
    return urls


def parse_image_urls(site_key: str, page_html: str, page_url: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"<img\b[^>]*>", page_html, re.I | re.S):
        tag = match.group(0)
        attrs = {
            attr.lower(): html.unescape(value)
            for attr, _quote, value in re.findall(r"""([a-zA-Z0-9_:\-]+)\s*=\s*(['"])(.*?)\2""", tag, re.S)
        }
        for attr in ("data-src", "data-lazy-src", "data-original", "data-cfsrc", "src"):
            value = attrs.get(attr)
            if value and image_allowed(site_key, value):
                urls.append(value)
        for attr in ("data-srcset", "srcset"):
            value = attrs.get(attr)
            if value:
                for url in srcset_urls(value):
                    if image_allowed(site_key, url):
                        urls.append(url)
    for raw in re.findall(r"https?://[^\"'<>\\\s)]+", page_html):
        url = html.unescape(raw)
        if image_allowed(site_key, url):
            urls.append(url)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.replace(" ", "%20")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def key_for(prefix: str, chapter: dict[str, Any], index: int, image_url: str, content_type: str | None = None) -> str:
    ext = extension_for_url(image_url, content_type)
    return f"{prefix.strip('/')}/{chapter['output_series']}/{chapter['chapter_slug']}/page-{index:04d}{ext}"


def download_one(
    *,
    chapter: dict[str, Any],
    image_url: str,
    index: int,
    bucket: str,
    prefix: str,
    overwrite: bool,
    network_mode: str,
    proxy_config: dict[str, Any],
) -> str:
    provisional_key = key_for(prefix, chapter, index, image_url)
    if not overwrite and object_exists(bucket, provisional_key):
        return "skipped"
    body, headers = request_bytes(
        image_url,
        referer=chapter["url"],
        accept="image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        timeout=60.0,
        retries=3,
        network_mode=network_mode,
        proxy_config=proxy_config,
    )
    content_type = headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        raise RuntimeError(f"unexpected content-type {content_type}")
    key = key_for(prefix, chapter, index, image_url, content_type)
    if key != provisional_key and not overwrite and object_exists(bucket, key):
        return "skipped"
    s3_call(
        "put_object",
        S3.put_object,
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={
            "source-site": str(chapter.get("site_key") or ""),
            "series": str(chapter.get("output_series") or ""),
            "chapter-url": str(chapter.get("url") or ""),
            "source-url": image_url,
            "page-index": str(index),
        },
    )
    return "written"


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    started = time.monotonic()
    row = event.get("chapter") or event.get("item") or event
    config = event.get("config") or {}
    bucket = str(config.get("bucket") or DEFAULT_BUCKET)
    prefix = str(config.get("prefix") or DEFAULT_PREFIX)
    image_concurrency = int(config.get("image_concurrency") or 8)
    overwrite = bool(config.get("overwrite") or False)
    proxy_secret_name = str(config.get("proxy_secret_name") or "")
    network_mode = str(config.get("network_mode") or "direct")
    if network_mode == "proxy":
        proxy_config = load_proxy_secret(proxy_secret_name)
    else:
        proxy_config = {}

    page_html = request_text(row["url"], referer=None, network_mode=network_mode, proxy_config=proxy_config)
    image_urls = parse_image_urls(str(row.get("site_key") or ""), page_html, row["url"])
    written = 0
    skipped = 0
    failures = 0
    samples: list[str] = []
    if image_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, image_concurrency)) as pool:
            futures = {
                pool.submit(
                    download_one,
                    chapter=row,
                    image_url=image_url,
                    index=index,
                    bucket=bucket,
                    prefix=prefix,
                    overwrite=overwrite,
                    network_mode=network_mode,
                    proxy_config=proxy_config,
                ): image_url
                for index, image_url in enumerate(image_urls, start=1)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    if len(samples) < 5:
                        samples.append(str(exc)[:500])
                    continue
                if result == "written":
                    written += 1
                else:
                    skipped += 1
    status = "accessible" if image_urls and failures == 0 else "partial_failure" if image_urls else "no_images"
    return {
        "status": status,
        "site": row.get("site_key"),
        "series": row.get("output_series"),
        "chapter_slug": row.get("chapter_slug"),
        "url": row.get("url"),
        "image_count": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "failure_samples": samples,
        "s3_prefix": f"s3://{bucket}/{prefix.strip('/')}/{row.get('output_series')}/{row.get('chapter_slug')}/",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
