#!/usr/bin/env python3
"""Lambda worker for authorized WEBTOON/manhwa episode downloads.

The worker only downloads image URLs present in the supplied web-visible episode
HTML. Source fetches can use Decodo proxies, but S3 writes always go direct.
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEST_BUCKET = os.environ.get("DEST_BUCKET", "drawtoon")
SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "datasets/pages/source/webtoon")
PROXY_MODE = os.environ.get("PROXY_MODE", "auto").strip().lower()
IMAGE_WORKERS = int(os.environ.get("IMAGE_WORKERS", "4"))
IMAGE_RETRIES = int(os.environ.get("IMAGE_RETRIES", "4"))
HTML_RETRIES = int(os.environ.get("HTML_RETRIES", "3"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "45"))
CHECK_EXISTING = os.environ.get("CHECK_EXISTING", "0").strip().lower() in {"1", "true", "yes"}
FAIL_ON_PARTIAL_IMAGE_FAILURE = os.environ.get("FAIL_ON_PARTIAL_IMAGE_FAILURE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
S3 = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 6, "mode": "adaptive"}))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMG_RE = re.compile(r"<img\b[^>]+(?:data-url|data-original|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class FetchResult:
    body: bytes
    content_type: str | None
    transport: str


class ViewerImageParser(HTMLParser):
    def __init__(self, episode_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.episode_url = episode_url
        self.urls: list[str] = []
        self._seen: set[str] = set()
        self._viewer_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name.lower(): value or "" for name, value in attrs}
        node_id = attr.get("id", "")
        node_class = attr.get("class", "")
        starts_viewer = node_id == "_imageList" or "viewer_img" in node_class or "_imageList" in node_class
        if starts_viewer:
            self._viewer_depth += 1
        elif self._viewer_depth:
            self._viewer_depth += 1

        if tag.lower() != "img" or not self._viewer_depth:
            return
        raw = attr.get("data-url") or attr.get("data-original") or attr.get("src")
        if raw:
            self.add_url(raw)

    def handle_endtag(self, _tag: str) -> None:
        if self._viewer_depth:
            self._viewer_depth -= 1

    def add_url(self, raw: str) -> None:
        url = normalize_image_url(self.episode_url, raw)
        if not url:
            return
        if url in self._seen:
            return
        self._seen.add(url)
        self.urls.append(url)


def normalize_image_url(episode_url: str, raw: str) -> str | None:
    url = urllib.parse.urljoin(episode_url, html.unescape(raw).strip())
    parsed = urllib.parse.urlsplit(url)
    if "webtoon-phinf.pstatic.net" not in parsed.netloc:
        return None
    path = urllib.parse.quote(parsed.path, safe="/%")
    query = urllib.parse.quote(parsed.query, safe="=&?/%:+,._-")
    normalized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, parsed.fragment))
    lower_path = urllib.parse.urlsplit(normalized).path.lower()
    if "thumb" in lower_path or "thumbnail" in lower_path:
        return None
    return normalized


def decodo_proxy_urls() -> list[str]:
    host = os.environ.get("DECODO_PROXY_HOST", "").strip()
    user = os.environ.get("DECODO_PROXY_USER", "").strip()
    password = os.environ.get("DECODO_PROXY_PASS", "")
    ports = [part.strip() for part in os.environ.get("DECODO_PROXY_PORTS", "").split(",") if part.strip()]
    if not (host and user and password and ports):
        return []
    safe_user = urllib.parse.quote(user, safe="")
    safe_password = urllib.parse.quote(password, safe="")
    return [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]


PROXIES = decodo_proxy_urls()
_OPENERS: dict[str, urllib.request.OpenerDirector] = {}


def opener_for(proxy_url: str | None) -> urllib.request.OpenerDirector:
    key = proxy_url or "direct"
    opener = _OPENERS.get(key)
    if opener is not None:
        return opener
    if proxy_url:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    else:
        opener = urllib.request.build_opener()
    _OPENERS[key] = opener
    return opener


def http_get_once(url: str, *, referer: str | None, proxy_url: str | None, timeout: float) -> FetchResult:
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with opener_for(proxy_url).open(request, timeout=timeout) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type")
    return FetchResult(body=body, content_type=content_type, transport="proxy" if proxy_url else "direct")


def proxy_for_attempt(attempt: int) -> str | None:
    if not PROXIES:
        return None
    return PROXIES[(attempt - 1) % len(PROXIES)]


def fetch(url: str, *, referer: str | None, retries: int, timeout: float, proxy_mode: str) -> FetchResult:
    last_exc: Exception | None = None
    attempts = max(1, retries)
    mode = (proxy_mode or PROXY_MODE or "auto").lower()

    if mode in {"direct", "auto"}:
        direct_attempts = 1 if mode == "auto" and PROXIES else attempts
        for attempt in range(1, direct_attempts + 1):
            try:
                return http_get_once(url, referer=referer, proxy_url=None, timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt < direct_attempts:
                    time.sleep(min(4.0, 0.25 * attempt))

    if mode in {"proxy", "auto"} and PROXIES:
        for attempt in range(1, attempts + 1):
            try:
                return http_get_once(url, referer=referer, proxy_url=proxy_for_attempt(attempt), timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(min(6.0, 0.35 * attempt))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("proxy mode requested but no Decodo proxy environment is configured")


def parse_image_urls(html_text: str, episode_url: str) -> list[str]:
    parser = ViewerImageParser(episode_url)
    parser.feed(html_text)
    if parser.urls:
        return parser.urls

    urls: list[str] = []
    seen: set[str] = set()
    for raw in IMG_RE.findall(html_text):
        url = normalize_image_url(episode_url, raw)
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def extension_for_url(url: str, content_type: str | None) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
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


def object_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def key_for_image(source_prefix: str, episode: dict[str, Any], index: int, image_url: str, content_type: str | None) -> str:
    return (
        f"{source_prefix.strip('/')}/{episode['series_slug']}/{episode['slug']}/"
        f"page-{index:04d}{extension_for_url(image_url, content_type)}"
    )


def download_one_image(
    *,
    bucket: str,
    source_prefix: str,
    episode: dict[str, Any],
    image_url: str,
    index: int,
    overwrite: bool,
    proxy_mode: str,
) -> dict[str, Any]:
    provisional_key = key_for_image(source_prefix, episode, index, image_url, None)
    if CHECK_EXISTING and not overwrite and object_exists(bucket, provisional_key):
        return {"status": "skipped", "transport": None, "key": provisional_key}

    result = fetch(image_url, referer=episode["url"], retries=IMAGE_RETRIES, timeout=HTTP_TIMEOUT, proxy_mode=proxy_mode)
    key = key_for_image(source_prefix, episode, index, image_url, result.content_type)
    if CHECK_EXISTING and key != provisional_key and not overwrite and object_exists(bucket, key):
        return {"status": "skipped", "transport": result.transport, "key": key}

    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=result.body,
        ContentType=result.content_type or "image/jpeg",
        Metadata={
            "source-url": image_url,
            "episode-url": episode["url"],
            "episode-no": str(episode["episode_no"]),
            "series": episode["series_slug"],
        },
    )
    return {"status": "written", "transport": result.transport, "key": key}


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    episode = event.get("episode") or event.get("item") or event
    bucket = event.get("bucket") or DEST_BUCKET
    source_prefix = event.get("source_prefix") or SOURCE_PREFIX
    proxy_mode = (event.get("proxy_mode") or PROXY_MODE or "auto").lower()
    overwrite = bool(event.get("overwrite", False))
    max_images = event.get("max_images")

    started = time.time()
    html_result = fetch(episode["url"], referer=episode.get("list_url"), retries=HTML_RETRIES, timeout=HTTP_TIMEOUT, proxy_mode=proxy_mode)
    image_urls = parse_image_urls(html_result.body.decode("utf-8", errors="replace"), episode["url"])
    if max_images:
        image_urls = image_urls[: int(max_images)]

    written = 0
    skipped = 0
    failures = 0
    direct = 1 if html_result.transport == "direct" else 0
    proxied = 1 if html_result.transport == "proxy" else 0
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, IMAGE_WORKERS)) as pool:
        futures = [
            pool.submit(
                download_one_image,
                bucket=bucket,
                source_prefix=source_prefix,
                episode=episode,
                image_url=url,
                index=index,
                overwrite=overwrite,
                proxy_mode=proxy_mode,
            )
            for index, url in enumerate(image_urls, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                if len(errors) < 8:
                    errors.append(str(exc)[:500])
                continue
            if result["status"] == "written":
                written += 1
            else:
                skipped += 1
            if result.get("transport") == "direct":
                direct += 1
            elif result.get("transport") == "proxy":
                proxied += 1

    response = {
        "series_slug": episode["series_slug"],
        "episode_no": episode["episode_no"],
        "images": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "direct_fetches": direct,
        "proxied_fetches": proxied,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if errors:
        response["errors"] = errors
    if failures and (FAIL_ON_PARTIAL_IMAGE_FAILURE or written + skipped == 0):
        raise RuntimeError(json.dumps(response, sort_keys=True))
    return response
