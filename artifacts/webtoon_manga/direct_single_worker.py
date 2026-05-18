#!/usr/bin/env python3
"""Direct WEBTOON/manhwa downloader into datasets/pages/single.

This worker avoids the old raw source prefix. It discovers episodes for one
series, samples episodes in deterministic random order, filters each episode by
image dimensions in memory, and writes only kept story pages directly to:

    s3://<bucket>/datasets/pages/single/<series_slug>_manwa/
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEST_BUCKET = os.environ.get("DEST_BUCKET", "drawtoon")
SINGLE_PREFIX = os.environ.get("SINGLE_PREFIX", "datasets/pages/single")
STATUS_PREFIX = os.environ.get("STATUS_PREFIX", "datasets/pages/manifests/webtoon_manga_direct_single/status")
PROXY_MODE = os.environ.get("PROXY_MODE", "auto").strip().lower()
IMAGE_WORKERS = int(os.environ.get("IMAGE_WORKERS", "64"))
IMAGE_RETRIES = int(os.environ.get("IMAGE_RETRIES", "4"))
HTML_RETRIES = int(os.environ.get("HTML_RETRIES", "4"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "45"))
MAX_LIST_PAGES = int(os.environ.get("MAX_LIST_PAGES", "200"))
MAX_PAGES_PER_SERIES = int(os.environ.get("MAX_PAGES_PER_SERIES", "3000"))
SEED = int(os.environ.get("SEED", "20260518"))

MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "100"))
MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "80"))
MIN_RATIO = float(os.environ.get("MIN_RATIO", "0.62"))
MAX_RATIO = float(os.environ.get("MAX_RATIO", "7.5"))
WIDTH_TOLERANCE_RATIO = float(os.environ.get("WIDTH_TOLERANCE_RATIO", "0.08"))
WIDTH_TOLERANCE_PX = int(os.environ.get("WIDTH_TOLERANCE_PX", "28"))
STORY_WINDOW = int(os.environ.get("STORY_WINDOW", "12"))
STORY_WINDOW_MIN_GOOD = int(os.environ.get("STORY_WINDOW_MIN_GOOD", "7"))

S3 = boto3.client(
    "s3",
    config=Config(
        max_pool_connections=max(96, IMAGE_WORKERS + 32),
        connect_timeout=5,
        read_timeout=45,
        retries={"max_attempts": 6, "mode": "adaptive"},
    ),
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EPISODE_LINK_RE = re.compile(r"<a\b[^>]+href=[\"']([^\"']*/viewer\?title_no=[^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
IMG_RE = re.compile(r"<img\b[^>]+(?:data-url|data-original|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
URL_REJECT_RE = re.compile(
    r"(?:thumb|thumbnail|language[-_]?warning|age[-_]?warning|warning|notice|recommend|"
    r"banner|download[-_]?app|app[-_]?download|subscribe|promotion|promo|"
    r"advert|credit|credits|author[-_]?note|creator[-_]?note|thanks|"
    r"instagram|facebook|twitter|youtube|wallpaper)",
    re.IGNORECASE,
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class FetchResult:
    body: bytes
    content_type: str | None
    transport: str


@dataclass
class ImageCandidate:
    index: int
    url: str
    body: bytes = b""
    content_type: str | None = None
    transport: str = ""
    width: int = 0
    height: int = 0
    ratio: float = 0.0
    status: str = "undecided"
    reason: str = ""
    key: str = ""
    error: str = ""


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
        if not url or url in self._seen:
            return
        self._seen.add(url)
        self.urls.append(url)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "episode"


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


def proxy_for_attempt(attempt: int, url: str) -> str | None:
    if not PROXIES:
        return None
    offset = abs(hash(url)) % len(PROXIES)
    return PROXIES[(offset + attempt - 1) % len(PROXIES)]


def http_get_once(url: str, *, referer: str | None, proxy_url: str | None, timeout: float) -> FetchResult:
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with opener_for(proxy_url).open(request, timeout=timeout) as response:
        return FetchResult(
            body=response.read(),
            content_type=response.headers.get("Content-Type"),
            transport="proxy" if proxy_url else "direct",
        )


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
                return http_get_once(url, referer=referer, proxy_url=proxy_for_attempt(attempt, url), timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(min(6.0, 0.35 * attempt))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("proxy mode requested but no Decodo proxy environment is configured")


def fetch_text(url: str, *, referer: str | None, proxy_mode: str) -> str:
    result = fetch(url, referer=referer, retries=HTML_RETRIES, timeout=HTTP_TIMEOUT, proxy_mode=proxy_mode)
    return result.body.decode("utf-8", errors="replace")


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


def text_from_anchor(anchor_html: str) -> str:
    text = TAG_RE.sub(" ", anchor_html)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def parse_episode_links(series: dict[str, Any], html_text: str, base_url: str) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    title_no_expected = int(series["title_no"])
    for raw_href, anchor_html in EPISODE_LINK_RE.findall(html_text):
        url = urllib.parse.urljoin(base_url, html.unescape(raw_href))
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        title_values = query.get("title_no") or []
        episode_values = query.get("episode_no") or []
        if not title_values or not episode_values:
            continue
        try:
            title_no = int(title_values[0])
            episode_no = int(episode_values[0])
        except ValueError:
            continue
        if title_no != title_no_expected or episode_no in seen:
            continue
        seen.add(episode_no)
        label = text_from_anchor(anchor_html)
        parts = [part for part in Path(parsed.path).parts if part not in {"/", ""}]
        path_slug = slugify(parts[-2] if len(parts) >= 2 else label)
        episodes.append(
            {
                "series_name": series["name"],
                "series_slug": series["series_slug"],
                "title_no": title_no,
                "list_url": series["list_url"],
                "episode_no": episode_no,
                "url": url,
                "slug": f"episode-{episode_no:06d}-{path_slug}",
                "label": label[:160],
            }
        )
    return episodes


def discover_episodes(series: dict[str, Any], *, max_list_pages: int, proxy_mode: str) -> list[dict[str, Any]]:
    episodes_by_no: dict[int, dict[str, Any]] = {}
    list_url = str(series["list_url"])
    for page in range(1, max_list_pages + 1):
        separator = "&" if "?" in list_url else "?"
        url = f"{list_url}{separator}page={page}"
        html_text = fetch_text(url, referer=list_url, proxy_mode=proxy_mode)
        page_episodes = parse_episode_links(series, html_text, url)
        if not page_episodes:
            break
        before = len(episodes_by_no)
        for episode in page_episodes:
            episodes_by_no.setdefault(int(episode["episode_no"]), episode)
        if len(episodes_by_no) == before:
            break
    return [episodes_by_no[key] for key in sorted(episodes_by_no)]


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
            return (
                int.from_bytes(data[start + 6 : start + 8], "little") & 0x3FFF,
                int.from_bytes(data[start + 8 : start + 10], "little") & 0x3FFF,
            )
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


def list_existing_image_count(bucket: str, prefix: str) -> int:
    total = 0
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if any(key.lower().endswith(suffix) for suffix in IMAGE_EXTENSIONS):
                total += 1
    return total


def object_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def dominant_width(rows: list[ImageCandidate]) -> int:
    counts: Counter[int] = Counter()
    for row in rows:
        if row.url and URL_REJECT_RE.search(row.url):
            continue
        if row.width >= MIN_WIDTH and row.height >= MIN_HEIGHT:
            counts[int(round(row.width / 10.0) * 10)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    fallback = sorted(row.width for row in rows if row.width > 0)
    return fallback[len(fallback) // 2] if fallback else 0


def base_candidate(row: ImageCandidate, dom_width: int) -> tuple[bool, str]:
    if row.reason == "download_error":
        return False, row.reason
    if row.reason == "dimension_error":
        return False, row.reason
    if row.width < MIN_WIDTH or row.height < MIN_HEIGHT:
        return False, "too_small"
    if row.ratio < MIN_RATIO:
        return False, "ratio_low"
    if row.ratio > MAX_RATIO:
        return False, "ratio_high"
    if dom_width and abs(row.width - dom_width) > max(WIDTH_TOLERANCE_PX, int(dom_width * WIDTH_TOLERANCE_RATIO)):
        return False, "width_outlier"
    if row.url and URL_REJECT_RE.search(row.url):
        return False, "url_reject"
    return True, "story_dimension_match"


def find_story_bounds(flags: list[bool]) -> tuple[int, int]:
    if not flags:
        return 0, -1
    if sum(flags) < max(3, STORY_WINDOW_MIN_GOOD):
        good = [i for i, flag in enumerate(flags) if flag]
        return (good[0], good[-1]) if good else (0, -1)

    start = 0
    for i in range(len(flags)):
        window = flags[i : i + STORY_WINDOW]
        if sum(window) >= min(STORY_WINDOW_MIN_GOOD, len(window)):
            start = i
            break

    end = len(flags) - 1
    for i in range(len(flags) - 1, -1, -1):
        window = flags[max(0, i - STORY_WINDOW + 1) : i + 1]
        if sum(window) >= min(STORY_WINDOW_MIN_GOOD, len(window)):
            end = i
            break
    return start, end


def download_candidate(episode: dict[str, Any], index: int, url: str, proxy_mode: str) -> ImageCandidate:
    row = ImageCandidate(index=index, url=url)
    try:
        result = fetch(url, referer=episode["url"], retries=IMAGE_RETRIES, timeout=HTTP_TIMEOUT, proxy_mode=proxy_mode)
        row.body = result.body
        row.content_type = result.content_type
        row.transport = result.transport
        row.width, row.height = parse_image_dimensions(result.body[:1048576])
        row.ratio = float(row.height) / max(1.0, float(row.width))
    except Exception as exc:
        row.status = "drop"
        row.reason = "download_error" if not row.body else "dimension_error"
        row.error = str(exc)[:500]
    return row


def filter_episode(episode: dict[str, Any], image_urls: list[str], proxy_mode: str) -> tuple[list[ImageCandidate], dict[str, Any]]:
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, IMAGE_WORKERS)) as pool:
        rows = list(pool.map(lambda item: download_candidate(episode, item[0], item[1], proxy_mode), enumerate(image_urls, start=1)))
    rows.sort(key=lambda row: row.index)

    dom_width = dominant_width(rows)
    candidate_flags: list[bool] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        ok, reason = base_candidate(row, dom_width)
        candidate_flags.append(ok)
        row.reason = reason
        reasons[reason] += 1

    start_idx, end_idx = find_story_bounds(candidate_flags)
    kept: list[ImageCandidate] = []
    dropped: list[ImageCandidate] = []
    for pos, row in enumerate(rows):
        ok = candidate_flags[pos] and start_idx <= pos <= end_idx
        if ok:
            row.status = "keep"
            kept.append(row)
        else:
            row.status = "drop"
            if candidate_flags[pos] and not (start_idx <= pos <= end_idx):
                row.reason = "outside_story_run"
            dropped.append(row)

    report = {
        "episode_no": episode["episode_no"],
        "input_images": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "dominant_width": dom_width,
        "story_start_position": start_idx + 1 if end_idx >= start_idx else None,
        "story_end_position": end_idx + 1 if end_idx >= start_idx else None,
        "drop_reasons": dict(sorted(Counter(row.reason for row in dropped).items())),
        "candidate_reasons": dict(sorted(reasons.items())),
        "direct_fetches": sum(1 for row in rows if row.transport == "direct"),
        "proxied_fetches": sum(1 for row in rows if row.transport == "proxy"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return kept, report


def output_key(single_prefix: str, series_slug: str, episode: dict[str, Any], row: ImageCandidate) -> str:
    ext = extension_for_url(row.url, row.content_type)
    return f"{single_prefix.strip('/')}/{series_slug}_manwa/{episode['slug']}__page-{row.index:04d}{ext}"


def put_image(bucket: str, key: str, episode: dict[str, Any], row: ImageCandidate) -> None:
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=row.body,
        ContentType=row.content_type or "image/jpeg",
        Metadata={
            "source-url": row.url,
            "episode-url": episode["url"],
            "episode-no": str(episode["episode_no"]),
            "series": episode["series_slug"],
            "filtered": "dimension_direct_single",
        },
    )


def write_status(bucket: str, status_prefix: str, run_id: str, series_slug: str, payload: dict[str, Any]) -> None:
    key = f"{status_prefix.strip('/')}/{run_id}/{series_slug}.json"
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    started = time.time()
    series = event.get("series") or event.get("item") or event
    bucket = str(event.get("bucket") or DEST_BUCKET)
    single_prefix = str(event.get("single_prefix") or SINGLE_PREFIX).strip("/")
    status_prefix = str(event.get("status_prefix") or STATUS_PREFIX).strip("/")
    proxy_mode = str(event.get("proxy_mode") or PROXY_MODE or "auto").lower()
    max_pages = int(event.get("max_pages") or MAX_PAGES_PER_SERIES)
    max_list_pages = int(event.get("max_list_pages") or MAX_LIST_PAGES)
    run_id = str(event.get("run_id") or "manual")
    seed = int(event.get("seed") or SEED)
    dry_run = bool(event.get("dry_run", False))
    overwrite = bool(event.get("overwrite", False))

    series_slug = str(series["series_slug"])
    output_prefix = f"{single_prefix}/{series_slug}_manwa/"
    existing = 0 if overwrite or dry_run else list_existing_image_count(bucket, output_prefix)
    if existing >= max_pages and not dry_run:
        response = {
            "series_slug": series_slug,
            "status": "already_complete",
            "existing": existing,
            "written": 0,
            "target_max_pages": max_pages,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_status(bucket, status_prefix, run_id, series_slug, response)
        return response

    episodes = discover_episodes(series, max_list_pages=max_list_pages, proxy_mode=proxy_mode)
    shuffled = list(episodes)
    random.Random(f"{seed}:{series_slug}").shuffle(shuffled)

    total_present = existing
    written = 0
    skipped_existing = 0
    filtered_kept = 0
    filtered_dropped = 0
    input_images = 0
    direct_fetches = 0
    proxied_fetches = 0
    episode_reports: list[dict[str, Any]] = []
    errors: list[str] = []

    base_status = {
        "schema_version": 1,
        "run_id": run_id,
        "series_slug": series_slug,
        "series_name": series.get("name"),
        "title_no": series.get("title_no"),
        "single_prefix": output_prefix,
        "target_max_pages": max_pages,
        "dry_run": dry_run,
        "existing_at_start": existing,
        "episodes_discovered": len(episodes),
    }

    for episode in shuffled:
        if total_present >= max_pages:
            break
        try:
            html_result = fetch(
                episode["url"],
                referer=episode.get("list_url"),
                retries=HTML_RETRIES,
                timeout=HTTP_TIMEOUT,
                proxy_mode=proxy_mode,
            )
            episode_direct = 1 if html_result.transport == "direct" else 0
            episode_proxied = 1 if html_result.transport == "proxy" else 0
            image_urls = parse_image_urls(html_result.body.decode("utf-8", errors="replace"), episode["url"])
            kept, report = filter_episode(episode, image_urls, proxy_mode)
            report["html_transport"] = html_result.transport
            report["episode_slug"] = episode["slug"]
            report["episode_url"] = episode["url"]
            input_images += int(report["input_images"])
            filtered_kept += int(report["kept"])
            filtered_dropped += int(report["dropped"])
            direct_fetches += int(report["direct_fetches"]) + episode_direct
            proxied_fetches += int(report["proxied_fetches"]) + episode_proxied

            episode_written = 0
            episode_skipped = 0
            for row in kept:
                if total_present >= max_pages:
                    break
                key = output_key(single_prefix, series_slug, episode, row)
                row.key = key
                if not overwrite and not dry_run and object_exists(bucket, key):
                    skipped_existing += 1
                    episode_skipped += 1
                    continue
                if not dry_run:
                    put_image(bucket, key, episode, row)
                    total_present += 1
                else:
                    total_present += 1
                written += 1
                episode_written += 1
                if total_present >= max_pages:
                    break

            report["written"] = episode_written
            report["skipped_existing"] = episode_skipped
            episode_reports.append(report)
        except Exception as exc:
            errors.append(f"episode {episode.get('episode_no')}: {str(exc)[:500]}")
            if len(errors) >= 12:
                break

        status_payload = {
            **base_status,
            "status": "running" if total_present < max_pages else "complete",
            "total_present_or_selected": total_present,
            "written": written,
            "skipped_existing": skipped_existing,
            "episodes_attempted": len(episode_reports),
            "input_images": input_images,
            "filtered_kept": filtered_kept,
            "filtered_dropped": filtered_dropped,
            "direct_fetches": direct_fetches,
            "proxied_fetches": proxied_fetches,
            "recent_episode_reports": episode_reports[-12:],
            "errors": errors[-12:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_status(bucket, status_prefix, run_id, series_slug, status_payload)

    final_status = {
        **base_status,
        "status": "complete" if total_present >= max_pages else "partial",
        "total_present_or_selected": total_present,
        "written": written,
        "skipped_existing": skipped_existing,
        "episodes_attempted": len(episode_reports),
        "episodes_remaining_unattempted": max(0, len(shuffled) - len(episode_reports)),
        "input_images": input_images,
        "filtered_kept": filtered_kept,
        "filtered_dropped": filtered_dropped,
        "direct_fetches": direct_fetches,
        "proxied_fetches": proxied_fetches,
        "episode_reports": episode_reports,
        "errors": errors[-20:],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_status(bucket, status_prefix, run_id, series_slug, final_status)
    if final_status["status"] != "complete" and written == 0 and existing == 0:
        raise RuntimeError(json.dumps(final_status, sort_keys=True))
    return {
        "series_slug": series_slug,
        "status": final_status["status"],
        "written": written,
        "total_present_or_selected": total_present,
        "target_max_pages": max_pages,
        "episodes_attempted": len(episode_reports),
        "direct_fetches": direct_fetches,
        "proxied_fetches": proxied_fetches,
        "elapsed_seconds": final_status["elapsed_seconds"],
    }
