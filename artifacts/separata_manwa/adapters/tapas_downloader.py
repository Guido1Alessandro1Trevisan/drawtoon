#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import html
import json
import mimetypes
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import boto3
import requests
from bs4 import BeautifulSoup
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_PREFIX = "datasets/pages/source/tapas"
DEFAULT_MANIFEST_DIR = Path("artifacts/separata_manwa/manifests")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclasses.dataclass(frozen=True)
class TapasSeries:
    requested_title: str
    official_title: str
    slug: str
    series_id: int
    free_note: str

    @property
    def info_url(self) -> str:
        return f"https://tapas.io/series/{self.slug}/info"


SERIES: tuple[TapasSeries, ...] = (
    TapasSeries(
        "Solo Leveling",
        "Solo Leveling",
        "solo-leveling-comic",
        202473,
        "first 4 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "Ranker Who Lives a Second Time",
        "Second Life Ranker",
        "second-life-ranker",
        194483,
        "first 3 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "SSS-Class Suicide Hunter",
        "SSS-Class Revival Hunter",
        "sss-class-revival-hunter",
        198789,
        "first 4 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "Overgeared",
        "Overgeared",
        "overgeared",
        214495,
        "first 4 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "A Returner's Magic Should Be Special",
        "A Returner's Magic Should Be Special",
        "a-returners-magic-should-be-special",
        202494,
        "first 3 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "The Great Mage Returns After 4,000 Years",
        "The Archmage Returns After 4000 Years",
        "the-archmage-returns-after-4000-years",
        193935,
        "first 3 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
    TapasSeries(
        "Trash of the Count's Family",
        "Lout of Count's Family",
        "lout-of-counts-family",
        267159,
        "first 4 public/free episodes; later WUF/Ink locked unless separately authorized",
    ),
)


class ProxyPool:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.index = 0
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "ProxyPool":
        host = os.environ.get("DECODO_PROXY_HOST", "").strip()
        ports_raw = os.environ.get("DECODO_PROXY_PORTS", "").strip()
        user = os.environ.get("DECODO_PROXY_USER", "").strip()
        password = os.environ.get("DECODO_PROXY_PASS", "")
        if not (host and ports_raw and user and password):
            return cls([])
        safe_user = quote(user, safe="")
        safe_password = quote(password, safe="")
        ports = [part.strip() for part in ports_raw.split(",") if part.strip()]
        urls = [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]
        return cls(urls)

    @property
    def enabled(self) -> bool:
        return bool(self.urls)

    def next_proxies(self) -> dict[str, str] | None:
        if not self.urls:
            return None
        with self.lock:
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
        return {"http": url, "https": url}


class HttpClient:
    def __init__(self, *, proxy_pool: ProxyPool, network_mode: str, timeout: float) -> None:
        self.proxy_pool = proxy_pool
        self.network_mode = network_mode
        self.timeout = timeout
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            self.local.session = session
        return session

    def get(
        self,
        url: str,
        *,
        referer: str | None = None,
        accept: str = "*/*",
        stream: bool = False,
        retries: int = 4,
        timeout: float | None = None,
    ) -> requests.Response:
        last_error: Exception | None = None
        headers = {"Accept": accept}
        if referer:
            headers["Referer"] = referer
        for attempt in range(1, retries + 1):
            try:
                proxies = self.proxy_pool.next_proxies() if self.network_mode == "proxy" else None
                response = self.session().get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=timeout or self.timeout,
                    stream=stream,
                )
                if response.status_code in {401, 403, 404, 429}:
                    return response
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(8.0, 0.4 * attempt + random.random() * 0.4))
        raise RuntimeError(f"GET failed status/url={url}: {last_error}")


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=256,
            connect_timeout=15,
            read_timeout=120,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "episode"


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


def select_series(names: list[str]) -> list[TapasSeries]:
    if not names:
        return list(SERIES)
    by_key: dict[str, TapasSeries] = {}
    for series in SERIES:
        keys = {
            series.slug,
            safe_slug(series.requested_title),
            safe_slug(series.official_title),
            series.requested_title.lower(),
            series.official_title.lower(),
        }
        for key in keys:
            by_key[key] = series
    selected: list[TapasSeries] = []
    missing: list[str] = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        series = by_key.get(key) or by_key.get(key.lower()) or by_key.get(safe_slug(key))
        if series is None:
            missing.append(name)
        elif series not in selected:
            selected.append(series)
    if missing:
        raise ValueError(f"unknown Tapas series: {', '.join(missing)}")
    return selected


def parse_series_id(info_html: str, fallback: int) -> int:
    candidates = [
        r"""data-series-id=["'](\d+)["']""",
        r"""series_id["']?\s*[:=]\s*["']?(\d+)""",
        r"""/series/(\d+)/episodes""",
    ]
    for pattern in candidates:
        match = re.search(pattern, info_html, re.I)
        if match:
            return int(match.group(1))
    return fallback


def html_text_from_fragment(fragment: str) -> str:
    return BeautifulSoup(fragment or "", "html.parser").get_text(" ", strip=True)


def episode_from_json(series: TapasSeries, item: dict[str, Any], position: int) -> dict[str, Any]:
    episode_id = int(item["id"])
    title = html.unescape(str(item.get("title") or f"Episode {position}"))
    free = bool(item.get("free") or item.get("free_access") or item.get("unlocked"))
    must_pay = bool(item.get("must_pay"))
    return {
        "platform": "tapas",
        "series": {
            "requested_title": series.requested_title,
            "official_title": series.official_title,
            "slug": series.slug,
            "series_id": series.series_id,
            "info_url": series.info_url,
        },
        "position": position,
        "episode_id": episode_id,
        "episode_slug": f"episode-{position:04d}-{episode_id}",
        "title": title,
        "episode_url": f"https://tapas.io/episode/{episode_id}",
        "publish_date": item.get("publish_date"),
        "free": free,
        "locked_hint": bool(must_pay or not free),
        "metadata": {
            "view_count": item.get("view_cnt"),
            "like_count": item.get("like_cnt"),
            "comment_count": item.get("comment_cnt"),
            "must_pay": must_pay,
            "nu": bool(item.get("nu")),
            "early_access": bool(item.get("early_access")),
            "mature": bool(item.get("mature")),
        },
    }


def parse_body_episode_flags(body_html: str) -> dict[int, dict[str, Any]]:
    flags: dict[int, dict[str, Any]] = {}
    soup = BeautifulSoup(body_html or "", "html.parser")
    for node in soup.select("[data-id]"):
        raw_id = node.get("data-id")
        if not raw_id or not raw_id.isdigit():
            continue
        episode_id = int(raw_id)
        flags[episode_id] = {
            "is_wuf": node.get("data-is-wuf") == "true",
            "is_wait_or_pay": node.get("data-is-wait-or-pay") == "true",
            "is_charging": node.get("data-is-charging") == "true",
            "requires_signin": "js-have-to-sign" in (node.get("class") or []),
            "body_label": html_text_from_fragment(str(node)),
        }
    return flags


def discover_series(
    http: HttpClient,
    series: TapasSeries,
    *,
    max_pages: int,
    max_episodes: int,
) -> list[dict[str, Any]]:
    info_response = http.get(
        series.info_url,
        referer=None,
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        retries=4,
        timeout=30.0,
    )
    if info_response.status_code >= 400:
        raise RuntimeError(f"Tapas info page returned {info_response.status_code} for {series.slug}")
    # Tapas info pages can contain links to related/fan-read series. Use the
    # vetted official series id from the title map instead of a broad HTML match.
    parse_series_id(info_response.text, series.series_id)

    episodes_by_id: dict[int, dict[str, Any]] = {}
    page = 1
    while page <= max_pages:
        url = f"https://tapas.io/series/{series.series_id}/episodes?page={page}"
        response = http.get(
            url,
            referer=series.info_url,
            accept="application/json, text/javascript, */*; q=0.01",
            retries=4,
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Tapas episode endpoint returned {response.status_code} for {series.slug} page={page}")
        payload = response.json()
        data = payload.get("data") or {}
        items = list(data.get("episodes") or [])
        if not items:
            break
        flags = parse_body_episode_flags(str(data.get("body") or ""))
        for item in items:
            position = len(episodes_by_id) + 1
            episode = episode_from_json(series, item, position)
            extra_flags = flags.get(int(episode["episode_id"])) or {}
            if extra_flags:
                episode["metadata"].update(extra_flags)
                if extra_flags.get("is_wuf") or extra_flags.get("is_wait_or_pay"):
                    episode["locked_hint"] = bool(not episode["free"])
            episodes_by_id.setdefault(int(episode["episode_id"]), episode)
            if max_episodes and len(episodes_by_id) >= max_episodes:
                break
        if max_episodes and len(episodes_by_id) >= max_episodes:
            break
        pagination = data.get("pagination") or {}
        if not pagination.get("has_next"):
            break
        next_page = int(pagination.get("page") or (page + 1))
        if next_page <= page:
            next_page = page + 1
        page = next_page

    return list(episodes_by_id.values())


def parse_episode_images(episode_html: str, episode_url: str) -> list[str]:
    soup = BeautifulSoup(episode_html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    selectors = [
        "article.viewer__body img.content__img[data-src]",
        "img.content__img[data-src]",
        "article.viewer__body img[data-src]",
    ]
    for selector in selectors:
        for img in soup.select(selector):
            raw = img.get("data-src") or img.get("src")
            if not raw:
                continue
            url = urljoin(episode_url, raw)
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc.endswith("tapas.io"):
                continue
            if "/pc/" not in parsed.path and "/c/" not in parsed.path:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        if urls:
            break
    return urls


def object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def s3_key(prefix: str, episode: dict[str, Any], index: int, image_url: str, content_type: str | None = None) -> str:
    series_slug = episode["series"]["slug"]
    episode_slug = episode["episode_slug"]
    ext = extension_for_url(image_url, content_type)
    return f"{prefix.strip('/')}/{series_slug}/{episode_slug}/page-{index:04d}{ext}"


def download_image(
    *,
    http: HttpClient,
    client: Any,
    bucket: str,
    prefix: str,
    episode: dict[str, Any],
    image_url: str,
    index: int,
    overwrite: bool,
) -> str:
    provisional_key = s3_key(prefix, episode, index, image_url)
    if not overwrite and object_exists(client, bucket, provisional_key):
        return "skipped"
    response = http.get(
        image_url,
        referer=str(episode["episode_url"]),
        accept="image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        stream=False,
        retries=5,
        timeout=90.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"image returned {response.status_code}")
    content_type = response.headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        raise RuntimeError(f"unexpected content-type {content_type}")
    key = s3_key(prefix, episode, index, image_url, content_type)
    if key != provisional_key and not overwrite and object_exists(client, bucket, key):
        return "skipped"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=response.content,
        ContentType=content_type,
        Metadata={
            "source-platform": "tapas",
            "series-slug": str(episode["series"]["slug"]),
            "episode-id": str(episode["episode_id"]),
            "episode-url": str(episode["episode_url"]),
            "page-index": str(index),
        },
    )
    return "written"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_file(client: Any, *, bucket: str, key: str, path: Path, content_type: str) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(), ContentType=content_type)


def choose_network_mode(proxy_pool: ProxyPool, requested: str) -> str:
    requested = requested.lower().strip()
    if requested == "never":
        return "direct"
    if requested == "always":
        if not proxy_pool.enabled:
            raise ValueError("proxy-mode=always requires DECODO_PROXY_* environment variables")
        return "proxy"
    try:
        response = requests.get(
            "https://tapas.io/series/solo-leveling-comic/info",
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=12.0,
        )
        response.raise_for_status()
        return "direct"
    except Exception:
        if not proxy_pool.enabled:
            raise
        return "proxy"


def summarize_counts(status_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "episodes": len(status_rows),
        "accessible": 0,
        "locked_or_unavailable": 0,
        "written": 0,
        "skipped": 0,
        "failures": 0,
        "images": 0,
    }
    for row in status_rows:
        counts["written"] += int(row.get("written") or 0)
        counts["skipped"] += int(row.get("skipped") or 0)
        counts["failures"] += int(row.get("failures") or 0)
        counts["images"] += int(row.get("image_count") or 0)
        if row.get("status") == "accessible":
            counts["accessible"] += 1
        elif row.get("status") == "locked_or_unavailable":
            counts["locked_or_unavailable"] += 1
    return counts


def download_episode(
    *,
    http: HttpClient,
    client: Any,
    bucket: str,
    prefix: str,
    episode: dict[str, Any],
    image_workers: int,
    max_images: int,
    overwrite: bool,
    probe_locked: bool,
) -> dict[str, Any]:
    if episode.get("locked_hint") and not probe_locked:
        return {
            "status": "locked_or_unavailable",
            "reason": "metadata_not_public_free",
            "series": episode["series"]["slug"],
            "episode_id": episode["episode_id"],
            "episode_slug": episode["episode_slug"],
            "title": episode["title"],
            "image_count": 0,
            "written": 0,
            "skipped": 0,
            "failures": 0,
        }

    response = http.get(
        str(episode["episode_url"]),
        referer=str(episode["series"]["info_url"]),
        accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        retries=4,
        timeout=45.0,
    )
    if response.status_code in {401, 403, 404, 429}:
        return {
            "status": "locked_or_unavailable",
            "reason": f"episode_http_{response.status_code}",
            "series": episode["series"]["slug"],
            "episode_id": episode["episode_id"],
            "episode_slug": episode["episode_slug"],
            "title": episode["title"],
            "image_count": 0,
            "written": 0,
            "skipped": 0,
            "failures": 0,
        }
    image_urls = parse_episode_images(response.text, str(episode["episode_url"]))
    if max_images > 0:
        image_urls = image_urls[:max_images]
    if not image_urls:
        return {
            "status": "locked_or_unavailable",
            "reason": "no_public_content_images",
            "series": episode["series"]["slug"],
            "episode_id": episode["episode_id"],
            "episode_slug": episode["episode_slug"],
            "title": episode["title"],
            "image_count": 0,
            "written": 0,
            "skipped": 0,
            "failures": 0,
        }

    written = 0
    skipped = 0
    failures = 0
    samples: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, image_workers)) as pool:
        future_map = {
            pool.submit(
                download_image,
                http=http,
                client=client,
                bucket=bucket,
                prefix=prefix,
                episode=episode,
                image_url=image_url,
                index=index,
                overwrite=overwrite,
            ): image_url
            for index, image_url in enumerate(image_urls, start=1)
        }
        for future in concurrent.futures.as_completed(future_map):
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                if len(samples) < 5:
                    samples.append(str(exc))
                continue
            if result == "written":
                written += 1
            else:
                skipped += 1

    status = "accessible" if failures == 0 else "partial_failure"
    return {
        "status": status,
        "reason": "public_content_images",
        "series": episode["series"]["slug"],
        "episode_id": episode["episode_id"],
        "episode_slug": episode["episode_slug"],
        "title": episode["title"],
        "image_count": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "failure_samples": samples,
        "s3_prefix": f"s3://{bucket}/{prefix.strip('/')}/{episode['series']['slug']}/{episode['episode_slug']}/",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Tapas public/free manhwa pages to S3.")
    parser.add_argument("--series", action="append", default=[], help="Tapas slug or title. Repeatable. Default: all supported.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--network-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--max-episodes-per-series", type=int, default=0)
    parser.add_argument("--max-images-per-episode", type=int, default=0)
    parser.add_argument("--episode-workers", type=int, default=8)
    parser.add_argument("--image-workers", type=int, default=12)
    parser.add_argument("--download", action="store_true", help="Download images to S3. Without this, only writes manifests.")
    parser.add_argument("--probe-locked", action="store_true", help="Probe non-public-free episode pages; still only downloads images if public HTML exposes them.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload-manifest", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    args = parser.parse_args()

    run_id = args.run_id or dt.datetime.utcnow().strftime("tapas_%Y%m%d_%H%M%S")
    manifest_dir = Path(args.manifest_dir)
    selected = select_series(args.series)
    proxy_pool = ProxyPool.from_env()
    network_mode = choose_network_mode(proxy_pool, args.network_mode)
    http = HttpClient(proxy_pool=proxy_pool, network_mode=network_mode, timeout=45.0)
    client = s3_client()

    print(
        f"setup: run_id={run_id} series={len(selected)} network_mode={network_mode} "
        f"proxy_configured={proxy_pool.enabled} download={args.download} "
        f"episode_workers={args.episode_workers} image_workers={args.image_workers}",
        flush=True,
    )

    all_episodes: list[dict[str, Any]] = []
    for series in selected:
        episodes = discover_series(
            http,
            series,
            max_pages=max(1, args.max_pages),
            max_episodes=max(0, args.max_episodes_per_series),
        )
        public_count = sum(1 for episode in episodes if not episode.get("locked_hint"))
        print(
            f"discover: series={series.slug} episodes={len(episodes)} public_or_unlocked_hint={public_count}",
            flush=True,
        )
        all_episodes.extend(episodes)

    manifest_path = manifest_dir / f"{run_id}_episodes.jsonl"
    status_path = manifest_dir / f"{run_id}_status.jsonl"
    summary_path = manifest_dir / f"{run_id}_summary.json"
    write_jsonl(manifest_path, all_episodes)
    print(f"manifest: path={manifest_path} episodes={len(all_episodes)}", flush=True)

    status_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    last_progress = 0.0
    total = len(all_episodes)

    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.episode_workers)) as pool:
            future_map = {
                pool.submit(
                    download_episode,
                    http=http,
                    client=client,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    episode=episode,
                    image_workers=max(1, args.image_workers),
                    max_images=max(0, args.max_images_per_episode),
                    overwrite=bool(args.overwrite),
                    probe_locked=bool(args.probe_locked),
                ): episode
                for episode in all_episodes
            }
            for future in concurrent.futures.as_completed(future_map):
                episode = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "reason": str(exc),
                        "series": episode["series"]["slug"],
                        "episode_id": episode["episode_id"],
                        "episode_slug": episode["episode_slug"],
                        "title": episode["title"],
                        "image_count": 0,
                        "written": 0,
                        "skipped": 0,
                        "failures": 1,
                    }
                status_rows.append(row)
                now = time.monotonic()
                if now - last_progress >= args.progress_interval or len(status_rows) == total:
                    last_progress = now
                    counts = summarize_counts(status_rows)
                    elapsed = max(0.1, now - start)
                    rate = len(status_rows) / elapsed
                    remaining = max(0, total - len(status_rows))
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        "progress: "
                        f"episodes={len(status_rows)}/{total} accessible={counts['accessible']} "
                        f"locked_or_unavailable={counts['locked_or_unavailable']} "
                        f"images={counts['images']} written={counts['written']} skipped={counts['skipped']} "
                        f"failures={counts['failures']} episode_rate={rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )
    else:
        for episode in all_episodes:
            if episode.get("locked_hint"):
                status_rows.append(
                    {
                        "status": "locked_or_unavailable",
                        "reason": "metadata_not_public_free",
                        "series": episode["series"]["slug"],
                        "episode_id": episode["episode_id"],
                        "episode_slug": episode["episode_slug"],
                        "title": episode["title"],
                        "image_count": 0,
                        "written": 0,
                        "skipped": 0,
                        "failures": 0,
                    }
                )

    write_jsonl(status_path, status_rows)
    counts = summarize_counts(status_rows)
    summary = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "network_mode": network_mode,
        "bucket": args.bucket,
        "prefix": args.prefix.strip("/"),
        "series": [series.slug for series in selected],
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "download": bool(args.download),
        "counts": counts,
        "notes": [
            "Only official Tapas public/free episode images exposed in HTML were downloaded.",
            "WUF/Ink/login/locked episodes are recorded as locked_or_unavailable and require an authorized export path.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary: " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.upload_manifest:
        manifest_prefix = f"{args.prefix.strip('/')}/_manifests/{run_id}"
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/episodes.jsonl", path=manifest_path, content_type="application/x-jsonlines")
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/status.jsonl", path=status_path, content_type="application/x-jsonlines")
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/summary.json", path=summary_path, content_type="application/json")
        print(f"uploaded_manifest: s3://{args.bucket}/{manifest_prefix}/", flush=True)


if __name__ == "__main__":
    main()
