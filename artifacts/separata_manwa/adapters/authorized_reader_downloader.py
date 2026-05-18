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
from urllib.parse import urljoin, urlparse

import boto3
import requests
from bs4 import BeautifulSoup
from botocore.config import Config
from botocore.exceptions import ClientError
from requests.utils import requote_uri


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_PREFIX = "datasets/pages/single"
DEFAULT_MANIFEST_DIR = Path("artifacts/separata_manwa/manifests")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclasses.dataclass(frozen=True)
class SiteConfig:
    key: str
    title: str
    series_slug: str
    start_url: str
    discover: str
    link_patterns: tuple[str, ...]
    image_patterns: tuple[str, ...]
    chapter_list_url: str = ""
    category_id: str = ""

    @property
    def output_series(self) -> str:
        return f"{self.series_slug}_manwa"


SITES: tuple[SiteConfig, ...] = (
    SiteConfig(
        key="solo",
        title="Solo Leveling",
        series_slug="solo-leveling",
        start_url="https://www.read-solo10.com/",
        discover="links",
        link_patterns=(r"/manga/solo-leveling-chapter-[^/]+/?$",),
        image_patterns=(r"eu2\.contabostorage\.com/.*[:/]manga/Solo%20Manwha/Chapter%20", r"eu2\.contabostorage\.com/.*[:/]manga/Solo Manwha/Chapter "),
    ),
    SiteConfig(
        key="sss",
        title="SSS-Class Suicide Hunter",
        series_slug="sss-class-suicide-hunter",
        start_url="https://www.read-sss-hunter.com/",
        discover="links",
        link_patterns=(r"/manga/sss-class-suicide-hunter-chapter-[^/]+/?$",),
        image_patterns=(r"eu2\.contabostorage\.com/.*[:/]manga/SSS-Class%20Revival%20Hunter/Chapter%20", r"eu2\.contabostorage\.com/.*[:/]manga/SSS-Class Revival Hunter/Chapter "),
    ),
    SiteConfig(
        key="second",
        title="Second Life Ranker",
        series_slug="second-life-ranker",
        start_url="https://w77.secondliferanker.com/",
        discover="links",
        link_patterns=(r"/second-life-ranker-(?:manga-)?chapter-\d+/?$", r"/second-life-ranker-\d+-manga-chapter/?$"),
        image_patterns=(r"i\.imgur\.com/[^/]+\.(?:jpg|jpeg|png|webp)(?:\?.*)?$",),
    ),
    SiteConfig(
        key="returner",
        title="A Returner's Magic Should Be Special",
        series_slug="a-returners-magic-should-be-special",
        start_url="https://www.mangavf.fr/en/manga/a-returners-magic-should-be-special-english.html",
        discover="mangavf",
        link_patterns=(r"/en/a-returners-magic-should-be-special-english/a-returners-magic-should-be-special-english-chapter-[^/]+\.html$",),
        image_patterns=(r"cdn\.mangavf\.fr/cdn/Book-en/A%20Returner%27s%20Magic%20Should%20Be%20Special%20ENGLISH/Chapter%20",),
        category_id="88",
    ),
    SiteConfig(
        key="great",
        title="The Great Mage Returns After 4000 Years",
        series_slug="the-great-mage-returns-after-4000-years",
        start_url="https://w2.greatmagereturns.com/",
        discover="links",
        link_patterns=(r"/the-great-mage-returns-after-4000-years-chapter-[^/]+/?$",),
        image_patterns=(r"abyssrift\.com/GMR/[^/]+\.(?:jpg|jpeg|png|webp)(?:\?.*)?$",),
    ),
    SiteConfig(
        key="lout",
        title="Lout of Count's Family",
        series_slug="lout-of-counts-family",
        start_url="https://manhwazone.com/series/lout-of-counts-family-z7m8q",
        discover="manhwazone",
        link_patterns=(r"/preview/[^/?#]+$",),
        image_patterns=(r"scans\.lastation\.us/manga/Trash-of-the-Counts-Family/",),
    ),
)


@dataclasses.dataclass(frozen=True)
class Chapter:
    site_key: str
    series_slug: str
    output_series: str
    title: str
    url: str
    chapter_slug: str
    sort_key: float


class Http:
    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
            self.local.session = session
        return session

    def get(self, url: str, *, referer: str | None = None, accept: str = "*/*", stream: bool = False, retries: int = 6, timeout: float | None = None) -> requests.Response:
        headers = {"Accept": accept}
        if referer:
            headers["Referer"] = referer
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.session().get(requote_uri(url), headers=headers, timeout=timeout or self.timeout, stream=stream)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(15.0, 0.8 * attempt + random.random() * 0.8))
        raise RuntimeError(f"GET failed for {url}: {last_error}")

    def post_json(self, url: str, *, referer: str, payload: dict[str, Any], headers: dict[str, str] | None = None, retries: int = 6) -> dict[str, Any]:
        request_headers = {"Accept": "application/json", "Content-Type": "application/json", "Referer": referer}
        if headers:
            request_headers.update(headers)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.session().post(url, headers=request_headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(15.0, 0.8 * attempt + random.random() * 0.8))
        raise RuntimeError(f"POST failed for {url}: {last_error}")


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=512,
            connect_timeout=15,
            read_timeout=300,
            retries={"max_attempts": 12, "mode": "adaptive"},
        ),
    )


def s3_retry(operation: str, func, *args, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= 6:
                break
            time.sleep(min(20.0, 1.2 * attempt + random.random()))
    raise RuntimeError(f"S3 {operation} failed after retries: {last_error}")


def slugify(value: str) -> str:
    value = html.unescape(value).strip().lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9.]+", "-", value)
    return value.strip("-") or "chapter"


def chapter_number(text: str) -> float:
    text = html.unescape(text)
    patterns = [
        r"chapter[-\s_]*(\d+(?:[.-]\d+)?)",
        r"episode[-\s_]*(\d+(?:[.-]\d+)?)",
        r"ranker-(\d+(?:[.-]\d+)?)-manga-chapter",
        r"-(\d+(?:[.-]\d+)?)-manga-chapter",
        r"/(\d+(?:[.-]\d+)?)(?:/|\.html|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            raw = match.group(1).replace("-", ".")
            try:
                return float(raw)
            except ValueError:
                pass
    return 0.0


def chapter_slug(title: str, url: str, sort_key: float) -> str:
    if sort_key >= 0 and re.search(r"(?:chapter|episode|ranker)", f"{title} {url}", re.I):
        raw = ("%07.1f" % sort_key).replace(".", "-") if not sort_key.is_integer() else f"{int(sort_key):06d}"
        return f"chapter-{raw}"
    last = Path(urlparse(url).path.rstrip("/")).name
    if last:
        return slugify(last)
    return slugify(title)


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


def select_sites(names: list[str]) -> list[SiteConfig]:
    if not names:
        return list(SITES)
    by_key = {site.key: site for site in SITES}
    by_key.update({site.series_slug: site for site in SITES})
    selected: list[SiteConfig] = []
    missing: list[str] = []
    for name in names:
        key = slugify(name)
        site = by_key.get(name) or by_key.get(key)
        if site is None:
            missing.append(name)
        elif site not in selected:
            selected.append(site)
    if missing:
        raise ValueError(f"unknown site/series: {', '.join(missing)}")
    return selected


def link_allowed(site: SiteConfig, url: str) -> bool:
    parsed = urlparse(url)
    path_url = parsed.path
    for pattern in site.link_patterns:
        if re.search(pattern, path_url, re.I) or re.search(pattern, url, re.I):
            return True
    return False


def image_allowed(site: SiteConfig, url: str) -> bool:
    if not url or url.startswith("data:"):
        return False
    cleaned = url.replace(" ", "%20")
    for pattern in site.image_patterns:
        if re.search(pattern, cleaned, re.I) or re.search(pattern, url, re.I):
            return True
    return False


def chapter_from_link(site: SiteConfig, title: str, url: str) -> Chapter:
    sort = chapter_number(f"{title} {url}")
    return Chapter(
        site_key=site.key,
        series_slug=site.series_slug,
        output_series=site.output_series,
        title=title or Path(urlparse(url).path).name,
        url=url,
        chapter_slug=chapter_slug(title, url, sort),
        sort_key=sort,
    )


def discover_links(http: Http, site: SiteConfig) -> list[Chapter]:
    response = http.get(site.start_url, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    soup = BeautifulSoup(response.text, "html.parser")
    chapters: dict[str, Chapter] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(response.url, anchor["href"])
        if not link_allowed(site, url):
            continue
        title = " ".join(anchor.get_text(" ", strip=True).split())
        chapters.setdefault(url, chapter_from_link(site, title, url))
    return sorted(chapters.values(), key=lambda chapter: (chapter.sort_key, chapter.url))


def discover_mangavf(http: Http, site: SiteConfig) -> list[Chapter]:
    response = http.get(site.start_url, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    nonce_match = re.search(r'nonce":"([^"]+)', response.text)
    category_match = re.search(r'class=["\']chapters-list["\'][^>]*data-category=["\'](\d+)', response.text)
    nonce = nonce_match.group(1) if nonce_match else ""
    category_id = site.category_id or (category_match.group(1) if category_match else "")
    if not nonce or not category_id:
        return discover_links(http, site)
    chapters: dict[str, Chapter] = {}
    page = 1
    while page <= 100:
        payload = {
            "action": "mangaverse_load_more",
            "nonce": nonce,
            "page": page,
            "type": "series",
            "category_id": category_id,
            "order": "desc",
            "lang": "en",
        }
        data = http.session().post(
            "https://www.mangavf.fr/wp-admin/admin-ajax.php",
            headers={"Accept": "application/json", "Referer": site.start_url},
            data=payload,
            timeout=45.0,
        )
        data.raise_for_status()
        body = data.json()
        if not body.get("success"):
            break
        html_fragment = str((body.get("data") or {}).get("html") or "")
        soup = BeautifulSoup(html_fragment, "html.parser")
        for anchor in soup.find_all("a", href=True):
            url = urljoin(site.start_url, anchor["href"])
            if not link_allowed(site, url):
                continue
            title = " ".join(anchor.get_text(" ", strip=True).split())
            chapters.setdefault(url, chapter_from_link(site, title, url))
        if not (body.get("data") or {}).get("has_more"):
            break
        page += 1
    return sorted(chapters.values(), key=lambda chapter: (chapter.sort_key, chapter.url))


def unpack_livewire_chapters(snapshot: str) -> tuple[list[dict[str, Any]], bool]:
    data = json.loads(snapshot)["data"]
    raw = data.get("chapters") or []
    chapters_container = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    chapters: list[dict[str, Any]] = []
    for item in chapters_container:
        if isinstance(item, list) and item and isinstance(item[0], dict):
            chapters.append(item[0])
        elif isinstance(item, dict):
            chapters.append(item)
    return chapters, bool(data.get("hasMore"))


def discover_manhwazone(http: Http, site: SiteConfig) -> list[Chapter]:
    response = http.get(site.start_url, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    csrf_match = re.search(r'<meta name="csrf-token" content="([^"]+)"', response.text)
    snapshot_match = re.search(r'wire:snapshot="([^"]+)"', response.text)
    if not csrf_match or not snapshot_match:
        return discover_links(http, site)
    csrf = csrf_match.group(1)
    snapshot = html.unescape(snapshot_match.group(1))
    headers = {"X-Livewire": "", "X-CSRF-TOKEN": csrf}

    def call(method: str, current_snapshot: str) -> str:
        payload = {
            "_token": csrf,
            "components": [
                {
                    "snapshot": current_snapshot,
                    "updates": {},
                    "calls": [{"path": "", "method": method, "params": []}],
                }
            ],
        }
        data = http.post_json("https://manhwazone.com/livewire/update", referer=site.start_url, payload=payload, headers=headers)
        return str(data["components"][0]["snapshot"])

    snapshot = call("bootLoad", snapshot)
    chapters_by_url: dict[str, Chapter] = {}
    while True:
        rows, has_more = unpack_livewire_chapters(snapshot)
        for row in rows:
            url = urljoin(site.start_url, str(row.get("web_url") or ""))
            if not link_allowed(site, url):
                continue
            title = str(row.get("name") or "")
            sort = float(row.get("chapter_no") or chapter_number(title) or 0)
            chapters_by_url.setdefault(
                url,
                Chapter(
                    site_key=site.key,
                    series_slug=site.series_slug,
                    output_series=site.output_series,
                    title=title,
                    url=url,
                    chapter_slug=chapter_slug(title, url, sort),
                    sort_key=sort,
                ),
            )
        if not has_more:
            break
        snapshot = call("loadMore", snapshot)
    return sorted(chapters_by_url.values(), key=lambda chapter: (chapter.sort_key, chapter.url))


def discover_site(http: Http, site: SiteConfig, max_chapters: int) -> list[Chapter]:
    if site.discover == "mangavf":
        chapters = discover_mangavf(http, site)
    elif site.discover == "manhwazone":
        chapters = discover_manhwazone(http, site)
    else:
        chapters = discover_links(http, site)
    if max_chapters > 0:
        chapters = chapters[:max_chapters]
    return chapters


def srcset_urls(value: str) -> list[str]:
    urls: list[str] = []
    for part in value.split(","):
        token = part.strip().split(" ")[0].strip()
        if token:
            urls.append(token)
    return urls


def parse_image_urls(site: SiteConfig, page_html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(page_html, "html.parser")
    urls: list[str] = []
    for img in soup.find_all("img"):
        candidates: list[str] = []
        for attr in ("data-src", "data-lazy-src", "data-original", "data-cfsrc", "src"):
            value = img.get(attr)
            if value:
                candidates.append(value)
        for attr in ("data-srcset", "srcset"):
            value = img.get(attr)
            if value:
                candidates.extend(srcset_urls(value))
        for raw in candidates:
            url = urljoin(page_url, html.unescape(raw))
            if image_allowed(site, url):
                urls.append(url)
    regex_urls = re.findall(r"https?://[^\"'<>\\\s)]+", page_html)
    for raw in regex_urls:
        url = html.unescape(raw)
        if image_allowed(site, url):
            urls.append(url)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.replace(" ", "%20")
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def object_exists(client: Any, bucket: str, key: str) -> bool:
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < 6:
            time.sleep(min(20.0, 1.2 * attempt + random.random()))
    raise RuntimeError(f"S3 head_object failed after retries: {last_error}")


def key_for(prefix: str, chapter: Chapter, index: int, image_url: str, content_type: str | None = None) -> str:
    ext = extension_for_url(image_url, content_type)
    return f"{prefix.strip('/')}/{chapter.output_series}/{chapter.chapter_slug}/page-{index:04d}{ext}"


def download_image(
    *,
    http: Http,
    client: Any,
    bucket: str,
    prefix: str,
    chapter: Chapter,
    image_url: str,
    index: int,
    overwrite: bool,
) -> str:
    provisional_key = key_for(prefix, chapter, index, image_url)
    if not overwrite and object_exists(client, bucket, provisional_key):
        return "skipped"
    response = http.get(
        image_url,
        referer=chapter.url,
        accept="image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        retries=5,
        timeout=120.0,
    )
    content_type = response.headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        raise RuntimeError(f"unexpected content-type {content_type}")
    key = key_for(prefix, chapter, index, image_url, content_type)
    if key != provisional_key and not overwrite and object_exists(client, bucket, key):
        return "skipped"
    s3_retry(
        "put_object",
        client.put_object,
        Bucket=bucket,
        Key=key,
        Body=response.content,
        ContentType=content_type,
        Metadata={
            "source-site": chapter.site_key,
            "series": chapter.output_series,
            "chapter-url": chapter.url,
            "source-url": image_url,
            "page-index": str(index),
        },
    )
    return "written"


def download_chapter(
    *,
    http: Http,
    client: Any,
    bucket: str,
    prefix: str,
    site: SiteConfig,
    chapter: Chapter,
    image_workers: int,
    max_images: int,
    overwrite: bool,
) -> dict[str, Any]:
    response = http.get(chapter.url, referer=site.start_url, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    image_urls = parse_image_urls(site, response.text, response.url)
    if max_images > 0:
        image_urls = image_urls[:max_images]
    if not image_urls:
        return {
            "status": "no_images",
            "site": site.key,
            "series": chapter.output_series,
            "chapter_slug": chapter.chapter_slug,
            "title": chapter.title,
            "url": chapter.url,
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
        futures = {
            pool.submit(
                download_image,
                http=http,
                client=client,
                bucket=bucket,
                prefix=prefix,
                chapter=chapter,
                image_url=image_url,
                index=index,
                overwrite=overwrite,
            ): image_url
            for index, image_url in enumerate(image_urls, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
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
    return {
        "status": "accessible" if failures == 0 else "partial_failure",
        "site": site.key,
        "series": chapter.output_series,
        "chapter_slug": chapter.chapter_slug,
        "title": chapter.title,
        "url": chapter.url,
        "image_count": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "failure_samples": samples,
        "s3_prefix": f"s3://{bucket}/{prefix.strip('/')}/{chapter.output_series}/{chapter.chapter_slug}/",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_file(client: Any, *, bucket: str, key: str, path: Path, content_type: str) -> None:
    s3_retry("put_object", client.put_object, Bucket=bucket, Key=key, Body=path.read_bytes(), ContentType=content_type)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {"chapters": len(rows), "accessible": 0, "no_images": 0, "written": 0, "skipped": 0, "failures": 0, "images": 0}
    for row in rows:
        result["written"] += int(row.get("written") or 0)
        result["skipped"] += int(row.get("skipped") or 0)
        result["failures"] += int(row.get("failures") or 0)
        result["images"] += int(row.get("image_count") or 0)
        if row.get("status") == "accessible":
            result["accessible"] += 1
        elif row.get("status") == "no_images":
            result["no_images"] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download user-authorized reader site chapters into datasets/pages/single/*_manwa.")
    parser.add_argument("--series", action="append", default=[], help="Site key or series slug. Repeatable. Default: all configured sites.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-chapters-per-series", type=int, default=0)
    parser.add_argument("--max-images-per-chapter", type=int, default=0)
    parser.add_argument("--chapter-workers", type=int, default=12)
    parser.add_argument("--image-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload-manifest", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    args = parser.parse_args()

    run_id = args.run_id or dt.datetime.utcnow().strftime("authorized_readers_%Y%m%dT%H%M%SZ")
    manifest_dir = Path(args.manifest_dir)
    http = Http()
    client = s3_client()
    selected = select_sites(args.series)
    print(
        f"setup: run_id={run_id} sites={len(selected)} prefix=s3://{args.bucket}/{args.prefix.strip('/')} "
        f"download={args.download} chapter_workers={args.chapter_workers} image_workers={args.image_workers}",
        flush=True,
    )

    chapters: list[Chapter] = []
    site_by_key = {site.key: site for site in selected}
    discovery_failures: list[dict[str, str]] = []
    for site in selected:
        try:
            site_chapters = discover_site(http, site, max_chapters=max(0, args.max_chapters_per_series))
        except Exception as exc:
            discovery_failures.append({"site": site.key, "series": site.output_series, "error": str(exc)})
            site_chapters = []
            print(f"discover_failed: site={site.key} series={site.output_series} error={exc}", flush=True)
        print(f"discover: site={site.key} series={site.output_series} chapters={len(site_chapters)}", flush=True)
        chapters.extend(site_chapters)

    manifest_rows = [dataclasses.asdict(chapter) for chapter in chapters]
    manifest_path = manifest_dir / f"{run_id}_chapters.jsonl"
    status_path = manifest_dir / f"{run_id}_status.jsonl"
    summary_path = manifest_dir / f"{run_id}_summary.json"
    write_jsonl(manifest_path, manifest_rows)
    print(f"manifest: path={manifest_path} chapters={len(chapters)}", flush=True)

    status_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    last = 0.0
    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.chapter_workers)) as pool:
            future_map = {
                pool.submit(
                    download_chapter,
                    http=http,
                    client=client,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    site=site_by_key[chapter.site_key],
                    chapter=chapter,
                    image_workers=max(1, args.image_workers),
                    max_images=max(0, args.max_images_per_chapter),
                    overwrite=bool(args.overwrite),
                ): chapter
                for chapter in chapters
            }
            for future in concurrent.futures.as_completed(future_map):
                chapter = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "reason": str(exc),
                        "site": chapter.site_key,
                        "series": chapter.output_series,
                        "chapter_slug": chapter.chapter_slug,
                        "title": chapter.title,
                        "url": chapter.url,
                        "image_count": 0,
                        "written": 0,
                        "skipped": 0,
                        "failures": 1,
                    }
                status_rows.append(row)
                now = time.monotonic()
                if now - last >= args.progress_interval or len(status_rows) == len(chapters):
                    last = now
                    current = counts(status_rows)
                    elapsed = max(0.1, now - start)
                    rate = len(status_rows) / elapsed
                    eta = max(0, len(chapters) - len(status_rows)) / rate if rate > 0 else 0
                    print(
                        "progress: "
                        f"chapters={len(status_rows)}/{len(chapters)} accessible={current['accessible']} "
                        f"no_images={current['no_images']} images={current['images']} written={current['written']} "
                        f"skipped={current['skipped']} failures={current['failures']} "
                        f"chapter_rate={rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )

    write_jsonl(status_path, status_rows)
    summary = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bucket": args.bucket,
        "prefix": args.prefix.strip("/"),
        "series": [site.output_series for site in selected],
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "download": bool(args.download),
        "counts": counts(status_rows),
        "discovery_failures": discovery_failures,
        "notes": [
            "User represented these source domains as authorized for this ingestion.",
            "Output goes directly under datasets/pages/single/<series>_manwa/.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary: " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.upload_manifest:
        manifest_prefix = f"{args.prefix.strip('/')}/_manifests/{run_id}"
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/chapters.jsonl", path=manifest_path, content_type="application/x-jsonlines")
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/status.jsonl", path=status_path, content_type="application/x-jsonlines")
        upload_file(client, bucket=args.bucket, key=f"{manifest_prefix}/summary.json", path=summary_path, content_type="application/json")
        print(f"uploaded_manifest: s3://{args.bucket}/{manifest_prefix}/", flush=True)


if __name__ == "__main__":
    main()
