from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.request import ProxyHandler, Request, build_opener

import boto3
from botocore.config import Config


DEST_BUCKET = os.environ.get("DEST_BUCKET", "drawtoon")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "45"))
HTML_RETRIES = int(os.environ.get("HTML_RETRIES", "4"))
PROXY_MODE = os.environ.get("PROXY_MODE", "auto").strip().lower()
REQUEST_DELAY_SECONDS = float(os.environ.get("REQUEST_DELAY_SECONDS", "0"))

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
REJECT_PATH_PARTS = {
    "author",
    "category",
    "contact",
    "dmca",
    "genre",
    "privacy-policy",
    "read-free-marvel-comics-online",
    "tag",
    "wp-content",
}

S3 = boto3.client(
    "s3",
    config=Config(max_pool_connections=32, connect_timeout=5, read_timeout=45, retries={"max_attempts": 5, "mode": "adaptive"}),
)
_OPENERS: dict[str, Any] = {}


@dataclass
class Link:
    url: str
    text: str
    rel: str
    css_class: str


@dataclass
class PageImage:
    url: str
    alt: str
    width: int
    height: int
    css_class: str


class CatalogParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.links: list[Link] = []
        self.images: list[PageImage] = []
        self._anchor_stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name.lower(): value or "" for name, value in attrs}
        if tag.lower() == "a":
            self._anchor_stack.append(
                {
                    "href": attr.get("href", ""),
                    "rel": attr.get("rel", ""),
                    "class": attr.get("class", ""),
                    "text": [],
                }
            )
            return
        if tag.lower() != "img":
            return
        raw = attr.get("data-src") or attr.get("data-original") or attr.get("data-lazy-src") or attr.get("src")
        if not raw and attr.get("srcset"):
            raw = first_srcset_url(attr.get("srcset", ""))
        if not raw:
            return
        url = normalize_image_url(self.page_url, raw)
        if not url:
            return
        self.images.append(
            PageImage(
                url=url,
                alt=html.unescape(attr.get("alt", "")),
                width=parse_int(attr.get("width")),
                height=parse_int(attr.get("height")),
                css_class=attr.get("class", ""),
            )
        )

    def handle_data(self, data: str) -> None:
        if self._anchor_stack:
            self._anchor_stack[-1]["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._anchor_stack:
            return
        anchor = self._anchor_stack.pop()
        raw = str(anchor.get("href") or "")
        if not raw:
            return
        url = normalize_site_url(self.page_url, raw)
        if not url:
            return
        text = re.sub(r"\s+", " ", html.unescape(" ".join(anchor.get("text") or []))).strip()
        self.links.append(Link(url=url, text=text, rel=str(anchor.get("rel") or ""), css_class=str(anchor.get("class") or "")))


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


def fetch_text(url: str, *, referer: str | None, proxy_mode: str) -> tuple[str, str]:
    mode = (proxy_mode or PROXY_MODE or "auto").lower()
    attempts: list[str | None] = []
    if mode in {"direct", "auto"}:
        attempts.extend([None] * max(1, HTML_RETRIES))
    if mode in {"proxy", "auto"}:
        attempts.extend(proxy_for_attempt(i, url) for i in range(1, min(HTML_RETRIES, max(1, len(PROXIES))) + 1))
    if not attempts:
        attempts.append(None)
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    last_error: Exception | None = None
    for attempt, proxy_url in enumerate(attempts, start=1):
        try:
            request = Request(url, headers=headers)
            with opener_for(proxy_url).open(request, timeout=HTTP_TIMEOUT) as response:
                return response.read().decode("utf-8", errors="replace"), "proxy" if proxy_url else "direct"
        except HTTPError as exc:
            last_error = exc
            if exc.code in {404, 410}:
                raise
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = retry_after_seconds(retry_after, attempt)
            if attempt < len(attempts):
                time.sleep(sleep_seconds)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < len(attempts):
                time.sleep(min(10.0, 0.75 * attempt))
    raise RuntimeError(str(last_error or "fetch failed")[:500])


def retry_after_seconds(value: str | None, attempt: int) -> float:
    if value:
        try:
            return min(30.0, max(1.0, float(value)))
        except ValueError:
            pass
    return min(30.0, 1.5 * attempt)


def parse_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "untitled"


def normalize_site_url(base_url: str, raw: str) -> str:
    raw = html.unescape(raw).strip()
    if not raw or raw.startswith(("data:", "blob:", "mailto:", "tel:", "#")):
        return ""
    url = urljoin(base_url, raw)
    parsed = urlparse(url)
    if parsed.netloc.lower() != "readfreecomicsonline.com":
        return ""
    path = re.sub(r"/+", "/", parsed.path or "/")
    return urlunparse((parsed.scheme or "https", parsed.netloc.lower(), path, "", parsed.query, ""))


def first_srcset_url(srcset: str) -> str:
    first = srcset.split(",", 1)[0].strip()
    return first.split()[0] if first else ""


def normalize_image_url(page_url: str, raw: str) -> str:
    url = normalize_site_url(page_url, raw)
    if not url:
        return ""
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return ""
    if "/wp-content/uploads/" not in parsed.path:
        return ""
    return url


def parse_catalog(html_text: str, page_url: str) -> CatalogParser:
    parser = CatalogParser(page_url)
    parser.feed(html_text)
    return parser


def path_parts(url: str) -> list[str]:
    return [part for part in urlparse(url).path.strip("/").split("/") if part]


def issue_slug_from_url(url: str) -> str:
    parts = path_parts(url)
    return slugify(parts[-1] if parts else "")


def looks_like_issue_url(url: str, text: str) -> bool:
    parts = path_parts(url)
    if len(parts) != 1:
        return False
    slug = slugify(parts[0])
    if not slug or slug in REJECT_PATH_PARTS:
        return False
    if any(part in REJECT_PATH_PARTS for part in parts):
        return False
    if slug.startswith(("read-free-", "category-", "tag-", "genre-")):
        return False
    text_l = text.lower()
    return "issue" in slug or "issue" in text_l or bool(re.search(r"(?:^|-)(?:19|20)\d{2}$", slug))


def pagination_links(links: list[Link]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for link in links:
        parsed = urlparse(link.url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        rel = link.rel.lower()
        css = link.css_class.lower()
        text = link.text.strip().lower()
        is_page = "/page/" in path or "paged=" in query or "page=" in query
        is_next = "next" in rel or "next" in css or text in {"next", ">", "›", "»"}
        if not is_page and not is_next:
            continue
        if looks_like_issue_url(link.url, link.text):
            continue
        if link.url not in seen:
            seen.add(link.url)
            urls.append(link.url)
    return urls


def guess_series_slug(issue_slug: str) -> str:
    value = re.sub(r"-issue-\d+.*$", "", issue_slug)
    value = re.sub(r"-(?:19|20)\d{2}$", "", value)
    return slugify(value or issue_slug)


def shard_key(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix.rstrip('/')}/{slugify(value)[:90]}-{digest}.jsonl"


def write_jsonl(bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    S3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/x-ndjson")


def discover_issues(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    seed_url = str(row["seed_url"])
    publisher = slugify(str(row.get("publisher") or "unknown"))
    seed_slug = slugify(str(row.get("seed_slug") or publisher))
    proxy_mode = str(config.get("proxy_mode") or PROXY_MODE or "auto")
    max_listing_pages = max(1, int(config.get("max_listing_pages") or row.get("max_listing_pages") or 200))
    bucket = str(config.get("bucket") or DEST_BUCKET)
    issue_shard_prefix = str(config["issue_shard_prefix"]).strip("/")

    queue = [normalize_site_url(seed_url, seed_url)]
    seen_pages: set[str] = set()
    issue_rows: dict[str, dict[str, Any]] = {}
    transports: dict[str, int] = {}
    while queue and len(seen_pages) < max_listing_pages:
        page_url = queue.pop(0)
        if not page_url or page_url in seen_pages:
            continue
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        html_text, transport = fetch_text(page_url, referer=seed_url, proxy_mode=proxy_mode)
        transports[transport] = transports.get(transport, 0) + 1
        seen_pages.add(page_url)
        parsed = parse_catalog(html_text, page_url)
        for link in parsed.links:
            if not looks_like_issue_url(link.url, link.text):
                continue
            issue_slug = issue_slug_from_url(link.url)
            issue_rows[link.url] = {
                "task_type": "rfco_issue_page",
                "publisher": publisher,
                "series_slug": guess_series_slug(issue_slug),
                "issue_slug": issue_slug,
                "title": link.text or issue_slug.replace("-", " ").title(),
                "url": link.url,
                "source_seed_url": seed_url,
                "source_seed_slug": seed_slug,
            }
        for next_url in pagination_links(parsed.links):
            if next_url not in seen_pages and next_url not in queue:
                queue.append(next_url)

    rows = sorted(issue_rows.values(), key=lambda item: (str(item.get("publisher")), str(item.get("issue_slug")), str(item.get("url"))))
    key = shard_key(issue_shard_prefix, seed_slug)
    write_jsonl(bucket, key, rows)
    return {
        "status": "written",
        "task_type": "discover_issues",
        "bucket": bucket,
        "key": key,
        "issue_count": len(rows),
        "listing_pages": len(seen_pages),
        "transport_counts": transports,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def discover_listing_page(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    listing_url = str(row["listing_url"])
    publisher = slugify(str(row.get("publisher") or "unknown"))
    seed_slug = slugify(str(row.get("seed_slug") or publisher))
    proxy_mode = str(config.get("proxy_mode") or PROXY_MODE or "auto")
    bucket = str(config.get("bucket") or DEST_BUCKET)
    issue_shard_prefix = str(config["issue_shard_prefix"]).strip("/")

    try:
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        html_text, transport = fetch_text(listing_url, referer=str(row.get("seed_url") or listing_url), proxy_mode=proxy_mode)
    except HTTPError as exc:
        if exc.code in {404, 410}:
            return {
                "status": "not_found",
                "task_type": "discover_listing_page",
                "listing_url": listing_url,
                "page_index": row.get("page_index"),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        raise

    parsed = parse_catalog(html_text, listing_url)
    issue_rows: dict[str, dict[str, Any]] = {}
    for link in parsed.links:
        if not looks_like_issue_url(link.url, link.text):
            continue
        issue_slug = issue_slug_from_url(link.url)
        issue_rows[link.url] = {
            "task_type": "rfco_issue_page",
            "publisher": publisher,
            "series_slug": guess_series_slug(issue_slug),
            "issue_slug": issue_slug,
            "title": link.text or issue_slug.replace("-", " ").title(),
            "url": link.url,
            "source_seed_url": str(row.get("seed_url") or ""),
            "source_seed_slug": seed_slug,
            "source_listing_url": listing_url,
            "source_listing_page_index": row.get("page_index"),
        }
    rows = sorted(issue_rows.values(), key=lambda item: (str(item.get("publisher")), str(item.get("issue_slug")), str(item.get("url"))))
    key = shard_key(issue_shard_prefix, f"{seed_slug}-page-{row.get('page_index') or 'unknown'}")
    write_jsonl(bucket, key, rows)
    return {
        "status": "written",
        "task_type": "discover_listing_page",
        "bucket": bucket,
        "key": key,
        "issue_count": len(rows),
        "listing_url": listing_url,
        "page_index": row.get("page_index"),
        "transport": transport,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def extract_page_images(html_text: str, page_url: str, min_width: int, min_height: int) -> list[PageImage]:
    parser = parse_catalog(html_text, page_url)
    images: list[PageImage] = []
    seen: set[str] = set()
    for image in parser.images:
        if image.url in seen:
            continue
        if image.width and image.width < min_width:
            continue
        if image.height and image.height < min_height:
            continue
        css = image.css_class.lower()
        if css and "size-full" not in css and "wp-image" not in css:
            continue
        seen.add(image.url)
        images.append(image)
    return images


def build_page_manifest(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    issue_url = str(row["url"])
    publisher = slugify(str(row.get("publisher") or "unknown"))
    issue_slug = slugify(str(row.get("issue_slug") or issue_slug_from_url(issue_url)))
    series_slug = slugify(str(row.get("series_slug") or guess_series_slug(issue_slug)))
    title = str(row.get("title") or issue_slug.replace("-", " ").title())
    proxy_mode = str(config.get("proxy_mode") or PROXY_MODE or "auto")
    min_width = int(config.get("min_width") or 500)
    min_height = int(config.get("min_height") or 500)
    bucket = str(config.get("bucket") or DEST_BUCKET)
    page_shard_prefix = str(config["page_shard_prefix"]).strip("/")
    output_root = str(config.get("output_root") or "datasets/pages/single").strip("/")
    authorization_ref = str(config.get("authorization_ref") or "")
    approval_period = str(config.get("approval_period") or "")
    accessed_at = str(config.get("accessed_at") or "")

    if REQUEST_DELAY_SECONDS > 0:
        time.sleep(REQUEST_DELAY_SECONDS)
    html_text, transport = fetch_text(issue_url, referer=str(row.get("source_seed_url") or issue_url), proxy_mode=proxy_mode)
    images = extract_page_images(html_text, issue_url, min_width, min_height)
    output_prefix = f"{output_root}/{publisher}_{issue_slug}_comic"
    rows: list[dict[str, Any]] = []
    for page_no, image in enumerate(images, start=1):
        rows.append(
            {
                "source_type": "authorized_third_party_public_browser_flow",
                "platform": "readfreecomicsonline",
                "series_slug": series_slug,
                "issue_slug": issue_slug,
                "page_no": page_no,
                "image_url": image.url,
                "referer": issue_url,
                "headers_profile": "public_browser_flow",
                "output_prefix": output_prefix,
                "metadata": {
                    "accessed_at": accessed_at,
                    "authorization_ref": authorization_ref,
                    "authorization_window": approval_period,
                    "issue_title": title,
                    "publisher": publisher,
                    "source_page_url": issue_url,
                    "source_seed_url": str(row.get("source_seed_url") or ""),
                    "source_site": "readfreecomicsonline.com",
                },
            }
        )
    key = shard_key(page_shard_prefix, f"{publisher}-{issue_slug}")
    write_jsonl(bucket, key, rows)
    return {
        "status": "written",
        "task_type": "build_page_manifest",
        "bucket": bucket,
        "key": key,
        "issue_slug": issue_slug,
        "page_count": len(rows),
        "transport": transport,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config = event.get("config") or {}
    row = event.get("row") or event.get("item") or event
    task_type = str(row.get("task_type") or config.get("task_type") or "").strip()
    try:
        if task_type == "discover_listing_page":
            return discover_listing_page(row, config)
        if task_type == "discover_issues":
            return discover_issues(row, config)
        if task_type == "rfco_issue_page":
            return build_page_manifest(row, config)
        raise ValueError(f"unsupported task_type={task_type!r}")
    except Exception as exc:
        return {
            "status": "failed",
            "task_type": task_type,
            "error": str(exc)[:500],
            "url": str(row.get("url") or row.get("seed_url") or ""),
        }
