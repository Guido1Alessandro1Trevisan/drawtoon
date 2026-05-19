#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adapters.base import ManifestRow, slugify, write_jsonl
else:
    from .base import ManifestRow, slugify, write_jsonl


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


@dataclass
class PageImage:
    url: str
    alt: str
    width: int
    height: int
    css_class: str


class ImageParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.images: list[PageImage] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr = {name.lower(): value or "" for name, value in attrs}
        raw = attr.get("data-src") or attr.get("data-original") or attr.get("data-lazy-src") or attr.get("src")
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


def parse_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def normalize_image_url(page_url: str, raw: str) -> str:
    raw = html.unescape(raw).strip()
    if raw.startswith(("data:", "blob:")):
        return ""
    url = urljoin(page_url, raw)
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return ""
    if parsed.netloc != "readfreecomicsonline.com":
        return ""
    if "/wp-content/uploads/" not in parsed.path:
        return ""
    return url


def fetch_page(url: str, timeout: float) -> str:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_page_images(html_text: str, page_url: str, min_width: int, min_height: int) -> list[PageImage]:
    parser = ImageParser(page_url)
    parser.feed(html_text)
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


def build_rows(args: argparse.Namespace) -> tuple[list[ManifestRow], dict[str, Any]]:
    config = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    authorization_ref = str(config["authorization_ref"])
    output_root = str(args.output_root or config.get("output_root") or "datasets/pages/single/marvel_dc_authorized").strip("/")
    output_prefix_template = str(config.get("output_prefix_template") or "{output_root}/{publisher}_{issue_slug}_comic")
    source_type = str(config.get("source_type") or "authorized_third_party_public_browser_flow")
    platform = str(config.get("platform") or "readfreecomicsonline")
    accessed_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    rows: list[ManifestRow] = []
    issue_summaries: list[dict[str, Any]] = []
    for index, issue in enumerate(config.get("issues") or [], start=1):
        issue_url = str(issue["url"])
        publisher = slugify(str(issue.get("publisher") or "unknown"))
        issue_slug = str(issue.get("issue_slug") or slugify(urlparse(issue_url).path.strip("/")))
        series_slug = str(issue.get("series_slug") or publisher)
        title = str(issue.get("title") or issue_slug)

        html_text = fetch_page(issue_url, args.http_timeout)
        images = extract_page_images(html_text, issue_url, args.min_width, args.min_height)
        output_prefix = output_prefix_template.format(
            output_root=output_root,
            publisher=publisher,
            issue_slug=issue_slug,
            series_slug=series_slug,
        ).strip("/")
        for page_no, image in enumerate(images, start=1):
            rows.append(
                ManifestRow(
                    source_type=source_type,
                    platform=platform,
                    series_slug=series_slug,
                    issue_slug=issue_slug,
                    page_no=page_no,
                    image_url=image.url,
                    referer=issue_url,
                    headers_profile="public_browser_flow",
                    output_prefix=output_prefix,
                    metadata={
                        "accessed_at": accessed_at,
                        "authorization_ref": authorization_ref,
                        "authorization_window": config.get("approval_period", ""),
                        "issue_title": title,
                        "publisher": publisher,
                        "source_page_url": issue_url,
                        "source_site": "readfreecomicsonline.com",
                    },
                )
            )
        issue_summaries.append(
            {
                "index": index,
                "publisher": publisher,
                "title": title,
                "url": issue_url,
                "issue_slug": issue_slug,
                "pages": len(images),
                "output_prefix": output_prefix,
            }
        )
        print(f"{publisher}/{issue_slug}: {len(images)} pages", flush=True)
        if args.sleep_seconds and index < len(config.get("issues") or []):
            time.sleep(args.sleep_seconds)

    report = {
        "authorization_ref": authorization_ref,
        "approval_period": config.get("approval_period", ""),
        "accessed_at": accessed_at,
        "source_site": "readfreecomicsonline.com",
        "manifest_path": args.output,
        "issue_count": len(issue_summaries),
        "page_count": len(rows),
        "issues": issue_summaries,
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest from authorized readfreecomicsonline public issue pages.")
    parser.add_argument("--sources", required=True, help="JSON source list with authorized issue URLs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--http-timeout", type=float, default=45.0)
    parser.add_argument("--min-width", type=int, default=500)
    parser.add_argument("--min-height", type=int, default=500)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    rows, report = build_rows(args)
    count = write_jsonl(Path(args.output), rows)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={args.output}", flush=True)
    print(f"rows={count}", flush=True)
    if args.report:
        print(f"report={args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
