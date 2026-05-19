#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adapters.base import ManifestRow, write_jsonl
else:
    from .base import ManifestRow, write_jsonl


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def is_image_url(url: str, mime_type: str = "") -> bool:
    if not url or url.startswith(("data:", "blob:")):
        return False
    if mime_type.lower().split(";", 1)[0].startswith("image/"):
        return True
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    guessed = mimetypes.guess_type(url)[0] or ""
    return guessed.startswith("image/")


def header_value(headers: list[dict[str, Any]], name: str) -> str:
    wanted = name.lower()
    for header in headers or []:
        if str(header.get("name", "")).lower() == wanted:
            return str(header.get("value") or "")
    return ""


def load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    entries = (((payload or {}).get("log") or {}).get("entries") or [])
    requests: list[dict[str, Any]] = []
    for entry in entries:
        request = entry.get("request") or {}
        response = entry.get("response") or {}
        content = response.get("content") or {}
        requests.append(
            {
                "url": request.get("url"),
                "method": request.get("method"),
                "request_headers": request.get("headers") or [],
                "mime_type": content.get("mimeType") or header_value(response.get("headers") or [], "content-type"),
                "status": response.get("status"),
            }
        )
    return requests


def build_rows(args: argparse.Namespace) -> list[ManifestRow]:
    requests = load_requests(Path(args.input))
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    output_prefix = args.output_prefix or f"{args.output_root.strip('/')}/{args.series_slug}/{args.issue_slug}"
    for request in requests:
        url = str(request.get("url") or "")
        if url in seen:
            continue
        if args.url_contains and args.url_contains not in url:
            continue
        if not is_image_url(url, str(request.get("mime_type") or "")):
            continue
        referer = args.referer or header_value(request.get("request_headers") or [], "referer")
        if args.referer_contains and args.referer_contains not in referer:
            continue
        seen.add(url)
        rows.append(
            ManifestRow(
                source_type=args.source_type,
                platform=args.platform,
                series_slug=args.series_slug,
                issue_slug=args.issue_slug,
                page_no=args.start_page_no + len(rows),
                image_url=url,
                referer=referer,
                headers_profile=args.headers_profile,
                output_prefix=output_prefix,
                metadata={
                    "authorization_ref": args.authorization_ref,
                    "source_manifest": str(Path(args.input).name),
                },
            )
        )
        if args.max_pages and len(rows) >= args.max_pages:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CDN image manifest from an authorized browser HAR/network export.")
    parser.add_argument("--input", required=True, help="HAR JSON or JSON list of request records.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default="dc")
    parser.add_argument("--source-type", default="official_rights_holder_cdn_manifest")
    parser.add_argument("--series-slug", required=True)
    parser.add_argument("--issue-slug", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--headers-profile", default="dc_browser_session")
    parser.add_argument("--referer", default="")
    parser.add_argument("--referer-contains", default="")
    parser.add_argument("--url-contains", default="")
    parser.add_argument("--output-root", default="datasets/pages/single/dc")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--start-page-no", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(args)
    count = write_jsonl(Path(args.output), rows)
    print(f"manifest={args.output}", flush=True)
    print(f"rows={count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

