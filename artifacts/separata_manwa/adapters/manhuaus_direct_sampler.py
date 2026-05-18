#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import mimetypes
import os
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_PREFIX = "datasets/pages/single"
DEFAULT_MANIFEST_DIR = Path("artifacts/separata_manwa/manifests")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_EXTENSIONS = ("jpg", "webp", "png", "jpeg")


@dataclasses.dataclass(frozen=True)
class TitleConfig:
    key: str
    title: str
    source_slug: str
    output_slug: str
    max_chapter: int
    first_chapter: int = 0

    @property
    def output_series(self) -> str:
        return f"{self.output_slug}_manhua"


TITLES: tuple[TitleConfig, ...] = (
    TitleConfig("evil-god", "I'm an Evil God", "im-an-evil-god-1", "im-an-evil-god", 600, 0),
    TitleConfig("magic-emperor", "Magic Emperor", "magic-emperor-0", "magic-emperor", 840, 1),
    TitleConfig(
        "big-villains",
        "My Disciples Are All Big Villains",
        "my-disciples-are-all-big-villains",
        "my-disciples-are-all-big-villains",
        500,
        0,
    ),
    TitleConfig(
        "three-thousand-years",
        "My Three Thousand Years To The Sky",
        "my-three-thousand-years-to-the-sky",
        "my-three-thousand-years-to-the-sky",
        380,
        1,
    ),
    TitleConfig(
        "top-tier-providence",
        "Top Tier Providence: Secretly Cultivate for a Thousand Years",
        "top-tier-providence-secretly-cultivate-for-a-thousand-years",
        "top-tier-providence-secretly-cultivate-for-a-thousand-years",
        280,
        0,
    ),
    TitleConfig("eternal-supreme", "The Eternal Supreme", "the-eternal-supreme", "the-eternal-supreme", 540, 1),
    TitleConfig("sect-leader", "All Hail the Sect Leader", "all-hail-the-sect-leader", "all-hail-the-sect-leader", 510, 1),
)


@dataclasses.dataclass(frozen=True)
class ChapterCandidate:
    title: TitleConfig
    chapter_no: int
    ext: str

    @property
    def chapter_slug(self) -> str:
        return f"chapter-{self.chapter_no:06d}"

    @property
    def source_chapter_slug(self) -> str:
        return f"chapter-{self.chapter_no}"

    @property
    def referer(self) -> str:
        return f"https://manhuaus.com/manga/{self.title.source_slug}/{self.source_chapter_slug}/"


@dataclasses.dataclass(frozen=True)
class ImageCandidate:
    chapter: ChapterCandidate
    page_no: int
    ext: str
    url: str


@dataclasses.dataclass
class ImageResult:
    status: str
    key: str = ""
    width: int = 0
    height: int = 0
    reason: str = ""
    error: str = ""
    bytes_written: int = 0


class Http:
    def __init__(self, timeout: float = 45.0) -> None:
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

    def head_image(self, url: str, referer: str, timeout: float | None = None) -> requests.Response | None:
        headers = {
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": referer,
        }
        try:
            response = self.session().head(url, headers=headers, timeout=timeout or self.timeout, allow_redirects=True)
            if response.status_code == 405:
                response = self.session().get(url, headers={**headers, "Range": "bytes=0-0"}, timeout=timeout or self.timeout)
            return response
        except requests.RequestException:
            return None

    def get_image(self, url: str, referer: str, retries: int = 4, timeout: float | None = None) -> requests.Response:
        headers = {
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": referer,
        }
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.session().get(url, headers=headers, timeout=timeout or self.timeout)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(10.0, 0.6 * attempt + random.random()))
        raise RuntimeError(f"GET failed for {url}: {last_error}")


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=512,
            connect_timeout=8,
            read_timeout=120,
            retries={"max_attempts": 8, "mode": "adaptive"},
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
            time.sleep(min(20.0, 0.8 * attempt + random.random()))
    raise RuntimeError(f"S3 {operation} failed after retries: {last_error}")


def select_titles(names: list[str]) -> list[TitleConfig]:
    if not names:
        return list(TITLES)
    by_name = {title.key: title for title in TITLES}
    by_name.update({title.source_slug: title for title in TITLES})
    by_name.update({title.output_slug: title for title in TITLES})
    selected: list[TitleConfig] = []
    missing: list[str] = []
    for raw in names:
        key = raw.strip().lower()
        title = by_name.get(key)
        if title is None:
            missing.append(raw)
        elif title not in selected:
            selected.append(title)
    if missing:
        raise ValueError(f"unknown title(s): {', '.join(missing)}")
    return selected


def image_url(title: TitleConfig, chapter_no: int, page_no: int, ext: str) -> str:
    return f"https://img.manhuaus.com/{title.source_slug}/chapter-{chapter_no}/{page_no:03d}.{ext}"


def is_image_response(response: requests.Response | None) -> bool:
    if response is None or response.status_code != 200:
        return False
    content_type = response.headers.get("content-type", "")
    return content_type.startswith("image/")


def discover_chapter(http: Http, title: TitleConfig, chapter_no: int) -> ChapterCandidate | None:
    referer = f"https://manhuaus.com/manga/{title.source_slug}/chapter-{chapter_no}/"
    for ext in IMAGE_EXTENSIONS:
        url = image_url(title, chapter_no, 1, ext)
        if is_image_response(http.head_image(url, referer, timeout=12.0)):
            return ChapterCandidate(title=title, chapter_no=chapter_no, ext=ext)
    return None


def discover_chapters(http: Http, title: TitleConfig, workers: int) -> list[ChapterCandidate]:
    chapter_numbers = list(range(title.first_chapter, title.max_chapter + 1))
    chapters: list[ChapterCandidate] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(discover_chapter, http, title, chapter_no): chapter_no for chapter_no in chapter_numbers}
        for future in concurrent.futures.as_completed(futures):
            chapter = future.result()
            if chapter is not None:
                chapters.append(chapter)
    return sorted(chapters, key=lambda item: item.chapter_no)


def discover_pages(http: Http, chapter: ChapterCandidate, max_pages: int, consecutive_misses: int) -> list[ImageCandidate]:
    pages: list[ImageCandidate] = []
    misses = 0
    for page_no in range(1, max_pages + 1):
        found: ImageCandidate | None = None
        for ext in (chapter.ext, *[ext for ext in IMAGE_EXTENSIONS if ext != chapter.ext]):
            url = image_url(chapter.title, chapter.chapter_no, page_no, ext)
            if is_image_response(http.head_image(url, chapter.referer, timeout=12.0)):
                found = ImageCandidate(chapter=chapter, page_no=page_no, ext=ext, url=url)
                break
        if found is None:
            misses += 1
            if pages and misses >= consecutive_misses:
                break
            continue
        misses = 0
        pages.append(found)
    return pages


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


def dimension_reason(width: int, height: int, args: argparse.Namespace) -> str:
    ratio = float(height) / max(1.0, float(width))
    if width < args.min_width:
        return "too_narrow"
    if height < args.min_height:
        return "too_short"
    if ratio < args.min_ratio:
        return "ratio_low"
    if ratio > args.max_ratio:
        return "ratio_high"
    return "keep"


def output_key(prefix: str, image: ImageCandidate) -> str:
    ext = "jpg" if image.ext == "jpeg" else image.ext
    return (
        f"{prefix.strip('/')}/{image.chapter.title.output_series}/"
        f"{image.chapter.chapter_slug}/page-{image.page_no:04d}.{ext}"
    )


def download_filter_upload(http: Http, s3: Any, bucket: str, prefix: str, image: ImageCandidate, args: argparse.Namespace) -> ImageResult:
    key = output_key(prefix, image)
    if not args.overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return ImageResult(status="skipped", key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
    response = http.get_image(image.url, image.chapter.referer, timeout=args.image_timeout)
    content_type = response.headers.get("content-type") or mimetypes.guess_type(image.url)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        return ImageResult(status="dropped", key=key, reason=f"content_type:{content_type}")
    body = response.content
    width, height = parse_image_dimensions(body[: min(len(body), 2_000_000)])
    reason = dimension_reason(width, height, args)
    if reason != "keep":
        return ImageResult(status="dropped", key=key, width=width, height=height, reason=reason)
    s3_retry(
        "put_object",
        s3.put_object,
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={
            "source-site": "manhuaus",
            "series": image.chapter.title.output_series,
            "chapter-url": image.chapter.referer,
            "source-url": image.url,
            "page-index": str(image.page_no),
            "width": str(width),
            "height": str(height),
        },
    )
    return ImageResult(status="written", key=key, width=width, height=height, bytes_written=len(body))


def run_title(title: TitleConfig, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    http = Http(timeout=args.http_timeout)
    s3 = s3_client()
    chapters = discover_chapters(http, title, args.discovery_workers)
    rng = random.Random(f"{args.seed}:{title.key}")
    sampled = list(chapters)
    rng.shuffle(sampled)
    written = 0
    skipped = 0
    dropped = 0
    failures = 0
    bytes_written = 0
    attempted_pages = 0
    chapter_summaries: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    failure_samples: list[str] = []
    used_chapters = 0

    for chapter in sampled:
        if written >= args.max_pages_per_title:
            break
        pages = discover_pages(http, chapter, args.max_pages_per_chapter, args.consecutive_misses)
        if not pages:
            continue
        remaining = args.max_pages_per_title - written
        selected_pages = pages[:remaining]
        attempted_pages += len(selected_pages)
        chapter_written = 0
        chapter_skipped = 0
        chapter_dropped = 0
        chapter_failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.image_workers)) as pool:
            futures = {
                pool.submit(download_filter_upload, http, s3, args.bucket, args.prefix, image, args): image
                for image in selected_pages
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    chapter_failures += 1
                    if len(failure_samples) < 10:
                        failure_samples.append(str(exc)[:500])
                    continue
                if result.status == "written":
                    written += 1
                    chapter_written += 1
                    bytes_written += result.bytes_written
                elif result.status == "skipped":
                    skipped += 1
                    chapter_skipped += 1
                else:
                    dropped += 1
                    chapter_dropped += 1
                    drop_reasons[result.reason or "unknown"] += 1
        used_chapters += 1
        chapter_summaries.append(
            {
                "chapter": chapter.chapter_no,
                "chapter_slug": chapter.chapter_slug,
                "discovered_pages": len(pages),
                "attempted_pages": len(selected_pages),
                "written": chapter_written,
                "skipped": chapter_skipped,
                "dropped": chapter_dropped,
                "failures": chapter_failures,
            }
        )
        print(
            json.dumps(
                {
                    "event": "title_progress",
                    "series": title.output_series,
                    "written": written,
                    "max_pages": args.max_pages_per_title,
                    "used_chapters": used_chapters,
                    "available_chapters": len(chapters),
                    "last_chapter": chapter.chapter_no,
                    "dropped": dropped,
                    "failures": failures,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "source": "manhuaus",
        "title": title.title,
        "source_slug": title.source_slug,
        "output_series": title.output_series,
        "s3_prefix": f"s3://{args.bucket}/{args.prefix.strip('/')}/{title.output_series}/",
        "available_chapters": len(chapters),
        "used_chapters": used_chapters,
        "attempted_pages": attempted_pages,
        "written": written,
        "skipped": skipped,
        "dropped": dropped,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "failures": failures,
        "failure_samples": failure_samples,
        "bytes_written": bytes_written,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "chapters": chapter_summaries,
    }
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample ManhuaUS CDN pages directly into datasets/pages/single/*_manhua.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--title", action="append", default=[])
    parser.add_argument("--max-pages-per-title", type=int, default=5000)
    parser.add_argument("--max-pages-per-chapter", type=int, default=180)
    parser.add_argument("--consecutive-misses", type=int, default=6)
    parser.add_argument("--seed", default="20260518-manhuaus")
    parser.add_argument("--title-workers", type=int, default=7)
    parser.add_argument("--discovery-workers", type=int, default=48)
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--http-timeout", type=float, default=45.0)
    parser.add_argument("--image-timeout", type=float, default=90.0)
    parser.add_argument("--min-width", type=int, default=450)
    parser.add_argument("--min-height", type=int, default=500)
    parser.add_argument("--min-ratio", type=float, default=0.55)
    parser.add_argument("--max-ratio", type=float, default=40.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    run_id = args.run_id or "manhuaus_direct_" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    titles = select_titles(args.title)
    summaries: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.title_workers)) as pool:
        futures = {pool.submit(run_title, title, args): title for title in titles}
        for future in concurrent.futures.as_completed(futures):
            title = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "source": "manhuaus",
                    "title": title.title,
                    "source_slug": title.source_slug,
                    "output_series": title.output_series,
                    "status": "failed",
                    "error": str(exc)[:1000],
                }
            summaries.append(summary)
            print(json.dumps({"event": "title_done", **summary}, ensure_ascii=False, sort_keys=True), flush=True)

    manifest_path = DEFAULT_MANIFEST_DIR / f"{run_id}_summary.jsonl"
    write_jsonl(manifest_path, sorted(summaries, key=lambda row: str(row.get("output_series"))))
    total_written = sum(int(row.get("written") or 0) for row in summaries)
    total_failures = sum(int(row.get("failures") or 0) for row in summaries)
    print(
        json.dumps(
            {
                "event": "run_done",
                "run_id": run_id,
                "summary": str(manifest_path),
                "titles": len(summaries),
                "total_written": total_written,
                "total_failures": total_failures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
