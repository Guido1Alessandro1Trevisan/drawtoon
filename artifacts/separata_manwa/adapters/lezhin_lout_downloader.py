#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_PREFIX = "datasets/pages/source/lezhin"
DEFAULT_MANIFEST_DIR = Path("artifacts/separata_manwa/manifests")
BASE_URL = "https://www.lezhinus.com/en/comic/lout_count"
SERIES_SLUG = "lout-of-counts-family"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def s3_client() -> Any:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=128,
            connect_timeout=15,
            read_timeout=180,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )


def session() -> requests.Session:
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return sess


def get_text(sess: requests.Session, url: str, *, referer: str | None = None) -> str:
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = sess.get(url, headers=headers, timeout=45.0)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt >= 4:
                break
            time.sleep(min(8.0, 0.6 * attempt + random.random() * 0.5))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


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


def discover_episodes(sess: requests.Session, *, max_episodes: int) -> list[dict[str, Any]]:
    text = get_text(sess, BASE_URL)
    seen: set[str] = set()
    names: list[str] = []
    for match in re.finditer(r"""href=["']([^"']*/en/comic/lout_count/([^"']+))["']""", text):
        name = html.unescape(match.group(2)).strip("/")
        if not re.fullmatch(r"(?:p\d+|n\d+|\d+)", name):
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    names.sort(key=lambda value: (-1000000 + int(value[1:]) if value.startswith("p") else int(value) if value.isdigit() else 1000000 + int(value[1:])))
    if max_episodes:
        names = names[:max_episodes]
    rows = []
    for position, name in enumerate(names, start=1):
        label = "Prologue" if name.startswith("p") else f"Episode {name}"
        rows.append(
            {
                "platform": "lezhin",
                "series": {
                    "slug": SERIES_SLUG,
                    "official_title": "Lout of Count's Family",
                    "source_url": BASE_URL,
                },
                "position": position,
                "episode_name": name,
                "episode_url": f"{BASE_URL}/{name}",
                "episode_slug": f"episode-{position:04d}-{name}",
                "title": label,
            }
        )
    return rows


def regex_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_reader(text: str, episode: dict[str, Any]) -> tuple[list[str], str]:
    paths = re.findall(r"""path\\":\\"([^\\"]*/contents/scrolls/\d+)""", text)
    paths = list(dict.fromkeys(path.replace("\\/", "/") for path in paths))
    policy = regex_first(r"""Policy\\":\\"([^\\"]+)""", text)
    signature = regex_first(r"""Signature\\":\\"([^\\"]+)""", text)
    key_pair_id = regex_first(r"""Key-Pair-Id\\":\\"([^\\"]+)""", text)
    if not paths or not (policy and signature and key_pair_id):
        reason = "no_public_reader_images"
        if "LOGIN_REQUIRED" in text:
            reason = "login_required"
        elif "PURCHASE" in text or "coin" in text:
            reason = "locked_or_purchase_required"
        return [], reason
    query = [
        ("purchased", "false"),
        ("q", "40"),
        ("Policy", policy),
        ("Signature", signature),
        ("Key-Pair-Id", key_pair_id),
    ]
    urls = [f"https://rcdn.lezhin.com/v2{path}?{urlencode(query)}" for path in paths]
    return urls, "public_reader_images"


def object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def key_for(prefix: str, episode: dict[str, Any], index: int, image_url: str, content_type: str | None = None) -> str:
    return (
        f"{prefix.strip('/')}/{SERIES_SLUG}/{episode['episode_slug']}/"
        f"page-{index:04d}{extension_for_url(image_url, content_type)}"
    )


def download_image(
    *,
    sess: requests.Session,
    client: Any,
    bucket: str,
    prefix: str,
    episode: dict[str, Any],
    image_url: str,
    index: int,
    overwrite: bool,
) -> str:
    provisional_key = key_for(prefix, episode, index, image_url)
    if not overwrite and object_exists(client, bucket, provisional_key):
        return "skipped"
    response = sess.get(
        image_url,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": str(episode["episode_url"]),
        },
        timeout=90.0,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        raise RuntimeError(f"unexpected content-type {content_type}")
    key = key_for(prefix, episode, index, image_url, content_type)
    if key != provisional_key and not overwrite and object_exists(client, bucket, key):
        return "skipped"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=response.content,
        ContentType=content_type,
        Metadata={
            "source-platform": "lezhin",
            "series-slug": SERIES_SLUG,
            "episode-name": str(episode["episode_name"]),
            "episode-url": str(episode["episode_url"]),
            "page-index": str(index),
        },
    )
    return "written"


def download_episode(
    *,
    sess: requests.Session,
    client: Any,
    bucket: str,
    prefix: str,
    episode: dict[str, Any],
    image_workers: int,
    max_images: int,
    overwrite: bool,
) -> dict[str, Any]:
    text = get_text(sess, str(episode["episode_url"]), referer=BASE_URL)
    image_urls, reason = parse_reader(text, episode)
    if max_images:
        image_urls = image_urls[:max_images]
    if not image_urls:
        return {
            "status": "locked_or_unavailable",
            "reason": reason,
            "series": SERIES_SLUG,
            "episode_name": episode["episode_name"],
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
        futures = {
            pool.submit(
                download_image,
                sess=sess,
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
        "reason": reason,
        "series": SERIES_SLUG,
        "episode_name": episode["episode_name"],
        "episode_slug": episode["episode_slug"],
        "title": episode["title"],
        "image_count": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "failure_samples": samples,
        "s3_prefix": f"s3://{bucket}/{prefix.strip('/')}/{SERIES_SLUG}/{episode['episode_slug']}/",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_file(client: Any, *, bucket: str, key: str, path: Path, content_type: str) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(), ContentType=content_type)


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {"episodes": len(rows), "accessible": 0, "locked_or_unavailable": 0, "written": 0, "skipped": 0, "failures": 0, "images": 0}
    for row in rows:
        result["written"] += int(row.get("written") or 0)
        result["skipped"] += int(row.get("skipped") or 0)
        result["failures"] += int(row.get("failures") or 0)
        result["images"] += int(row.get("image_count") or 0)
        if row.get("status") == "accessible":
            result["accessible"] += 1
        elif row.get("status") == "locked_or_unavailable":
            result["locked_or_unavailable"] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Lezhin public reader pages for Lout of Count's Family.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-images-per-episode", type=int, default=0)
    parser.add_argument("--episode-workers", type=int, default=4)
    parser.add_argument("--image-workers", type=int, default=6)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload-manifest", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    args = parser.parse_args()

    run_id = args.run_id or dt.datetime.utcnow().strftime("lezhin_lout_%Y%m%d_%H%M%S")
    manifest_dir = Path(args.manifest_dir)
    sess = session()
    client = s3_client()
    episodes = discover_episodes(sess, max_episodes=max(0, args.max_episodes))
    manifest_path = manifest_dir / f"{run_id}_episodes.jsonl"
    status_path = manifest_dir / f"{run_id}_status.jsonl"
    summary_path = manifest_dir / f"{run_id}_summary.json"
    write_jsonl(manifest_path, episodes)
    print(f"setup: run_id={run_id} episodes={len(episodes)} download={args.download}", flush=True)
    print(f"manifest: path={manifest_path} episodes={len(episodes)}", flush=True)

    status_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    last = 0.0
    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.episode_workers)) as pool:
            future_map = {
                pool.submit(
                    download_episode,
                    sess=sess,
                    client=client,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    episode=episode,
                    image_workers=max(1, args.image_workers),
                    max_images=max(0, args.max_images_per_episode),
                    overwrite=bool(args.overwrite),
                ): episode
                for episode in episodes
            }
            for future in concurrent.futures.as_completed(future_map):
                episode = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "reason": str(exc),
                        "series": SERIES_SLUG,
                        "episode_name": episode["episode_name"],
                        "episode_slug": episode["episode_slug"],
                        "title": episode["title"],
                        "image_count": 0,
                        "written": 0,
                        "skipped": 0,
                        "failures": 1,
                    }
                status_rows.append(row)
                now = time.monotonic()
                if now - last >= args.progress_interval or len(status_rows) == len(episodes):
                    last = now
                    current = counts(status_rows)
                    elapsed = max(0.1, now - start)
                    rate = len(status_rows) / elapsed
                    eta = max(0, len(episodes) - len(status_rows)) / rate if rate > 0 else 0
                    print(
                        "progress: "
                        f"episodes={len(status_rows)}/{len(episodes)} accessible={current['accessible']} "
                        f"locked_or_unavailable={current['locked_or_unavailable']} images={current['images']} "
                        f"written={current['written']} skipped={current['skipped']} failures={current['failures']} "
                        f"episode_rate={rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )

    write_jsonl(status_path, status_rows)
    summary = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bucket": args.bucket,
        "prefix": args.prefix.strip("/"),
        "series": [SERIES_SLUG],
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "download": bool(args.download),
        "counts": counts(status_rows),
        "notes": [
            "Only official Lezhin reader pages with public embedded cut metadata and signed CDN parameters were downloaded.",
            "LOGIN_REQUIRED, coin, and missing-reader episodes are recorded as locked_or_unavailable.",
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
