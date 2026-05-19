#!/usr/bin/env python3
"""Launcher for the Drawtoon remove_text workflow.

Lists filtered manga pages, skips pages that already have an output PNG, writes
a manifest JSONL, and invokes ``modal run modal_klein.py::annotate_manifest_local``
(or ``::annotate_manifest_spawn`` when ``--detach`` is set).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3


WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_REGION = os.environ.get("AWS_REGION") or "us-east-1"
DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_OUTPUT_PREFIX_KLEIN_LOCAL = "datasets/pages/text_removed"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NON_MANGA_SUFFIXES = ("_manwa", "_manhwa", "_manha", "_manhua", "_comic")
KNOWN_PLAIN_MANGA_CHAPTERS = {
    "jujutsu-kaisen",
    "monster",
    "my-hero-academia",
    "the-fragrant-flower-blooms-with-dignity",
    "vagabond",
    "vinland-saga",
}


def boto3_session(profile: str) -> boto3.Session:
    if profile:
        return boto3.Session(profile_name=profile, region_name=DEFAULT_REGION)
    return boto3.Session(region_name=DEFAULT_REGION)


def _list_existing(s3, bucket: str, output_prefix: str, chapter: str) -> set[str]:
    prefix = f"{output_prefix.rstrip('/')}/{chapter}/"
    paginator = s3.get_paginator("list_objects_v2")
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if key.endswith(".png"):
                seen.add(Path(key).stem)
    return seen


def normalize_manga_chapter_name(chapter: str) -> str:
    if (
        not chapter
        or chapter.startswith("_")
        or chapter.endswith("_manga")
        or chapter.endswith(NON_MANGA_SUFFIXES)
    ):
        return chapter
    if chapter.endswith(("_mangazero", "_manga109")) or chapter in KNOWN_PLAIN_MANGA_CHAPTERS:
        return f"{chapter}_manga"
    return chapter


def list_pages(
    *,
    session: boto3.Session,
    bucket: str,
    source_prefix: str,
    output_prefix: str,
    skip_existing_prefixes: list[str],
    chapters: list[str],
    chapter_regex: str,
    overwrite: bool,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    s3 = session.client("s3")
    rows: list[dict[str, Any]] = []
    stats = {"source_image_count": 0, "skipped_existing_count": 0, "chapter_count": 0}
    include_re = re.compile(chapter_regex) if chapter_regex else None

    if not chapters:
        chapters = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix.rstrip("/") + "/", Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []) or []:
                chapter = str(prefix.get("Prefix") or "").rstrip("/").split("/")[-1]
                if chapter and (not include_re or include_re.search(chapter)):
                    chapters.append(chapter)
    chapters = sorted(set(chapters))
    stats["chapter_count"] = len(chapters)

    for chapter in chapters:
        output_chapter = normalize_manga_chapter_name(chapter)
        chapter_prefix = f"{source_prefix.rstrip('/')}/{chapter}/"
        existing_prefixes = [output_prefix] + [p.strip() for p in skip_existing_prefixes if p.strip()]
        existing = set()
        if not overwrite:
            for existing_prefix in dict.fromkeys(existing_prefixes):
                existing.update(_list_existing(s3, bucket, existing_prefix, output_chapter))
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=chapter_prefix):
            for obj in page.get("Contents", []) or []:
                key = str(obj.get("Key") or "")
                if Path(key).suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                stats["source_image_count"] += 1
                page_id = Path(key).stem
                if not overwrite and page_id in existing:
                    stats["skipped_existing_count"] += 1
                    continue
                rows.append({
                    "chapter": chapter,
                    "output_chapter": output_chapter,
                    "page_id": page_id,
                    "sample_id": f"{output_chapter}__{page_id}",
                    "page_key": key,
                    "output_key": f"{output_prefix.rstrip('/')}/{output_chapter}/{page_id}.png",
                })
                if max_pages > 0 and len(rows) >= max_pages:
                    return rows, stats
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="default")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument(
        "--variant",
        choices=["klein_local_9b_4step"],
        default="klein_local_9b_4step",
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument(
        "--skip-existing-prefix",
        action="append",
        default=[],
        help="Optional S3 prefix to check for existing page IDs while writing outputs to --output-prefix.",
    )
    parser.add_argument("--chapters", nargs="*", default=[])
    parser.add_argument("--chapter-regex", default="")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pages-per-shard", type=int, default=8)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt-override-file",
        default="",
        help="Path to a prompt .md file that overrides prompts/master_prompt.md for this run (e.g. prompts/comic_prompt.md).",
    )
    args = parser.parse_args()

    default_output_prefixes = {"klein_local_9b_4step": DEFAULT_OUTPUT_PREFIX_KLEIN_LOCAL}
    output_prefix = args.output_prefix or default_output_prefixes[args.variant]

    session = boto3_session(args.profile.strip())
    run_id = args.run_id.strip() or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    rows, stats = list_pages(
        session=session, bucket=args.bucket,
        source_prefix=args.source_prefix,
        output_prefix=output_prefix,
        skip_existing_prefixes=args.skip_existing_prefix,
        chapters=args.chapters, chapter_regex=args.chapter_regex,
        overwrite=args.overwrite, max_pages=args.max_pages,
    )
    if not rows:
        print(json.dumps({"event": "noop", "stats": stats, "reason": "no pages to process"}, indent=2))
        return 0

    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(f"/tmp/remove_text_{run_id}.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "event": "manifest_written", "manifest_path": str(manifest_path),
        "run_id": run_id, "variant": args.variant, "output_prefix": output_prefix,
        "page_count": len(rows), "stats": stats,
        "chapters": sorted({row["chapter"] for row in rows}),
    }, indent=2), flush=True)

    if args.dry_run:
        return 0

    # .map() in detached mode prints a "calls may be cancelled" warning because
    # the local caller goes away. Use the spawn-based entrypoint for --detach:
    # it fires every shard via .spawn(), prints call IDs, and exits cleanly
    # while Modal keeps running the shards.
    entrypoint = "annotate_manifest_spawn" if args.detach else "annotate_manifest_local"
    cmd = ["modal", "run"]
    if args.detach:
        cmd.append("--detach")
    cmd.extend([
        f"{WORKFLOW_DIR / 'modal_klein.py'}::{entrypoint}",
        "--manifest-path", str(manifest_path),
        "--variant", args.variant,
        "--bucket", args.bucket,
        "--run-id", run_id,
        "--pages-per-shard", str(args.pages_per_shard),
        "--trust-manifest",
    ])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.prompt_override_file:
        cmd.extend(["--prompt-override-file", args.prompt_override_file])
    print(json.dumps({"event": "modal_run", "cmd": cmd}), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
