#!/usr/bin/env python3
"""Launcher for the Drawtoon manga_annotate workflow.

Lists filtered manga pages under ``s3://drawtoon/datasets/pages/filtered/``,
skips pages that already have an annotation, writes a manifest JSONL, and
invokes ``modal run modal_magi.py::annotate_manifest_local`` so the H100
pool fans the work out.
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
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_S3_REGION") or "us-east-1"
DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_OUTPUT_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_GEMINI_VERIFIER_MODEL = os.environ.get("GEMINI_VERIFIER_MODEL", "gemini-3-flash-preview")
DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL = os.environ.get("GEMINI_VERIFIER_THINKING_LEVEL", "HIGH")
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


def infer_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _list_existing_annotations(s3, bucket: str, output_prefix: str, chapter: str) -> set[str]:
    """Set of page_id stems that already have an annotation for the given chapter."""
    prefix = f"{output_prefix.rstrip('/')}/{chapter}/"
    paginator = s3.get_paginator("list_objects_v2")
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if key.endswith(".jsonl"):
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
        existing = set() if overwrite else _list_existing_annotations(s3, bucket, output_prefix, output_chapter)
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
                rows.append(
                    {
                        "chapter": chapter,
                        "output_chapter": output_chapter,
                        "page_id": page_id,
                        "sample_id": f"{output_chapter}__{page_id}",
                        "page_key": key,
                        "output_key": f"{output_prefix.rstrip('/')}/{output_chapter}/{page_id}.jsonl",
                    }
                )
                if max_pages > 0 and len(rows) >= max_pages:
                    return rows, stats
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="lineart2-s3")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--chapters",
        nargs="*",
        default=[],
        help="One or more chapter folder names (e.g. jujutsu-kaisen). Empty = all under source prefix.",
    )
    parser.add_argument("--chapter-regex", default="", help="Used only when --chapters is empty.")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--pages-per-shard", type=int, default=16)
    # Gemini character verification is always on and mandatory: after Magi v3
    # detection, Gemini drops/corrects character boxes and stores audit reasons.
    # A Gemini failure on a page fails that page (it lands in _failed/).
    parser.add_argument("--gemini-verifier-model", default=DEFAULT_GEMINI_VERIFIER_MODEL)
    parser.add_argument("--gemini-verifier-thinking-level", default=DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Where to write the manifest JSONL. Default: /tmp/manga_annotate_<run_id>.jsonl",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Pass --detach to modal run so the run keeps going if the local CLI exits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List pages and write manifest but skip modal run.")
    args = parser.parse_args()

    session = boto3_session(args.profile.strip())
    run_id = args.run_id.strip() or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    rows, stats = list_pages(
        session=session,
        bucket=args.bucket,
        source_prefix=args.source_prefix,
        output_prefix=args.output_prefix,
        chapters=args.chapters,
        chapter_regex=args.chapter_regex,
        overwrite=args.overwrite,
        max_pages=args.max_pages,
    )
    if not rows:
        print(json.dumps({"event": "noop", "stats": stats, "reason": "no pages to annotate"}, indent=2))
        return 0

    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(f"/tmp/manga_annotate_{run_id}.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "event": "manifest_written",
                "manifest_path": str(manifest_path),
                "run_id": run_id,
                "page_count": len(rows),
                "stats": stats,
                "chapters": sorted({row["chapter"] for row in rows}),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        return 0

    cmd = [
        "modal",
        "run",
    ]
    if args.detach:
        cmd.append("--detach")
    cmd.extend(
        [
            f"{WORKFLOW_DIR / 'modal_magi.py'}::annotate_manifest_local",
            "--manifest-path",
            str(manifest_path),
            "--bucket",
            args.bucket,
            "--run-id",
            run_id,
            "--git-sha",
            infer_git_sha(),
            "--gpu-batch-size",
            str(args.gpu_batch_size),
            "--pages-per-shard",
            str(args.pages_per_shard),
        ]
    )
    cmd.extend(
        [
            "--gemini-verifier-model",
            args.gemini_verifier_model,
            "--gemini-verifier-thinking-level",
            args.gemini_verifier_thinking_level,
        ]
    )
    if args.overwrite:
        cmd.append("--overwrite")
    print(json.dumps({"event": "modal_run", "cmd": cmd}), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
