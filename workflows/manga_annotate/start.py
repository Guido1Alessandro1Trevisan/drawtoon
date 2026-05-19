#!/usr/bin/env python3
"""Launcher for the Drawtoon manga_annotate workflow.

Lists filtered manga pages under ``s3://drawtoon/datasets/pages/filtered/``,
skips pages that already have an annotation, writes a manifest JSONL, and
invokes ``modal run modal_magi.py::annotate_manifest_local`` so the H200
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


def list_manwa_pages_by_chapter(
    *,
    session: boto3.Session,
    bucket: str,
    source_prefix: str,
    chapters: list[str],
    chapter_regex: str,
) -> dict[str, list[str]]:
    """Discover raw manhwa page S3 keys per chapter under ``source_prefix``.

    Each chapter contributes a sorted list of S3 keys for its raw single pages
    (the inputs the manwa filter has not yet stitched). When ``chapters`` is
    empty, every immediate sub-prefix of ``source_prefix`` is treated as a
    chapter; ``chapter_regex`` filters that list.
    """
    s3 = session.client("s3")
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

    by_chapter: dict[str, list[str]] = {}
    for chapter in chapters:
        chapter_prefix = f"{source_prefix.rstrip('/')}/{chapter}/"
        keys: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=chapter_prefix):
            for obj in page.get("Contents", []) or []:
                key = str(obj.get("Key") or "")
                if Path(key).suffix.lower() in SUPPORTED_SUFFIXES:
                    keys.append(key)
        keys.sort()
        if keys:
            by_chapter[chapter] = keys
    return by_chapter


def build_manwa_manifest_rows(
    *,
    session: boto3.Session,
    bucket: str,
    pages_by_chapter: dict[str, list[str]],
    output_prefix: str,
    pages_per_chunk: int,
    max_parallel: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """For each chapter, run the Gemini-driven sheet builder locally and emit
    one manifest row per chapter. The row carries every sheet's slice plan
    (which the modal worker stitches in memory + annotates).

    The on-S3 annotation key for each sheet mirrors the manga layout:
        ``<output_prefix>/<output_chapter>/<output_chapter>__sheet-NNNN.jsonl``
    so downstream pipelines (captioner, trainer) discover them with no special
    logic.
    """
    # Local import so non-manwa runs don't pay for the boto/PIL/genai warmup
    # just to parse args. manwa_sheets.py lives next to this file.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from manwa_sheets import build_chapter_sheets  # type: ignore  # noqa: E402

    s3 = session.client("s3")
    rows: list[dict[str, Any]] = []
    stats = {
        "chapter_count": 0,
        "source_image_count": 0,
        "sheet_count": 0,
        "dropped_page_count": 0,
        "skipped_existing_count": 0,
    }
    stats["chapter_count"] = len(pages_by_chapter)
    for chapter, page_keys in pages_by_chapter.items():
        stats["source_image_count"] += len(page_keys)
        output_chapter = normalize_manga_chapter_name(chapter)
        page_bytes_list: list[bytes] = []
        page_etags: list[str] = []
        for key in page_keys:
            response = s3.get_object(Bucket=bucket, Key=key)
            page_bytes_list.append(response["Body"].read())
            page_etags.append(str(response.get("ETag") or "").strip('"'))
        result = build_chapter_sheets(
            page_bytes=page_bytes_list,
            page_keys=page_keys,
            page_etags=page_etags,
            pages_per_chunk=pages_per_chunk,
            max_parallel=max_parallel,
        )
        sheets = result.get("sheets") or []
        stats["dropped_page_count"] += len(result.get("dropped_page_indices") or [])
        if not sheets:
            continue
        existing_sheet_ids: set[str] = set()
        if not overwrite:
            existing_sheet_ids = _list_existing_annotations(s3, bucket, output_prefix, output_chapter)
        sheet_rows: list[dict[str, Any]] = []
        for sheet in sheets:
            sheet_id = str(sheet["sheet_id"])  # sheet_NNNN
            sample_id = f"{output_chapter}__{sheet_id}"
            output_key = f"{output_prefix.rstrip('/')}/{output_chapter}/{sample_id}.jsonl"
            if not overwrite and sample_id in existing_sheet_ids:
                stats["skipped_existing_count"] += 1
                continue
            sheet_rows.append(
                {
                    "sheet_id": sheet_id,
                    "sample_id": sample_id,
                    "output_key": output_key,
                    "width": int(sheet["width"]),
                    "height": int(sheet["height"]),
                    # Slice plan: each entry tells the worker which source page
                    # contributes which y-band of the sheet image. The worker
                    # downloads and stitches these into one PIL.Image per sheet.
                    "slices": [
                        {
                            "source_page_key": str(s["source_page_key"]),
                            "source_page_width": int(s.get("source_page_width") or 0),
                            "source_page_height": int(s.get("source_page_height") or 0),
                            "source_page_etag": str(s.get("source_page_etag") or ""),
                            "source_y_start": int(s["source_y_start"]),
                            "source_y_end": int(s["source_y_end"]),
                            "sheet_y_start": int(s["sheet_y_start"]),
                            "sheet_y_end": int(s["sheet_y_end"]),
                        }
                        for s in sheet.get("slices") or []
                    ],
                }
            )
        if not sheet_rows:
            continue
        stats["sheet_count"] += len(sheet_rows)
        rows.append(
            {
                "row_type": "manwa_chapter",
                "chapter": chapter,
                "output_chapter": output_chapter,
                "sheets": sheet_rows,
            }
        )
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
    parser.add_argument(
        "--manwa",
        action="store_true",
        help=(
            "Treat the source as raw manhwa/webtoon pages: for each chapter, "
            "run Gemini-driven sheet construction locally and emit a chapter-"
            "level manifest whose rows describe in-memory sheets (with slice "
            "provenance). Each sheet becomes one magi-v3 annotation row that "
            "is shape-compatible with manga page annotations (only the source "
            "block differs)."
        ),
    )
    parser.add_argument(
        "--manwa-source-prefix",
        default="datasets/pages/single",
        help="S3 prefix for raw manhwa pages when --manwa is set. Defaults to datasets/pages/single.",
    )
    parser.add_argument(
        "--manwa-pages-per-chunk",
        type=int,
        default=2,
        help="Number of pages stitched per Gemini cut chunk (manwa mode only).",
    )
    parser.add_argument(
        "--manwa-max-parallel",
        type=int,
        default=10,
        help="Max parallel Gemini cut requests per chapter (manwa mode only).",
    )
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

    if args.manwa:
        pages_by_chapter = list_manwa_pages_by_chapter(
            session=session,
            bucket=args.bucket,
            source_prefix=args.manwa_source_prefix,
            chapters=args.chapters,
            chapter_regex=args.chapter_regex,
        )
        if not pages_by_chapter:
            print(json.dumps({"event": "noop", "reason": "no manwa chapters found"}, indent=2))
            return 0
        rows, stats = build_manwa_manifest_rows(
            session=session,
            bucket=args.bucket,
            pages_by_chapter=pages_by_chapter,
            output_prefix=args.output_prefix,
            pages_per_chunk=args.manwa_pages_per_chunk,
            max_parallel=args.manwa_max_parallel,
            overwrite=args.overwrite,
        )
        if not rows:
            print(json.dumps({"event": "noop", "stats": stats, "reason": "no manwa sheets to annotate"}, indent=2))
            return 0
        manifest_entrypoint = "annotate_manwa_manifest_local"
        manifest_event = {
            "event": "manwa_manifest_written",
            "stats": stats,
            "chapters": sorted(pages_by_chapter.keys()),
        }
    else:
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
        manifest_entrypoint = "annotate_manifest_local"
        manifest_event = {
            "event": "manifest_written",
            "stats": stats,
            "chapters": sorted({row["chapter"] for row in rows}),
        }

    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(f"/tmp/manga_annotate_{run_id}.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                **manifest_event,
                "manifest_path": str(manifest_path),
                "run_id": run_id,
                "row_count": len(rows),
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
            f"{WORKFLOW_DIR / 'modal_magi.py'}::{manifest_entrypoint}",
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
