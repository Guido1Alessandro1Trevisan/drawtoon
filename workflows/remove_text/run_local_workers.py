#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import handlers


DEFAULT_MANIFEST_URI = (
    "s3://drawtoon/datasets/pages/text_removed/"
    "qwen2511_master_prompt_mangazero_v1/_jobs/20260516T195116Z-3122/page_manifest.jsonl"
)
DEFAULT_WORKER_CONFIG_URI = (
    "s3://drawtoon/datasets/pages/text_removed/"
    "qwen2511_master_prompt_mangazero_v1/_jobs/20260516T195116Z-3122/worker_config.json"
)


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


class LocalContext:
    def __init__(self, timeout_seconds: float) -> None:
        self._deadline = time.time() + float(timeout_seconds)

    def get_remaining_time_in_millis(self) -> int:
        return max(0, int((self._deadline - time.time()) * 1000))


class Counters:
    def __init__(self) -> None:
        self.started = time.time()
        self.enqueued = 0
        self.done = 0
        self.completed = 0
        self.skipped = 0
        self.errors = 0
        self.retries = 0
        self.lock = threading.Lock()

    def bump(self, name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, name, int(getattr(self, name)) + amount)

    def snapshot(self, *, queue_size: int) -> dict[str, Any]:
        with self.lock:
            elapsed = max(0.001, time.time() - self.started)
            finished = self.completed + self.skipped + self.errors
            return {
                "elapsed_seconds": round(elapsed, 1),
                "enqueued": self.enqueued,
                "queue_remaining": queue_size,
                "done": self.done,
                "completed": self.completed,
                "skipped_existing": self.skipped,
                "errors": self.errors,
                "retries": self.retries,
                "finished_pages_per_min": round(finished / elapsed * 60.0, 2),
                "new_outputs_per_min": round(self.completed / elapsed * 60.0, 2),
            }


STOP = threading.Event()


def parse_s3_uri(value: str) -> S3Uri:
    raw = str(value).strip()
    if not raw.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {value!r}")
    bucket, _, key = raw[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got {value!r}")
    return S3Uri(bucket=bucket, key=key)


def load_manifest(uri: S3Uri, *, start_index: int, limit: int) -> list[tuple[int, dict[str, Any]]]:
    response = handlers._s3_client().get_object(Bucket=uri.bucket, Key=uri.key)
    rows: list[tuple[int, dict[str, Any]]] = []
    for row_index, raw_line in enumerate(response["Body"].iter_lines()):
        if row_index < start_index:
            continue
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {row_index} is not an object")
        rows.append((row_index, row))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def process_one(
    *,
    row_index: int,
    row: dict[str, Any],
    config_ref: dict[str, str],
    timeout_seconds: float,
    max_attempts: int,
    counters: Counters,
) -> None:
    for attempt in range(1, max_attempts + 1):
        if STOP.is_set():
            return
        result = handlers.remove_text_page(
            {"row_index": row_index, "page": row, "config_ref": config_ref},
            LocalContext(timeout_seconds),
        )
        status = str(result.get("status") or "")
        if status == "completed":
            counters.bump("completed")
            counters.bump("done")
            return
        if status == "skipped_existing":
            counters.bump("skipped")
            counters.bump("done")
            return
        if attempt < max_attempts:
            counters.bump("retries")
            time.sleep(min(30.0, 2.0 * attempt))
            continue
        counters.bump("errors")
        counters.bump("done")
        return


def worker_loop(
    *,
    name: str,
    tasks: "queue.Queue[tuple[int, dict[str, Any]] | None]",
    config_ref: dict[str, str],
    timeout_seconds: float,
    max_attempts: int,
    counters: Counters,
) -> None:
    while not STOP.is_set():
        item = tasks.get()
        try:
            if item is None:
                return
            row_index, row = item
            process_one(
                row_index=row_index,
                row=row,
                config_ref=config_ref,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                counters=counters,
            )
        except Exception as exc:
            counters.bump("errors")
            counters.bump("done")
            print(json.dumps({"worker": name, "status": "exception", "error": str(exc), "row_index": item[0] if item else None}), flush=True)
        finally:
            tasks.task_done()


def update_worker_config(uri: S3Uri, *, poll_interval_seconds: float) -> None:
    config = handlers._get_s3_json(uri.bucket, uri.key)
    if poll_interval_seconds > 0:
        config["poll_interval_seconds"] = float(poll_interval_seconds)
        handlers._put_s3_json(uri.bucket, uri.key, config, pretty=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI)
    parser.add_argument("--worker-config-uri", default=DEFAULT_WORKER_CONFIG_URI)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=840.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    args = parser.parse_args()

    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("FAL_SECRET_NAME", "drawtoon-fal-key")

    manifest_uri = parse_s3_uri(args.manifest_uri)
    worker_config_uri = parse_s3_uri(args.worker_config_uri)
    update_worker_config(worker_config_uri, poll_interval_seconds=float(args.poll_interval_seconds))

    def handle_signal(signum: int, _frame: object) -> None:
        print(json.dumps({"status": "stopping", "signal": signum}), flush=True)
        STOP.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    rows = load_manifest(
        manifest_uri,
        start_index=max(0, int(args.start_index)),
        limit=max(0, int(args.limit)),
    )
    tasks: "queue.Queue[tuple[int, dict[str, Any]] | None]" = queue.Queue(maxsize=max(1, int(args.workers)) * 4)
    counters = Counters()
    counters.enqueued = len(rows)
    config_ref = {"bucket": worker_config_uri.bucket, "key": worker_config_uri.key}

    print(
        json.dumps(
            {
                "status": "starting",
                "workers": int(args.workers),
                "rows": len(rows),
                "start_index": max(0, int(args.start_index)),
                "manifest_uri": args.manifest_uri,
                "worker_config_uri": args.worker_config_uri,
                "poll_interval_seconds": float(args.poll_interval_seconds),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    threads = [
        threading.Thread(
            target=worker_loop,
            kwargs={
                "name": f"worker-{i:02d}",
                "tasks": tasks,
                "config_ref": config_ref,
                "timeout_seconds": float(args.timeout_seconds),
                "max_attempts": max(1, int(args.max_attempts)),
                "counters": counters,
            },
            daemon=True,
        )
        for i in range(max(1, int(args.workers)))
    ]
    for thread in threads:
        thread.start()

    def feeder() -> None:
        for item in rows:
            if STOP.is_set():
                break
            tasks.put(item)
        for _ in threads:
            tasks.put(None)

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    while any(thread.is_alive() for thread in threads):
        print(json.dumps({"status": "progress", **counters.snapshot(queue_size=tasks.qsize())}, sort_keys=True), flush=True)
        time.sleep(max(1.0, float(args.progress_seconds)))

    print(json.dumps({"status": "finished", **counters.snapshot(queue_size=tasks.qsize())}, sort_keys=True), flush=True)
    if counters.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
