from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from filelock import FileLock


def _wait_for_everyone(accelerator) -> None:
    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()


def _load_marker_payload(marker_path: Path) -> dict:
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _marker_matches(marker_path: Path, marker_payload: dict | None) -> bool:
    if not marker_path.exists():
        return False
    stored_payload = _load_marker_payload(marker_path).get("payload", {})
    return stored_payload == (marker_payload or {})


def _failed_marker_payload_matches(failed_path: Path, marker_payload: dict | None) -> bool:
    if not failed_path.exists():
        return False
    stored_payload = _load_marker_payload(failed_path).get("payload", {})
    return stored_payload == (marker_payload or {})


def _wait_for_marker(
    *,
    marker_path: Path,
    failed_path: Path,
    marker_payload: dict | None,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _marker_matches(marker_path, marker_payload):
            return
        if _failed_marker_payload_matches(failed_path, marker_payload):
            failed_payload = _load_marker_payload(failed_path)
            raise RuntimeError(
                f"Cache seeding failed for marker {marker_path}: "
                f"{failed_payload.get('error', 'unknown error')}"
            )
        time.sleep(2.0)
    raise TimeoutError(
        f"Timed out waiting for cache marker {marker_path} "
        f"with expected payload match after {timeout_seconds}s"
    )


def run_once_with_filelock(
    *,
    accelerator,
    cache_dir: str | os.PathLike,
    marker_name: str,
    fn: Callable[[], None],
    logger=None,
    should_run: Callable[[], bool] | None = None,
    marker_payload: dict | None = None,
    timeout_seconds: int = 3600,
) -> None:
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    marker_path = cache_dir_path / f"{marker_name}.done.json"
    failed_path = cache_dir_path / f"{marker_name}.failed.json"
    lock_path = cache_dir_path / f"{marker_name}.lock"

    _wait_for_everyone(accelerator)

    if getattr(accelerator, "is_main_process", True):
        with FileLock(str(lock_path), timeout=timeout_seconds):
            if failed_path.exists():
                try:
                    failed_path.unlink()
                except FileNotFoundError:
                    pass
            needs_run = True
            if marker_path.exists():
                marker_matches = _marker_matches(marker_path, marker_payload)
                needs_run = (not marker_matches) or (bool(should_run()) if should_run is not None else False)

            if not needs_run:
                if logger is not None:
                    logger.event(
                        "cache_seed_skip",
                        print_main=True,
                        marker=str(marker_path),
                        payload=marker_payload or {},
                    )
            else:
                if logger is not None:
                    logger.event(
                        "cache_seed_start",
                        print_main=True,
                        marker=str(marker_path),
                        payload=marker_payload or {},
                    )

                try:
                    fn()
                except Exception as exc:
                    failed_path.write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "time": time.time(),
                                "pid": os.getpid(),
                                "error": str(exc),
                                "payload": marker_payload or {},
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    raise

                tmp_path = marker_path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps(
                        {
                            "status": "done",
                            "time": time.time(),
                            "pid": os.getpid(),
                            "payload": marker_payload or {},
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp_path, marker_path)

                if logger is not None:
                    logger.event(
                        "cache_seed_done",
                        print_main=True,
                        marker=str(marker_path),
                        payload=marker_payload or {},
                    )

    if getattr(accelerator, "is_main_process", True):
        return
    _wait_for_marker(
        marker_path=marker_path,
        failed_path=failed_path,
        marker_payload=marker_payload,
        timeout_seconds=timeout_seconds,
    )
