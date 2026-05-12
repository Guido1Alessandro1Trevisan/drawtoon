from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm

from toolkit.accelerator import get_accelerator


def _env_truthy(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


class DistributedLogger:
    def __init__(self, accelerator=None, log_dir: str | os.PathLike | None = None):
        self.accelerator = accelerator or get_accelerator()
        self.rank = int(getattr(self.accelerator, "process_index", 0))
        self.local_rank = int(getattr(self.accelerator, "local_process_index", -1))
        self.world_size = int(getattr(self.accelerator, "num_processes", 1))
        self.is_main = bool(getattr(self.accelerator, "is_main_process", True))
        self.log_all_ranks = _env_truthy("AITK_LOG_ALL_RANKS")

        self.log_dir = Path(log_dir) if log_dir else None
        self.log_path: Path | None = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / f"rank_{self.rank:02d}.jsonl"

    def event(
        self,
        name: str,
        *,
        print_main: bool = False,
        print_all: bool = False,
        **payload: Any,
    ) -> None:
        record = {
            "time": time.time(),
            "event": name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            **payload,
        }

        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

        if print_all or self.log_all_ranks or (print_main and self.is_main):
            print("[aitk-event] " + json.dumps(record, sort_keys=True, ensure_ascii=False), flush=True)

    def progress(
        self,
        iterable: Iterable,
        *,
        desc: str,
        total: Optional[int] = None,
        **kwargs,
    ):
        if self.is_main:
            kwargs.setdefault("dynamic_ncols", False)
            kwargs.setdefault("mininterval", 2.0)
            return tqdm(iterable, total=total, desc=desc, **kwargs)
        return iterable

    def gather_event(self, name: str, **payload: Any) -> None:
        """
        Must be called by all distributed ranks at the same logical point.
        Do not call from rank-conditional branches.
        """
        record = {
            "time": time.time(),
            "event": name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            **payload,
        }

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(gathered, record)
            if self.rank == 0:
                print("[aitk-gather] " + json.dumps(gathered, sort_keys=True, ensure_ascii=False), flush=True)
            return

        if self.is_main:
            print("[aitk-gather] " + json.dumps([record], sort_keys=True, ensure_ascii=False), flush=True)


def rank_tqdm(
    iterable: Iterable,
    *,
    desc: str,
    total: Optional[int] = None,
    accelerator=None,
    logger: DistributedLogger | None = None,
    **kwargs,
):
    if logger is not None:
        return logger.progress(iterable, desc=desc, total=total, **kwargs)

    resolved_accelerator = accelerator or get_accelerator()
    if getattr(resolved_accelerator, "is_main_process", True):
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("mininterval", 2.0)
        return tqdm(iterable, total=total, desc=desc, **kwargs)
    return iterable
