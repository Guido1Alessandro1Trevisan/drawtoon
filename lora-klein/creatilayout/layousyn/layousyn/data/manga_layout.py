"""Drawtoon MangaLayout dataset for LayouSyn fine-tunes.

Vendored into the drawtoon-LayouSyn tree at this path so the upstream
__init__.py can register it cleanly (no runtime patching).

Input JSONL schema (one row per panel; identical for the page model):

    {
      "prompt": "medium shot, eye-level, two characters fighting in a street, 1 speech from character 1",
      "width":  0.683,
      "height": 0.412,
      "items": [
        {"label": "character 1",                    "bbox": [-0.847, -0.812, -0.286,  0.562]},
        {"label": "character 2",                    "bbox": [ 0.071, -0.750,  0.836,  0.687]},
        {"label": "speech bubble from character 1", "bbox": [-0.796, -0.937, -0.439, -0.625]}
      ]
    }

Conventions:
- `bbox` is `[x0, y0, x1, y1]` in `[-1, 1]` panel-local coordinates (LayouSyn
  convention; paper Sec. 3.2).
- `width` and `height` are normalized to the parent page (panel model) or to
  the longest page side (page model). Both in `[0, 1]`. The model consumes
  `aspect_ratio = width / height` through the upstream `ar_embedder` —
  storing the pair instead of a single derived scalar keeps absolute scale
  available for future use without changing the model.
- Backward-compat fallback: rows that only carry `aspect_ratio` still work.

Per-item dict emitted to `EmbeddedDataset` matches the COCO pipeline contract:
`caption`, `bboxs` (in `[0, 1]`), `cats`, `gt_bboxs`, `gt_cats`, `width`,
`height`, `layout: Layout(...)`. We seed `Layout(width, height)` with
integers proportional to the actual normalized values so that the upstream
derivation `aspect_ratio = layout.width / layout.height` exactly recovers
panel_w / panel_h.
"""

from __future__ import annotations

import json
import random
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
from torch.utils.data import Dataset

from layout_evaluation import Layout

from .utils import cleanup_prompt, crop_layouts, shuffle_bboxs


# Resolution at which the normalized (width, height) pair is materialized
# into integer `Layout` dimensions. Only the ratio matters since upstream
# derives `aspect_ratio = layout.width / layout.height`.
_BASE_DIM = 1024


def _bbox_pm1_to_unit(bbox: list[float]) -> list[float]:
    """[-1, 1]^4 -> [0, 1]^4, clipped."""
    return [float(np.clip((v + 1.0) / 2.0, 0.0, 1.0)) for v in bbox]


def _resolve_width_height(row: dict[str, Any]) -> tuple[float, float]:
    """Return (width, height) in normalized units.

    Prefers explicit `width`/`height` from the row; falls back to deriving
    them from a legacy `aspect_ratio` if the row predates the schema change.
    """
    width = row.get("width")
    height = row.get("height")
    if width is not None and height is not None:
        w = max(1e-3, float(width))
        h = max(1e-3, float(height))
        return w, h
    aspect = max(0.1, float(row.get("aspect_ratio") or 1.0))
    return 1.0, max(1e-3, 1.0 / aspect)


class MangaLayout(Dataset):
    """Drawtoon JSONL → LayouSyn-compatible per-item dict."""

    def __init__(
        self,
        jsonl_path: str,
        shuffle_bbox: bool = False,
        crop_augment: bool = False,
        max_items: Optional[int] = None,
        seed: int = 0,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.shuffle_bbox = shuffle_bbox
        self.crop_augment = crop_augment
        self.max_items = max_items
        self._rng = random.Random(seed)

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"MangaLayout jsonl not found: {jsonl_path}")
        self.rows: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no} JSON decode error: {exc}") from exc
            if not isinstance(row, dict):
                continue
            items = row.get("items") or []
            items = [item for item in items if isinstance(item, dict) and "bbox" in item]
            if not items:
                continue
            if self.max_items is not None:
                items = items[: self.max_items]
            width_norm, height_norm = _resolve_width_height(row)
            self.rows.append(
                {
                    "prompt": str(row.get("prompt") or "").strip(),
                    "width": width_norm,
                    "height": height_norm,
                    "items": items,
                }
            )

        if logger:
            logger.info(
                f"Initialized MangaLayout dataset from {jsonl_path} with "
                f"{len(self.rows):,} examples (shuffle_bbox={shuffle_bbox}, "
                f"crop_augment={crop_augment})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        caption = cleanup_prompt(row["prompt"])
        width_norm = row["width"]
        height_norm = row["height"]

        # Integer width/height for upstream's `Layout` field. Pick the larger
        # of (w, h) as the BASE so the smaller dim still gets a useful integer
        # resolution. The ratio = w/h is preserved exactly.
        longer = max(width_norm, height_norm)
        legacy_width = max(1, int(round(_BASE_DIM * (width_norm / longer))))
        legacy_height = max(1, int(round(_BASE_DIM * (height_norm / longer))))

        bboxs: list[list[float]] = []
        cats: list[str] = []
        for item in row["items"]:
            bbox_in = item.get("bbox")
            if not isinstance(bbox_in, (list, tuple)) or len(bbox_in) != 4:
                continue
            bboxs.append(_bbox_pm1_to_unit(list(bbox_in)))
            cats.append(str(item.get("label") or ""))

        if self.shuffle_bbox:
            bboxs, cats = shuffle_bboxs(bboxs, cats)

        if self.crop_augment and bboxs:
            bboxs, cw, ch = crop_layouts(
                bboxs=bboxs,
                width=legacy_width,
                height=legacy_height,
            )
            legacy_width = int(cw)
            legacy_height = int(ch)

        layout = Layout(bboxs=bboxs, labels=cats, width=legacy_width, height=legacy_height)
        return {
            "caption_id": index,
            "caption": caption,
            "image_id": index,
            "bboxs": bboxs,
            "cats": cats,
            "gt_bboxs": bboxs,
            "gt_cats": cats,
            "width": legacy_width,
            "height": legacy_height,
            "layout": layout,
        }

    def get_captions(self, idx: int) -> list[str]:
        return [cleanup_prompt(self.rows[idx]["prompt"])]
