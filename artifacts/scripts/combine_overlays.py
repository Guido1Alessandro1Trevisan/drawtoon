"""Concatenate every existing per-detector cuts_overlay.jpg into one wide image.

Reads ``artifacts/gutter_comparison/<detector>/the-mafia-nanny_manwa/episode-000025-episode-25/cuts_overlay.jpg``
for each detector subdir, glues them side-by-side, and writes
``artifacts/gutter_comparison/cuts_overlay_side_by_side.jpg``.

Run:
    uv run --active --with pillow python artifacts/scripts/combine_overlays.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts" / "gutter_comparison"
CHAPTER = "the-mafia-nanny_manwa/episode-000025-episode-25"
OUT = ROOT / "cuts_overlay_side_by_side.jpg"

# Order detectors from baseline -> classical -> ML.
ORDER = [
    "baseline_white_black",
    "kcc_find_edges",
    "canny_laplacian",
    "saliency_finegrained",
    "pelt_rbf",
    "persistent_homology",
    "rembg_birefnet",
]


def load_stats(detector: str) -> dict:
    manifest_path = ROOT / detector / CHAPTER / "manifest.json"
    if not manifest_path.exists():
        return {}
    m = json.loads(manifest_path.read_text())
    segs = m.get("segments") or []
    heights = [int(s.get("rendered_height") or 0) for s in segs]
    return {
        "cut_count": int((m.get("segmentation") or {}).get("detected_gutter_cuts") or len(segs)),
        "segment_count": len(segs),
        "sheet_count": int(m.get("sheet_count") or len(m.get("sheets") or [])),
        "max_height": max(heights) if heights else 0,
        "median_height": int(sorted(heights)[len(heights) // 2]) if heights else 0,
    }


def main() -> None:
    panels: list[tuple[str, Image.Image, dict]] = []
    for det in ORDER:
        overlay_path = ROOT / det / CHAPTER / "cuts_overlay.jpg"
        if not overlay_path.exists():
            print(f"  skip {det}: no overlay")
            continue
        img = Image.open(overlay_path).convert("RGB")
        stats = load_stats(det)
        panels.append((det, img, stats))
    if not panels:
        print("nothing to combine", file=sys.stderr)
        sys.exit(1)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=16)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=13)
    except Exception:
        font = ImageFont.load_default()
        small = font
    gap = 10
    footer_h = 56
    panel_width = panels[0][1].width
    panel_height = max(p[1].height for p in panels)
    combined_width = panel_width * len(panels) + gap * (len(panels) - 1)
    combined_height = panel_height + footer_h
    combined = Image.new("RGB", (combined_width, combined_height), (15, 15, 15))
    draw = ImageDraw.Draw(combined)
    x = 0
    print(f"{'detector':<24} cuts segments sheets median_h max_h")
    for det, img, stats in panels:
        combined.paste(img, (x, 0))
        # Footer text under each panel.
        cx, cy = x + 6, panel_height + 4
        draw.text((cx, cy), det, fill=(220, 220, 220), font=font)
        line2 = f"cuts={stats.get('cut_count', '?')} segs={stats.get('segment_count', '?')} sheets={stats.get('sheet_count', '?')}"
        line3 = f"max={stats.get('max_height', 0)}px med={stats.get('median_height', 0)}px"
        draw.text((cx, cy + 18), line2, fill=(180, 180, 180), font=small)
        draw.text((cx, cy + 34), line3, fill=(180, 180, 180), font=small)
        print(f"{det:<24} {stats.get('cut_count', '?'):>4} {stats.get('segment_count', '?'):>8} {stats.get('sheet_count', '?'):>6} {stats.get('median_height', 0):>8} {stats.get('max_height', 0):>5}")
        x += panel_width + gap
    out_bytes = OUT
    combined.save(out_bytes, format="JPEG", quality=82, optimize=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
