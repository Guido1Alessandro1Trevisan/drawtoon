#!/usr/bin/env python3
"""Render local review samples from manga_change_of_angle output JSONs."""
from __future__ import annotations

import argparse
import html
import io
import json
import random
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont, ImageOps


def s3_client(region: str):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"mode": "adaptive", "max_attempts": 10},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=96,
        ),
    )


def list_json_keys(client, bucket: str, prefix: str) -> list[str]:
    root = prefix.rstrip("/") + "/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            if "/_jobs/" in key or "/_audit/" in key:
                continue
            keys.append(key)
    return keys


def fetch_json(client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except Exception as exc:
        return {"_fetch_error": f"{type(exc).__name__}: {exc}", "_key": key}


def candidate_groups(payload: dict[str, Any], output_key: str) -> list[dict[str, Any]]:
    if payload.get("_fetch_error"):
        return []
    groups = payload.get("angle_groups") or []
    panels = payload.get("panels_in_reading_order") or []
    if not groups or not panels:
        return []
    chapter = str(payload.get("chapter") or "")
    page_id = str(payload.get("page_id") or "")
    rows: list[dict[str, Any]] = []
    for group_i, group in enumerate(groups):
        indices = group.get("panel_indices")
        if not isinstance(indices, list) or len(indices) < 2:
            continue
        try:
            indices = [int(x) for x in indices]
        except (TypeError, ValueError):
            continue
        if any(i < 0 or i >= len(panels) for i in indices):
            continue
        rows.append(
            {
                "chapter": chapter,
                "page_id": page_id,
                "output_key": output_key,
                "page_key": str(payload.get("page_key") or ""),
                "sample_id": str(payload.get("sample_id") or f"{chapter}__{page_id}"),
                "group_index": group_i,
                "panel_indices": indices,
                "reason": str(group.get("reason") or ""),
                "panels_in_reading_order": panels,
                "verification": payload.get("verification") or {},
                "summary": payload.get("summary") or {},
            }
        )
    return rows


def head_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def filter_existing_sources(
    client,
    bucket: str,
    candidates: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    page_keys = sorted({row["page_key"] for row in candidates if row.get("page_key")})
    exists: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_key = {pool.submit(head_exists, client, bucket, key): key for key in page_keys}
        for i, future in enumerate(as_completed(future_to_key), 1):
            key = future_to_key[future]
            try:
                exists[key] = future.result()
            except Exception as exc:
                print(f"warning: head failed for {key}: {type(exc).__name__}: {exc}")
                exists[key] = False
            if i % 1000 == 0:
                kept = sum(1 for value in exists.values() if value)
                print(f"checked {i}/{len(page_keys)} source pages; existing={kept}")
    return [row for row in candidates if exists.get(row["page_key"], False)]


def round_robin_sample(candidates: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_chapter: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_chapter.setdefault(row["chapter"], []).append(row)
    chapters = sorted(by_chapter)
    rng.shuffle(chapters)
    for chapter in chapters:
        rng.shuffle(by_chapter[chapter])
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for chapter in chapters:
            rows = by_chapter[chapter]
            if depth >= len(rows):
                continue
            selected.append(rows[depth])
            added = True
            if len(selected) >= count:
                break
        if not added:
            break
        depth += 1
    return selected


def safe_name(text: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return value[:max_len] or "sample"


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 220),
    )
    draw.text((x, y), text, fill=fill, font=font)


def render_one(client, bucket: str, sample: dict[str, Any], image_path: Path, thumb_path: Path) -> dict[str, Any]:
    page_key = sample["page_key"]
    body = client.get_object(Bucket=bucket, Key=page_key)["Body"].read()
    with Image.open(io.BytesIO(body)) as raw:
        raw.load()
        image = ImageOps.exif_transpose(raw).convert("RGB")

    max_side = 1800
    scale = 1.0
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)

    header_h = 124
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = load_font(max(18, image.width // 44))
    small_font = load_font(max(14, image.width // 62))
    label_font = load_font(max(18, image.width // 42))

    title = f"{sample['review_id']:04d}  {sample['chapter']} / {sample['page_id']}  group {sample['group_index']}"
    indices = ", ".join(str(i) for i in sample["panel_indices"])
    reason = sample["reason"] or "(no reason)"
    draw.rectangle((0, 0, canvas.width, header_h), fill=(255, 255, 255, 255))
    draw.text((12, 10), title[:150], fill=(0, 0, 0, 255), font=title_font)
    draw.text((12, 42), f"panel_indices: [{indices}]", fill=(180, 0, 0, 255), font=small_font)
    wrapped = textwrap.wrap(reason, width=max(55, canvas.width // 15))[:2]
    for line_i, line in enumerate(wrapped):
        draw.text((12, 70 + line_i * 24), line, fill=(0, 0, 0, 255), font=small_font)

    selected = set(sample["panel_indices"])
    colors = [
        (230, 0, 0, 255),
        (0, 140, 0, 255),
        (0, 92, 230, 255),
        (210, 120, 0, 255),
        (160, 0, 200, 255),
    ]
    for idx, panel in enumerate(sample["panels_in_reading_order"]):
        bbox = panel.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [round(float(v) * scale) for v in bbox[:4]]
        y0 += header_h
        y1 += header_h
        if idx in selected:
            color = colors[list(sample["panel_indices"]).index(idx) % len(colors)]
            width = max(5, round(canvas.width / 180))
            for offset in range(width):
                draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=color)
            draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, 30))
            draw_label(draw, (x0 + 8, y0 + 8), str(idx), label_font, (255, 255, 255, 255))
        else:
            draw.rectangle((x0, y0, x1, y1), outline=(90, 90, 90, 150), width=1)

    canvas.save(image_path, quality=92, optimize=True)
    thumb = canvas.copy()
    thumb.thumbnail((360, 520), Image.LANCZOS)
    thumb.save(thumb_path, quality=84, optimize=True)

    return {
        **{k: v for k, v in sample.items() if k not in {"panels_in_reading_order"}},
        "image_path": str(image_path),
        "thumb_path": str(thumb_path),
    }


def build_contact_sheets(samples: list[dict[str, Any]], sheets_dir: Path, cols: int = 5, rows: int = 4) -> list[Path]:
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[Path] = []
    per_sheet = cols * rows
    cell_w, cell_h = 360, 560
    font = load_font(15)
    for sheet_i in range(0, len(samples), per_sheet):
        batch = samples[sheet_i : sheet_i + per_sheet]
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for i, sample in enumerate(batch):
            thumb = Image.open(sample["thumb_path"]).convert("RGB")
            x = (i % cols) * cell_w
            y = (i // cols) * cell_h
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 8))
            label = f"{sample['review_id']:04d} [{','.join(map(str, sample['panel_indices']))}]"
            draw.text((x + 8, y + cell_h - 28), label, fill=(0, 0, 0), font=font)
        path = sheets_dir / f"sheet_{len(sheet_paths) + 1:03d}.jpg"
        sheet.save(path, quality=88, optimize=True)
        sheet_paths.append(path)
    return sheet_paths


def write_gallery(samples: list[dict[str, Any]], sheet_paths: list[Path], out_dir: Path) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Change Angle Review Samples</title>",
        "<style>body{font-family:sans-serif;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px} img{max-width:100%;height:auto;border:1px solid #ccc} .card{break-inside:avoid}.meta{font-size:13px;color:#333}</style>",
        "<h1>Change Angle Review Samples</h1>",
        "<p>500 sampled Kimi K2.6 reasoning-on angle groups. Red/colored boxes are panels in the selected group; gray boxes are other detected panels.</p>",
        "<h2>Contact Sheets</h2>",
    ]
    for path in sheet_paths:
        rel = path.relative_to(out_dir)
        lines.append(f"<p><a href='{html.escape(str(rel))}'>{html.escape(path.name)}</a></p>")
    lines.append("<h2>Samples</h2><div class='grid'>")
    for sample in samples:
        rel = Path(sample["image_path"]).relative_to(out_dir)
        reason = html.escape(sample.get("reason") or "")
        lines.append("<div class='card'>")
        lines.append(f"<a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}' loading='lazy'></a>")
        lines.append(
            f"<div class='meta'><b>{sample['review_id']:04d}</b> "
            f"{html.escape(sample['chapter'])}<br>{html.escape(sample['page_id'])}<br>"
            f"panels {html.escape(str(sample['panel_indices']))}<br>{reason}</div>"
        )
        lines.append("</div>")
    lines.append("</div>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="drawtoon")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--run-prefix",
        default="datasets/pages/change_angle/kimi_k26_on_diverse_manga_comic_75k_v1",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/change_angle_review_500_kimi_k26_on",
    )
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--seed", default="change-angle-review-500-kimi-k26-on-v1")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    client = s3_client(args.region)
    keys = list_json_keys(client, args.bucket, args.run_prefix)
    print(f"listed {len(keys)} page JSON keys")

    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_key = {pool.submit(fetch_json, client, args.bucket, key): key for key in keys}
        for i, future in enumerate(as_completed(future_to_key), 1):
            key = future_to_key[future]
            payload = future.result()
            if payload:
                candidates.extend(candidate_groups(payload, key))
            if i % 5000 == 0:
                print(f"scanned {i}/{len(keys)} jsons; candidates={len(candidates)}")
    print(f"candidate groups: {len(candidates)}")
    print("checking candidate source pages still exist")
    candidates = filter_existing_sources(client, args.bucket, candidates, args.workers)
    print(f"candidate groups with existing source pages: {len(candidates)}")
    if len(candidates) < args.sample_count:
        print(f"warning: only {len(candidates)} groups available")

    selected = round_robin_sample(candidates, min(args.sample_count, len(candidates)), args.seed)
    for i, row in enumerate(selected, 1):
        row["review_id"] = i

    rendered: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 24)) as pool:
        futures = []
        for row in selected:
            name = safe_name(f"{row['review_id']:04d}_{row['chapter']}__{row['page_id']}__g{row['group_index']}")
            image_path = images_dir / f"{name}.jpg"
            thumb_path = thumbs_dir / f"{name}.jpg"
            futures.append(pool.submit(render_one, client, args.bucket, row, image_path, thumb_path))
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rendered.append(future.result())
            except Exception as exc:
                print(f"warning: render failed: {type(exc).__name__}: {exc}")
            if i % 50 == 0:
                print(f"rendered {i}/{len(futures)}")
    rendered.sort(key=lambda row: row["review_id"])

    manifest_path = out_dir / "samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rendered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "bucket": args.bucket,
        "run_prefix": args.run_prefix,
        "sample_count": len(rendered),
        "candidate_groups": len(candidates),
        "seed": args.seed,
        "chapters_sampled": len({row["chapter"] for row in rendered}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sheet_paths = build_contact_sheets(rendered, out_dir / "sheets")
    write_gallery(rendered, sheet_paths, out_dir)
    (out_dir / "README.md").write_text(
        "# Change Angle Review Samples\n\n"
        "Open `index.html` for the gallery, or inspect `sheets/` for contact sheets.\n"
        "Each image highlights one sampled Kimi K2.6 reasoning-on angle group.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
