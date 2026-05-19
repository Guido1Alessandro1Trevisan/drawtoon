"""Smoke runner for the page-layout dataset.

Pulls 100 random pages from `s3://drawtoon/captions/gemini3_flash_page_panel_v1/`,
sends each page image + the MAGI panel-bbox context to Gemini Vision, asks
for a 10–15 word page-level prompt, and assembles the LayouSyn page-layout
training row (prompt + width/height + items[].label=size_class + bbox).

Prints 10 sampled rows so you can eyeball the format before committing to
the full Distributed-Map build.

Run:
    cd lora-klein/creatilayout/workflows/build_page_dataset
    uv run --quiet --with boto3 --with google-genai --with pillow \\
        python tests/test_100_pages.py
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from PIL import Image


BUCKET_SOURCE = "drawtoon"
BUCKET_SHORT = "drawtoon-layousyn"
SOURCE_PREFIX = "captions/gemini3_flash_page_panel_v1/"
MODEL = "gemini-3-flash-preview"
THINKING_LEVEL = "off"
N_PAGES = 100
N_DISPLAY = 10
SEED = 17
MAX_PAGE_SIDE = 1536
MAX_PARALLEL = 32
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
SECRET_ID = "drawtoon/gemini-api-key"


# Closed enums Gemini must pick from. The page model's prompt only needs the
# scene-level archetype + rhythm + reading direction; size_class per panel
# is computed deterministically.
PAGE_ARCHETYPES = (
    "splash",
    "action",
    "dialogue",
    "establishing",
    "transition",
    "reaction",
    "atmospheric",
    "montage",
)
RHYTHMS = ("fast", "steady", "slow", "decompressed")

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_archetype": {"type": "string", "format": "enum", "enum": list(PAGE_ARCHETYPES)},
        "rhythm": {"type": "string", "format": "enum", "enum": list(RHYTHMS)},
    },
    "required": ["page_archetype", "rhythm"],
}

SYSTEM_INSTRUCTION = (
    "You classify manga/manhwa pages for a page-layout supervision model.\n"
    "\n"
    "Given a full-page image and the count of pre-detected panels, return:\n"
    "- page_archetype: one of [splash, action, dialogue, establishing, "
    "transition, reaction, atmospheric, montage]\n"
    "- rhythm: one of [fast, steady, slow, decompressed]\n"
    "\n"
    "Definitions:\n"
    "- splash: 1-2 big panels carrying the page; usually a climactic image.\n"
    "- action: combat, motion, dynamic posing; usually mixed panel sizes.\n"
    "- dialogue: characters talking, many speech bubbles, similar sized panels.\n"
    "- establishing: a location or scene-setting shot, often wide panoramic.\n"
    "- transition: bridges scenes; often a single wide panel or abstract image.\n"
    "- reaction: close-ups of faces reacting to something, beat-driven.\n"
    "- atmospheric: mood/place focused, silent, sparse text.\n"
    "- montage: multiple disjoint moments, often time-passage.\n"
    "\n"
    "Rhythm: fast = many small beats; steady = dialogue or balanced panels; "
    "slow = decompressed splash or wide shots; decompressed = single beat "
    "stretched across many similar panels.\n"
    "\n"
    "Do not mention characters by name, dialogue text, mood adjectives, or "
    "the title. Output JSON only.\n"
)

USER_INSTRUCTION_TEMPLATE = (
    "This page has {n} pre-detected panels. "
    "Classify page_archetype and rhythm."
)


def _s3():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )


def _resolve_api_key() -> str:
    if os.environ.get(GEMINI_API_KEY_ENV):
        return os.environ[GEMINI_API_KEY_ENV]
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    secret = sm.get_secret_value(SecretId=SECRET_ID)["SecretString"]
    try:
        parsed = json.loads(secret)
        if isinstance(parsed, dict) and parsed.get(GEMINI_API_KEY_ENV):
            return parsed[GEMINI_API_KEY_ENV]
    except json.JSONDecodeError:
        pass
    return secret.strip()


def _pick_pages(seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    s3 = _s3()
    chapters: list[str] = []
    for obj in s3.list_objects_v2(Bucket=BUCKET_SOURCE, Prefix=SOURCE_PREFIX, Delimiter="/").get("CommonPrefixes", []):
        path = obj["Prefix"][len(SOURCE_PREFIX):].rstrip("/")
        if path.startswith("_"):
            continue
        chapters.append(path)
    rng.shuffle(chapters)

    chosen: list[str] = []
    for chapter in chapters:
        if len(chosen) >= N_PAGES:
            break
        page = s3.list_objects_v2(Bucket=BUCKET_SOURCE, Prefix=f"{SOURCE_PREFIX}{chapter}/", MaxKeys=20)
        keys = [c["Key"] for c in page.get("Contents", []) if c["Key"].endswith(".json")]
        if not keys:
            continue
        chosen.append(rng.choice(keys))
    return chosen


def _load_source(key: str) -> dict:
    body = _s3().get_object(Bucket=BUCKET_SOURCE, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def _load_page_image(page_key: str) -> Image.Image:
    body = _s3().get_object(Bucket=BUCKET_SOURCE, Key=page_key)["Body"].read()
    return Image.open(io.BytesIO(body)).convert("RGB")


def _encode_page(img: Image.Image) -> tuple[bytes, str]:
    if max(img.size) > MAX_PAGE_SIDE:
        img = img.copy()
        img.thumbnail((MAX_PAGE_SIDE, MAX_PAGE_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


def size_class_for(panel_bbox: list[float], page_w: int, page_h: int) -> str:
    px0, py0, px1, py1 = panel_bbox
    pw = max(1.0, px1 - px0)
    ph = max(1.0, py1 - py0)
    area = (pw * ph) / max(1, page_w * page_h)
    aspect = pw / ph
    if aspect >= 2.5:
        return "panoramic panel"
    if aspect <= 0.4:
        return "vertical panel"
    if area > 0.60:
        return "splash panel"
    if area > 0.30:
        return "hero panel"
    if area > 0.10:
        return "standard panel"
    return "beat panel"


def reading_direction(chapter: str) -> str:
    c = chapter.lower()
    if c.endswith("_manwa"):
        return "left-to-right"
    return "right-to-left"


def _classify_page(*, client, page_png: bytes, panel_count: int) -> tuple[dict, dict]:
    from google.genai import types  # type: ignore

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PAGE_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
        if THINKING_LEVEL == "off"
        else types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=page_png, mime_type="image/png"),
            USER_INSTRUCTION_TEMPLATE.format(n=panel_count),
        ],
        config=config,
    )
    raw = json.loads(resp.text or "{}")
    arch = str(raw.get("page_archetype") or "dialogue").strip()
    rhythm = str(raw.get("rhythm") or "steady").strip()
    if arch not in PAGE_ARCHETYPES:
        arch = "dialogue"
    if rhythm not in RHYTHMS:
        rhythm = "steady"
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
        "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
        "thoughts_tokens": int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
    }
    return {"page_archetype": arch, "rhythm": rhythm}, usage


def bbox_to_pm1_pagewise(panel_bbox: list[float], page_w: int, page_h: int) -> list[float]:
    """Pixel bbox → page-local [-1, 1] xyxy."""
    px0, py0, px1, py1 = panel_bbox
    return [
        round((px0 / page_w) * 2.0 - 1.0, 6),
        round((py0 / page_h) * 2.0 - 1.0, 6),
        round((px1 / page_w) * 2.0 - 1.0, 6),
        round((py1 / page_h) * 2.0 - 1.0, 6),
    ]


def build_row(source: dict, gemini: dict, *, chapter: str) -> dict:
    page_size = source.get("page_size") or {}
    page_w = int(page_size.get("width_px") or 0)
    page_h = int(page_size.get("height_px") or 0)
    longer = max(page_w, page_h, 1)
    width_norm = round(page_w / longer, 6)
    height_norm = round(page_h / longer, 6)
    rtl = reading_direction(chapter)

    panels = source.get("panels") or []
    items: list[dict[str, Any]] = []
    for p in panels:
        bbox = p.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        sz = size_class_for(bbox, page_w, page_h)
        items.append(
            {
                "label": sz,
                "bbox": bbox_to_pm1_pagewise(bbox, page_w, page_h),
            }
        )

    n = len(items)
    prompt = (
        f"{gemini['page_archetype']} page, {n} panels, "
        f"{gemini['rhythm']} rhythm, {rtl}"
    )
    return {
        "prompt": prompt,
        "width": width_norm,
        "height": height_norm,
        "items": items,
        "_meta": {
            "chapter": chapter,
            "page_id": source.get("page_id"),
            "page_archetype": gemini["page_archetype"],
            "rhythm": gemini["rhythm"],
            "reading_direction": rtl,
        },
    }


def main() -> None:
    api_key = _resolve_api_key()
    from google import genai
    from google.genai import types  # noqa: F401

    client = genai.Client(api_key=api_key)

    keys = _pick_pages()
    print(f"Picked {len(keys)} pages.\n")

    tasks: list[dict] = []
    for key in keys:
        try:
            source = _load_source(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {key}: load failed: {exc}")
            continue
        page_key = source.get("sources", {}).get("page_key")
        panels = source.get("panels") or []
        if not page_key or not panels:
            continue
        tasks.append(
            {
                "key": key,
                "source": source,
                "page_key": page_key,
                "panel_count": len(panels),
                "chapter": source.get("chapter") or key.split("/")[2],
            }
        )

    results: list[dict] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
    started = time.time()

    def _process(task: dict) -> dict:
        page_img = _load_page_image(task["page_key"])
        page_png, _mime = _encode_page(page_img)
        gemini, usage = _classify_page(
            client=client, page_png=page_png, panel_count=task["panel_count"]
        )
        row = build_row(task["source"], gemini, chapter=task["chapter"])
        return {"row": row, "usage": usage, "key": task["key"]}

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_process, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {futures[fut]['key']}: failed: {exc}")
                continue
            results.append(out)
            for k in usage_totals:
                usage_totals[k] += out["usage"].get(k, 0)
            if done % 25 == 0 or done == len(tasks):
                print(f"  ...{done}/{len(tasks)} pages done")

    elapsed = time.time() - started
    print(f"\nProcessed {len(results)} pages in {elapsed:.1f}s")
    print(f"Tokens: in={usage_totals['input_tokens']:,}  out={usage_totals['output_tokens']:,}  "
          f"think={usage_totals['thoughts_tokens']:,}")

    print(f"\n{'=' * 100}\nSAMPLED {N_DISPLAY} ROWS\n{'=' * 100}")
    rng = random.Random(SEED + 1)
    for sample in rng.sample(results, min(N_DISPLAY, len(results))):
        row = sample["row"]
        meta = row["_meta"]
        print(f"\n# {meta['chapter']}/{meta['page_id']}  (s3://{BUCKET_SOURCE}/{sample['key']})")
        print(f"  prompt: \"{row['prompt']}\"")
        print(f"  width:  {row['width']}   height: {row['height']}")
        print(f"  items:  ({len(row['items'])})")
        for it in row["items"]:
            print(f"    - {it['label']:<20s}  bbox={it['bbox']}")


if __name__ == "__main__":
    main()
