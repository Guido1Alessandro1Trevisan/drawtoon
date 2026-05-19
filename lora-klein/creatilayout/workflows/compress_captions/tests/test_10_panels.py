"""End-to-end test for the vision+thinking compress workflow.

Picks 10 real pages from `s3://drawtoon/captions/gemini3_flash_page_panel_v1/`,
loads each page image, crops every panel, calls Gemini in parallel, and
prints the assembled short_caption for every panel.

Run:
    cd lora-klein/creatilayout/workflows/compress_captions
    uv run --quiet --with boto3 --with google-genai --with pillow \\
        python tests/test_10_panels.py
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import boto3
from botocore.config import Config
from PIL import Image

from src.assembly import (
    assemble_short_caption,
    attribute_bubbles_to_characters,
    word_count,
)
from src.prompts.compress_prompt import (
    CAMERA_ANGLE_VALUES,
    PANEL_EXTRACT_SCHEMA,
    SHOT_SIZE_VALUES,
    SYSTEM_INSTRUCTION,
    USER_INSTRUCTION,
)


BUCKET = "drawtoon"
SOURCE_PREFIX = "captions/gemini3_flash_page_panel_v1/"
MODEL = "gemini-3-flash-preview"
THINKING_LEVEL = "high"
N_PAGES = 10
SEED = 11
MAX_PANEL_SIDE = 1024
MAX_PARALLEL_PANELS = 24
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
SECRET_ID = "drawtoon/gemini-api-key"


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
    for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=SOURCE_PREFIX, Delimiter="/").get("CommonPrefixes", []):
        path = obj["Prefix"][len(SOURCE_PREFIX):].rstrip("/")
        if path.startswith("_"):
            continue
        chapters.append(path)
    rng.shuffle(chapters)

    chosen: list[str] = []
    for chapter in chapters:
        if len(chosen) >= N_PAGES:
            break
        page = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{SOURCE_PREFIX}{chapter}/", MaxKeys=12)
        keys = [c["Key"] for c in page.get("Contents", []) if c["Key"].endswith(".json")]
        if not keys:
            continue
        chosen.append(rng.choice(keys))
    return chosen


def _load_source(key: str) -> dict:
    body = _s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def _load_page_image(page_key: str) -> Image.Image:
    body = _s3().get_object(Bucket=BUCKET, Key=page_key)["Body"].read()
    return Image.open(io.BytesIO(body)).convert("RGB")


def _crop_panel_png(page_image: Image.Image, panel_bbox: list[int]) -> bytes:
    x0, y0, x1, y1 = [int(round(v)) for v in panel_bbox]
    crop = page_image.crop((x0, y0, x1, y1))
    if max(crop.size) > MAX_PANEL_SIDE:
        crop = crop.copy()
        crop.thumbnail((MAX_PANEL_SIDE, MAX_PANEL_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _extract(*, client, panel_png: bytes) -> tuple[dict, dict]:
    from google.genai import types  # type: ignore

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PANEL_EXTRACT_SCHEMA,
        system_instruction=SYSTEM_INSTRUCTION,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=panel_png, mime_type="image/png"),
            USER_INSTRUCTION,
        ],
        config=config,
    )
    raw = json.loads(resp.text or "{}")
    shot = str(raw.get("shot_size") or "ambiguous")
    angle = str(raw.get("camera_angle") or "ambiguous")
    action = str(raw.get("action_phrase") or "").strip()
    if shot not in SHOT_SIZE_VALUES:
        shot = "ambiguous"
    if angle not in CAMERA_ANGLE_VALUES:
        angle = "ambiguous"
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
        "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
        "thoughts_tokens": int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
    }
    return {"shot_size": shot, "camera_angle": angle, "action_phrase": action}, usage


def main() -> None:
    api_key = _resolve_api_key()
    from google import genai
    from google.genai import types  # noqa: F401

    client = genai.Client(api_key=api_key)

    keys = _pick_pages()
    print(f"Picked {len(keys)} pages (vision + thinking_level={THINKING_LEVEL}):")
    for k in keys:
        print(f"  - s3://{BUCKET}/{k}")
    print()

    tasks: list[dict] = []
    page_info: dict[str, tuple[str, str, tuple[int, int]]] = {}
    for page_idx, key in enumerate(keys, start=1):
        source = _load_source(key)
        chapter = source.get("chapter") or ""
        page_id = source.get("page_id") or Path(key).stem
        page_key = source.get("sources", {}).get("page_key")
        panels_in = source.get("panels") or []
        if not page_key or not panels_in:
            print(f"[{page_idx}/{len(keys)}] {chapter}/{page_id}: missing page or panels, skipping")
            continue
        try:
            page_image = _load_page_image(page_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[{page_idx}/{len(keys)}] {chapter}/{page_id}: page download failed: {exc}")
            continue
        page_info[key] = (chapter, page_id, page_image.size)
        for panel in panels_in:
            idx = int(panel.get("panel_index") or 0)
            panel_bbox = panel.get("bbox")
            if not panel_bbox:
                continue
            try:
                panel_png = _crop_panel_png(page_image, panel_bbox)
            except Exception as exc:  # noqa: BLE001
                print(f"  {chapter}/{page_id} panel {idx}: crop failed: {exc}")
                continue
            tasks.append(
                {
                    "page_key": key,
                    "panel_index": idx,
                    "characters": panel.get("characters") or [],
                    "text_bubbles": panel.get("text_bubbles") or [],
                    "panel_png": panel_png,
                }
            )

    print(
        f"\nQueued {len(tasks)} panels across {len(page_info)} pages, "
        f"running with {MAX_PARALLEL_PANELS} workers...\n"
    )

    results: dict[tuple[str, int], dict] = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
    started = time.time()

    def _run(task: dict) -> tuple[tuple[str, int], dict]:
        extract, usage = _extract(client=client, panel_png=task["panel_png"])
        attributed = attribute_bubbles_to_characters(
            characters=task["characters"],
            text_bubbles=task["text_bubbles"],
        )
        short = assemble_short_caption(
            shot_size=extract["shot_size"],
            camera_angle=extract["camera_angle"],
            action_phrase=extract["action_phrase"],
            character_count=len(task["characters"]),
            attributed_bubbles=attributed,
        )
        return (task["page_key"], task["panel_index"]), {
            "extract": extract,
            "usage": usage,
            "short": short,
            "char_count": len(task["characters"]),
            "bubble_types": [(b.get("type") or "Speech Bubble") for b in task["text_bubbles"]],
        }

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PANELS) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                k, payload = fut.result()
                results[k] = payload
            except Exception as exc:  # noqa: BLE001
                task = futures[fut]
                print(f"  {task['page_key']} panel {task['panel_index']}: failed: {exc}")
            if done % 10 == 0 or done == len(tasks):
                print(f"  ...{done}/{len(tasks)} panels done")

    word_counts: list[int] = []
    over_cap = 0
    total_panels = 0
    for page_idx, key in enumerate(keys, start=1):
        if key not in page_info:
            continue
        chapter, page_id, size = page_info[key]
        print("=" * 100)
        print(f"[{page_idx}/{len(keys)}] {chapter}/{page_id}  (page {size})")
        print("=" * 100)
        source = _load_source(key)
        for panel in source.get("panels") or []:
            idx = int(panel.get("panel_index") or 0)
            payload = results.get((key, idx))
            if payload is None:
                continue
            short = payload["short"]
            wc = word_count(short)
            word_counts.append(wc)
            total_panels += 1
            if wc > 20:
                over_cap += 1
            for k in total_usage:
                total_usage[k] += payload["usage"].get(k, 0)
            print(
                f"\n  Panel {idx}  |  chars={payload['char_count']}  "
                f"bubbles={payload['bubble_types'] or '[]'}  "
                f"think={payload['usage']['thoughts_tokens']}t"
            )
            print(f"    SHORT ({wc}w): {short}")
        print()

    elapsed = time.time() - started
    print("=" * 100)
    print(f"TOTAL — {total_panels} panels across {len(keys)} pages in {elapsed:.1f}s")
    if word_counts:
        word_counts.sort()
        print(
            f"  word counts: min={word_counts[0]}, "
            f"p50={word_counts[len(word_counts)//2]}, "
            f"p90={word_counts[int(len(word_counts)*0.9)]}, "
            f"max={word_counts[-1]}, "
            f"over_cap(>20)={over_cap}"
        )
    print(
        f"  Gemini tokens: in={total_usage['input_tokens']}, "
        f"out={total_usage['output_tokens']}, "
        f"thoughts={total_usage['thoughts_tokens']}"
    )


if __name__ == "__main__":
    main()
