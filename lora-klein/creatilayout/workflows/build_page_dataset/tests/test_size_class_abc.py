"""3-way A/B test for panel size-class classification on 100 random pages.

Approaches:
- A. arithmetic: deterministic from bbox (tuned thresholds: splash > 0.50 area;
     panoramic requires aspect >= 2.5 AND area >= 0.04).
- B. Gemini-6: closed enum of the existing 6 tokens.
- C. Gemini-8: same as B plus `borderless panel` and `impact panel`.

Per-page: one Gemini call for B, one for C, A is computed locally.
Outputs:
- ~/tmp/size_class_abc/results.jsonl  (per-panel triples)
- ~/tmp/size_class_abc/disagree_cases.jsonl  (A != B or A != C)
- prints agreement matrix, confusion, token frequency, sampled disagreements.

Run:
    cd lora-klein/creatilayout/workflows/build_page_dataset
    uv run --quiet --with boto3 --with google-genai --with pillow \\
        python tests/test_size_class_abc.py
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from PIL import Image


BUCKET_SOURCE = "drawtoon"
SOURCE_PREFIX = "captions/gemini3_flash_page_panel_v1/"
MODEL = "gemini-3-flash-preview"
N_PAGES = 100
SEED = 23
MAX_PAGE_SIDE = 1536
MAX_PARALLEL = 24
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
SECRET_ID = "drawtoon/gemini-api-key"

OUT_DIR = Path.home() / "tmp" / "size_class_abc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENS_6 = [
    "splash panel",
    "hero panel",
    "panoramic panel",
    "vertical panel",
    "standard panel",
    "beat panel",
]
TOKENS_8 = TOKENS_6 + ["borderless panel", "impact panel"]


SCHEMA_B = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "panel_index": {"type": "integer"},
                    "size_class": {"type": "string", "format": "enum", "enum": TOKENS_6},
                },
                "required": ["panel_index", "size_class"],
            },
        }
    },
    "required": ["panels"],
}

SCHEMA_C = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "panel_index": {"type": "integer"},
                    "size_class": {"type": "string", "format": "enum", "enum": TOKENS_8},
                },
                "required": ["panel_index", "size_class"],
            },
        }
    },
    "required": ["panels"],
}

SYS_B = (
    "You classify each panel on a manga/manhwa page into ONE of 6 size-class "
    "tokens for a layout supervision model.\n"
    "\n"
    "Definitions (visual + size, in order of priority):\n"
    "- panoramic panel: aspect ratio >= 2.5 (very wide letterbox), used for "
    "establishing shots or sweeping action.\n"
    "- vertical panel: aspect ratio <= 0.4 (very tall, narrow column).\n"
    "- splash panel: a single panel covering >= ~50% of page area; the dominant image.\n"
    "- hero panel: large panel ~30-50% of page area, squarish or moderately wide.\n"
    "- standard panel: ordinary panel ~10-30% of page area, squarish.\n"
    "- beat panel: small panel < 10% of page area (close-ups, reactions, small inserts).\n"
    "\n"
    "Use the panel images + the page image + the bbox_norm coordinates (xyxy in "
    "[0,1] page coords) to pick the best label. If multiple rules apply, pick the "
    "MORE SPECIFIC visual one (panoramic / vertical) before falling back to area buckets.\n"
    "\n"
    "Return one entry per provided panel_index. JSON only."
)

SYS_C = (
    "You classify each panel on a manga/manhwa page into ONE of 8 size-class "
    "tokens for a layout supervision model.\n"
    "\n"
    "Definitions (in order of priority):\n"
    "- borderless panel: panel has NO visible rectangular frame border (the art "
    "bleeds into the page or background, no inked rectangle around it). "
    "Common in modern manga for emotional/mood moments.\n"
    "- impact panel: panel is tilted >= ~8 degrees from page axis OR breaks/cuts "
    "across an adjacent panel's gutter. Used for high-energy hits and action.\n"
    "- panoramic panel: aspect ratio >= 2.5 (very wide letterbox).\n"
    "- vertical panel: aspect ratio <= 0.4 (very tall, narrow column).\n"
    "- splash panel: single panel >= ~50% of page area; dominant image.\n"
    "- hero panel: large panel ~30-50% of page area.\n"
    "- standard panel: ordinary panel ~10-30% of page area.\n"
    "- beat panel: small panel < 10% of page area.\n"
    "\n"
    "Priority: if a panel is BOTH borderless and impact, prefer impact. If it has "
    "a strong shape signal (panoramic/vertical) AND is also borderless, prefer the "
    "shape token (panoramic/vertical). Only assign borderless if there is clearly "
    "no frame line. Only assign impact if the panel is visibly rotated or breaks "
    "a gutter (don't use it for every action panel).\n"
    "\n"
    "Return one entry per provided panel_index. JSON only."
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
    for obj in s3.list_objects_v2(
        Bucket=BUCKET_SOURCE, Prefix=SOURCE_PREFIX, Delimiter="/"
    ).get("CommonPrefixes", []):
        path = obj["Prefix"][len(SOURCE_PREFIX):].rstrip("/")
        if path.startswith("_"):
            continue
        chapters.append(path)
    rng.shuffle(chapters)

    chosen: list[str] = []
    for chapter in chapters:
        if len(chosen) >= N_PAGES:
            break
        page = s3.list_objects_v2(
            Bucket=BUCKET_SOURCE,
            Prefix=f"{SOURCE_PREFIX}{chapter}/",
            MaxKeys=40,
        )
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


def _encode_page(img: Image.Image) -> bytes:
    if max(img.size) > MAX_PAGE_SIDE:
        img = img.copy()
        img.thumbnail((MAX_PAGE_SIDE, MAX_PAGE_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---- A: arithmetic (TUNED) ----

def size_class_arithmetic(panel_bbox: list[float], page_w: int, page_h: int) -> str:
    px0, py0, px1, py1 = panel_bbox
    pw = max(1.0, px1 - px0)
    ph = max(1.0, py1 - py0)
    area = (pw * ph) / max(1, page_w * page_h)
    aspect = pw / ph
    # tuned: panoramic needs both wide AND >=4% area (sliver guard)
    if aspect >= 2.5 and area >= 0.04:
        return "panoramic panel"
    if aspect <= 0.4:
        return "vertical panel"
    # tuned: splash trips at 50% (not 60%)
    if area > 0.50:
        return "splash panel"
    if area > 0.30:
        return "hero panel"
    if area > 0.10:
        return "standard panel"
    return "beat panel"


# ---- Gemini call ----

def _classify_with_gemini(
    *,
    client,
    page_png: bytes,
    panel_descs: list[dict],
    schema: dict,
    sys_instr: str,
) -> tuple[dict[int, str], dict]:
    from google.genai import types  # type: ignore

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=sys_instr,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    user_text = (
        "Classify the size_class for each panel below. Panels are listed with "
        "panel_index and their bbox_norm (xyxy, normalized to page).\n\n"
        + "\n".join(
            f"- panel_index={p['panel_index']} bbox_norm={p['bbox_norm']} "
            f"area_frac={p['area_frac']:.3f} aspect={p['aspect']:.2f}"
            for p in panel_descs
        )
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=page_png, mime_type="image/png"),
            user_text,
        ],
        config=config,
    )
    raw = json.loads(resp.text or "{}")
    out: dict[int, str] = {}
    for entry in raw.get("panels", []):
        try:
            idx = int(entry.get("panel_index"))
            sz = str(entry.get("size_class") or "").strip()
            if sz:
                out[idx] = sz
        except (TypeError, ValueError):
            continue
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
        "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0) if um else 0,
        "thoughts_tokens": int(getattr(um, "thoughts_token_count", 0) or 0) if um else 0,
    }
    return out, usage


def _build_panel_descs(panels: list[dict], page_w: int, page_h: int) -> list[dict]:
    out: list[dict] = []
    for p in panels:
        bbox = p.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        px0, py0, px1, py1 = bbox
        pw = max(1.0, px1 - px0)
        ph = max(1.0, py1 - py0)
        area = (pw * ph) / max(1, page_w * page_h)
        aspect = pw / ph
        bbox_norm = [
            round(px0 / max(1, page_w), 4),
            round(py0 / max(1, page_h), 4),
            round(px1 / max(1, page_w), 4),
            round(py1 / max(1, page_h), 4),
        ]
        out.append(
            {
                "panel_index": int(p.get("panel_index", len(out))),
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "area_frac": area,
                "aspect": aspect,
            }
        )
    return out


def _process_page(client, task: dict) -> dict:
    source = task["source"]
    page_w = int((source.get("page_size") or {}).get("width_px") or 0)
    page_h = int((source.get("page_size") or {}).get("height_px") or 0)
    panels = source.get("panels") or []
    descs = _build_panel_descs(panels, page_w, page_h)
    if not descs:
        return {"key": task["key"], "skipped": True}

    page_png = _encode_page(_load_page_image(task["page_key"]))

    # A: arithmetic
    a_labels = {d["panel_index"]: size_class_arithmetic(d["bbox"], page_w, page_h) for d in descs}

    # B: Gemini-6
    b_labels, usage_b = _classify_with_gemini(
        client=client,
        page_png=page_png,
        panel_descs=descs,
        schema=SCHEMA_B,
        sys_instr=SYS_B,
    )
    # C: Gemini-8
    c_labels, usage_c = _classify_with_gemini(
        client=client,
        page_png=page_png,
        panel_descs=descs,
        schema=SCHEMA_C,
        sys_instr=SYS_C,
    )

    rows: list[dict] = []
    for d in descs:
        idx = d["panel_index"]
        rows.append(
            {
                "key": task["key"],
                "chapter": task["chapter"],
                "page_id": source.get("page_id"),
                "page_key": task["page_key"],
                "panel_index": idx,
                "bbox_norm": d["bbox_norm"],
                "area_frac": round(d["area_frac"], 4),
                "aspect": round(d["aspect"], 3),
                "A": a_labels.get(idx),
                "B": b_labels.get(idx),
                "C": c_labels.get(idx),
            }
        )

    return {
        "key": task["key"],
        "rows": rows,
        "usage": {
            k: usage_b.get(k, 0) + usage_c.get(k, 0)
            for k in ("input_tokens", "output_tokens", "thoughts_tokens")
        },
    }


def main() -> None:
    api_key = _resolve_api_key()
    if not api_key:
        print("ERROR: no Gemini API key", file=sys.stderr)
        sys.exit(2)

    from google import genai

    client = genai.Client(api_key=api_key)

    keys = _pick_pages()
    print(f"Picked {len(keys)} pages (seed={SEED}).\n")

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
                "chapter": source.get("chapter") or key.split("/")[2],
            }
        )
    print(f"Filtered to {len(tasks)} usable pages.\n")

    all_rows: list[dict] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
    started = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_process_page, client, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  fail {futures[fut]['key']}: {exc}")
                continue
            if out.get("skipped"):
                continue
            all_rows.extend(out["rows"])
            for k in usage_totals:
                usage_totals[k] += out["usage"].get(k, 0)
            if done % 10 == 0 or done == len(tasks):
                print(f"  ...{done}/{len(tasks)} pages")

    elapsed = time.time() - started
    print(f"\nProcessed {len(all_rows)} panels across {done} pages in {elapsed:.1f}s")
    print(
        f"Gemini tokens (B+C combined): in={usage_totals['input_tokens']:,} "
        f"out={usage_totals['output_tokens']:,} think={usage_totals['thoughts_tokens']:,}"
    )

    # Persist raw rows
    out_path = OUT_DIR / "results.jsonl"
    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path}")

    # ---- Agreement stats ----
    n = len(all_rows)
    ab = sum(1 for r in all_rows if r["A"] is not None and r["B"] is not None and r["A"] == r["B"])
    bc = sum(1 for r in all_rows if r["B"] is not None and r["C"] is not None and r["B"] == r["C"])
    ac = sum(1 for r in all_rows if r["A"] is not None and r["C"] is not None and r["A"] == r["C"])
    n_ab = sum(1 for r in all_rows if r["A"] is not None and r["B"] is not None)
    n_bc = sum(1 for r in all_rows if r["B"] is not None and r["C"] is not None)
    n_ac = sum(1 for r in all_rows if r["A"] is not None and r["C"] is not None)

    print("\n=== AGREEMENT ===")
    print(f"  A vs B: {ab}/{n_ab} = {100*ab/max(1,n_ab):.1f}%")
    print(f"  B vs C: {bc}/{n_bc} = {100*bc/max(1,n_bc):.1f}%")
    print(f"  A vs C: {ac}/{n_ac} = {100*ac/max(1,n_ac):.1f}%")

    # ---- Confusion A vs B ----
    print("\n=== CONFUSION A (rows) vs B (cols) ===")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in all_rows:
        if r["A"] and r["B"]:
            confusion[r["A"]][r["B"]] += 1
    cols = TOKENS_6
    header = "  A\\B".ljust(18) + " ".join(c[:6].rjust(7) for c in cols)
    print(header)
    for row_lab in TOKENS_6:
        line = ("  " + row_lab[:16]).ljust(18)
        for col_lab in cols:
            line += str(confusion[row_lab].get(col_lab, 0)).rjust(7) + " "
        print(line)

    # ---- Token frequency in C ----
    print("\n=== TOKEN FREQUENCY (C, 8-token) ===")
    cfreq = Counter(r["C"] for r in all_rows if r["C"])
    total_c = sum(cfreq.values())
    for tok in TOKENS_8:
        c = cfreq.get(tok, 0)
        print(f"  {tok:<20s} {c:>5d}  {100*c/max(1,total_c):5.1f}%")

    # ---- Token frequency for A and B too ----
    print("\n=== TOKEN FREQUENCY (A, arithmetic) ===")
    afreq = Counter(r["A"] for r in all_rows if r["A"])
    total_a = sum(afreq.values())
    for tok in TOKENS_6:
        c = afreq.get(tok, 0)
        print(f"  {tok:<20s} {c:>5d}  {100*c/max(1,total_a):5.1f}%")

    print("\n=== TOKEN FREQUENCY (B, Gemini-6) ===")
    bfreq = Counter(r["B"] for r in all_rows if r["B"])
    total_b = sum(bfreq.values())
    for tok in TOKENS_6:
        c = bfreq.get(tok, 0)
        print(f"  {tok:<20s} {c:>5d}  {100*c/max(1,total_b):5.1f}%")

    # ---- Disagreement cases ----
    disagree = [r for r in all_rows if (r["A"] and r["B"] and r["A"] != r["B"]) or (r["A"] and r["C"] and r["A"] != r["C"])]
    print(f"\n=== DISAGREEMENTS (A vs B or A vs C): {len(disagree)}/{n} = {100*len(disagree)/max(1,n):.1f}% ===")

    # Sample 20 disagreement cases for inspection
    rng = random.Random(SEED + 7)
    sample = rng.sample(disagree, min(20, len(disagree)))
    dis_path = OUT_DIR / "disagree_cases.jsonl"
    with dis_path.open("w") as f:
        for r in disagree:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {dis_path}  ({len(disagree)} rows)")

    print("\n--- 20 sampled disagreement cases ---")
    for r in sample:
        print(
            f"  {r['chapter']}/{r['page_id']} panel={r['panel_index']:<2d} "
            f"area={r['area_frac']:.3f} aspect={r['aspect']:.2f}  "
            f"A={r['A']:<16s} B={r['B']:<16s} C={r['C']}"
        )


if __name__ == "__main__":
    main()
