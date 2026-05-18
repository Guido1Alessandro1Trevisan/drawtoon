#!/usr/bin/env python3
"""One-shot FLUX.2 Klein validation run on Modal.

Run one model at a time:

    modal run lora-klein/validation/validate_run.py --base true --eval-id klein_eval_001
    modal run lora-klein/validation/validate_run.py --base false --checkpoint-uri s3://... --eval-id klein_eval_001

Each run samples or reuses 200 fixed manifest rows, fans out to 10 H100
containers, generates 20 panels per container, then evaluates generated panels
with the internal CMMD/SigLIP2/DINOv3 metric suite plus optional Haiku bubble
judging. Existing generated images can be re-evaluated without regeneration.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("DRAWTOON_VALIDATION_APP_NAME", "drawtoon-flux2-klein-validation")
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
S3_BUCKET = os.environ.get("DRAWTOON_S3_BUCKET") or os.environ.get("S3_BUCKET") or "drawtoon"
S3_OUTPUT_PREFIX = os.environ.get("DRAWTOON_VALIDATION_PREFIX", "validation/flux2_klein_panel_eval")
DEFAULT_CAPTION_RUN = "haiku45_mangazero_page_panel_v1"
DEFAULT_INCLUDE_CHAPTER_REGEX = "_mangazero$"
DEFAULT_PAGES_PREFIX = "datasets/pages/filtered"
DEFAULT_CAPTIONS_PREFIX = "captions"
DEFAULT_ANNOTATIONS_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_SAMPLE_COUNT = 200
DEFAULT_SHARD_COUNT = 10
DEFAULT_HAIKU_SHARD_COUNT = 200
DEFAULT_METRIC_BATCH_SIZE = 256
DEFAULT_VALIDATION_DATASET = "generalist"
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
DEFAULT_SEED = 20260515
DEFAULT_HF_SECRET_NAME = "drawtoon-flux2-inference-hf-token"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"
DINOV3_MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
DINOV3_TIMM_MODEL_NAME = "hf_hub:timm/vit_base_patch16_dinov3.lvd1689m"
CMMD_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0
DEFAULT_HAIKU_BUBBLE_MODEL = os.environ.get(
    "DRAWTOON_HAIKU_BUBBLE_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "1800"))
MAX_CONTROL_IMAGE_SLOTS = 7
MAX_CHARACTER_REFS = 5
SCORED_TEXT_BUBBLE_TYPES = ("Speech Bubble", "Narration Bubble", "Shout Bubble")
REMOTE_VALIDATION_DATASETS_ROOT = "/root/validation_datasets"
LAYOUT_TEXT_COLORS = {
    "Speech Bubble": (0, 96, 255),
    "Narration Bubble": (255, 128, 0),
    "Shout Bubble": (128, 0, 255),
}
LAYOUT_TEXT_COLOR = LAYOUT_TEXT_COLORS["Speech Bubble"]
LAYOUT_BACKGROUND_COLOR = (0, 0, 0)
LAYOUT_CHARACTER_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (255, 220, 0),
    (255, 0, 255),
    (0, 220, 255),
]
LAYOUT_CHARACTER_COLOR_NAMES = ["red", "green", "yellow", "magenta", "cyan"]
LAYOUT_CHARACTER_OUTLINE_WIDTHS = [12, 10, 8, 6, 4]
CHARACTER_REF_BORDER_WIDTH = 3

VALIDATION_DIR = Path(__file__).parent.resolve()
LORA_ROOT = VALIDATION_DIR.parent
DOCKERFILE_PATH = LORA_ROOT / "Dockerfile.modal"

image = (
    modal.Image.from_dockerfile(str(DOCKERFILE_PATH))
    .env(
        {
            "HF_HOME": "/root/validation_cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/root/validation_cache/huggingface/hub",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/ai-toolkit",
        }
    )
    .add_local_dir(str(LORA_ROOT / "ai-toolkit"), remote_path="/root/ai-toolkit", copy=False)
    .add_local_dir(str(VALIDATION_DIR / "datasets"), remote_path=REMOTE_VALIDATION_DATASETS_ROOT, copy=False)
)

app = modal.App(APP_NAME, image=image)
aws_secret = modal.Secret.from_name(os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3"))
hf_secret = modal.Secret.from_name(os.environ.get("DRAWTOON_HF_SECRET_NAME", DEFAULT_HF_SECRET_NAME))
validation_cache = modal.Volume.from_name("flux-klein-validation-cache", create_if_missing=True)

BUBBLE_TYPE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "target_description": {
            "type": "string",
            "description": "Brief visible description of the target panel text containers.",
        },
        "generated_description": {
            "type": "string",
            "description": "Brief visible description of the generated panel text containers.",
        },
        "overall_respected_percent": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Percent of target speech/narration/shout regions whose type is respected in the generated image.",
        },
        "type_scores": {
            "type": "object",
            "properties": {
                "Speech Bubble": {
                    "type": "object",
                    "properties": {
                        "expected_count": {"type": "integer", "minimum": 0},
                        "respected_count": {"type": "integer", "minimum": 0},
                        "respected_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "notes": {"type": "string"},
                    },
                    "required": ["expected_count", "respected_count", "respected_percent", "notes"],
                    "additionalProperties": False,
                },
                "Narration Bubble": {
                    "type": "object",
                    "properties": {
                        "expected_count": {"type": "integer", "minimum": 0},
                        "respected_count": {"type": "integer", "minimum": 0},
                        "respected_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "notes": {"type": "string"},
                    },
                    "required": ["expected_count", "respected_count", "respected_percent", "notes"],
                    "additionalProperties": False,
                },
                "Shout Bubble": {
                    "type": "object",
                    "properties": {
                        "expected_count": {"type": "integer", "minimum": 0},
                        "respected_count": {"type": "integer", "minimum": 0},
                        "respected_percent": {"type": "number", "minimum": 0, "maximum": 100},
                        "notes": {"type": "string"},
                    },
                    "required": ["expected_count", "respected_count", "respected_percent", "notes"],
                    "additionalProperties": False,
                },
            },
            "required": ["Speech Bubble", "Narration Bubble", "Shout Bubble"],
            "additionalProperties": False,
        },
        "region_judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_region_index": {"type": "integer"},
                    "expected_type": {"type": "string", "enum": list(SCORED_TEXT_BUBBLE_TYPES)},
                    "generated_type": {
                        "type": "string",
                        "enum": ["Speech Bubble", "Narration Bubble", "Shout Bubble", "None", "Missing", "Ambiguous"],
                    },
                    "respected": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "string",
                        "description": "Short visible evidence for the judgement.",
                    },
                },
                "required": [
                    "text_region_index",
                    "expected_type",
                    "generated_type",
                    "respected",
                    "confidence",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "target_description",
        "generated_description",
        "overall_respected_percent",
        "type_scores",
        "region_judgements",
    ],
    "additionalProperties": False,
}

BUBBLE_TYPE_JUDGE_SYSTEM_PROMPT = """You judge whether generated manga panels preserve text-bubble container types from a target panel.

Return structured output only with the requested tool.
Compare visible bubble shapes and text containers, not text content or story meaning.
Speech Bubble: a smooth round or oval spoken-dialogue balloon, often with a tail or pointer toward a speaker.
Narration Bubble: a square or rectangular caption/narration box with straight sides or corners, usually no speaker tail.
Shout Bubble: a jagged, spiky, starburst, or angular burst balloon used for shouting or emphasis.
None: free-floating text, signs, SFX, sound effects, labels, or scene text not inside one of the three bubble containers.
Missing: the generated image has no visible corresponding text container for that target region.
Ambiguous: a visible text container exists, but the type cannot be determined.
Use the target image and target-region list as ground truth. The first image is the target panel. The second image is the generated panel.
Judge whether each target speech/narration/shout region appears in the generated image with the same type. Do not reward a generated bubble if it is the wrong type, even if it is near the right location."""


def s3_uri(*parts: str) -> str:
    clean = [part.strip("/") for part in parts if part and part.strip("/")]
    return f"s3://{S3_BUCKET}/{'/'.join(clean)}" if clean else f"s3://{S3_BUCKET}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return bucket, key


def stable_hash(value: Any, seed: int = DEFAULT_SEED) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(f"{seed}:{payload}".encode("utf-8")).hexdigest()


def sanitize_filename(value: str, fallback: str = "item") -> str:
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value or "").strip()).strip("_")
    return safe[:180] or fallback


def sanitize_dataset_name(value: str, fallback: str = DEFAULT_VALIDATION_DATASET) -> str:
    return sanitize_filename(value, fallback=fallback)


def validation_dataset_root(dataset: str) -> Path:
    return Path(REMOTE_VALIDATION_DATASETS_ROOT) / sanitize_dataset_name(dataset)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def upload_json(s3_client, payload: Any, uri: str) -> None:
    bucket, key = parse_s3_uri(uri)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def upload_jsonl(s3_client, rows: list[dict[str, Any]], uri: str) -> None:
    bucket, key = parse_s3_uri(uri)
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/jsonl")


def load_jsonl_s3(s3_client, uri: str) -> list[dict[str, Any]]:
    bucket, key = parse_s3_uri(uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    rows: list[dict[str, Any]] = []
    for line in obj["Body"].read().decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def object_exists(s3_client, uri: str) -> bool:
    bucket, key = parse_s3_uri(uri)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def download_s3_uri(s3_client, uri: str, destination: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = int(s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        if destination.exists() and destination.stat().st_size == size:
            return destination
    except Exception:
        pass
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    s3_client.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(destination)
    return destination


def cache_path_for_s3(uri: str, root: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    digest = hashlib.sha1(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:18]
    suffix = Path(key).suffix or ".bin"
    return root / digest[:2] / f"{Path(key).stem[:96]}_{digest}{suffix}"


def cache_path_for_structured(asset_ref: dict[str, Any], root: Path, suffix: str = ".png") -> Path:
    digest = hashlib.sha1(json.dumps(asset_ref, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return root / digest[:2] / f"{digest}{suffix}"


def coerce_pixel_box(box: Any, *, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def box_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def panel_relative_norm_box(
    entity_box: tuple[float, float, float, float],
    panel_box: tuple[float, float, float, float],
    *,
    padded_width: int,
    padded_height: int,
    min_entity_overlap: float = 0.05,
) -> list[float] | None:
    ex0, ey0, ex1, ey1 = entity_box
    px0, py0, px1, py1 = panel_box
    ix0, iy0 = max(ex0, px0), max(ey0, py0)
    ix1, iy1 = min(ex1, px1), min(ey1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    if box_area((ix0, iy0, ix1, iy1)) / max(1.0, box_area(entity_box)) < min_entity_overlap:
        return None
    return [
        round((ix0 - px0) / max(1, padded_width), 6),
        round((iy0 - py0) / max(1, padded_height), 6),
        round((ix1 - px0) / max(1, padded_width), 6),
        round((iy1 - py0) / max(1, padded_height), 6),
    ]


def normalize_layout_text_bubble_type(value: object) -> str:
    text = " ".join(str(value or "").split()).strip().lower()
    if text in {"", "none", "null", "no", "not a bubble"}:
        return "None"
    if any(token in text for token in ("free", "floating", "standalone", "sign", "sfx")):
        return "None"
    if "narration" in text or "caption" in text:
        return "Narration Bubble"
    if any(token in text for token in ("shout", "yell", "scream", "burst")):
        return "Shout Bubble"
    if any(token in text for token in ("speech", "dialogue", "dialog", "bubble")):
        return "Speech Bubble"
    return "None"


def panel_text_regions(panel: dict[str, Any]) -> list[Any]:
    for field_name in ("text_bubbles", "speech_bubbles", "text_regions", "texts", "bubbles"):
        regions = panel.get(field_name)
        if isinstance(regions, list):
            return regions
    return []


def caption_page_key(caption_payload: dict[str, Any], *, pages_prefix: str) -> str:
    sources = caption_payload.get("sources") if isinstance(caption_payload.get("sources"), dict) else {}
    page_key = str(sources.get("page_key") or "").strip()
    chapter = str(caption_payload.get("chapter") or "").strip()
    page_id = str(caption_payload.get("page_id") or "").strip()
    if not chapter or not page_id:
        raise ValueError("Caption payload is missing sources.page_key and chapter/page_id fallback fields")
    normalized_prefix = str(pages_prefix).strip().strip("/")
    if page_key and page_key.startswith(f"{normalized_prefix}/"):
        return page_key
    extension = ".png" if "text_removed" in normalized_prefix.split("/") else (Path(page_key).suffix or ".jpg")
    return f"{normalized_prefix}/{chapter}/{page_id}{extension}"


def caption_page_size(caption_payload: dict[str, Any]) -> tuple[int, int] | None:
    page_size = caption_payload.get("page_size")
    if not isinstance(page_size, dict):
        return None
    try:
        width = int(page_size.get("width_px") or page_size.get("width") or 0)
        height = int(page_size.get("height_px") or page_size.get("height") or 0)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def caption_panels(caption_payload: dict[str, Any]) -> list[Any]:
    panels = caption_payload.get("panels")
    return panels if isinstance(panels, list) else []


def panel_character_contexts(panels: list[Any], *, page_width: int, page_height: int) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        panel_index = int(panel.get("panel_index") or 0)
        panel_box = coerce_pixel_box(panel.get("bbox"), width=page_width, height=page_height)
        if panel_box is None:
            panel_box = coerce_pixel_box(panel.get("bbox_norm"), width=page_width, height=page_height)
        if panel_box is None:
            continue
        characters = []
        for character in panel.get("characters") or []:
            if not isinstance(character, dict):
                continue
            char_box = coerce_pixel_box(character.get("bbox"), width=page_width, height=page_height)
            if char_box is None:
                char_box = coerce_pixel_box(character.get("bbox_norm"), width=page_width, height=page_height)
            if char_box is not None:
                characters.append({**character, "_pixel_box": char_box, "_panel_index": panel_index})
        contexts.append({**panel, "_pixel_box": panel_box, "_characters": characters, "_panel_index": panel_index})
    return contexts


def select_character_reference(
    *,
    target_character: dict[str, Any],
    target_panel_index: int,
    panel_contexts: list[dict[str, Any]],
    seed_key: str,
) -> tuple[dict[str, Any], str]:
    target_source_id = str(target_character.get("source_character_id") or "").strip()
    same_character_other_panel: list[dict[str, Any]] = []
    if target_source_id:
        for panel in panel_contexts:
            candidate_panel_index = int(panel.get("_panel_index") or 0)
            if candidate_panel_index == target_panel_index:
                continue
            for character in panel.get("_characters") or []:
                if str(character.get("source_character_id") or "").strip() == target_source_id:
                    same_character_other_panel.append(character)
    if same_character_other_panel:
        same_character_other_panel.sort(key=lambda item: stable_hash([seed_key, item.get("_pixel_box")]))
        return same_character_other_panel[0], "same_character_stable_other_panel"
    return target_character, "same_character_target_panel_fallback"


def build_panel_layout_control(
    *,
    panel: dict[str, Any],
    target_box: tuple[float, float, float, float],
    page_width: int,
    page_height: int,
    padded_width: int,
    padded_height: int,
    character_layout_items: list[dict[str, Any]],
) -> dict[str, Any]:
    drawn_characters: list[dict[str, Any]] = []
    for item in character_layout_items[:MAX_CHARACTER_REFS]:
        box_norm = item.get("bbox_norm")
        color = item.get("rgb")
        if isinstance(box_norm, list) and isinstance(color, tuple):
            drawn_characters.append(
                {
                    "character_label": item.get("character_label"),
                    "control_slot": item.get("control_slot"),
                    "color": item.get("color"),
                    "rgb": list(color),
                    "line_width": int(item.get("line_width") or 6),
                    "ref_border_width": CHARACTER_REF_BORDER_WIDTH,
                    "bbox_norm": box_norm,
                    "source_character_id": item.get("source_character_id"),
                    "ref_policy": item.get("ref_policy"),
                }
            )

    text_count = 0
    type_counts = {"Speech Bubble": 0, "Narration Bubble": 0, "Shout Bubble": 0, "None": 0}
    regions: list[dict[str, Any]] = []
    for text_region in panel_text_regions(panel):
        if not isinstance(text_region, dict):
            continue
        text_box = coerce_pixel_box(text_region.get("bbox"), width=page_width, height=page_height)
        if text_box is None:
            text_box = coerce_pixel_box(text_region.get("bbox_norm"), width=page_width, height=page_height)
        if text_box is None:
            continue
        text_box_norm = panel_relative_norm_box(
            text_box,
            target_box,
            padded_width=padded_width,
            padded_height=padded_height,
        )
        if text_box_norm is None:
            continue
        bubble_type = normalize_layout_text_bubble_type(text_region.get("type"))
        type_counts[bubble_type] = int(type_counts.get(bubble_type, 0)) + 1
        masked = bubble_type != "None"
        if masked:
            text_count += 1
        regions.append(
            {
                "text_region_index": text_region.get("text_region_index"),
                "type": bubble_type,
                "bbox_norm": text_box_norm,
                "masked": masked,
            }
        )

    return {
        "control_slot": 1,
        "text": {
            "color": "type_color",
            "rgb": list(LAYOUT_TEXT_COLOR),
            "colors": {text_type: list(rgb) for text_type, rgb in LAYOUT_TEXT_COLORS.items()},
            "count": text_count,
            "type_counts": type_counts,
            "regions": regions,
        },
        "characters": drawn_characters,
    }


def iter_panel_rows_for_caption(
    s3_client,
    *,
    bucket: str,
    caption_key: str,
    pages_prefix: str,
    annotations_prefix: str,
    target_multiple: int = 16,
) -> list[dict[str, Any]]:
    from PIL import Image

    caption_obj = s3_client.get_object(Bucket=bucket, Key=caption_key)
    caption_payload = json.loads(caption_obj["Body"].read().decode("utf-8"))
    if str(caption_payload.get("status") or "") != "ok":
        return []

    chapter = str(caption_payload.get("chapter") or "").strip()
    page_id = str(caption_payload.get("page_id") or "").strip()
    panels = caption_panels(caption_payload)
    if not chapter or not page_id or not panels:
        return []

    page_key = caption_page_key(caption_payload, pages_prefix=pages_prefix)
    page_size = caption_page_size(caption_payload)
    if page_size is None:
        page_bytes = s3_client.get_object(Bucket=bucket, Key=page_key)["Body"].read()
        with Image.open(io.BytesIO(page_bytes)) as page_image:
            page_width, page_height = page_image.size
    else:
        page_width, page_height = page_size

    panel_contexts = panel_character_contexts(panels, page_width=page_width, page_height=page_height)
    rows: list[dict[str, Any]] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        panel_index = int(panel.get("panel_index") or 0)
        panel_box = coerce_pixel_box(panel.get("bbox"), width=page_width, height=page_height)
        if panel_box is None:
            panel_box = coerce_pixel_box(panel.get("bbox_norm"), width=page_width, height=page_height)
        caption = str(panel.get("caption") or "").strip()
        if panel_box is None or not caption:
            continue

        px0, py0, px1, py1 = panel_box
        crop_box = [int(math.floor(px0)), int(math.floor(py0)), int(math.ceil(px1)), int(math.ceil(py1))]
        target_box = tuple(float(value) for value in crop_box)
        padded_width = int(math.ceil(max(1, crop_box[2] - crop_box[0]) / target_multiple) * target_multiple)
        padded_height = int(math.ceil(max(1, crop_box[3] - crop_box[1]) / target_multiple) * target_multiple)
        control_paths: list[dict[str, Any]] = []
        character_layout_items: list[dict[str, Any]] = []

        for character in panel.get("characters") or []:
            if len(control_paths) >= MAX_CHARACTER_REFS or not isinstance(character, dict):
                continue
            char_box = coerce_pixel_box(character.get("bbox"), width=page_width, height=page_height)
            if char_box is None:
                char_box = coerce_pixel_box(character.get("bbox_norm"), width=page_width, height=page_height)
            if char_box is None:
                continue
            char_box_norm = panel_relative_norm_box(
                char_box,
                target_box,
                padded_width=padded_width,
                padded_height=padded_height,
            )
            if char_box_norm is None:
                continue
            character_index = len(control_paths)
            ref_character, ref_policy = select_character_reference(
                target_character={**character, "_pixel_box": char_box, "_panel_index": panel_index},
                target_panel_index=panel_index,
                panel_contexts=panel_contexts,
                seed_key=f"{caption_key}:{panel_index}:{character_index}",
            )
            rx0, ry0, rx1, ry1 = ref_character["_pixel_box"]
            control_paths.append(
                {
                    "image": f"s3://{bucket}/{page_key}",
                    "crop_box": [int(math.floor(rx0)), int(math.floor(ry0)), int(math.ceil(rx1)), int(math.ceil(ry1))],
                    "pad_multiple": target_multiple,
                    "border_rgb": list(LAYOUT_CHARACTER_COLORS[character_index]),
                    "border_width": CHARACTER_REF_BORDER_WIDTH,
                }
            )
            character_layout_items.append(
                {
                    "character_label": f"Character {character_index + 1}",
                    "control_slot": character_index + 2,
                    "bbox_norm": char_box_norm,
                    "color": LAYOUT_CHARACTER_COLOR_NAMES[character_index],
                    "rgb": LAYOUT_CHARACTER_COLORS[character_index],
                    "line_width": LAYOUT_CHARACTER_OUTLINE_WIDTHS[character_index],
                    "source_character_id": str(character.get("source_character_id") or ""),
                    "ref_policy": ref_policy,
                }
            )

        if not control_paths:
            continue

        layout_metadata = build_panel_layout_control(
            panel=panel,
            target_box=target_box,
            page_width=page_width,
            page_height=page_height,
            padded_width=padded_width,
            padded_height=padded_height,
            character_layout_items=character_layout_items,
        )
        rows.append(
            {
                "schema_version": 1,
                "source": "drawtoon",
                "sample_type": "panel_prediction",
                "sample_id": f"{chapter}/{page_id}/panel_{panel_index:03d}",
                "chapter": chapter,
                "page_id": page_id,
                "page_root": f"{chapter}/{page_id}",
                "panel_index": panel_index,
                "source_page": f"s3://{bucket}/{page_key}",
                "source_caption": f"s3://{bucket}/{caption_key}",
                "source_annotation": f"s3://{bucket}/{annotations_prefix.rstrip('/')}/{chapter}/{page_id}.jsonl",
                "target_panel": {
                    "image": f"s3://{bucket}/{page_key}",
                    "crop_box": crop_box,
                    "pad_multiple": target_multiple,
                },
                "target_width": padded_width,
                "target_height": padded_height,
                "caption": caption.rstrip(".") + ".",
                "character_count": len(control_paths),
                "controls": {
                    "layout_control": layout_metadata,
                    "character_ref_paths": control_paths,
                    "has_previous_control": False,
                    "character_ref_policy": "same_character_stable_other_panel_else_target_panel",
                },
            }
        )
    return rows


def row_stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    char_count = int(row.get("character_count") or 0)
    char_bucket = "1" if char_count <= 1 else "2" if char_count == 2 else "3plus"
    text_payload = ((row.get("controls") or {}).get("layout_control") or {}).get("text") or {}
    text_count = int(text_payload.get("count") or 0)
    text_bucket = "0" if text_count == 0 else "1" if text_count == 1 else "2plus"
    width = max(1, int(row.get("target_width") or 1))
    height = max(1, int(row.get("target_height") or 1))
    ratio = width / height
    aspect = "landscape" if ratio > 1.2 else "portrait" if ratio < 0.83 else "square"
    return char_bucket, text_bucket, aspect


def select_eval_rows(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("sample_type") == "panel_prediction"
        and row.get("caption")
        and row.get("target_panel")
        and (row.get("controls") or {}).get("layout_control")
        and (row.get("controls") or {}).get("character_ref_paths")
    ]
    eligible.sort(key=lambda row: stable_hash(row.get("sample_id"), seed))

    quotas = {"1": int(count * 0.45), "2": int(count * 0.35)}
    quotas["3plus"] = count - quotas["1"] - quotas["2"]
    chapter_cap = max(3, math.ceil(count / 50))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    chapter_counts: dict[str, int] = {}

    def can_take(row: dict[str, Any]) -> bool:
        sample_id = str(row.get("sample_id") or "")
        chapter = str(row.get("chapter") or sample_id.split("/", 1)[0])
        return sample_id not in selected_ids and chapter_counts.get(chapter, 0) < chapter_cap

    def take(row: dict[str, Any]) -> None:
        sample_id = str(row.get("sample_id") or "")
        chapter = str(row.get("chapter") or sample_id.split("/", 1)[0])
        selected.append(row)
        selected_ids.add(sample_id)
        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1

    for char_bucket, quota in quotas.items():
        bucket_rows = [row for row in eligible if row_stratum(row)[0] == char_bucket]
        for row in bucket_rows:
            if len([r for r in selected if row_stratum(r)[0] == char_bucket]) >= quota:
                break
            if can_take(row):
                take(row)

    for row in eligible:
        if len(selected) >= count:
            break
        if can_take(row):
            take(row)

    if len(selected) < count:
        for row in eligible:
            if len(selected) >= count:
                break
            if str(row.get("sample_id") or "") not in selected_ids:
                take(row)

    if len(selected) < count:
        raise RuntimeError(f"Only selected {len(selected)} eligible rows, need {count}.")
    selected = selected[:count]
    for index, row in enumerate(selected):
        row["eval_index"] = index
        row["eval_seed"] = seed + index
        row["eval_stratum"] = "/".join(row_stratum(row))
    return selected


def draw_layout_region(
    draw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    line_width: int = 6,
) -> None:
    if not isinstance(box_norm, (list, tuple)) or len(box_norm) != 4:
        return
    x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
    y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
    x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
    if x1 > x0 and y1 > y0:
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, int(line_width)))


def draw_text_layout_shape(draw, box_norm: list[float], *, width: int, height: int, text_bubble_type: str) -> None:
    if not isinstance(box_norm, (list, tuple)) or len(box_norm) != 4:
        return
    normalized = " ".join(str(text_bubble_type or "").split()).strip()
    if normalized in {"", "None", "null"}:
        return
    text_color = LAYOUT_TEXT_COLORS.get(normalized)
    if text_color is None:
        return
    x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
    y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
    x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
    if x1 <= x0 or y1 <= y0:
        return
    draw.rectangle((x0, y0, x1, y1), fill=text_color)


def materialize_layout_control(layout_metadata: dict[str, Any], *, width: int, height: int) -> Any:
    from PIL import Image, ImageDraw

    layout = Image.new("RGB", (width, height), LAYOUT_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(layout)
    for character in layout_metadata.get("characters") or []:
        if isinstance(character, dict) and isinstance(character.get("rgb"), list):
            draw_layout_region(
                draw,
                character.get("bbox_norm"),
                width=width,
                height=height,
                color=tuple(int(value) for value in character["rgb"]),
                line_width=int(character.get("line_width") or 6),
            )
    text_payload = layout_metadata.get("text") if isinstance(layout_metadata.get("text"), dict) else {}
    for region in text_payload.get("regions") or []:
        if isinstance(region, dict) and region.get("masked", True):
            draw_text_layout_shape(
                draw,
                region.get("bbox_norm"),
                width=width,
                height=height,
                text_bubble_type=str(region.get("type") or "Speech Bubble"),
            )
    return layout


def materialize_asset(s3_client, asset_ref: Any, *, asset_root: Path) -> Path:
    from PIL import Image, ImageOps

    if isinstance(asset_ref, dict):
        image_ref = asset_ref.get("image") or asset_ref.get("path")
        if not image_ref:
            raise ValueError(f"Structured asset missing image/path: {asset_ref}")
        output_path = cache_path_for_structured(asset_ref, asset_root)
        if output_path.exists():
            return output_path
        source = materialize_asset(s3_client, image_ref, asset_root=asset_root)
        with Image.open(source) as image:
            image = image.convert("RGB")
            crop_box = asset_ref.get("crop_box") or asset_ref.get("bbox")
            if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
                image = image.crop(tuple(int(round(float(v))) for v in crop_box))
            pad_multiple = int(asset_ref.get("pad_multiple") or 0)
            border_width = int(asset_ref.get("border_width") or 0)
            border_rgb = asset_ref.get("border_rgb")
            if border_width > 0 and isinstance(border_rgb, (list, tuple)) and len(border_rgb) == 3:
                if pad_multiple > 1:
                    inner_width = int(math.ceil((image.width + border_width * 2) / pad_multiple) * pad_multiple) - border_width * 2
                    inner_height = int(math.ceil((image.height + border_width * 2) / pad_multiple) * pad_multiple) - border_width * 2
                    if inner_width != image.width or inner_height != image.height:
                        padded = Image.new("RGB", (max(image.width, inner_width), max(image.height, inner_height)), "white")
                        padded.paste(image, (0, 0))
                        image = padded
                image = ImageOps.expand(image, border=border_width, fill=tuple(int(value) for value in border_rgb))
            elif pad_multiple > 1:
                padded_width = int(math.ceil(image.width / pad_multiple) * pad_multiple)
                padded_height = int(math.ceil(image.height / pad_multiple) * pad_multiple)
                if (padded_width, padded_height) != image.size:
                    padded = Image.new("RGB", (padded_width, padded_height), "white")
                    padded.paste(image, (0, 0))
                    image = padded
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG", optimize=True)
        return output_path

    raw = str(asset_ref).strip()
    if raw.startswith("s3://"):
        return download_s3_uri(s3_client, raw, cache_path_for_s3(raw, asset_root))
    path = Path(raw)
    if path.is_file():
        return path
    raise ValueError(f"Unsupported asset reference: {asset_ref!r}")


def current_control_encoding_row(row: dict[str, Any]) -> dict[str, Any]:
    encoded = json.loads(json.dumps(row, ensure_ascii=False))
    controls = encoded.get("controls")
    if not isinstance(controls, dict):
        return encoded

    layout = controls.get("layout_control")
    if isinstance(layout, dict):
        characters = layout.get("characters")
        if isinstance(characters, list):
            for index, character in enumerate(characters[:MAX_CHARACTER_REFS]):
                if not isinstance(character, dict):
                    continue
                if index < len(LAYOUT_CHARACTER_COLORS):
                    character.setdefault("rgb", list(LAYOUT_CHARACTER_COLORS[index]))
                    character.setdefault("color", LAYOUT_CHARACTER_COLOR_NAMES[index])
                    character["line_width"] = LAYOUT_CHARACTER_OUTLINE_WIDTHS[index]
                    character["ref_border_width"] = CHARACTER_REF_BORDER_WIDTH
        text_payload = layout.get("text")
        if isinstance(text_payload, dict):
            text_payload["color"] = "type_color"
            text_payload["rgb"] = list(LAYOUT_TEXT_COLOR)
            text_payload["colors"] = {text_type: list(rgb) for text_type, rgb in LAYOUT_TEXT_COLORS.items()}
            text_payload["types"] = {
                "Speech Bubble": "solid blue filled rectangle",
                "Narration Bubble": "solid orange filled rectangle",
                "Shout Bubble": "solid violet filled rectangle",
                "None": "no mask",
            }
            for region in text_payload.get("regions") or []:
                if isinstance(region, dict) and region.get("masked", True):
                    region["shape"] = "filled_rectangle"
                    region["preserve_text"] = False
        controls.pop("layout_control_path", None)

    encoded_refs: list[Any] = []
    for index, ref in enumerate((controls.get("character_ref_paths") or [])[:MAX_CHARACTER_REFS]):
        if index >= len(LAYOUT_CHARACTER_COLORS):
            break
        if isinstance(ref, dict):
            ref_payload = dict(ref)
            ref_payload.setdefault("border_rgb", list(LAYOUT_CHARACTER_COLORS[index]))
            ref_payload.setdefault("border_width", CHARACTER_REF_BORDER_WIDTH)
        else:
            ref_payload = {
                "path": ref,
                "border_rgb": list(LAYOUT_CHARACTER_COLORS[index]),
                "border_width": CHARACTER_REF_BORDER_WIDTH,
            }
        encoded_refs.append(ref_payload)
    controls["character_ref_paths"] = encoded_refs
    return encoded


@app.function(
    image=image,
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    secrets=[aws_secret],
)
def prepare_eval_manifest(
    *,
    eval_id: str,
    dataset: str = DEFAULT_VALIDATION_DATASET,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    caption_run: str = DEFAULT_CAPTION_RUN,
    include_chapter_regex: str = DEFAULT_INCLUDE_CHAPTER_REGEX,
    pages_prefix: str = DEFAULT_PAGES_PREFIX,
    captions_prefix: str = DEFAULT_CAPTIONS_PREFIX,
    annotations_prefix: str = DEFAULT_ANNOTATIONS_PREFIX,
    force: bool = False,
) -> dict[str, Any]:
    import boto3

    s3_client = boto3.client("s3")
    dataset = sanitize_dataset_name(dataset)
    manifest_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "sample_manifest.jsonl")
    summary_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "sample_manifest_summary.json")
    fixed_manifest = validation_dataset_root(dataset) / "manifest.jsonl"
    if fixed_manifest.is_file():
        rows = []
        with fixed_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if len(rows) < sample_count:
            raise RuntimeError(
                f"Fixed validation set has {len(rows)} rows but {sample_count} were requested: {fixed_manifest}"
            )
        selected = [current_control_encoding_row(row) for row in rows[:sample_count]]
        upload_jsonl(s3_client, selected, manifest_uri)
        summary = {
            "eval_id": eval_id,
            "dataset": dataset,
            "manifest_uri": manifest_uri,
            "source": "local_fixed_validation_set",
            "fixed_manifest": str(fixed_manifest),
            "sample_count": len(selected),
            "seed": seed,
            "strata": {},
        }
        for row in selected:
            stratum = row.get("eval_stratum", "unknown")
            summary["strata"][stratum] = int(summary["strata"].get(stratum, 0)) + 1
        upload_json(s3_client, summary, summary_uri)
        return {**summary, "summary_uri": summary_uri, "reused": False}
    datasets_root = Path(REMOTE_VALIDATION_DATASETS_ROOT)
    available = sorted(
        path.name for path in datasets_root.iterdir()
        if path.is_dir() and (path / "manifest.jsonl").is_file()
    ) if datasets_root.is_dir() else []
    if dataset:
        raise FileNotFoundError(
            f"Validation dataset {dataset!r} is missing {fixed_manifest}. "
            f"Available datasets: {available}"
        )
    if not force and object_exists(s3_client, manifest_uri):
        rows = load_jsonl_s3(s3_client, manifest_uri)
        return {
            "eval_id": eval_id,
            "dataset": dataset,
            "manifest_uri": manifest_uri,
            "summary_uri": summary_uri,
            "sample_count": len(rows),
            "reused": True,
        }

    caption_root = f"{captions_prefix.rstrip('/')}/{caption_run.strip('/')}"
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=caption_root.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            rel = key[len(caption_root.rstrip("/") + "/") :]
            parts = rel.split("/")
            if rel.startswith("_") or len(parts) != 2:
                continue
            if include_re and not include_re.search(parts[0]):
                continue
            keys.append(key)
    keys.sort()
    if not keys:
        raise RuntimeError(f"No caption JSONs found under s3://{S3_BUCKET}/{caption_root}/")

    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            rows.extend(
                iter_panel_rows_for_caption(
                    s3_client,
                    bucket=S3_BUCKET,
                    caption_key=key,
                    pages_prefix=pages_prefix,
                    annotations_prefix=annotations_prefix,
                    target_multiple=16,
                )
            )
        except Exception as exc:
            print(f"[manifest] skipped {key}: {exc}", flush=True)

    selected = select_eval_rows(rows, count=sample_count, seed=seed)
    upload_jsonl(s3_client, selected, manifest_uri)
    summary = {
        "eval_id": eval_id,
        "dataset": dataset,
        "manifest_uri": manifest_uri,
        "caption_run": caption_run,
        "include_chapter_regex": include_chapter_regex,
        "source_caption_count": len(keys),
        "eligible_row_count": len(rows),
        "sample_count": len(selected),
        "seed": seed,
        "strata": {},
    }
    for row in selected:
        stratum = row.get("eval_stratum", "unknown")
        summary["strata"][stratum] = int(summary["strata"].get(stratum, 0)) + 1
    upload_json(s3_client, summary, summary_uri)
    return {**summary, "summary_uri": summary_uri, "reused": False}


def resolve_checkpoint(s3_client, checkpoint_uri: str, cache_root: Path) -> Path:
    if not checkpoint_uri:
        raise ValueError("checkpoint_uri is required when base=False")
    raw = checkpoint_uri.strip()
    if raw.startswith("s3://"):
        bucket, key = parse_s3_uri(raw)
        if key.endswith(".safetensors"):
            return download_s3_uri(s3_client, raw, cache_path_for_s3(raw, cache_root))
        prefix = key.rstrip("/") + "/"
        newest: dict[str, Any] | None = None
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                obj_key = str(obj.get("Key") or "")
                if not obj_key.endswith(".safetensors"):
                    continue
                if newest is None or obj.get("LastModified") > newest.get("LastModified"):
                    newest = obj
        if newest is None:
            raise FileNotFoundError(f"No .safetensors checkpoint found under {raw}")
        return download_s3_uri(s3_client, f"s3://{bucket}/{newest['Key']}", cache_path_for_s3(f"s3://{bucket}/{newest['Key']}", cache_root))
    path = Path(raw)
    if path.is_dir():
        candidates = sorted(path.glob("*.safetensors"), key=lambda item: item.stat().st_mtime)
        if candidates:
            return candidates[-1]
    if path.is_file():
        return path
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_uri}")


@app.function(
    image=image,
    timeout=10 * 60,
    cpu=1,
    memory=2048,
    single_use_containers=True,
    secrets=[hf_secret],
)
def hf_access_preflight() -> dict[str, Any]:
    import requests

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"Modal secret {os.environ.get('DRAWTOON_HF_SECRET_NAME', DEFAULT_HF_SECRET_NAME)!r} "
            "did not inject HF_TOKEN."
        )
    filename = "flux-2-klein-base-9b.safetensors"
    url = f"https://huggingface.co/{BASE_MODEL_ID}/resolve/main/{filename}"
    response = requests.head(
        url,
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False,
        timeout=30,
    )
    if response.status_code in {401, 403}:
        raise RuntimeError(
            f"HF_TOKEN cannot access gated model file {BASE_MODEL_ID}/{filename}; "
            f"status={response.status_code}. Use the same gated-repo-enabled token as dashboard/backend."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Hugging Face access preflight failed for {BASE_MODEL_ID}/{filename}; "
            f"status={response.status_code}"
        )
    return {
        "ok": True,
        "model": BASE_MODEL_ID,
        "filename": filename,
        "status_code": response.status_code,
        "hf_secret_name": os.environ.get("DRAWTOON_HF_SECRET_NAME", DEFAULT_HF_SECRET_NAME),
    }


def is_lora_state_dict(state: dict[str, Any]) -> bool:
    return any(str(key).endswith(".lora_A.weight") for key in state)


def normalize_ai_toolkit_lora_module_key(key: str) -> str:
    normalized = str(key)
    for prefix in ("diffusion_model.", "transformer.", "model.", "module."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def merge_ai_toolkit_lora_into_transformer(transformer, state: dict[str, Any], *, scale: float = 1.0) -> dict[str, Any]:
    import torch

    params = dict(transformer.named_parameters())
    lora_a_suffix = ".lora_A.weight"
    loaded = 0
    skipped_missing_b = 0
    skipped_missing_target = 0
    skipped_shape = 0
    ranks: set[int] = set()
    examples: list[dict[str, Any]] = []

    with torch.no_grad():
        for lora_a_key in sorted(str(key) for key in state if str(key).endswith(lora_a_suffix)):
            lora_b_key = f"{lora_a_key[:-len(lora_a_suffix)]}.lora_B.weight"
            if lora_b_key not in state:
                skipped_missing_b += 1
                continue

            target_key = normalize_ai_toolkit_lora_module_key(lora_a_key[:-len(lora_a_suffix)]) + ".weight"
            target = params.get(target_key)
            if target is None:
                skipped_missing_target += 1
                continue

            lora_a = state[lora_a_key]
            lora_b = state[lora_b_key]
            ranks.add(int(lora_a.shape[0]))
            if lora_a.ndim != 2 or lora_b.ndim != 2:
                skipped_shape += 1
                continue

            if tuple(lora_b.shape) != (target.shape[0], lora_a.shape[0]) or lora_a.shape[1] != target.shape[1]:
                skipped_shape += 1
                continue

            delta = torch.matmul(
                lora_b.to(device=target.device, dtype=torch.float32),
                lora_a.to(device=target.device, dtype=torch.float32),
            )
            target.add_(delta.to(dtype=target.dtype) * float(scale))
            loaded += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "target_key": target_key,
                        "rank": int(lora_a.shape[0]),
                        "shape": list(target.shape),
                    }
                )
            del delta

    return {
        "checkpoint_format": "ai_toolkit_lora",
        "loaded_lora_modules": loaded,
        "lora_a_tensors": sum(1 for key in state if str(key).endswith(lora_a_suffix)),
        "ranks": sorted(ranks),
        "scale": float(scale),
        "skipped_missing_b": skipped_missing_b,
        "skipped_missing_target": skipped_missing_target,
        "skipped_shape": skipped_shape,
        "examples": examples,
    }


def load_flux_pipeline(
    *,
    base: bool,
    checkpoint_uri: str = "",
    overlay_lora_uri: str = "",
    lora_scale: float = 1.0,
):
    import sys

    import torch
    from safetensors.torch import load_file

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"Modal secret {os.environ.get('DRAWTOON_HF_SECRET_NAME', DEFAULT_HF_SECRET_NAME)!r} "
            "did not inject HF_TOKEN."
        )
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    sys.path.insert(0, "/root/ai-toolkit")
    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein9BModel
    from toolkit.config_modules import ModelConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model_config = ModelConfig(
        name_or_path=BASE_MODEL_ID,
        arch="flux2_klein_9b",
        quantize=False,
        quantize_te=False,
        low_vram=False,
        dtype="bf16",
        vae_dtype="bf16",
        te_dtype="bf16",
    )
    model = Flux2Klein9BModel(device="cuda", model_config=model_config, dtype="bf16")
    model.load_model()
    load_stats: dict[str, Any] = {"base_model": BASE_MODEL_ID}
    if not base:
        import boto3

        s3_client = boto3.client("s3")
        checkpoint_path = resolve_checkpoint(s3_client, checkpoint_uri, Path("/root/validation_cache/checkpoints"))
        state = load_file(str(checkpoint_path), device="cpu")
        if is_lora_state_dict(state):
            lora_stats = merge_ai_toolkit_lora_into_transformer(model.model, state, scale=lora_scale)
            if int(lora_stats["loaded_lora_modules"]) < 10:
                raise RuntimeError(
                    "LoRA checkpoint matched too few FLUX2 target modules: "
                    f"{lora_stats}. Use the raw ai-toolkit .safetensors checkpoint, not a PEFT adapter dir."
                )
            load_stats.update(
                {
                    "checkpoint_uri": checkpoint_uri,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_tensors": len(state),
                    **lora_stats,
                }
            )
        else:
            target_state = model.model.state_dict()
            filtered = {}
            skipped_shape = 0
            for key, value in state.items():
                normalized_key = key
                for prefix in ("transformer.", "model.", "module."):
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key[len(prefix) :]
                if not (normalized_key.startswith("double_blocks.") or normalized_key.startswith("single_blocks.")):
                    continue
                if normalized_key in target_state and tuple(value.shape) == tuple(target_state[normalized_key].shape):
                    filtered[normalized_key] = value.to(dtype=target_state[normalized_key].dtype)
                else:
                    skipped_shape += 1
            if len(filtered) < 100:
                raise RuntimeError(f"Checkpoint matched too few FLUX2 block tensors: {len(filtered)}")
            missing, unexpected = model.model.load_state_dict(filtered, strict=False)
            load_stats.update(
                {
                    "checkpoint_format": "full_finetune_blocks",
                    "checkpoint_uri": checkpoint_uri,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_tensors": len(state),
                    "loaded_block_tensors": len(filtered),
                    "skipped_shape_or_key": skipped_shape,
                    "missing_tensors": len(missing),
                    "unexpected_tensors": len(unexpected),
                }
            )
        if overlay_lora_uri.strip():
            overlay_lora_path = resolve_checkpoint(
                s3_client,
                overlay_lora_uri,
                Path("/root/validation_cache/checkpoints"),
            )
            overlay_state = load_file(str(overlay_lora_path), device="cpu")
            if not is_lora_state_dict(overlay_state):
                raise RuntimeError(
                    "overlay_lora_uri must point to a raw ai-toolkit LoRA .safetensors checkpoint; "
                    f"got {overlay_lora_uri}"
                )
            overlay_stats = merge_ai_toolkit_lora_into_transformer(
                model.model,
                overlay_state,
                scale=lora_scale,
            )
            if int(overlay_stats["loaded_lora_modules"]) < 10:
                raise RuntimeError(
                    "Overlay LoRA checkpoint matched too few FLUX2 target modules: "
                    f"{overlay_stats}. Use the raw ai-toolkit .safetensors checkpoint."
                )
            load_stats["overlay_lora"] = {
                "checkpoint_uri": overlay_lora_uri,
                "checkpoint_path": str(overlay_lora_path),
                "checkpoint_tensors": len(overlay_state),
                **overlay_stats,
            }
    pipe = model.pipeline
    pipe.to("cuda")
    return pipe, load_stats


@app.function(
    image=image,
    gpu="H100",
    timeout=10 * 60 * 60,
    cpu=8,
    memory=65536,
    max_containers=10,
    single_use_containers=True,
    volumes={"/root/validation_cache": validation_cache},
    secrets=[aws_secret, hf_secret],
)
def generate_shard(spec: dict[str, Any]) -> dict[str, Any]:
    import boto3
    import torch
    from PIL import Image

    s3_client = boto3.client("s3")
    base = bool(spec["base"])
    model_key = "base" if base else "finetuned"
    eval_id = str(spec["eval_id"])
    output_prefix = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{eval_id}/{model_key}"
    shard_index = int(spec["shard_index"])
    rows = list(spec["rows"])
    steps = int(spec.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(spec.get("guidance_scale") or DEFAULT_GUIDANCE)
    checkpoint_uri = str(spec.get("checkpoint_uri") or "")
    overlay_lora_uri = str(spec.get("overlay_lora_uri") or "")
    lora_scale = float(spec.get("lora_scale") or 1.0)
    asset_root = Path("/root/validation_cache/assets") / eval_id

    pipe, load_stats = load_flux_pipeline(
        base=base,
        checkpoint_uri=checkpoint_uri,
        overlay_lora_uri=overlay_lora_uri,
        lora_scale=lora_scale,
    )
    outputs: list[dict[str, Any]] = []
    started = time.time()
    for row in rows:
        sample_index = int(row["eval_index"])
        sample_id = str(row["sample_id"])
        sample_dir = f"{output_prefix}/samples/{sample_index:04d}_{sanitize_filename(sample_id)}"
        width = int(row["target_width"])
        height = int(row["target_height"])
        caption = str(row["caption"])
        seed = int(row["eval_seed"])
        controls = row.get("controls") or {}
        layout = controls.get("layout_control") or {}

        target_path = materialize_asset(s3_client, row["target_panel"], asset_root=asset_root)
        target_image = Image.open(target_path).convert("RGB")
        if controls.get("layout_control_path"):
            layout_path = materialize_asset(s3_client, controls["layout_control_path"], asset_root=asset_root)
            layout_image = Image.open(layout_path).convert("RGB")
        else:
            layout_image = materialize_layout_control(layout, width=width, height=height)
        control_images = [layout_image]
        ref_paths: list[Path] = []
        for ref in (controls.get("character_ref_paths") or [])[: MAX_CONTROL_IMAGE_SLOTS - 1]:
            ref_path = materialize_asset(s3_client, ref, asset_root=asset_root)
            ref_paths.append(ref_path)
            control_images.append(Image.open(ref_path).convert("RGB"))

        generator = torch.Generator(device="cuda").manual_seed(seed)
        t0 = time.time()
        generated = pipe(
            prompt=caption,
            negative_prompt="",
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            control_img_list=control_images,
        ).images[0]
        seconds = round(time.time() - t0, 3)

        local_out = Path("/tmp/validation_outputs") / eval_id / model_key / f"{sample_index:04d}"
        local_out.mkdir(parents=True, exist_ok=True)
        generated_path = local_out / "generated.png"
        target_copy_path = local_out / "target.png"
        layout_path = local_out / "ctrl_img_1.png"
        generated.save(generated_path)
        target_image.save(target_copy_path)
        layout_image.save(layout_path)

        upload_files = [
            (generated_path, f"{sample_dir}/generated.png"),
            (target_copy_path, f"{sample_dir}/target.png"),
            (layout_path, f"{sample_dir}/ctrl_img_1.png"),
        ]
        ref_keys = []
        for idx, ref_path in enumerate(ref_paths, start=2):
            ref_copy_path = local_out / f"ctrl_img_{idx}.png"
            Image.open(ref_path).convert("RGB").save(ref_copy_path)
            ref_key = f"{sample_dir}/ctrl_img_{idx}.png"
            upload_files.append((ref_copy_path, ref_key))
            ref_keys.append(s3_uri(ref_key))
        for src, key in upload_files:
            s3_client.upload_file(str(src), S3_BUCKET, key)

        record = {
            "eval_id": eval_id,
            "model": model_key,
            "base": base,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "caption": caption,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "overlay_lora_uri": overlay_lora_uri,
            "lora_scale": lora_scale,
            "seconds": seconds,
            "generated_uri": s3_uri(f"{sample_dir}/generated.png"),
            "target_uri": s3_uri(f"{sample_dir}/target.png"),
            "layout_uri": s3_uri(f"{sample_dir}/ctrl_img_1.png"),
            "ref_uris": ref_keys,
            "row": row,
            "load_stats": load_stats,
        }
        upload_json(s3_client, record, s3_uri(f"{sample_dir}/metadata.json"))
        outputs.append(record)
        print(f"[{model_key} shard {shard_index}] generated {sample_index:04d} {sample_id} in {seconds}s", flush=True)

    shard_summary = {
        "eval_id": eval_id,
        "model": model_key,
        "shard_index": shard_index,
        "count": len(outputs),
        "seconds": round(time.time() - started, 3),
        "outputs": outputs,
        "load_stats": load_stats,
    }
    upload_json(s3_client, shard_summary, s3_uri(output_prefix, "shards", f"shard_{shard_index:02d}.json"))
    validation_cache.commit()
    return {k: v for k, v in shard_summary.items() if k != "outputs"} | {"output_count": len(outputs)}


def pil_to_uint8_tensor(image):
    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def crop_norm(image, box_norm: list[float]):
    width, height = image.size
    x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
    y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
    x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def chunks(items: list[Any], size: int):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cmmd_distance(x, y, *, sigma: float = CMMD_SIGMA, scale: float = CMMD_SCALE) -> float:
    import torch

    x = torch.nn.functional.normalize(x.float(), dim=-1)
    y = torch.nn.functional.normalize(y.float(), dim=-1)
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    gamma = 1.0 / (2.0 * float(sigma) ** 2)
    k_xx = torch.mean(
        torch.exp(
            -gamma
            * (
                -2.0 * torch.matmul(x, x.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(x_sqnorms, 0)
            )
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2.0 * torch.matmul(x, y.T)
                + torch.unsqueeze(x_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2.0 * torch.matmul(y, y.T)
                + torch.unsqueeze(y_sqnorms, 1)
                + torch.unsqueeze(y_sqnorms, 0)
            )
        )
    )
    return float((scale * (k_xx + k_yy - 2.0 * k_xy)).detach().cpu().item())


def bootstrap_mean_delta(values: list[float], *, iterations: int = 1000, seed: int = DEFAULT_SEED) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return {"mean": sum(values) / len(values), "ci95_low": lo, "ci95_high": hi}


def split_bedrock_model_ref(model_ref: str) -> tuple[str, str]:
    raw = str(model_ref or "").strip()
    if "|" in raw:
        region, model = raw.split("|", 1)
        region = region.strip() or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid Bedrock model ref: {model_ref!r}")
        return region, model
    if not raw:
        raise ValueError("Bedrock model ref must not be empty")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return region, raw


def bedrock_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    input_tokens = int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def extract_bedrock_tool_input(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    content = ((response.get("output") or {}).get("message") or {}).get("content") or []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and str(tool_use.get("name") or "") == tool_name:
            value = tool_use.get("input")
            if not isinstance(value, dict):
                raise ValueError(f"Bedrock tool {tool_name!r} returned non-object input")
            return value
        if block.get("text"):
            texts.append(str(block["text"]))
    for text in texts:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Bedrock response did not contain tool input for {tool_name!r}")


def prepare_bedrock_image_block(image_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        scale = min(
            1.0,
            BEDROCK_MAX_IMAGE_SIDE / max(1, source_width),
            BEDROCK_MAX_IMAGE_SIDE / max(1, source_height),
        )
        if scale < 1.0:
            image = image.resize(
                (max(1, int(source_width * scale)), max(1, int(source_height * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in (92, 84, 76, 68, 60, 52):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= BEDROCK_MAX_IMAGE_BYTES:
                return (
                    {"image": {"format": "jpeg", "source": {"bytes": encoded}}},
                    {
                        "image_format": "jpeg",
                        "image_bytes": len(encoded),
                        "image_width": image.width,
                        "image_height": image.height,
                        "source_image_width": source_width,
                        "source_image_height": source_height,
                        "jpeg_quality": quality,
                    },
                )
    raise RuntimeError(f"Could not encode {image_path} below {BEDROCK_MAX_IMAGE_BYTES} bytes for Bedrock")


def expected_bubble_regions_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    row = record.get("row") if isinstance(record.get("row"), dict) else {}
    layout = ((row.get("controls") or {}).get("layout_control") or {}) if isinstance(row.get("controls"), dict) else {}
    text_payload = layout.get("text") if isinstance(layout.get("text"), dict) else {}
    regions: list[dict[str, Any]] = []
    for fallback_index, region in enumerate(text_payload.get("regions") or []):
        if not isinstance(region, dict):
            continue
        bubble_type = normalize_layout_text_bubble_type(region.get("type"))
        if bubble_type not in SCORED_TEXT_BUBBLE_TYPES:
            continue
        box_norm = region.get("bbox_norm")
        if not isinstance(box_norm, list) or len(box_norm) != 4:
            continue
        regions.append(
            {
                "text_region_index": int(region.get("text_region_index") if region.get("text_region_index") is not None else fallback_index),
                "expected_type": bubble_type,
                "bbox_norm": [round(float(value), 6) for value in box_norm],
            }
        )
    return regions


def type_count_template() -> dict[str, dict[str, Any]]:
    return {
        bubble_type: {
            "expected_count": 0,
            "respected_count": 0,
            "respected_percent": None,
            "notes": "",
        }
        for bubble_type in SCORED_TEXT_BUBBLE_TYPES
    }


def summarize_region_judgements(
    *,
    expected_regions: list[dict[str, Any]],
    judgement: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], float | None, list[dict[str, Any]]]:
    expected_by_index = {int(region["text_region_index"]): region for region in expected_regions}
    raw_judgements = judgement.get("region_judgements") if isinstance(judgement.get("region_judgements"), list) else []
    returned_by_index: dict[int, dict[str, Any]] = {}
    for item in raw_judgements:
        if not isinstance(item, dict):
            continue
        try:
            returned_by_index[int(item["text_region_index"])] = item
        except Exception:
            continue

    scores = type_count_template()
    normalized_judgements: list[dict[str, Any]] = []
    total_respected = 0
    for expected in expected_regions:
        region_index = int(expected["text_region_index"])
        expected_type = str(expected["expected_type"])
        item = returned_by_index.get(region_index) or {}
        generated_type = str(item.get("generated_type") or "Missing")
        if generated_type not in {"Speech Bubble", "Narration Bubble", "Shout Bubble", "None", "Missing", "Ambiguous"}:
            generated_type = "Ambiguous"
        respected = bool(item.get("respected")) and generated_type == expected_type
        confidence = item.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except Exception:
            confidence = 0.0
        evidence = " ".join(str(item.get("evidence") or "").split())[:240]
        scores[expected_type]["expected_count"] += 1
        if respected:
            scores[expected_type]["respected_count"] += 1
            total_respected += 1
        normalized_judgements.append(
            {
                "text_region_index": region_index,
                "bbox_norm": expected.get("bbox_norm"),
                "expected_type": expected_type,
                "generated_type": generated_type,
                "respected": respected,
                "confidence": confidence,
                "evidence": evidence,
            }
        )

    for bubble_type, score in scores.items():
        expected_count = int(score["expected_count"])
        score["respected_percent"] = (
            100.0 * int(score["respected_count"]) / expected_count if expected_count else None
        )
    overall = 100.0 * total_respected / len(expected_regions) if expected_regions else None
    return scores, overall, normalized_judgements


def call_haiku_bubble_judge(
    *,
    bedrock_client,
    model_id: str,
    record: dict[str, Any],
    expected_regions: list[dict[str, Any]],
    target_image_block: dict[str, Any],
    generated_image_block: dict[str, Any],
    max_output_tokens: int,
    request_id: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    expected_counts = {bubble_type: 0 for bubble_type in SCORED_TEXT_BUBBLE_TYPES}
    for region in expected_regions:
        expected_counts[str(region["expected_type"])] += 1
    user_text = "\n".join(
        [
            "Judge text-bubble type preservation between two manga panel images.",
            "Image 1 is the target panel. Image 2 is the generated panel.",
            "Use the target region list as the expected bubble types and approximate normalized panel boxes.",
            "Coordinates are [x0, y0, x1, y1] in normalized panel coordinates.",
            "",
            f"Sample id: {record.get('sample_id')}",
            f"Expected type counts: {json.dumps(expected_counts, sort_keys=True)}",
            "Expected target text regions:",
            json.dumps(expected_regions, sort_keys=True, ensure_ascii=False),
            "",
            "For each expected region, inspect the corresponding area and nearby area in the generated image.",
            "Return whether the generated image preserved the same bubble type.",
            "Do not judge OCR quality, exact wording, or whether lettering is readable.",
            "If a generated region exists but the container shape is wrong, respected=false.",
            "If there is no corresponding visible text container, generated_type=Missing and respected=false.",
        ]
    )
    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": BUBBLE_TYPE_JUDGE_SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": user_text},
                    target_image_block,
                    generated_image_block,
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": max(256, int(max_output_tokens)),
            "temperature": 0,
        },
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": "bubble_type_judgement",
                        "description": "Judge whether generated text containers preserve target speech/narration/shout bubble types.",
                        "inputSchema": {"json": BUBBLE_TYPE_JUDGE_SCHEMA},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "bubble_type_judgement"}},
        },
        requestMetadata={"client_request_id": request_id[:256]},
    )
    return extract_bedrock_tool_input(response, tool_name="bubble_type_judgement"), bedrock_usage(response)


def judge_bubble_types_for_record(
    *,
    s3_client,
    bedrock_client,
    model_id: str,
    record: dict[str, Any],
    local_root: Path,
    retries: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    expected_regions = expected_bubble_regions_from_record(record)
    base_row = {
        "eval_id": record.get("eval_id"),
        "model": record.get("model"),
        "sample_index": int(record.get("sample_index") or 0),
        "sample_id": str(record.get("sample_id") or ""),
        "expected_region_count": len(expected_regions),
        "expected_type_counts": {bubble_type: 0 for bubble_type in SCORED_TEXT_BUBBLE_TYPES},
        "available": False,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    for region in expected_regions:
        base_row["expected_type_counts"][str(region["expected_type"])] += 1
    if not expected_regions:
        return {
            **base_row,
            "status": "skipped_no_expected_bubbles",
            "overall_respected_percent": None,
            "type_scores": type_count_template(),
            "region_judgements": [],
        }

    sample_dir = local_root / f"{int(record.get('sample_index') or 0):04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    target_path = download_s3_uri(s3_client, str(record["target_uri"]), sample_dir / "target.png")
    generated_path = download_s3_uri(s3_client, str(record["generated_uri"]), sample_dir / "generated.png")
    target_block, target_meta = prepare_bedrock_image_block(target_path)
    generated_block, generated_meta = prepare_bedrock_image_block(generated_path)
    request_source = f"{record.get('eval_id')}:{record.get('model')}:{record.get('sample_id')}:bubble-type"
    request_id = hashlib.sha1(request_source.encode("utf-8")).hexdigest()

    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            judgement, usage = call_haiku_bubble_judge(
                bedrock_client=bedrock_client,
                model_id=model_id,
                record=record,
                expected_regions=expected_regions,
                target_image_block=target_block,
                generated_image_block=generated_block,
                max_output_tokens=max_output_tokens,
                request_id=f"{request_id}-{attempt}",
            )
            type_scores, overall, normalized_judgements = summarize_region_judgements(
                expected_regions=expected_regions,
                judgement=judgement,
            )
            return {
                **base_row,
                "available": True,
                "status": "ok",
                "overall_respected_percent": overall,
                "type_scores": type_scores,
                "region_judgements": normalized_judgements,
                "target_description": str(judgement.get("target_description") or "")[:1000],
                "generated_description": str(judgement.get("generated_description") or "")[:1000],
                "raw_overall_respected_percent": judgement.get("overall_respected_percent"),
                "raw_type_scores": judgement.get("type_scores"),
                "usage": usage,
                "bedrock_images": {"target": target_meta, "generated": generated_meta},
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(min(20.0, 1.5 * (2**attempt)) + random.random())
    return {
        **base_row,
        "available": False,
        "status": "error",
        "error": str(last_error),
        "overall_respected_percent": None,
        "type_scores": type_count_template(),
        "region_judgements": [],
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    cpu=8,
    memory=65536,
    single_use_containers=True,
    volumes={"/root/validation_cache": validation_cache},
    secrets=[aws_secret, hf_secret],
)
def evaluate_run(
    *,
    eval_id: str,
    base: bool,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    shard_count: int = DEFAULT_SHARD_COUNT,
    metric_batch_size: int = DEFAULT_METRIC_BATCH_SIZE,
) -> dict[str, Any]:
    import boto3
    import timm
    import torch
    from PIL import Image
    from transformers import (
        AutoModel,
        AutoProcessor,
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    s3_client = boto3.client("s3")
    model_key = "base" if base else "finetuned"
    output_prefix = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{eval_id}/{model_key}"
    metrics_prefix = f"{output_prefix}/metrics"
    local_root = Path("/tmp/validation_eval") / eval_id / model_key
    local_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for shard_idx in range(shard_count):
        shard_uri = s3_uri(output_prefix, "shards", f"shard_{shard_idx:02d}.json")
        if not object_exists(s3_client, shard_uri):
            continue
        shard = json.loads(s3_client.get_object(Bucket=S3_BUCKET, Key=parse_s3_uri(shard_uri)[1])["Body"].read().decode("utf-8"))
        records.extend(shard.get("outputs") or [])
    records.sort(key=lambda item: int(item["sample_index"]))
    if len(records) < sample_count:
        raise RuntimeError(f"Expected {sample_count} generated records for {model_key}, found {len(records)}")

    metric_batch_size = max(1, int(metric_batch_size or DEFAULT_METRIC_BATCH_SIZE))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_started = time.time()

    def metric_log(message: str) -> None:
        gpu = ""
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu = f" gpu_alloc={allocated:.2f}GiB gpu_reserved={reserved:.2f}GiB"
        print(f"[metrics {model_key}] +{time.time() - metric_started:.1f}s {message}{gpu}", flush=True)

    def log_progress(stage: str, index: int, total: int, detail: str = "") -> None:
        if total <= 0:
            return
        if index == 1 or index == total or index % 5 == 0:
            suffix = f" {detail}" if detail else ""
            metric_log(f"{stage} {index}/{total}{suffix}")

    metric_log(
        f"start internal-v2 eval_id={eval_id} samples={sample_count} records={len(records)} "
        f"metric_batch_size={metric_batch_size} device={device}"
    )
    metric_log(
        "loading metric models "
        f"siglip2={SIGLIP2_MODEL_ID} dinov3={DINOV3_MODEL_ID} cmmd={CMMD_MODEL_ID}"
    )
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    siglip_model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID).to(device).eval()
    dinov3_model = timm.create_model(DINOV3_TIMM_MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    dinov3_data_config = timm.data.resolve_model_data_config(dinov3_model)
    dinov3_transform = timm.data.create_transform(**dinov3_data_config, is_training=False)
    cmmd_processor = CLIPImageProcessor.from_pretrained(CMMD_MODEL_ID)
    cmmd_model = CLIPVisionModelWithProjection.from_pretrained(CMMD_MODEL_ID).to(device).eval()
    cmmd_input_size = int(cmmd_processor.crop_size.get("height") or cmmd_processor.size.get("shortest_edge") or 336)
    metric_log("loaded SigLIP2, DINOv3, and CMMD CLIP vision model")

    def siglip2_text_scores(images, texts: list[str]) -> list[float]:
        inputs = siglip_processor(
            text=[str(text).lower() for text in texts],
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(device)
        with torch.inference_mode():
            output = siglip_model(**inputs)
            image_features = torch.nn.functional.normalize(output.image_embeds, dim=-1)
            text_features = torch.nn.functional.normalize(output.text_embeds, dim=-1)
            return [float(value) for value in (image_features * text_features).sum(dim=-1).detach().cpu().tolist()]

    def dinov3_embedding_batch(images):
        inputs = torch.stack([dinov3_transform(image.convert("RGB")) for image in images]).to(device)
        with torch.inference_mode():
            emb = dinov3_model(inputs)
            if emb.ndim == 3:
                emb = emb[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def cmmd_embedding_batch(images):
        resized = [image.convert("RGB").resize((cmmd_input_size, cmmd_input_size), Image.Resampling.BICUBIC) for image in images]
        inputs = cmmd_processor(
            images=resized,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            image_features = cmmd_model(**inputs).image_embeds
            return torch.nn.functional.normalize(image_features, dim=-1)

    sample_items: list[dict[str, Any]] = []

    metric_log("downloading generated and target images")
    for record in records[:sample_count]:
        idx = int(record["sample_index"])
        sample_dir = local_root / f"{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        generated_path = download_s3_uri(s3_client, record["generated_uri"], sample_dir / "generated.png")
        target_path = download_s3_uri(s3_client, record["target_uri"], sample_dir / "target.png")
        generated = Image.open(generated_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        sample_items.append(
            {
                "record": record,
                "sample_index": idx,
                "sample_id": record["sample_id"],
                "sample_dir": sample_dir,
                "generated": generated,
                "target": target,
            }
        )
        downloaded = len(sample_items)
        if downloaded == 1 or downloaded == sample_count or downloaded % 25 == 0:
            metric_log(f"downloaded {downloaded}/{sample_count} image pairs")

    siglip2_t_by_index: dict[int, float] = {}
    dinov3_i_by_index: dict[int, float] = {}
    generated_cmmd_embeddings = []
    target_cmmd_embeddings = []
    total_batches = math.ceil(len(sample_items) / metric_batch_size)
    metric_log(f"starting SigLIP2-T, DINOv3-I, and CMMD embedding batches={total_batches}")
    batch_index = 0
    for batch in chunks(sample_items, metric_batch_size):
        batch_index += 1
        generated_images = [item["generated"] for item in batch]
        target_images = [item["target"] for item in batch]
        captions = [str(item["record"]["caption"]) for item in batch]
        for item, score in zip(batch, siglip2_text_scores(generated_images, captions), strict=True):
            siglip2_t_by_index[int(item["sample_index"])] = score
        gen_dinov3_emb = dinov3_embedding_batch(generated_images)
        target_dinov3_emb = dinov3_embedding_batch(target_images)
        dinov3_scores = (gen_dinov3_emb * target_dinov3_emb).sum(dim=-1).detach().cpu().tolist()
        for item, score in zip(batch, dinov3_scores, strict=True):
            dinov3_i_by_index[int(item["sample_index"])] = float(score)
        generated_cmmd_embeddings.append(cmmd_embedding_batch(generated_images).detach().cpu())
        target_cmmd_embeddings.append(cmmd_embedding_batch(target_images).detach().cpu())
        log_progress(
            "SigLIP2/DINOv3-I/CMMD batch",
            batch_index,
            total_batches,
            f"batch_size={len(batch)}",
        )
    metric_log("finished SigLIP2-T, DINOv3-I, and CMMD embeddings")

    ref_scores_by_index: dict[int, list[float]] = {int(item["sample_index"]): [] for item in sample_items}
    crop_pairs: list[dict[str, Any]] = []
    metric_log("preparing generated character crops and reference crops for DINOv3-C")
    for item in sample_items:
        record = item["record"]
        idx = int(item["sample_index"])
        generated = item["generated"]
        sample_dir = item["sample_dir"]

        row = record.get("row") or {}
        layout = ((row.get("controls") or {}).get("layout_control") or {})
        characters = layout.get("characters") or []
        ref_uris = record.get("ref_uris") or []
        for character in characters:
            if not isinstance(character, dict):
                continue
            slot = int(character.get("control_slot") or 0)
            ref_idx = slot - 2
            if ref_idx < 0 or ref_idx >= len(ref_uris):
                continue
            gen_crop = crop_norm(generated, character.get("bbox_norm") or [])
            if gen_crop is None:
                continue
            ref_path = download_s3_uri(s3_client, ref_uris[ref_idx], sample_dir / f"ref_{ref_idx + 1}.png")
            ref_image = Image.open(ref_path).convert("RGB")
            crop_pairs.append({"sample_index": idx, "generated_crop": gen_crop, "ref_image": ref_image})

    dinov3_c_total_batches = math.ceil(len(crop_pairs) / metric_batch_size) if crop_pairs else 0
    metric_log(f"starting DINOv3-C crop scoring crop_pairs={len(crop_pairs)} batches={dinov3_c_total_batches}")
    dinov3_c_batch_index = 0
    for batch in chunks(crop_pairs, metric_batch_size):
        dinov3_c_batch_index += 1
        gen_emb = dinov3_embedding_batch([item["generated_crop"] for item in batch])
        ref_emb = dinov3_embedding_batch([item["ref_image"] for item in batch])
        scores = (gen_emb * ref_emb).sum(dim=-1).detach().cpu().tolist()
        for item, score in zip(batch, scores, strict=True):
            ref_scores_by_index[int(item["sample_index"])].append(float(score))
        log_progress(
            "DINOv3-C batch",
            dinov3_c_batch_index,
            dinov3_c_total_batches,
            f"batch_size={len(batch)}",
        )
    metric_log("finished DINOv3-C crop scoring")

    per_sample: list[dict[str, Any]] = []
    for item in sample_items:
        record = item["record"]
        idx = int(item["sample_index"])
        ref_scores = ref_scores_by_index.get(idx, [])

        per_sample.append(
            {
                "eval_id": eval_id,
                "model": model_key,
                "sample_index": idx,
                "sample_id": record["sample_id"],
                "siglip2_t": siglip2_t_by_index[idx],
                "dinov3_i": dinov3_i_by_index[idx],
                "dinov3_c": mean(ref_scores),
                "character_crop_count": len(ref_scores),
            }
        )

    metric_log("computing aggregate CMMD and metric means")
    generated_cmmd = torch.cat(generated_cmmd_embeddings, dim=0).to(device)
    target_cmmd = torch.cat(target_cmmd_embeddings, dim=0).to(device)
    cmmd_value = cmmd_distance(target_cmmd, generated_cmmd)
    summary = {
        "eval_id": eval_id,
        "model": model_key,
        "metric_suite": "internal_v2_cmmd_siglip2_dinov3",
        "sample_count": len(per_sample),
        "metric_batch_size": metric_batch_size,
        "metric_models": {
            "cmmd": CMMD_MODEL_ID,
            "siglip2_t": SIGLIP2_MODEL_ID,
            "dinov3": DINOV3_MODEL_ID,
        },
        "cmmd": cmmd_value,
        "cmmd_sigma": CMMD_SIGMA,
        "cmmd_scale": CMMD_SCALE,
        "siglip2_t_mean": mean([row["siglip2_t"] for row in per_sample]),
        "dinov3_i_mean": mean([row["dinov3_i"] for row in per_sample]),
        "dinov3_c_mean": mean([row["dinov3_c"] for row in per_sample if row["dinov3_c"] is not None]),
        "metrics_uri": s3_uri(metrics_prefix, "summary.json"),
        "per_sample_uri": s3_uri(metrics_prefix, "per_sample.jsonl"),
    }
    metric_log(
        "aggregate metrics ready "
        f"cmmd={summary['cmmd']:.4f} siglip2_t={summary['siglip2_t_mean']:.4f} "
        f"dinov3_i={summary['dinov3_i_mean']:.4f} dinov3_c={summary['dinov3_c_mean']:.4f}"
    )
    metric_log(f"uploading per-sample metrics to {summary['per_sample_uri']}")
    upload_jsonl(s3_client, per_sample, summary["per_sample_uri"])
    metric_log(f"uploading summary metrics to {summary['metrics_uri']}")
    upload_json(s3_client, summary, summary["metrics_uri"])

    metric_log("checking for paired base/finetuned summary")
    paired = maybe_write_paired_summary(s3_client, eval_id=eval_id)
    if paired:
        summary["paired_summary_uri"] = paired["paired_summary_uri"]
        metric_log(f"paired summary written to {paired['paired_summary_uri']}")
    else:
        metric_log("paired summary not written yet; sibling run is not complete")
    validation_cache.commit()
    metric_log("committed validation cache and finished numeric evaluation")
    return summary


@app.function(
    image=image,
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16384,
    single_use_containers=True,
    volumes={"/root/validation_cache": validation_cache},
    secrets=[aws_secret],
)
def evaluate_bubble_types_with_haiku(
    *,
    eval_id: str,
    base: bool,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    shard_count: int = DEFAULT_SHARD_COUNT,
    haiku_sample_count: int = 0,
    haiku_shard_count: int = DEFAULT_HAIKU_SHARD_COUNT,
    model: str = DEFAULT_HAIKU_BUBBLE_MODEL,
    retries: int = 3,
    max_output_tokens: int = 1400,
) -> dict[str, Any]:
    import boto3

    s3_client = boto3.client("s3")
    region, model_id = split_bedrock_model_ref(model)
    model_key = "base" if base else "finetuned"
    output_prefix = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{eval_id}/{model_key}"
    metrics_prefix = f"{output_prefix}/metrics/haiku_bubble"

    records: list[dict[str, Any]] = []
    for shard_idx in range(shard_count):
        shard_uri = s3_uri(output_prefix, "shards", f"shard_{shard_idx:02d}.json")
        if not object_exists(s3_client, shard_uri):
            continue
        shard = json.loads(s3_client.get_object(Bucket=S3_BUCKET, Key=parse_s3_uri(shard_uri)[1])["Body"].read().decode("utf-8"))
        records.extend(shard.get("outputs") or [])
    records.sort(key=lambda item: int(item["sample_index"]))
    if len(records) < sample_count:
        raise RuntimeError(f"Expected {sample_count} generated records for {model_key}, found {len(records)}")

    judged_limit = min(sample_count, int(haiku_sample_count or sample_count))
    judged_records = records[:judged_limit]
    effective_haiku_shards = min(
        max(1, int(haiku_shard_count or shard_count or DEFAULT_HAIKU_SHARD_COUNT)),
        max(1, len(judged_records)),
    )
    per_haiku_shard = math.ceil(max(1, len(judged_records)) / effective_haiku_shards)
    specs = []
    for haiku_shard_idx in range(effective_haiku_shards):
        chunk = judged_records[
            haiku_shard_idx * per_haiku_shard : (haiku_shard_idx + 1) * per_haiku_shard
        ]
        if not chunk:
            continue
        specs.append(
            {
                "eval_id": eval_id,
                "base": base,
                "model": model,
                "model_id": model_id,
                "region": region,
                "model_key": model_key,
                "shard_index": haiku_shard_idx,
                "records": chunk,
                "retries": retries,
                "max_output_tokens": max_output_tokens,
            }
        )
    print(
        f"[haiku_bubble {model_key}] launching {len(specs)} judge shard(s) "
        f"for {len(judged_records)} sample(s)",
        flush=True,
    )
    shard_results = list(evaluate_bubble_types_with_haiku_shard.map(specs, order_outputs=False))

    per_sample: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    aggregate_scores = type_count_template()

    for shard_result in shard_results:
        for key, value in (shard_result.get("usage") or {}).items():
            usage_total[key] = int(usage_total.get(key, 0)) + int(value or 0)
        for row in shard_result.get("per_sample") or []:
            per_sample.append(row)
    per_sample.sort(key=lambda row: int(row.get("sample_index") or 0))

    for row in per_sample:
        for key, value in (row.get("usage") or {}).items():
            # Shard-level usage is authoritative; this keeps summaries correct if
            # a shard result is replayed from S3 in a future recovery path.
            usage_total.setdefault(key, int(usage_total.get(key, 0)))
        if row.get("available"):
            for bubble_type in SCORED_TEXT_BUBBLE_TYPES:
                score = (row.get("type_scores") or {}).get(bubble_type) or {}
                aggregate_scores[bubble_type]["expected_count"] += int(score.get("expected_count") or 0)
                aggregate_scores[bubble_type]["respected_count"] += int(score.get("respected_count") or 0)

    for bubble_type, score in aggregate_scores.items():
        expected_count = int(score["expected_count"])
        score["respected_percent"] = (
            100.0 * int(score["respected_count"]) / expected_count if expected_count else None
        )
    total_expected = sum(int(score["expected_count"]) for score in aggregate_scores.values())
    total_respected = sum(int(score["respected_count"]) for score in aggregate_scores.values())
    available_rows = [row for row in per_sample if row.get("available")]
    error_rows = [row for row in per_sample if row.get("status") == "error"]
    skipped_rows = [row for row in per_sample if row.get("status") == "skipped_no_expected_bubbles"]
    summary = {
        "eval_id": eval_id,
        "model": model_key,
        "judge_model": model_id,
        "judge_region": region,
        "sample_count_requested": sample_count,
        "sample_count_judged": len(per_sample),
        "judge_shard_count": len(specs),
        "available_sample_count": len(available_rows),
        "skipped_no_expected_bubbles": len(skipped_rows),
        "error_count": len(error_rows),
        "expected_region_count": total_expected,
        "respected_region_count": total_respected,
        "overall_respected_percent": 100.0 * total_respected / total_expected if total_expected else None,
        "type_scores": aggregate_scores,
        "usage": usage_total,
        "bubble_type_definitions": {
            "Speech Bubble": "smooth round or oval spoken-dialogue balloon, often with a speaker tail or pointer",
            "Narration Bubble": "square or rectangular caption/narration box with straight sides or corners and usually no speaker tail",
            "Shout Bubble": "jagged, spiky, starburst, or angular burst balloon used for shouting or emphasis",
        },
        "per_sample_uri": s3_uri(metrics_prefix, "per_sample.jsonl"),
        "summary_uri": s3_uri(metrics_prefix, "summary.json"),
    }
    upload_jsonl(s3_client, per_sample, summary["per_sample_uri"])
    upload_json(s3_client, summary, summary["summary_uri"])
    paired = maybe_write_paired_haiku_bubble_summary(s3_client, eval_id=eval_id)
    if paired:
        summary["paired_haiku_bubble_summary_uri"] = paired["summary_uri"]
    validation_cache.commit()
    return summary


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    cpu=2,
    memory=8192,
    max_containers=200,
    single_use_containers=True,
    volumes={"/root/validation_cache": validation_cache},
    secrets=[aws_secret],
)
def evaluate_bubble_types_with_haiku_shard(spec: dict[str, Any]) -> dict[str, Any]:
    import boto3
    from botocore.config import Config

    s3_client = boto3.client("s3")
    region = str(spec["region"])
    model_id = str(spec["model_id"])
    model_key = str(spec["model_key"])
    eval_id = str(spec["eval_id"])
    shard_index = int(spec["shard_index"])
    records = list(spec["records"])
    retries = int(spec.get("retries") or 3)
    max_output_tokens = int(spec.get("max_output_tokens") or 1400)
    metrics_prefix = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{eval_id}/{model_key}/metrics/haiku_bubble"
    local_root = Path("/tmp/validation_haiku_bubble") / eval_id / model_key / f"shard_{shard_index:02d}"
    local_root.mkdir(parents=True, exist_ok=True)

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            region_name=region,
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=10,
            read_timeout=180,
        ),
    )
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    per_sample: list[dict[str, Any]] = []
    started = time.time()
    for record in records:
        row = judge_bubble_types_for_record(
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            model_id=model_id,
            record=record,
            local_root=local_root,
            retries=retries,
            max_output_tokens=max_output_tokens,
        )
        for key, value in (row.get("usage") or {}).items():
            usage_total[key] = int(usage_total.get(key, 0)) + int(value or 0)
        per_sample.append(row)
        print(
            f"[haiku_bubble {model_key} shard {shard_index}] "
            f"{row['sample_index']:04d} {row['sample_id']} "
            f"{row['status']} overall={row.get('overall_respected_percent')}",
            flush=True,
        )

    per_sample.sort(key=lambda row: int(row.get("sample_index") or 0))
    shard_summary = {
        "eval_id": eval_id,
        "model": model_key,
        "shard_index": shard_index,
        "sample_count": len(per_sample),
        "seconds": round(time.time() - started, 3),
        "usage": usage_total,
        "per_sample": per_sample,
        "per_sample_uri": s3_uri(metrics_prefix, "shards", f"shard_{shard_index:02d}.jsonl"),
        "summary_uri": s3_uri(metrics_prefix, "shards", f"shard_{shard_index:02d}.json"),
    }
    upload_jsonl(s3_client, per_sample, shard_summary["per_sample_uri"])
    upload_json(
        s3_client,
        {key: value for key, value in shard_summary.items() if key != "per_sample"},
        shard_summary["summary_uri"],
    )
    validation_cache.commit()
    return shard_summary


def maybe_write_paired_summary(s3_client, *, eval_id: str) -> dict[str, Any] | None:
    base_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "base", "metrics", "per_sample.jsonl")
    finetuned_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "finetuned", "metrics", "per_sample.jsonl")
    base_summary_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "base", "metrics", "summary.json")
    finetuned_summary_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "finetuned", "metrics", "summary.json")
    if not object_exists(s3_client, base_uri) or not object_exists(s3_client, finetuned_uri):
        return None
    base_rows = {str(row["sample_id"]): row for row in load_jsonl_s3(s3_client, base_uri)}
    fine_rows = {str(row["sample_id"]): row for row in load_jsonl_s3(s3_client, finetuned_uri)}
    base_summary = json.loads(s3_client.get_object(Bucket=S3_BUCKET, Key=parse_s3_uri(base_summary_uri)[1])["Body"].read().decode("utf-8"))
    fine_summary = json.loads(s3_client.get_object(Bucket=S3_BUCKET, Key=parse_s3_uri(finetuned_summary_uri)[1])["Body"].read().decode("utf-8"))
    metric_suite = "internal_v2_cmmd_siglip2_dinov3"
    if base_summary.get("metric_suite") != metric_suite or fine_summary.get("metric_suite") != metric_suite:
        return None

    def delta_value(fine_value: Any, base_value: Any) -> float | None:
        if fine_value is None or base_value is None:
            return None
        return float(fine_value) - float(base_value)

    common_ids = sorted(set(base_rows) & set(fine_rows))
    deltas: list[dict[str, Any]] = []
    for sample_id in common_ids:
        base_row = base_rows[sample_id]
        fine_row = fine_rows[sample_id]
        delta = {
            "sample_id": sample_id,
            "sample_index": fine_row.get("sample_index"),
            "siglip2_t_delta": delta_value(fine_row.get("siglip2_t"), base_row.get("siglip2_t")),
            "dinov3_i_delta": delta_value(fine_row.get("dinov3_i"), base_row.get("dinov3_i")),
            "dinov3_c_delta": delta_value(fine_row.get("dinov3_c"), base_row.get("dinov3_c")),
        }
        deltas.append(delta)
    summary = {
        "eval_id": eval_id,
        "paired_count": len(deltas),
        "metric_suite": metric_suite,
        "cmmd_delta": fine_summary.get("cmmd") - base_summary.get("cmmd"),
        "siglip2_t_delta": bootstrap_mean_delta([row["siglip2_t_delta"] for row in deltas if row["siglip2_t_delta"] is not None]),
        "dinov3_i_delta": bootstrap_mean_delta([row["dinov3_i_delta"] for row in deltas if row["dinov3_i_delta"] is not None]),
        "dinov3_c_delta": bootstrap_mean_delta([row["dinov3_c_delta"] for row in deltas if row["dinov3_c_delta"] is not None]),
        "paired_delta_uri": s3_uri(S3_OUTPUT_PREFIX, eval_id, "paired", "paired_delta.jsonl"),
        "paired_summary_uri": s3_uri(S3_OUTPUT_PREFIX, eval_id, "paired", "paired_summary.json"),
    }
    upload_jsonl(s3_client, deltas, summary["paired_delta_uri"])
    upload_json(s3_client, summary, summary["paired_summary_uri"])
    return summary


def maybe_write_paired_haiku_bubble_summary(s3_client, *, eval_id: str) -> dict[str, Any] | None:
    base_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "base", "metrics", "haiku_bubble", "per_sample.jsonl")
    finetuned_uri = s3_uri(S3_OUTPUT_PREFIX, eval_id, "finetuned", "metrics", "haiku_bubble", "per_sample.jsonl")
    if not object_exists(s3_client, base_uri) or not object_exists(s3_client, finetuned_uri):
        return None
    base_rows = {str(row["sample_id"]): row for row in load_jsonl_s3(s3_client, base_uri) if row.get("available")}
    fine_rows = {str(row["sample_id"]): row for row in load_jsonl_s3(s3_client, finetuned_uri) if row.get("available")}
    common_ids = sorted(set(base_rows) & set(fine_rows))
    deltas: list[dict[str, Any]] = []
    for sample_id in common_ids:
        base_row = base_rows[sample_id]
        fine_row = fine_rows[sample_id]
        base_overall = base_row.get("overall_respected_percent")
        fine_overall = fine_row.get("overall_respected_percent")
        type_deltas = {}
        for bubble_type in SCORED_TEXT_BUBBLE_TYPES:
            base_score = ((base_row.get("type_scores") or {}).get(bubble_type) or {}).get("respected_percent")
            fine_score = ((fine_row.get("type_scores") or {}).get(bubble_type) or {}).get("respected_percent")
            type_deltas[bubble_type] = (
                fine_score - base_score if fine_score is not None and base_score is not None else None
            )
        deltas.append(
            {
                "sample_id": sample_id,
                "sample_index": fine_row.get("sample_index"),
                "overall_respected_percent_delta": (
                    fine_overall - base_overall if fine_overall is not None and base_overall is not None else None
                ),
                "type_respected_percent_delta": type_deltas,
            }
        )
    summary = {
        "eval_id": eval_id,
        "paired_count": len(deltas),
        "overall_respected_percent_delta": bootstrap_mean_delta(
            [
                row["overall_respected_percent_delta"]
                for row in deltas
                if row.get("overall_respected_percent_delta") is not None
            ]
        ),
        "type_respected_percent_delta": {
            bubble_type: bootstrap_mean_delta(
                [
                    row["type_respected_percent_delta"][bubble_type]
                    for row in deltas
                    if row["type_respected_percent_delta"].get(bubble_type) is not None
                ]
            )
            for bubble_type in SCORED_TEXT_BUBBLE_TYPES
        },
        "delta_uri": s3_uri(S3_OUTPUT_PREFIX, eval_id, "paired", "haiku_bubble_delta.jsonl"),
        "summary_uri": s3_uri(S3_OUTPUT_PREFIX, eval_id, "paired", "haiku_bubble_summary.json"),
    }
    upload_jsonl(s3_client, deltas, summary["delta_uri"])
    upload_json(s3_client, summary, summary["summary_uri"])
    return summary


@app.function(
    image=image,
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16384,
    single_use_containers=True,
    secrets=[aws_secret],
)
def run_validation(
    *,
    base: bool,
    checkpoint_uri: str = "",
    overlay_lora_uri: str = "",
    lora_scale: float = 1.0,
    eval_id: str = "",
    dataset: str = DEFAULT_VALIDATION_DATASET,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    shard_count: int = DEFAULT_SHARD_COUNT,
    metric_batch_size: int = DEFAULT_METRIC_BATCH_SIZE,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int = DEFAULT_SEED,
    force_manifest: bool = False,
    haiku_bubble_judge: bool = False,
    haiku_bubble_sample_count: int = 0,
    haiku_bubble_shard_count: int = DEFAULT_HAIKU_SHARD_COUNT,
    haiku_bubble_model: str = DEFAULT_HAIKU_BUBBLE_MODEL,
) -> dict[str, Any]:
    eval_id = eval_id.strip() or time.strftime("klein_eval_%Y%m%d_%H%M%S", time.gmtime())
    dataset = sanitize_dataset_name(dataset)
    if not base and not checkpoint_uri.strip():
        raise ValueError("checkpoint_uri is required when base=False")
    print(
        f"[run_validation] start eval_id={eval_id} model={'base' if base else 'finetuned'} "
        f"dataset={dataset} samples={sample_count} shards={shard_count} steps={steps} "
        f"haiku_bubble={haiku_bubble_judge} overlay_lora={bool(overlay_lora_uri.strip())}",
        flush=True,
    )
    manifest = prepare_eval_manifest.remote(
        eval_id=eval_id,
        dataset=dataset,
        sample_count=sample_count,
        seed=seed,
        force=force_manifest,
    )
    print(f"[run_validation] manifest ready: {manifest['manifest_uri']}", flush=True)
    print("[run_validation] checking Hugging Face gated model access before GPU launch", flush=True)
    hf_preflight = hf_access_preflight.remote()
    print(
        f"[run_validation] Hugging Face access ok via {hf_preflight['hf_secret_name']}",
        flush=True,
    )
    import boto3

    rows = load_jsonl_s3(boto3.client("s3"), manifest["manifest_uri"])
    rows = rows[:sample_count]
    per_shard = math.ceil(sample_count / shard_count)
    specs = []
    for shard_index in range(shard_count):
        chunk = rows[shard_index * per_shard : (shard_index + 1) * per_shard]
        if not chunk:
            continue
        specs.append(
            {
                "eval_id": eval_id,
                "base": base,
                "checkpoint_uri": checkpoint_uri,
                "overlay_lora_uri": overlay_lora_uri,
                "lora_scale": lora_scale,
                "shard_index": shard_index,
                "rows": chunk,
                "steps": steps,
                "guidance_scale": guidance_scale,
            }
        )
    print(f"[run_validation] launching {len(specs)} generation shard(s)", flush=True)
    shard_results = list(generate_shard.map(specs, order_outputs=False))
    print("[run_validation] generation complete; starting numeric evaluation", flush=True)
    eval_summary = evaluate_run.remote(
        eval_id=eval_id,
        base=base,
        sample_count=sample_count,
        shard_count=shard_count,
        metric_batch_size=metric_batch_size,
    )
    print("[run_validation] numeric evaluation complete", flush=True)
    haiku_bubble_summary = None
    if haiku_bubble_judge:
        print("[run_validation] starting Haiku bubble-type judge", flush=True)
        haiku_bubble_summary = evaluate_bubble_types_with_haiku.remote(
            eval_id=eval_id,
            base=base,
            sample_count=sample_count,
            shard_count=shard_count,
            haiku_sample_count=haiku_bubble_sample_count,
            haiku_shard_count=haiku_bubble_shard_count,
            model=haiku_bubble_model,
        )
        print("[run_validation] Haiku bubble-type judge complete", flush=True)
    result = {
        "eval_id": eval_id,
        "dataset": dataset,
        "model": "base" if base else "finetuned",
        "base": base,
        "checkpoint_uri": checkpoint_uri if not base else "",
        "overlay_lora_uri": overlay_lora_uri if not base else "",
        "lora_scale": lora_scale,
        "manifest_uri": manifest["manifest_uri"],
        "output_prefix": s3_uri(S3_OUTPUT_PREFIX, eval_id, "base" if base else "finetuned"),
        "shards": shard_results,
        "evaluation": eval_summary,
        "haiku_bubble_evaluation": haiku_bubble_summary,
    }
    import boto3 as _boto3

    upload_json(
        _boto3.client("s3"),
        result,
        s3_uri(S3_OUTPUT_PREFIX, eval_id, "base" if base else "finetuned", "run_summary.json"),
    )
    return result


@app.function(
    image=image,
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16384,
    single_use_containers=True,
    secrets=[aws_secret],
)
def run_existing_evaluation(
    *,
    base: bool,
    eval_id: str,
    dataset: str = DEFAULT_VALIDATION_DATASET,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    shard_count: int = DEFAULT_SHARD_COUNT,
    metric_batch_size: int = DEFAULT_METRIC_BATCH_SIZE,
    haiku_bubble_judge: bool = False,
    haiku_bubble_sample_count: int = 0,
    haiku_bubble_shard_count: int = DEFAULT_HAIKU_SHARD_COUNT,
    haiku_bubble_model: str = DEFAULT_HAIKU_BUBBLE_MODEL,
) -> dict[str, Any]:
    model_key = "base" if base else "finetuned"
    dataset = sanitize_dataset_name(dataset)
    print(
        f"[run_existing_evaluation] start eval_id={eval_id} model={model_key} "
        f"dataset={dataset} samples={sample_count} shards={shard_count} haiku_bubble={haiku_bubble_judge}",
        flush=True,
    )
    eval_summary = evaluate_run.remote(
        eval_id=eval_id,
        base=base,
        sample_count=sample_count,
        shard_count=shard_count,
        metric_batch_size=metric_batch_size,
    )
    print("[run_existing_evaluation] internal metric evaluation complete", flush=True)
    haiku_bubble_summary = None
    if haiku_bubble_judge:
        print("[run_existing_evaluation] starting Haiku bubble-type judge", flush=True)
        haiku_bubble_summary = evaluate_bubble_types_with_haiku.remote(
            eval_id=eval_id,
            base=base,
            sample_count=sample_count,
            shard_count=shard_count,
            haiku_sample_count=haiku_bubble_sample_count,
            haiku_shard_count=haiku_bubble_shard_count,
            model=haiku_bubble_model,
        )
        print("[run_existing_evaluation] Haiku bubble-type judge complete", flush=True)
    result = {
        "eval_id": eval_id,
        "dataset": dataset,
        "model": model_key,
        "base": base,
        "mode": "evaluate_existing",
        "output_prefix": s3_uri(S3_OUTPUT_PREFIX, eval_id, model_key),
        "evaluation": eval_summary,
        "haiku_bubble_evaluation": haiku_bubble_summary,
    }
    import boto3 as _boto3

    upload_json(
        _boto3.client("s3"),
        result,
        s3_uri(S3_OUTPUT_PREFIX, eval_id, model_key, "existing_eval_summary.json"),
    )
    return result


@app.function(image=image, timeout=60 * 60, cpu=1, memory=1024, single_use_containers=True)
@modal.fastapi_endpoint(method="POST")
def start(item: dict):
    base = parse_bool(item.get("base"), default=True)
    checkpoint_uri = str(item.get("checkpoint_uri") or "")
    overlay_lora_uri = str(item.get("overlay_lora_uri") or "")
    lora_scale = float(item.get("lora_scale") or 1.0)
    eval_id = str(item.get("eval_id") or "").strip() or time.strftime("klein_eval_%Y%m%d_%H%M%S", time.gmtime())
    dataset = sanitize_dataset_name(str(item.get("dataset") or DEFAULT_VALIDATION_DATASET))
    sample_count = int(item.get("sample_count") or DEFAULT_SAMPLE_COUNT)
    shard_count = int(item.get("shard_count") or DEFAULT_SHARD_COUNT)
    metric_batch_size = int(item.get("metric_batch_size") or DEFAULT_METRIC_BATCH_SIZE)
    steps = int(item.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(item.get("guidance_scale") or DEFAULT_GUIDANCE)
    seed = int(item.get("seed") or DEFAULT_SEED)
    force_manifest = parse_bool(item.get("force_manifest"), default=False)
    haiku_bubble_judge = parse_bool(item.get("haiku_bubble_judge"), default=False)
    haiku_bubble_sample_count = int(item.get("haiku_bubble_sample_count") or 0)
    haiku_bubble_shard_count = int(item.get("haiku_bubble_shard_count") or DEFAULT_HAIKU_SHARD_COUNT)
    haiku_bubble_model = str(item.get("haiku_bubble_model") or DEFAULT_HAIKU_BUBBLE_MODEL)
    evaluate_existing = parse_bool(item.get("evaluate_existing"), default=False) or parse_bool(
        item.get("numeric_eval_only"),
        default=False,
    )
    if evaluate_existing:
        call = run_existing_evaluation.spawn(
            base=base,
            eval_id=eval_id,
            dataset=dataset,
            sample_count=sample_count,
            shard_count=shard_count,
            metric_batch_size=metric_batch_size,
            haiku_bubble_judge=haiku_bubble_judge,
            haiku_bubble_sample_count=haiku_bubble_sample_count,
            haiku_bubble_shard_count=haiku_bubble_shard_count,
            haiku_bubble_model=haiku_bubble_model,
        )
        mode = "evaluate_existing"
    else:
        call = run_validation.spawn(
            base=base,
            checkpoint_uri=checkpoint_uri,
            overlay_lora_uri=overlay_lora_uri,
            lora_scale=lora_scale,
            eval_id=eval_id,
            dataset=dataset,
            sample_count=sample_count,
            shard_count=shard_count,
            metric_batch_size=metric_batch_size,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            force_manifest=force_manifest,
            haiku_bubble_judge=haiku_bubble_judge,
            haiku_bubble_sample_count=haiku_bubble_sample_count,
            haiku_bubble_shard_count=haiku_bubble_shard_count,
            haiku_bubble_model=haiku_bubble_model,
        )
        mode = "regenerate_images"
    return {
        "call_id": call.object_id,
        "eval_id": eval_id,
        "dataset": dataset,
        "model": "base" if base else "finetuned",
        "mode": mode,
        "output_root": s3_uri(S3_OUTPUT_PREFIX, eval_id),
        "checkpoint_uri": checkpoint_uri if not base else "",
        "overlay_lora_uri": overlay_lora_uri if not base else "",
        "lora_scale": lora_scale,
        "haiku_bubble_judge": haiku_bubble_judge,
    }


@app.local_entrypoint()
def main(
    base: bool = True,
    checkpoint_uri: str = "",
    overlay_lora_uri: str = "",
    lora_scale: float = 1.0,
    eval_id: str = "",
    dataset: str = DEFAULT_VALIDATION_DATASET,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    shard_count: int = DEFAULT_SHARD_COUNT,
    metric_batch_size: int = DEFAULT_METRIC_BATCH_SIZE,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int = DEFAULT_SEED,
    force_manifest: bool = False,
    haiku_bubble_judge: bool = False,
    haiku_bubble_sample_count: int = 0,
    haiku_bubble_shard_count: int = DEFAULT_HAIKU_SHARD_COUNT,
    haiku_bubble_model: str = DEFAULT_HAIKU_BUBBLE_MODEL,
    haiku_bubble_only: bool = False,
    numeric_eval_only: bool = False,
    evaluate_existing: bool = False,
    regenerate_images: bool = False,
    hf_preflight_only: bool = False,
    wait: bool = False,
):
    eval_id = eval_id.strip() or time.strftime("klein_eval_%Y%m%d_%H%M%S", time.gmtime())
    dataset = sanitize_dataset_name(dataset)
    if hf_preflight_only:
        print("Starting Hugging Face gated-model access preflight", flush=True)
        call = hf_access_preflight.spawn()
        print(f"Modal HF access preflight call id: {call.object_id}")
    elif haiku_bubble_only:
        print(
            f"Starting Haiku bubble judge only: eval_id={eval_id} model={'base' if base else 'finetuned'}",
            flush=True,
        )
        call = evaluate_bubble_types_with_haiku.spawn(
            eval_id=eval_id,
            base=base,
            sample_count=sample_count,
            shard_count=shard_count,
            haiku_sample_count=haiku_bubble_sample_count,
            haiku_shard_count=haiku_bubble_shard_count,
            model=haiku_bubble_model,
        )
        print(f"Modal Haiku bubble judge call id: {call.object_id}")
    elif numeric_eval_only or evaluate_existing:
        print(
            f"Starting evaluation on existing generated images: eval_id={eval_id} "
            f"model={'base' if base else 'finetuned'} dataset={dataset}",
            flush=True,
        )
        call = run_existing_evaluation.spawn(
            base=base,
            eval_id=eval_id,
            dataset=dataset,
            sample_count=sample_count,
            shard_count=shard_count,
            metric_batch_size=metric_batch_size,
            haiku_bubble_judge=haiku_bubble_judge,
            haiku_bubble_sample_count=haiku_bubble_sample_count,
            haiku_bubble_shard_count=haiku_bubble_shard_count,
            haiku_bubble_model=haiku_bubble_model,
        )
        print(f"Modal existing-evaluation call id: {call.object_id}")
    else:
        print(
            f"Starting validation: eval_id={eval_id} model={'base' if base else 'finetuned'} "
            f"dataset={dataset} samples={sample_count} shards={shard_count} steps={steps} "
            "mode=regenerate_images "
            f"haiku_bubble={haiku_bubble_judge} overlay_lora={bool(overlay_lora_uri.strip())}",
            flush=True,
        )
        call = run_validation.spawn(
            base=base,
            checkpoint_uri=checkpoint_uri,
            overlay_lora_uri=overlay_lora_uri,
            lora_scale=lora_scale,
            eval_id=eval_id,
            dataset=dataset,
            sample_count=sample_count,
            shard_count=shard_count,
            metric_batch_size=metric_batch_size,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            force_manifest=force_manifest,
            haiku_bubble_judge=haiku_bubble_judge,
            haiku_bubble_sample_count=haiku_bubble_sample_count,
            haiku_bubble_shard_count=haiku_bubble_shard_count,
            haiku_bubble_model=haiku_bubble_model,
        )
        print(f"Modal validation call id: {call.object_id}")
    print(f"Model: {'base' if base else 'finetuned'}")
    print(f"Dataset: {dataset}")
    print(f"Output root: {s3_uri(S3_OUTPUT_PREFIX, eval_id)}")
    if haiku_bubble_judge or haiku_bubble_only:
        print(f"Haiku bubble judge: enabled model={haiku_bubble_model}")
    if wait:
        print(json.dumps(call.get(), indent=2, sort_keys=True))
