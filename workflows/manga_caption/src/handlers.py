from __future__ import annotations

import datetime as dt
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

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image


DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DATASET_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_CAPTION_MODEL = os.environ.get(
    "DEFAULT_CAPTION_MODEL",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_SOURCE_PREFIX = "datasets/pages/filtered"
DEFAULT_ANNOTATION_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_OUTPUT_PREFIX = "captions"
DEFAULT_CAPTION_RUN = "haiku45_mangazero_panel_simple_v1"
DEFAULT_MANGA_METADATA_REF = os.environ.get(
    "MANGA_METADATA_JSON",
    "metadata/mangazero_manga_credits_20260511.json",
)
DEFAULT_MAX_CONCURRENCY = int(os.environ.get("DEFAULT_MANGA_CAPTION_MAX_CONCURRENCY", "400"))
BEDROCK_MAX_IMAGE_BYTES = int(os.environ.get("BEDROCK_MAX_IMAGE_BYTES", "3600000"))
BEDROCK_MAX_IMAGE_SIDE = int(os.environ.get("BEDROCK_MAX_IMAGE_SIDE", "1800"))
MAX_CHARACTERS_PER_PANEL = 5
MAX_TEXT_REGIONS_PER_PANEL = 7
MIN_PANEL_ENTITY_OVERLAP_RATIO = float(os.environ.get("MIN_PANEL_ENTITY_OVERLAP_RATIO", "0.05"))
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TEXT_BUBBLE_TYPES = ("Speech Bubble", "Narration Bubble", "Shout Bubble")

_S3_CLIENT = None
_BEDROCK_RUNTIME_CLIENTS: dict[tuple[str, int], Any] = {}
_MANGA_METADATA_CACHE: dict[str, dict[str, Any]] = {}

PANEL_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "text_bubble_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_region_index": {"type": "integer"},
                    "type": {"type": "string", "enum": list(ALLOWED_TEXT_BUBBLE_TYPES)},
                },
                "required": ["text_region_index", "type"],
                "additionalProperties": False,
            },
        },
        "caption": {
            "type": "string",
            "description": "One 30-80 word panel-caption body. Do not include manga title, author, or Black and White Manga/Colored Manga prefix.",
        },
    },
    "required": ["text_bubble_types", "caption"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You write concise manga panel captions for image-generation training.

Use the supplied entity labels exactly: Character 1, Character 2, and so on.
Do not describe character identity, hair, clothes, accessories, age, gender, art style, or other visual features.
The caption field must be only the panel-caption body. Do not include the manga title, author, or Black and White Manga/Colored Manga prefix; code will inject that after your response.
Write one caption-body sentence as a medium FLUX prompt, 30-80 words. Start with the visible subject/action, then add setting/background, shot size and camera angle when visible, and useful panel composition details such as speed lines, screen tone, stark white space, dark fill, silhouettes, debris, overlap, or tilted framing.
Be descriptive, not interpretive: do not infer story meaning, intent, mood, cause, hidden emotion, or off-panel events. Describe visible posture, gesture, gaze, mouth shape, facial tension, and panel composition instead.
Never use interpretive words such as distress, distressed, anguish, anxious, overwhelmed, agitation, chaos, foreboding, suggesting, implying, indicating, or conveying. Replace them with visible facts like wide eyes, open mouth shape, hunched posture, speed lines, scattered papers, or tilted framing.
Never mention character clothing or accessories, including glasses, masks, hats, coats, shirts, uniforms, or shoes. Use only Character N plus visible pose, gesture, gaze, mouth shape, and facial tension.
Do not quote, translate, or transcribe any visible text, including signs, narration, dialogue, and standalone non-dialogue lettering; describe it only as a sign, text area, speech bubble, or narration bubble when relevant.
Never mention SFX, sound effects, or sound-effect marks in the caption.
If the background is blank, mostly white, mostly black, speed lines, or abstract screen tone, say that plainly.
Classify each text region as exactly one of: Speech Bubble, Narration Bubble, Shout Bubble.
For text areas, mention only simple bubble presence/type when relevant. Do not transcribe dialogue or infer the written words.
Return only the structured tool output."""


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            region_name=DEFAULT_REGION,
            config=Config(
                region_name=DEFAULT_REGION,
                retries={"mode": "adaptive", "max_attempts": 10},
                max_pool_connections=256,
                connect_timeout=10,
                read_timeout=300,
            ),
        )
    return _S3_CLIENT


def _bedrock_runtime_client(timeout_seconds: float, *, region: str):
    read_timeout = max(1, min(900, int(math.ceil(float(timeout_seconds or 900.0)))))
    cache_key = (region, read_timeout)
    client = _BEDROCK_RUNTIME_CLIENTS.get(cache_key)
    if client is None:
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                region_name=region,
                retries={"mode": "standard", "total_max_attempts": 1},
                connect_timeout=10,
                read_timeout=read_timeout,
            ),
        )
        _BEDROCK_RUNTIME_CLIENTS[cache_key] = client
    return client


def _json_dumps(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None) + "\n"


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_token() -> str:
    return f"{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000, 9999)}"


def _normalize_prefix(value: object, *, default: str) -> str:
    prefix = str(value or default).strip().strip("/")
    if not prefix:
        raise ValueError("S3 prefix must not be empty")
    return prefix


def _join_key(*parts: object) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    text = str(uri or "").strip()
    if not text.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = text[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def _split_bedrock_model_ref(model_ref: str) -> tuple[str, str]:
    raw = str(model_ref or "").strip()
    if "|" in raw:
        region, model = raw.split("|", 1)
        region = region.strip() or DEFAULT_REGION
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid Bedrock model ref: {model_ref!r}")
        return region, model
    if not raw:
        raise ValueError("Bedrock model must not be empty")
    return DEFAULT_REGION, raw


def _get_s3_bytes(bucket: str, key: str) -> bytes:
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def _put_s3_json(bucket: str, key: str, payload: object, *, pretty: bool = False) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=_json_dumps(payload, pretty=pretty).encode("utf-8"),
        ContentType="application/json",
    )


def _object_exists(bucket: str, key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _get_s3_json_or_jsonl(bucket: str, key: str) -> dict[str, Any]:
    text = _get_s3_bytes(bucket, key).decode("utf-8").strip()
    if not text:
        raise ValueError(f"Empty JSON object at {_join_s3_uri(bucket, key)}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for line in text.splitlines():
            line = line.strip()
            if line:
                payload = json.loads(line)
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object/JSONL row at {_join_s3_uri(bucket, key)}")
    return payload


def _load_json_ref(ref: str) -> object:
    if ref.startswith("s3://"):
        bucket, key = _parse_s3_uri(ref)
        return json.loads(_get_s3_bytes(bucket, key).decode("utf-8"))
    path = Path(ref)
    if not path.is_absolute():
        path = WORKFLOW_ROOT / path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_dataset_suffix(value: object) -> str:
    return re.sub(r"_(?:mangazero|manga109)$", "", str(value or "").strip(), flags=re.IGNORECASE)


def _metadata_lookup_key(value: object) -> str:
    text = _strip_dataset_suffix(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _metadata_key_candidates(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    candidates = [
        raw,
        _strip_dataset_suffix(raw),
        raw.replace("_", "-"),
        _strip_dataset_suffix(raw).replace("_", "-"),
        raw.replace("-", " "),
        _strip_dataset_suffix(raw).replace("-", " "),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _metadata_lookup_key(candidate)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _title_from_chapter(chapter: object) -> str:
    text = _strip_dataset_suffix(chapter)
    text = re.sub(r"[-_]+", " ", text).strip()
    return " ".join(word if word.isupper() else word[:1].upper() + word[1:] for word in text.split())


def _iter_metadata_records(payload: object):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield "", item
        return
    if not isinstance(payload, dict):
        return
    for list_key in ("items", "mangas", "metadata", "chapters", "credits", "series"):
        rows = payload.get(list_key)
        if isinstance(rows, list):
            for item in rows:
                if isinstance(item, dict):
                    yield "", item
            return
    for key, value in payload.items():
        if isinstance(value, dict):
            yield str(key), value
        elif isinstance(value, str):
            yield str(key), {"author": value}


def _clean_text(value: object, *, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def _load_manga_metadata_index(metadata_ref: object) -> dict[str, Any]:
    ref = str(metadata_ref or "").strip()
    if not ref:
        return {}
    cached = _MANGA_METADATA_CACHE.get(ref)
    if cached is not None:
        return cached
    try:
        payload = _load_json_ref(ref)
    except Exception:
        _MANGA_METADATA_CACHE[ref] = {}
        return {}

    index: dict[str, Any] = {}
    for record_key, record in _iter_metadata_records(payload):
        title = _clean_text(
            record.get("title")
            or record.get("display_title")
            or record.get("manga_title")
            or record.get("manga_name")
            or record.get("name")
            or record_key,
            max_chars=180,
        )
        mangaka = _clean_text(
            record.get("mangaka")
            or record.get("credit_name")
            or record.get("author")
            or record.get("artist")
            or record.get("creator"),
            max_chars=180,
        )
        if not title and not mangaka:
            continue
        credit = {"title": title, "mangaka": mangaka, "metadata_key": record_key, "metadata_source": ref}
        for key_value in (
            record_key,
            record.get("key"),
            record.get("slug"),
            record.get("manga_slug"),
            record.get("chapter"),
            record.get("folder"),
            record.get("target_folder"),
            record.get("source_book"),
            title,
        ):
            for lookup_key in _metadata_key_candidates(key_value):
                index.setdefault(lookup_key, credit)
    _MANGA_METADATA_CACHE[ref] = index
    return index


def _lookup_manga_credit(chapter: str, annotation: dict[str, Any], metadata_ref: str) -> dict[str, Any]:
    index = _load_manga_metadata_index(metadata_ref)
    source = annotation.get("source") if isinstance(annotation.get("source"), dict) else {}
    candidates = [
        chapter,
        annotation.get("chapter"),
        annotation.get("sample_id"),
        source.get("chapter"),
        source.get("target_folder"),
        source.get("source_book"),
    ]
    for candidate in candidates:
        for lookup_key in _metadata_key_candidates(candidate):
            credit = index.get(lookup_key)
            if credit:
                return {
                    "title": str(credit.get("title") or _title_from_chapter(chapter)),
                    "mangaka": str(credit.get("mangaka") or ""),
                    "metadata_key": str(credit.get("metadata_key") or ""),
                    "metadata_source": str(credit.get("metadata_source") or metadata_ref),
                    "fallback": False,
                }
    return {
        "title": _title_from_chapter(chapter),
        "mangaka": "",
        "metadata_key": "",
        "metadata_source": metadata_ref,
        "fallback": True,
    }


def _page_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def _infer_manga_rendering_label(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256), Image.Resampling.BILINEAR)
            pixels = list(image.getdata())
        if not pixels:
            return "Black and White Manga"
        step = max(1, len(pixels) // 20000)
        sampled = pixels[::step]
        colorful = 0
        chroma_total = 0.0
        for red, green, blue in sampled:
            high = max(red, green, blue)
            low = min(red, green, blue)
            chroma = high - low
            chroma_total += chroma
            if chroma >= 24 and high >= 40:
                colorful += 1
        if colorful / max(1, len(sampled)) >= 0.03 and chroma_total / max(1, len(sampled)) >= 8.0:
            return "Colored Manga"
    except Exception:
        pass
    return "Black and White Manga"


def _caption_prefix(rendering_label: str, manga_credit: dict[str, Any]) -> str:
    style = "Colored Manga" if str(rendering_label or "").strip() == "Colored Manga" else "Black and White Manga"
    title = _clean_text(manga_credit.get("title"), max_chars=180)
    mangaka = _clean_text(manga_credit.get("mangaka"), max_chars=180)
    parts = [f"{style}."]
    if title and mangaka:
        parts.append(f"{title} by {mangaka}.")
    elif title:
        parts.append(f"{title}.")
    elif mangaka:
        parts.append(f"by {mangaka}.")
    return " ".join(parts)


def _enforce_prefix(caption: object, prefix: str) -> str:
    clean_prefix = _clean_text(prefix, max_chars=420)
    text = _clean_text(caption, max_chars=1000)
    if not clean_prefix:
        return text
    if text.lower().startswith(clean_prefix.lower()):
        rest = text[len(clean_prefix) :].strip(" -:")
        return f"{clean_prefix} {rest}".strip() if rest else clean_prefix
    text = re.sub(r"^(?:black\s+and\s+white|colored)\s+manga\.?\s*", "", text, flags=re.IGNORECASE).strip(" -:")
    return f"{clean_prefix} {text}".strip() if text else clean_prefix


def _normalize_text_bubble_type(value: object) -> str:
    text = " ".join(str(value or "").split()).strip().lower()
    if "narration" in text or "caption" in text:
        return "Narration Bubble"
    if "shout" in text or "yell" in text or "scream" in text or "burst" in text:
        return "Shout Bubble"
    return "Speech Bubble"


def _apply_text_bubble_types(panel: dict[str, Any], typed_regions: object) -> dict[str, Any]:
    by_index: dict[int, str] = {}
    if isinstance(typed_regions, list):
        for item in typed_regions:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("text_region_index"))
            except (TypeError, ValueError):
                continue
            by_index[index] = _normalize_text_bubble_type(item.get("type"))

    updated_bubbles: list[dict[str, Any]] = []
    for index, text_region in enumerate(panel.get("text_bubbles") or []):
        if not isinstance(text_region, dict):
            continue
        text_region_index = int(text_region.get("text_region_index") or index)
        updated_bubbles.append(
            {
                **text_region,
                "type": by_index.get(text_region_index, _normalize_text_bubble_type(text_region.get("type"))),
            }
        )
    return {**panel, "text_bubbles": updated_bubbles}


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _coerce_pixel_box(box: object, *, width: int, height: int) -> list[float] | None:
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
    return [x0, y0, x1, y1]


def _bbox_norm(box: list[float], *, width: int, height: int) -> list[float]:
    return [
        round(box[0] / max(1, width), 6),
        round(box[1] / max(1, height), 6),
        round(box[2] / max(1, width), 6),
        round(box[3] / max(1, height), 6),
    ]


def _panel_relative_norm_box(entity_box: list[float], panel_box: list[float]) -> list[float] | None:
    ex0, ey0, ex1, ey1 = entity_box
    px0, py0, px1, py1 = panel_box
    ix0 = max(ex0, px0)
    iy0 = max(ey0, py0)
    ix1 = min(ex1, px1)
    iy1 = min(ey1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    entity_area = _bbox_area(entity_box)
    if entity_area <= 0:
        return None
    if _bbox_area([ix0, iy0, ix1, iy1]) / entity_area < MIN_PANEL_ENTITY_OVERLAP_RATIO:
        return None
    panel_width = max(1.0, px1 - px0)
    panel_height = max(1.0, py1 - py0)
    return [
        round((ix0 - px0) / panel_width, 6),
        round((iy0 - py0) / panel_height, 6),
        round((ix1 - px0) / panel_width, 6),
        round((iy1 - py0) / panel_height, 6),
    ]


def _entity_overlaps_panel(entity_box: list[float] | None, panel_box: list[float] | None) -> bool:
    if entity_box is None or panel_box is None:
        return False
    ex0, ey0, ex1, ey1 = entity_box
    px0, py0, px1, py1 = panel_box
    ix0 = max(ex0, px0)
    iy0 = max(ey0, py0)
    ix1 = min(ex1, px1)
    iy1 = min(ey1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    area = _bbox_area(entity_box)
    return area > 0 and _bbox_area([ix0, iy0, ix1, iy1]) / area >= MIN_PANEL_ENTITY_OVERLAP_RATIO


def _image_block_from_crop(page_image: Image.Image, panel_box: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    x0, y0, x1, y1 = panel_box
    crop = page_image.crop((int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1)))).convert("RGB")
    original_width, original_height = crop.size
    if max(crop.size) > BEDROCK_MAX_IMAGE_SIDE:
        crop.thumbnail((BEDROCK_MAX_IMAGE_SIDE, BEDROCK_MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    for quality in (92, 84, 76, 68):
        output = io.BytesIO()
        crop.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = output.getvalue()
        if len(encoded) <= BEDROCK_MAX_IMAGE_BYTES:
            width, height = crop.size
            return (
                {"image": {"format": "jpeg", "source": {"bytes": encoded}}},
                {
                    "image_format": "jpeg",
                    "image_bytes": len(encoded),
                    "image_width": int(width),
                    "image_height": int(height),
                    "source_panel_width": int(original_width),
                    "source_panel_height": int(original_height),
                },
            )
    raise ValueError(f"Panel crop could not be encoded below {BEDROCK_MAX_IMAGE_BYTES} bytes")


def _annotation_panels(annotation: dict[str, Any], *, width: int, height: int) -> list[dict[str, Any]]:
    detections = annotation.get("detections") if isinstance(annotation.get("detections"), dict) else {}
    raw_panels = detections.get("panels") if isinstance(detections.get("panels"), list) else []
    raw_characters = detections.get("characters") if isinstance(detections.get("characters"), list) else []
    raw_texts = (
        detections.get("texts")
        if isinstance(detections.get("texts"), list)
        else detections.get("text_regions")
        if isinstance(detections.get("text_regions"), list)
        else []
    )

    character_rows = []
    for source_index, character in enumerate(raw_characters):
        if not isinstance(character, dict):
            continue
        box = _coerce_pixel_box(character.get("bbox") or character.get("bbox_norm"), width=width, height=height)
        if box is None:
            continue
        character_rows.append(
            {
                "source_character_index": source_index,
                "source_character_id": str(character.get("source_character_id") or character.get("cluster_id") or ""),
                "bbox": [int(round(v)) for v in box],
                "bbox_norm": _bbox_norm(box, width=width, height=height),
                "_pixel_box": box,
            }
        )

    text_rows = []
    for source_index, text_region in enumerate(raw_texts):
        if not isinstance(text_region, dict):
            continue
        box = _coerce_pixel_box(text_region.get("bbox") or text_region.get("bbox_norm"), width=width, height=height)
        if box is None:
            continue
        text_rows.append(
            {
                "source_text_index": source_index,
                "bbox": [int(round(v)) for v in box],
                "bbox_norm": _bbox_norm(box, width=width, height=height),
                "_pixel_box": box,
            }
        )

    if not raw_panels:
        raw_panels = [{"bbox": [0, 0, width, height], "panel_id": "full_page"}]

    panels: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(raw_panels):
        if not isinstance(panel, dict):
            continue
        panel_box = _coerce_pixel_box(panel.get("bbox") or panel.get("bbox_norm"), width=width, height=height)
        if panel_box is None:
            continue
        panel_characters = []
        for local_index, character in enumerate(
            row for row in character_rows if _entity_overlaps_panel(row["_pixel_box"], panel_box)
        ):
            if local_index >= MAX_CHARACTERS_PER_PANEL:
                break
            panel_relative = _panel_relative_norm_box(character["_pixel_box"], panel_box)
            panel_characters.append(
                {
                    "id": f"Character {local_index + 1}",
                    "character_index": local_index,
                    "source_character_index": character["source_character_index"],
                    "source_character_id": character["source_character_id"],
                    "bbox": character["bbox"],
                    "bbox_norm": character["bbox_norm"],
                    "panel_bbox_norm": panel_relative,
                }
            )
        panel_texts = []
        for local_index, text_region in enumerate(row for row in text_rows if _entity_overlaps_panel(row["_pixel_box"], panel_box)):
            if local_index >= MAX_TEXT_REGIONS_PER_PANEL:
                break
            panel_relative = _panel_relative_norm_box(text_region["_pixel_box"], panel_box)
            panel_texts.append(
                {
                    "id": f"Text Region {local_index + 1}",
                    "text_region_index": local_index,
                    "source_text_index": text_region["source_text_index"],
                    "type": "Speech Bubble",
                    "bbox": text_region["bbox"],
                    "bbox_norm": text_region["bbox_norm"],
                    "panel_bbox_norm": panel_relative,
                }
            )
        panels.append(
            {
                "panel_index": panel_index,
                "panel_id": str(panel.get("panel_id") or f"panel_{panel_index:03d}"),
                "bbox": [int(round(v)) for v in panel_box],
                "bbox_norm": _bbox_norm(panel_box, width=width, height=height),
                "characters": panel_characters,
                "text_bubbles": panel_texts,
            }
        )
    return panels


def _bedrock_tool_input(response: dict[str, Any], *, tool_name: str) -> dict[str, Any]:
    output = response.get("output") if isinstance(response.get("output"), dict) else {}
    message = output.get("message") if isinstance(output.get("message"), dict) else {}
    for block in message.get("content") or []:
        tool_use = block.get("toolUse") if isinstance(block, dict) else None
        if isinstance(tool_use, dict) and str(tool_use.get("name") or "") == tool_name:
            value = tool_use.get("input")
            if not isinstance(value, dict):
                raise ValueError(f"Bedrock tool {tool_name!r} returned non-object input")
            return value
    raise ValueError(f"Bedrock response did not contain tool input for {tool_name!r}")


def _bedrock_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
        "total_tokens": int(usage.get("totalTokens") or 0),
    }


def _caption_one_panel(
    *,
    model: str,
    timeout_seconds: float,
    retries: int,
    max_output_tokens: int,
    image_block: dict[str, Any],
    caption_prefix: str,
    panel: dict[str, Any],
    request_id: str,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    character_lines = [
        f"- Character {idx + 1}: bbox_norm_in_panel={character.get('panel_bbox_norm')}"
        for idx, character in enumerate(panel.get("characters") or [])
    ]
    text_lines = [
        f"- Text Region index {int(text_region.get('text_region_index') or idx)}: bbox_norm_in_panel={text_region.get('panel_bbox_norm')}"
        for idx, text_region in enumerate(panel.get("text_bubbles") or [])
    ]
    user_text = "\n".join(
        [
            "Caption this single manga panel.",
            "A metadata prefix with manga rendering, title, and author will be injected by code after your response.",
            "Do not include Black and White Manga/Colored Manga, manga title, or author in your caption field.",
            "",
            "Characters to reference:",
            *(character_lines or ["- none"]),
            "",
            "Text regions:",
            *(text_lines or ["- none"]),
            "",
            "Write one single 30-80 word caption-body sentence. Use Character N labels for characters.",
            "Use this FLUX-friendly order when possible: subject/action, setting/background, shot size and camera angle, useful panel composition details.",
            "Focus on visible actions, poses, gaze, mouth shape, facial tension, and interactions.",
            "Do not infer story meaning, intent, mood, cause, or hidden emotion.",
            "Avoid these words: distress, distressed, anguish, anxious, overwhelmed, agitation, chaos, foreboding, suggesting, implying, indicating, conveying.",
            "Do not describe appearance details, clothing, accessories, hair, age, or identity; the character reference images provide identity.",
            "Do not mention glasses, masks, hats, coats, shirts, uniforms, shoes, or other accessories.",
            "Do not quote, translate, or transcribe visible text, signs, narration, dialogue, or standalone non-dialogue lettering.",
            "Never mention SFX, sound effects, or sound-effect marks in the caption.",
            "Do not mention bbox coordinates or panel positions in the caption.",
            "For every listed text region, return its type as Speech Bubble, Narration Bubble, or Shout Bubble.",
        ]
    )

    region, model_id = _split_bedrock_model_ref(model)
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            response = _bedrock_runtime_client(timeout_seconds, region=region).converse(
                modelId=model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_text}, image_block]}],
                inferenceConfig={
                    "maxTokens": max(128, int(max_output_tokens)),
                    "temperature": 0,
                },
                toolConfig={
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "panel_caption",
                                "description": "Return one simple panel caption using Character N labels.",
                                "inputSchema": {"json": PANEL_CAPTION_SCHEMA},
                            }
                        }
                    ],
                    "toolChoice": {"tool": {"name": "panel_caption"}},
                },
                requestMetadata={"client_request_id": request_id[:256]},
            )
            parsed = _bedrock_tool_input(response, tool_name="panel_caption")
            caption = _enforce_prefix(parsed.get("caption"), caption_prefix)
            text_bubble_types = parsed.get("text_bubble_types")
            return (
                caption,
                text_bubble_types if isinstance(text_bubble_types, list) else [],
                _bedrock_usage(response),
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(min(20.0, 1.5 * (2**attempt)) + random.random())
    raise RuntimeError(f"Panel caption Bedrock call failed after {max(1, retries)} attempts: {last_error}")


def _list_annotation_relatives(*, bucket: str, annotation_prefix: str) -> set[str]:
    prefix = annotation_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key.endswith(".jsonl") or key.endswith(".json"):
                relatives.add(key[len(prefix) :])
    return relatives


def _list_existing_output_relatives(*, bucket: str, output_root: str) -> set[str]:
    prefix = output_root.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    relatives: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if "/_jobs/" in key or "/_audit/" in key:
                continue
            if key.endswith(".json"):
                relatives.add(key[len(prefix) :])
    return relatives


def _list_page_rows(
    *,
    bucket: str,
    source_prefix: str,
    annotation_prefix: str,
    output_root: str,
    include_chapter_regex: str,
    overwrite: bool,
    require_annotations: bool,
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotation_relatives = _list_annotation_relatives(bucket=bucket, annotation_prefix=annotation_prefix)
    existing_output_relatives = (
        set() if overwrite else _list_existing_output_relatives(bucket=bucket, output_root=output_root)
    )
    include_re = re.compile(include_chapter_regex) if include_chapter_regex else None
    source_root = source_prefix.rstrip("/") + "/"
    paginator = _s3_client().get_paginator("list_objects_v2")
    rows: list[dict[str, Any]] = []
    stats = {
        "source_image_count": 0,
        "skipped_non_image_count": 0,
        "skipped_chapter_regex_count": 0,
        "skipped_missing_annotation_count": 0,
        "skipped_existing_count": 0,
    }

    for page in paginator.paginate(Bucket=bucket, Prefix=source_root):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            suffix = Path(key).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                stats["skipped_non_image_count"] += 1
                continue
            stats["source_image_count"] += 1
            relative = key[len(source_root) :]
            parts = relative.split("/")
            if len(parts) != 2:
                continue
            chapter, filename = parts
            if include_re and not include_re.search(chapter):
                stats["skipped_chapter_regex_count"] += 1
                continue
            page_id = Path(filename).stem
            annotation_relative = f"{chapter}/{page_id}.jsonl"
            if require_annotations and annotation_relative not in annotation_relatives:
                stats["skipped_missing_annotation_count"] += 1
                continue
            output_relative = f"{chapter}/{page_id}.json"
            output_key = _join_key(output_root, output_relative)
            if not overwrite and output_relative in existing_output_relatives:
                stats["skipped_existing_count"] += 1
                continue
            rows.append(
                {
                    "chapter": chapter,
                    "page_id": page_id,
                    "relative_path": relative,
                    "page_key": key,
                    "annotation_key": _join_key(annotation_prefix, annotation_relative),
                    "output_key": output_key,
                    "page_s3_uri": _join_s3_uri(bucket, key),
                    "annotation_s3_uri": _join_s3_uri(bucket, _join_key(annotation_prefix, annotation_relative)),
                }
            )
            if max_pages > 0 and len(rows) >= max_pages:
                return rows, stats
    return rows, stats


def prepare_manga_caption_config(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or DATASET_BUCKET).strip()
    source_prefix = _normalize_prefix(event.get("source_prefix"), default=DEFAULT_SOURCE_PREFIX)
    annotation_prefix = _normalize_prefix(event.get("annotation_prefix"), default=DEFAULT_ANNOTATION_PREFIX)
    output_prefix = _normalize_prefix(event.get("output_prefix"), default=DEFAULT_OUTPUT_PREFIX)
    caption_run = str(event.get("caption_run") or DEFAULT_CAPTION_RUN).strip().strip("/")
    if not caption_run:
        raise ValueError("caption_run must not be empty")
    output_root = _join_key(output_prefix, caption_run)
    run_id = str(event.get("run_id") or _run_token()).strip()
    include_chapter_regex = str(event.get("include_chapter_regex") or "").strip()
    max_pages = max(0, int(event.get("max_pages") or 0))
    overwrite = bool(event.get("overwrite", False))
    require_annotations = bool(event.get("require_annotations", True))
    model = str(event.get("model") or DEFAULT_CAPTION_MODEL).strip()
    manga_metadata_json = str(event.get("manga_metadata_json") or DEFAULT_MANGA_METADATA_REF).strip()

    rows, page_stats = _list_page_rows(
        bucket=bucket,
        source_prefix=source_prefix,
        annotation_prefix=annotation_prefix,
        output_root=output_root,
        include_chapter_regex=include_chapter_regex,
        overwrite=overwrite,
        require_annotations=require_annotations,
        max_pages=max_pages,
    )
    manifest_key = _join_key(output_root, "_jobs", run_id, "page_manifest.jsonl")
    manifest_body = "".join(_json_dumps(row) for row in rows)
    _s3_client().put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_body.encode("utf-8"),
        ContentType="application/x-ndjson",
    )

    worker_config = {
        "bucket": bucket,
        "source_prefix": source_prefix,
        "annotation_prefix": annotation_prefix,
        "output_prefix": output_prefix,
        "caption_run": caption_run,
        "output_root": output_root,
        "run_id": run_id,
        "model": model,
        "manga_metadata_json": manga_metadata_json,
        "timeout_seconds": float(event.get("timeout_seconds") or 180.0),
        "retries": max(1, int(event.get("retries") or 3)),
        "max_output_tokens": max(128, int(event.get("max_output_tokens") or 512)),
        "overwrite": overwrite,
        "created_at": _now_utc_iso(),
        "git_sha": str(event.get("git_sha") or ""),
    }
    config_key = _join_key(output_root, "_jobs", run_id, "worker_config.json")
    _put_s3_json(bucket, config_key, worker_config, pretty=True)

    return {
        "schema_version": 1,
        "source": {"bucket": bucket, "page_manifest_key": manifest_key},
        "worker_config": {"bucket": bucket, "key": config_key},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or DEFAULT_MAX_CONCURRENCY))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 0))},
        "audit": {"bucket": bucket, "prefix": _join_key(output_root, "_audit", run_id) + "/"},
        "output": {"bucket": bucket, "prefix": output_root, "caption_run": caption_run, "run_id": run_id},
        "stats": {**page_stats, "manifest_count": len(rows)},
    }


def caption_manga_page(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    config_ref = event.get("config_ref") if isinstance(event.get("config_ref"), dict) else {}
    config = _get_s3_json_or_jsonl(str(config_ref["bucket"]), str(config_ref["key"]))
    bucket = str(config["bucket"])
    page = event.get("page") if isinstance(event.get("page"), dict) else event
    output_key = str(page["output_key"])
    if not bool(config.get("overwrite")) and _object_exists(bucket, output_key):
        return {"status": "skipped_existing", "output_key": output_key, "page_id": str(page.get("page_id") or "")}

    page_bytes = _get_s3_bytes(bucket, str(page["page_key"]))
    width, height = _page_image_dimensions(page_bytes)
    annotation = _get_s3_json_or_jsonl(bucket, str(page["annotation_key"]))
    panels = _annotation_panels(annotation, width=width, height=height)
    rendering_label = _infer_manga_rendering_label(page_bytes)
    manga_credit = _lookup_manga_credit(str(page["chapter"]), annotation, str(config.get("manga_metadata_json") or ""))
    caption_prefix = _caption_prefix(rendering_label, manga_credit)

    with Image.open(io.BytesIO(page_bytes)) as image:
        page_image = image.convert("RGB")
        usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        captioned_panels: list[dict[str, Any]] = []
        for panel in panels:
            image_block, image_meta = _image_block_from_crop(page_image, panel["bbox"])
            request_source = f"{config.get('run_id')}:{page.get('chapter')}:{page.get('page_id')}:{panel['panel_index']}"
            request_id = hashlib.sha1(request_source.encode("utf-8")).hexdigest()
            caption, text_bubble_types, usage = _caption_one_panel(
                model=str(config["model"]),
                timeout_seconds=float(config.get("timeout_seconds") or 180.0),
                retries=int(config.get("retries") or 3),
                max_output_tokens=int(config.get("max_output_tokens") or 512),
                image_block=image_block,
                caption_prefix=caption_prefix,
                panel=panel,
                request_id=request_id,
            )
            for key, value in usage.items():
                usage_total[key] = int(usage_total.get(key, 0)) + int(value)
            typed_panel = _apply_text_bubble_types(panel, text_bubble_types)
            captioned_panels.append({**typed_panel, "caption": caption, "image": image_meta})

    output = {
        "schema_version": 2,
        "caption_type": "haiku_panel_simple_v1",
        "status": "ok",
        "caption_run": str(config["caption_run"]),
        "run_id": str(config["run_id"]),
        "created_at": _now_utc_iso(),
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "page_size": {"width_px": int(width), "height_px": int(height)},
        "caption_prefix": caption_prefix,
        "manga_rendering": rendering_label,
        "manga_credit": manga_credit,
        "sources": {
            "page": str(page.get("page_s3_uri") or _join_s3_uri(bucket, str(page["page_key"]))),
            "annotation": str(page.get("annotation_s3_uri") or _join_s3_uri(bucket, str(page["annotation_key"]))),
            "annotation_type": "magi_v3",
            "page_key": str(page["page_key"]),
            "annotation_key": str(page["annotation_key"]),
        },
        "model": {"id": str(config["model"]), "provider": "bedrock"},
        "panels": captioned_panels,
        "usage": usage_total,
    }
    _put_s3_json(bucket, output_key, output, pretty=False)
    return {
        "status": "ok",
        "chapter": str(page["chapter"]),
        "page_id": str(page["page_id"]),
        "output_key": output_key,
        "panel_count": len(captioned_panels),
        "usage": usage_total,
    }
