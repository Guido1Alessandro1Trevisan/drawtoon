"""Modal app: distributed Magi v3 annotation on 40 H200s.

Inspired by ``lineart2/6magi_3.py``. Loads ``ragavsachdeva/magiv3`` once per
container (via ``@modal.enter(snap=True)`` so the load is captured in the
memory snapshot), runs ``predict_detections_and_associations`` in batches,
and writes one JSONL annotation per manga page to S3 in the schema the
manga_caption workflow already consumes.

Usage
-----

Deploy::

    modal deploy modal_magi.py

Smoke-test one page::

    modal run modal_magi.py::smoke_test

Bulk-annotate via the launcher::

    python start.py --chapters jujutsu-kaisen --gpu-batch-size 8
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


MAGI_V3_REPO = "ragavsachdeva/magiv3"
HF_HOME = "/root/.cache/huggingface"

S3_BUCKET_DEFAULT = os.environ.get("DRAWTOON_S3_BUCKET", "drawtoon")
AWS_REGION_DEFAULT = os.environ.get("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3")
MODAL_REGION = os.environ.get("DRAWTOON_MODAL_REGION", "us-east-1")

DEFAULT_MAX_CONTAINERS = int(os.environ.get("MAGI_V3_MAX_CONTAINERS", "40"))
DEFAULT_GPU_BATCH_SIZE = int(os.environ.get("MAGI_V3_BATCH_SIZE", "8"))
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GEMINI_API_KEY_SECRET_NAME = os.environ.get("GEMINI_API_KEY_SECRET_NAME", "drawtoon/gemini-api-key")
GEMINI_MODAL_SECRET_NAME = os.environ.get("GEMINI_MODAL_SECRET_NAME", "gemini-api-key")
GEMINI_API_KEY_ENV = "GOOGLE_GENERATIVE_AI_API_KEY"
DEFAULT_GEMINI_VERIFIER_MODEL = os.environ.get("GEMINI_VERIFIER_MODEL", "gemini-3-flash-preview")
DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL = os.environ.get("GEMINI_VERIFIER_THINKING_LEVEL", "HIGH")
GEMINI_VERIFIER_MAX_IMAGE_SIDE = int(os.environ.get("GEMINI_VERIFIER_MAX_IMAGE_SIDE", "2048"))
GEMINI_VERIFIER_MAX_IMAGE_BYTES = int(os.environ.get("GEMINI_VERIFIER_MAX_IMAGE_BYTES", "8000000"))
# How many pages in a shard run their Gemini verifier in parallel. Gemini calls
# are I/O-bound (multi-second LLM latency) so a thread pool yields near-linear
# speedup. Matches the shard size (16) so every page in a shard fires its
# Gemini call concurrently — 16 workers x 40 containers = up to 640 in-flight
# calls cluster-wide.
GEMINI_VERIFIER_PARALLELISM = int(os.environ.get("GEMINI_VERIFIER_PARALLELISM", "16"))
GEMINI_CHARACTER_VERIFIER_PROMPT_VERSION = "magi_v3_gemini_character_verifier_v2_clean_image_coords"

GEMINI_CHARACTER_VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "character_boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_id": {"type": "string"},
                    "final_label": {
                        "type": "string",
                        "description": "Short visible human description, or exactly NoCharacter.",
                    },
                    "decision": {
                        "type": "string",
                        "format": "enum",
                        "enum": [
                            "keep_magi",
                            "correct_magi",
                            "drop_not_character",
                            "drop_duplicate",
                            "drop_bystander_or_silhouette",
                            "uncertain_keep",
                        ],
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short visible-evidence explanation, not hidden reasoning.",
                    },
                },
                "required": ["bbox_id", "final_label", "decision", "reason"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["character_boxes"],
}

# `hf_transfer` accelerates HF Hub downloads ~3x on a cold volume; pinning
# torch/torchvision so transformers==4.49.0 stays compatible.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate==1.4.0",
        "boto3==1.35.99",
        "einops==0.8.0",
        "hf_transfer==0.1.8",
        "matplotlib==3.10.0",
        "networkx==3.4.2",
        "pillow==11.1.0",
        "pytorch-metric-learning==2.8.1",
        "safetensors==0.5.2",
        "shapely==2.0.6",
        "google-genai>=1.0.0",
        "timm==1.0.13",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.49.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_HOME})
)

app = modal.App("drawtoon-manga-annotate", image=image)
hf_volume = modal.Volume.from_name("magi-hf-cache", create_if_missing=True)
aws_secret = modal.Secret.from_name(AWS_SECRET_NAME)
gemini_secret = modal.Secret.from_name(GEMINI_MODAL_SECRET_NAME)
worker_secrets = [aws_secret, gemini_secret]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    value = str(uri).strip()
    if not value.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = value[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 URI: {uri!r}")
    return bucket, key


def _join_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def _json_friendly(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_friendly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_friendly(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_friendly(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _s3_client_cached() -> Any:
    import boto3
    from botocore.config import Config

    if not hasattr(_s3_client_cached, "_client"):
        _s3_client_cached._client = boto3.client(  # type: ignore[attr-defined]
            "s3",
            region_name=AWS_REGION_DEFAULT,
            config=Config(
                retries={"mode": "adaptive", "max_attempts": 10},
                connect_timeout=10,
                read_timeout=120,
                max_pool_connections=128,
            ),
        )
    return _s3_client_cached._client  # type: ignore[attr-defined]


def _head_exists(bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        _s3_client_cached().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _download_rgb_image(bucket: str, key: str) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageOps

    response = _s3_client_cached().get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()
    image_obj = Image.open(io.BytesIO(image_bytes))
    image_obj = ImageOps.exif_transpose(image_obj).convert("RGB")
    return image_obj, {
        "bucket": bucket,
        "key": key,
        "s3_uri": _join_s3_uri(bucket, key),
        "etag": str(response.get("ETag", "")).strip('"'),
        "content_length": int(response.get("ContentLength", 0) or 0),
    }


def _put_jsonl_object(bucket: str, key: str, payload: dict[str, Any]) -> None:
    body = (json.dumps(_json_friendly(payload), ensure_ascii=False) + "\n").encode("utf-8")
    _s3_client_cached().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/x-ndjson; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Optional Gemini verifier
# ---------------------------------------------------------------------------


def _resolve_gemini_api_key() -> str:
    env_value = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if env_value:
        return env_value
    for fallback_env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        env_value = os.environ.get(fallback_env, "").strip()
        if env_value:
            return env_value

    import boto3

    response = boto3.client("secretsmanager", region_name=AWS_REGION_DEFAULT).get_secret_value(
        SecretId=GEMINI_API_KEY_SECRET_NAME
    )
    value = str(response.get("SecretString") or "").strip()
    if not value:
        raise RuntimeError(f"empty Gemini API secret {GEMINI_API_KEY_SECRET_NAME!r}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict):
        for key in (GEMINI_API_KEY_ENV, "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            if parsed.get(key):
                return str(parsed[key]).strip()
    return value


# Per-thread genai client. Sharing one client across the verifier ThreadPool
# (16 workers per shard) caused `[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC]`
# read errors — the underlying httpx connection's TLS state is not safe to
# multiplex across concurrent threads. Each thread now gets its own client
# (and therefore its own connection pool); construction is cheap.
_gemini_client_tls = threading.local()


def _gemini_client_cached() -> Any:
    client = getattr(_gemini_client_tls, "client", None)
    if client is None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=_resolve_gemini_api_key(),
            http_options=types.HttpOptions(timeout=240000),
        )
        _gemini_client_tls.client = client
    return client


def _clean_verifier_label(value: object) -> str:
    label = " ".join(str(value or "").strip().split())
    if not label:
        return "NoCharacter"
    if re.sub(r"[^a-z]", "", label.lower()) in {"nocharacter", "none", "notcharacter"}:
        return "NoCharacter"
    return label[:160]


def _encode_image_for_gemini(image_obj: Any, *, format_hint: str = "JPEG") -> tuple[bytes, str, dict[str, int]]:
    from PIL import Image

    image = image_obj.copy()
    if max(image.size) > GEMINI_VERIFIER_MAX_IMAGE_SIDE:
        image.thumbnail(
            (GEMINI_VERIFIER_MAX_IMAGE_SIDE, GEMINI_VERIFIER_MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    if format_hint.upper() == "PNG":
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
        mime = "image/jpeg"
    data = buffer.getvalue()
    if len(data) > GEMINI_VERIFIER_MAX_IMAGE_BYTES and mime == "image/png":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        data = buffer.getvalue()
        mime = "image/jpeg"
    return data, mime, {"image_width": int(image.width), "image_height": int(image.height), "image_bytes": len(data)}


def _build_gemini_verifier_prompt(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are auditing Magi character detections on one manga/webtoon page.",
            "",
            "Input:",
            "- Image 1 is the clean page with no boxes drawn.",
            "- Metadata lists every bbox id and exact pixel coordinates in that image.",
            "- Coordinates are [x0, y0, x1, y1] measured from the top-left of the image.",
            "",
            "Task:",
            "For each bbox id, inspect the region described by its coordinates in the image and classify it.",
            "",
            "Rules:",
            "- Use only bbox ids present in metadata. Do not invent extra bbox ids.",
            "- Assign a HUMAN/PERSON character only.",
            "- Use the same short visible description for the same person on this page.",
            "- Prefer the original Magi label only if it visibly matches; correct it if wrong.",
            "- Return exactly NoCharacter for animals, pets, objects, text, speech bubbles, effects, background, unusable body fragments, or duplicate smaller boxes.",
            "- If two boxes overlap heavily on the same person, keep the larger/clearer box and mark the duplicate NoCharacter.",
            "- Do not mark a visible secondary character NoCharacter just because another person is also nearby or partly inside the box.",
            "- Mark silhouettes, tiny background bystanders, or crowd figures as NoCharacter only when they are not visually identifiable enough to track.",
            "- Do not drop a real face, upper body, back view, masked person, occluded person, or close-up when the person is visually identifiable.",
            "- Do not output story names. Use visual descriptions like 'red-haired woman in red dress'.",
            "- The reason must be a short visible-evidence explanation.",
            "",
            "Return JSON only using the requested schema.",
            "",
            "Metadata JSON:",
            json.dumps(metadata, ensure_ascii=False, indent=2),
        ]
    )


def _normalize_gemini_verifier_payload(payload: dict[str, Any], *, expected_count: int) -> list[dict[str, str]]:
    rows = payload.get("character_boxes")
    if not isinstance(rows, list):
        raise ValueError("Gemini verifier response missing character_boxes[]")
    by_id: dict[str, dict[str, str]] = {}
    valid_decisions = {
        "keep_magi",
        "correct_magi",
        "drop_not_character",
        "drop_duplicate",
        "drop_bystander_or_silhouette",
        "uncertain_keep",
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        bbox_id = str(row.get("bbox_id") or "").strip()
        if not bbox_id:
            continue
        decision = str(row.get("decision") or "").strip()
        if decision not in valid_decisions:
            decision = "drop_not_character" if _clean_verifier_label(row.get("final_label")) == "NoCharacter" else "uncertain_keep"
        by_id[bbox_id] = {
            "bbox_id": bbox_id,
            "final_label": _clean_verifier_label(row.get("final_label")),
            "decision": decision,
            "reason": " ".join(str(row.get("reason") or "").split())[:500],
        }

    expected_ids = [f"bbox{idx}" for idx in range(1, expected_count + 1)]
    missing = [bbox_id for bbox_id in expected_ids if bbox_id not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"Gemini verifier bbox mismatch: missing={missing} extra={extra}")
    return [by_id[bbox_id] for bbox_id in expected_ids]


def _call_gemini_character_verifier(
    *,
    image_obj: Any,
    characters: list[dict[str, Any]],
    sample_id: str,
    model: str,
    thinking_level: str,
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int]]:
    from google.genai import types

    metadata = {
        "sample_id": sample_id,
        "image_size": {"width": int(image_obj.width), "height": int(image_obj.height)},
        "character_boxes_to_classify_exactly_once": [
            {
                "bbox_id": f"bbox{idx + 1}",
                "character_index": idx,
                "magi_source_character_id": str(character.get("source_character_id") or ""),
                "bbox": character.get("bbox"),
                "score": character.get("score"),
            }
            for idx, character in enumerate(characters)
        ],
    }
    clean_bytes, clean_mime, clean_meta = _encode_image_for_gemini(image_obj)

    thinking_enum = getattr(types.ThinkingLevel, str(thinking_level or "HIGH").upper(), types.ThinkingLevel.HIGH)
    response = _gemini_client_cached().models.generate_content(
        model=model,
        contents=[
            "IMAGE 1: clean page. Use metadata coordinates to locate each bbox region.",
            types.Part.from_bytes(data=clean_bytes, mime_type=clean_mime),
            _build_gemini_verifier_prompt(metadata),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_CHARACTER_VERIFIER_SCHEMA,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_enum,
                include_thoughts=False,
            ),
        ),
    )
    payload = json.loads(str(response.text or "{}"))
    rows = _normalize_gemini_verifier_payload(payload, expected_count=len(characters))
    usage_meta = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": int(getattr(usage_meta, "prompt_token_count", 0) or 0) if usage_meta else 0,
        "output_tokens": int(getattr(usage_meta, "candidates_token_count", 0) or 0) if usage_meta else 0,
        "reasoning_tokens": int(getattr(usage_meta, "thoughts_token_count", 0) or 0) if usage_meta else 0,
        "total_tokens": int(getattr(usage_meta, "total_token_count", 0) or 0) if usage_meta else 0,
    }
    image_meta = {
        "input_images": ["clean_image_only"],
        "uses_coordinate_metadata": True,
        "clean_image_width": clean_meta["image_width"],
        "clean_image_height": clean_meta["image_height"],
        "clean_image_bytes": clean_meta["image_bytes"],
    }
    return rows, usage, image_meta


def _remap_text_character_associations(associations: object, old_to_new: dict[int, int]) -> list[Any]:
    remapped: list[Any] = []
    if not isinstance(associations, list):
        return remapped
    for association in associations:
        if isinstance(association, (list, tuple)) and len(association) >= 2:
            try:
                old_character_index = int(association[1])
            except (TypeError, ValueError):
                continue
            if old_character_index not in old_to_new:
                continue
            updated = list(association)
            updated[1] = old_to_new[old_character_index]
            remapped.append(updated)
        elif isinstance(association, dict):
            raw_index = association.get("character_index", association.get("character"))
            try:
                old_character_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if old_character_index not in old_to_new:
                continue
            updated = dict(association)
            updated["character_index"] = old_to_new[old_character_index]
            remapped.append(updated)
    return remapped


def _apply_gemini_character_verification(
    *,
    annotation: dict[str, Any],
    image_obj: Any,
    sample_id: str,
    model: str,
    thinking_level: str,
) -> dict[str, Any]:
    detections = annotation.get("detections") if isinstance(annotation.get("detections"), dict) else {}
    characters = detections.get("characters") if isinstance(detections.get("characters"), list) else []
    annotation["verification"] = {
        "enabled": True,
        "status": "skipped_no_characters" if not characters else "pending",
        "provider": "gemini",
        "model": model,
        "thinking_level": str(thinking_level or "HIGH").upper(),
        "prompt_version": GEMINI_CHARACTER_VERIFIER_PROMPT_VERSION,
    }
    if not characters:
        return annotation

    # Gemini verification is mandatory: any failure propagates to the caller's
    # per-page try/except, which records the page as an error. No fallback to
    # raw MAGI output.
    verifier_rows, usage, image_meta = _call_gemini_character_verifier(
        image_obj=image_obj,
        characters=characters,
        sample_id=sample_id,
        model=model,
        thinking_level=thinking_level,
    )

    kept_characters: list[dict[str, Any]] = []
    old_to_new: dict[int, int] = {}
    dropped: list[dict[str, Any]] = []
    for old_index, (character, verifier_row) in enumerate(zip(characters, verifier_rows)):
        label = verifier_row["final_label"]
        decision = verifier_row["decision"]
        should_drop = label == "NoCharacter" or decision.startswith("drop_")
        audit = {
            "bbox_id": verifier_row["bbox_id"],
            "final_label": label,
            "decision": decision,
            "reason": verifier_row.get("reason") or "",
        }
        if should_drop:
            dropped.append({"character_index": old_index, **audit})
            continue
        updated = dict(character)
        raw_source_id = str(updated.get("source_character_id") or "")
        updated["magi_source_character_id"] = raw_source_id
        updated["source_character_id"] = label
        updated["gemini_character_label"] = label
        updated["gemini_verification"] = audit
        old_to_new[old_index] = len(kept_characters)
        kept_characters.append(updated)

    detections["characters"] = kept_characters
    detections["character_cluster_labels"] = [str(character.get("source_character_id") or "") for character in kept_characters]
    detections["text_character_associations"] = _remap_text_character_associations(
        detections.get("text_character_associations"), old_to_new
    )
    annotation["detections"] = detections
    annotation["tasks"] = list(dict.fromkeys(list(annotation.get("tasks") or []) + ["gemini_character_verification"]))
    summary = annotation.get("summary") if isinstance(annotation.get("summary"), dict) else {}
    summary.update(
        {
            "raw_magi_character_count": len(characters),
            "character_count": len(kept_characters),
            "dropped_character_count": len(dropped),
            "character_cluster_count": len(set(detections["character_cluster_labels"])),
            "text_character_association_count": len(detections["text_character_associations"]),
        }
    )
    annotation["summary"] = summary
    annotation["verification"].update(
        {
            "status": "ok",
            "character_boxes": verifier_rows,
            "dropped_characters": dropped,
            "kept_character_count": len(kept_characters),
            "dropped_character_count": len(dropped),
            "usage": usage,
            "image": image_meta,
        }
    )
    return annotation


# ---------------------------------------------------------------------------
# Output schema — matches the existing magi_v3_page_annotation files on S3.
# Caption pipeline reads: panels[].bbox/panel_id, characters[].bbox/source_character_id,
# texts[].bbox, image_size. Cluster labels + associations are retained for downstream
# attribution. Everything else (panel_index/character_index/text_region_index/etc.)
# is invented and unused — dropped.
# ---------------------------------------------------------------------------


def _annotation_payload(
    *,
    sample_id: str,
    image_obj: Any,
    source: dict[str, Any],
    raw_detections: dict[str, Any],
    run_id: str,
    git_sha: str,
) -> dict[str, Any]:
    detections = dict(raw_detections or {})
    panels = list(detections.get("panels") or [])
    characters = list(detections.get("characters") or [])
    texts = list(detections.get("texts") or [])
    tails = list(detections.get("tails") or [])
    cluster_labels = [str(label) for label in (detections.get("character_cluster_labels") or [])]

    def _coerce_box(box: Any) -> list[int] | None:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    def _ensure_dict(item: Any, *, default_score: float = 1.0) -> dict[str, Any] | None:
        if isinstance(item, dict):
            box = _coerce_box(item.get("bbox") or item.get("box") or item.get("xyxy"))
            if box is None:
                return None
            score = item.get("score")
            try:
                score_value = float(score) if score is not None else default_score
            except (TypeError, ValueError):
                score_value = default_score
            return {"bbox": box, "score": score_value}
        box = _coerce_box(item)
        if box is None:
            return None
        return {"bbox": box, "score": default_score}

    normalized_panels: list[dict[str, Any]] = []
    for idx, panel in enumerate(panels):
        entry = _ensure_dict(panel)
        if entry is None:
            continue
        entry["panel_id"] = f"{sample_id}__panel_{idx:03d}"
        normalized_panels.append(entry)

    normalized_characters: list[dict[str, Any]] = []
    for idx, character in enumerate(characters):
        entry = _ensure_dict(character)
        if entry is None:
            continue
        # The caption pipeline groups recurring characters by source_character_id.
        if idx < len(cluster_labels):
            entry["source_character_id"] = cluster_labels[idx]
        normalized_characters.append(entry)

    normalized_texts = [entry for entry in (_ensure_dict(text) for text in texts) if entry]
    normalized_tails = [entry for entry in (_ensure_dict(tail) for tail in tails) if entry]

    detections_out = {
        "panels": normalized_panels,
        "characters": normalized_characters,
        "texts": normalized_texts,
        "tails": normalized_tails,
        "character_cluster_labels": cluster_labels,
        "text_character_associations": list(detections.get("text_character_associations") or []),
        "text_tail_associations": list(detections.get("text_tail_associations") or []),
    }

    return _json_friendly(
        {
            "schema_name": "magi_v3_page_annotation",
            "model_repo": MAGI_V3_REPO,
            "sample_id": sample_id,
            "source": source,
            "image_size": {"width": int(image_obj.width), "height": int(image_obj.height)},
            "tasks": ["detections"],
            "detections": detections_out,
            "summary": {
                "panel_count": len(normalized_panels),
                "character_count": len(normalized_characters),
                "text_count": len(normalized_texts),
                "tail_count": len(normalized_tails),
                "character_cluster_count": len(set(cluster_labels)),
                "text_character_association_count": len(detections_out["text_character_associations"]),
                "text_tail_association_count": len(detections_out["text_tail_associations"]),
            },
            "run": {
                "run_id": run_id,
                "git_sha": git_sha,
                "annotated_at": datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
            },
        }
    )


# ---------------------------------------------------------------------------
# Model class — @app.cls with snapshotting for fast cold starts
# ---------------------------------------------------------------------------


# target_inputs=8 lets one H200 container hold 8 shards at once. The GPU phase
# of each shard is ~1-2s (MAGI v3 inference) but the Gemini verification phase
# is ~10s wall (16 parallel API calls per shard, tail-bound). Without input
# concurrency the H200 sits idle 80-90% of the time waiting on httpx reads;
# with target_inputs=8 the GPU runs shard A's MAGI pass while shards B-H block
# on Gemini, then rotates. PyTorch serializes CUDA kernels on the shared
# context so the model itself is safe — no extra synchronization needed.
# max_inputs=10 gives 2 slots of burst headroom during autoscaler ramps.
# Memory cost: 8 shards x 16 PIL images ≈ 4 GB, well under the 32 GB budget.
@app.cls(
    region=MODAL_REGION,
    gpu="H200",
    timeout=3600,
    startup_timeout=1500,
    cpu=8.0,
    memory=32768,
    secrets=worker_secrets,
    volumes={HF_HOME: hf_volume},
    max_containers=DEFAULT_MAX_CONTAINERS,
    scaledown_window=300,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class MagiAnnotator:
    @modal.enter(snap=True)
    def load_model_to_cpu(self) -> None:
        """Pre-snapshot: download Magi v3 weights and load to CPU."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MAGI_V3_REPO,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(MAGI_V3_REPO, trust_remote_code=True)

    @modal.enter(snap=False)
    def move_model_to_gpu(self) -> None:
        """Post-snapshot: move to GPU and warm CUDA kernels.

        Florence2's first decoder.generate call recompiles JIT kernels and
        takes 30-60s. Doing a dummy 768x768 forward pass here moves that
        cost into the container's startup_timeout budget so the first real
        inference call lands fully warm.
        """
        import torch
        from PIL import Image

        self.model = self.model.cuda()
        try:
            dummy = Image.new("RGB", (768, 768), (128, 128, 128))
            with torch.inference_mode():
                _ = self.model.predict_detections_and_associations([dummy], self.processor)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] skipped due to {exc!r}", flush=True)

    @modal.method()
    def annotate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Annotate one shard of page items.

        Each item is ``{page_key, output_key, sample_id, ...}`` or
        ``{output_key, sample_id, source: {"type": "page", "page_key": ...}}``.
        """
        return _annotate_batch_impl(payload, model=self.model, processor=self.processor)


def _verify_and_persist_page(
    *,
    image_obj: Any,
    source: dict[str, Any],
    sample_id: str,
    output_key: str,
    raw: Any,
    bucket: str,
    run_id: str,
    git_sha: str,
    failed_prefix: str,
    gemini_verifier_model: str,
    gemini_verifier_thinking_level: str,
) -> tuple[str, dict[str, Any]]:
    """Run Gemini verification on one page and write either the verified annotation
    or a failure record to S3. Designed to be called concurrently from a
    ThreadPoolExecutor: every dependency (boto3 + genai clients) is module-level
    and thread-safe, and exceptions are caught here so one bad page does not
    poison the rest of the shard.
    """
    try:
        annotation = _annotation_payload(
            sample_id=sample_id,
            image_obj=image_obj,
            source=source,
            raw_detections=raw or {},
            run_id=run_id,
            git_sha=git_sha,
        )
        annotation = _apply_gemini_character_verification(
            annotation=annotation,
            image_obj=image_obj,
            sample_id=sample_id,
            model=gemini_verifier_model,
            thinking_level=gemini_verifier_thinking_level,
        )
        _put_jsonl_object(bucket, output_key, annotation)
        return (
            "ok",
            {
                "sample_id": sample_id,
                "page_key": source.get("key"),
                "output_key": output_key,
                "status": "ok",
                "verification_status": annotation["verification"]["status"],
                "summary": annotation["summary"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        failure_key = f"{failed_prefix}/{run_id}/{sample_id}.json"
        try:
            _put_jsonl_object(
                bucket,
                failure_key,
                {
                    "sample_id": sample_id,
                    "page_key": source.get("key"),
                    "output_key": output_key,
                    "run_id": run_id,
                    "error": repr(exc)[:1000],
                },
            )
        except Exception:
            pass
        return (
            "error",
            {
                "sample_id": sample_id,
                "page_key": source.get("key"),
                "output_key": output_key,
                "failure_key": failure_key,
                "status": "error",
                "error": repr(exc)[:500],
            },
        )


def _annotate_batch_impl(
    payload: dict[str, Any],
    *,
    model: Any,
    processor: Any,
) -> dict[str, Any]:
    pages = list(payload.get("pages") or [])
    if not pages:
        raise ValueError("payload.pages is required and must be non-empty")

    bucket = str(payload.get("bucket") or S3_BUCKET_DEFAULT).strip()
    run_id = str(payload.get("run_id") or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = str(payload.get("git_sha") or "").strip()
    overwrite = bool(payload.get("overwrite", False))
    gpu_batch_size = max(1, int(payload.get("gpu_batch_size") or DEFAULT_GPU_BATCH_SIZE))
    failed_prefix = str(payload.get("failed_prefix") or "datasets/annotations/magi_v3/_failed").strip().strip("/")
    # Gemini character verification is always-on and mandatory. The only knobs
    # are the verifier model + thinking level.
    gemini_verifier_model = str(payload.get("gemini_verifier_model") or DEFAULT_GEMINI_VERIFIER_MODEL).strip()
    gemini_verifier_thinking_level = str(
        payload.get("gemini_verifier_thinking_level") or DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL
    ).strip()

    # Skip pages that already have an annotation when overwrite=False.
    todo: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_key = str(page.get("page_key") or "").strip()
        output_key = str(page.get("output_key") or "").strip()
        if not page_key or not output_key:
            continue
        if not overwrite and _head_exists(bucket, output_key):
            skipped.append({"page_key": page_key, "output_key": output_key, "status": "skipped_existing"})
            continue
        todo.append(page)

    if not todo:
        return {"ok": True, "annotated": [], "skipped": skipped, "errors": []}

    # Parallel S3 download — typical page is 100-400 KB, 32 threads keeps the
    # H200 fed and overlaps with the previous batch's inference under
    # @modal.concurrent(target_inputs=8, max_inputs=10) on MagiAnnotator.
    download_start = time.perf_counter()
    images: list[Any] = []
    sources: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    output_keys: list[str] = []

    def _download(page: dict[str, Any]) -> tuple[Any, dict[str, Any], str, str]:
        # Items can be either flat rows (``page_key``) or wrapped rows with a
        # ``source`` block ({"type": "page", "page_key": ...}).
        output_key = str(page["output_key"])
        source_block = page.get("source")
        if not isinstance(source_block, dict):
            page_key = str(page.get("page_key") or "")
            if not page_key:
                raise ValueError("item is missing both source and page_key")
            source_block = {"type": "page", "page_key": page_key}
        image_obj, source_meta = load_asset_image(source_block, bucket)
        chapter = str(page.get("chapter") or "")
        derived_id = Path(str(source_meta.get("key") or source_meta.get("page_key") or "")).stem
        page_id = str(page.get("page_id") or derived_id)
        sample_id = str(page.get("sample_id") or (f"{chapter}__{page_id}" if chapter else page_id))
        return image_obj, source_meta, sample_id, output_key

    with ThreadPoolExecutor(max_workers=min(32, len(todo))) as pool:
        for image_obj, source, sample_id, output_key in pool.map(_download, todo):
            images.append(image_obj)
            sources.append(source)
            sample_ids.append(sample_id)
            output_keys.append(output_key)
    download_sec = time.perf_counter() - download_start

    import torch

    inference_start = time.perf_counter()
    detections: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(images), gpu_batch_size):
            batch_images = images[start : start + gpu_batch_size]
            detections.extend(model.predict_detections_and_associations(batch_images, processor))
    torch.cuda.synchronize()
    inference_sec = time.perf_counter() - inference_start

    annotated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Fan out per-page Gemini verification + S3 write across a thread pool.
    # Each Gemini call blocks on multi-second LLM latency; the only shared
    # state (boto3 + genai clients, cached module-level) is thread-safe, and
    # _verify_and_persist_page handles its own try/except so a failure on one
    # page does not affect the others. Results are collected as they complete
    # so the slowest page does not block reporting.
    workers = min(GEMINI_VERIFIER_PARALLELISM, max(1, len(images)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _verify_and_persist_page,
                image_obj=img,
                source=src,
                sample_id=sid,
                output_key=okey,
                raw=raw,
                bucket=bucket,
                run_id=run_id,
                git_sha=git_sha,
                failed_prefix=failed_prefix,
                gemini_verifier_model=gemini_verifier_model,
                gemini_verifier_thinking_level=gemini_verifier_thinking_level,
            )
            for img, src, sid, okey, raw in zip(
                images, sources, sample_ids, output_keys, detections
            )
        ]
        for fut in as_completed(futures):
            status, entry = fut.result()
            if status == "ok":
                annotated.append(entry)
            else:
                errors.append(entry)

    return {
        "ok": not errors,
        "annotated": annotated,
        "skipped": skipped,
        "errors": errors,
        "stats": {
            "batch_size": len(images),
            "gpu_batch_size": gpu_batch_size,
            "download_sec": round(download_sec, 3),
            "inference_sec": round(inference_sec, 3),
            "pages_per_sec": round(len(images) / inference_sec, 3) if inference_sec > 0 else None,
        },
    }


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------


def load_asset_image(source: dict[str, Any], bucket: str) -> tuple[Any, dict[str, Any]]:
    """Single chokepoint for image acquisition during annotation.

    Returns ``(image, source_meta)``. ``source_meta`` is the dict written to
    ``annotation['source']`` — it captures provenance the trainer needs to
    reload the same bytes later.
    """
    page_key = str(source.get("page_key") or "").strip()
    if not page_key:
        raise ValueError("page source missing page_key")
    image_obj, meta = _download_rgb_image(bucket, page_key)
    meta["type"] = "page"
    meta["page_key"] = page_key
    return image_obj, meta


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.function(
    region=MODAL_REGION,
    gpu="H200",
    timeout=3600,
    startup_timeout=1500,
    cpu=8.0,
    memory=65536,
    secrets=worker_secrets,
    volumes={HF_HOME: hf_volume},
    max_containers=1,
)
def benchmark_batch_sizes(
    bucket: str = S3_BUCKET_DEFAULT,
    chapter: str = "jujutsu-kaisen",
    source_prefix: str = "datasets/pages/filtered",
    n_images: int = 128,
    batch_sizes: list[int] | None = None,
    warmup_batch_size: int = 1,
    candidate_repeats: int = 2,
) -> dict[str, Any]:
    """Run a batch-size sweep on one H200 to find the throughput peak.

    Loads ``n_images`` real pages from S3, warms up, then times each batch
    size in ``batch_sizes`` for ``candidate_repeats`` iterations each.
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    # Load model
    load_start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            MAGI_V3_REPO,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        .cuda()
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MAGI_V3_REPO, trust_remote_code=True)
    load_sec = time.perf_counter() - load_start

    # List + download images
    s3 = _s3_client_cached()
    prefix = f"{source_prefix.rstrip('/')}/{chapter}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if Path(key).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                keys.append(key)
                if len(keys) >= n_images:
                    break
        if len(keys) >= n_images:
            break
    if not keys:
        raise RuntimeError(f"no pages found at s3://{bucket}/{prefix}")

    download_start = time.perf_counter()
    images: list[Any] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for image_obj, _ in pool.map(lambda k: _download_rgb_image(bucket, k), keys):
            images.append(image_obj)
    download_sec = time.perf_counter() - download_start

    # Warmup so the first measured batch is not cold
    if warmup_batch_size > 0:
        with torch.inference_mode():
            _ = model.predict_detections_and_associations(
                images[: max(1, min(warmup_batch_size, len(images)))], processor
            )
        torch.cuda.synchronize()

    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size > len(images):
            results.append({"batch_size": batch_size, "skipped": "not enough images"})
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        candidate_images = images[:batch_size]
        encoded = 0
        oom = False
        error = ""
        start = time.perf_counter()
        try:
            for _ in range(max(1, int(candidate_repeats))):
                with torch.inference_mode():
                    _ = model.predict_detections_and_associations(candidate_images, processor)
                encoded += batch_size
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            oom = True
            error = str(exc)[:300]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                oom = True
                error = str(exc)[:300]
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                raise
        wall = time.perf_counter() - start
        peak_alloc_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        result = {
            "batch_size": batch_size,
            "oom": oom,
            "error": error,
            "images_completed": encoded,
            "wall_sec": round(wall, 3),
            "pages_per_sec": round(encoded / wall, 3) if wall > 0 else 0.0,
            "sec_per_image": round(wall / encoded, 4) if encoded > 0 else None,
            "peak_allocated_gb": round(peak_alloc_gb, 2),
        }
        print(f"[benchmark] {json.dumps(result, sort_keys=True)}", flush=True)
        results.append(result)

    viable = [r for r in results if not r.get("oom") and r.get("images_completed", 0) > 0]
    best = max(viable, key=lambda r: r["pages_per_sec"]) if viable else None
    return {
        "ok": best is not None,
        "model_repo": MAGI_V3_REPO,
        "chapter": chapter,
        "image_count": len(images),
        "load_sec": round(load_sec, 2),
        "download_sec": round(download_sec, 2),
        "best": best,
        "results": results,
    }


@app.local_entrypoint()
def run_benchmark(
    chapter: str = "jujutsu-kaisen",
    n_images: int = 128,
):
    result = benchmark_batch_sizes.remote(chapter=chapter, n_images=n_images)
    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def smoke_test(
    page_s3_uri: str = "",
    gemini_verifier_model: str = DEFAULT_GEMINI_VERIFIER_MODEL,
    gemini_verifier_thinking_level: str = DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL,
):
    """Annotate one page end-to-end as a deploy sanity check.

    Gemini character verification is always part of the pipeline; this entrypoint
    exposes only the verifier model/thinking-level so the smoke test mirrors a
    real production run.
    """
    if not page_s3_uri:
        page_s3_uri = f"s3://{S3_BUCKET_DEFAULT}/datasets/pages/filtered/jujutsu-kaisen/0001.jpg"
    bucket, page_key = _parse_s3_uri(page_s3_uri)
    chapter = page_key.split("/")[-2] if "/" in page_key else ""
    page_id = Path(page_key).stem
    output_key = f"datasets/annotations/magi_v3/_smoke/{chapter}/{page_id}.jsonl"
    payload = {
        "bucket": bucket,
        "run_id": "smoke",
        "overwrite": True,
        "gpu_batch_size": 1,
        "gemini_verifier_model": gemini_verifier_model,
        "gemini_verifier_thinking_level": gemini_verifier_thinking_level,
        "pages": [
            {
                "chapter": chapter,
                "page_id": page_id,
                "sample_id": f"{chapter}__{page_id}" if chapter else page_id,
                "page_key": page_key,
                "output_key": output_key,
            }
        ],
    }
    annotator = MagiAnnotator()
    result = annotator.annotate_batch.remote(payload)
    print(json.dumps(result, indent=2, default=str))


def _build_manifest_shards(
    manifest_path: str,
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    git_sha: str = "",
    overwrite: bool = False,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    pages_per_shard: int = 16,
    gemini_verifier_model: str = DEFAULT_GEMINI_VERIFIER_MODEL,
    gemini_verifier_thinking_level: str = DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL,
) -> tuple[str, list[dict[str, Any]], int, int]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    effective_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shard_size = max(1, int(pages_per_shard))
    if not rows:
        return effective_run_id, [], 0, shard_size

    shards = [
        {
            "bucket": bucket,
            "run_id": effective_run_id,
            "git_sha": git_sha,
            "overwrite": overwrite,
            "gpu_batch_size": gpu_batch_size,
            "gemini_verifier_model": gemini_verifier_model,
            "gemini_verifier_thinking_level": gemini_verifier_thinking_level,
            "pages": rows[start : start + shard_size],
        }
        for start in range(0, len(rows), shard_size)
    ]
    return effective_run_id, shards, len(rows), shard_size


@app.local_entrypoint()
def annotate_manifest_local(
    manifest_path: str,
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    git_sha: str = "",
    overwrite: bool = False,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    pages_per_shard: int = 16,
    gemini_verifier_model: str = DEFAULT_GEMINI_VERIFIER_MODEL,
    gemini_verifier_thinking_level: str = DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL,
):
    """Read a JSONL manifest of pages and fan them out across the H200 pool.

    Gemini character verification is a mandatory second stage. A Gemini failure
    on a page fails that page; it is recorded in the per-run ``_failed/`` prefix.
    """
    effective_run_id, shards, row_count, shard_size = _build_manifest_shards(
        manifest_path=manifest_path,
        bucket=bucket,
        run_id=run_id,
        git_sha=git_sha,
        overwrite=overwrite,
        gpu_batch_size=gpu_batch_size,
        pages_per_shard=pages_per_shard,
        gemini_verifier_model=gemini_verifier_model,
        gemini_verifier_thinking_level=gemini_verifier_thinking_level,
    )

    print(
        json.dumps(
            {
                "event": "start",
                "run_id": effective_run_id,
                "pages": row_count,
                "shards": len(shards),
                "shard_size": shard_size,
                "gpu_batch_size": gpu_batch_size,
                "gemini_verifier_model": gemini_verifier_model,
                "gemini_verifier_thinking_level": gemini_verifier_thinking_level,
                "max_containers": DEFAULT_MAX_CONTAINERS,
            }
        ),
        flush=True,
    )

    if not shards:
        print("manifest is empty; nothing to do")
        return

    annotated_total = 0
    skipped_total = 0
    error_total = 0
    wall_start = time.perf_counter()
    annotator = MagiAnnotator()
    for idx, result in enumerate(annotator.annotate_batch.map(shards, order_outputs=False)):
        annotated_total += len(result.get("annotated", []))
        skipped_total += len(result.get("skipped", []))
        error_total += len(result.get("errors", []))
        elapsed = time.perf_counter() - wall_start
        rate = annotated_total / elapsed if elapsed > 0 else 0.0
        print(
            json.dumps(
                {
                    "event": "shard_done",
                    "completed_shards": idx + 1,
                    "annotated_total": annotated_total,
                    "skipped_total": skipped_total,
                    "error_total": error_total,
                    "elapsed_sec": round(elapsed, 1),
                    "pages_per_sec_cluster": round(rate, 2),
                    "stats": result.get("stats"),
                }
            ),
            flush=True,
        )
    wall_sec = time.perf_counter() - wall_start
    print(
        json.dumps(
            {
                "event": "done",
                "run_id": effective_run_id,
                "annotated_total": annotated_total,
                "skipped_total": skipped_total,
                "error_total": error_total,
                "wall_sec": round(wall_sec, 1),
                "pages_per_sec_cluster": round(annotated_total / wall_sec, 2) if wall_sec > 0 else None,
            }
        ),
        flush=True,
    )


@app.local_entrypoint()
def annotate_manifest_spawn(
    manifest_path: str,
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    git_sha: str = "",
    overwrite: bool = False,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    pages_per_shard: int = 16,
    gemini_verifier_model: str = DEFAULT_GEMINI_VERIFIER_MODEL,
    gemini_verifier_thinking_level: str = DEFAULT_GEMINI_VERIFIER_THINKING_LEVEL,
):
    """Detach-safe variant: submit every shard with .spawn() and exit."""
    effective_run_id, shards, row_count, shard_size = _build_manifest_shards(
        manifest_path=manifest_path,
        bucket=bucket,
        run_id=run_id,
        git_sha=git_sha,
        overwrite=overwrite,
        gpu_batch_size=gpu_batch_size,
        pages_per_shard=pages_per_shard,
        gemini_verifier_model=gemini_verifier_model,
        gemini_verifier_thinking_level=gemini_verifier_thinking_level,
    )

    if not shards:
        print("manifest is empty; nothing to do")
        return

    print(
        json.dumps(
            {
                "event": "spawn_start",
                "run_id": effective_run_id,
                "pages": row_count,
                "shards": len(shards),
                "shard_size": shard_size,
                "gpu_batch_size": gpu_batch_size,
                "gemini_verifier_model": gemini_verifier_model,
                "gemini_verifier_thinking_level": gemini_verifier_thinking_level,
                "max_containers": DEFAULT_MAX_CONTAINERS,
            }
        ),
        flush=True,
    )

    annotator = MagiAnnotator()
    call_ids: list[str] = []
    for idx, shard in enumerate(shards):
        call = annotator.annotate_batch.spawn(shard)
        call_ids.append(call.object_id)
        if idx == 0 or (idx + 1) % 100 == 0 or idx == len(shards) - 1:
            print(
                json.dumps(
                    {
                        "event": "shard_spawned",
                        "shard_index": idx,
                        "call_id": call.object_id,
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "spawn_done",
                "run_id": effective_run_id,
                "spawned": len(call_ids),
                "first_call_id": call_ids[0] if call_ids else None,
                "last_call_id": call_ids[-1] if call_ids else None,
                "note": "shards run remotely; track via Modal dashboard. Local exit is safe.",
            }
        ),
        flush=True,
    )
