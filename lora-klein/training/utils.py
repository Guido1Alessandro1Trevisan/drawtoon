#!/usr/bin/env python3
"""Prepare an ai-toolkit DDP config for direct EC2 training.

This mirrors the small runtime rewrites that run_modal.py normally performs
before launching Modal, without importing Modal or changing the trainer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


S3_BUCKET = os.environ.get("DRAWTOON_S3_BUCKET") or os.environ.get("S3_BUCKET") or "drawtoon"
S3_MODELS_PREFIX = "models"
DEFAULT_OUTPUT_ROOT = "/tmp/lineart2_training_output"
DEFAULT_DATASET_CACHE_ROOT = "/mnt/local/training/datasets_cache"
DEFAULT_DRAWTOON_PAGES_PREFIX = "datasets/pages/filtered"
DEFAULT_DRAWTOON_ANNOTATIONS_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_DRAWTOON_CAPTIONS_PREFIX = "captions"
DEFAULT_DRAWTOON_CAPTION_RUN = "haiku45_mangazero_page_panel_v1"
MAX_CONTROL_IMAGE_SLOTS = 7
LAYOUT_TEXT_COLORS = {
    "Speech Bubble": (0, 96, 255),
    "Narration Bubble": (255, 128, 0),
    "Shout Bubble": (128, 0, 255),
}
LAYOUT_TEXT_COLOR = LAYOUT_TEXT_COLORS["Speech Bubble"]
LAYOUT_BACKGROUND_COLOR = (0, 0, 0)
LAYOUT_TAIL_COLOR = (192, 192, 192)
LAYOUT_CHARACTER_OUTLINE_WIDTHS = [12, 10, 8, 6, 4]
CHARACTER_REF_BORDER_WIDTH = 3
REVIEW_CHARACTER_COLORS = [
    ("Character 1", (255, 0, 0)),
    ("Character 2", (0, 180, 0)),
    ("Character 3", (220, 180, 0)),
    ("Character 4", (220, 0, 220)),
    ("Character 5", (0, 190, 190)),
]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket_and_key = uri[5:]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return bucket, key


def s3_uri(*parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    return f"s3://{S3_BUCKET}/{'/'.join(clean_parts)}" if clean_parts else f"s3://{S3_BUCKET}"


def s3_head(s3_client, uri: str) -> dict[str, Any]:
    bucket, key = parse_s3_uri(uri)
    return s3_client.head_object(Bucket=bucket, Key=key)


def download_s3_uri(s3_client, uri: str, destination: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = int(s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if destination.exists() and destination.stat().st_size == remote_size:
        return destination
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    s3_client.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(destination)
    return destination


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def iter_manifest_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def cache_path_for_s3_asset(asset_s3: str, root: Path) -> Path:
    bucket, key = parse_s3_uri(asset_s3)
    digest = hashlib.sha1(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:16]
    suffix = Path(key).suffix or ".bin"
    name = Path(key).stem[:80]
    return root / digest[:2] / f"{name}_{digest}{suffix}"


def cache_path_for_structured_asset(asset_ref: dict[str, Any], root: Path, suffix: str = ".png") -> Path:
    digest = hashlib.sha1(json.dumps(asset_ref, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return root / digest[:2] / f"{digest}{suffix}"


def pad_image_to_multiple(image: Image.Image, *, multiple: int) -> Image.Image:
    if multiple <= 1:
        return image
    padded_width = int(math.ceil(image.width / multiple) * multiple)
    padded_height = int(math.ceil(image.height / multiple) * multiple)
    if padded_width == image.width and padded_height == image.height:
        return image
    padded = Image.new(image.mode, (padded_width, padded_height), "white")
    padded.paste(image, (0, 0))
    return padded


def apply_ref_border_and_padding(
    image: Image.Image,
    *,
    border_width: int,
    border_rgb: Any,
    multiple: int,
) -> Image.Image:
    if border_width > 0 and isinstance(border_rgb, (list, tuple)) and len(border_rgb) == 3:
        if multiple > 1:
            inner_width = int(math.ceil((image.width + border_width * 2) / multiple) * multiple) - border_width * 2
            inner_height = int(math.ceil((image.height + border_width * 2) / multiple) * multiple) - border_width * 2
            if inner_width != image.width or inner_height != image.height:
                padded = Image.new("RGB", (max(image.width, inner_width), max(image.height, inner_height)), "white")
                padded.paste(image, (0, 0))
                image = padded
        return ImageOps.expand(image, border=border_width, fill=tuple(int(value) for value in border_rgb))
    return pad_image_to_multiple(image, multiple=multiple)


def coerce_crop_box(asset_ref: dict[str, Any]) -> tuple[int, int, int, int] | None:
    crop_box = asset_ref.get("crop_box") or asset_ref.get("bbox")
    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def materialize_asset(
    s3_client,
    asset_ref: Any,
    *,
    asset_root: Path,
    dataset_s3_path: str,
) -> Path:
    if isinstance(asset_ref, dict):
        image_ref = asset_ref.get("image") or asset_ref.get("path")
        if not image_ref:
            raise ValueError(f"Structured validation asset is missing image/path: {asset_ref}")
        output_path = cache_path_for_structured_asset(asset_ref, asset_root)
        if output_path.exists():
            return output_path
        source_path = materialize_asset(
            s3_client,
            image_ref,
            asset_root=asset_root,
            dataset_s3_path=dataset_s3_path,
        )
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            crop_box = coerce_crop_box(asset_ref)
            if crop_box is not None:
                image = image.crop(crop_box)
            pad_multiple = int(asset_ref.get("pad_multiple") or 0)
            border_width = int(asset_ref.get("border_width") or 0)
            image = apply_ref_border_and_padding(
                image,
                border_width=border_width,
                border_rgb=asset_ref.get("border_rgb"),
                multiple=pad_multiple,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG", optimize=True)
        return output_path

    raw = str(asset_ref).strip()
    if not raw:
        raise ValueError("Empty validation asset reference")
    if raw.startswith("s3://"):
        return download_s3_uri(s3_client, raw, cache_path_for_s3_asset(raw, asset_root))
    if os.path.isabs(raw):
        return Path(raw)
    if not dataset_s3_path:
        raise ValueError(f"Relative asset reference needs dataset_s3_path: {raw!r}")
    return materialize_asset(
        s3_client,
        dataset_s3_path.rstrip("/") + "/" + raw.lstrip("/"),
        asset_root=asset_root,
        dataset_s3_path=dataset_s3_path,
    )


def norm_box_to_pixel_box(
    box_norm: list[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(box_norm, (list, tuple)) or len(box_norm) != 4:
        return None
    x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
    y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
    x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_layout_blob(
    draw: ImageDraw.ImageDraw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    line_width: int = 6,
) -> bool:
    pixel_box = norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    draw.rectangle(pixel_box, outline=color, width=max(1, int(line_width)))
    return True


def draw_text_layout_region(
    draw: ImageDraw.ImageDraw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    text_bubble_type: str,
    line_width: int = 0,
) -> bool:
    normalized_type = " ".join(str(text_bubble_type or "").split()).strip().lower()
    if normalized_type in {"", "none", "null"}:
        return False
    text_color = LAYOUT_TEXT_COLORS.get(text_bubble_type)
    if text_color is None:
        return False
    pixel_box = norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    draw.rectangle(pixel_box, fill=text_color)
    return True


def draw_tail_layout_region(
    draw: ImageDraw.ImageDraw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int] = LAYOUT_TAIL_COLOR,
) -> bool:
    pixel_box = norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    draw.rectangle(pixel_box, fill=color)
    return True


def materialize_layout_control(
    layout_metadata: dict[str, Any],
    *,
    asset_root: Path,
    sample_id: str,
    width: int,
    height: int,
) -> Path:
    payload = {"sample_id": sample_id, "layout_control": layout_metadata, "width": width, "height": height}
    output_path = cache_path_for_structured_asset(payload, asset_root, suffix=".png")
    if output_path.exists():
        return output_path
    layout = Image.new("RGB", (width, height), LAYOUT_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(layout)
    for character in layout_metadata.get("characters") or []:
        if not isinstance(character, dict):
            continue
        box_norm = character.get("bbox_norm")
        rgb = character.get("rgb")
        if isinstance(box_norm, list) and isinstance(rgb, list):
            draw_layout_blob(
                draw,
                box_norm,
                width=width,
                height=height,
                color=tuple(int(value) for value in rgb),
                line_width=int(character.get("line_width") or 6),
            )
    text_payload = layout_metadata.get("text") if isinstance(layout_metadata.get("text"), dict) else {}
    for region in text_payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        draw_text_layout_region(
            draw,
            region.get("bbox_norm"),
            width=width,
            height=height,
            text_bubble_type=str(region.get("type") or "Speech Bubble"),
            line_width=int(region.get("line_width") or 0),
        )
    tail_payload = layout_metadata.get("tail") if isinstance(layout_metadata.get("tail"), dict) else {}
    for region in tail_payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        draw_tail_layout_region(
            draw,
            region.get("bbox_norm"),
            width=width,
            height=height,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout.save(output_path, format="PNG", optimize=True)
    return output_path


def build_static_validation_samples(
    s3_client,
    *,
    manifest_path: Path,
    dataset_s3_path: str,
    output_root: Path,
    sample_count: int,
    max_character_refs: int,
) -> list[dict[str, Any]]:
    asset_root = output_root / "_validation_assets"
    samples: list[dict[str, Any]] = []
    for row in iter_manifest_rows(manifest_path):
        if row.get("sample_type") not in {"character_ref_to_panel", "panel_prediction"}:
            continue
        target_path = materialize_asset(
            s3_client,
            row["target_panel"],
            asset_root=asset_root,
            dataset_s3_path=dataset_s3_path,
        )
        with Image.open(target_path) as image:
            width, height = image.size
        sample: dict[str, Any] = {
            "prompt": str(row["caption"]),
            "width": int(width),
            "height": int(height),
            "validation_sample_id": str(row.get("sample_id", "")),
            "validation_target_path": str(target_path),
            "validation_caption": str(row["caption"]),
        }
        control_paths: list[str] = []
        controls = row.get("controls", {}) or {}
        layout_control = controls.get("layout_control_path")
        if layout_control:
            local_layout = materialize_asset(
                s3_client,
                layout_control,
                asset_root=asset_root,
                dataset_s3_path=dataset_s3_path,
            )
            control_paths.append(str(local_layout))
        elif isinstance(controls.get("layout_control"), dict):
            local_layout = materialize_layout_control(
                controls["layout_control"],
                asset_root=asset_root,
                sample_id=str(row.get("sample_id") or len(samples)),
                width=int(width),
                height=int(height),
            )
            control_paths.append(str(local_layout))

        for ref_path in controls.get("character_ref_paths", [])[:max_character_refs]:
            if len(control_paths) >= MAX_CONTROL_IMAGE_SLOTS:
                break
            local_ref = materialize_asset(
                s3_client,
                ref_path,
                asset_root=asset_root,
                dataset_s3_path=dataset_s3_path,
            )
            control_paths.append(str(local_ref))
        for idx, control_path in enumerate(control_paths[:MAX_CONTROL_IMAGE_SLOTS], start=1):
            sample[f"ctrl_img_{idx}"] = control_path
        sample["validation_control_paths"] = control_paths[:MAX_CONTROL_IMAGE_SLOTS]
        samples.append(sample)
        if len(samples) >= sample_count:
            break
    if not samples:
        raise RuntimeError(f"No validation samples could be built from {manifest_path}")
    return samples


def compute_steps_per_epoch(row_count: int, *, batch_size: int, gradient_accumulation_steps: int, world_size: int) -> int:
    emitted_microbatches = max(1, math.ceil(row_count / max(1, batch_size)))
    per_rank_microbatches = max(1, math.ceil(emitted_microbatches / max(1, world_size)))
    return max(1, math.ceil(per_rank_microbatches / max(1, gradient_accumulation_steps)))


def prepare_ec2_ai_toolkit_config(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True, help="Local or s3:// manifest.jsonl prepared from Drawtoon canonical data.")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--target-epochs", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-samples", type=int, default=16)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_path = Path(args.output)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    manifest_source = args.manifest.strip()
    if manifest_source.startswith("s3://"):
        manifest_head = s3_head(s3_client, manifest_source)
        manifest_digest = hashlib.sha1(manifest_source.encode("utf-8")).hexdigest()[:16]
        local_manifest = output_root / "_resolved_manifests" / f"{manifest_digest}.jsonl"
        download_s3_uri(s3_client, manifest_source, local_manifest)
        manifest_size = int(manifest_head["ContentLength"])
    else:
        local_manifest = Path(manifest_source)
        if not local_manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {local_manifest}")
        manifest_size = local_manifest.stat().st_size

    row_count = count_jsonl_rows(local_manifest)

    parsed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    process_config = parsed_config.setdefault("config", {}).setdefault("process", [{}])[0]
    train_config = process_config.setdefault("train", {})
    datasets = process_config.setdefault("datasets", [])
    if not datasets:
        raise RuntimeError("Training config has no datasets")
    dataset = datasets[0]

    model_id = args.model_id.strip() or process_config.get("name") or parsed_config["config"].get("name")
    if not model_id:
        raise RuntimeError("Training config is missing a job name")
    parsed_config["config"]["name"] = model_id
    process_config["name"] = model_id
    parsed_config.setdefault("meta", {})["name"] = model_id

    process_config["training_folder"] = str(output_root)
    process_config["sample_copy_root"] = str(output_root / "validate")
    process_config["device"] = "cuda"

    dataset["manifest_path"] = str(local_manifest)
    dataset["_prepared_row_count"] = row_count
    dataset["_prepared_manifest_summary"] = {"source": manifest_source, "row_count": row_count}
    dataset.setdefault("dataset_s3_path", str(local_manifest.parent))
    dataset_s3_path = str(dataset.get("dataset_s3_path") or "").strip()
    max_character_refs = int(dataset.get("max_character_refs") or 6)

    batch_size = int(train_config.get("batch_size") or 1)
    gradient_accumulation_steps = int(train_config.get("gradient_accumulation_steps") or 1)
    steps_per_epoch = compute_steps_per_epoch(
        row_count,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=args.world_size,
    )
    total_steps = steps_per_epoch * max(1, args.target_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    train_config["steps"] = total_steps
    train_config["force_first_sample"] = False
    train_config["skip_first_sample"] = True

    save_config = process_config.setdefault("save", {})
    save_every = int(save_config.get("save_every", 0) or 0)
    if save_every <= 0:
        save_config["save_every"] = total_steps if args.max_train_steps > 0 else steps_per_epoch
    save_config.setdefault("max_step_saves_to_keep", 5)

    sample_config = process_config.setdefault("sample", {})
    sample_config.pop("prompts", None)
    validation_samples: list[dict[str, Any]] = []
    if args.validation_samples > 0:
        validation_samples = build_static_validation_samples(
            s3_client,
            manifest_path=local_manifest,
            dataset_s3_path=dataset_s3_path,
            output_root=output_root,
            sample_count=args.validation_samples,
            max_character_refs=max_character_refs,
        )
        sample_config["samples"] = validation_samples
        if args.world_size > 1:
            sample_config["distributed_sampling"] = True
            sample_config["distributed_sample_total"] = len(validation_samples)
            sample_config["distributed_samples_per_rank"] = max(
                1,
                math.ceil(len(validation_samples) / args.world_size),
            )
            sample_config["distributed_sample_strategy"] = "round_robin"
        sample_config["dynamic_validation_enabled"] = True
        sample_config["dynamic_validation_manifest_paths"] = [str(local_manifest)]
        sample_config["dynamic_validation_bucket_tolerance"] = int(dataset.get("bucket_tolerance") or 16)
        sample_config["dynamic_validation_character_panel_count"] = len(validation_samples)
        sample_config["dynamic_validation_character_panel_fixed_count"] = max(
            0,
            min(int(args.world_size), len(validation_samples)),
        )
    else:
        sample_config["sample_every"] = 0
        sample_config["samples"] = []
        sample_config["distributed_sampling"] = False
        sample_config["distributed_sample_total"] = 0
        sample_config["distributed_samples_per_rank"] = 0
        sample_config["distributed_sample_strategy"] = "none"
        sample_config["dynamic_validation_enabled"] = False
        sample_config["dynamic_validation_manifest_paths"] = []
        sample_config["dynamic_validation_bucket_tolerance"] = int(dataset.get("bucket_tolerance") or 16)
        sample_config["dynamic_validation_character_panel_count"] = 0
        sample_config["dynamic_validation_character_panel_fixed_count"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(parsed_config, sort_keys=False), encoding="utf-8")

    summary = {
        "config": str(output_path),
        "job_name": model_id,
        "manifest_source": manifest_source,
        "manifest_local": str(local_manifest),
        "manifest_size": manifest_size,
        "row_count": row_count,
        "steps_per_epoch": steps_per_epoch,
        "target_epochs": args.target_epochs,
        "total_steps": total_steps,
        "validation_samples": len(validation_samples),
        "dynamic_validation_enabled": bool(sample_config["dynamic_validation_enabled"]),
        "dynamic_validation_fixed_samples": int(sample_config["dynamic_validation_character_panel_fixed_count"]),
        "dynamic_validation_random_samples": int(
            len(validation_samples) - sample_config["dynamic_validation_character_panel_fixed_count"]
        ),
        "s3_model_prefix": s3_uri(S3_MODELS_PREFIX, model_id),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _install_noop_modal() -> None:
    """Let EC2 reuse run_modal.py helpers without creating Modal apps."""
    fake = types.ModuleType("modal")

    class _NoopVolume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

        def commit(self):
            return None

        def reload(self):
            return None

    class _NoopImage:
        @classmethod
        def from_dockerfile(cls, *args, **kwargs):
            return cls()

        @classmethod
        def debian_slim(cls, *args, **kwargs):
            return cls()

        def add_local_dir(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def apt_install(self, *args, **kwargs):
            return self

        def run_commands(self, *args, **kwargs):
            return self

        def env(self, *args, **kwargs):
            return self

    class _NoopSecret:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class _NoopCloudBucketMount:
        def __init__(self, *args, **kwargs):
            pass

    class _NoopApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def cls(self, *args, **kwargs):
            def decorator(cls):
                return cls

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    def _identity_decorator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    fake.Volume = _NoopVolume
    fake.Image = _NoopImage
    fake.Secret = _NoopSecret
    fake.CloudBucketMount = _NoopCloudBucketMount
    fake.App = _NoopApp
    fake.enter = _identity_decorator
    fake.method = _identity_decorator
    sys.modules["modal"] = fake


def _load_training_module_for_ec2(cache_root: str):
    _install_noop_modal()
    module_name = "_drawtoon_run_modal_ec2"
    module_path = Path(__file__).with_name("run_modal.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.DATASET_CACHE_ROOT = cache_root
    return module


def build_drawtoon_panel_cache(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the temporary Drawtoon panel cache directly on EC2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--caption-run", default=None)
    parser.add_argument("--pages-prefix", default=None)
    parser.add_argument("--annotations-prefix", default=None)
    parser.add_argument("--captions-prefix", default=None)
    parser.add_argument("--include-chapter-regex", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--caption-field", default=None)
    parser.add_argument("--caption-format", default=None, choices=["text", "compact_json"])
    parser.add_argument("--shard-count", type=int, default=0)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--cache-root", default=DEFAULT_DATASET_CACHE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    parsed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    drawtoon_config = parsed_config.get("drawtoon", {}) if isinstance(parsed_config, dict) else {}
    local_exclude_path = ""
    raw_exclude_path = ""
    if isinstance(drawtoon_config, dict):
        raw_exclude_path = str(drawtoon_config.get("exclude_sample_ids_path") or "").strip()
    if raw_exclude_path:
        exclude_path = Path(raw_exclude_path)
        try:
            exclude_exists = exclude_path.exists()
        except OSError:
            exclude_exists = False
        if not exclude_exists:
            local_candidate = config_path.parent / exclude_path.name
            if local_candidate.exists():
                local_exclude_path = str(local_candidate)

    module = _load_training_module_for_ec2(str(Path(args.cache_root)))
    overrides = {
        key: value
        for key, value in {
            "bucket": args.bucket,
            "caption_run": args.caption_run,
            "pages_prefix": args.pages_prefix,
            "annotations_prefix": args.annotations_prefix,
            "captions_prefix": args.captions_prefix,
            "caption_field": args.caption_field,
            "caption_format": args.caption_format,
            "include_chapter_regex": args.include_chapter_regex,
            "max_pages": args.max_pages,
        }.items()
        if value is not None
    }
    if local_exclude_path:
        overrides["exclude_sample_ids_path"] = local_exclude_path
    plan = module.plan_drawtoon_panel_cache(
        args.config,
        overrides,
        shard_count=args.shard_count,
        overwrite=args.overwrite,
    )
    if not plan.get("enabled") or plan.get("ready"):
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    print(
        "Building Drawtoon panel EC2 cache: "
        f"{plan['caption_key_count']} page captions across {plan['shard_count']} shards "
        f"with {max(1, args.workers)} workers",
        flush=True,
    )
    workers = max(1, min(int(args.workers), len(plan["shards"])))
    shard_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(module.build_drawtoon_panel_cache_shard, shard) for shard in plan["shards"]]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            shard_results.append(result)
            stats = result.get("stats") or {}
            print(
                f"cache shard {index}/{len(futures)} done: "
                f"rows={stats.get('panels', 0)} errors={stats.get('errors', 0)} "
                f"skipped={stats.get('skipped_panels', 0)}",
                flush=True,
            )

    final = module.finalize_drawtoon_panel_cache(args.config, plan)
    final["shard_results_observed"] = len(shard_results)
    print(json.dumps(final, indent=2, sort_keys=True))


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2]))


def _is_text_layout_pixel(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return blue >= 140 and red <= 90 and green <= 140


def _is_character_layout_pixel(rgb: tuple[int, int, int], target: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    if target == (255, 0, 0):
        return red >= 130 and red - max(green, blue) >= 55
    if target == (0, 180, 0):
        return green >= 110 and green - max(red, blue) >= 45
    if target == (220, 180, 0):
        return red >= 130 and green >= 110 and blue <= 80 and min(red, green) - blue >= 60
    if target == (220, 0, 220):
        return red >= 120 and blue >= 120 and green <= 100 and min(red, blue) - green >= 45
    if target == (0, 190, 190):
        return green >= 120 and blue >= 120 and red <= 100 and min(green, blue) - red >= 45
    return _color_distance(rgb, target) <= 80


def _is_colored_layout_pixel(rgb: tuple[int, int, int]) -> bool:
    if _is_text_layout_pixel(rgb) and not (rgb[0] <= 45 and rgb[1] <= 45 and rgb[2] <= 45):
        return True
    return any(_is_character_layout_pixel(rgb, color) for _, color in REVIEW_CHARACTER_COLORS)


def _looks_like_layout_control(image: Image.Image) -> bool:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    colored = 0
    step = max(1, (width * height) // 50000)
    sampled = 0
    for index in range(0, width * height, step):
        x = index % width
        y = index // width
        if y >= height:
            break
        sampled += 1
        if _is_colored_layout_pixel(pixels[x, y]):
            colored += 1
    return colored >= max(20, int(sampled * 0.002))


def _connected_component_boxes(
    image: Image.Image,
    predicate,
    *,
    min_pixels: int = 12,
    min_extent: int = 3,
) -> list[tuple[int, int, int, int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    visited = bytearray(width * height)
    boxes: list[tuple[int, int, int, int, int]] = []

    for y in range(height):
        row_offset = y * width
        for x in range(width):
            index = row_offset + x
            if visited[index] or not predicate(pixels[x, y]):
                continue

            stack = [(x, y)]
            visited[index] = 1
            x0 = x1 = x
            y0 = y1 = y
            count = 0

            while stack:
                cx, cy = stack.pop()
                count += 1
                if cx < x0:
                    x0 = cx
                elif cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                elif cy > y1:
                    y1 = cy

                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    nindex = ny * width + nx
                    if visited[nindex] or not predicate(pixels[nx, ny]):
                        continue
                    visited[nindex] = 1
                    stack.append((nx, ny))

            if count >= min_pixels and (x1 - x0 + 1) >= min_extent and (y1 - y0 + 1) >= min_extent:
                boxes.append((x0, y0, x1 + 1, y1 + 1, count))

    return boxes


def _detect_layout_boxes(layout_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with Image.open(layout_path) as layout:
        layout = layout.convert("RGB")
        if not _looks_like_layout_control(layout):
            return [], []
        characters: list[dict[str, Any]] = []
        for label, color in REVIEW_CHARACTER_COLORS:
            for x0, y0, x1, y1, area in _connected_component_boxes(
                layout,
                lambda rgb, target=color: _is_character_layout_pixel(rgb, target),
            ):
                characters.append({"label": label, "bbox": [x0, y0, x1, y1], "area": area, "color": color})

        text_regions = [
            {"label": f"Text {idx + 1}", "bbox": [x0, y0, x1, y1], "area": area, "color": (0, 90, 255)}
            for idx, (x0, y0, x1, y1, area) in enumerate(
                _connected_component_boxes(layout, _is_text_layout_pixel, min_pixels=12)
            )
        ]
    return characters, text_regions


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    *,
    label: str,
    color: tuple[int, int, int],
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    label_width = text_bbox[2] - text_bbox[0] + 6
    label_height = text_bbox[3] - text_bbox[1] + 4
    label_y0 = max(0, y0 - label_height)
    draw.rectangle((x0, label_y0, x0 + label_width, label_y0 + label_height), fill=color)
    draw.text((x0 + 3, label_y0 + 2), label, fill=(255, 255, 255), font=font)


def _overlay_generated_bboxes(generated_path: Path, layout_path: Path) -> Image.Image:
    with Image.open(generated_path) as generated:
        output = generated.convert("RGB")
    with Image.open(layout_path) as layout_image:
        layout_size = layout_image.size

    characters, text_regions = _detect_layout_boxes(layout_path)
    scale_x = output.width / max(1, layout_size[0])
    scale_y = output.height / max(1, layout_size[1])
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    line_width = max(2, min(output.size) // 160)

    for item in sorted(characters, key=lambda row: row["area"], reverse=True):
        x0, y0, x1, y1 = item["bbox"]
        _draw_labeled_box(
            draw,
            [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
            label=item["label"].replace("Character ", "C"),
            color=tuple(item["color"]),
            width=line_width,
            font=font,
        )

    for item in sorted(text_regions, key=lambda row: row["area"], reverse=True):
        x0, y0, x1, y1 = item["bbox"]
        _draw_labeled_box(
            draw,
            [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
            label=item["label"].replace("Text ", "T"),
            color=(0, 90, 255),
            width=line_width,
            font=font,
        )

    return output


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return fitted


def _labeled_panel(title: str, image: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    font = ImageFont.load_default()
    image = _fit_image(image, max_width=max_width, max_height=max_height)
    label_height = 22
    canvas = Image.new("RGB", (max(max_width, image.width), image.height + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, label_height), fill=(30, 30, 30))
    draw.text((6, 5), title, fill=(255, 255, 255), font=font)
    canvas.paste(image, ((canvas.width - image.width) // 2, label_height))
    return canvas


def _reference_panel(ref_paths: list[Path], *, max_width: int, max_height: int) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 22
    canvas = Image.new("RGB", (max_width, max_height + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, label_height), fill=(30, 30, 30))
    draw.text((6, 5), "character refs", fill=(255, 255, 255), font=font)
    if not ref_paths:
        draw.text((12, label_height + 12), "none", fill=(70, 70, 70), font=font)
        return canvas

    thumb_max = 132
    gap = 8
    x = gap
    y = label_height + gap
    row_height = 0
    for index, ref_path in enumerate(ref_paths, start=1):
        try:
            with Image.open(ref_path) as ref_image:
                thumb = _fit_image(ref_image, max_width=thumb_max, max_height=thumb_max)
        except Exception:
            continue
        if x + thumb.width + gap > max_width:
            x = gap
            y += row_height + gap + 14
            row_height = 0
        if y + thumb.height + 14 > canvas.height:
            break
        canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb.height + 1), f"ref {index}", fill=(20, 20, 20), font=font)
        x += thumb.width + gap
        row_height = max(row_height, thumb.height)
    return canvas


def _first_existing_control_image(sample_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = sample_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    matches = sorted(path for path in sample_dir.glob(f"{stem}.*") if path.is_file())
    return matches[0] if matches else None


def _compose_validation_review(
    *,
    generated_path: Path,
    sample_dir: Path,
    output_path: Path,
) -> bool:
    target_path = sample_dir / "target.jpg"
    layout_path = _first_existing_control_image(sample_dir, "ctrl_img_1")
    if not target_path.is_file() or not generated_path.is_file():
        return False

    ref_paths = sorted(
        path
        for path in sample_dir.glob("ctrl_img_*.*")
        if path.is_file() and path.stem != "ctrl_img_1"
    )
    with Image.open(target_path) as target_image:
        target_panel = _labeled_panel("target", target_image, max_width=520, max_height=720)

    condition_panel = None
    if layout_path is not None and layout_path.is_file():
        with Image.open(layout_path) as layout_image:
            condition_panel = _labeled_panel("conditioning ctrl_img_1", layout_image, max_width=360, max_height=720)

    refs_panel = _reference_panel(ref_paths, max_width=300, max_height=max(120, target_panel.height - 22))
    if layout_path is not None and layout_path.is_file():
        generated_image = _overlay_generated_bboxes(generated_path, layout_path)
    else:
        with Image.open(generated_path) as generated:
            generated_image = generated.convert("RGB")
    generated_panel = _labeled_panel("generated + boxes", generated_image, max_width=520, max_height=720)

    gap = 12
    panels = [target_panel]
    if condition_panel is not None:
        panels.append(condition_panel)
    panels.extend([refs_panel, generated_panel])
    width = sum(panel.width for panel in panels) + gap * (len(panels) + 1)
    height = max(panel.height for panel in panels) + gap * 2
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    x = gap
    for panel in panels:
        sheet.paste(panel, (x, gap))
        x += panel.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def build_validation_review_sheets(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build local validation review images from synced ai-toolkit validation folders."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--step", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args(argv)

    job_dir = Path(args.job_dir)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Validation job directory does not exist: {job_dir}")

    if args.step:
        step_dirs = [job_dir / args.step]
    else:
        step_dirs = sorted(path for path in job_dir.glob("step_*") if path.is_dir())

    built = 0
    skipped = 0
    missing = 0
    for step_dir in step_dirs:
        if not step_dir.is_dir():
            continue
        review_dir = step_dir / "_review_sheets" / "per_sample"
        generated_paths = sorted(
            path
            for path in step_dir.glob("*.jpg")
            if path.is_file() and not path.name.startswith("prompt_")
        )
        for generated_path in generated_paths:
            if args.max_samples > 0 and built >= args.max_samples:
                break
            sample_dir = step_dir / generated_path.stem
            output_path = review_dir / f"{generated_path.stem}__target_refs_generated_bboxes.jpg"
            if output_path.exists() and not args.force:
                skipped += 1
                continue
            if not sample_dir.is_dir():
                missing += 1
                continue
            if _compose_validation_review(
                generated_path=generated_path,
                sample_dir=sample_dir,
                output_path=output_path,
            ):
                built += 1
            else:
                missing += 1

    print(
        json.dumps(
            {
                "job_dir": str(job_dir),
                "steps": len(step_dirs),
                "built": built,
                "skipped_existing": skipped,
                "missing_inputs": missing,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _define_checkpoint_sync_modal_app():
    import modal

    volume_mount = Path("/mnt/models")
    app = modal.App("lineart2-sync-existing-checkpoints-to-s3")
    model_volume = modal.Volume.from_name("flux-lora-models")
    aws_secret = modal.Secret.from_name("lineart2-aws-s3")
    image = modal.Image.debian_slim().pip_install("boto3")

    @app.function(
        image=image,
        volumes={str(volume_mount): model_volume},
        secrets=[aws_secret],
        timeout=6 * 60 * 60,
        cpu=4,
        memory=8192,
    )
    def sync_existing_checkpoints(job_names: list[str]) -> dict:
        import boto3
        from boto3.s3.transfer import TransferConfig

        client = boto3.client("s3")
        transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=128 * 1024 * 1024,
            max_concurrency=16,
            use_threads=True,
        )
        uploaded = []
        missing = []

        def upload_file(path: Path, key: str) -> None:
            print(f"upload {path} -> s3://{S3_BUCKET}/{key}", flush=True)
            client.upload_file(str(path), S3_BUCKET, key, Config=transfer_config)
            uploaded.append(
                {
                    "path": str(path),
                    "s3": f"s3://{S3_BUCKET}/{key}",
                    "bytes": path.stat().st_size,
                }
            )

        for job_name in job_names:
            save_root = volume_mount / job_name
            if not save_root.exists():
                missing.append({"job": job_name, "reason": "missing volume directory"})
                continue

            model_prefix = f"models/{job_name}"
            for name in ("config.yaml", "optimizer.pt"):
                path = save_root / name
                if path.is_file():
                    upload_file(path, f"{model_prefix}/{name}")

            checkpoint_paths = [path for path in sorted(save_root.glob(f"{job_name}*")) if path.is_file()]
            if not checkpoint_paths:
                missing.append({"job": job_name, "reason": "no checkpoint files"})
                continue

            for path in checkpoint_paths:
                upload_file(path, f"{model_prefix}/checkpoints/{path.name}")

        return {"uploaded": uploaded, "missing": missing}

    @app.local_entrypoint()
    def sync_existing_checkpoints_to_s3(jobs: str):
        job_names = [part.strip() for part in jobs.split(",") if part.strip()]
        if not job_names:
            raise ValueError("Pass --jobs with one job name or comma-separated job names")
        result = sync_existing_checkpoints.remote(job_names)
        print(result)

    return app, sync_existing_checkpoints


try:
    app, sync_existing_checkpoints = _define_checkpoint_sync_modal_app()
except Exception:
    app = None
    sync_existing_checkpoints = None


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in {"build-drawtoon-panel-cache", "build-ec2-cache"}:
        build_drawtoon_panel_cache(sys.argv[2:])
        return
    if command in {"build-validation-review-sheets", "build-validation-review"}:
        build_validation_review_sheets(sys.argv[2:])
        return
    if command in {"prepare-ec2-ai-toolkit-config", "prepare-ec2-config"}:
        prepare_ec2_ai_toolkit_config(sys.argv[2:])
        return
    if command and not command.startswith("-"):
        raise SystemExit(f"Unknown utils command: {command}")
    prepare_ec2_ai_toolkit_config(sys.argv[1:])


if __name__ == "__main__":
    main()
