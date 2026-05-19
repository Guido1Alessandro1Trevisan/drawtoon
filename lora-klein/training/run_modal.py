"""
AI-Toolkit FLUX Training on Modal - Best Practices Implementation
Clean workflow with external config - no fallbacks
"""

import json
import io
import os
import random
import re
import shutil
import subprocess
import hashlib
import math
import sys
import time
import threading
from types import MethodType
from pathlib import Path
from typing import Any, Literal

import boto3
import modal
import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from PIL import Image, ImageDraw, ImageOps

# Define volumes for persistent storage
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
dataset_cache_volume = modal.Volume.from_name("flux-dataset-cache", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("flux-hf-cache", create_if_missing=True)
MOUNT_DIR = "/root/ai-toolkit/modal_output"
DDP_OUTPUT_ROOT = "/tmp/lineart2_training_output"
DATASET_CACHE_ROOT = "/root/training/datasets_cache"
HF_HOME = "/root/.cache/huggingface"
TRAINING_DIR = Path(__file__).parent.resolve()
REPO_ROOT = TRAINING_DIR.parent
LOCAL_DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.modal"
S3_BUCKET = os.environ.get("DRAWTOON_S3_BUCKET") or os.environ.get("S3_BUCKET") or "drawtoon"
S3_MOUNT_ROOT = "/mnt/datasets"
S3_MODELS_PREFIX = "models"
REMOTE_CONFIGS_DIR = "/root/training/configs"
DDP_WORLD_SIZE_8 = 8
DEFAULT_CONFIG_PATH = f"{REMOTE_CONFIGS_DIR}/mngrm12_klein4b_dataset_jpg_r16_768x1024_s12000_v1.yaml"
MODAL_SINGLE_GPU = os.environ.get("LINEART2_MODAL_SINGLE_GPU", "H200")
MODAL_DDP8_GPU = os.environ.get("LINEART2_MODAL_DDP8_GPU", "B200:8")
AWS_SECRET_NAME = os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3")
DEFAULT_DRAWTOON_PAGES_PREFIX = "datasets/pages/filtered"
DEFAULT_DRAWTOON_ANNOTATIONS_PREFIX = "datasets/annotations/magi_v3"
DEFAULT_DRAWTOON_CAPTIONS_PREFIX = "captions"
DEFAULT_DRAWTOON_CAPTION_RUN = "haiku45_mangazero_page_panel_v1"
MAX_CONTROL_IMAGE_SLOTS = 7
MAX_LAYOUT_CHARACTER_REFS = 5
MAX_LAYOUT_TEXT_REGIONS = 7
LAYOUT_TEXT_BUBBLE_TYPES = ("Speech Bubble", "Narration Bubble", "Shout Bubble", "None")
LAYOUT_TEXT_COLORS = {
    "Speech Bubble": (0, 96, 255),
    "Narration Bubble": (255, 128, 0),
    "Shout Bubble": (128, 0, 255),
}
LAYOUT_TEXT_COLOR = LAYOUT_TEXT_COLORS["Speech Bubble"]
LAYOUT_TAIL_COLOR = (192, 192, 192)
MAX_LAYOUT_TAIL_REGIONS = 14
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


class LaunchArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    config_path: str = DEFAULT_CONFIG_PATH
    dataset_subpath: str = ""
    model_id: str = ""
    preset: str = ""
    target_epochs: int = Field(default=3, ge=1)
    ddp_world_size: int = Field(default=1)
    max_train_steps: int = Field(default=0, ge=0)
    sample_every: int = Field(default=0, ge=0)

    @field_validator("preset", mode="before")
    @classmethod
    def normalize_preset(cls, value: object) -> str:
        if value is None:
            return ""
        return "".join(part.strip() for part in str(value).splitlines()).strip()

    @model_validator(mode="after")
    def validate_launch_args(self) -> "LaunchArgs":
        if self.preset and self.config_path != DEFAULT_CONFIG_PATH:
            raise ValueError("Use either --preset or --config-path, not both.")
        if self.ddp_world_size not in {1, DDP_WORLD_SIZE_8}:
            raise ValueError(
                f"Unsupported ddp_world_size={self.ddp_world_size}. "
                f"Supported values are 1 and {DDP_WORLD_SIZE_8}."
            )
        return self

    def resolved_config_path(self) -> str:
        return resolve_training_config_path(self.config_path, self.preset or None)


def find_raw_lora_path(job_name: str) -> Path:
    candidates = (
        Path(os.environ.get("LINEART2_TRAINING_OUTPUT_ROOT", DDP_OUTPUT_ROOT)) / job_name / f"{job_name}.safetensors",
        Path(MOUNT_DIR) / job_name / f"{job_name}.safetensors",
        Path(MOUNT_DIR) / f"{job_name}.safetensors",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find trained LoRA weights for {job_name}")


def resolve_training_config_path(
    config_path: str | None,
    preset: str | None = None,
) -> str:
    raw_config_path = (config_path or "").strip()
    if raw_config_path and raw_config_path != DEFAULT_CONFIG_PATH:
        if raw_config_path.startswith(REMOTE_CONFIGS_DIR):
            return raw_config_path

        local_candidate = Path(raw_config_path)
        if not local_candidate.exists():
            alt_candidate = TRAINING_DIR / "configs" / raw_config_path
            if alt_candidate.exists():
                local_candidate = alt_candidate

        if local_candidate.exists():
            try:
                relative_to_configs = local_candidate.resolve().relative_to((TRAINING_DIR / "configs").resolve())
            except ValueError:
                return str(local_candidate)
            return f"{REMOTE_CONFIGS_DIR}/{relative_to_configs.as_posix()}"

        return raw_config_path

    normalized_preset = "".join(part.strip() for part in str(preset or "").splitlines()).strip()
    if normalized_preset:
        config_root = (TRAINING_DIR / "configs").resolve()
        base_candidate = config_root / normalized_preset
        candidates = []
        if base_candidate.suffix.lower() in {".yaml", ".yml"}:
            candidates.append(base_candidate)
        else:
            candidates.append(base_candidate.with_suffix(".yaml"))
            candidates.append(base_candidate.with_suffix(".yml"))

        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            raise FileNotFoundError(
                f"Could not infer a training config for preset '{normalized_preset}'. "
                f"Expected a file under {config_root}."
            )
        relative_to_configs = match.resolve().relative_to(config_root)
        return f"{REMOTE_CONFIGS_DIR}/{relative_to_configs.as_posix()}"

    return DEFAULT_CONFIG_PATH


def resolve_local_training_config_path(config_path: str, preset: str | None = None) -> Path | None:
    normalized_preset = "".join(part.strip() for part in str(preset or "").splitlines()).strip()
    config_root = (TRAINING_DIR / "configs").resolve()
    if normalized_preset:
        base_candidate = config_root / normalized_preset
        candidates = (
            [base_candidate]
            if base_candidate.suffix.lower() in {".yaml", ".yml"}
            else [base_candidate.with_suffix(".yaml"), base_candidate.with_suffix(".yml")]
        )
        return next((candidate for candidate in candidates if candidate.exists()), None)

    raw = str(config_path or "").strip()
    if raw.startswith(REMOTE_CONFIGS_DIR + "/"):
        candidate = config_root / raw.removeprefix(REMOTE_CONFIGS_DIR + "/")
        return candidate if candidate.exists() else None
    if raw and raw != DEFAULT_CONFIG_PATH:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        candidate = config_root / raw
        return candidate if candidate.exists() else None
    return None


def sanitize_filename(value: str, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value or "").strip())
    normalized = normalized.strip("_")
    return normalized[:180] or fallback


def build_peft_adapter(
    source_path: Path,
    output_dir: Path,
    base_model: str,
    rank: int | None,
    alpha: int | None,
) -> Path:
    from safetensors.torch import load_file, save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    source_state_dict = load_file(str(source_path))
    converted_state_dict = {}

    for key, value in source_state_dict.items():
        new_key = key.replace("diffusion_model.", "transformer.")
        new_key = new_key.replace(".double_blocks.", ".transformer_blocks.")
        new_key = new_key.replace(".single_blocks.", ".single_transformer_blocks.")
        new_key = new_key.replace(".img_attn.qkv.", ".attn.to_qkv.")
        new_key = new_key.replace(".img_attn.proj.", ".attn.to_out.")
        new_key = new_key.replace(".txt_attn.qkv.", ".attn.add_kv_proj.")
        new_key = new_key.replace(".txt_attn.proj.", ".attn.to_add_out.")
        new_key = new_key.replace(".img_mlp.0.", ".ff.linear_in.")
        new_key = new_key.replace(".img_mlp.2.", ".ff.linear_out.")
        new_key = new_key.replace(".txt_mlp.0.", ".ff_context.linear_in.")
        new_key = new_key.replace(".txt_mlp.2.", ".ff_context.linear_out.")
        new_key = new_key.replace(".linear1.", ".to_qkv_mlp_proj.")
        new_key = new_key.replace(".linear2.", ".to_out.")
        converted_state_dict[new_key] = value

    save_file(converted_state_dict, str(output_dir / "adapter_model.safetensors"))

    adapter_config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha or rank or 16,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank or 16,
        "revision": None,
        "target_modules": [
            "to_qkv",
            "add_kv_proj",
            "to_out",
            "to_add_out",
            "linear_in",
            "linear_out",
            "to_qkv_mlp_proj",
        ],
        "task_type": None,
    }

    (output_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    return output_dir


def validate_peft_adapter_dir(path: Path) -> None:
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
    )
    missing = [str(file_path) for file_path in required if not file_path.exists()]
    if missing:
        raise RuntimeError(f"Missing PEFT adapter files: {missing}")


def dataset_sample_prompts(dataset_dir: str, limit: int = 20) -> list[str]:
    captions = []
    for path in sorted(Path(dataset_dir).rglob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            captions.append(text)
    if not captions:
        raise RuntimeError(f"No caption files found in dataset: {dataset_dir}")
    if len(captions) <= limit:
        return captions
    return random.sample(captions, limit)


def count_training_pairs(dataset_dir: Path) -> tuple[int, int, int]:
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = {
        path.relative_to(dataset_dir).with_suffix("")
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    }
    captions = {
        path.relative_to(dataset_dir).with_suffix("")
        for path in dataset_dir.rglob("*.txt")
        if path.is_file()
    }
    matched = images & captions
    return len(images), len(captions), len(matched)


def is_manifest_training(process_config: dict) -> bool:
    datasets = process_config.get("datasets", [])
    return any(dataset.get("type") == "manifest" for dataset in datasets)


def s3_uri_to_mount_path(s3_path: str) -> Path:
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Expected an s3:// path, got {s3_path}")
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    if bucket != S3_BUCKET:
        raise ValueError(f"Expected bucket {S3_BUCKET}, got {bucket}")
    return Path(S3_MOUNT_ROOT) / key


def load_manifest_entries(manifest_path: Path) -> list[dict]:
    entries: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def resolve_manifest_input_path(path_or_s3: str) -> Path:
    raw = str(path_or_s3 or "").strip()
    if not raw:
        raise ValueError("Expected a manifest path")
    if raw.startswith("s3://"):
        return s3_uri_to_mount_path(raw)
    return Path(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class CharacterPanelControls(BaseModel):
    model_config = ConfigDict(extra="ignore")

    character_ref_paths: list[Any] = Field(default_factory=list)
    has_previous_control: bool = False


class CharacterPanelManifestEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int | None = None
    sample_id: str | None = None
    target_panel: Any
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    bucket_key: str | None = None
    target_geometry: dict[str, Any] | None = None
    caption: str = Field(min_length=1)
    sample_type: Literal["character_ref_to_panel", "panel_prediction"]
    page_root: str = Field(min_length=1)
    panel_index: int = Field(ge=0)
    source_page: str | None = None
    character_count: int = Field(ge=0)
    controls: CharacterPanelControls


CharacterPanelManifestEntryList = TypeAdapter(list[CharacterPanelManifestEntry])


def materialize_manifest_dataset_view(
    manifest_dataset: dict,
    *,
    job_name: str,
    dataset_index: int,
) -> tuple[Path, list[dict], dict[str, object]]:
    layout = str(manifest_dataset.get("layout") or "manifest").strip().lower()
    if layout not in {"manifest", ""}:
        raise ValueError(
            f"Unsupported manifest layout={layout!r}; Drawtoon training expects a local manifest_path "
            "prepared from canonical pages, annotations, and captions."
        )
    base_manifest_source = str(manifest_dataset["manifest_path"]).strip()
    base_manifest_path = resolve_manifest_input_path(base_manifest_source)
    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Base manifest path does not exist: {base_manifest_path}")

    base_entries = load_manifest_entries(base_manifest_path)
    return base_manifest_path, base_entries, {
        "base_manifest_source": base_manifest_source,
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "resolved_manifest_path": str(base_manifest_path),
        "resolved_manifest_sha256": sha256_file(base_manifest_path),
        "resolved_row_count": len(base_entries),
        "materialized_overlay": False,
    }


def _manifest_entry_identity(entry: dict) -> str:
    sample_id = str(entry.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    return json.dumps(entry.get("target_panel") or entry, sort_keys=True, ensure_ascii=False)


def _stable_sample(entries: list[dict], count: int, seed_key: str) -> list[dict]:
    if count <= 0:
        return []
    decorated: list[tuple[str, dict]] = []
    for entry in entries:
        entry_key = _manifest_entry_identity(entry)
        stable_key = hashlib.sha1(f"{seed_key}:{entry_key}".encode("utf-8")).hexdigest()
        decorated.append((stable_key, entry))
    decorated.sort(key=lambda item: item[0])
    return [entry for _, entry in decorated[:count]]


def _manifest_sample_types(entries: list[dict]) -> set[str]:
    return {str(entry.get("sample_type", "")).strip() for entry in entries if entry.get("sample_type")}


def _resolve_manifest_validation_asset_path(manifest_path: Path, asset_ref: Any) -> Path:
    raw_value = str(asset_ref).strip()
    if not raw_value:
        raise ValueError("Validation asset reference must be non-empty")
    if raw_value.startswith("s3://"):
        return s3_uri_to_mount_path(raw_value)
    if os.path.isabs(raw_value):
        return Path(raw_value)

    normalized = raw_value.lstrip("/")
    mounted_bucket_path = Path(S3_MOUNT_ROOT) / normalized
    if mounted_bucket_path.exists():
        return mounted_bucket_path

    return manifest_path.parent / normalized


def _manifest_validation_asset_cache_path(asset_ref: Any, suffix: str) -> Path:
    digest = hashlib.sha1(json.dumps(asset_ref, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return Path("/tmp/drawtoon_manifest_validation_assets") / digest[:2] / f"{digest}{suffix}"


def _coerce_manifest_crop_box(asset_ref: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(asset_ref, dict):
        return None
    crop_box = asset_ref.get("crop_box") or asset_ref.get("bbox")
    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def _materialize_manifest_validation_asset(manifest_path: Path, asset_ref: Any) -> Path:
    if not isinstance(asset_ref, dict):
        return _resolve_manifest_validation_asset_path(manifest_path, asset_ref)
    image_ref = asset_ref.get("image") or asset_ref.get("path")
    if not image_ref:
        raise ValueError(f"Lazy manifest validation asset is missing image/path: {asset_ref}")
    output_path = _manifest_validation_asset_cache_path(asset_ref, ".jpg")
    if output_path.exists():
        return output_path
    source_path = _resolve_manifest_validation_asset_path(manifest_path, image_ref)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        crop_box = _coerce_manifest_crop_box(asset_ref)
        if crop_box is not None:
            image = image.crop(crop_box)
        pad_multiple = int(asset_ref.get("pad_multiple") or 0)
        border_width = int(asset_ref.get("border_width") or 0)
        image = _apply_ref_border_and_padding(
            image,
            border_width=border_width,
            border_rgb=asset_ref.get("border_rgb"),
            multiple=pad_multiple,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95, optimize=True)
    return output_path


def _layout_control_image_from_metadata(layout_metadata: dict[str, Any], width: int, height: int) -> Image.Image:
    layout = Image.new("RGB", (width, height), LAYOUT_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(layout)
    for character in layout_metadata.get("characters") or []:
        if not isinstance(character, dict):
            continue
        box_norm = character.get("bbox_norm")
        rgb = character.get("rgb")
        if isinstance(box_norm, list) and isinstance(rgb, list):
            _draw_layout_blob(
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
        _draw_text_layout_region(
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
        _draw_tail_layout_region(
            draw,
            region.get("bbox_norm"),
            width=width,
            height=height,
        )
    return layout


def _materialize_manifest_validation_layout(manifest_path: Path, entry: dict[str, Any]) -> Path | None:
    controls = entry.get("controls") if isinstance(entry.get("controls"), dict) else {}
    layout_metadata = controls.get("layout_control")
    if not isinstance(layout_metadata, dict):
        return None
    width = int(entry.get("target_width") or 0)
    height = int(entry.get("target_height") or 0)
    if width <= 0 or height <= 0:
        return None
    output_path = _manifest_validation_asset_cache_path(
        {
            "manifest": str(manifest_path),
            "sample_id": entry.get("sample_id"),
            "layout_control": layout_metadata,
            "width": width,
            "height": height,
        },
        ".png",
    )
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _layout_control_image_from_metadata(layout_metadata, width, height).save(output_path, format="PNG", optimize=True)
    return output_path


def _manifest_validation_control_paths_for_entry(
    manifest_path: Path,
    entry: dict[str, Any],
    *,
    include_previous: bool = True,
    max_controls: int = MAX_CONTROL_IMAGE_SLOTS,
) -> list[Path]:
    controls = entry.get("controls") if isinstance(entry.get("controls"), dict) else {}
    control_paths: list[Path] = []

    layout_control = controls.get("layout_control_path")
    if layout_control:
        control_paths.append(_materialize_manifest_validation_asset(manifest_path, layout_control))
    else:
        layout_path = _materialize_manifest_validation_layout(manifest_path, entry)
        if layout_path is not None:
            control_paths.append(layout_path)

    previous_control = controls.get("previous_panel") or controls.get("previous_page")
    if include_previous and previous_control and len(control_paths) < max_controls:
        control_paths.append(_materialize_manifest_validation_asset(manifest_path, previous_control))

    for ref_path in controls.get("character_ref_paths", []):
        if len(control_paths) >= max_controls:
            break
        control_paths.append(_materialize_manifest_validation_asset(manifest_path, ref_path))

    return control_paths[:max_controls]


def _build_manifest_validation_sample_dicts(
    manifest_path: Path,
    description_only_count: int = 5,
    next_page_first_panel_count: int = 5,
    within_page_next_panel_count: int = 10,
    bucket_tolerance: int = 16,
) -> list[dict]:
    entries = load_manifest_entries(manifest_path)
    if not entries:
        raise RuntimeError(f"No manifest entries found in {manifest_path}")

    grouped_entries: dict[str, list[dict]] = {
        "description_only": [],
        "next_page_first_panel": [],
        "within_page_next_panel": [],
    }
    for entry in entries:
        sample_type = entry.get("sample_type")
        if sample_type in grouped_entries:
            grouped_entries[sample_type].append(entry)

    total_counts = {
        "description_only": description_only_count,
        "next_page_first_panel": next_page_first_panel_count,
        "within_page_next_panel": within_page_next_panel_count,
    }

    fixed_counts = {
        "description_only": 2,
        "next_page_first_panel": 3,
        "within_page_next_panel": 5,
    }

    fixed_selected: list[dict] = []
    random_selected: list[dict] = []
    for sample_type, total_count in total_counts.items():
        candidates = grouped_entries[sample_type]
        if not candidates or total_count <= 0:
            continue

        effective_total = min(total_count, len(candidates))
        if effective_total < total_count:
            print(
                f"[validation] {sample_type}: requested {total_count}, found {len(candidates)}; using {effective_total}"
            )

        fixed_count = min(fixed_counts.get(sample_type, 0), effective_total)
        fixed_selection = _stable_sample(
            candidates,
            fixed_count,
            f"{manifest_path}:{sample_type}:fixed",
        )
        fixed_keys = {_manifest_entry_identity(entry) for entry in fixed_selection}
        random_pool = [
            entry for entry in candidates if _manifest_entry_identity(entry) not in fixed_keys
        ]
        random_count = effective_total - len(fixed_selection)
        random_selection = random.sample(random_pool, random_count) if random_count > 0 else []
        fixed_selected.extend(fixed_selection)
        random_selected.extend(random_selection)

    selected = fixed_selected + random_selected

    if not selected:
        raise RuntimeError(
            f"No validation samples could be built from {manifest_path}; available sample types: "
            f"{sorted(k for k, v in grouped_entries.items() if v)}"
        )

    samples: list[dict] = []
    for entry in selected:
        target_path = _materialize_manifest_validation_asset(
            manifest_path,
            entry["target_panel"],
        )
        sample_width, sample_height = _infer_validation_dimensions(
            target_path=target_path,
            bucket_tolerance=bucket_tolerance,
        )
        sample = {
            "prompt": entry["caption"],
            "width": sample_width,
            "height": sample_height,
        }
        control_paths = _manifest_validation_control_paths_for_entry(manifest_path, entry)

        for idx, path in enumerate(control_paths, start=1):
            sample[f"ctrl_img_{idx}"] = str(path)
        sample["validation_sample_id"] = str(entry.get("sample_id", ""))
        sample["validation_target_path"] = str(target_path)
        sample["validation_caption"] = str(entry["caption"])
        sample["validation_control_paths"] = [str(path) for path in control_paths]

        samples.append(sample)

    return samples


def _build_character_panel_validation_sample_dicts(
    manifest_path: Path,
    sample_count: int = 20,
    fixed_count: int | None = None,
    bucket_tolerance: int = 16,
) -> list[dict]:
    entries = load_manifest_entries(manifest_path)
    if not entries:
        raise RuntimeError(f"No manifest entries found in {manifest_path}")

    candidates = [
        entry
        for entry in entries
        if entry.get("sample_type") in {"character_ref_to_panel", "panel_prediction"}
    ]
    if not candidates:
        raise RuntimeError(
            f"No character panel validation candidates found in {manifest_path}"
        )

    effective_total = min(sample_count, len(candidates))
    if effective_total < sample_count:
        print(
            f"[validation] character_ref_to_panel: requested {sample_count}, found {len(candidates)}; using {effective_total}"
        )

    effective_fixed_count = min(
        fixed_count if fixed_count is not None else 8,
        effective_total,
    )
    fixed_selection = _stable_sample(
        candidates,
        effective_fixed_count,
        f"{manifest_path}:character_ref_to_panel:fixed",
    )
    fixed_keys = {_manifest_entry_identity(entry) for entry in fixed_selection}
    random_pool = [entry for entry in candidates if _manifest_entry_identity(entry) not in fixed_keys]
    random_count = effective_total - len(fixed_selection)
    random_selection = random.sample(random_pool, random_count) if random_count > 0 else []
    selected = fixed_selection + random_selection

    samples: list[dict] = []
    for entry in selected:
        target_path = _materialize_manifest_validation_asset(
            manifest_path,
            entry["target_panel"],
        )
        sample_width, sample_height = _infer_validation_dimensions(
            target_path=target_path,
            bucket_tolerance=bucket_tolerance,
        )
        sample = {
            "prompt": entry["caption"],
            "width": sample_width,
            "height": sample_height,
        }
        control_paths = _manifest_validation_control_paths_for_entry(manifest_path, entry)

        for idx, path in enumerate(control_paths, start=1):
            sample[f"ctrl_img_{idx}"] = str(path)
        sample["validation_sample_id"] = str(entry.get("sample_id", ""))
        sample["validation_target_path"] = str(target_path)
        sample["validation_caption"] = str(entry["caption"])
        sample["validation_control_paths"] = [str(path) for path in control_paths]
        samples.append(sample)

    return samples


def build_manifest_validation_samples(
    manifest_path: Path,
    description_only_count: int = 5,
    next_page_first_panel_count: int = 5,
    within_page_next_panel_count: int = 10,
    character_panel_count: int | None = None,
    character_panel_fixed_count: int | None = None,
    bucket_tolerance: int = 16,
) -> list[dict]:
    entries = load_manifest_entries(manifest_path)
    sample_types = _manifest_sample_types(entries)
    if sample_types and sample_types.issubset({"character_ref_to_panel", "panel_prediction"}):
        return _build_character_panel_validation_sample_dicts(
            manifest_path=manifest_path,
            sample_count=character_panel_count
            if character_panel_count is not None
            else description_only_count + next_page_first_panel_count + within_page_next_panel_count,
            fixed_count=character_panel_fixed_count,
            bucket_tolerance=bucket_tolerance,
        )
    return _build_manifest_validation_sample_dicts(
        manifest_path=manifest_path,
        description_only_count=description_only_count,
        next_page_first_panel_count=next_page_first_panel_count,
        within_page_next_panel_count=within_page_next_panel_count,
        bucket_tolerance=bucket_tolerance,
    )


def _infer_validation_dimensions(
    *,
    target_path: Path,
    bucket_tolerance: int,
) -> tuple[int, int]:
    with Image.open(target_path) as image:
        width, height = image.size

    divisibility = max(1, int(bucket_tolerance))
    if width % divisibility != 0 or height % divisibility != 0:
        raise ValueError(
            "Validation target image dimensions must be divisible by "
            f"{divisibility}; got {width}x{height} for {target_path}"
        )
    return int(width), int(height)


def build_multi_manifest_validation_samples(
    manifest_paths: list[Path],
    description_only_count: int = 5,
    next_page_first_panel_count: int = 5,
    within_page_next_panel_count: int = 10,
    character_panel_count: int | None = None,
    character_panel_fixed_count: int | None = None,
    bucket_tolerance: int = 16,
) -> list[dict]:
    valid_paths = [path for path in manifest_paths if path is not None]
    if not valid_paths:
        raise RuntimeError("No manifest paths were provided for validation sampling.")
    if len(valid_paths) == 1:
        return build_manifest_validation_samples(
            manifest_path=valid_paths[0],
            description_only_count=description_only_count,
            next_page_first_panel_count=next_page_first_panel_count,
            within_page_next_panel_count=within_page_next_panel_count,
            character_panel_count=character_panel_count,
            character_panel_fixed_count=character_panel_fixed_count,
            bucket_tolerance=bucket_tolerance,
        )

    sample_count = (
        character_panel_count
        if character_panel_count is not None
        else description_only_count + next_page_first_panel_count + within_page_next_panel_count
    )
    fixed_count = character_panel_fixed_count

    fixed_selected: list[dict] = []
    random_selected: list[dict] = []
    base_sample_count = sample_count // len(valid_paths)
    sample_remainder = sample_count % len(valid_paths)
    base_fixed_count = (fixed_count // len(valid_paths)) if fixed_count is not None else None
    fixed_remainder = (fixed_count % len(valid_paths)) if fixed_count is not None else 0

    for index, manifest_path in enumerate(valid_paths):
        manifest_sample_count = base_sample_count + (1 if index < sample_remainder else 0)
        manifest_fixed_count = None
        if fixed_count is not None:
            manifest_fixed_count = base_fixed_count + (1 if index < fixed_remainder else 0)
        if manifest_sample_count <= 0:
            continue
        manifest_selected = build_manifest_validation_samples(
            manifest_path=manifest_path,
            description_only_count=description_only_count,
            next_page_first_panel_count=next_page_first_panel_count,
            within_page_next_panel_count=within_page_next_panel_count,
            character_panel_count=manifest_sample_count,
            character_panel_fixed_count=manifest_fixed_count,
            bucket_tolerance=bucket_tolerance,
        )
        effective_manifest_fixed_count = min(manifest_fixed_count or 0, len(manifest_selected))
        fixed_selected.extend(manifest_selected[:effective_manifest_fixed_count])
        random_selected.extend(manifest_selected[effective_manifest_fixed_count:])
    return fixed_selected + random_selected


def infer_manifest_validation_divisibility(
    manifest_datasets: list[dict],
    *,
    default_bucket_tolerance: int = 16,
) -> int:
    if not manifest_datasets:
        return default_bucket_tolerance
    first_dataset = manifest_datasets[0] or {}
    return int(first_dataset.get("bucket_tolerance") or default_bucket_tolerance)


def s3_uri(*parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    if clean_parts:
        return f"s3://{S3_BUCKET}/{'/'.join(clean_parts)}"
    return f"s3://{S3_BUCKET}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket_and_key = uri[5:]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return bucket, key


def checkpoint_upload_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=int(os.environ.get("S3_CHECKPOINT_MULTIPART_THRESHOLD_MB", "64")) * 1024 * 1024,
        multipart_chunksize=int(os.environ.get("S3_CHECKPOINT_MULTIPART_CHUNK_MB", "128")) * 1024 * 1024,
        max_concurrency=int(os.environ.get("S3_CHECKPOINT_UPLOAD_CONCURRENCY", "16")),
        use_threads=True,
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def path_snapshot(path: Path) -> tuple[int, int, int]:
    if path.is_file():
        stat = path.stat()
        return (1, int(stat.st_size), int(stat.st_mtime_ns))
    file_count = 0
    total_size = 0
    max_mtime_ns = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        stat = child.stat()
        file_count += 1
        total_size += int(stat.st_size)
        max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
    return (file_count, total_size, max_mtime_ns)


def iter_checkpoint_upload_candidates(save_root: Path, job_name: str) -> list[Path]:
    if not save_root.exists():
        return []

    candidates: list[Path] = []
    for fixed_name in ("config.yaml", "optimizer.pt"):
        fixed_path = save_root / fixed_name
        if fixed_path.is_file():
            candidates.append(fixed_path)

    for path in sorted(save_root.glob(f"{job_name}*")):
        if path.name in {"samples", "rank_logs", "peft_adapter"}:
            continue
        if path.is_file() or path.is_dir():
            candidates.append(path)
    return candidates


def checkpoint_destination_uri(path: Path, *, job_name: str, model_prefix: str) -> str:
    if path.name == "config.yaml":
        return s3_uri(model_prefix, "config.yaml")
    if path.name == "optimizer.pt":
        return s3_uri(model_prefix, "optimizer.pt")
    return s3_uri(model_prefix, "checkpoints", path.name)


def verify_s3_object_size(*, s3_client, bucket: str, key: str, expected_size: int) -> None:
    head = s3_client.head_object(Bucket=bucket, Key=key)
    actual_size = int(head.get("ContentLength") or 0)
    if actual_size != expected_size:
        raise RuntimeError(
            "S3 checkpoint upload verification failed: "
            f"s3://{bucket}/{key} expected={expected_size} actual={actual_size}"
        )


def upload_path_to_s3(path: Path, destination_uri: str, *, s3_client, transfer_config) -> None:
    if path.is_file():
        bucket, key = parse_s3_uri(destination_uri)
        s3_client.upload_file(str(path), bucket, key, Config=transfer_config)
        verify_s3_object_size(
            s3_client=s3_client,
            bucket=bucket,
            key=key,
            expected_size=path.stat().st_size,
        )
        return

    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative_key = child.relative_to(path).as_posix()
        bucket, key_prefix = parse_s3_uri(destination_uri.rstrip("/"))
        s3_client.upload_file(
            str(child),
            bucket,
            f"{key_prefix}/{relative_key}",
            Config=transfer_config,
        )
        verify_s3_object_size(
            s3_client=s3_client,
            bucket=bucket,
            key=f"{key_prefix}/{relative_key}",
            expected_size=child.stat().st_size,
        )


def is_prunable_checkpoint_artifact(
    path: Path,
    *,
    job_name: str,
    include_metadata: bool = False,
) -> bool:
    if include_metadata and path.name in {"config.yaml", "optimizer.pt"}:
        return True
    if path.name in {"config.yaml", "optimizer.pt", "samples", "rank_logs", "peft_adapter"}:
        return False
    return path.name.startswith(job_name)


def checkpoint_artifact_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"_(\d{6,12})(?:\.|$)", path.name)
    step = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    return (step, mtime, path.name)


def checkpoint_name_sort_key(name: str, last_modified: Any = None) -> tuple[int, float, str]:
    match = re.search(r"_(\d{6,12})(?:\.|$)", name)
    step = int(match.group(1)) if match else -1
    timestamp = 0.0
    if last_modified is not None:
        try:
            timestamp = float(last_modified.timestamp())
        except Exception:
            timestamp = 0.0
    return (step, timestamp, name)


def delete_local_checkpoint_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def latest_local_checkpoint_artifact(save_root: Path, job_name: str) -> Path | None:
    candidates = [
        path
        for path in iter_checkpoint_upload_candidates(save_root, job_name)
        if is_prunable_checkpoint_artifact(path, job_name=job_name)
    ]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_artifact_sort_key)


def newest_s3_checkpoint_object(*, s3_client, model_prefix: str, job_name: str) -> dict | None:
    prefix = f"{model_prefix.strip('/')}/checkpoints/"
    paginator = s3_client.get_paginator("list_objects_v2")
    newest: dict | None = None
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            name = key.rsplit("/", 1)[-1]
            if not name.startswith(job_name):
                continue
            if newest is None or checkpoint_name_sort_key(name, obj.get("LastModified")) > checkpoint_name_sort_key(
                newest["Key"].rsplit("/", 1)[-1],
                newest.get("LastModified"),
            ):
                newest = obj
    return newest


def download_s3_object_to_path(*, s3_client, bucket: str, key: str, destination: Path, transfer_config) -> bool:
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    expected_size = int(head.get("ContentLength") or 0)
    if destination.exists() and destination.is_file() and destination.stat().st_size == expected_size:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.download")
    if tmp_path.exists():
        tmp_path.unlink()
    s3_client.download_file(bucket, key, str(tmp_path), Config=transfer_config)
    verify_s3_object_size(
        s3_client=s3_client,
        bucket=bucket,
        key=key,
        expected_size=tmp_path.stat().st_size,
    )
    os.replace(tmp_path, destination)
    return True


def hydrate_checkpoint_resume_state_from_s3(
    *,
    save_root: Path,
    job_name: str,
    model_prefix: str,
) -> set[str]:
    if not env_flag("S3_CHECKPOINT_HYDRATE_ON_START", default=True):
        return set()

    import boto3

    s3_client = boto3.client("s3")
    transfer_config = checkpoint_upload_config()
    protected_paths: set[str] = set()
    save_root.mkdir(parents=True, exist_ok=True)

    remote_checkpoint = newest_s3_checkpoint_object(
        s3_client=s3_client,
        model_prefix=model_prefix,
        job_name=job_name,
    )
    if remote_checkpoint is not None:
        remote_key = str(remote_checkpoint["Key"])
        remote_name = remote_key.rsplit("/", 1)[-1]
        local_latest = latest_local_checkpoint_artifact(save_root, job_name)
        should_download = local_latest is None
        if local_latest is not None:
            should_download = checkpoint_name_sort_key(remote_name, remote_checkpoint.get("LastModified")) > checkpoint_artifact_sort_key(local_latest)
        local_checkpoint = save_root / remote_name
        if should_download:
            print(
                f"☁️  Hydrating latest S3 checkpoint for resume: s3://{S3_BUCKET}/{remote_key} -> {local_checkpoint}",
                flush=True,
            )
            if download_s3_object_to_path(
                s3_client=s3_client,
                bucket=S3_BUCKET,
                key=remote_key,
                destination=local_checkpoint,
                transfer_config=transfer_config,
            ):
                protected_paths.add(str(local_checkpoint))
        elif local_latest is not None:
            protected_paths.add(str(local_latest))

    for name in ("config.yaml", "optimizer.pt"):
        key = f"{model_prefix.strip('/')}/{name}"
        destination = save_root / name
        if download_s3_object_to_path(
            s3_client=s3_client,
            bucket=S3_BUCKET,
            key=key,
            destination=destination,
            transfer_config=transfer_config,
        ):
            protected_paths.add(str(destination))

    return protected_paths


def upload_ready_checkpoints_once(
    *,
    save_root: Path,
    job_name: str,
    model_prefix: str,
    seen_snapshots: dict[str, tuple[tuple[int, int, int], float]],
    uploaded_snapshots: dict[str, tuple[int, int, int]],
    stable_seconds: float,
    protected_paths: set[str] | None = None,
    force: bool = False,
) -> int:
    import boto3

    now = time.monotonic()
    s3_client = boto3.client("s3")
    transfer_config = checkpoint_upload_config()
    uploaded_count = 0
    delete_local_after_upload = env_flag("S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD", default=False)
    delete_metadata_after_upload = env_flag(
        "S3_CHECKPOINT_DELETE_LOCAL_METADATA_AFTER_UPLOAD",
        default=delete_local_after_upload,
    )
    keep_latest_local = env_flag("S3_CHECKPOINT_KEEP_LATEST_LOCAL", default=True)
    candidates = iter_checkpoint_upload_candidates(save_root, job_name)
    keep_local_paths: set[str] = set()
    protected_paths = protected_paths or set()
    if delete_local_after_upload and keep_latest_local:
        prunable = [
            candidate
            for candidate in candidates
            if is_prunable_checkpoint_artifact(
                candidate,
                job_name=job_name,
                include_metadata=delete_metadata_after_upload,
            )
        ]
        if prunable:
            keep_local_paths.add(str(max(prunable, key=checkpoint_artifact_sort_key)))

    for path in candidates:
        path_key = str(path)
        try:
            snapshot = path_snapshot(path)
        except FileNotFoundError:
            continue
        if snapshot[0] <= 0:
            continue
        if uploaded_snapshots.get(path_key) == snapshot:
            continue

        previous = seen_snapshots.get(path_key)
        if previous is None or previous[0] != snapshot:
            seen_snapshots[path_key] = (snapshot, now)
            if not force:
                continue

        stable_since = seen_snapshots.get(path_key, (snapshot, now))[1]
        if not force and now - stable_since < stable_seconds:
            continue

        destination_uri = checkpoint_destination_uri(
            path,
            job_name=job_name,
            model_prefix=model_prefix,
        )
        print(
            f"☁️  Uploading checkpoint artifact to S3: {path} -> {destination_uri}",
            flush=True,
        )
        upload_path_to_s3(path, destination_uri, s3_client=s3_client, transfer_config=transfer_config)
        uploaded_snapshots[path_key] = snapshot
        uploaded_count += 1
        print(f"✅ Uploaded checkpoint artifact to S3: {destination_uri}", flush=True)
        if (
            delete_local_after_upload
            and path_key not in keep_local_paths
            and path_key not in protected_paths
            and is_prunable_checkpoint_artifact(
                path,
                job_name=job_name,
                include_metadata=delete_metadata_after_upload,
            )
        ):
            try:
                delete_local_checkpoint_artifact(path)
                print(f"🧹 Deleted local checkpoint artifact after S3 upload: {path}", flush=True)
            except FileNotFoundError:
                pass

    return uploaded_count


def start_checkpoint_s3_uploader(
    *,
    save_root: Path,
    job_name: str,
    model_prefix: str,
    stop_event: threading.Event,
    protected_paths: set[str] | None = None,
) -> threading.Thread:
    interval_seconds = float(os.environ.get("S3_CHECKPOINT_UPLOAD_INTERVAL_SECONDS", "30"))
    stable_seconds = float(os.environ.get("S3_CHECKPOINT_STABLE_SECONDS", "20"))
    delete_local_after_upload = env_flag("S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD", default=False)
    delete_metadata_after_upload = env_flag(
        "S3_CHECKPOINT_DELETE_LOCAL_METADATA_AFTER_UPLOAD",
        default=delete_local_after_upload,
    )
    keep_latest_local = env_flag("S3_CHECKPOINT_KEEP_LATEST_LOCAL", default=True)
    seen_snapshots: dict[str, tuple[tuple[int, int, int], float]] = {}
    uploaded_snapshots: dict[str, tuple[int, int, int]] = {}
    protected_paths = protected_paths or set()
    commit_model_volume = str(save_root).startswith(str(Path(MOUNT_DIR)))

    def worker() -> None:
        print(
            "☁️  Live checkpoint S3 uploader started "
            f"(interval={interval_seconds}s stable={stable_seconds}s "
            f"prefix={s3_uri(model_prefix)} "
            f"delete_local_after_upload={delete_local_after_upload} "
            f"delete_metadata_after_upload={delete_metadata_after_upload} "
            f"keep_latest_local={keep_latest_local})",
            flush=True,
        )
        while not stop_event.is_set():
            try:
                uploaded_count = upload_ready_checkpoints_once(
                    save_root=save_root,
                    job_name=job_name,
                    model_prefix=model_prefix,
                    seen_snapshots=seen_snapshots,
                    uploaded_snapshots=uploaded_snapshots,
                    stable_seconds=stable_seconds,
                    protected_paths=protected_paths,
                )
                if uploaded_count and commit_model_volume:
                    try:
                        model_volume.commit()
                    except Exception as exc:
                        print(f"⚠️  Modal volume commit after checkpoint upload failed: {exc}", flush=True)
            except Exception as exc:
                print(f"⚠️  Live checkpoint S3 uploader error: {exc}", flush=True)
            stop_event.wait(interval_seconds)

        try:
            uploaded_count = upload_ready_checkpoints_once(
                save_root=save_root,
                job_name=job_name,
                model_prefix=model_prefix,
                seen_snapshots=seen_snapshots,
                uploaded_snapshots=uploaded_snapshots,
                stable_seconds=0,
                protected_paths=protected_paths,
                force=True,
            )
            if uploaded_count and commit_model_volume:
                model_volume.commit()
        except Exception as exc:
            print(f"⚠️  Final checkpoint S3 upload failed: {exc}", flush=True)
        print("☁️  Live checkpoint S3 uploader stopped", flush=True)

    thread = threading.Thread(target=worker, name="checkpoint-s3-uploader", daemon=True)
    thread.start()
    return thread


def apply_runtime_dataset_defaults(
    dataset_config: dict,
    *,
    folder_path: str | None = None,
) -> None:
    dataset_config["num_repeats"] = 1
    if folder_path is not None:
        dataset_config["folder_path"] = folder_path


def prepare_manifest_training_datasets(
    process_config: dict,
    *,
    dataset_subpath: str | None,
    staged_dataset_dir: Path | None,
) -> dict[str, object]:
    manifest_datasets = [
        dataset for dataset in process_config.get("datasets", []) if dataset.get("type") == "manifest"
    ]
    if not manifest_datasets:
        raise RuntimeError("Manifest training was detected but no manifest datasets were configured.")

    validation_bucket_tolerance = infer_manifest_validation_divisibility(manifest_datasets)
    manifest_paths: list[Path] = []
    resolved_staged_dataset_dir = staged_dataset_dir
    total_samples = 0
    job_name = str(process_config.get("name") or "manifest_job").strip() or "manifest_job"

    for dataset_index, manifest_dataset in enumerate(manifest_datasets):
        manifest_path, manifest_entries, overlay_summary = materialize_manifest_dataset_view(
            manifest_dataset,
            job_name=job_name,
            dataset_index=dataset_index,
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest dataset path does not exist: {manifest_path}")
        original_manifest_path = str(manifest_dataset.get("manifest_path") or "").strip()
        if overlay_summary.get("materialized_overlay") or not original_manifest_path.startswith("s3://"):
            manifest_dataset["manifest_path"] = str(manifest_path)
        manifest_paths.append(manifest_path)
        if resolved_staged_dataset_dir is None:
            dataset_s3_path = str(manifest_dataset.get("dataset_s3_path") or "").strip()
            if dataset_s3_path:
                resolved_staged_dataset_dir = (
                    s3_uri_to_mount_path(dataset_s3_path)
                    if dataset_s3_path.startswith("s3://")
                    else Path(dataset_s3_path)
                )
            else:
                resolved_staged_dataset_dir = manifest_path.parent

        dataset_row_count = int(
            overlay_summary.get("resolved_row_count")
            or overlay_summary.get("row_count")
            or len(manifest_entries)
        )
        if dataset_row_count <= 0:
            raise RuntimeError(f"Manifest dataset resolved to zero rows: {manifest_path}")
        total_samples += dataset_row_count
        manifest_dataset["_prepared_row_count"] = dataset_row_count
        manifest_dataset["_prepared_manifest_summary"] = dict(overlay_summary)
        apply_runtime_dataset_defaults(manifest_dataset)

    return {
        "manifest_datasets": manifest_datasets,
        "manifest_paths": manifest_paths,
        "validation_bucket_tolerance": validation_bucket_tolerance,
        "total_samples": total_samples,
        "staged_dataset_dir": resolved_staged_dataset_dir,
    }


def configure_manifest_validation_samples(
    sample_config: dict,
    *,
    manifest_paths: list[Path],
    validation_bucket_tolerance: int,
    ddp_world_size: int,
) -> None:
    sample_config.pop("prompts", None)
    if ddp_world_size > 1:
        distributed_samples_per_rank = 2
        distributed_sample_total = max(2, ddp_world_size * distributed_samples_per_rank)
        sample_config["samples"] = build_multi_manifest_validation_samples(
            manifest_paths=manifest_paths,
            description_only_count=4,
            next_page_first_panel_count=4,
            within_page_next_panel_count=8,
            character_panel_count=distributed_sample_total,
            character_panel_fixed_count=ddp_world_size,
            bucket_tolerance=validation_bucket_tolerance,
        )
        sample_config.setdefault("sample_every", 31)
        sample_config["distributed_sampling"] = True
        sample_config["distributed_sample_total"] = distributed_sample_total
        sample_config["distributed_samples_per_rank"] = distributed_samples_per_rank
        sample_config["distributed_sample_strategy"] = "round_robin"
        sample_config["dynamic_validation_enabled"] = True
        sample_config["dynamic_validation_manifest_paths"] = [
            str(path) for path in manifest_paths
        ]
        sample_config["dynamic_validation_bucket_tolerance"] = validation_bucket_tolerance
        sample_config["dynamic_validation_character_panel_count"] = distributed_sample_total
        sample_config["dynamic_validation_character_panel_fixed_count"] = ddp_world_size
        return

    sample_config["samples"] = build_multi_manifest_validation_samples(
        manifest_paths=manifest_paths,
        description_only_count=5,
        next_page_first_panel_count=5,
        within_page_next_panel_count=10,
        character_panel_count=8,
        bucket_tolerance=validation_bucket_tolerance,
    )
    sample_config.setdefault("sample_every", 31)


def parse_gradient_accumulation_steps(raw: Any, default: int = 1) -> int:
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        raise ValueError("gradient_accumulation_steps must be an integer >= 1")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("gradient_accumulation_steps must be an integer >= 1")
    if value < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    return value


def resolve_training_steps(
    process_config: dict,
    *,
    steps_per_epoch: int,
    target_epochs: int,
    max_train_steps: int = 0,
) -> int:
    train_config = process_config.get("train", {})
    fixed_steps = int(train_config.get("fixed_steps") or 0)
    if fixed_steps > 0:
        if max_train_steps > 0:
            return min(fixed_steps, max_train_steps)
        return fixed_steps

    total_steps = steps_per_epoch * max(1, target_epochs)
    if max_train_steps > 0:
        total_steps = min(total_steps, max_train_steps)
    return total_steps


def resolve_gradient_accumulation_steps(process_config: dict) -> int:
    train_config = process_config.get("train", {})
    if "gradient_accumulation" in train_config:
        raise ValueError(
            "Use train.gradient_accumulation_steps; train.gradient_accumulation is not supported"
        )
    return parse_gradient_accumulation_steps(
        train_config.get("gradient_accumulation_steps", 1)
    )


def compute_steps_per_epoch(
    *,
    num_train_items: int,
    process_config: dict,
    world_size: int = 1,
) -> int:
    train_config = process_config.get("train", {})
    per_gpu_batch = max(1, int(train_config.get("batch_size", 1) or 1))
    grad_accum = resolve_gradient_accumulation_steps(process_config)
    global_batch = per_gpu_batch * max(1, int(world_size)) * grad_accum
    return max(1, math.ceil(max(1, int(num_train_items)) / global_batch))


def _infer_bucket_divisibility_for_manifest(process_config: dict, dataset_config: dict) -> int:
    model_arch = str(process_config.get("model", {}).get("arch") or "").strip().lower()
    if model_arch.startswith("flux2"):
        return 16
    return max(1, int(dataset_config.get("bucket_tolerance", 64) or 64))


def estimate_manifest_emitted_batch_stats(
    *,
    process_config: dict,
    manifest_dataset_config: dict,
) -> dict:
    if "/root/ai-toolkit" not in sys.path:
        sys.path.insert(0, "/root/ai-toolkit")

    from toolkit.config_modules import DatasetConfig
    from toolkit.manifest_dataset import ManifestDataset

    class _ManifestStepEstimateSD:
        def __init__(self, *, bucket_divisibility: int, use_raw_multi_control: bool):
            self.use_raw_control_images = use_raw_multi_control
            self.has_multiple_control_images = use_raw_multi_control
            self._bucket_divisibility = bucket_divisibility

        def get_bucket_divisibility(self) -> int:
            return self._bucket_divisibility

    estimate_config_dict = dict(manifest_dataset_config)
    estimate_config_dict["cache_latents"] = False
    estimate_config_dict["cache_latents_to_disk"] = False
    estimate_config_dict.setdefault("fast_image_size", True)

    bucket_divisibility = _infer_bucket_divisibility_for_manifest(
        process_config,
        estimate_config_dict,
    )
    dataset_config = DatasetConfig(**estimate_config_dict)
    dataset = ManifestDataset(
        dataset_config,
        batch_size=max(1, int(process_config.get("train", {}).get("batch_size", 1) or 1)),
        sd=_ManifestStepEstimateSD(
            bucket_divisibility=bucket_divisibility,
            use_raw_multi_control=True,
        ),
    )

    microbatch_sizes = [len(batch) for batch in dataset.batch_indices]
    emitted_batches = len(dataset)
    total_samples = len(dataset.file_list)
    average_batch_size = (
        float(total_samples) / float(emitted_batches) if emitted_batches > 0 else 0.0
    )
    return {
        "total_samples": total_samples,
        "emitted_batches": emitted_batches,
        "min_batch_size": min(microbatch_sizes) if microbatch_sizes else 0,
        "max_batch_size": max(microbatch_sizes) if microbatch_sizes else 0,
        "avg_batch_size": average_batch_size,
    }


def compute_manifest_steps_per_epoch(
    *,
    process_config: dict,
    manifest_dataset_config: dict,
    world_size: int = 1,
) -> tuple[int, dict]:
    stats = estimate_manifest_emitted_batch_stats(
        process_config=process_config,
        manifest_dataset_config=manifest_dataset_config,
    )
    grad_accum = resolve_gradient_accumulation_steps(process_config)
    per_rank_microbatches = max(
        1,
        math.ceil(int(stats["emitted_batches"]) / max(1, int(world_size))),
    )
    steps_per_epoch = max(1, math.ceil(per_rank_microbatches / grad_accum))
    stats = {
        **stats,
        "per_rank_microbatches": per_rank_microbatches,
        "gradient_accumulation_steps": grad_accum,
        "world_size": max(1, int(world_size)),
    }
    return steps_per_epoch, stats


def compute_multi_manifest_steps_per_epoch(
    *,
    process_config: dict,
    manifest_dataset_configs: list[dict],
    world_size: int = 1,
) -> tuple[int, dict]:
    if not manifest_dataset_configs:
        raise RuntimeError("No manifest dataset configs were provided.")
    per_gpu_batch = max(1, int(process_config.get("train", {}).get("batch_size", 1) or 1))
    if per_gpu_batch == 1 and all(
        int(dataset_config.get("_prepared_row_count") or 0) > 0
        for dataset_config in manifest_dataset_configs
    ):
        total_samples = sum(
            int(dataset_config.get("_prepared_row_count") or 0)
            for dataset_config in manifest_dataset_configs
        )
        emitted_batches = total_samples
        grad_accum = resolve_gradient_accumulation_steps(process_config)
        per_rank_microbatches = max(
            1,
            math.ceil(int(emitted_batches) / max(1, int(world_size))),
        )
        steps_per_epoch = max(1, math.ceil(per_rank_microbatches / grad_accum))
        return steps_per_epoch, {
            "total_samples": total_samples,
            "emitted_batches": emitted_batches,
            "min_batch_size": 1,
            "max_batch_size": 1,
            "avg_batch_size": 1.0,
            "per_rank_microbatches": per_rank_microbatches,
            "gradient_accumulation_steps": grad_accum,
            "world_size": max(1, int(world_size)),
            "dataset_count": len(manifest_dataset_configs),
            "estimation_mode": "prepared_manifest_row_count",
        }
    if len(manifest_dataset_configs) == 1:
        return compute_manifest_steps_per_epoch(
            process_config=process_config,
            manifest_dataset_config=manifest_dataset_configs[0],
            world_size=world_size,
        )

    per_dataset_stats = [
        estimate_manifest_emitted_batch_stats(
            process_config=process_config,
            manifest_dataset_config=manifest_dataset_config,
        )
        for manifest_dataset_config in manifest_dataset_configs
    ]
    total_samples = sum(int(stats["total_samples"]) for stats in per_dataset_stats)
    emitted_batches = sum(int(stats["emitted_batches"]) for stats in per_dataset_stats)
    grad_accum = resolve_gradient_accumulation_steps(process_config)
    per_rank_microbatches = max(
        1,
        math.ceil(int(emitted_batches) / max(1, int(world_size))),
    )
    steps_per_epoch = max(1, math.ceil(per_rank_microbatches / grad_accum))
    weighted_avg_batch_size = (
        float(total_samples) / float(emitted_batches) if emitted_batches > 0 else 0.0
    )
    stats = {
        "total_samples": total_samples,
        "emitted_batches": emitted_batches,
        "min_batch_size": min(int(stats["min_batch_size"]) for stats in per_dataset_stats),
        "max_batch_size": max(int(stats["max_batch_size"]) for stats in per_dataset_stats),
        "avg_batch_size": weighted_avg_batch_size,
        "per_rank_microbatches": per_rank_microbatches,
        "gradient_accumulation_steps": grad_accum,
        "world_size": max(1, int(world_size)),
        "dataset_count": len(per_dataset_stats),
    }
    return steps_per_epoch, stats



# Define Modal image with a Dockerfile so requirements are cached independently
# from ai-toolkit source changes. Fast-changing source code is added at
# container startup instead of being baked into image layers.
base_training_image = (
    modal.Image.from_dockerfile(str(LOCAL_DOCKERFILE_PATH))
    .add_local_dir(
        str(REPO_ROOT / "ai-toolkit"),
        remote_path="/root/ai-toolkit",
        copy=False,
    )
    .add_local_dir(
        str(TRAINING_DIR / "configs"),
        remote_path="/root/training/configs",
        copy=False,
    )
)

image_4b = base_training_image
image_9b = base_training_image

app = modal.App(name="flux-lora-training", image=image_4b)
aws_secret = modal.Secret.from_name(AWS_SECRET_NAME)
hf_secret = modal.Secret.from_name("flux2-klein-user-hf-token")
s3_mount = modal.CloudBucketMount(
    bucket_name=S3_BUCKET,
    secret=aws_secret,
    read_only=True,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _process_config(parsed_config: dict[str, Any]) -> dict[str, Any]:
    processes = parsed_config.get("config", {}).get("process", [])
    if not processes:
        raise RuntimeError("Training config has no config.process entries")
    process_config = processes[0]
    if not isinstance(process_config, dict):
        raise RuntimeError("Training config process[0] is not an object")
    return process_config


def _manifest_dataset_configs(parsed_config: dict[str, Any]) -> list[dict[str, Any]]:
    process_config = _process_config(parsed_config)
    return [
        dataset
        for dataset in process_config.get("datasets", []) or []
        if isinstance(dataset, dict) and dataset.get("type") == "manifest"
    ]


def _drawtoon_config_from_parsed(
    parsed_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    drawtoon: dict[str, Any] = {}
    if isinstance(parsed_config.get("drawtoon"), dict):
        drawtoon.update(parsed_config["drawtoon"])
    if isinstance(parsed_config.get("config"), dict) and isinstance(parsed_config["config"].get("drawtoon"), dict):
        drawtoon.update(parsed_config["config"]["drawtoon"])

    for key, value in (overrides or {}).items():
        if value is not None and str(value).strip() != "":
            drawtoon[key] = value

    caption_run = str(drawtoon.get("caption_run") or DEFAULT_DRAWTOON_CAPTION_RUN).strip()
    if not caption_run:
        raise ValueError("drawtoon.caption_run is required")
    caption_field = str(drawtoon.get("caption_field") or "caption").strip()
    caption_format = str(drawtoon.get("caption_format") or "text").strip().lower()
    if caption_format not in {"text", "compact_json"}:
        raise ValueError(
            f"Unsupported drawtoon.caption_format={caption_format!r}; expected text or compact_json"
        )
    return {
        "bucket": str(drawtoon.get("bucket") or S3_BUCKET).strip(),
        "pages_prefix": str(drawtoon.get("pages_prefix") or DEFAULT_DRAWTOON_PAGES_PREFIX).strip().strip("/"),
        "annotations_prefix": str(
            drawtoon.get("annotations_prefix") or DEFAULT_DRAWTOON_ANNOTATIONS_PREFIX
        ).strip().strip("/"),
        "captions_prefix": str(drawtoon.get("captions_prefix") or DEFAULT_DRAWTOON_CAPTIONS_PREFIX).strip().strip("/"),
        "caption_run": caption_run,
        "caption_field": caption_field,
        "caption_format": caption_format,
        "include_chapter_regex": str(drawtoon.get("include_chapter_regex") or "").strip(),
        "max_pages": max(0, int(drawtoon.get("max_pages") or 0)),
        "target_multiple": max(1, int(drawtoon.get("target_multiple") or 16)),
        "exclude_sample_ids_path": str(drawtoon.get("exclude_sample_ids_path") or "").strip(),
    }


def _config_needs_drawtoon_cache(parsed_config: dict[str, Any]) -> bool:
    manifest_datasets = _manifest_dataset_configs(parsed_config)
    if not manifest_datasets:
        return False
    return any(not str(dataset.get("manifest_path") or "").strip() for dataset in manifest_datasets)


def _drawtoon_cache_key(drawtoon: dict[str, Any]) -> str:
    fingerprint_payload = {
        "bucket": drawtoon["bucket"],
        "pages_prefix": drawtoon["pages_prefix"],
        "annotations_prefix": drawtoon["annotations_prefix"],
        "captions_prefix": drawtoon["captions_prefix"],
        "caption_run": drawtoon["caption_run"],
        "caption_field": drawtoon.get("caption_field", "caption"),
        "caption_format": drawtoon.get("caption_format", "text"),
        "include_chapter_regex": drawtoon["include_chapter_regex"],
        "max_pages": int(drawtoon["max_pages"]),
        "target_multiple": int(drawtoon["target_multiple"]),
        "exclude_sample_ids_path": drawtoon.get("exclude_sample_ids_path", ""),
        "schema": "drawtoon_panel_cache_v12_with_tails_manga_suffix_real_dims_tail_region_index",
    }
    digest = hashlib.sha1(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    run_label = sanitize_filename(str(drawtoon["caption_run"]), fallback="caption_run")
    return f"{run_label}_{digest}"


def _caption_chapter_from_key(caption_root: str, key: str) -> str:
    rel = key[len(caption_root.rstrip("/") + "/") :]
    parts = rel.split("/")
    return parts[0] if len(parts) == 2 else ""


def _sort_caption_key(caption_root: str, key: str) -> tuple[str, str]:
    rel = key[len(caption_root.rstrip("/") + "/") :]
    parts = rel.split("/")
    if len(parts) != 2:
        return ("", rel)
    return (parts[0], parts[1])


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _drawtoon_sample_identity(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    return json.dumps(row.get("target_panel") or row, sort_keys=True, ensure_ascii=False)


def load_drawtoon_excluded_sample_ids(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Drawtoon exclude_sample_ids_path does not exist: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def write_resolved_drawtoon_cache_config(
    *,
    parsed_config: dict[str, Any],
    manifest_path: Path,
    cache_root: Path,
    row_count: int,
    output_path: Path,
) -> None:
    process_config = _process_config(parsed_config)
    manifest_datasets = [
        dataset
        for dataset in process_config.get("datasets", []) or []
        if isinstance(dataset, dict) and dataset.get("type") == "manifest"
    ]
    if not manifest_datasets:
        raise RuntimeError("No manifest dataset found to receive the Drawtoon cache manifest_path")
    for dataset in manifest_datasets:
        dataset["manifest_path"] = str(manifest_path)
        dataset["dataset_s3_path"] = str(cache_root)
        dataset["local_cache_dir"] = f"/tmp/drawtoon_panel_sample_cache/{cache_root.name}"
        dataset["layout"] = "manifest"
        dataset["_prepared_row_count"] = int(row_count)

    sample_config = process_config.setdefault("sample", {})
    sample_config["dynamic_validation_manifest_paths"] = [str(manifest_path)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(parsed_config, sort_keys=False), encoding="utf-8")


def _box_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _coerce_pixel_box(
    box: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
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


def _panel_relative_norm_box(
    entity_box: tuple[float, float, float, float],
    panel_box: tuple[float, float, float, float],
    *,
    padded_width: int,
    padded_height: int,
    min_entity_overlap: float = 0.05,
) -> list[float] | None:
    ex0, ey0, ex1, ey1 = entity_box
    px0, py0, px1, py1 = panel_box
    ix0 = max(ex0, px0)
    iy0 = max(ey0, py0)
    ix1 = min(ex1, px1)
    iy1 = min(ey1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    entity_area = _box_area(entity_box)
    if entity_area <= 0.0:
        return None
    if _box_area((ix0, iy0, ix1, iy1)) / entity_area < min_entity_overlap:
        return None
    return [
        round((ix0 - px0) / max(1, padded_width), 6),
        round((iy0 - py0) / max(1, padded_height), 6),
        round((ix1 - px0) / max(1, padded_width), 6),
        round((iy1 - py0) / max(1, padded_height), 6),
    ]


def _pad_image_to_multiple(image: Image.Image, multiple: int = 16) -> Image.Image:
    width, height = image.size
    padded_width = int(math.ceil(width / multiple) * multiple)
    padded_height = int(math.ceil(height / multiple) * multiple)
    if padded_width == width and padded_height == height:
        return image
    padded = Image.new("RGB", (padded_width, padded_height), "white")
    padded.paste(image, (0, 0))
    return padded


def _apply_ref_border_and_padding(
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
    if multiple > 1:
        return _pad_image_to_multiple(image, multiple=multiple)
    return image


def _panel_text_regions(panel: dict[str, Any]) -> list[Any]:
    for field_name in ("text_bubbles", "speech_bubbles", "text_regions", "texts", "bubbles"):
        regions = panel.get(field_name)
        if isinstance(regions, list):
            return regions
    return []


def _norm_box_to_pixel_box(
    box_norm: list[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if len(box_norm) != 4:
        return None
    x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
    y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
    x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _draw_layout_blob(
    draw: ImageDraw.ImageDraw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    line_width: int = 6,
) -> bool:
    pixel_box = _norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    x0, y0, x1, y1 = pixel_box
    draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, int(line_width)))
    return True


def _normalize_layout_text_bubble_type(value: object) -> str:
    text = " ".join(str(value or "").split()).strip().lower()
    if text in {"", "none", "null", "no", "not a bubble"}:
        return "None"
    if "free" in text or "floating" in text or "standalone" in text or "sign" in text or "sfx" in text:
        return "None"
    if "narration" in text or "caption" in text:
        return "Narration Bubble"
    if "shout" in text or "yell" in text or "scream" in text or "burst" in text:
        return "Shout Bubble"
    if "speech" in text or "dialogue" in text or "dialog" in text or "bubble" in text:
        return "Speech Bubble"
    return "None"


def _draw_text_layout_region(
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
    pixel_box = _norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    draw.rectangle(pixel_box, fill=text_color)
    return True


def _draw_tail_layout_region(
    draw: ImageDraw.ImageDraw,
    box_norm: list[float],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int] = LAYOUT_TAIL_COLOR,
) -> bool:
    pixel_box = _norm_box_to_pixel_box(box_norm, width=width, height=height)
    if pixel_box is None:
        return False
    draw.rectangle(pixel_box, fill=color)
    return True


def _panel_tail_regions(panel: dict[str, Any]) -> list[Any]:
    for field_name in ("tails", "speech_bubble_tails", "bubble_tails"):
        regions = panel.get(field_name)
        if isinstance(regions, list):
            return regions
    return []


def _build_panel_layout_control(
    *,
    panel: dict[str, Any],
    target_box: tuple[float, float, float, float],
    page_width: int,
    page_height: int,
    padded_width: int,
    padded_height: int,
    character_layout_items: list[dict[str, Any]],
) -> tuple[Image.Image, dict[str, Any]]:
    layout = Image.new("RGB", (padded_width, padded_height), LAYOUT_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(layout)
    drawn_characters: list[dict[str, Any]] = []
    for item in character_layout_items[:MAX_LAYOUT_CHARACTER_REFS]:
        box_norm = item.get("bbox_norm")
        color = item.get("rgb")
        if not isinstance(box_norm, list) or not isinstance(color, tuple):
            continue
        line_width = int(item.get("line_width") or 6)
        if _draw_layout_blob(
            draw,
            box_norm,
            width=padded_width,
            height=padded_height,
            color=color,
            line_width=line_width,
        ):
            drawn_characters.append(
                {
                    "character_label": item.get("character_label"),
                    "control_slot": item.get("control_slot"),
                    "color": item.get("color"),
                    "rgb": list(color),
                    "line_width": line_width,
                    "ref_border_width": CHARACTER_REF_BORDER_WIDTH,
                    "bbox_norm": box_norm,
                }
            )

    text_count = 0
    text_type_counts = {text_type: 0 for text_type in LAYOUT_TEXT_BUBBLE_TYPES}
    drawn_text_regions: list[dict[str, Any]] = []
    for text_region in _panel_text_regions(panel):
        if text_count >= MAX_LAYOUT_TEXT_REGIONS:
            break
        if not isinstance(text_region, dict):
            continue
        text_box = _coerce_pixel_box(text_region.get("bbox"), width=page_width, height=page_height)
        if text_box is None:
            text_box = _coerce_pixel_box(text_region.get("bbox_norm"), width=page_width, height=page_height)
        if text_box is None:
            continue
        text_box_norm = _panel_relative_norm_box(
            text_box,
            target_box,
            padded_width=padded_width,
            padded_height=padded_height,
        )
        if text_box_norm is None:
            continue
        text_bubble_type = _normalize_layout_text_bubble_type(text_region.get("type"))
        if text_bubble_type == "None":
            text_type_counts["None"] = int(text_type_counts.get("None", 0)) + 1
            drawn_text_regions.append(
                {
                    "text_region_index": text_region.get("text_region_index"),
                    "type": text_bubble_type,
                    "bbox_norm": text_box_norm,
                    "masked": False,
                }
            )
            continue
        if _draw_text_layout_region(
            draw,
            text_box_norm,
            width=padded_width,
            height=padded_height,
            text_bubble_type=text_bubble_type,
        ):
            text_type_counts[text_bubble_type] = int(text_type_counts.get(text_bubble_type, 0)) + 1
            drawn_text_regions.append(
                {
                    "text_region_index": text_region.get("text_region_index"),
                    "type": text_bubble_type,
                    "bbox_norm": text_box_norm,
                    "masked": True,
                    "shape": "filled_rectangle",
                    "preserve_text": False,
                }
            )
            text_count += 1

    tail_count = 0
    drawn_tail_regions: list[dict[str, Any]] = []
    for tail_region in _panel_tail_regions(panel):
        if tail_count >= MAX_LAYOUT_TAIL_REGIONS:
            break
        if not isinstance(tail_region, dict):
            continue
        tail_box = _coerce_pixel_box(tail_region.get("bbox"), width=page_width, height=page_height)
        if tail_box is None:
            tail_box = _coerce_pixel_box(tail_region.get("bbox_norm"), width=page_width, height=page_height)
        if tail_box is None:
            continue
        tail_box_norm = _panel_relative_norm_box(
            tail_box,
            target_box,
            padded_width=padded_width,
            padded_height=padded_height,
        )
        if tail_box_norm is None:
            continue
        if _draw_tail_layout_region(
            draw,
            tail_box_norm,
            width=padded_width,
            height=padded_height,
        ):
            drawn_tail_regions.append(
                {
                    "tail_region_index": tail_region.get("tail_region_index", tail_region.get("tail_index")),
                    "source_tail_index": tail_region.get("source_tail_index"),
                    "bbox_norm": tail_box_norm,
                    "shape": "filled_rectangle",
                }
            )
            tail_count += 1

    metadata = {
        "control_slot": 1,
        "text": {
            "color": "type_color",
            "rgb": list(LAYOUT_TEXT_COLOR),
            "colors": {text_type: list(rgb) for text_type, rgb in LAYOUT_TEXT_COLORS.items()},
            "count": text_count,
            "type_counts": text_type_counts,
            "types": {
                "Speech Bubble": "solid blue filled rectangle",
                "Narration Bubble": "solid orange filled rectangle",
                "Shout Bubble": "solid violet filled rectangle",
                "None": "no mask",
            },
            "regions": drawn_text_regions,
        },
        "tail": {
            "color": "silver",
            "rgb": list(LAYOUT_TAIL_COLOR),
            "count": tail_count,
            "shape": "filled_rectangle",
            "regions": drawn_tail_regions,
        },
        "characters": drawn_characters,
    }
    return layout, metadata


def _chapter_storage_key(chapter: str) -> str:
    """Caption JSON stores `chapter` unsuffixed (e.g. `*_mangazero`) but the
    durable S3 layout under `pages/*/` and `annotations/magi_v3/` uses the
    `*_mangazero_manga/` form. Idempotent for already-suffixed inputs."""
    return chapter if chapter.endswith("_manga") else f"{chapter}_manga"


def _caption_page_key(caption_payload: dict[str, Any], drawtoon: dict[str, Any]) -> str:
    sources = caption_payload.get("sources") if isinstance(caption_payload.get("sources"), dict) else {}
    page_key = str(sources.get("page_key") or "").strip()
    chapter = str(caption_payload.get("chapter") or "").strip()
    page_id = str(caption_payload.get("page_id") or "").strip()
    if not chapter or not page_id:
        raise ValueError("Caption payload is missing sources.page_key and chapter/page_id fallback fields")
    pages_prefix = str(drawtoon["pages_prefix"]).strip().strip("/")
    chapter_storage = _chapter_storage_key(chapter)
    if page_key.startswith(f"{pages_prefix}/{chapter_storage}/"):
        return page_key
    extension = ".png" if "text_removed" in pages_prefix.split("/") else (Path(page_key).suffix or ".jpg")
    return f"{pages_prefix}/{chapter_storage}/{page_id}{extension}"


def _panel_character_contexts(
    panels: list[Any],
    *,
    page_width: int,
    page_height: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        panel_index = int(panel.get("panel_index") or 0)
        panel_box = _coerce_pixel_box(panel.get("bbox"), width=page_width, height=page_height)
        if panel_box is None:
            panel_box = _coerce_pixel_box(panel.get("bbox_norm"), width=page_width, height=page_height)
        if panel_box is None:
            continue
        characters: list[dict[str, Any]] = []
        for character in panel.get("characters") or []:
            if not isinstance(character, dict):
                continue
            char_box = _coerce_pixel_box(character.get("bbox"), width=page_width, height=page_height)
            if char_box is None:
                char_box = _coerce_pixel_box(character.get("bbox_norm"), width=page_width, height=page_height)
            if char_box is None:
                continue
            characters.append({**character, "_pixel_box": char_box, "_panel_index": panel_index})
        contexts.append({**panel, "_pixel_box": panel_box, "_characters": characters, "_panel_index": panel_index})
    return contexts


def _select_character_reference(
    *,
    target_character: dict[str, Any],
    target_panel_index: int,
    panel_contexts: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    target_source_id = str(target_character.get("source_character_id") or "").strip()
    same_character_other_panel: list[dict[str, Any]] = []
    if target_source_id:
        for panel in panel_contexts:
            candidate_panel_index = int(panel.get("_panel_index") or 0)
            if candidate_panel_index == target_panel_index:
                continue
            for character in panel.get("_characters") or []:
                candidate_source_id = str(character.get("source_character_id") or "").strip()
                if candidate_source_id == target_source_id:
                    same_character_other_panel.append(character)

    if same_character_other_panel:
        return random.choice(same_character_other_panel), "same_character_random_other_panel"

    return target_character, "same_character_target_panel_fallback"



def _caption_panels(caption_payload: dict[str, Any]) -> list[Any]:
    panels = caption_payload.get("panels")
    if isinstance(panels, list):
        return panels
    return []


def _caption_page_size(caption_payload: dict[str, Any]) -> tuple[int, int] | None:
    page_size = caption_payload.get("page_size")
    if not isinstance(page_size, dict):
        return None
    try:
        width = int(page_size.get("width_px") or page_size.get("width") or 0)
        height = int(page_size.get("height_px") or page_size.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _panel_caption(
    panel: dict[str, Any],
    *,
    caption_field: str = "caption",
    caption_format: str = "text",
) -> str:
    value = panel.get(caption_field)
    if value is None:
        return ""
    if caption_format == "compact_json":
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return ""
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _attach_panel_tails(
    *,
    s3_client,
    bucket: str,
    annotations_prefix: str,
    chapter: str,
    page_id: str,
    panels: list[Any],
    page_width: int,
    page_height: int,
) -> None:
    """Read the magi_v3 annotation, group tail bboxes by enclosing panel, and
    write each panel["tails"] in-place. Safe no-op if the annotation cannot be
    loaded or no tails are present."""
    if not annotations_prefix or not chapter or not page_id:
        return

    chapter_suffixed = _chapter_storage_key(chapter)
    candidate_keys = [
        f"{annotations_prefix}/{chapter_suffixed}/{page_id}.jsonl",
    ]
    if chapter_suffixed != chapter:
        candidate_keys.append(f"{annotations_prefix}/{chapter}/{page_id}.jsonl")

    annotation_payload: dict[str, Any] | None = None
    for key in candidate_keys:
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
        except Exception:
            continue
        try:
            text = obj["Body"].read().decode("utf-8").strip()
            annotation_payload = json.loads(text.splitlines()[0]) if text else None
        except Exception:
            annotation_payload = None
        if annotation_payload is not None:
            break
    if not isinstance(annotation_payload, dict):
        return

    detections = annotation_payload.get("detections") if isinstance(annotation_payload.get("detections"), dict) else {}
    raw_tails = detections.get("tails") if isinstance(detections.get("tails"), list) else []
    if not raw_tails:
        return

    tail_records: list[tuple[tuple[int, int, int, int], tuple[float, float], dict[str, Any]]] = []
    for source_index, tail in enumerate(raw_tails):
        if not isinstance(tail, dict):
            continue
        box = _coerce_pixel_box(
            tail.get("bbox") or tail.get("bbox_norm"),
            width=page_width,
            height=page_height,
        )
        if box is None:
            continue
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        tail_records.append((box, center, {"source_tail_index": source_index}))

    if not tail_records:
        return

    for panel in panels:
        if not isinstance(panel, dict):
            continue
        # Respect tails already present (e.g. future caption runs).
        existing = panel.get("tails")
        if isinstance(existing, list) and existing:
            continue
        panel_box = _coerce_pixel_box(
            panel.get("bbox") or panel.get("bbox_norm"),
            width=page_width,
            height=page_height,
        )
        if panel_box is None:
            continue
        px0, py0, px1, py1 = panel_box
        panel_tails: list[dict[str, Any]] = []
        for tail_box, (cx, cy), meta in tail_records:
            if not (px0 <= cx <= px1 and py0 <= cy <= py1):
                continue
            panel_relative = _panel_relative_norm_box(
                tail_box,
                (float(px0), float(py0), float(px1), float(py1)),
                padded_width=max(1, px1 - px0),
                padded_height=max(1, py1 - py0),
            )
            panel_tails.append(
                {
                    "id": f"Tail {len(panel_tails) + 1}",
                    "tail_region_index": len(panel_tails),
                    "source_tail_index": meta.get("source_tail_index"),
                    "bbox": [int(round(v)) for v in tail_box],
                    "bbox_norm": [
                        tail_box[0] / page_width,
                        tail_box[1] / page_height,
                        tail_box[2] / page_width,
                        tail_box[3] / page_height,
                    ],
                    "panel_bbox_norm": panel_relative,
                }
            )
        if panel_tails:
            panel["tails"] = panel_tails


def _iter_panel_rows_for_caption(
    *,
    s3_client,
    bucket: str,
    caption_key: str,
    drawtoon: dict[str, Any],
    cache_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    caption_obj = s3_client.get_object(Bucket=bucket, Key=caption_key)
    caption_payload = json.loads(caption_obj["Body"].read().decode("utf-8"))
    if str(caption_payload.get("status") or "") != "ok":
        return [], {"skipped_status": 1}

    chapter = str(caption_payload.get("chapter") or "").strip()
    page_id = str(caption_payload.get("page_id") or "").strip()
    panels = _caption_panels(caption_payload)
    if not chapter or not page_id or not panels:
        return [], {"skipped_no_panels": 1}

    page_key = _caption_page_key(caption_payload, drawtoon)
    # Always measure dimensions from the *active* page (pages_prefix), not the
    # caption's cached page_size — text_removed pages can be a few pixels off
    # from the filtered originals used at caption time, which makes the
    # manifest's crop_box / target_width drift and trips the loader's exact
    # dimension check.
    page_bytes = s3_client.get_object(Bucket=bucket, Key=page_key)["Body"].read()
    with Image.open(io.BytesIO(page_bytes)) as page_image:
        page_width, page_height = page_image.size

    # Tails from MAGI v3 inference live in detections.tails, which the old
    # caption runs do not surface. Pull them straight from the annotation file
    # and attach to whichever panel encloses the tail centre. This is what
    # makes the silver tail mask show up in the layout-control image.
    _attach_panel_tails(
        s3_client=s3_client,
        bucket=bucket,
        annotations_prefix=str(drawtoon.get("annotations_prefix") or "").strip().strip("/"),
        chapter=chapter,
        page_id=page_id,
        panels=panels,
        page_width=page_width,
        page_height=page_height,
    )
    target_multiple = int(drawtoon.get("target_multiple") or 16)
    caption_field = str(drawtoon.get("caption_field") or "caption").strip()
    caption_format = str(drawtoon.get("caption_format") or "text").strip().lower()
    panel_contexts = _panel_character_contexts(panels, page_width=page_width, page_height=page_height)
    rows: list[dict[str, Any]] = []
    stats = {
        "pages": 1,
        "panels": 0,
        "characters": 0,
        "skipped_panels": 0,
        "target_panel_ref_fallbacks": 0,
    }

    for panel in panels:
        if not isinstance(panel, dict):
            continue
        panel_index = int(panel.get("panel_index") or 0)
        panel_box = _coerce_pixel_box(panel.get("bbox"), width=page_width, height=page_height)
        if panel_box is None:
            panel_box = _coerce_pixel_box(panel.get("bbox_norm"), width=page_width, height=page_height)
        caption = _panel_caption(panel, caption_field=caption_field, caption_format=caption_format)
        if panel_box is None or not caption:
            stats["skipped_panels"] += 1
            continue

        px0, py0, px1, py1 = panel_box
        crop_box = (int(math.floor(px0)), int(math.floor(py0)), int(math.ceil(px1)), int(math.ceil(py1)))
        target_box = tuple(float(value) for value in crop_box)
        padded_width = int(math.ceil(max(1, crop_box[2] - crop_box[0]) / target_multiple) * target_multiple)
        padded_height = int(math.ceil(max(1, crop_box[3] - crop_box[1]) / target_multiple) * target_multiple)
        control_paths: list[dict[str, Any]] = []
        character_layout_items: list[dict[str, Any]] = []
        for character in panel.get("characters") or []:
            if len(control_paths) >= MAX_LAYOUT_CHARACTER_REFS:
                break
            if not isinstance(character, dict):
                continue
            char_box = _coerce_pixel_box(character.get("bbox"), width=page_width, height=page_height)
            if char_box is None:
                char_box = _coerce_pixel_box(character.get("bbox_norm"), width=page_width, height=page_height)
            if char_box is None:
                continue
            char_box_norm = _panel_relative_norm_box(
                char_box,
                target_box,
                padded_width=padded_width,
                padded_height=padded_height,
            )
            if char_box_norm is None:
                continue
            character_index = len(control_paths)
            ref_character, ref_policy = _select_character_reference(
                target_character={**character, "_pixel_box": char_box, "_panel_index": panel_index},
                target_panel_index=panel_index,
                panel_contexts=panel_contexts,
            )
            rx0, ry0, rx1, ry1 = ref_character["_pixel_box"]
            if ref_policy == "same_character_target_panel_fallback":
                stats["target_panel_ref_fallbacks"] += 1
            control_paths.append(
                {
                    "image": f"s3://{bucket}/{page_key}",
                    "crop_box": [
                        int(math.floor(rx0)),
                        int(math.floor(ry0)),
                        int(math.ceil(rx1)),
                        int(math.ceil(ry1)),
                    ],
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
            stats["characters"] += 1

        if not control_paths:
            stats["skipped_panels"] += 1
            continue

        _, layout_metadata = _build_panel_layout_control(
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
                "source_annotation": (
                    f"s3://{bucket}/{drawtoon['annotations_prefix'].rstrip('/')}/{_chapter_storage_key(chapter)}/{page_id}.jsonl"
                ),
                "target_panel": {
                    "image": f"s3://{bucket}/{page_key}",
                    "crop_box": list(crop_box),
                    "pad_multiple": target_multiple,
                },
                "target_width": int(padded_width),
                "target_height": int(padded_height),
                "caption": caption.rstrip(".") + ".",
                "character_count": len(control_paths),
                "controls": {
                    "layout_control": layout_metadata,
                    "character_ref_paths": control_paths,
                    "has_previous_control": False,
                    "character_ref_policy": "same_character_random_other_panel_else_target_panel",
                },
            }
        )
        stats["panels"] += 1
    return rows, stats


@app.function(
    image=image_4b,
    volumes={DATASET_CACHE_ROOT: dataset_cache_volume},
    secrets=[aws_secret],
    timeout=24 * 60 * 60,
    cpu=2,
    memory=8192,
)
def plan_drawtoon_panel_cache(
    config_path: str,
    overrides: dict[str, Any] | None = None,
    *,
    shard_count: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    parsed_config = _read_yaml(config_path)
    if not _config_needs_drawtoon_cache(parsed_config):
        return {"enabled": False, "resolved_config_path": config_path}

    drawtoon = _drawtoon_config_from_parsed(parsed_config, overrides)
    bucket = drawtoon["bucket"]
    caption_root = f"{drawtoon['captions_prefix'].rstrip('/')}/{drawtoon['caption_run'].strip('/')}"
    cache_root = Path(DATASET_CACHE_ROOT) / "drawtoon_panel" / _drawtoon_cache_key(drawtoon)
    manifest_path = cache_root / "manifest.jsonl"
    summary_path = cache_root / "summary.json"
    resolved_config_path = cache_root / "resolved_config.yaml"
    success_path = cache_root / "_SUCCESS"
    if overwrite and cache_root.exists():
        shutil.rmtree(cache_root)

    include_re = re.compile(drawtoon["include_chapter_regex"]) if drawtoon["include_chapter_regex"] else None
    s3_client = boto3.client("s3")
    caption_keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=caption_root.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key.endswith(".json"):
                continue
            rel = key[len(caption_root.rstrip("/") + "/") :]
            if rel.startswith("_") or len(rel.split("/")) != 2:
                continue
            chapter = _caption_chapter_from_key(caption_root, key)
            if include_re and not include_re.search(chapter):
                continue
            caption_keys.append(key)

    caption_keys.sort(key=lambda key: _sort_caption_key(caption_root, key))
    if drawtoon["max_pages"] > 0:
        caption_keys = caption_keys[: int(drawtoon["max_pages"])]
    if not caption_keys:
        raise RuntimeError(f"No Drawtoon caption JSONs found under s3://{bucket}/{caption_root}/")

    if success_path.exists() and manifest_path.exists() and resolved_config_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        if (
            isinstance(summary, dict)
            and summary.get("source") == drawtoon
            and int(summary.get("caption_key_count") or 0) == len(caption_keys)
            and int(summary.get("row_count") or 0) > 0
        ):
            row_count = int(summary.get("row_count") or 0)
            write_resolved_drawtoon_cache_config(
                parsed_config=parsed_config,
                manifest_path=manifest_path,
                cache_root=cache_root,
                row_count=row_count,
                output_path=resolved_config_path,
            )
            dataset_cache_volume.commit()
            return {
                "enabled": True,
                "ready": True,
                "drawtoon": drawtoon,
                "cache_root": str(cache_root),
                "manifest_path": str(manifest_path),
                "resolved_config_path": str(resolved_config_path),
                "summary": summary,
                "row_count": row_count,
            }
        shutil.rmtree(cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    work_dir = cache_root / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_shards = max(1, int(shard_count or 0))
    if shard_count <= 0:
        effective_shards = min(128, max(1, math.ceil(len(caption_keys) / 500)))
    effective_shards = min(effective_shards, len(caption_keys))
    shards: list[dict[str, Any]] = []
    for shard_index in range(effective_shards):
        keys = caption_keys[shard_index::effective_shards]
        key_file = work_dir / f"shard_{shard_index:04d}.keys.json"
        key_file.write_text(json.dumps(keys, ensure_ascii=False) + "\n", encoding="utf-8")
        shards.append(
            {
                "shard_index": shard_index,
                "key_file": str(key_file),
                "output_jsonl": str(work_dir / f"shard_{shard_index:04d}.jsonl"),
                "stats_json": str(work_dir / f"shard_{shard_index:04d}.stats.json"),
                "bucket": bucket,
                "drawtoon": drawtoon,
                "cache_root": str(cache_root),
            }
        )

    dataset_cache_volume.commit()
    return {
        "enabled": True,
        "ready": False,
        "drawtoon": drawtoon,
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "resolved_config_path": str(resolved_config_path),
        "caption_key_count": len(caption_keys),
        "shard_count": len(shards),
        "shards": shards,
    }


@app.function(
    image=image_4b,
    volumes={DATASET_CACHE_ROOT: dataset_cache_volume},
    secrets=[aws_secret],
    timeout=12 * 60 * 60,
    cpu=4,
    memory=16384,
)
def build_drawtoon_panel_cache_shard(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        dataset_cache_volume.reload()
    except Exception:
        pass
    key_file = Path(spec["key_file"])
    output_jsonl = Path(spec["output_jsonl"])
    stats_json = Path(spec["stats_json"])
    cache_root = Path(spec["cache_root"])
    bucket = str(spec["bucket"])
    drawtoon = dict(spec["drawtoon"])
    caption_keys = json.loads(key_file.read_text(encoding="utf-8"))
    s3_client = boto3.client("s3")

    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "caption_keys": len(caption_keys),
        "pages": 0,
        "panels": 0,
        "characters": 0,
        "skipped_panels": 0,
        "skipped_status": 0,
        "skipped_no_panels": 0,
        "errors": 0,
    }
    error_samples: list[dict[str, str]] = []

    for caption_key in caption_keys:
        try:
            page_rows, page_stats = _iter_panel_rows_for_caption(
                s3_client=s3_client,
                bucket=bucket,
                caption_key=str(caption_key),
                drawtoon=drawtoon,
                cache_root=cache_root,
            )
            rows.extend(page_rows)
            for key, value in page_stats.items():
                stats[key] = int(stats.get(key, 0)) + int(value)
        except Exception as exc:
            stats["errors"] += 1
            if len(error_samples) < 20:
                error_samples.append({"caption_key": str(caption_key), "error": str(exc)})

    _jsonl_write(output_jsonl, rows)
    stats_payload = {
        "shard_index": int(spec["shard_index"]),
        "stats": stats,
        "error_samples": error_samples,
        "output_jsonl": str(output_jsonl),
    }
    stats_json.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_cache_volume.commit()
    return stats_payload


@app.function(
    image=image_4b,
    volumes={DATASET_CACHE_ROOT: dataset_cache_volume},
    secrets=[aws_secret],
    timeout=24 * 60 * 60,
    cpu=2,
    memory=8192,
)
def finalize_drawtoon_panel_cache(config_path: str, plan: dict[str, Any]) -> dict[str, Any]:
    try:
        dataset_cache_volume.reload()
    except Exception:
        pass

    parsed_config = _read_yaml(config_path)
    cache_root = Path(plan["cache_root"])
    manifest_path = Path(plan["manifest_path"])
    resolved_config_path = Path(plan["resolved_config_path"])
    summary_path = cache_root / "summary.json"
    success_path = cache_root / "_SUCCESS"

    part_paths = sorted((cache_root / "_work").glob("shard_*.jsonl"))
    if not part_paths:
        raise RuntimeError(f"No Drawtoon cache shards found under {cache_root / '_work'}")

    rows: list[dict[str, Any]] = []
    for part_path in part_paths:
        with part_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    raw_row_count = len(rows)
    if raw_row_count <= 0:
        raise RuntimeError(f"Drawtoon cache built zero manifest rows at {manifest_path}")

    drawtoon = plan.get("drawtoon") or {}
    excluded_sample_ids = load_drawtoon_excluded_sample_ids(
        str(drawtoon.get("exclude_sample_ids_path") or "")
    )
    if excluded_sample_ids:
        rows = [
            row for row in rows
            if _drawtoon_sample_identity(row) not in excluded_sample_ids
        ]
    excluded_match_count = raw_row_count - len(rows)

    row_count = len(rows)
    if row_count <= 0:
        raise RuntimeError(f"Drawtoon exclude list removed all training rows at {manifest_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _jsonl_write(manifest_path, rows)

    shard_stats = []
    totals: dict[str, int] = {}
    for stats_path in sorted((cache_root / "_work").glob("shard_*.stats.json")):
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        shard_stats.append(payload)
        for key, value in (payload.get("stats") or {}).items():
            totals[key] = int(totals.get(key, 0)) + int(value)

    parsed_config["drawtoon_cache"] = {
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "raw_row_count": raw_row_count,
        "row_count": row_count,
        "exclude_sample_ids_path": str(drawtoon.get("exclude_sample_ids_path") or ""),
        "excluded_sample_id_count": len(excluded_sample_ids),
        "excluded_manifest_match_count": excluded_match_count,
        "source": drawtoon,
    }
    write_resolved_drawtoon_cache_config(
        parsed_config=parsed_config,
        manifest_path=manifest_path,
        cache_root=cache_root,
        row_count=row_count,
        output_path=resolved_config_path,
    )

    summary = {
        "schema_version": 1,
        "cache_type": "drawtoon_panel_modal_cache",
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "resolved_config_path": str(resolved_config_path),
        "raw_row_count": raw_row_count,
        "row_count": row_count,
        "exclude_sample_ids_path": str(drawtoon.get("exclude_sample_ids_path") or ""),
        "excluded_sample_id_count": len(excluded_sample_ids),
        "excluded_manifest_match_count": excluded_match_count,
        "caption_key_count": int(plan.get("caption_key_count") or 0),
        "source": drawtoon,
        "totals": totals,
        "shard_count": len(part_paths),
        "shards": shard_stats,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success_path.write_text(json.dumps({"ok": True, "row_count": row_count}, sort_keys=True) + "\n", encoding="utf-8")
    dataset_cache_volume.commit()
    return {
        "enabled": True,
        "ready": True,
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "resolved_config_path": str(resolved_config_path),
        "row_count": row_count,
        "summary": summary,
    }


def prepare_drawtoon_panel_cache_for_launch(
    *,
    config_path: str,
    overrides: dict[str, Any],
    shard_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    plan = plan_drawtoon_panel_cache.remote(
        config_path,
        overrides,
        shard_count=shard_count,
        overwrite=overwrite,
    )
    if not plan.get("enabled") or plan.get("ready"):
        return plan

    print(
        "🧩 Building Drawtoon panel Modal cache: "
        f"{plan['caption_key_count']} page captions across {plan['shard_count']} CPU shards",
        flush=True,
    )
    completed_shards = 0
    for _ in build_drawtoon_panel_cache_shard.map(plan["shards"], order_outputs=False):
        completed_shards += 1
        if completed_shards % 25 == 0 or completed_shards == plan["shard_count"]:
            print(f"🧩 Drawtoon cache shards complete: {completed_shards}/{plan['shard_count']}", flush=True)
    final = finalize_drawtoon_panel_cache.remote(config_path, plan)
    print(
        "🧩 Drawtoon cache ready: "
        f"{final['row_count']} panel rows at {final['manifest_path']}",
        flush=True,
    )
    return final

@app.cls(
    image=image_4b,
    gpu=MODAL_SINGLE_GPU,
    timeout=24 * 60 * 60,
    volumes={
        MOUNT_DIR: model_volume,
        DATASET_CACHE_ROOT: dataset_cache_volume,
        HF_HOME: hf_cache_volume,
        S3_MOUNT_ROOT: s3_mount,
    },
    enable_memory_snapshot=True,  # Cache heavy imports and initialization
    secrets=[aws_secret, hf_secret],
)
class FluxTrainer:
    @modal.enter(snap=True)  # This gets captured in memory snapshot
    def setup(self):
        import sys
        import os
        from pathlib import Path

        if os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"]

        sys.path.insert(0, "/root/ai-toolkit")
        
        # Heavy imports - these get cached in memory snapshot
        import torch
        import diffusers
        import transformers
        from toolkit.job import get_job
        
        # FLUX.2-klein specific fix: Set optimum-quanto for proper quantization
        os.environ["QUANTO_BYPASS_OBJECT_COPY"] = "1"
        
        # Store references for use in methods
        self.get_job = get_job
        self.Path = Path
        
        print("✅ Heavy libraries loaded and cached in memory snapshot")

    def _run(self, cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    def _copy_caption_stage(self, image_dir: Path, caption_dir: Path, target_dir: Path) -> None:
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Expected matching image stage at {image_dir} for caption stage {caption_dir}"
            )
        print(f"📥 Staging images from {image_dir} to {target_dir}")
        self._run(
            [
                "rsync",
                "-a",
                "--delete",
                f"{image_dir.as_posix().rstrip('/')}/",
                f"{target_dir.as_posix().rstrip('/')}/",
            ]
        )
        print(f"📝 Overlaying captions from {caption_dir} into {target_dir}")
        self._run(
            [
                "rsync",
                "-a",
                "--include=*/",
                "--include=*.txt",
                "--include=*.json",
                "--exclude=*",
                f"{caption_dir.as_posix().rstrip('/')}/",
                f"{target_dir.as_posix().rstrip('/')}/",
            ]
        )

    def _sync_file_to_s3(self, source_path: Path, destination_uri: str) -> None:
        if not source_path.exists():
            return
        self._run(
            [
                "aws",
                "s3",
                "cp",
                str(source_path),
                destination_uri,
                "--only-show-errors",
                "--no-progress",
            ]
        )

    def _sync_dir_to_s3(self, source_dir: Path, destination_uri: str) -> None:
        if not source_dir.exists():
            return
        self._run(
            [
                "aws",
                "s3",
                "sync",
                str(source_dir),
                destination_uri,
                "--only-show-errors",
                "--no-progress",
            ]
        )

    def _export_training_state_to_s3(
        self,
        *,
        job_name: str,
        model_id: str,
        save_root: Path,
        validate_root: Path,
        save_path: Path | None = None,
        include_all_checkpoints: bool = False,
    ) -> None:
        model_prefix = f"{S3_MODELS_PREFIX}/{model_id}"

        self._sync_file_to_s3(save_root / "config.yaml", s3_uri(model_prefix, "config.yaml"))
        self._sync_file_to_s3(save_root / "optimizer.pt", s3_uri(model_prefix, "optimizer.pt"))

        if save_path is not None and save_path.exists():
            if save_path.is_dir():
                self._sync_dir_to_s3(
                    save_path,
                    s3_uri(model_prefix, "checkpoints", save_path.name),
                )
            else:
                self._sync_file_to_s3(
                    save_path,
                    s3_uri(model_prefix, "checkpoints", save_path.name),
                )

        if include_all_checkpoints:
            for path in sorted(save_root.glob(f"{job_name}*")):
                if path.name in {"samples", "peft_adapter", "config.yaml", "optimizer.pt"}:
                    continue
                if path.is_dir():
                    self._sync_dir_to_s3(
                        path,
                        s3_uri(model_prefix, "checkpoints", path.name),
                    )
                else:
                    self._sync_file_to_s3(
                        path,
                        s3_uri(model_prefix, "checkpoints", path.name),
                    )

            peft_dir = save_root / "peft_adapter"
            if peft_dir.exists():
                self._sync_dir_to_s3(
                    peft_dir,
                    s3_uri(model_prefix, "final", "peft_adapter"),
                )

        if validate_root.exists():
            self._sync_dir_to_s3(
                validate_root,
                s3_uri(model_prefix, "validate"),
            )

    def _attach_s3_export_hooks(self, job, *, model_id: str, job_name: str) -> None:
        for process in job.process:
            original_post_save_hook = process.post_save_hook

            def patched_post_save_hook(proc_self, save_path, original_hook=original_post_save_hook):
                original_hook(save_path)
                model_volume.commit()
                self._export_training_state_to_s3(
                    job_name=job_name,
                    model_id=model_id,
                    save_root=self.Path(proc_self.save_root),
                    validate_root=self.Path(MOUNT_DIR) / "validate" / job_name,
                    save_path=self.Path(save_path),
                    include_all_checkpoints=False,
                )

            process.post_save_hook = MethodType(patched_post_save_hook, process)

    def _attach_manifest_validation_refresh_hook(
        self,
        job,
        *,
        manifest_paths: list[Path],
        description_only_count: int = 5,
        next_page_first_panel_count: int = 5,
        within_page_next_panel_count: int = 10,
        character_panel_count: int = 8,
        character_panel_fixed_count: int | None = None,
        bucket_tolerance: int = 16,
    ) -> None:
        from toolkit.config_modules import SampleItem

        for process in job.process:
            original_sample = process.sample

            def patched_sample(
                proc_self,
                step=None,
                is_first=False,
                original_method=original_sample,
            ):
                if not is_first:
                    sample_dicts = build_multi_manifest_validation_samples(
                        manifest_paths=manifest_paths,
                        description_only_count=description_only_count,
                        next_page_first_panel_count=next_page_first_panel_count,
                        within_page_next_panel_count=within_page_next_panel_count,
                        character_panel_count=character_panel_count,
                        character_panel_fixed_count=character_panel_fixed_count,
                        bucket_tolerance=bucket_tolerance,
                    )
                    proc_self.sample_config.samples = [
                        SampleItem(proc_self.sample_config, **item)
                        for item in sample_dicts
                    ]
                    proc_self.cache_sample_prompts()
                return original_method(step=step, is_first=is_first)

            process.sample = MethodType(patched_sample, process)
    
    @modal.method()
    def train_flux_lora(
        self,
        config_path: str = f"{REMOTE_CONFIGS_DIR}/mngrm12_klein4b_dataset_jpg_r16_768x1024_s12000_v1.yaml",
        dataset_subpath: str | None = None,
        model_id: str | None = None,
        target_epochs: int = 3,
        max_train_steps: int = 0,
        sample_every: int = 0,
        staged_dataset_dir: str | None = None,
        image_count: int | None = None,
        caption_count: int | None = None,
        matched_count: int | None = None,
        sample_prompts: list[str] | None = None,
    ):
        """Run FLUX LoRA training using ai-toolkit - clean implementation with no fallbacks"""
        config_file = self.Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(
                f"Expected config at {config_path}. "
                f"Provide it via add_local_dir(...) in the image."
            )

        ai_toolkit_config = self.Path("/root/ai-toolkit/config/flux_klein_train.yaml")
        ai_toolkit_config.parent.mkdir(parents=True, exist_ok=True)
        parsed_config = yaml.safe_load(config_file.read_text())
        process_config = parsed_config.get("config", {}).get("process", [{}])[0]
        process_config["training_folder"] = MOUNT_DIR
        process_config["device"] = "cuda"
        manifest_training = is_manifest_training(process_config)
        if model_id:
            parsed_config.setdefault("config", {})
            parsed_config["config"]["name"] = model_id
            process_config["name"] = model_id
            parsed_config.setdefault("meta", {})
            parsed_config["meta"]["name"] = model_id
        if not manifest_training and staged_dataset_dir is None:
            raise RuntimeError("dataset_subpath staging has been removed; use manifest dataset configs")
        try:
            dataset_cache_volume.reload()
        except Exception:
            pass
        if staged_dataset_dir is not None:
            staged_dataset_dir = self.Path(staged_dataset_dir)
        process_config["sample_copy_root"] = f"{MOUNT_DIR}/validate"

        manifest_datasets: list[dict] = []
        manifest_paths: list[Path] = []
        validation_bucket_tolerance = 16
        if manifest_training:
            manifest_state = prepare_manifest_training_datasets(
                process_config,
                dataset_subpath=dataset_subpath,
                staged_dataset_dir=staged_dataset_dir,
            )
            manifest_datasets = manifest_state["manifest_datasets"]
            manifest_paths = manifest_state["manifest_paths"]
            validation_bucket_tolerance = int(manifest_state["validation_bucket_tolerance"])
            staged_dataset_dir = manifest_state["staged_dataset_dir"]
            total_samples = int(manifest_state["total_samples"])
            image_count = total_samples
            caption_count = total_samples
            matched_count = total_samples
            sample_config = process_config.setdefault("sample", {})
            configure_manifest_validation_samples(
                sample_config,
                manifest_paths=manifest_paths,
                validation_bucket_tolerance=validation_bucket_tolerance,
                ddp_world_size=1,
            )
        else:
            for dataset in process_config.get("datasets", []):
                apply_runtime_dataset_defaults(
                    dataset,
                    folder_path=str(staged_dataset_dir),
                )
            if image_count is None or caption_count is None or matched_count is None:
                image_count, caption_count, matched_count = count_training_pairs(staged_dataset_dir)
            if matched_count == 0:
                raise RuntimeError(
                    f"No matched image/caption pairs found in staged dataset {staged_dataset_dir} "
                    f"(images={image_count}, captions={caption_count})"
                )

        manifest_epoch_stats = None
        if manifest_training:
            try:
                steps_per_epoch, manifest_epoch_stats = compute_multi_manifest_steps_per_epoch(
                    process_config=process_config,
                    manifest_dataset_configs=manifest_datasets,
                    world_size=1,
                )
            except Exception as exc:
                print(
                    "[epoch-estimate] Falling back to naive sample-count math for manifest training: "
                    f"{exc}"
                )
                steps_per_epoch = compute_steps_per_epoch(
                    num_train_items=matched_count,
                    process_config=process_config,
                    world_size=1,
                )
        else:
            steps_per_epoch = compute_steps_per_epoch(
                num_train_items=matched_count,
                process_config=process_config,
                world_size=1,
            )
        total_steps = resolve_training_steps(
            process_config,
            steps_per_epoch=steps_per_epoch,
            target_epochs=target_epochs,
            max_train_steps=max_train_steps,
        )
        process_config.setdefault("train", {})
        process_config["train"]["steps"] = total_steps
        process_config["train"]["force_first_sample"] = False
        process_config["train"]["skip_first_sample"] = True
        process_config.setdefault("save", {})
        configured_save_every = int(process_config["save"].get("save_every", 0) or 0)
        process_config["save"]["save_every"] = configured_save_every if configured_save_every > 0 else steps_per_epoch
        configured_keep = int(process_config["save"].get("max_step_saves_to_keep", 0) or 0)
        process_config["save"]["max_step_saves_to_keep"] = configured_keep if configured_keep > 0 else 5
        sample_config = process_config.setdefault("sample", {})
        if not manifest_training:
            sample_config["sample_every"] = 31
            sample_config["prompts"] = dataset_sample_prompts(str(staged_dataset_dir), limit=10)
        if sample_every > 0:
            sample_config["sample_every"] = int(sample_every)
        ai_toolkit_config.write_text(yaml.safe_dump(parsed_config, sort_keys=False))
        parsed_config = yaml.safe_load(ai_toolkit_config.read_text())

        os.makedirs(MOUNT_DIR, exist_ok=True)

        process_config = parsed_config.get("config", {}).get("process", [{}])[0]
        job_name = process_config.get("name") or parsed_config.get("config", {}).get("name")
        if not job_name:
            raise RuntimeError("Training config is missing the job name")
        if not model_id:
            model_id = job_name
        model_name = process_config.get("model", {}).get("name_or_path", "unknown-model")
        print(f"🚀 Starting {model_name} training")
        print("✅ Heavy libraries already cached from setup()")
        print(f"✅ Loading config from: {config_file}")
        if manifest_training and manifest_paths:
            print(f"📚 Manifest datasets: {len(manifest_paths)}")
            for manifest_path in manifest_paths:
                print(f"   - {manifest_path}")
        else:
            print(f"📚 Staged dataset: {staged_dataset_dir}")
        print(
            f"🧮 Training pairs: {matched_count} "
            f"(images={image_count}, captions={caption_count})"
        )
        print(
            f"📏 Epochs: {target_epochs}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}"
        )
        if max_train_steps > 0:
            print(f"🧪 Actual-model smoke step cap: {max_train_steps}")
        if manifest_epoch_stats is not None:
            print(
                "📦 Manifest batch estimate: "
                f"emitted_microbatches={manifest_epoch_stats['emitted_batches']} "
                f"per_rank_microbatches={manifest_epoch_stats['per_rank_microbatches']} "
                f"avg_bs={manifest_epoch_stats['avg_batch_size']:.2f} "
                f"min_bs={manifest_epoch_stats['min_batch_size']} "
                f"max_bs={manifest_epoch_stats['max_batch_size']}"
            )
        print(f"🖼️  Validation samples every {sample_config['sample_every']} steps")
        print("🖼️  No validation sample before training start")
        print(f"💾 Checkpoints every {process_config['save']['save_every']} steps")
        print(f"💾 Training outputs will be saved to: {MOUNT_DIR}")
        print(f"☁️  S3 model prefix: {s3_uri(S3_MODELS_PREFIX, model_id)}")

        job = self.get_job(str(ai_toolkit_config), None) 
        job.config['process'][0]['training_folder'] = MOUNT_DIR
        self._attach_s3_export_hooks(job, model_id=model_id, job_name=job_name)
        if manifest_training:
            self._attach_manifest_validation_refresh_hook(
                job,
                manifest_paths=manifest_paths,
                description_only_count=5,
                next_page_first_panel_count=5,
                within_page_next_panel_count=10,
                character_panel_count=8,
                bucket_tolerance=validation_bucket_tolerance,
            )
        job.run()
        job.cleanup()
        
        print("✅ Training completed successfully!")

        network_config = process_config.get("network", {})
        model_config = process_config.get("model", {})
        is_lora_run = bool(network_config)
        raw_lora_path = None
        adapter_dir = None

        if is_lora_run:
            raw_lora_path = find_raw_lora_path(job_name)
            adapter_dir = raw_lora_path.parent / "peft_adapter"

            build_peft_adapter(
                source_path=raw_lora_path,
                output_dir=adapter_dir,
                base_model=model_config.get("name_or_path", "black-forest-labs/FLUX.2-klein-base-4B"),
                rank=network_config.get("linear"),
                alpha=network_config.get("linear_alpha"),
            )
            validate_peft_adapter_dir(adapter_dir)
            print(f"✅ PEFT adapter prepared at {adapter_dir}")
        else:
            print("✅ Full fine-tune run detected; skipping PEFT adapter export")

        self._export_training_state_to_s3(
            job_name=job_name,
            model_id=model_id,
            save_root=self.Path(MOUNT_DIR) / job_name,
            validate_root=self.Path(MOUNT_DIR) / "validate" / job_name,
            include_all_checkpoints=True,
        )
        print(f"✅ Exported checkpoints and validation images to {s3_uri(S3_MODELS_PREFIX, model_id)}")
        
        return (
            f"Training completed. Live resume state is in Modal volume 'flux-lora-models'. "
            f"Artifacts mirrored to {s3_uri(S3_MODELS_PREFIX, model_id)}."
        )


@app.function(
    image=image_4b,
    gpu=MODAL_DDP8_GPU,
    timeout=24 * 60 * 60,
    cpu=16,
    memory=131072,
    volumes={
        HF_HOME: hf_cache_volume,
        S3_MOUNT_ROOT: s3_mount,
        DATASET_CACHE_ROOT: dataset_cache_volume,
    },
    secrets=[aws_secret, hf_secret],
)
def train_flux_lora_ddp8(
    config_path: str = f"{REMOTE_CONFIGS_DIR}/mngrm12_klein4b_dataset_jpg_r16_768x1024_s12000_v1.yaml",
    dataset_subpath: str | None = None,
    model_id: str | None = None,
    target_epochs: int = 3,
    ddp_smoke: bool = False,
    max_train_steps: int = 0,
    sample_every: int = 0,
    staged_dataset_dir: str | None = None,
    image_count: int | None = None,
    caption_count: int | None = None,
    matched_count: int | None = None,
):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Expected config at {config_path}. "
            f"Provide it via add_local_dir(...) in the image."
        )

    ai_toolkit_config = Path("/root/ai-toolkit/config/flux_klein_train_ddp.yaml")
    ai_toolkit_config.parent.mkdir(parents=True, exist_ok=True)
    parsed_config = yaml.safe_load(config_file.read_text())
    process_config = parsed_config.get("config", {}).get("process", [{}])[0]
    training_output_root = Path(os.environ.get("LINEART2_TRAINING_OUTPUT_ROOT", DDP_OUTPUT_ROOT))
    training_output_root.mkdir(parents=True, exist_ok=True)
    process_config["training_folder"] = str(training_output_root)
    process_config["device"] = "cuda"
    manifest_training = is_manifest_training(process_config)
    if model_id:
        parsed_config.setdefault("config", {})
        parsed_config["config"]["name"] = model_id
        process_config["name"] = model_id
        parsed_config.setdefault("meta", {})
        parsed_config["meta"]["name"] = model_id

    try:
        dataset_cache_volume.reload()
    except Exception:
        pass

    if staged_dataset_dir is not None:
        staged_dataset_dir = Path(staged_dataset_dir)
    process_config["sample_copy_root"] = str(training_output_root / "validate")

    manifest_datasets: list[dict] = []
    manifest_paths: list[Path] = []
    validation_bucket_tolerance = 16
    if manifest_training:
        manifest_state = prepare_manifest_training_datasets(
            process_config,
            dataset_subpath=dataset_subpath,
            staged_dataset_dir=staged_dataset_dir,
        )
        manifest_datasets = manifest_state["manifest_datasets"]
        manifest_paths = manifest_state["manifest_paths"]
        validation_bucket_tolerance = int(manifest_state["validation_bucket_tolerance"])
        staged_dataset_dir = manifest_state["staged_dataset_dir"]
        total_samples = int(manifest_state["total_samples"])
        image_count = total_samples
        caption_count = total_samples
        matched_count = total_samples
        sample_config = process_config.setdefault("sample", {})
        configure_manifest_validation_samples(
            sample_config,
            manifest_paths=manifest_paths,
            validation_bucket_tolerance=validation_bucket_tolerance,
            ddp_world_size=DDP_WORLD_SIZE_8,
        )
    else:
        if staged_dataset_dir is None:
            raise RuntimeError("staged_dataset_dir is required for non-manifest DDP runs")
        for dataset in process_config.get("datasets", []):
            apply_runtime_dataset_defaults(
                dataset,
                folder_path=str(staged_dataset_dir),
            )
        if image_count is None or caption_count is None or matched_count is None:
            image_count, caption_count, matched_count = count_training_pairs(staged_dataset_dir)
        if matched_count == 0:
            raise RuntimeError(
                f"No matched image/caption pairs found in staged dataset {staged_dataset_dir} "
                f"(images={image_count}, captions={caption_count})"
            )

    manifest_epoch_stats = None
    if manifest_training:
        try:
            steps_per_epoch, manifest_epoch_stats = compute_multi_manifest_steps_per_epoch(
                process_config=process_config,
                manifest_dataset_configs=manifest_datasets,
                world_size=DDP_WORLD_SIZE_8,
            )
        except Exception as exc:
            print(
                "[epoch-estimate] Falling back to naive sample-count math for manifest training: "
                f"{exc}"
            )
            steps_per_epoch = compute_steps_per_epoch(
                num_train_items=matched_count,
                process_config=process_config,
                world_size=DDP_WORLD_SIZE_8,
            )
    else:
        steps_per_epoch = compute_steps_per_epoch(
            num_train_items=matched_count,
            process_config=process_config,
            world_size=DDP_WORLD_SIZE_8,
        )
    total_steps = resolve_training_steps(
        process_config,
        steps_per_epoch=steps_per_epoch,
        target_epochs=target_epochs,
        max_train_steps=max_train_steps,
    )
    process_config.setdefault("train", {})
    process_config["train"]["steps"] = total_steps
    process_config["train"]["force_first_sample"] = False
    process_config["train"]["skip_first_sample"] = True
    process_config.setdefault("save", {})
    configured_save_every = int(process_config["save"].get("save_every", 0) or 0)
    process_config["save"]["save_every"] = configured_save_every if configured_save_every > 0 else steps_per_epoch
    configured_keep = int(process_config["save"].get("max_step_saves_to_keep", 0) or 0)
    process_config["save"]["max_step_saves_to_keep"] = configured_keep if configured_keep > 0 else 5
    sample_config = process_config.setdefault("sample", {})
    if not manifest_training:
        sample_config["sample_every"] = 31
        sample_config["prompts"] = dataset_sample_prompts(str(staged_dataset_dir), limit=10)
    if sample_every > 0:
        sample_config["sample_every"] = int(sample_every)
    ai_toolkit_config.write_text(yaml.safe_dump(parsed_config, sort_keys=False))
    parsed_config = yaml.safe_load(ai_toolkit_config.read_text())

    training_output_root.mkdir(parents=True, exist_ok=True)
    process_config = parsed_config.get("config", {}).get("process", [{}])[0]
    job_name = process_config.get("name") or parsed_config.get("config", {}).get("name")
    if not job_name:
        raise RuntimeError("Training config is missing the job name")
    if not model_id:
        model_id = job_name

    print(f"🚀 Starting distributed training for {job_name}")
    if manifest_training and manifest_paths:
        print(f"📚 Manifest datasets: {len(manifest_paths)}")
        for manifest_path in manifest_paths:
            print(f"   - {manifest_path}")
    else:
        print(f"📚 Staged dataset: {staged_dataset_dir}")
    print(
        f"🧮 Training pairs: {matched_count} "
        f"(images={image_count}, captions={caption_count})"
    )
    print(
        f"📏 Epochs: {target_epochs}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}"
    )
    if max_train_steps > 0:
        print(f"🧪 Actual-model smoke step cap: {max_train_steps}")
    if manifest_epoch_stats is not None:
        print(
            "📦 Manifest batch estimate: "
            f"emitted_microbatches={manifest_epoch_stats['emitted_batches']} "
            f"per_rank_microbatches={manifest_epoch_stats['per_rank_microbatches']} "
            f"avg_bs={manifest_epoch_stats['avg_batch_size']:.2f} "
            f"min_bs={manifest_epoch_stats['min_batch_size']} "
            f"max_bs={manifest_epoch_stats['max_batch_size']}"
        )
    print(f"🖼️  Validation samples every {sample_config['sample_every']} steps")
    print(f"💾 Checkpoints every {process_config['save']['save_every']} steps")

    save_root = training_output_root / job_name
    validate_root = training_output_root / "validate" / job_name
    model_prefix = f"{S3_MODELS_PREFIX}/{model_id}"
    print(f"💾 Training outputs will be staged on ephemeral local disk: {save_root}")
    print(f"☁️  Checkpoints will be mirrored during training to: {s3_uri(model_prefix, 'checkpoints')}")

    network_config = process_config.get("network")
    train_config = process_config.get("train", {})
    if network_config is None and train_config.get("train_unet", False):
        print(
            "⚠️ No network/LoRA config found and train_unet=true. "
            "This looks like base-model fine-tuning, not LoRA-only training."
        )

    env = os.environ.copy()
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    env["HF_HOME"] = HF_HOME
    env["QUANTO_BYPASS_OBJECT_COPY"] = "1"
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_NVLS_ENABLE"] = "0"
    env["AITK_DDP_STATIC_GRAPH"] = "1"
    env["AITK_DDP_FIND_UNUSED_PARAMETERS"] = "0"
    # Bump NCCL process-group timeout to 30 min so a single slow rank does not
    # detonate the whole job after only 10 min (PyTorch default). Flows through
    # `InitProcessGroupKwargs` in lora-klein/ai-toolkit/toolkit/accelerator.py.
    env.setdefault("AITK_NCCL_TIMEOUT_SECONDS", "1800")
    env["S3_VALIDATION_UPLOAD_ROOT"] = s3_uri(model_prefix, "validate")
    if ddp_smoke:
        env["AITK_DDP_SMOKE"] = "1"
    resolve_gradient_accumulation_steps(process_config)
    env["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = "1"
    os.environ.setdefault("S3_CHECKPOINT_DELETE_LOCAL_AFTER_UPLOAD", "1")
    os.environ.setdefault("S3_CHECKPOINT_KEEP_LATEST_LOCAL", "1")
    os.environ.setdefault("S3_CHECKPOINT_HYDRATE_ON_START", "1")

    protected_resume_paths = hydrate_checkpoint_resume_state_from_s3(
        save_root=save_root,
        job_name=job_name,
        model_prefix=model_prefix,
    )
    if protected_resume_paths:
        print(
            "☁️  Protected hydrated resume artifacts for this launch: "
            f"{len(protected_resume_paths)}",
            flush=True,
        )

    checkpoint_uploader_stop = threading.Event()
    checkpoint_uploader_thread = start_checkpoint_s3_uploader(
        save_root=save_root,
        job_name=job_name,
        model_prefix=model_prefix,
        stop_event=checkpoint_uploader_stop,
        protected_paths=protected_resume_paths,
    )
    try:
        subprocess.run(
            [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={DDP_WORLD_SIZE_8}",
                "/root/ai-toolkit/run.py",
                str(ai_toolkit_config),
            ],
            cwd="/root/ai-toolkit",
            env=env,
            check=True,
        )
    finally:
        checkpoint_uploader_stop.set()
        checkpoint_uploader_thread.join(timeout=float(os.environ.get("S3_CHECKPOINT_FINAL_JOIN_SECONDS", "3600")))
        if checkpoint_uploader_thread.is_alive():
            print("⚠️  Live checkpoint S3 uploader did not finish before timeout", flush=True)

    network_config = process_config.get("network", {})
    model_config = process_config.get("model", {})

    def run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    if network_config:
        raw_lora_path = find_raw_lora_path(job_name)
        adapter_dir = raw_lora_path.parent / "peft_adapter"
        build_peft_adapter(
            source_path=raw_lora_path,
            output_dir=adapter_dir,
            base_model=model_config.get("name_or_path", "black-forest-labs/FLUX.2-klein-base-4B"),
            rank=network_config.get("linear"),
            alpha=network_config.get("linear_alpha"),
        )
        validate_peft_adapter_dir(adapter_dir)

    if save_root.exists():
        config_yaml = save_root / "config.yaml"
        optimizer_pt = save_root / "optimizer.pt"
        if config_yaml.exists():
            run(["aws", "s3", "cp", str(config_yaml), s3_uri(model_prefix, "config.yaml"), "--only-show-errors", "--no-progress"])
        if optimizer_pt.exists():
            run(["aws", "s3", "cp", str(optimizer_pt), s3_uri(model_prefix, "optimizer.pt"), "--only-show-errors", "--no-progress"])
        run(["aws", "s3", "sync", str(save_root), s3_uri(model_prefix, "checkpoints"), "--only-show-errors", "--no-progress"])
        peft_dir = save_root / "peft_adapter"
        if peft_dir.exists():
            run(["aws", "s3", "sync", str(peft_dir), s3_uri(model_prefix, "final", "peft_adapter"), "--only-show-errors", "--no-progress"])
    if validate_root.exists():
        run(["aws", "s3", "sync", str(validate_root), s3_uri(model_prefix, "validate"), "--only-show-errors", "--no-progress"])

    if str(save_root).startswith(str(Path(MOUNT_DIR))):
        model_volume.commit()

    return (
        f"Distributed training completed. Resume artifacts are sourced from S3 and staged locally on launch. "
        f"Artifacts mirrored to {s3_uri(S3_MODELS_PREFIX, model_id)}."
    )


@app.function(
    image=image_4b,
    timeout=24 * 60 * 60,
    cpu=4,
    memory=16384,
    volumes={DATASET_CACHE_ROOT: dataset_cache_volume},
    secrets=[aws_secret, hf_secret],
)
def run_drawtoon_cache_then_train(
    *,
    config_path: str,
    dataset_subpath: str | None,
    model_id: str | None,
    target_epochs: int,
    ddp_world_size: int,
    ddp_smoke: bool,
    max_train_steps: int,
    sample_every: int,
    drawtoon_bucket: str,
    drawtoon_caption_run: str,
    drawtoon_pages_prefix: str,
    drawtoon_annotations_prefix: str,
    drawtoon_captions_prefix: str,
    drawtoon_include_chapter_regex: str,
    drawtoon_max_pages: int,
    drawtoon_shard_count: int,
    drawtoon_overwrite_cache: bool,
):
    print("🚀 Starting AI-toolkit training orchestration", flush=True)
    print(f"📄 Config path: {config_path}", flush=True)
    print(f"🔁 Target epochs: {target_epochs}", flush=True)
    print(f"🌐 DDP world size: {ddp_world_size}", flush=True)
    if max_train_steps > 0:
        print(f"🧪 Max train steps: {max_train_steps}", flush=True)
    if sample_every > 0:
        print(f"🖼️  Validation interval override: {sample_every}", flush=True)

    cache_result = prepare_drawtoon_panel_cache_for_launch(
        config_path=config_path,
        overrides={
            "bucket": drawtoon_bucket,
            "caption_run": drawtoon_caption_run,
            "pages_prefix": drawtoon_pages_prefix,
            "annotations_prefix": drawtoon_annotations_prefix,
            "captions_prefix": drawtoon_captions_prefix,
            "include_chapter_regex": drawtoon_include_chapter_regex,
            "max_pages": drawtoon_max_pages,
        },
        shard_count=drawtoon_shard_count,
        overwrite=drawtoon_overwrite_cache,
    )
    if cache_result.get("enabled"):
        config_path = str(cache_result["resolved_config_path"])
        print(f"📚 Drawtoon cache manifest: {cache_result['manifest_path']}", flush=True)
        if cache_result.get("row_count"):
            print(f"📚 Drawtoon cache rows: {cache_result['row_count']}", flush=True)
    else:
        print("📚 Using manifest_path already present in the training config.", flush=True)

    if ddp_world_size == DDP_WORLD_SIZE_8:
        function_call = train_flux_lora_ddp8.spawn(
            config_path=config_path,
            dataset_subpath=dataset_subpath or None,
            model_id=model_id or None,
            target_epochs=target_epochs,
            ddp_smoke=ddp_smoke,
            max_train_steps=max_train_steps,
            sample_every=sample_every,
        )
    elif ddp_world_size == 1:
        trainer = FluxTrainer()
        function_call = trainer.train_flux_lora.spawn(
            config_path=config_path,
            dataset_subpath=dataset_subpath or None,
            model_id=model_id or None,
            target_epochs=target_epochs,
            max_train_steps=max_train_steps,
            sample_every=sample_every,
        )
    else:
        raise ValueError(f"Unsupported ddp_world_size={ddp_world_size}")

    print(f"🆔 Training Modal call id: {function_call.object_id}", flush=True)
    print("✅ Training job submitted after cache success.", flush=True)
    return function_call.get()


@app.local_entrypoint()
def main(
    config_path: str = DEFAULT_CONFIG_PATH,
    dataset_subpath: str = "",
    model_id: str = "",
    preset: str = "",
    target_epochs: int = 3,
    ddp_world_size: int = 1,
    ddp_smoke: bool = False,
    max_train_steps: int = 0,
    sample_every: int = 0,
    drawtoon_bucket: str = S3_BUCKET,
    drawtoon_caption_run: str = "",
    drawtoon_pages_prefix: str = DEFAULT_DRAWTOON_PAGES_PREFIX,
    drawtoon_annotations_prefix: str = DEFAULT_DRAWTOON_ANNOTATIONS_PREFIX,
    drawtoon_captions_prefix: str = DEFAULT_DRAWTOON_CAPTIONS_PREFIX,
    drawtoon_include_chapter_regex: str = "",
    drawtoon_max_pages: int = 0,
    drawtoon_shard_count: int = 0,
    drawtoon_overwrite_cache: bool = False,
):
    launch_args = LaunchArgs(
        config_path=config_path,
        dataset_subpath=dataset_subpath,
        model_id=model_id,
        preset=preset,
        target_epochs=target_epochs,
        ddp_world_size=ddp_world_size,
        max_train_steps=max_train_steps,
        sample_every=sample_every,
    )
    config_path = launch_args.resolved_config_path()
    dataset_subpath = launch_args.dataset_subpath
    model_id = launch_args.model_id
    preset = launch_args.preset
    target_epochs = launch_args.target_epochs
    ddp_world_size = launch_args.ddp_world_size
    max_train_steps = launch_args.max_train_steps
    sample_every = launch_args.sample_every
    print("🚀 Starting AI-toolkit training")
    print("🔧 Optimized with memory snapshots for fast cold starts")
    print("=" * 60)
    print(f"📄 Config path: {config_path}")
    if preset:
        print(f"🎯 Preset: {preset}")
    if dataset_subpath:
        raise RuntimeError("dataset_subpath staging has been removed; use manifest dataset configs")
    if model_id:
        print(f"🧾 Model ID: {model_id}")
    print(f"🔁 Target epochs: {target_epochs}")
    print(f"🌐 DDP world size: {ddp_world_size}")
    if max_train_steps > 0:
        print(f"🧪 Max train steps: {max_train_steps}")
    if sample_every > 0:
        print(f"🖼️  Validation interval override: {sample_every}")
    if ddp_smoke:
        print("🧪 DDP smoke logging enabled")

    function_call = run_drawtoon_cache_then_train.spawn(
        config_path=config_path,
        dataset_subpath=dataset_subpath or None,
        model_id=model_id or None,
        target_epochs=target_epochs,
        ddp_world_size=ddp_world_size,
        ddp_smoke=ddp_smoke,
        max_train_steps=max_train_steps,
        sample_every=sample_every,
        drawtoon_bucket=drawtoon_bucket,
        drawtoon_caption_run=drawtoon_caption_run,
        drawtoon_pages_prefix=drawtoon_pages_prefix,
        drawtoon_annotations_prefix=drawtoon_annotations_prefix,
        drawtoon_captions_prefix=drawtoon_captions_prefix,
        drawtoon_include_chapter_regex=drawtoon_include_chapter_regex,
        drawtoon_max_pages=drawtoon_max_pages,
        drawtoon_shard_count=drawtoon_shard_count,
        drawtoon_overwrite_cache=drawtoon_overwrite_cache,
    )
    print(f"🆔 Orchestration Modal call id: {function_call.object_id}")
    print("✅ Detached orchestration submitted; it will launch training after cache success.")
    print("⏳ Waiting on the spawned training call so detached Modal keeps the live job attached.")
    print("💻 You can close your computer now; Modal will keep running the job.")

    result = function_call.get()

    print("\n🎉 Training workflow completed!")
    print("📁 Resume artifacts are sourced from S3 and staged locally on launch.")
    print("☁️ Artifacts are mirrored to S3 under models/<model_id>/.")
    if result:
        print(result)
