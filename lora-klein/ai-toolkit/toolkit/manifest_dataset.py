#!/usr/bin/env python3
"""Manifest-backed dataset with canonical source IDs and FLUX-safe control handling."""

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import boto3
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.dataloader_mixins import BucketsMixin, accelerator
from toolkit.config_modules import DatasetConfig
from toolkit.distributed_cache import run_once_with_filelock
from toolkit.distributed_logging import rank_tqdm
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.print import print_acc
from extensions_built_in.diffusion_models.flux2.src.gia import (
    build_gia_inputs_from_lamic_sample,
    build_gia_prompt,
)


class ManifestDataset(BucketsMixin, Dataset):
    """Dataset that loads target/control images from a manifest row."""

    CHARACTER_CONTROL_PREPARED_MARKER = "page_controls_prepared/v3_minpad_768"

    def __init__(self, config: DatasetConfig, **kwargs):
        self.config = config
        self.dataset_config = config
        self.sd = kwargs.get("sd", None)
        self.batch_size = kwargs.get("batch_size", 1)
        self.epoch_num = 0
        self.is_audio_model = False
        self.is_video = False
        self.is_caching_latents = config.cache_latents or config.cache_latents_to_disk
        self.is_caching_latents_to_memory = config.cache_latents
        self.is_caching_latents_to_disk = config.cache_latents_to_disk
        self.file_list: List[FileItemDTO] = []
        self.batch_indices: List[List[int]] = []
        self.deferred_target_size_count = 0
        self.materialized_target_startup_count = 0
        self.resize_mode = "bucket_crop" if config.buckets else "native"
        self.use_bucket_batches = self.resize_mode == "bucket_crop"
        self.dataset_config.buckets = self.use_bucket_batches
        if self.sd is not None:
            self.dataset_config.bucket_tolerance = self.sd.get_bucket_divisibility()
        if self.sd is None and self.is_caching_latents:
            raise ValueError("sd is required for manifest latent caching")
        if self.resize_mode == "native" and self.batch_size > 1:
            raise ValueError(
                "Native manifest images require batch_size=1 because images can have "
                "different dimensions. Use buckets=true for multi-sample batches."
            )

        self._validate_config()

        self.manifest_path = config.manifest_path
        if not self.manifest_path:
            raise ValueError("manifest_path is required for manifest datasets")
        self.dataset_path = self.manifest_path

        self.manifest_ref = self._parse_source_ref(self.manifest_path, allow_relative=False)
        self.local_cache_dir = Path(config.local_cache_dir or "/tmp/lora-klein-cache")
        self.sample_cache_dir = self.local_cache_dir / "manifest_samples"
        self.sample_cache_dir.mkdir(parents=True, exist_ok=True)
        self._s3_client = None

        dataset_root_hint = config.dataset_s3_path or self._default_dataset_root()
        self.dataset_root_ref = self._parse_source_ref(
            dataset_root_hint,
            allow_relative=False,
        ) if dataset_root_hint else None

        self.samples = self._load_manifest_rows()
        self.target_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda image: image * 2 - 1),
            ]
        )
        self.control_transform = transforms.ToTensor()

        for sample in self.samples:
            target_ref = self._parse_source_ref(sample["target_panel"])
            sample_id = sample.get("sample_id") or sample.get("page_root") or target_ref["canonical"]
            file_item = self._build_manifest_file_item(
                sample=sample,
                target_ref=target_ref,
                sample_id=str(sample_id),
            )
            file_item.manifest_sample = sample
            file_item.manifest_target_ref = target_ref
            self.file_list.append(file_item)

        if self.deferred_target_size_count or self.materialized_target_startup_count:
            print_acc(
                "ManifestDataset target startup: "
                f"deferred={self.deferred_target_size_count:,} "
                f"materialized={self.materialized_target_startup_count:,} "
                f"total={len(self.file_list):,}"
            )

        self._expand_flips()
        self.setup_epoch()

    def _build_manifest_file_item(
        self,
        *,
        sample: dict,
        target_ref: Dict[str, str],
        sample_id: str,
    ) -> FileItemDTO:
        preset_width, preset_height = self._declared_target_size(sample)
        if (
            self.resize_mode == "native"
            and not self.is_caching_latents
            and preset_width is not None
            and preset_height is not None
        ):
            self.deferred_target_size_count += 1
            local_target_path = self._path_hint_for_ref(target_ref)
            preset_kwargs = {
                "preset_width": preset_width,
                "preset_height": preset_height,
                "preset_file_signature": f"manifest:{sample_id}:{preset_width}x{preset_height}",
            }
        else:
            self.materialized_target_startup_count += 1
            local_target_path = self._materialize_ref(target_ref)
            preset_kwargs = {}
        file_item = FileItemDTO(
            path=local_target_path,
            dataset_config=self.config,
            sd=self.sd,
            dataset_root=str(self.sample_cache_dir),
            canonical_image_id=sample_id,
            raw_caption=sample["caption"],
            **preset_kwargs,
        )
        gia_inputs = build_gia_inputs_from_lamic_sample(sample)
        if gia_inputs is not None:
            file_item.gia_inputs = gia_inputs
            file_item.gia_prompt = build_gia_prompt(gia_inputs)[0]
        if self.resize_mode == "native":
            self._set_native_target_geometry(file_item)
        return file_item

    def setup_buckets(self, quiet=False):
        super().setup_buckets(quiet=quiet)

    def _validate_config(self):
        unsupported = []
        if self.config.num_frames > 1 or self.config.auto_frame_count:
            unsupported.append("video/auto_frame_count")
        if self.config.cache_clip_vision_to_disk:
            unsupported.append("clip vision caching")
        if self.config.cache_text_embeddings:
            unsupported.append("text embedding caching")
        if self.config.standardize_images:
            unsupported.append("standardize_images")
        if self.config.random_scale:
            unsupported.append("random_scale")
        if self.config.augmentations:
            unsupported.append("augmentations")
        if self.config.augments:
            unsupported.append("augments")
        if self.config.controls:
            unsupported.append("generated controls")
        if self.config.inpaint_path is not None:
            unsupported.append("inpaint_path")
        if self.config.clip_image_path is not None:
            unsupported.append("clip_image_path")
        if self.config.mask_path is not None:
            unsupported.append("mask_path")
        if self.config.unconditional_path is not None:
            unsupported.append("unconditional_path")
        if self.config.num_repeats != 1:
            unsupported.append("num_repeats")
        if (
            self._uses_raw_multi_control_images()
            and self.batch_size > 1
        ):
            unsupported.append(
                "batch_size > 1 with raw FLUX multi-control images. "
                "Use batch_size=1 + gradient accumulation."
            )
        if unsupported:
            raise ValueError(
                "ManifestDataset does not currently support: " + ", ".join(unsupported)
            )

    def _target_multiple(self) -> int:
        if self.sd is not None and hasattr(self.sd, "get_bucket_divisibility"):
            return max(1, int(self.sd.get_bucket_divisibility()))
        return 16

    def _pad_size_to_target_multiple(self, width: int, height: int) -> tuple[int, int]:
        multiple = self._target_multiple()
        padded_width = int(math.ceil(int(width) / multiple) * multiple)
        padded_height = int(math.ceil(int(height) / multiple) * multiple)
        return padded_width, padded_height

    def _set_native_target_geometry(self, file_item: FileItemDTO) -> tuple[int, int]:
        width = int(file_item.width)
        height = int(file_item.height)
        output_width, output_height = self._pad_size_to_target_multiple(width, height)
        file_item.scale_to_width = width
        file_item.scale_to_height = height
        file_item.crop_x = 0
        file_item.crop_y = 0
        file_item.crop_width = output_width
        file_item.crop_height = output_height
        return output_width, output_height

    def _uses_raw_multi_control_images(self) -> bool:
        return bool(
            self.sd is not None
            and getattr(self.sd, "use_raw_control_images", False)
            and getattr(self.sd, "has_multiple_control_images", False)
        )

    def _default_dataset_root(self) -> Optional[str]:
        if self.manifest_ref["kind"] == "s3":
            return self.manifest_ref["canonical"].rsplit("/", 1)[0]
        if self.manifest_ref["kind"] == "local":
            return str(Path(self.manifest_ref["path"]).parent)
        return None

    def _declared_target_size(self, sample: dict) -> tuple[int | None, int | None]:
        try:
            width = int(sample.get("target_width") or 0)
            height = int(sample.get("target_height") or 0)
        except (TypeError, ValueError):
            return None, None
        if width <= 0 or height <= 0:
            return None, None
        return width, height

    def _mounted_path_for_s3_ref(self, source_ref: Dict[str, str]) -> str | None:
        if source_ref.get("kind") != "s3":
            return None
        key = str(source_ref.get("key") or "").lstrip("/")
        if not key:
            return None
        return os.path.join("/mnt/datasets", key)

    def _path_hint_for_ref(self, source_ref: Dict[str, str]) -> str:
        if source_ref["kind"] == "local":
            return source_ref["path"]
        mounted_path = self._mounted_path_for_s3_ref(source_ref)
        if mounted_path:
            return mounted_path
        return source_ref["canonical"]

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _resolve_s3_object(self, source_ref: Dict[str, str]) -> tuple[dict, str]:
        head = None
        resolved_key = None
        last_error = None
        candidate_keys = [source_ref["key"], *source_ref.get("alternate_keys", [])]
        for candidate_key in OrderedDict.fromkeys(candidate_keys):
            try:
                head = self.s3_client.head_object(
                    Bucket=source_ref["bucket"],
                    Key=candidate_key,
                )
                resolved_key = candidate_key
                break
            except Exception as exc:
                last_error = exc

        if head is None or resolved_key is None:
            if last_error is not None:
                raise last_error
            raise FileNotFoundError(
                f"Unable to resolve S3 object for {source_ref['canonical']}"
            )

        return head, resolved_key

    def _load_manifest_rows(self) -> List[dict]:
        if self.manifest_ref["kind"] == "s3":
            response = self.s3_client.get_object(
                Bucket=self.manifest_ref["bucket"],
                Key=self.manifest_ref["key"],
            )
            manifest_content = response["Body"].read().decode("utf-8")
        else:
            with open(self.manifest_ref["path"], "r", encoding="utf-8") as f:
                manifest_content = f.read()

        rows = []
        for line in manifest_content.splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
        return rows

    def _parse_source_ref(self, path_or_key: Any, allow_relative: bool = True) -> Dict[str, str]:
        if isinstance(path_or_key, dict):
            path_or_key = path_or_key.get("image") or path_or_key.get("path")
        if not path_or_key:
            raise ValueError("Expected a non-empty manifest image path")

        raw_value = str(path_or_key).strip()
        if raw_value.startswith("s3://"):
            bucket, key = raw_value[5:].split("/", 1)
            return {
                "kind": "s3",
                "bucket": bucket,
                "key": key.lstrip("/"),
                "canonical": f"s3://{bucket}/{key.lstrip('/')}",
            }
        if raw_value.startswith("file://"):
            local_path = os.path.abspath(unquote(urlparse(raw_value).path))
            return {
                "kind": "local",
                "path": local_path,
                "canonical": local_path,
            }
        if raw_value.startswith("http://") or raw_value.startswith("https://"):
            raise ValueError(f"HTTP(S) sources are not supported in ManifestDataset: {raw_value}")
        if os.path.isabs(raw_value):
            return {
                "kind": "local",
                "path": os.path.abspath(raw_value),
                "canonical": os.path.abspath(raw_value),
            }
        if raw_value.startswith("datasets/"):
            bucket = None
            if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "s3":
                bucket = self.dataset_root_ref["bucket"]
            elif self.manifest_ref["kind"] == "s3":
                bucket = self.manifest_ref["bucket"]
            if bucket is not None:
                key = raw_value.lstrip("/")
                return {
                    "kind": "s3",
                    "bucket": bucket,
                    "key": key,
                    "canonical": f"s3://{bucket}/{key}",
                }
        if not allow_relative:
            local_path = os.path.abspath(raw_value)
            return {
                "kind": "local",
                "path": local_path,
                "canonical": local_path,
            }

        if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "s3":
            prefix = self.dataset_root_ref.get("key", "").strip("/")
            key = raw_value.lstrip("/")
            alternate_keys = []
            if prefix and not key.startswith(prefix):
                alternate_keys.append(key)
                key = f"{prefix}/{key}"
            return {
                "kind": "s3",
                "bucket": self.dataset_root_ref["bucket"],
                "key": key,
                "canonical": f"s3://{self.dataset_root_ref['bucket']}/{key}",
                "alternate_keys": alternate_keys,
            }

        if self.manifest_ref["kind"] == "s3":
            prefix = self.manifest_ref["key"].rsplit("/", 1)[0]
            key = raw_value.lstrip("/")
            if prefix and not key.startswith(prefix):
                key = f"{prefix}/{key}"
            return {
                "kind": "s3",
                "bucket": self.manifest_ref["bucket"],
                "key": key,
                "canonical": f"s3://{self.manifest_ref['bucket']}/{key}",
            }

        base_dir = (
            self.dataset_root_ref["path"]
            if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "local"
            else os.path.dirname(self.manifest_ref["path"])
        )
        local_path = os.path.abspath(os.path.join(base_dir, raw_value))
        return {
            "kind": "local",
            "path": local_path,
            "canonical": local_path,
        }

    def _materialize_ref(self, source_ref: Dict[str, str]) -> str:
        if source_ref["kind"] == "local":
            return source_ref["path"]

        mounted_path = self._mounted_path_for_s3_ref(source_ref)
        if mounted_path and os.path.exists(mounted_path):
            return mounted_path

        head, resolved_key = self._resolve_s3_object(source_ref)

        etag = str(head.get("ETag", "")).strip('"')
        version_id = str(head.get("VersionId", "") or "")
        cache_identity = f"s3://{source_ref['bucket']}/{resolved_key}"
        if version_id:
            cache_identity += f"?versionId={version_id}"
        elif etag:
            cache_identity += f"?etag={etag}"
        suffix = Path(resolved_key).suffix or ".png"
        digest = hashlib.sha1(cache_identity.encode("utf-8")).hexdigest()
        local_path = self.sample_cache_dir / digest[:2] / f"{digest}{suffix}"
        if local_path.exists():
            return str(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            suffix=local_path.suffix + ".tmp",
            prefix=local_path.stem + ".",
            dir=local_path.parent,
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        response = self.s3_client.get_object(
            Bucket=source_ref["bucket"],
            Key=resolved_key,
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(response["Body"].read())
            os.replace(tmp_path, local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return str(local_path)

    def _source_signature(self, source_ref: Dict[str, str]) -> Dict[str, Any]:
        if source_ref["kind"] == "s3":
            head, resolved_key = self._resolve_s3_object(source_ref)
            return {
                "kind": "s3",
                "bucket": source_ref["bucket"],
                "key": resolved_key,
                "etag": str(head.get("ETag", "")).strip('"'),
                "version_id": str(head.get("VersionId", "") or ""),
                "content_length": int(head.get("ContentLength", 0) or 0),
                "last_modified": head.get("LastModified").isoformat()
                if head.get("LastModified")
                else "",
            }

        local_path = os.path.abspath(source_ref["path"])
        stat = os.stat(local_path)
        return {
            "kind": "local",
            "path": local_path,
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }

    def _canonical_from_signature(
        self, source_ref: Dict[str, str], source_signature: Dict[str, Any]
    ) -> str:
        kind = str(source_signature.get("kind") or source_ref.get("kind") or "").strip()
        if kind == "s3":
            bucket = str(source_signature.get("bucket") or source_ref.get("bucket") or "").strip()
            key = str(source_signature.get("key") or source_ref.get("key") or "").lstrip("/")
            if bucket and key:
                return f"s3://{bucket}/{key}"
        if kind == "local":
            path = str(source_signature.get("path") or source_ref.get("path") or "").strip()
            if path:
                return os.path.abspath(path)
        return str(source_ref.get("canonical") or "").strip()

    def _target_geometry_for_file_item(self, file_item: FileItemDTO) -> Dict[str, int]:
        resolution = int(self.dataset_config.resolution)
        if self.resize_mode == "native":
            width = int(file_item.width)
            height = int(file_item.height)
            output_width, output_height = self._set_native_target_geometry(file_item)
            return {
                "resize_mode": "native",
                "resolution": resolution,
                "bucket_tolerance": int(self.dataset_config.bucket_tolerance or 16),
                "scale_to_width": width,
                "scale_to_height": height,
                "crop_x": 0,
                "crop_y": 0,
                "crop_width": output_width,
                "crop_height": output_height,
                "output_width": output_width,
                "output_height": output_height,
            }

        if self.resize_mode == "bucket_crop":
            return {
                "resize_mode": "bucket_crop",
                "resolution": resolution,
                "bucket_tolerance": int(self.dataset_config.bucket_tolerance or 16),
                "scale_to_width": int(file_item.scale_to_width),
                "scale_to_height": int(file_item.scale_to_height),
                "crop_x": int(file_item.crop_x),
                "crop_y": int(file_item.crop_y),
                "crop_width": int(file_item.crop_width),
                "crop_height": int(file_item.crop_height),
                "output_width": int(file_item.crop_width),
                "output_height": int(file_item.crop_height),
            }

        raise ValueError(f"Unsupported manifest resize mode: {self.resize_mode}")

    def _expand_flips(self):
        if self.dataset_config.flip_x:
            current_items = [item for item in self.file_list]
            for file_item in current_items:
                flipped = copy.deepcopy(file_item)
                flipped.flip_x = True
                self.file_list.append(flipped)
        if self.dataset_config.flip_y:
            current_items = [item for item in self.file_list]
            for file_item in current_items:
                flipped = copy.deepcopy(file_item)
                flipped.flip_y = True
                self.file_list.append(flipped)

    def setup_epoch(self):
        if self.epoch_num == 0:
            if self.use_bucket_batches:
                self.setup_buckets()
            if self.is_caching_latents:
                self.cache_latents_all_latents()
        elif self.use_bucket_batches:
            self.setup_buckets(quiet=True)
        self.epoch_num += 1

    def __len__(self):
        if self.use_bucket_batches:
            return len(self.batch_indices)
        return len(self.file_list)

    def _load_image(self, source_ref: Dict[str, str]) -> Image.Image:
        local_path = self._materialize_ref(source_ref)
        image = Image.open(local_path)
        image = exif_transpose(image)
        return image.convert("RGB")

    def _apply_flip(self, image: Image.Image, file_item: FileItemDTO) -> Image.Image:
        if file_item.flip_x:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if file_item.flip_y:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def _apply_target_geometry(self, image: Image.Image, file_item: FileItemDTO) -> Image.Image:
        if self.resize_mode == "native":
            width, height = image.size
            declared_width = int(file_item.width)
            declared_height = int(file_item.height)
            if (declared_width, declared_height) != (width, height):
                raise ValueError(
                    "Manifest target size does not match loaded target image: "
                    f"{file_item.path} manifest={declared_width}x{declared_height} "
                    f"image={width}x{height}"
                )
            output_width, output_height = self._set_native_target_geometry(file_item)
            if output_width == width and output_height == height:
                return image
            padded = Image.new("RGB", (output_width, output_height), (255, 255, 255))
            padded.paste(image, (0, 0))
            return padded

        if self.resize_mode == "bucket_crop":
            image = image.resize((file_item.scale_to_width, file_item.scale_to_height), Image.BICUBIC)
            return image.crop(
                (
                    file_item.crop_x,
                    file_item.crop_y,
                    file_item.crop_x + file_item.crop_width,
                    file_item.crop_y + file_item.crop_height,
                )
            )

        raise ValueError(f"Unsupported manifest resize mode: {self.resize_mode}")

    def _apply_control_geometry(self, image: Image.Image, file_item: FileItemDTO) -> Image.Image:
        if getattr(self.sd, "use_raw_control_images", False):
            return image
        if getattr(file_item, "use_raw_control_images", False):
            return image
        return self._apply_target_geometry(image, file_item)

    def _load_manifest_controls(self, file_item: FileItemDTO):
        sample = file_item.manifest_sample
        controls = sample.get("controls", {})
        control_refs: List[Dict[str, str]] = []

        prev_control = controls.get("previous_panel") or controls.get("previous_page")
        if prev_control:
            control_refs.append(self._parse_source_ref(prev_control))

        for char_ref_path in controls.get("character_ref_paths", [])[: self.config.max_character_refs]:
            control_refs.append(self._parse_source_ref(char_ref_path))

        if not control_refs:
            return

        tensors: List[torch.Tensor] = []
        for control_ref in control_refs:
            control_img = self._load_image(control_ref)
            control_img = self._apply_flip(control_img, file_item)
            control_img = self._apply_control_geometry(control_img, file_item)
            control_width, control_height = control_img.size
            target_multiple = self._target_multiple()
            if control_width <= 0 or control_height <= 0:
                raise ValueError(
                    f"Control image has invalid geometry: {control_ref['canonical']} = "
                    f"{control_width}x{control_height}"
                )
            if control_width % target_multiple != 0 or control_height % target_multiple != 0:
                raise ValueError(
                    f"Control image is not divisible by {target_multiple}: "
                    f"{control_ref['canonical']} = {control_width}x{control_height}"
                )
            tensors.append(self.control_transform(control_img))

        if not tensors:
            return

        if getattr(self.sd, "has_multiple_control_images", False) or getattr(file_item, "use_raw_control_images", False):
            file_item.control_tensor_list = tensors
        elif len(tensors) == 1:
            file_item.control_tensor = tensors[0]
        else:
            file_item.control_tensor = torch.stack(tensors, dim=0)

    def _prepare_file_item_target_tensor(self, file_item: FileItemDTO):
        target_img = self._load_image(file_item.manifest_target_ref)
        target_img = self._apply_flip(target_img, file_item)
        target_img = self._apply_target_geometry(target_img, file_item)
        file_item.tensor = self.target_transform(target_img)

    def cache_latents_all_latents(self):
        if self.is_caching_latents_to_disk:
            marker_payload = self._target_latent_cache_marker_payload()
            run_once_with_filelock(
                accelerator=accelerator,
                cache_dir=self.sample_cache_dir,
                marker_name=self._target_latent_cache_marker_name(),
                logger=self._distributed_logger(),
                should_run=self._target_latent_cache_incomplete,
                marker_payload=marker_payload,
                fn=self._cache_latents_all_latents_impl,
            )
            self._mark_cached_latents_from_disk()
            return

        self._cache_latents_all_latents_impl()

    def _distributed_logger(self):
        return getattr(self.sd, "distributed_logger", None)

    def _target_latent_cache_marker_name(self) -> str:
        payload = self._target_latent_cache_marker_payload()
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        return f"target_latents_{digest}"

    def _target_latent_cache_marker_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset_path": self.dataset_path,
            "sample_count": len(self.file_list),
            "torch_dtype": str(self.sd.torch_dtype),
            "to_disk": bool(self.is_caching_latents_to_disk),
            "to_memory": bool(self.is_caching_latents_to_memory),
        }

        first_latent_path = self.file_list[0].get_latent_path(recalculate=True) if self.file_list else ""
        last_latent_path = self.file_list[-1].get_latent_path(recalculate=True) if self.file_list else ""
        payload["first_latent_path"] = first_latent_path
        payload["last_latent_path"] = last_latent_path

        return payload

    def _target_latent_cache_incomplete(self) -> bool:
        for file_item in self.file_list:
            latent_path = file_item.get_latent_path(recalculate=True)
            if not os.path.exists(latent_path):
                return True
            if not self._latent_path_matches_expected(file_item, latent_path):
                return True
        return False

    def _latent_path_matches_expected(self, file_item: FileItemDTO, latent_path: str) -> bool:
        metadata = load_metadata_from_safetensors(latent_path)
        if not metadata:
            return False

        expected_info = file_item.get_latent_info_dict()
        for key, expected_value in expected_info.items():
            if metadata.get(key) != expected_value:
                return False

        return True

    def _mark_cached_latents_from_disk(self):
        dtype = self.sd.torch_dtype
        missing_paths: list[str] = []
        for file_item in self.file_list:
            file_item.is_caching_to_disk = self.is_caching_latents_to_disk
            file_item.is_caching_to_memory = self.is_caching_latents_to_memory
            file_item.latent_load_device = self.sd.device
            latent_path = file_item.get_latent_path(recalculate=True)
            if not os.path.exists(latent_path):
                file_item.is_latent_cached = False
                missing_paths.append(latent_path)
                continue
            if not self._latent_path_matches_expected(file_item, latent_path):
                file_item.is_latent_cached = False
                missing_paths.append(latent_path)
                continue
            file_item.is_latent_cached = True
            if self.is_caching_latents_to_memory:
                state_dict = load_file(latent_path, device="cpu")
                file_item._encoded_latent = state_dict["latent"].to("cpu", dtype=dtype)

        if missing_paths:
            preview = missing_paths[:5]
            raise RuntimeError(
                "Latent cache was expected to be complete after distributed cache seeding, "
                f"but {len(missing_paths)} latent files are missing. First missing paths: {preview}"
            )

        dist_logger = self._distributed_logger()
        if dist_logger is not None:
            cached_count = sum(1 for file_item in self.file_list if file_item.is_latent_cached)
            dist_logger.gather_event(
                "latent_cache_state",
                cached=cached_count,
                total=len(self.file_list),
                cache_to_memory=bool(self.is_caching_latents_to_memory),
            )

    def _cache_latents_all_latents_impl(self):
        started_at = time.monotonic()
        dist_logger = self._distributed_logger()
        print_acc(f"Caching latents for {self.dataset_path}")
        to_disk = self.is_caching_latents_to_disk
        to_memory = self.is_caching_latents_to_memory

        if to_disk:
            print_acc(" - Saving latents to disk")
        if to_memory:
            print_acc(" - Keeping latents in memory")

        self.sd.set_device_state_preset("cache_latents")
        dtype = self.sd.torch_dtype
        device = self.sd.device_torch

        progress_iterable = rank_tqdm(
            self.file_list,
            desc=f'Caching latents{" to disk" if to_disk else ""}',
            total=len(self.file_list),
            logger=dist_logger,
        )
        for file_item in progress_iterable:
            file_item.is_caching_to_disk = to_disk
            file_item.is_caching_to_memory = to_memory
            file_item.latent_load_device = self.sd.device
            latent_path = file_item.get_latent_path(recalculate=True)

            if os.path.exists(latent_path) and self._latent_path_matches_expected(file_item, latent_path):
                if to_memory:
                    state_dict = load_file(latent_path, device="cpu")
                    file_item._encoded_latent = state_dict["latent"].to("cpu", dtype=dtype)
                file_item.is_latent_cached = True
                continue

            self._prepare_file_item_target_tensor(file_item)
            imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
            latent = self.sd.encode_images(imgs).squeeze(0)

            state_dict = OrderedDict()
            if to_disk:
                state_dict["latent"] = latent.clone().detach().cpu()
                latent_meta = OrderedDict(file_item.get_latent_info_dict())
                meta = get_meta_for_safetensors(latent_meta)
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                save_file(state_dict, latent_path, metadata=meta)

            if to_memory:
                file_item._encoded_latent = latent.to("cpu", dtype=dtype)

            del imgs
            del latent
            del state_dict
            file_item.tensor = None
            file_item.is_latent_cached = True

        self.sd.restore_device_state()
        if dist_logger is not None:
            dist_logger.event(
                "target_latent_cache_ready",
                print_main=True,
                sample_count=len(self.file_list),
                seconds=round(time.monotonic() - started_at, 3),
                to_disk=bool(to_disk),
                to_memory=bool(to_memory),
            )

    def _get_single_item(self, index: int) -> FileItemDTO:
        file_item = copy.deepcopy(self.file_list[index])
        if not file_item.is_latent_cached:
            self._prepare_file_item_target_tensor(file_item)
        file_item.load_caption()
        gia_prompt = getattr(file_item, "gia_prompt", None)
        if gia_prompt:
            file_item.caption = gia_prompt
            file_item.caption_short = gia_prompt
        self._load_manifest_controls(file_item)
        return file_item

    def __getitem__(self, item):
        if self.use_bucket_batches:
            if len(self.batch_indices) - 1 < item:
                item = 0
            idx_list = self.batch_indices[item]
            return [self._get_single_item(idx) for idx in idx_list]
        return self._get_single_item(item)

    def collate_fn(self, batch: List[FileItemDTO]) -> DataLoaderBatchDTO:
        return DataLoaderBatchDTO(file_items=batch)
