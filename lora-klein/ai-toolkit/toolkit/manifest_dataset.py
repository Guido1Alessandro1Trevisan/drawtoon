#!/usr/bin/env python3
"""Manifest-backed dataset with canonical source IDs and FLUX-safe control handling."""

import copy
import hashlib
import json
import math
import os
import pickle
import tempfile
import time

import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import boto3
import torch
from PIL import Image, ImageDraw, ImageOps
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.dataloader_mixins import BucketsMixin, accelerator
from toolkit.config_modules import DatasetConfig, MAX_CONTROL_IMAGE_SLOTS
from toolkit.distributed_cache import run_once_with_filelock
from toolkit.distributed_logging import rank_tqdm
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.print import print_acc


LAYOUT_BACKGROUND_COLOR = (0, 0, 0)
LAYOUT_TEXT_COLORS = {
    "Speech Bubble": (0, 96, 255),
    "Narration Bubble": (255, 128, 0),
    "Shout Bubble": (128, 0, 255),
}


class _TorchSerializedList:
    """Memory-efficient list replacement that fixes the classic CoW-defeat leak
    in fork-based DataLoader workers (PyTorch issue #13246, Yuxin Wu / Detectron2
    pattern).

    Holding a Python ``list[dict]`` or ``list[FileItemDTO]`` of N items means
    every worker that even READS an entry bumps a Py_REFCNT, writes the page
    header, and triggers glibc to copy the 4KB page out of CoW into the
    worker's private memory. Over an epoch, all workers end up with private
    copies of the full manifest → slow ~10 GB/hr RSS creep with 32 workers.

    Wrapping the list in this class stores all entries as one contiguous
    ``torch.Tensor`` of uint8 bytes. PyTorch's tensor pickler relocates
    storage to ``/dev/shm``, so workers genuinely share one buffer across
    ranks instead of materializing private copies. Per-item access pays a
    microsecond ``pickle.loads`` — vastly cheaper than image I/O.

    Returned items are independent fresh copies, so callers can mutate them
    freely without affecting the underlying store.
    """

    def __init__(self, lst):
        serialized = [
            np.frombuffer(pickle.dumps(item, protocol=-1), dtype=np.uint8)
            for item in lst
        ]
        sizes = np.asarray([len(b) for b in serialized], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(sizes))
        self._lst = torch.from_numpy(np.concatenate(serialized)) if serialized else torch.empty(0, dtype=torch.uint8)

    def __len__(self) -> int:
        return len(self._addr)

    def __getitem__(self, idx: int):
        start = 0 if idx == 0 else int(self._addr[idx - 1])
        end = int(self._addr[idx])
        return pickle.loads(memoryview(self._lst[start:end].numpy()))

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def _normalize_slice_entry(raw: Any) -> Dict[str, Any]:
    """Coerce a manwa-sheet slice into the canonical dict shape.

    Each slice describes one band of a stitched manwa sheet:
      - ``source_page_key``: S3 key (or s3:// URI) of the raw source page
      - ``source_y_start`` / ``source_y_end``: y-range cropped from that page
      - ``sheet_y_start`` / ``sheet_y_end``: where the cropped band lands in
        the final sheet image. The reconstructor pastes at ``sheet_y_start``
        exactly and asserts the slice plan is contiguous, so any future drift
        between annotation-time and train-time surfaces immediately rather
        than producing a silently shifted sheet.
      - ``source_page_width`` / ``source_page_height`` / ``source_page_etag``
        (optional): captured at annotation-time. If present and the reloaded
        page disagrees, the reconstructor raises rather than producing a
        sheet whose pixel coords no longer match the annotation bboxes.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"manwa-sheet slice must be a dict, got {type(raw).__name__}")
    key = str(raw.get("source_page_key") or raw.get("page_key") or "").strip()
    if not key:
        raise ValueError(f"manwa-sheet slice missing source_page_key: {raw}")
    return {
        "source_page_key": key,
        "source_y_start": int(raw.get("source_y_start") or 0),
        "source_y_end": int(raw.get("source_y_end") or 0),
        "sheet_y_start": int(raw.get("sheet_y_start") or 0),
        "sheet_y_end": int(raw.get("sheet_y_end") or 0),
        "source_page_width": int(raw.get("source_page_width") or 0),
        "source_page_height": int(raw.get("source_page_height") or 0),
        "source_page_etag": str(raw.get("source_page_etag") or ""),
    }


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

        # Swap the Python list manifest for a shared-memory tensor-backed store
        # to fix the fork-CoW refcount memory leak (PyTorch #13246).
        # `self.samples` is dead after the init loop above — only referenced
        # at init time — so wrapping it is always safe.
        self.samples = _TorchSerializedList(self.samples)
        # `self.file_list` is read-only post-init ONLY in native mode without
        # latent caching: `_get_single_item` already deepcopies before mutating
        # (line ~1238), so workers never write the underlying entries. Latent
        # caching (`cache_latents_all_latents`) and POI-rebuild bucketing both
        # mutate file_items in place — keep them on the plain list.
        if not self.is_caching_latents and not self.use_bucket_batches:
            self.file_list = _TorchSerializedList(self.file_list)

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

    def _source_ref_options(self, raw_value: Any) -> Dict[str, Any]:
        if not isinstance(raw_value, dict):
            return {}
        options: Dict[str, Any] = {}
        crop_box = raw_value.get("crop_box") or raw_value.get("bbox")
        if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
            options["crop_box"] = [float(value) for value in crop_box]
        pad_multiple = raw_value.get("pad_multiple")
        if pad_multiple is not None:
            options["pad_multiple"] = max(1, int(pad_multiple))
        # Border options used when a control image needs a coloured frame.
        border_width = raw_value.get("border_width")
        if border_width is not None:
            options["border_width"] = max(0, int(border_width))
        border_rgb = raw_value.get("border_rgb")
        if isinstance(border_rgb, (list, tuple)) and len(border_rgb) == 3:
            options["border_rgb"] = [int(v) for v in border_rgb]
        return options

    def _parse_source_ref(self, path_or_key: Any, allow_relative: bool = True) -> Dict[str, str]:
        source_options = self._source_ref_options(path_or_key)

        def with_options(source_ref: Dict[str, Any]) -> Dict[str, Any]:
            if source_options:
                source_ref.update(source_options)
            return source_ref

        # Manwa sheet refs carry a ``slices`` list instead of a single
        # ``image``/``path``. Each slice points at one source page that
        # contributes a vertical band to the sheet image. The materializer
        # downloads each source page once, crops per slice, and stitches the
        # band stack to a cached JPEG on disk.
        if isinstance(path_or_key, dict) and isinstance(path_or_key.get("slices"), list) and path_or_key["slices"]:
            slices = [_normalize_slice_entry(s) for s in path_or_key["slices"]]
            bucket_hint = None
            for s in slices:
                src_key = s.get("source_page_key") or ""
                if src_key.startswith("s3://"):
                    bucket_hint, _, _ = src_key[5:].partition("/")
                    break
            if bucket_hint is None:
                if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "s3":
                    bucket_hint = self.dataset_root_ref["bucket"]
                elif self.manifest_ref["kind"] == "s3":
                    bucket_hint = self.manifest_ref["bucket"]
            canonical = "manwa_sheet:" + hashlib.sha1(
                json.dumps(slices, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            return with_options({
                "kind": "manwa_sheet",
                "slices": slices,
                "bucket": bucket_hint,
                "canonical": canonical,
            })

        if isinstance(path_or_key, dict):
            path_or_key = path_or_key.get("image") or path_or_key.get("path")
        if not path_or_key:
            raise ValueError("Expected a non-empty manifest image path")

        raw_value = str(path_or_key).strip()
        if raw_value.startswith("s3://"):
            bucket, key = raw_value[5:].split("/", 1)
            return with_options({
                "kind": "s3",
                "bucket": bucket,
                "key": key.lstrip("/"),
                "canonical": f"s3://{bucket}/{key.lstrip('/')}",
            })
        if raw_value.startswith("file://"):
            local_path = os.path.abspath(unquote(urlparse(raw_value).path))
            return with_options({
                "kind": "local",
                "path": local_path,
                "canonical": local_path,
            })
        if raw_value.startswith("http://") or raw_value.startswith("https://"):
            raise ValueError(f"HTTP(S) sources are not supported in ManifestDataset: {raw_value}")
        if os.path.isabs(raw_value):
            return with_options({
                "kind": "local",
                "path": os.path.abspath(raw_value),
                "canonical": os.path.abspath(raw_value),
            })
        if raw_value.startswith("datasets/"):
            bucket = None
            if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "s3":
                bucket = self.dataset_root_ref["bucket"]
            elif self.manifest_ref["kind"] == "s3":
                bucket = self.manifest_ref["bucket"]
            if bucket is not None:
                key = raw_value.lstrip("/")
                return with_options({
                    "kind": "s3",
                    "bucket": bucket,
                    "key": key,
                    "canonical": f"s3://{bucket}/{key}",
                })
        if not allow_relative:
            local_path = os.path.abspath(raw_value)
            return with_options({
                "kind": "local",
                "path": local_path,
                "canonical": local_path,
            })

        if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "s3":
            prefix = self.dataset_root_ref.get("key", "").strip("/")
            key = raw_value.lstrip("/")
            alternate_keys = []
            if prefix and not key.startswith(prefix):
                alternate_keys.append(key)
                key = f"{prefix}/{key}"
            return with_options({
                "kind": "s3",
                "bucket": self.dataset_root_ref["bucket"],
                "key": key,
                "canonical": f"s3://{self.dataset_root_ref['bucket']}/{key}",
                "alternate_keys": alternate_keys,
            })

        if self.manifest_ref["kind"] == "s3":
            prefix = self.manifest_ref["key"].rsplit("/", 1)[0]
            key = raw_value.lstrip("/")
            if prefix and not key.startswith(prefix):
                key = f"{prefix}/{key}"
            return with_options({
                "kind": "s3",
                "bucket": self.manifest_ref["bucket"],
                "key": key,
                "canonical": f"s3://{self.manifest_ref['bucket']}/{key}",
            })

        base_dir = (
            self.dataset_root_ref["path"]
            if self.dataset_root_ref is not None and self.dataset_root_ref["kind"] == "local"
            else os.path.dirname(self.manifest_ref["path"])
        )
        local_path = os.path.abspath(os.path.join(base_dir, raw_value))
        return with_options({
            "kind": "local",
            "path": local_path,
            "canonical": local_path,
        })

    def _materialize_ref(self, source_ref: Dict[str, str]) -> str:
        if source_ref["kind"] == "local":
            return source_ref["path"]

        if source_ref["kind"] == "manwa_sheet":
            return self._materialize_manwa_sheet(source_ref)

        mounted_path = self._mounted_path_for_s3_ref(source_ref)
        if mounted_path and os.path.exists(mounted_path) and not source_ref.get("crop_box"):
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

    def _materialize_manwa_sheet(self, source_ref: Dict[str, Any]) -> str:
        """Reconstruct a manwa sheet image from its slice plan and cache to disk.

        The sheet is produced once per slice-plan hash and reused across
        epochs. Each slice's source page is downloaded once (boto3 reads),
        cropped to its y-band, and pasted into a vertically stacked canvas.
        The resulting JPEG path is returned for downstream PIL.Image.open.
        """
        slices = source_ref.get("slices") or []
        if not slices:
            raise ValueError("manwa-sheet source_ref carries no slices")
        digest = hashlib.sha1(
            json.dumps(slices, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        local_path = self.sample_cache_dir / digest[:2] / f"manwa_sheet_{digest}.jpg"
        if local_path.exists():
            return str(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        bucket = source_ref.get("bucket") or (
            self.dataset_root_ref.get("bucket") if self.dataset_root_ref else None
        )
        if not bucket:
            raise ValueError("manwa-sheet source_ref has no resolvable bucket")

        # Download each unique source page once (s3 mounted fast-path if
        # available), then crop per-slice.
        source_images: Dict[str, Image.Image] = {}
        for entry in slices:
            key = entry["source_page_key"]
            if key in source_images:
                continue
            if key.startswith("s3://"):
                key_no_scheme = key[5:]
                _, _, page_key = key_no_scheme.partition("/")
            else:
                page_key = key
            mounted_path = None
            if hasattr(self, "_mounted_path_for_s3_ref"):
                mounted_path = self._mounted_path_for_s3_ref({"bucket": bucket, "key": page_key})
            if mounted_path and os.path.exists(mounted_path):
                with Image.open(mounted_path) as im:
                    source_images[key] = im.convert("RGB")
            else:
                response = self.s3_client.get_object(Bucket=bucket, Key=page_key)
                data = response["Body"].read()
                from io import BytesIO

                with Image.open(BytesIO(data)) as im:
                    source_images[key] = im.convert("RGB")

        # Drift guard: every slice carries the dimensions the annotator saw.
        # If the underlying page changed (re-encoded, resized, swapped) the
        # bboxes in the annotation no longer line up with reconstructed
        # coordinates, so refuse to silently emit a mismatched sheet.
        for s in slices:
            recorded_w = int(s.get("source_page_width") or 0)
            recorded_h = int(s.get("source_page_height") or 0)
            src = source_images[s["source_page_key"]]
            if recorded_w and recorded_h and (src.width != recorded_w or src.height != recorded_h):
                raise ValueError(
                    "manwa sheet source-page dimension drift: "
                    f"{s['source_page_key']} recorded={recorded_w}x{recorded_h} "
                    f"observed={src.width}x{src.height}"
                )

        target_width = max(im.width for im in source_images.values())
        sorted_slices = sorted(slices, key=lambda s: int(s["sheet_y_start"]))
        # Slice plan must be contiguous and cover [0, total_h).
        expected = 0
        for s in sorted_slices:
            sy0 = int(s["sheet_y_start"])
            sy1 = int(s["sheet_y_end"])
            if sy0 != expected:
                raise ValueError(
                    f"manwa sheet slice plan non-contiguous: expected sheet_y_start={expected}, got {sy0}"
                )
            if sy1 <= sy0:
                raise ValueError(f"manwa sheet slice has non-positive height: {s}")
            expected = sy1
        total_h = expected
        sheet = Image.new("RGB", (int(target_width), int(total_h)), (255, 255, 255))
        for s in sorted_slices:
            src = source_images[s["source_page_key"]]
            crop = src.crop((0, int(s["source_y_start"]), src.width, int(s["source_y_end"])))
            try:
                if crop.width != target_width:
                    new_h = int(round(crop.height * target_width / crop.width))
                    crop = crop.resize((target_width, new_h), Image.BICUBIC)
                sheet.paste(crop, (0, int(s["sheet_y_start"])))
            finally:
                crop.close()
        fd, tmp_name = tempfile.mkstemp(
            suffix=local_path.suffix + ".tmp",
            prefix=local_path.stem + ".",
            dir=local_path.parent,
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            sheet.save(tmp_path, format="JPEG", quality=92, optimize=True)
            os.replace(tmp_path, local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            sheet.close()
            for im in source_images.values():
                im.close()
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

        if source_ref["kind"] == "manwa_sheet":
            # The slice plan itself is the deterministic identity. We hash
            # the canonical JSON so cache keys are stable across processes.
            slice_blob = json.dumps(source_ref.get("slices") or [], sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha1(slice_blob.encode("utf-8")).hexdigest()
            return {
                "kind": "manwa_sheet",
                "bucket": source_ref.get("bucket"),
                "slice_count": len(source_ref.get("slices") or []),
                "slice_digest": digest,
                "canonical": source_ref.get("canonical"),
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

    def _apply_source_ref_geometry(self, image: Image.Image, source_ref: Dict[str, Any]) -> Image.Image:
        crop_box = source_ref.get("crop_box")
        if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
            width, height = image.size
            x0, y0, x1, y1 = [float(value) for value in crop_box]
            x0 = max(0, min(width, int(math.floor(x0))))
            y0 = max(0, min(height, int(math.floor(y0))))
            x1 = max(0, min(width, int(math.ceil(x1))))
            y1 = max(0, min(height, int(math.ceil(y1))))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Invalid manifest crop_box for {source_ref['canonical']}: {crop_box}")
            image = image.crop((x0, y0, x1, y1))

        pad_multiple = int(source_ref.get("pad_multiple") or 0)
        border_width = int(source_ref.get("border_width") or 0)
        border_rgb = source_ref.get("border_rgb")
        if border_width > 0 and isinstance(border_rgb, (list, tuple)) and len(border_rgb) == 3:
            if pad_multiple > 1:
                inner_width = int(math.ceil((image.width + border_width * 2) / pad_multiple) * pad_multiple) - border_width * 2
                inner_height = int(math.ceil((image.height + border_width * 2) / pad_multiple) * pad_multiple) - border_width * 2
                if inner_width != image.width or inner_height != image.height:
                    padded = Image.new("RGB", (max(image.width, inner_width), max(image.height, inner_height)), (255, 255, 255))
                    padded.paste(image, (0, 0))
                    image = padded
            image = ImageOps.expand(image, border=border_width, fill=tuple(int(value) for value in border_rgb))
        elif pad_multiple > 1:
            padded_width = int(math.ceil(image.width / pad_multiple) * pad_multiple)
            padded_height = int(math.ceil(image.height / pad_multiple) * pad_multiple)
            if (padded_width, padded_height) != image.size:
                padded = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
                padded.paste(image, (0, 0))
                image = padded
        return image

    def _load_image(self, source_ref: Dict[str, str]) -> Image.Image:
        local_path = self._materialize_ref(source_ref)
        # Context-managed open + explicit close on the rotated intermediate
        # releases the file descriptor and the PNG tile decoder's internal
        # buffers immediately. Without this, 32 worker procs accumulate
        # un-closed lazy Image fps + tile state, contributing to the slow
        # host RAM creep documented in production runs (Pillow #7961, #5180).
        with Image.open(local_path) as src:
            src.load()
            rotated = exif_transpose(src)
            image = rotated.convert("RGB")
            if rotated is not src:
                rotated.close()
        return self._apply_source_ref_geometry(image, source_ref)

    @staticmethod
    def _norm_box_to_pixel_box(box_norm: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
        if not isinstance(box_norm, list) or len(box_norm) != 4:
            return None
        x0 = int(round(max(0.0, min(1.0, float(box_norm[0]))) * width))
        y0 = int(round(max(0.0, min(1.0, float(box_norm[1]))) * height))
        x1 = int(round(max(0.0, min(1.0, float(box_norm[2]))) * width))
        y1 = int(round(max(0.0, min(1.0, float(box_norm[3]))) * height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _draw_metadata_box(
        draw: ImageDraw.ImageDraw,
        box_norm: list[float],
        width: int,
        height: int,
        color,
        line_width: int = 6,
    ) -> bool:
        pixel_box = ManifestDataset._norm_box_to_pixel_box(box_norm, width, height)
        if pixel_box is None:
            return False
        draw.rectangle(pixel_box, outline=tuple(int(value) for value in color), width=max(1, int(line_width)))
        return True

    @staticmethod
    def _draw_metadata_text_region(
        draw: ImageDraw.ImageDraw,
        box_norm: list[float],
        width: int,
        height: int,
        text_bubble_type: str,
    ) -> bool:
        normalized = " ".join(str(text_bubble_type or "").split()).strip()
        if normalized.lower() in {"", "none", "null"}:
            return False
        pixel_box = ManifestDataset._norm_box_to_pixel_box(box_norm, width, height)
        if pixel_box is None:
            return False
        text_color = LAYOUT_TEXT_COLORS.get(normalized)
        if text_color is None:
            return False
        draw.rectangle(pixel_box, fill=text_color)
        return True

    def _layout_control_from_metadata(self, layout_metadata: dict[str, Any], width: int, height: int) -> Image.Image:
        image = Image.new("RGB", (width, height), LAYOUT_BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        for character in layout_metadata.get("characters") or []:
            if not isinstance(character, dict):
                continue
            box_norm = character.get("bbox_norm")
            rgb = character.get("rgb")
            if isinstance(box_norm, list) and isinstance(rgb, list):
                self._draw_metadata_box(
                    draw,
                    box_norm,
                    width,
                    height,
                    rgb,
                    line_width=int(character.get("line_width") or 6),
                )
        text_payload = layout_metadata.get("text") if isinstance(layout_metadata.get("text"), dict) else {}
        for region in text_payload.get("regions") or []:
            if not isinstance(region, dict):
                continue
            self._draw_metadata_text_region(
                draw,
                region.get("bbox_norm"),
                width,
                height,
                str(region.get("type") or "Speech Bubble"),
            )
        return image

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
        tensors: List[torch.Tensor] = []

        def append_control_image(control_img: Image.Image, canonical: str) -> None:
            control_img = self._apply_flip(control_img, file_item)
            control_img = self._apply_control_geometry(control_img, file_item)
            control_width, control_height = control_img.size
            target_multiple = self._target_multiple()
            if control_width <= 0 or control_height <= 0:
                raise ValueError(
                    f"Control image has invalid geometry: {canonical} = "
                    f"{control_width}x{control_height}"
                )
            if control_width % target_multiple != 0 or control_height % target_multiple != 0:
                padded_width = int(math.ceil(control_width / target_multiple) * target_multiple)
                padded_height = int(math.ceil(control_height / target_multiple) * target_multiple)
                padded = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
                padded.paste(control_img, (0, 0))
                control_img = padded
            tensors.append(self.control_transform(control_img))

        layout_control = controls.get("layout_control_path")
        if layout_control:
            control_refs.append(self._parse_source_ref(layout_control))
        elif isinstance(controls.get("layout_control"), dict):
            append_control_image(
                self._layout_control_from_metadata(
                    controls["layout_control"],
                    int(file_item.width),
                    int(file_item.height),
                ),
                "manifest:layout_control",
            )

        prev_control = controls.get("previous_panel") or controls.get("previous_page")
        if prev_control and len(control_refs) < MAX_CONTROL_IMAGE_SLOTS:
            control_refs.append(self._parse_source_ref(prev_control))

        remaining_character_slots = max(0, MAX_CONTROL_IMAGE_SLOTS - len(control_refs))
        max_character_refs = min(self.config.max_character_refs, remaining_character_slots)
        for char_ref_path in controls.get("character_ref_paths", [])[:max_character_refs]:
            control_refs.append(self._parse_source_ref(char_ref_path))

        if not control_refs and not tensors:
            return

        for control_ref in control_refs:
            control_img = self._load_image(control_ref)
            append_control_image(control_img, control_ref["canonical"])

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
