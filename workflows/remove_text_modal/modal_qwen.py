"""Modal app: Qwen-Image-Edit-2511 text removal on H100.

Production configurations:

- ``QwenVanilla``   — full 40-step bf16, matches fal-ai/qwen-image-edit-2511 settings
- ``QwenLightning`` — 4-step Lightning distilled LoRA (lightx2v/Qwen-Image-Edit-2511-Lightning)
- ``QwenLightning8Step`` — 8-step Lightning distilled LoRA, slower but less aggressive

Experimental smoke/benchmark-only configurations:

- ``QwenFp8Single`` — 40-step fp8 single-file base transformer; loads but expands to bf16 in Diffusers
- ``QwenFp8RuntimeWO`` — 40-step runtime torchao fp8 weight-only transformer; slower than vanilla

Inspired by ``workflows/manga_annotate/modal_magi.py`` — same @app.cls +
@modal.enter pattern, HF cache volume, S3 read/write, ``_failed/`` log on error.
The Qwen model is ~57 GB at bf16 so memory snapshots aren't used (too large);
instead the HF cache volume keeps weights warm across container starts.

Run modes
---------

Bulk annotate (drives 40 H100s)::

    modal run modal_qwen.py::annotate_manifest_local \
      --manifest-path /tmp/manifest.jsonl --variant lightning
    modal run modal_qwen.py::annotate_manifest_local \
      --manifest-path /tmp/manifest.jsonl --variant lightning_8step

Smoke (one page each)::

    modal run modal_qwen.py::smoke_test --variant lightning

Compare quality on N test pages (writes PNGs to artifacts/)::

    modal run modal_qwen.py::compare_variants --pages 8

Benchmark batch sweep on 1 H100 each::

    modal run modal_qwen.py::run_benchmark --variant lightning --n-images 16
    modal run modal_qwen.py::run_benchmark --variant lightning_8step --n-images 8
"""

from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


QWEN_REPO = "Qwen/Qwen-Image-Edit-2511"
LIGHTNING_LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
QWEN_FP8_BASE_WEIGHT = "qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors"
QWEN_FP8_LIGHTNING_4STEP_WEIGHT = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors"
QWEN_FP8_LIGHTNING_8STEP_WEIGHT = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"
HF_HOME = "/root/.cache/huggingface"

S3_BUCKET_DEFAULT = os.environ.get("DRAWTOON_S3_BUCKET", "drawtoon")
AWS_REGION_DEFAULT = os.environ.get("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3")
MODAL_REGION = os.environ.get("DRAWTOON_MODAL_REGION", "us-east-1")

DEFAULT_MAX_CONTAINERS = int(os.environ.get("QWEN_MAX_CONTAINERS", "40"))
DEFAULT_GPU_BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "1"))
DEFAULT_GPU_TYPE = os.environ.get("QWEN_GPU", "H100")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Default Qwen-Image-Edit-2511 settings from the model card
VANILLA_STEPS = 40
VANILLA_TRUE_CFG = 4.0
LIGHTNING_STEPS = 4
LIGHTNING_TRUE_CFG = 1.0

NEGATIVE_PROMPT = " "  # The model card uses a single space


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "accelerate==1.4.0",
        "boto3==1.35.99",
        "diffusers==0.36.0",
        "einops==0.8.0",
        "hf_transfer==0.1.8",
        "hf_xet==1.4.3",
        "pillow==11.1.0",
        "safetensors==0.5.2",
        "sentencepiece==0.2.0",
        "peft==0.17.0",
        "torch==2.5.1",
        "torchao==0.12.0",
        "torchvision==0.20.1",
        "transformers==4.57.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": HF_HOME,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "QWEN_40STEP_COMPILE_BLOCKS": os.environ.get("QWEN_40STEP_COMPILE_BLOCKS", "0"),
        "QWEN_40STEP_COMPILE_DYNAMIC": os.environ.get("QWEN_40STEP_COMPILE_DYNAMIC", "0"),
        "QWEN_40STEP_FAST_MATH": os.environ.get("QWEN_40STEP_FAST_MATH", "1"),
        "QWEN_40STEP_FUSE_QKV": os.environ.get("QWEN_40STEP_FUSE_QKV", "0"),
        "QWEN_40STEP_MEMORY_MODE": os.environ.get("QWEN_40STEP_MEMORY_MODE", "auto"),
        "QWEN_40STEP_MIN_FULL_GPU_GB": os.environ.get("QWEN_40STEP_MIN_FULL_GPU_GB", "110"),
        "QWEN_40STEP_WARMUP_STEPS": os.environ.get("QWEN_40STEP_WARMUP_STEPS", "1"),
    })
    .add_local_dir(
        str(Path(__file__).resolve().parent / "prompts"),
        remote_path="/root/prompts",
        copy=False,
    )
)

app = modal.App("drawtoon-remove-text-modal", image=image)
hf_volume = modal.Volume.from_name("qwen-image-edit-hf-cache", create_if_missing=True)
aws_secret = modal.Secret.from_name(AWS_SECRET_NAME)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_prompt() -> str:
    path = Path("/root/prompts/master_prompt.md")
    if not path.exists():
        # Local fallback when running locally
        path = Path(__file__).resolve().parent / "prompts" / "master_prompt.md"
    return path.read_text(encoding="utf-8").strip()


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


def _s3_client():
    import boto3
    from botocore.config import Config

    if not hasattr(_s3_client, "_client"):
        _s3_client._client = boto3.client(  # type: ignore[attr-defined]
            "s3",
            region_name=AWS_REGION_DEFAULT,
            config=Config(
                retries={"mode": "adaptive", "max_attempts": 10},
                connect_timeout=10,
                read_timeout=120,
                max_pool_connections=128,
            ),
        )
    return _s3_client._client  # type: ignore[attr-defined]


def _head_exists(bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _download_rgb_image(bucket: str, key: str) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageOps

    response = _s3_client().get_object(Bucket=bucket, Key=key)
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


def _put_png_to_s3(bucket: str, key: str, image_obj: Any) -> int:
    buf = io.BytesIO()
    image_obj.save(buf, format="PNG", optimize=True)
    body = buf.getvalue()
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="image/png",
    )
    return len(body)


def _put_json_to_s3(bucket: str, key: str, payload: dict[str, Any]) -> None:
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple // 2) // multiple) * multiple)


def _normalize_size(image_obj: Any, *, target_long: int = 1024, multiple: int = 16) -> Any:
    """Resize to longest side == target_long and snap to multiple-of-16 dims."""
    w, h = image_obj.size
    if max(w, h) == target_long and w % multiple == 0 and h % multiple == 0:
        return image_obj
    scale = target_long / float(max(w, h))
    new_w = _round_to_multiple(int(round(w * scale)), multiple)
    new_h = _round_to_multiple(int(round(h * scale)), multiple)
    from PIL import Image

    return image_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Pipeline classes — one per variant
# ---------------------------------------------------------------------------


GPU_KW = dict(
    region=MODAL_REGION,
    gpu=DEFAULT_GPU_TYPE,
    timeout=3600,
    startup_timeout=1800,
    cpu=8.0,
    memory=65536,
    secrets=[aws_secret],
    volumes={HF_HOME: hf_volume},
    max_containers=DEFAULT_MAX_CONTAINERS,
    scaledown_window=300,
)


LIGHTNING_4STEP_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_8STEP_WEIGHT = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _modal_gpu_request(value: str) -> str | list[str]:
    """Parse a Modal GPU fallback list from an env var, e.g. ``H200,H100``."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return "H100"
    return items if len(items) > 1 else items[0]


# 40-step vanilla is the quality path: 40 denoise steps and true CFG 4.0.
# Prefer H200 so the full pipeline can stay resident on GPU; H100 remains a
# fallback with automatic CPU offload when VRAM is not enough.
VANILLA_40STEP_GPU_KW = {
    **GPU_KW,
    "gpu": _modal_gpu_request(os.environ.get("QWEN_40STEP_GPU", "H200,H100")),
    "memory": _env_int("QWEN_40STEP_CPU_MEMORY_MB", 131072),
    "max_containers": _env_int("QWEN_40STEP_MAX_CONTAINERS", DEFAULT_MAX_CONTAINERS),
    "scaledown_window": _env_int("QWEN_40STEP_SCALEDOWN_WINDOW", 900),
}

VANILLA_40STEP_FAST_MATH = _env_flag("QWEN_40STEP_FAST_MATH", True)
VANILLA_40STEP_FUSE_QKV = _env_flag("QWEN_40STEP_FUSE_QKV", False)
VANILLA_40STEP_COMPILE_BLOCKS = _env_flag("QWEN_40STEP_COMPILE_BLOCKS", False)
VANILLA_40STEP_COMPILE_DYNAMIC = _env_flag("QWEN_40STEP_COMPILE_DYNAMIC", False)
VANILLA_40STEP_WARMUP_STEPS = _env_int("QWEN_40STEP_WARMUP_STEPS", 1)
VANILLA_40STEP_MIN_FULL_GPU_GB = float(os.environ.get("QWEN_40STEP_MIN_FULL_GPU_GB", "110"))

BENCHMARK_GPU_KW = {
    **GPU_KW,
    "gpu": _modal_gpu_request(os.environ.get("QWEN_BENCHMARK_GPU", "H200,H100")),
    "memory": _env_int("QWEN_BENCHMARK_CPU_MEMORY_MB", 131072),
    "max_containers": 2,
}


def _load_pipeline(
    *,
    with_lightning: bool,
    lightning_weight_name: str | None = None,
    compile_transformer: bool = False,
    compile_repeated_blocks: bool = False,
    compile_dynamic: bool = False,
    compile_fullgraph: bool = True,
    fast_math: bool = False,
    fuse_qkv: bool = False,
    fp8_transformer_mode: str = "",
    fp8_single_file_name: str | None = None,
):
    """Load QwenImageEditPlusPipeline + optional Lightning LoRA / compile / fp8."""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    if fast_math:
        _enable_cuda_fast_math(torch)

    if fp8_single_file_name:
        from diffusers import QwenImageTransformer2DModel
        from huggingface_hub import hf_hub_download

        transformer_path = hf_hub_download(
            repo_id=LIGHTNING_LORA_REPO,
            filename=fp8_single_file_name,
            cache_dir=HF_HOME,
        )
        transformer = QwenImageTransformer2DModel.from_single_file(
            transformer_path,
            config=QWEN_REPO,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        print(f"[opt] loaded fp8 single-file transformer: {fp8_single_file_name}", flush=True)
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_REPO,
            torch_dtype=torch.bfloat16,
        )

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    if with_lightning:
        kwargs: dict[str, Any] = {}
        if lightning_weight_name:
            kwargs["weight_name"] = lightning_weight_name
        pipe.load_lora_weights(LIGHTNING_LORA_REPO, **kwargs)
        pipe.fuse_lora()

    if fuse_qkv:
        try:
            pipe.transformer.fuse_qkv_projections()
            print("[opt] fused transformer QKV projections", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[opt] qkv fusion failed: {exc!r}", flush=True)

    if fp8_transformer_mode:
        # torchao 0.12.0 + H100. Filter to nn.Linear only (skip norms /
        # embeddings) to avoid quality regressions.
        try:
            import torch.nn as nn
            from torchao.quantization import (
                Float8WeightOnlyConfig,
                Float8DynamicActivationFloat8WeightConfig,
                quantize_,
            )
            from torchao.quantization.granularity import PerRow

            if fp8_transformer_mode == "weight_only":
                cfg = Float8WeightOnlyConfig()
            elif fp8_transformer_mode == "dynamic":
                cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
            else:
                raise ValueError(f"unknown fp8_transformer_mode={fp8_transformer_mode!r}")
            quantize_(
                pipe.transformer,
                cfg,
                filter_fn=lambda m, _: isinstance(m, nn.Linear),
            )
            print(f"[opt] applied torchao fp8 ({fp8_transformer_mode}) to transformer.Linear layers", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[opt] fp8 quantization failed ({fp8_transformer_mode}): {exc!r}", flush=True)

    if compile_repeated_blocks:
        try:
            _configure_torch_compile(torch)
            pipe.transformer.compile_repeated_blocks(
                mode="default",
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
            )
            print(
                "[opt] torch.compile applied via "
                f"compile_repeated_blocks(mode=default, fullgraph={compile_fullgraph}, dynamic={compile_dynamic})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[opt] compile_repeated_blocks failed: {exc!r}", flush=True)

    if compile_transformer:
        # Compiling the whole transformer trips TorchDynamo on Qwen-Image's
        # pos_embed module ("'int' object has no attribute 'pos_freqs'").
        # Workaround: compile only the inner transformer_blocks. They're where
        # >95% of the FLOPs live, and they have nicely traceable shapes.
        compiled = 0
        try:
            blocks = pipe.transformer.transformer_blocks
            for i, block in enumerate(blocks):
                # mode="default" is safe; "reduce-overhead" uses CUDA graphs
                # which clobber residual tensors in Qwen blocks.
                blocks[i] = torch.compile(
                    block,
                    mode="default",
                    fullgraph=False,
                    dynamic=False,
                )
                compiled += 1
            print(f"[opt] torch.compile applied to {compiled} transformer_blocks (mode=default)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[opt] block-level compile failed: {exc!r}", flush=True)

    _log_transformer_footprint(pipe.transformer)
    return pipe


def _enable_cuda_fast_math(torch_mod: Any) -> None:
    """Enable CUDA math backends that preserve the 40-step/CFG-4 pipeline path."""
    try:
        torch_mod.backends.cuda.matmul.allow_tf32 = True
        torch_mod.backends.cudnn.allow_tf32 = True
        torch_mod.set_float32_matmul_precision("high")
        torch_mod.backends.cuda.enable_flash_sdp(True)
        torch_mod.backends.cuda.enable_mem_efficient_sdp(True)
        torch_mod.backends.cuda.enable_math_sdp(True)
        print("[opt] enabled CUDA fast-math backend flags", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[opt] enabling CUDA fast-math flags failed: {exc!r}", flush=True)


def _configure_torch_compile(torch_mod: Any) -> None:
    """Compiler settings from Diffusers' inference optimization guide."""
    try:
        torch_mod._inductor.config.conv_1x1_as_mm = True
        torch_mod._inductor.config.coordinate_descent_tuning = True
        torch_mod._inductor.config.epilogue_fusion = False
        torch_mod._inductor.config.coordinate_descent_check_all_directions = True
        print("[opt] configured TorchInductor compile flags", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[opt] TorchInductor compile flag setup failed: {exc!r}", flush=True)


def _cuda_total_memory_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.get_device_properties(0).total_memory / (1024**3))


def _configure_40step_memory(pipe: Any, *, min_full_gpu_gb: float = VANILLA_40STEP_MIN_FULL_GPU_GB) -> str:
    """Keep 40-step CFG on GPU when VRAM allows, otherwise fit safely."""
    total_gb = _cuda_total_memory_gb()
    mode = os.environ.get("QWEN_40STEP_MEMORY_MODE", "auto").strip().lower()
    if mode not in {"auto", "full_gpu", "cpu_offload"}:
        raise ValueError("QWEN_40STEP_MEMORY_MODE must be one of: auto, full_gpu, cpu_offload")

    use_cpu_offload = mode == "cpu_offload" or (mode == "auto" and total_gb < min_full_gpu_gb)
    if use_cpu_offload:
        print(
            json.dumps({
                "event": "40step_memory_mode",
                "mode": "cpu_offload",
                "total_vram_gb": round(total_gb, 1),
                "min_full_gpu_gb": min_full_gpu_gb,
            }),
            flush=True,
        )
        try:
            pipe.enable_model_cpu_offload()
        except Exception as exc:  # noqa: BLE001
            print(f"[mem] enable_model_cpu_offload failed: {exc!r}", flush=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        return "cpu_offload"

    print(
        json.dumps({
            "event": "40step_memory_mode",
            "mode": "full_gpu",
            "total_vram_gb": round(total_gb, 1),
            "min_full_gpu_gb": min_full_gpu_gb,
        }),
        flush=True,
    )
    return "full_gpu"


def _warmup_edit_pipe(
    pipe: Any,
    *,
    steps: int,
    true_cfg_scale: float,
    size: tuple[int, int] = (1024, 1024),
    prompt: str = "warmup",
) -> None:
    """Run a real-shape CFG warmup before work arrives."""
    try:
        from PIL import Image
        import torch

        dummy = Image.new("RGB", size, (128, 128, 128))
        warmup_steps = max(1, min(int(steps), VANILLA_40STEP_WARMUP_STEPS))
        with torch.inference_mode():
            _ = pipe(
                image=[dummy],
                prompt=[prompt],
                negative_prompt=[NEGATIVE_PROMPT],
                num_inference_steps=warmup_steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
        print(
            json.dumps({
                "event": "warmup_done",
                "steps": warmup_steps,
                "true_cfg_scale": true_cfg_scale,
                "size": list(size),
            }),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] skipped: {exc!r}", flush=True)


def _enable_batched_true_cfg(pipe: Any) -> None:
    """Batch Qwen true-CFG cond/uncond transformer passes into one H200-sized call."""
    import torch
    import numpy as np
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
        QwenImagePipelineOutput,
        XLA_AVAILABLE,
        calculate_dimensions,
        calculate_shift,
        logger,
        retrieve_timesteps,
    )

    base_cls = pipe.__class__
    if getattr(base_cls, "_drawtoon_batched_true_cfg", False):
        return

    @torch.no_grad()
    def _batched_true_cfg_call(
        self,
        image: Any | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: Any | None = None,
        latents: Any | None = None,
        prompt_embeds: Any | None = None,
        prompt_embeds_mask: Any | None = None,
        negative_prompt_embeds: Any | None = None,
        negative_prompt_embeds_mask: Any | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Any | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if batch_size > 1:
            raise ValueError(
                f"QwenImageEditPlusPipeline currently only supports batch_size=1, but received batch_size={batch_size}."
            )

        device = self._execution_device
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
        else:
            condition_images = image
            vae_image_sizes = [(width, height)]
            vae_images = image

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if do_true_cfg:
                    cfg_hidden_states = torch.cat([latent_model_input, latent_model_input], dim=0)
                    cfg_timestep = torch.cat([timestep, timestep], dim=0)
                    cfg_guidance = torch.cat([guidance, guidance], dim=0) if guidance is not None else None
                    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                        max_prompt_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])

                        def _pad_seq(tensor: Any, *, value: float = 0.0) -> Any:
                            pad_len = max_prompt_len - tensor.shape[1]
                            if pad_len <= 0:
                                return tensor
                            pad_shape = list(tensor.shape)
                            pad_shape[1] = pad_len
                            return torch.cat(
                                [tensor, torch.full(pad_shape, value, device=tensor.device, dtype=tensor.dtype)],
                                dim=1,
                            )

                        prompt_embeds = _pad_seq(prompt_embeds)
                        negative_prompt_embeds = _pad_seq(negative_prompt_embeds)
                        prompt_embeds_mask = _pad_seq(prompt_embeds_mask)
                        negative_prompt_embeds_mask = _pad_seq(negative_prompt_embeds_mask)
                    cfg_prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                    cfg_prompt_mask = torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0)
                    cfg_txt_seq_lens = [int(x) for x in cfg_prompt_mask.to(torch.bool).sum(dim=1).tolist()]
                    with self.transformer.cache_context("batched_cfg"):
                        cfg_noise_pred = self.transformer(
                            hidden_states=cfg_hidden_states,
                            timestep=cfg_timestep / 1000,
                            guidance=cfg_guidance,
                            encoder_hidden_states_mask=cfg_prompt_mask,
                            encoder_hidden_states=cfg_prompt_embeds,
                            img_shapes=img_shapes * 2,
                            txt_seq_lens=cfg_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    cfg_noise_pred = cfg_noise_pred[:, : latents.size(1)]
                    noise_pred, neg_noise_pred = cfg_noise_pred.chunk(2, dim=0)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                else:
                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    from torch_xla.core import xla_model as xm

                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)

    batched_cls = type(
        "DrawtoonBatchedCFGQwenImageEditPlusPipeline",
        (base_cls,),
        {"__call__": _batched_true_cfg_call, "_drawtoon_batched_true_cfg": True},
    )
    pipe.__class__ = batched_cls
    print("[opt] enabled batched true-CFG transformer pass", flush=True)


def _log_transformer_footprint(transformer: Any) -> None:
    try:
        counts: dict[str, int] = {}
        bytes_total = 0
        for param in transformer.parameters():
            dtype_name = str(param.dtype).replace("torch.", "")
            counts[dtype_name] = counts.get(dtype_name, 0) + int(param.numel())
            bytes_total += int(param.numel()) * int(param.element_size())
        by_dtype = {k: round(v / 1_000_000_000, 3) for k, v in sorted(counts.items())}
        print(
            json.dumps({
                "event": "transformer_footprint",
                "param_billion_by_dtype": by_dtype,
                "estimated_param_gb": round(bytes_total / (1024**3), 2),
            }),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[opt] transformer footprint logging failed: {exc!r}", flush=True)


def _edit_batch_impl(
    payload: dict[str, Any],
    *,
    pipe: Any,
    variant: str,
    steps: int,
    true_cfg_scale: float,
    target_long_side: int = 1024,
) -> dict[str, Any]:
    import torch

    pages = list(payload.get("pages") or [])
    if not pages:
        raise ValueError("payload.pages is required and non-empty")

    bucket = str(payload.get("bucket") or S3_BUCKET_DEFAULT).strip()
    run_id = str(payload.get("run_id") or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    overwrite = bool(payload.get("overwrite", False))
    gpu_batch_size = max(1, int(payload.get("gpu_batch_size") or DEFAULT_GPU_BATCH_SIZE))
    failed_prefix = str(payload.get("failed_prefix") or "datasets/pages/text_removed/_failed").strip().strip("/")
    prompt_override = str(payload.get("prompt_override") or "").strip()
    trust_manifest = bool(payload.get("trust_manifest", False))
    seed = int(payload.get("seed") or 0)

    prompt_text = prompt_override or _load_prompt()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    todo: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_key = str(page.get("page_key") or "").strip()
        output_key = str(page.get("output_key") or "").strip()
        if not page_key or not output_key:
            continue
        if not overwrite and not trust_manifest and _head_exists(bucket, output_key):
            skipped.append({"page_key": page_key, "output_key": output_key, "status": "skipped_existing"})
            continue
        todo.append(page)

    if not todo:
        return {"ok": True, "annotated": [], "skipped": skipped, "errors": [], "stats": {}}

    # Parallel S3 download
    download_start = time.perf_counter()
    images: list[Any] = []
    sources: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    output_keys: list[str] = []
    original_sizes: list[tuple[int, int]] = []

    def _download(page: dict[str, Any]) -> tuple[Any, dict[str, Any], str, str]:
        page_key = str(page["page_key"])
        output_key = str(page["output_key"])
        chapter = str(page.get("chapter") or "")
        page_id = str(page.get("page_id") or Path(page_key).stem)
        sample_id = str(page.get("sample_id") or (f"{chapter}__{page_id}" if chapter else page_id))
        image_obj, source = _download_rgb_image(bucket, page_key)
        return image_obj, source, sample_id, output_key

    with ThreadPoolExecutor(max_workers=min(16, len(todo))) as pool:
        for image_obj, source, sample_id, output_key in pool.map(_download, todo):
            original_sizes.append(image_obj.size)
            images.append(_normalize_size(image_obj, target_long=target_long_side))
            sources.append(source)
            sample_ids.append(sample_id)
            output_keys.append(output_key)
    download_sec = time.perf_counter() - download_start

    # Inference (Qwen-Image-Edit accepts batches via list-of-images on `image=`)
    annotated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), gpu_batch_size):
            batch_imgs = images[start : start + gpu_batch_size]
            batch_sources = sources[start : start + gpu_batch_size]
            batch_outputs = output_keys[start : start + gpu_batch_size]
            batch_samples = sample_ids[start : start + gpu_batch_size]
            batch_orig_sizes = original_sizes[start : start + gpu_batch_size]
            try:
                # QwenImageEditPlusPipeline supports `image=` as list[PIL].
                result = pipe(
                    image=batch_imgs,
                    prompt=[prompt_text] * len(batch_imgs),
                    negative_prompt=[NEGATIVE_PROMPT] * len(batch_imgs),
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
                    guidance_scale=1.0,
                    generator=generator,
                )
                out_images = result.images if hasattr(result, "images") else result
            except Exception as exc:  # noqa: BLE001
                for source, output_key, sample_id in zip(batch_sources, batch_outputs, batch_samples):
                    failure_key = f"{failed_prefix}/{run_id}/{sample_id}.json"
                    try:
                        _put_json_to_s3(
                            bucket, failure_key,
                            {
                                "sample_id": sample_id, "page_key": source.get("key"),
                                "output_key": output_key, "run_id": run_id,
                                "variant": variant, "error": repr(exc)[:1000],
                            },
                        )
                    except Exception:
                        pass
                    errors.append({
                        "sample_id": sample_id, "page_key": source.get("key"),
                        "output_key": output_key, "failure_key": failure_key,
                        "status": "error", "error": repr(exc)[:500],
                    })
                continue

            for source, output_key, sample_id, out_img, orig_size in zip(
                batch_sources, batch_outputs, batch_samples, out_images, batch_orig_sizes
            ):
                try:
                    # Restore original aspect/size before save
                    if out_img.size != orig_size:
                        from PIL import Image
                        out_img = out_img.resize(orig_size, Image.Resampling.LANCZOS)
                    body_size = _put_png_to_s3(bucket, output_key, out_img)
                    annotated.append({
                        "sample_id": sample_id,
                        "page_key": source.get("key"),
                        "output_key": output_key,
                        "status": "ok",
                        "bytes": body_size,
                    })
                except Exception as exc:  # noqa: BLE001
                    errors.append({
                        "sample_id": sample_id, "page_key": source.get("key"),
                        "output_key": output_key, "status": "error",
                        "error": repr(exc)[:500],
                    })

    torch.cuda.synchronize()
    inference_sec = time.perf_counter() - inference_start

    return {
        "ok": not errors,
        "annotated": annotated,
        "skipped": skipped,
        "errors": errors,
        "stats": {
            "variant": variant,
            "batch_size": len(images),
            "gpu_batch_size": gpu_batch_size,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "download_sec": round(download_sec, 3),
            "inference_sec": round(inference_sec, 3),
            "pages_per_sec": round(len(images) / inference_sec, 3) if inference_sec > 0 else None,
        },
    }


@app.cls(**VANILLA_40STEP_GPU_KW)
class QwenVanilla:
    @modal.enter()
    def load(self) -> None:
        total_gb = _cuda_total_memory_gb()
        compile_blocks = VANILLA_40STEP_COMPILE_BLOCKS and total_gb >= VANILLA_40STEP_MIN_FULL_GPU_GB
        self.pipe = _load_pipeline(
            with_lightning=False,
            fast_math=VANILLA_40STEP_FAST_MATH,
            fuse_qkv=VANILLA_40STEP_FUSE_QKV,
            compile_repeated_blocks=compile_blocks,
            compile_dynamic=VANILLA_40STEP_COMPILE_DYNAMIC,
            compile_fullgraph=False,
        )
        self.memory_mode = _configure_40step_memory(self.pipe)
        _warmup_edit_pipe(
            self.pipe,
            steps=VANILLA_STEPS,
            true_cfg_scale=VANILLA_TRUE_CFG,
            prompt=_load_prompt(),
        )

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = _edit_batch_impl(
            payload,
            pipe=self.pipe,
            variant="vanilla_40step_optimized",
            steps=VANILLA_STEPS,
            true_cfg_scale=VANILLA_TRUE_CFG,
        )
        result.setdefault("stats", {})["memory_mode"] = getattr(self, "memory_mode", "unknown")
        return result


@app.cls(**GPU_KW)
class QwenLightning:
    @modal.enter()
    def load(self) -> None:
        self.pipe = _load_pipeline(with_lightning=True)
        try:
            from PIL import Image
            import torch

            dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))
            with torch.inference_mode():
                _ = self.pipe(
                    image=[dummy],
                    prompt=["warmup"],
                    negative_prompt=[NEGATIVE_PROMPT],
                    num_inference_steps=LIGHTNING_STEPS,
                    true_cfg_scale=LIGHTNING_TRUE_CFG,
                    guidance_scale=1.0,
                )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] lightning skipped: {exc!r}", flush=True)

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _edit_batch_impl(
            payload,
            pipe=self.pipe,
            variant="lightning",
            steps=LIGHTNING_STEPS,
            true_cfg_scale=LIGHTNING_TRUE_CFG,
        )


@app.cls(**GPU_KW)
class QwenLightning8Step:
    @modal.enter()
    def load(self) -> None:
        self.pipe = _load_pipeline(
            with_lightning=True,
            lightning_weight_name=LIGHTNING_8STEP_WEIGHT,
        )
        try:
            from PIL import Image
            import torch

            dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))
            with torch.inference_mode():
                _ = self.pipe(
                    image=[dummy],
                    prompt=["warmup"],
                    negative_prompt=[NEGATIVE_PROMPT],
                    num_inference_steps=8,
                    true_cfg_scale=LIGHTNING_TRUE_CFG,
                    guidance_scale=1.0,
                )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] lightning_8step skipped: {exc!r}", flush=True)

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _edit_batch_impl(
            payload,
            pipe=self.pipe,
            variant="lightning_8step",
            steps=8,
            true_cfg_scale=LIGHTNING_TRUE_CFG,
        )


@app.cls(**GPU_KW)
class QwenFp8Single:
    @modal.enter()
    def load(self) -> None:
        self.pipe = _load_pipeline(
            with_lightning=False,
            fp8_single_file_name=QWEN_FP8_BASE_WEIGHT,
        )
        try:
            from PIL import Image
            import torch

            dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))
            with torch.inference_mode():
                _ = self.pipe(
                    image=[dummy],
                    prompt=["warmup"],
                    negative_prompt=[NEGATIVE_PROMPT],
                    num_inference_steps=2,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] fp8_single skipped: {exc!r}", flush=True)

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _edit_batch_impl(
            payload,
            pipe=self.pipe,
            variant="fp8_single",
            steps=VANILLA_STEPS,
            true_cfg_scale=VANILLA_TRUE_CFG,
        )


@app.cls(**GPU_KW)
class QwenFp8RuntimeWO:
    @modal.enter()
    def load(self) -> None:
        self.pipe = _load_pipeline(
            with_lightning=False,
            fp8_transformer_mode="weight_only",
        )
        try:
            from PIL import Image
            import torch

            dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))
            with torch.inference_mode():
                _ = self.pipe(
                    image=[dummy],
                    prompt=["warmup"],
                    negative_prompt=[NEGATIVE_PROMPT],
                    num_inference_steps=2,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] fp8_runtime_wo skipped: {exc!r}", flush=True)

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _edit_batch_impl(
            payload,
            pipe=self.pipe,
            variant="fp8_runtime_wo",
            steps=VANILLA_STEPS,
            true_cfg_scale=VANILLA_TRUE_CFG,
        )


def _load_ad_hoc_pipeline(variant: str) -> tuple[Any, int, float]:
    """Load a one-shot pipeline for smoke/benchmark helpers."""
    if variant in {"vanilla", "vanilla_40step"}:
        return _load_pipeline(with_lightning=False), VANILLA_STEPS, VANILLA_TRUE_CFG
    if variant in {"vanilla_40step_optimized", "vanilla_40step_h200"}:
        total_gb = _cuda_total_memory_gb()
        pipe = _load_pipeline(
            with_lightning=False,
            fast_math=VANILLA_40STEP_FAST_MATH,
            fuse_qkv=VANILLA_40STEP_FUSE_QKV,
            compile_repeated_blocks=VANILLA_40STEP_COMPILE_BLOCKS and total_gb >= VANILLA_40STEP_MIN_FULL_GPU_GB,
            compile_dynamic=VANILLA_40STEP_COMPILE_DYNAMIC,
            compile_fullgraph=False,
        )
        _configure_40step_memory(pipe)
        return pipe, VANILLA_STEPS, VANILLA_TRUE_CFG
    if variant == "vanilla_40step_fastmath":
        return _load_pipeline(with_lightning=False, fast_math=True), VANILLA_STEPS, VANILLA_TRUE_CFG
    if variant == "vanilla_40step_compile":
        return (
            _load_pipeline(
                with_lightning=False,
                fast_math=True,
                compile_repeated_blocks=True,
                compile_dynamic=True,
                compile_fullgraph=False,
            ),
            VANILLA_STEPS,
            VANILLA_TRUE_CFG,
        )
    if variant == "vanilla_40step_flash_compile":
        return (
            _load_pipeline(
                with_lightning=False,
                fast_math=True,
                compile_repeated_blocks=True,
                compile_dynamic=True,
                compile_fullgraph=False,
            ),
            VANILLA_STEPS,
            VANILLA_TRUE_CFG,
        )
    if variant == "vanilla_40step_qkv":
        return _load_pipeline(with_lightning=False, fuse_qkv=True), VANILLA_STEPS, VANILLA_TRUE_CFG
    if variant == "vanilla_40step_qkv_compile":
        return (
            _load_pipeline(
                with_lightning=False,
                fuse_qkv=True,
                compile_repeated_blocks=True,
                compile_dynamic=True,
                compile_fullgraph=False,
            ),
            VANILLA_STEPS,
            VANILLA_TRUE_CFG,
        )
    if variant in {"lightning", "lightning_4step"}:
        return _load_pipeline(with_lightning=True), LIGHTNING_STEPS, LIGHTNING_TRUE_CFG
    if variant == "lightning_8step":
        return (
            _load_pipeline(with_lightning=True, lightning_weight_name=LIGHTNING_8STEP_WEIGHT),
            8,
            LIGHTNING_TRUE_CFG,
        )
    if variant in {"fp8_single", "fp8_single_40step"}:
        return (
            _load_pipeline(with_lightning=False, fp8_single_file_name=QWEN_FP8_BASE_WEIGHT),
            VANILLA_STEPS,
            VANILLA_TRUE_CFG,
        )
    if variant in {"fp8_runtime_wo", "fp8_runtime_wo_40step"}:
        return (
            _load_pipeline(with_lightning=False, fp8_transformer_mode="weight_only"),
            VANILLA_STEPS,
            VANILLA_TRUE_CFG,
        )
    if variant == "fp8_single_lightning_4step":
        return (
            _load_pipeline(with_lightning=False, fp8_single_file_name=QWEN_FP8_LIGHTNING_4STEP_WEIGHT),
            LIGHTNING_STEPS,
            LIGHTNING_TRUE_CFG,
        )
    if variant == "fp8_single_lightning_8step":
        return (
            _load_pipeline(with_lightning=False, fp8_single_file_name=QWEN_FP8_LIGHTNING_8STEP_WEIGHT),
            8,
            LIGHTNING_TRUE_CFG,
        )
    raise ValueError(f"unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Benchmark — sweep batch sizes for one variant on one GPU
# ---------------------------------------------------------------------------


@app.function(**BENCHMARK_GPU_KW)
def benchmark_batch_sizes(
    variant: str = "lightning",
    bucket: str = S3_BUCKET_DEFAULT,
    chapter: str = "vinland-saga",
    source_prefix: str = "datasets/pages/filtered",
    n_images: int = 8,
    batch_sizes: list[int] | None = None,
    candidate_repeats: int = 2,
) -> dict[str, Any]:
    import time
    import torch

    if batch_sizes is None:
        batch_sizes = [1, 2, 4]

    load_start = time.perf_counter()
    pipe, steps, true_cfg = _load_ad_hoc_pipeline(variant)
    load_sec = time.perf_counter() - load_start

    s3 = _s3_client()
    prefix = f"{source_prefix.rstrip('/')}/{chapter}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = str(obj.get("Key") or "")
            if Path(k).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                keys.append(k)
                if len(keys) >= n_images:
                    break
        if len(keys) >= n_images:
            break
    if not keys:
        raise RuntimeError(f"no pages found at s3://{bucket}/{prefix}")

    images: list[Any] = []
    for k in keys:
        img, _ = _download_rgb_image(bucket, k)
        images.append(_normalize_size(img, target_long=1024))

    # Warmup
    prompt_text = _load_prompt()
    attention_backend_name = "_native_flash" if variant == "vanilla_40step_flash_compile" else ""

    with torch.inference_mode():
        _ = pipe(
            image=[images[0]],
            prompt=[prompt_text],
            negative_prompt=[NEGATIVE_PROMPT],
            num_inference_steps=steps,
            true_cfg_scale=true_cfg,
            guidance_scale=1.0,
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
        candidate = images[:batch_size]
        encoded = 0
        oom = False
        error = ""
        start = time.perf_counter()
        try:
            if not attention_backend_name:
                for _ in range(max(1, int(candidate_repeats))):
                    with torch.inference_mode():
                        _ = pipe(
                            image=candidate,
                            prompt=[prompt_text] * len(candidate),
                            negative_prompt=[NEGATIVE_PROMPT] * len(candidate),
                            num_inference_steps=steps,
                            true_cfg_scale=true_cfg,
                            guidance_scale=1.0,
                        )
                    encoded += len(candidate)
                    torch.cuda.synchronize()
            else:
                from diffusers import attention_backend

                with attention_backend(attention_backend_name):
                    for _ in range(max(1, int(candidate_repeats))):
                        with torch.inference_mode():
                            _ = pipe(
                                image=candidate,
                                prompt=[prompt_text] * len(candidate),
                                negative_prompt=[NEGATIVE_PROMPT] * len(candidate),
                                num_inference_steps=steps,
                                true_cfg_scale=true_cfg,
                                guidance_scale=1.0,
                            )
                        encoded += len(candidate)
                        torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            oom = True
            error = str(exc)[:300]
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                oom = True
                error = str(exc)[:300]
                torch.cuda.empty_cache()
            else:
                raise
        wall = time.perf_counter() - start
        peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        item = {
            "batch_size": batch_size,
            "oom": oom,
            "error": error,
            "images_completed": encoded,
            "wall_sec": round(wall, 3),
            "pages_per_sec": round(encoded / wall, 4) if wall > 0 else 0.0,
            "sec_per_image": round(wall / encoded, 4) if encoded > 0 else None,
            "peak_allocated_gb": round(peak_gb, 2),
        }
        print(f"[benchmark/{variant}] {json.dumps(item, sort_keys=True)}", flush=True)
        results.append(item)

    viable = [r for r in results if not r.get("oom") and r.get("images_completed", 0) > 0]
    best = max(viable, key=lambda r: r["pages_per_sec"]) if viable else None
    return {
        "ok": best is not None,
        "variant": variant,
        "steps": steps,
        "true_cfg_scale": true_cfg,
        "image_count": len(images),
        "load_sec": round(load_sec, 2),
        "best": best,
        "results": results,
    }


@app.local_entrypoint()
def run_benchmark(
    variant: str = "lightning",
    chapter: str = "vinland-saga",
    n_images: int = 8,
):
    result = benchmark_batch_sizes.remote(variant=variant, chapter=chapter, n_images=n_images)
    print(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Smoke + compare entrypoints
# ---------------------------------------------------------------------------


@app.function(
    region=MODAL_REGION,
    gpu=DEFAULT_GPU_TYPE,
    timeout=2400,
    startup_timeout=1800,
    cpu=8.0,
    memory=65536,
    secrets=[aws_secret],
    volumes={HF_HOME: hf_volume},
)
def edit_pages_remote(
    keys: list[str],
    bucket: str = S3_BUCKET_DEFAULT,
    variant: str = "lightning",
    return_bytes: bool = True,
) -> list[dict[str, Any]]:
    """One-shot variant test: run the chosen variant on `keys`, return PNG bytes
    (for local download / artifact comparison) and timing per image.
    """
    import base64
    import torch

    pipe, steps, true_cfg = _load_ad_hoc_pipeline(variant)
    prompt_text = _load_prompt()

    # Warmup
    from PIL import Image as PILImage
    dummy = PILImage.new("RGB", (1024, 1024), (128, 128, 128))
    with torch.inference_mode():
        _ = pipe(
            image=[dummy], prompt=["warmup"], negative_prompt=[NEGATIVE_PROMPT],
            num_inference_steps=steps, true_cfg_scale=true_cfg, guidance_scale=1.0,
        )
    torch.cuda.synchronize()

    results = []
    for k in keys:
        img, _ = _download_rgb_image(bucket, k)
        orig_size = img.size
        norm = _normalize_size(img, target_long=1024)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pipe(
                image=[norm],
                prompt=[prompt_text],
                negative_prompt=[NEGATIVE_PROMPT],
                num_inference_steps=steps,
                true_cfg_scale=true_cfg,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out_img = out.images[0].resize(orig_size, PILImage.Resampling.LANCZOS)
        entry = {"key": k, "sec": round(dt, 3), "size": list(orig_size)}
        if return_bytes:
            buf = io.BytesIO()
            out_img.save(buf, format="PNG", optimize=True)
            entry["png_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Tweak experiment — try multiple Lightning/vanilla configs on one page
# ---------------------------------------------------------------------------


TWEAK_PROMPT_STRICT = (
    "ONLY erase the manga text and SFX lettering. Do not modify any non-text content. "
    "Preserve every character, line, shadow, screentone, panel border, and detail exactly as it is. "
    "Do not redraw, simplify, or stylize anything except where text was."
)
TWEAK_PROMPT_MINIMAL = "Erase only the text and sfx. Leave all art unchanged."
TWEAK_NEG_STRONG = (
    "modify art, simplify lines, smooth, redraw, low detail, distort, "
    "change composition, remove characters, erase art, blur, soften, stylize, "
    "delete details, lose texture"
)


@app.function(
    region=MODAL_REGION,
    gpu=DEFAULT_GPU_TYPE,
    timeout=2400,
    startup_timeout=1800,
    cpu=8.0,
    memory=65536,
    secrets=[aws_secret],
    volumes={HF_HOME: hf_volume},
    max_containers=8,
)
def edit_with_config(payload: dict[str, Any]) -> dict[str, Any]:
    """One container loads a specific pipeline config and edits N pages.

    payload:
        variant: 'lightning_4step' | 'lightning_4step_cfg15' | 'lightning_4step_strongneg'
               | 'lightning_4step_strict' | 'lightning_8step' | 'vanilla_20step' | 'vanilla_40step'
               | 'vanilla_40step_fastmath' | 'vanilla_40step_qkv'
               | 'vanilla_40step_compile' | 'vanilla_40step_qkv_compile'
               | 'vanilla_40step_optimized' | 'vanilla_40step_batched_cfg'
               | 'fp8_single_40step' | 'fp8_single_40step_compiled'
               | 'fp8_runtime_wo_40step' | 'fp8_runtime_wo_40step_compiled'
        keys: list of S3 keys (in `bucket`)
        bucket: S3 bucket
    """
    import base64
    import time
    import torch

    variant = str(payload["variant"])
    keys = list(payload["keys"])
    bucket = str(payload.get("bucket") or S3_BUCKET_DEFAULT)

    # Pick LoRA + step count + cfg per variant.
    # When true_cfg_scale > 1 the pipeline runs the model twice per step (CFG),
    # which doubles activation memory and OOMs on H100-80GB at 1024px without
    # mitigation. Use attention_slicing + (optionally) cpu_offload to fit.
    needs_cpu_offload = False
    needs_attention_slicing = False
    if variant == "vanilla_40step":
        pipe = _load_pipeline(with_lightning=False)
        steps, true_cfg = 40, 4.0
    elif variant == "vanilla_40step_offload":
        pipe = _load_pipeline(with_lightning=False)
        steps, true_cfg = 40, 4.0
        needs_cpu_offload = True
    elif variant == "vanilla_40step_fastmath":
        pipe = _load_pipeline(with_lightning=False, fast_math=True)
        steps, true_cfg = 40, 4.0
    elif variant in {"vanilla_40step_optimized", "vanilla_40step_h200", "vanilla_40step_batched_cfg"}:
        total_gb = _cuda_total_memory_gb()
        pipe = _load_pipeline(
            with_lightning=False,
            fast_math=VANILLA_40STEP_FAST_MATH,
            fuse_qkv=VANILLA_40STEP_FUSE_QKV,
            compile_repeated_blocks=VANILLA_40STEP_COMPILE_BLOCKS and total_gb >= VANILLA_40STEP_MIN_FULL_GPU_GB,
            compile_dynamic=VANILLA_40STEP_COMPILE_DYNAMIC,
            compile_fullgraph=False,
        )
        if variant == "vanilla_40step_batched_cfg":
            _enable_batched_true_cfg(pipe)
        steps, true_cfg = 40, 4.0
        _configure_40step_memory(pipe)
    elif variant == "vanilla_40step_qkv":
        pipe = _load_pipeline(with_lightning=False, fuse_qkv=True)
        steps, true_cfg = 40, 4.0
    elif variant == "vanilla_40step_compile":
        pipe = _load_pipeline(
            with_lightning=False,
            compile_repeated_blocks=True,
            compile_dynamic=True,
        )
        steps, true_cfg = 40, 4.0
    elif variant == "vanilla_40step_qkv_compile":
        pipe = _load_pipeline(
            with_lightning=False,
            fuse_qkv=True,
            compile_repeated_blocks=True,
            compile_dynamic=True,
        )
        steps, true_cfg = 40, 4.0
    elif variant == "vanilla_40step_compiled":
        # Same 40-step pipeline + torch.compile on the transformer.
        # Expected ~1.3-1.5x speedup, identical quality.
        pipe = _load_pipeline(with_lightning=False, compile_transformer=True)
        steps, true_cfg = 40, 4.0
        needs_attention_slicing = True
    elif variant == "vanilla_40step_fp8":
        # Historical failed experiment: dynamic activation fp8 breaks the
        # useful fused attention path and was much slower than vanilla.
        pipe = _load_pipeline(with_lightning=False, fp8_transformer_mode="dynamic")
        steps, true_cfg = 40, 4.0
        needs_attention_slicing = True
    elif variant == "vanilla_40step_compiled_fp8":
        # Historical failed experiment: compile + dynamic fp8 was slower than
        # either vanilla or compiled vanilla in the one-page tweak test.
        pipe = _load_pipeline(
            with_lightning=False, compile_transformer=True, fp8_transformer_mode="dynamic",
        )
        steps, true_cfg = 40, 4.0
        needs_attention_slicing = True
    elif variant == "fp8_runtime_wo_40step":
        # Experiment-only: measured 59.179 s on the Vinland smoke page, slower
        # than the 39.038 s vanilla result on that same page.
        pipe = _load_pipeline(with_lightning=False, fp8_transformer_mode="weight_only")
        steps, true_cfg = 40, 4.0
    elif variant == "fp8_runtime_wo_40step_compiled":
        pipe = _load_pipeline(
            with_lightning=False,
            fp8_transformer_mode="weight_only",
            compile_transformer=True,
        )
        steps, true_cfg = 40, 4.0
    elif variant == "fp8_single_40step":
        pipe = _load_pipeline(with_lightning=False, fp8_single_file_name=QWEN_FP8_BASE_WEIGHT)
        steps, true_cfg = 40, 4.0
    elif variant == "fp8_single_40step_compiled":
        pipe = _load_pipeline(
            with_lightning=False,
            fp8_single_file_name=QWEN_FP8_BASE_WEIGHT,
            compile_transformer=True,
        )
        steps, true_cfg = 40, 4.0
    elif variant == "fp8_single_lightning_4step":
        pipe = _load_pipeline(
            with_lightning=False,
            fp8_single_file_name=QWEN_FP8_LIGHTNING_4STEP_WEIGHT,
        )
        steps, true_cfg = 4, 1.0
    elif variant == "fp8_single_lightning_8step":
        pipe = _load_pipeline(
            with_lightning=False,
            fp8_single_file_name=QWEN_FP8_LIGHTNING_8STEP_WEIGHT,
        )
        steps, true_cfg = 8, 1.0
    elif variant == "vanilla_20step":
        pipe = _load_pipeline(with_lightning=False)
        steps, true_cfg = 20, 4.0
        needs_cpu_offload = True
    elif variant == "lightning_8step":
        pipe = _load_pipeline(with_lightning=True, lightning_weight_name=LIGHTNING_8STEP_WEIGHT)
        steps, true_cfg = 8, 1.0
    elif variant.startswith("lightning_4step"):
        pipe = _load_pipeline(with_lightning=True, lightning_weight_name=LIGHTNING_4STEP_WEIGHT)
        steps = 4
        true_cfg = 1.5 if "cfg15" in variant else 1.0
        if true_cfg > 1.0:
            needs_cpu_offload = True
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    if needs_cpu_offload:
        # Last-resort: keeps weights in CPU RAM, moves sub-module to GPU per
        # forward pass. ~2-3x slower than on-GPU but always fits inside 80GB.
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"[mem] enable_model_cpu_offload failed: {e!r}", flush=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    elif needs_attention_slicing:
        # Slices attention QKV into chunks; ~5% slower than no slicing but
        # cuts peak activation memory enough to keep the full pipe on GPU
        # while CFG runs the model twice per step.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # Choose prompt + negative based on variant suffix
    prompt = _load_prompt()
    negative = NEGATIVE_PROMPT
    if variant.endswith("_strict"):
        prompt = TWEAK_PROMPT_STRICT
    if variant.endswith("_minimal"):
        prompt = TWEAK_PROMPT_MINIMAL
    if variant.endswith("_strongneg"):
        negative = TWEAK_NEG_STRONG
    # Allow per-payload override (last-write-wins) for ad-hoc prompt tests
    if payload.get("prompt_override"):
        prompt = str(payload["prompt_override"])
    if payload.get("negative_override"):
        negative = str(payload["negative_override"])

    # Warmup
    from PIL import Image as PILImage
    dummy = PILImage.new("RGB", (1024, 1024), (128, 128, 128))
    with torch.inference_mode():
        _ = pipe(
            image=[dummy], prompt=["warmup"], negative_prompt=[negative],
            num_inference_steps=steps, true_cfg_scale=true_cfg, guidance_scale=1.0,
        )
    torch.cuda.synchronize()

    results = []
    for k in keys:
        img, _ = _download_rgb_image(bucket, k)
        orig_size = img.size
        norm = _normalize_size(img, target_long=1024)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pipe(
                image=[norm],
                prompt=[prompt],
                negative_prompt=[negative],
                num_inference_steps=steps,
                true_cfg_scale=true_cfg,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out_img = out.images[0].resize(orig_size, PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG", optimize=True)
        results.append({
            "key": k,
            "sec": round(dt, 3),
            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        })

    return {
        "variant": variant,
        "steps": steps,
        "true_cfg_scale": true_cfg,
        "prompt": prompt,
        "negative_prompt": negative,
        "results": results,
    }


DEFAULT_TWEAK_VARIANTS = [
    "lightning_4step",
    "lightning_4step_cfg15",
    "lightning_4step_strongneg",
    "lightning_4step_strict",
    "lightning_8step",
    "vanilla_20step",
    "vanilla_40step",
]


@app.local_entrypoint()
def compare_tweaks(
    page_s3_uri: str = "",
    extra_pages: str = "",
    variants: str = "",  # comma-separated; empty = all
):
    """Run a problem page through Lightning/vanilla configs in parallel and
    save outputs to artifacts/. Default page is the vinland-saga page the user
    flagged as over-edited. Pass `--variants` to run a subset.
    """
    import base64

    if not page_s3_uri:
        page_s3_uri = f"s3://{S3_BUCKET_DEFAULT}/datasets/pages/filtered/vinland-saga/vinland-saga-chapter-0__006_6.jpg"

    keys: list[tuple[str, str]] = []
    for uri in [page_s3_uri] + [u.strip() for u in extra_pages.split(",") if u.strip()]:
        b, k = _parse_s3_uri(uri)
        keys.append((b, k))

    out_root = Path(__file__).resolve().parents[2] / "artifacts" / "remove_text_modal_tweaks"
    out_root.mkdir(parents=True, exist_ok=True)
    originals_dir = out_root / "original"
    originals_dir.mkdir(exist_ok=True, parents=True)

    # Download originals locally for the side-by-side
    import boto3
    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "default"))
    s3 = session.client("s3", region_name=AWS_REGION_DEFAULT)
    for b, k in keys:
        local = originals_dir / Path(k).name
        if not local.exists():
            s3.download_file(b, k, str(local))

    requested = [v.strip() for v in variants.split(",") if v.strip()] or DEFAULT_TWEAK_VARIANTS

    bucket = keys[0][0]
    payloads = [
        {"variant": v, "keys": [k for _, k in keys], "bucket": bucket}
        for v in requested
    ]

    overall: dict[str, Any] = {"selected_keys": [k for _, k in keys], "variants": {}}
    # Fire all variants in parallel — each gets its own H100 container.
    # `return_exceptions=True` so one OOM doesn't kill the rest.
    for result in edit_with_config.map(payloads, order_outputs=False, return_exceptions=True):
        if isinstance(result, BaseException):
            print(json.dumps({"event": "variant_failed", "error": repr(result)[:500]}), flush=True)
            continue
        variant = result["variant"]
        vdir = out_root / variant
        vdir.mkdir(exist_ok=True, parents=True)
        per_image = []
        for r in result["results"]:
            png = base64.b64decode(r["png_b64"])
            local_path = vdir / (Path(r["key"]).stem + ".png")
            local_path.write_bytes(png)
            per_image.append({"key": r["key"], "sec": r["sec"], "local_path": str(local_path)})
        avg = sum(x["sec"] for x in per_image) / len(per_image) if per_image else None
        overall["variants"][variant] = {
            "steps": result["steps"],
            "true_cfg_scale": result["true_cfg_scale"],
            "prompt": result["prompt"],
            "negative_prompt": result["negative_prompt"],
            "avg_sec_per_image": round(avg, 3) if avg else None,
            "per_image": per_image,
        }
        print(json.dumps({
            "variant": variant, "avg_sec": round(avg, 3) if avg else None,
            "steps": result["steps"], "true_cfg": result["true_cfg_scale"],
        }), flush=True)

    (out_root / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(f"\nArtifacts saved to: {out_root}")


@app.local_entrypoint()
def rerun_compare_variant(variant: str = "lightning_8step", out_subdir: str = ""):
    """Re-run the original 6 compare_variants pages through any variant.

    Example::

        modal run modal_qwen.py::rerun_compare_variant --variant lightning_8step
        modal run modal_qwen.py::rerun_compare_variant --variant lightning_4step_minimal

    Output dir defaults to ``artifacts/remove_text_modal_compare/<variant>/``.
    """
    import base64

    compare_root = Path(__file__).resolve().parents[2] / "artifacts" / "remove_text_modal_compare"
    summary_path = compare_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path} — run compare_variants first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = list(summary["selected_keys"])
    if not keys:
        raise RuntimeError("no selected_keys in compare summary")

    payload = {"variant": variant, "keys": keys, "bucket": S3_BUCKET_DEFAULT}
    result = edit_with_config.remote(payload)

    out_dir = compare_root / (out_subdir or variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image = []
    for r in result["results"]:
        png = base64.b64decode(r["png_b64"])
        local_path = out_dir / (Path(r["key"]).stem + ".png")
        local_path.write_bytes(png)
        per_image.append({"key": r["key"], "sec": r["sec"], "local_path": str(local_path)})

    avg = sum(x["sec"] for x in per_image) / len(per_image)
    print(json.dumps({
        "variant": variant,
        "steps": result["steps"],
        "true_cfg_scale": result["true_cfg_scale"],
        "prompt": result["prompt"],
        "negative_prompt": result["negative_prompt"],
        "avg_sec_per_image": round(avg, 3),
        "per_image": per_image,
    }, indent=2))
    print(f"\nArtifacts: {out_dir}")


# ---------------------------------------------------------------------------
# Existing smoke_test / annotate_manifest_local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_test(variant: str = "lightning", page_s3_uri: str = ""):
    if not page_s3_uri:
        page_s3_uri = f"s3://{S3_BUCKET_DEFAULT}/datasets/pages/filtered/vinland-saga/vinland-saga-chapter-100__010.jpg"
    bucket, key = _parse_s3_uri(page_s3_uri)
    out_dir = Path(__file__).resolve().parents[2] / "artifacts" / "remove_text_modal_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = edit_pages_remote.remote([key], bucket=bucket, variant=variant, return_bytes=True)
    for r in results:
        if "png_b64" in r:
            import base64
            png_bytes = base64.b64decode(r["png_b64"])
            out_path = out_dir / f"{variant}__{Path(r['key']).stem}.png"
            out_path.write_bytes(png_bytes)
            r["local_path"] = str(out_path)
            r.pop("png_b64", None)
    print(json.dumps({"variant": variant, "results": results}, indent=2, default=str))


@app.local_entrypoint()
def compare_variants(
    pages: int = 6,
    bucket: str = S3_BUCKET_DEFAULT,
    source_prefix: str = "datasets/pages/filtered",
    chapters: str = "jujutsu-kaisen,monster,my-hero-academia,vagabond,vinland-saga,the-fragrant-flower-blooms-with-dignity",
):
    """Pick N test pages spread across given chapters, run BOTH variants,
    save the PNG outputs side-by-side under artifacts/ for visual A/B.
    """
    import base64
    import boto3

    out_root = Path(__file__).resolve().parents[2] / "artifacts" / "remove_text_modal_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    # Pick `pages` total, distributed across chapters
    chapter_list = [c.strip() for c in chapters.split(",") if c.strip()]
    per_chapter = max(1, pages // len(chapter_list))
    selected: list[str] = []
    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "default"))
    s3 = session.client("s3", region_name=AWS_REGION_DEFAULT)
    for ch in chapter_list:
        prefix = f"{source_prefix.rstrip('/')}/{ch}/"
        paginator = s3.get_paginator("list_objects_v2")
        taken = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                k = str(obj.get("Key") or "")
                if Path(k).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    selected.append(k)
                    taken += 1
                    if taken >= per_chapter:
                        break
            if taken >= per_chapter:
                break
        if len(selected) >= pages:
            break
    selected = selected[:pages]

    # Also copy originals for side-by-side comparison
    originals_dir = out_root / "original"
    originals_dir.mkdir(exist_ok=True, parents=True)
    for k in selected:
        out_path = originals_dir / Path(k).name
        if not out_path.exists():
            s3.download_file(bucket, k, str(out_path))

    overall: dict[str, Any] = {"selected_keys": selected, "variants": {}}
    for variant in ("vanilla", "lightning"):
        variant_dir = out_root / variant
        variant_dir.mkdir(exist_ok=True, parents=True)
        results = edit_pages_remote.remote(selected, bucket=bucket, variant=variant, return_bytes=True)
        per_image = []
        for r in results:
            local_name = Path(r["key"]).stem + ".png"
            (variant_dir / local_name).write_bytes(base64.b64decode(r["png_b64"]))
            per_image.append({"key": r["key"], "sec": r["sec"], "local_path": str(variant_dir / local_name)})
        avg_sec = sum(x["sec"] for x in per_image) / len(per_image) if per_image else None
        overall["variants"][variant] = {
            "avg_sec_per_image": round(avg_sec, 3) if avg_sec else None,
            "per_image": per_image,
        }

    # Write a summary JSON
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2, default=str))
    print(f"\nArtifacts saved to: {out_root}")


# ---------------------------------------------------------------------------
# Bulk runner (mirrors manga_annotate annotate_manifest_local)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def annotate_manifest_local(
    manifest_path: str,
    variant: str = "lightning",
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    overwrite: bool = False,
    gpu_batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    pages_per_shard: int = 8,
    trust_manifest: bool = False,
):
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        print("manifest is empty; nothing to do")
        return

    effective_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shard_size = max(1, int(pages_per_shard))
    shards = [
        {
            "bucket": bucket,
            "run_id": effective_run_id,
            "overwrite": overwrite,
            "trust_manifest": trust_manifest,
            "gpu_batch_size": gpu_batch_size,
            "pages": rows[start : start + shard_size],
        }
        for start in range(0, len(rows), shard_size)
    ]

    print(json.dumps({
        "event": "start", "run_id": effective_run_id, "variant": variant,
        "pages": len(rows), "shards": len(shards), "shard_size": shard_size,
        "gpu_batch_size": gpu_batch_size, "max_containers": DEFAULT_MAX_CONTAINERS,
    }), flush=True)

    pipeline_map = {
        "vanilla": QwenVanilla,
        "vanilla_40step": QwenVanilla,
        "vanilla_40step_optimized": QwenVanilla,
        "vanilla_40step_batched_cfg": QwenVanilla,
        "lightning": QwenLightning,
        "lightning_8step": QwenLightning8Step,
    }
    if variant not in pipeline_map:
        raise ValueError(f"unknown production variant {variant!r}; expected one of {sorted(pipeline_map)}")
    pipeline_cls = pipeline_map[variant]
    cls = pipeline_cls()
    annotated_total = 0
    skipped_total = 0
    error_total = 0
    wall_start = time.perf_counter()
    for idx, result in enumerate(cls.edit_batch.map(shards, order_outputs=False)):
        annotated_total += len(result.get("annotated", []))
        skipped_total += len(result.get("skipped", []))
        error_total += len(result.get("errors", []))
        elapsed = time.perf_counter() - wall_start
        rate = annotated_total / elapsed if elapsed > 0 else 0.0
        print(json.dumps({
            "event": "shard_done", "completed_shards": idx + 1,
            "annotated_total": annotated_total, "skipped_total": skipped_total,
            "error_total": error_total, "elapsed_sec": round(elapsed, 1),
            "pages_per_sec_cluster": round(rate, 2), "stats": result.get("stats"),
        }), flush=True)
    wall_sec = time.perf_counter() - wall_start
    print(json.dumps({
        "event": "done", "run_id": effective_run_id, "variant": variant,
        "annotated_total": annotated_total, "skipped_total": skipped_total,
        "error_total": error_total, "wall_sec": round(wall_sec, 1),
        "pages_per_sec_cluster": round(annotated_total / wall_sec, 2) if wall_sec > 0 else None,
    }), flush=True)
