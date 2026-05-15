import base64
import io
import os
import random
import time
from pathlib import Path

import modal

APP_NAME = "drawtoon-flux-compare"
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
CHECKPOINT_URI = ""
SERVER_VERSION = "controls-v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate",
        "boto3",
        "diffusers",
        "fastapi[standard]",
        "safetensors",
        "torch",
        "transformers",
    )
)

app = modal.App(APP_NAME)
ckpt_cache = modal.Volume.from_name("drawtoon-inference-checkpoints", create_if_missing=True)
aws_secret = modal.Secret.from_name("lineart2-aws-s3")
hf_secret = modal.Secret.from_name("drawtoon-flux2-inference-hf-token")
deploy_values = {
    "BASE_MODEL_ID": os.environ.get("BASE_MODEL_ID", BASE_MODEL_ID),
    "FINETUNED_CHECKPOINT_URI": os.environ.get("FINETUNED_CHECKPOINT_URI", CHECKPOINT_URI),
}
deploy_env = modal.Secret.from_dict(deploy_values)

_base_pipe = None
_finetuned_pipe = None
_load_stats = {}


def _download_s3(uri: str) -> Path:
    import boto3

    bucket, key = uri[5:].split("/", 1)
    path = Path("/checkpoints") / Path(key).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(path))
        ckpt_cache.commit()
    return path


def _resolve_checkpoint(uri: str) -> Path:
    if not uri:
        raise RuntimeError("Set FINETUNED_CHECKPOINT_URI to a .safetensors file or Diffusers directory.")
    if uri.startswith("s3://"):
        return _download_s3(uri)
    return Path(uri)


def _convert_ai_toolkit_transformer_state(state: dict, target_state: dict) -> dict:
    converted = {}
    for key, value in state.items():
        parts = key.split(".")
        if parts[0] == "double_blocks" and len(parts) >= 4:
            block = parts[1]
            stream = parts[2]
            leaf = ".".join(parts[3:])
            base = f"transformer_blocks.{block}"
            if stream == "img_attn" and leaf == "qkv.weight":
                q, k, v = value.chunk(3, dim=0)
                converted[f"{base}.attn.to_q.weight"] = q
                converted[f"{base}.attn.to_k.weight"] = k
                converted[f"{base}.attn.to_v.weight"] = v
            elif stream == "txt_attn" and leaf == "qkv.weight":
                q, k, v = value.chunk(3, dim=0)
                converted[f"{base}.attn.add_q_proj.weight"] = q
                converted[f"{base}.attn.add_k_proj.weight"] = k
                converted[f"{base}.attn.add_v_proj.weight"] = v
            elif stream == "img_attn" and leaf == "proj.weight":
                converted[f"{base}.attn.to_out.0.weight"] = value
            elif stream == "txt_attn" and leaf == "proj.weight":
                converted[f"{base}.attn.to_add_out.weight"] = value
            elif stream == "img_attn" and leaf == "norm.query_norm.scale":
                converted[f"{base}.attn.norm_q.weight"] = value
            elif stream == "img_attn" and leaf == "norm.key_norm.scale":
                converted[f"{base}.attn.norm_k.weight"] = value
            elif stream == "txt_attn" and leaf == "norm.query_norm.scale":
                converted[f"{base}.attn.norm_added_q.weight"] = value
            elif stream == "txt_attn" and leaf == "norm.key_norm.scale":
                converted[f"{base}.attn.norm_added_k.weight"] = value
            elif stream == "img_mlp" and leaf == "0.weight":
                converted[f"{base}.ff.linear_in.weight"] = value
            elif stream == "img_mlp" and leaf == "2.weight":
                converted[f"{base}.ff.linear_out.weight"] = value
            elif stream == "txt_mlp" and leaf == "0.weight":
                converted[f"{base}.ff_context.linear_in.weight"] = value
            elif stream == "txt_mlp" and leaf == "2.weight":
                converted[f"{base}.ff_context.linear_out.weight"] = value
        elif parts[0] == "single_blocks" and len(parts) >= 4:
            block = parts[1]
            leaf = ".".join(parts[2:])
            base = f"single_transformer_blocks.{block}.attn"
            if leaf == "linear1.weight":
                converted[f"{base}.to_qkv_mlp_proj.weight"] = value
            elif leaf == "linear2.weight":
                converted[f"{base}.to_out.weight"] = value
            elif leaf == "norm.query_norm.scale":
                converted[f"{base}.norm_q.weight"] = value
            elif leaf == "norm.key_norm.scale":
                converted[f"{base}.norm_k.weight"] = value

    return {
        key: value
        for key, value in converted.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }


def _load_pipeline(finetuned: bool):
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        os.environ.get("BASE_MODEL_ID", BASE_MODEL_ID),
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    ).to("cuda")

    if finetuned:
        path = _resolve_checkpoint(os.environ.get("FINETUNED_CHECKPOINT_URI", CHECKPOINT_URI))
        if path.is_dir():
            pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda")
        else:
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
            target = getattr(pipe, "transformer", None) or getattr(pipe, "unet")
            target_state = target.state_dict()
            candidates = [state, _convert_ai_toolkit_transformer_state(state, target_state)]
            for prefix in ("transformer.", "model.", "module."):
                candidates.append({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
            best = max(
                candidates,
                key=lambda sd: sum(
                    1 for k, v in sd.items() if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
                ),
            )
            matched = sum(
                1 for k, v in best.items() if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
            )
            if matched < 150:
                raise RuntimeError(f"Checkpoint keys do not match {type(target).__name__}; matched {matched}.")
            missing, unexpected = target.load_state_dict(best, strict=False)
            _load_stats["finetuned"] = {
                "matched_tensors": matched,
                "checkpoint_tensors": len(state),
                "missing_tensors": len(missing),
                "unexpected_tensors": len(unexpected),
            }
    return pipe


def _pipe(finetuned: bool):
    global _base_pipe, _finetuned_pipe
    if finetuned:
        if _finetuned_pipe is None:
            _finetuned_pipe = _load_pipeline(True)
        return _finetuned_pipe
    if _base_pipe is None:
        _base_pipe = _load_pipeline(False)
    return _base_pipe


def _decode_control_images(payload: dict):
    from PIL import Image

    raw_items = payload.get("control_images") or []
    if not isinstance(raw_items, list):
        return []

    images = []
    for item in raw_items[:7]:
        raw_value = item.get("data_url") if isinstance(item, dict) else item
        if not isinstance(raw_value, str):
            continue
        data_url = raw_value.strip()
        if not data_url:
            continue
        if "," in data_url and data_url.startswith("data:image/"):
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url, validate=True)
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def _generate(payload: dict, *, finetuned: bool) -> dict:
    import torch

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")

    count = min(max(int(payload.get("count", 1)), 1), 2)
    seed = int(payload.get("seed") or random.randint(0, 2**31 - 1))
    width = int(payload.get("width", 768))
    height = int(payload.get("height", 768))
    steps = int(payload.get("steps", 30))
    guidance = float(payload.get("guidance_scale", 4.0))
    control_images = _decode_control_images(payload)

    pipe = _pipe(finetuned)
    started = time.time()
    images = []
    for i in range(count):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if control_images:
            kwargs["image"] = control_images
        image = pipe(
            **kwargs,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        images.append({"seed": seed + i, "data_url": f"data:image/png;base64,{data}"})

    return {
        "model": "finetuned" if finetuned else "base",
        "server_version": SERVER_VERSION,
        "seconds": round(time.time() - started, 2),
        "images": images,
        "load": _load_stats.get("finetuned") if finetuned else None,
        "controls_used": len(control_images),
    }


function_kwargs = dict(
    image=image,
    gpu="H100",
    timeout=20 * 60,
    scaledown_window=20 * 60,
    volumes={"/checkpoints": ckpt_cache},
    secrets=[aws_secret, hf_secret, deploy_env],
)


@app.function(**function_kwargs)
@modal.fastapi_endpoint(method="POST")
def base(item: dict):
    try:
        return _generate(item, finetuned=False)
    except Exception as exc:
        return {
            "model": "base",
            "server_version": SERVER_VERSION,
            "images": [],
            "controls_used": len(_decode_control_images(item)),
            "error": str(exc)[:1000],
        }


@app.function(**function_kwargs)
@modal.fastapi_endpoint(method="POST")
def finetuned(item: dict):
    try:
        return _generate(item, finetuned=True)
    except Exception as exc:
        return {
            "model": "finetuned",
            "server_version": SERVER_VERSION,
            "images": [],
            "controls_used": len(_decode_control_images(item)),
            "error": str(exc)[:1000],
        }
