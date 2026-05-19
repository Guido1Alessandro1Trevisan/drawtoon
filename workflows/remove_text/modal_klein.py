"""Modal app: local FLUX.2 Klein 9B text removal.

Production path:

- ``FluxKleinLocal`` loads ``black-forest-labs/FLUX.2-klein-9B`` with Diffusers
  once per GPU container and edits pages with exactly 4 inference steps.

S3 is the durable source of truth for inputs, outputs, and failure logs.
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


KLEIN_REPO = os.environ.get("FLUX2_KLEIN_REPO", "black-forest-labs/FLUX.2-klein-9B")
HF_HOME = "/root/.cache/huggingface"

S3_BUCKET_DEFAULT = os.environ.get("DRAWTOON_S3_BUCKET", "drawtoon")
AWS_REGION_DEFAULT = os.environ.get("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.environ.get("DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3")
HF_SECRET_NAME = os.environ.get("DRAWTOON_HF_SECRET_NAME", "lineart2-hf-token")
MODAL_REGION = os.environ.get("DRAWTOON_MODAL_REGION", "us-east-1")

KLEIN_GPU = os.environ.get("KLEIN_GPU", "H200")
KLEIN_MAX_CONTAINERS = int(os.environ.get("KLEIN_MAX_CONTAINERS", "40"))
KLEIN_STEPS = int(os.environ.get("KLEIN_STEPS", "4"))
KLEIN_CPU = float(os.environ.get("KLEIN_CPU", "8"))
KLEIN_CPU_MEMORY_MB = int(os.environ.get("KLEIN_CPU_MEMORY_MB", "65536"))
KLEIN_SCALEDOWN_WINDOW = int(os.environ.get("KLEIN_SCALEDOWN_WINDOW", "900"))
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

KLEIN_NO_NEW_BUBBLES_PROMPT = (
    "\n\nCritical constraint: do not add, draw, generate, or replace anything with speech bubbles, "
    "narration bubbles, white ovals, white circles, white boxes, or blank bubble shapes. "
    "If text or SFX is outside an existing bubble, erase only the lettering and leave the original "
    "background/art in that same area unchanged. Only preserve bubbles that already existed in the input image; "
    "never create new bubbles."
)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "accelerate",
        "boto3==1.35.99",
        "diffusers",
        "hf_xet==1.4.3",
        "pillow==11.1.0",
        "safetensors",
        "sentencepiece==0.2.0",
        "torch",
        "torchvision",
        "transformers",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": HF_HOME,
        "HUGGINGFACE_HUB_CACHE": f"{HF_HOME}/hub",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(
        str(Path(__file__).resolve().parent / "prompts"),
        remote_path="/root/prompts",
        copy=False,
    )
)

app = modal.App("drawtoon-remove-text-modal", image=image)
hf_volume = modal.Volume.from_name("flux2-klein-9b-hf-cache", create_if_missing=True)
aws_secret = modal.Secret.from_name(AWS_SECRET_NAME)
hf_secrets = [aws_secret]
if os.environ.get("HF_TOKEN"):
    hf_secrets.append(modal.Secret.from_local_environ(["HF_TOKEN"]))
else:
    hf_secrets.append(modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"]))

KLEIN_KW = dict(
    region=MODAL_REGION,
    gpu=KLEIN_GPU,
    timeout=3600,
    startup_timeout=1800,
    cpu=KLEIN_CPU,
    memory=KLEIN_CPU_MEMORY_MB,
    secrets=hf_secrets,
    volumes={HF_HOME: hf_volume},
    max_containers=KLEIN_MAX_CONTAINERS,
    scaledown_window=KLEIN_SCALEDOWN_WINDOW,
    # Snapshot captures the bf16 9B weights on CPU after the first cold start;
    # subsequent containers restore from the snapshot in ~10s instead of ~70s.
    enable_memory_snapshot=True,
)


def _load_prompt() -> str:
    path = Path("/root/prompts/master_prompt.md")
    if not path.exists():
        path = Path(__file__).resolve().parent / "prompts" / "master_prompt.md"
    return path.read_text(encoding="utf-8").strip()


def _klein_prompt(prompt_override: str = "") -> str:
    base = prompt_override.strip() or _load_prompt()
    if "never create new bubbles" in base.lower():
        return base
    return base + KLEIN_NO_NEW_BUBBLES_PROMPT


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


def _enable_cuda_fast_math(torch_mod: Any) -> None:
    try:
        torch_mod.backends.cuda.matmul.allow_tf32 = True
        torch_mod.backends.cudnn.allow_tf32 = True
        torch_mod.set_float32_matmul_precision("high")
        print("[opt] enabled CUDA fast-math backend flags", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[opt] enabling CUDA fast-math flags failed: {exc!r}", flush=True)


def _load_klein_pipeline_cpu():
    """Pre-snapshot loader: weights end up on CPU so the snapshot captures them."""
    import torch
    from diffusers import DiffusionPipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN is required to load gated model {KLEIN_REPO}")

    # device_map="cuda" + dtype= leaves sub-modules (e.g. text encoder) in fp32
    # and bombs with "BFloat16 != float" matmul errors on the first step. The
    # working pattern is torch_dtype=bf16 + .to("cuda"), which casts every
    # submodule uniformly. Here we load to CPU only so the memory snapshot is
    # GPU-free (Modal cannot snapshot GPU state); the .to("cuda") happens in
    # the snap=False enter below.
    pipe = DiffusionPipeline.from_pretrained(
        KLEIN_REPO,
        torch_dtype=torch.bfloat16,
        token=token,
        cache_dir=HF_HOME,
    )
    pipe.set_progress_bar_config(disable=True)
    print(
        json.dumps({
            "event": "klein_loaded_cpu",
            "repo": KLEIN_REPO,
            "steps": KLEIN_STEPS,
            "gpu": KLEIN_GPU,
        }),
        flush=True,
    )
    return pipe


def _call_klein_pipe(pipe: Any, image_obj: Any, *, prompt: str, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            image=image_obj,
            prompt=prompt,
            num_inference_steps=KLEIN_STEPS,
            generator=generator,
        )
    torch.cuda.synchronize()
    return result.images[0] if hasattr(result, "images") else result[0]


def _warmup_klein_pipe(pipe: Any) -> None:
    try:
        from PIL import Image

        dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))
        _ = _call_klein_pipe(pipe, dummy, prompt="Erase only the text.", seed=0)
        print(json.dumps({"event": "warmup_done", "steps": KLEIN_STEPS}), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] skipped: {exc!r}", flush=True)


def _edit_batch_klein_impl(
    payload: dict[str, Any],
    *,
    pipe: Any,
    variant: str = "klein_local_9b_4step",
    target_long_side: int = 1024,
) -> dict[str, Any]:
    pages = list(payload.get("pages") or [])
    if not pages:
        raise ValueError("payload.pages is required and non-empty")

    bucket = str(payload.get("bucket") or S3_BUCKET_DEFAULT).strip()
    run_id = str(payload.get("run_id") or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    overwrite = bool(payload.get("overwrite", False))
    trust_manifest = bool(payload.get("trust_manifest", False))
    failed_prefix = str(payload.get("failed_prefix") or "datasets/pages/text_removed/_failed").strip().strip("/")
    prompt_text = _klein_prompt(str(payload.get("prompt_override") or ""))
    seed_base = int(payload.get("seed") or 0)

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

    download_start = time.perf_counter()
    work_items: list[dict[str, Any]] = []

    def _download(page: dict[str, Any], index: int) -> dict[str, Any]:
        page_key = str(page["page_key"])
        output_key = str(page["output_key"])
        chapter = str(page.get("chapter") or "")
        page_id = str(page.get("page_id") or Path(page_key).stem)
        sample_id = str(page.get("sample_id") or (f"{chapter}__{page_id}" if chapter else page_id))
        image_obj, source = _download_rgb_image(bucket, page_key)
        return {
            "index": index,
            "image": _normalize_size(image_obj, target_long=target_long_side),
            "orig_size": image_obj.size,
            "source": source,
            "sample_id": sample_id,
            "output_key": output_key,
        }

    with ThreadPoolExecutor(max_workers=min(16, len(todo))) as pool:
        for item in pool.map(lambda pair: _download(pair[1], pair[0]), enumerate(todo)):
            work_items.append(item)
    download_sec = time.perf_counter() - download_start

    annotated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    inference_start = time.perf_counter()
    for item in work_items:
        try:
            item_start = time.perf_counter()
            out_img = _call_klein_pipe(
                pipe,
                item["image"],
                prompt=prompt_text,
                seed=seed_base + int(item["index"]),
            )
            inference_item_sec = time.perf_counter() - item_start
            if out_img.size != item["orig_size"]:
                from PIL import Image

                out_img = out_img.resize(item["orig_size"], Image.Resampling.LANCZOS)
            body_size = _put_png_to_s3(bucket, str(item["output_key"]), out_img)
            annotated.append({
                "sample_id": item["sample_id"],
                "page_key": item["source"].get("key"),
                "output_key": item["output_key"],
                "status": "ok",
                "bytes": body_size,
                "sec": round(inference_item_sec, 3),
            })
        except Exception as exc:  # noqa: BLE001
            failure_key = f"{failed_prefix}/{run_id}/{item['sample_id']}.json"
            try:
                _put_json_to_s3(
                    bucket,
                    failure_key,
                    {
                        "sample_id": item["sample_id"],
                        "page_key": item["source"].get("key"),
                        "output_key": item["output_key"],
                        "run_id": run_id,
                        "variant": variant,
                        "error": repr(exc)[:1000],
                    },
                )
            except Exception:
                pass
            errors.append({
                "sample_id": item["sample_id"],
                "page_key": item["source"].get("key"),
                "output_key": item["output_key"],
                "failure_key": failure_key,
                "status": "error",
                "error": repr(exc)[:500],
            })

    inference_sec = time.perf_counter() - inference_start
    return {
        "ok": not errors,
        "annotated": annotated,
        "skipped": skipped,
        "errors": errors,
        "stats": {
            "variant": variant,
            "repo": KLEIN_REPO,
            "batch_size": len(work_items),
            "steps": KLEIN_STEPS,
            "download_sec": round(download_sec, 3),
            "inference_sec": round(inference_sec, 3),
            "pages_per_sec": round(len(work_items) / inference_sec, 3) if inference_sec > 0 else None,
        },
    }


@app.cls(**KLEIN_KW)
class FluxKleinLocal:
    @modal.enter(snap=True)
    def load_model_to_cpu(self) -> None:
        """Pre-snapshot: load Klein weights to CPU. Snapshotted once per app
        version; subsequent containers restore in ~10s instead of re-loading."""
        self.pipe = _load_klein_pipeline_cpu()

    @modal.enter(snap=False)
    def move_model_to_gpu(self) -> None:
        """Post-snapshot: move the bf16 pipe to CUDA and warm the JIT kernels
        so the first real call lands on a hot GPU."""
        import torch

        _enable_cuda_fast_math(torch)
        self.pipe = self.pipe.to("cuda")
        print(
            json.dumps({"event": "klein_on_gpu", "gpu": KLEIN_GPU, "steps": KLEIN_STEPS}),
            flush=True,
        )
        _warmup_klein_pipe(self.pipe)

    @modal.method()
    def edit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _edit_batch_klein_impl(payload, pipe=self.pipe)


def _build_shards(
    *,
    manifest_path: str,
    variant: str,
    bucket: str,
    run_id: str,
    overwrite: bool,
    pages_per_shard: int,
    trust_manifest: bool,
) -> tuple[str, list[dict[str, Any]]]:
    if variant != "klein_local_9b_4step":
        raise ValueError("Only production variant is 'klein_local_9b_4step'")

    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    effective_run_id = run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shard_size = max(1, int(pages_per_shard))
    shards = [
        {
            "bucket": bucket,
            "run_id": effective_run_id,
            "overwrite": overwrite,
            "trust_manifest": trust_manifest,
            "pages": rows[start : start + shard_size],
        }
        for start in range(0, len(rows), shard_size)
    ]
    return effective_run_id, shards


@app.local_entrypoint()
def annotate_manifest_local(
    manifest_path: str,
    variant: str = "klein_local_9b_4step",
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    overwrite: bool = False,
    pages_per_shard: int = 8,
    trust_manifest: bool = False,
):
    effective_run_id, shards = _build_shards(
        manifest_path=manifest_path,
        variant=variant,
        bucket=bucket,
        run_id=run_id,
        overwrite=overwrite,
        pages_per_shard=pages_per_shard,
        trust_manifest=trust_manifest,
    )
    if not shards:
        print("manifest is empty; nothing to do")
        return

    print(
        json.dumps({
            "event": "start",
            "run_id": effective_run_id,
            "variant": variant,
            "pages": sum(len(s["pages"]) for s in shards),
            "shards": len(shards),
            "shard_size": max(1, int(pages_per_shard)),
            "max_containers": KLEIN_MAX_CONTAINERS,
            "gpu": KLEIN_GPU,
            "steps": KLEIN_STEPS,
        }),
        flush=True,
    )

    cls = FluxKleinLocal()
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
        print(
            json.dumps({
                "event": "shard_done",
                "completed_shards": idx + 1,
                "annotated_total": annotated_total,
                "skipped_total": skipped_total,
                "error_total": error_total,
                "elapsed_sec": round(elapsed, 1),
                "pages_per_sec_cluster": round(rate, 2),
                "stats": result.get("stats"),
            }),
            flush=True,
        )
    wall_sec = time.perf_counter() - wall_start
    print(
        json.dumps({
            "event": "done",
            "run_id": effective_run_id,
            "variant": variant,
            "annotated_total": annotated_total,
            "skipped_total": skipped_total,
            "error_total": error_total,
            "wall_sec": round(wall_sec, 1),
            "pages_per_sec_cluster": round(annotated_total / wall_sec, 2) if wall_sec > 0 else None,
        }),
        flush=True,
    )


@app.local_entrypoint()
def annotate_manifest_spawn(
    manifest_path: str,
    variant: str = "klein_local_9b_4step",
    bucket: str = S3_BUCKET_DEFAULT,
    run_id: str = "",
    overwrite: bool = False,
    pages_per_shard: int = 8,
    trust_manifest: bool = False,
):
    """Detach-safe variant: submit every shard with .spawn() and exit.

    .map() in a --detach run prints a warning that pending calls may be cancelled
    if the local caller disconnects. .spawn() returns a FunctionCall immediately,
    so we can fire all shards and exit cleanly while Modal keeps running them.
    Progress is visible at the printed dashboard URL.
    """
    effective_run_id, shards = _build_shards(
        manifest_path=manifest_path,
        variant=variant,
        bucket=bucket,
        run_id=run_id,
        overwrite=overwrite,
        pages_per_shard=pages_per_shard,
        trust_manifest=trust_manifest,
    )
    if not shards:
        print("manifest is empty; nothing to do")
        return

    print(
        json.dumps({
            "event": "spawn_start",
            "run_id": effective_run_id,
            "variant": variant,
            "pages": sum(len(s["pages"]) for s in shards),
            "shards": len(shards),
            "shard_size": max(1, int(pages_per_shard)),
            "max_containers": KLEIN_MAX_CONTAINERS,
            "gpu": KLEIN_GPU,
            "steps": KLEIN_STEPS,
        }),
        flush=True,
    )

    cls = FluxKleinLocal()
    call_ids: list[str] = []
    for idx, shard in enumerate(shards):
        call = cls.edit_batch.spawn(shard)
        call_ids.append(call.object_id)
        if idx == 0 or (idx + 1) % 100 == 0 or idx == len(shards) - 1:
            print(
                json.dumps({
                    "event": "shard_spawned",
                    "shard_index": idx,
                    "call_id": call.object_id,
                }),
                flush=True,
            )

    print(
        json.dumps({
            "event": "spawn_done",
            "run_id": effective_run_id,
            "variant": variant,
            "spawned": len(call_ids),
            "first_call_id": call_ids[0] if call_ids else None,
            "last_call_id": call_ids[-1] if call_ids else None,
            "note": "shards run remotely; track via Modal dashboard. Local exit is safe.",
        }),
        flush=True,
    )
