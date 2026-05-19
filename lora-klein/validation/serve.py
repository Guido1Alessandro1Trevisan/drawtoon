#!/usr/bin/env python3
"""Modal serving app for the FLUX.2 Klein 9B full fine-tune.

Loads the canonical training checkpoint once per container, exposes a single
POST endpoint that takes a panel-level generation request, and returns the
S3 URL of the generated PNG.

Deploy:

    modal deploy lora-klein/validation/serve.py

Test:

    curl -X POST "$(modal app endpoint show drawtoon-flux2-klein-serve)/generate" \
         -H 'content-type: application/json' \
         -H 'x-auth-token: $TOKEN' \
         -d '{"prompt":"...","width":696,"height":446,"layout_metadata":{...},"character_ref_urls":["..."]}'
"""

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import modal
from fastapi import Response
from pydantic import BaseModel, Field


APP_NAME = os.environ.get("DRAWTOON_SERVE_APP_NAME", "drawtoon-flux2-klein-serve")


class GenerateRequest(BaseModel):
    auth_token: str = ""
    prompt: str = ""
    width: int = 0
    height: int = 0
    seed: int = 42
    steps: int = 50
    guidance_scale: float = 4.0
    layout_metadata: dict = Field(default_factory=dict)
    character_ref_urls: list[str] = Field(default_factory=list)
S3_BUCKET = os.environ.get("DRAWTOON_S3_BUCKET") or os.environ.get("S3_BUCKET") or "drawtoon"
S3_OUTPUT_PREFIX = os.environ.get(
    "DRAWTOON_SERVE_PREFIX", "serving/flux2_klein/generated"
)
DEFAULT_HF_SECRET_NAME = os.environ.get(
    "DRAWTOON_HF_SECRET_NAME", "drawtoon-flux2-inference-hf-token"
)
DEFAULT_AWS_SECRET_NAME = os.environ.get(
    "DRAWTOON_AWS_SECRET_NAME", "lineart2-aws-s3"
)
SERVE_AUTH_SECRET_NAME = os.environ.get(
    "DRAWTOON_SERVE_AUTH_SECRET_NAME", "drawtoon-flux2-serve-token"
)
DEFAULT_CHECKPOINT_URI = os.environ.get(
    "DRAWTOON_DEFAULT_CHECKPOINT_URI",
    "s3://drawtoon/models/mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lr28e7_ga8_1epoch/checkpoints/mangazero_flux2_klein9b_panel_prediction_same_page_refs_native_pad16_haiku45_lr28e7_ga8_1epoch_000002000.safetensors",
)

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
    .pip_install("requests")
    .add_local_dir(
        str(LORA_ROOT / "ai-toolkit"), remote_path="/root/ai-toolkit", copy=False
    )
    .add_local_python_source("validate_run")
)

app = modal.App(APP_NAME, image=image)
aws_secret = modal.Secret.from_name(DEFAULT_AWS_SECRET_NAME)
hf_secret = modal.Secret.from_name(DEFAULT_HF_SECRET_NAME)
auth_secret = modal.Secret.from_name(SERVE_AUTH_SECRET_NAME)
validation_cache = modal.Volume.from_name(
    "flux-klein-validation-cache", create_if_missing=True
)


@app.cls(
    gpu="H100",
    cpu=8,
    memory=65536,
    secrets=[aws_secret, hf_secret, auth_secret],
    volumes={"/root/validation_cache": validation_cache},
    scaledown_window=900,
    timeout=15 * 60,
    max_containers=2,
)
class FluxKleinServer:
    @modal.enter()
    def load(self):
        from validate_run import load_flux_pipeline

        self.checkpoint_uri = DEFAULT_CHECKPOINT_URI
        self.pipe, self.load_stats = load_flux_pipeline(
            base=False,
            checkpoint_uri=self.checkpoint_uri,
            overlay_lora_uri="",
            lora_scale=1.0,
        )
        print(
            f"[serve] loaded checkpoint={self.checkpoint_uri} "
            f"stats={json.dumps(self.load_stats)}",
            flush=True,
        )

    @modal.fastapi_endpoint(method="POST", docs=False)
    def generate(self, payload: GenerateRequest):
        import boto3
        import requests
        import torch
        from PIL import Image, ImageDraw
        from validate_run import draw_text_layout_shape

        LAYOUT_BACKGROUND_COLOR = (0, 0, 0)
        LAYOUT_TAIL_COLOR = (192, 192, 192)

        def materialize_layout_control_filled(layout_metadata, width, height):
            """Match training pipeline: SOLID FILLED character rectangles.

            run_modal.py:_draw_layout_blob renders training-time character
            boxes as solid filled rects (line 1921 fill=color). validate_run's
            reconstruction uses outlined rects which diverges from training.
            For full fine-tune inference we want filled to match training.
            """
            img = Image.new("RGB", (width, height), LAYOUT_BACKGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            for character in layout_metadata.get("characters") or []:
                if not isinstance(character, dict):
                    continue
                rgb = character.get("rgb")
                box = character.get("bbox_norm")
                if not isinstance(rgb, list) or not isinstance(box, list) or len(box) != 4:
                    continue
                x0 = int(round(max(0.0, min(1.0, float(box[0]))) * width))
                y0 = int(round(max(0.0, min(1.0, float(box[1]))) * height))
                x1 = int(round(max(0.0, min(1.0, float(box[2]))) * width))
                y1 = int(round(max(0.0, min(1.0, float(box[3]))) * height))
                if x1 > x0 and y1 > y0:
                    draw.rectangle(
                        (x0, y0, x1, y1),
                        fill=tuple(int(v) for v in rgb),
                    )
            text_payload = (
                layout_metadata.get("text")
                if isinstance(layout_metadata.get("text"), dict)
                else {}
            )
            for region in text_payload.get("regions") or []:
                if isinstance(region, dict) and region.get("masked", True):
                    draw_text_layout_shape(
                        draw,
                        region.get("bbox_norm"),
                        width=width,
                        height=height,
                        text_bubble_type=str(region.get("type") or "Speech Bubble"),
                    )
            tail_payload = (
                layout_metadata.get("tail")
                if isinstance(layout_metadata.get("tail"), dict)
                else {}
            )
            for region in tail_payload.get("regions") or []:
                if not isinstance(region, dict):
                    continue
                box = region.get("bbox_norm")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                x0 = int(round(max(0.0, min(1.0, float(box[0]))) * width))
                y0 = int(round(max(0.0, min(1.0, float(box[1]))) * height))
                x1 = int(round(max(0.0, min(1.0, float(box[2]))) * width))
                y1 = int(round(max(0.0, min(1.0, float(box[3]))) * height))
                if x1 > x0 and y1 > y0:
                    draw.rectangle((x0, y0, x1, y1), fill=LAYOUT_TAIL_COLOR)
            return img

        materialize_layout_control = materialize_layout_control_filled

        expected = os.environ.get("DRAWTOON_SERVE_AUTH_TOKEN") or ""
        if not expected or payload.auth_token != expected:
            return Response(
                content=json.dumps({"error": "unauthorized"}),
                status_code=401,
                media_type="application/json",
            )

        prompt = payload.prompt.strip()
        width = int(payload.width)
        height = int(payload.height)
        if not prompt or width <= 0 or height <= 0:
            return Response(
                content=json.dumps({"error": "missing prompt/width/height"}),
                status_code=400,
                media_type="application/json",
            )
        seed = int(payload.seed)
        steps = int(payload.steps)
        guidance_scale = float(payload.guidance_scale)
        layout_metadata = payload.layout_metadata or {}
        character_ref_urls = list(payload.character_ref_urls or [])[:6]

        layout_image = materialize_layout_control(
            layout_metadata, width=width, height=height
        )
        layout_buf = io.BytesIO()
        layout_image.save(layout_buf, format="PNG", optimize=True)
        layout_bytes = layout_buf.getvalue()
        control_images = [layout_image]
        for url in character_ref_urls:
            res = requests.get(url, timeout=60)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            control_images.append(img)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        started = time.time()
        result = self.pipe(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            control_img_list=control_images,
        )
        generated = result.images[0]
        seconds = round(time.time() - started, 3)

        buf = io.BytesIO()
        generated.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        run_id = uuid.uuid4().hex
        key = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{run_id}.png"
        layout_key = f"{S3_OUTPUT_PREFIX.rstrip('/')}/{run_id}-ctrl_img_1.png"
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key, Body=png_bytes, ContentType="image/png"
        )
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=layout_key,
            Body=layout_bytes,
            ContentType="image/png",
        )
        presigned = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        layout_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": layout_key},
            ExpiresIn=7 * 24 * 3600,
        )

        return {
            "url": presigned,
            "s3_uri": f"s3://{S3_BUCKET}/{key}",
            "layout_control_url": layout_url,
            "layout_control_s3_uri": f"s3://{S3_BUCKET}/{layout_key}",
            "seconds": seconds,
            "checkpoint_uri": self.checkpoint_uri,
            "control_image_count": len(control_images),
            "width": width,
            "height": height,
            "seed": seed,
        }
