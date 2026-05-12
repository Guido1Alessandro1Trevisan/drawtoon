from __future__ import annotations

from pathlib import Path

import modal


S3_BUCKET = "drawtoon"
VOLUME_MOUNT = Path("/mnt/models")

app = modal.App("lineart2-sync-existing-checkpoints-to-s3")
model_volume = modal.Volume.from_name("flux-lora-models")
aws_secret = modal.Secret.from_name("lineart2-aws-s3")
image = modal.Image.debian_slim().pip_install("boto3")


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT): model_volume},
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
        save_root = VOLUME_MOUNT / job_name
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
def main(jobs: str):
    job_names = [part.strip() for part in jobs.split(",") if part.strip()]
    if not job_names:
        raise ValueError("Pass --jobs with one job name or comma-separated job names")
    result = sync_existing_checkpoints.remote(job_names)
    print(result)
