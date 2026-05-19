"""Modal launcher for the LayouSyn DiT fine-tune.

Trains either the **panel-layout** or **page-layout** model (same code, two
configs). Both warm-start from the released HF checkpoint
`dsrivastavv/Lay-Your-Scene/grit/model.pt` and fine-tune on the drawtoon
JSONL corpus produced by `data_prep/build_panel_dataset.py`.

The whole LayouSyn source tree is **vendored** at
`lora-klein/creatilayout/layousyn/` (a pristine `mlpc-ucsd/Lay-Your-Scene`
@769e9bf snapshot plus the in-tree `MangaLayout` dataset). The Modal image
bundles it via `add_local_dir`; no `git clone`, no runtime patching.

Usage:

    modal run lora-klein/creatilayout/training/modal_train_layousyn.py \\
        --variant panel \\
        --run-name layousyn-panel-v1 \\
        --epochs 30

Defaults — overridable via CLI flags:

  --train-jsonl-s3   s3://drawtoon-layousyn/datasets/panel_layout/train.jsonl
  --output-s3-prefix s3://drawtoon-layousyn/models/<run_name>/

Outputs land at:

    s3://drawtoon-layousyn/models/<run_name>/
        checkpoints/<step>.pt
        final.pt
        config.json
        log.txt
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import modal


APP_NAME = "drawtoon-layousyn-train"
REPO_DIR = "/opt/Lay-Your-Scene"
HF_WARMSTART_REPO = "dsrivastavv/Lay-Your-Scene"
HF_WARMSTART_FILE = "grit/model.pt"
WARMSTART_LOCAL = "/cache/warmstart/grit_model.pt"

DEFAULT_S3_BUCKET = "drawtoon-layousyn"
DEFAULT_OUTPUT_PREFIX = "models"
DEFAULT_TRAIN_JSONL_S3 = "s3://drawtoon-layousyn/datasets/panel_layout/train.jsonl"
DEFAULT_VAL_JSONL_S3 = "s3://drawtoon-layousyn/datasets/panel_layout/val.jsonl"

LOCAL_DIR = Path(__file__).resolve().parent
VENDORED_LAYOUSYN = LOCAL_DIR / "layousyn"


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        # torch>=2.6 required by recent transformers for `torch.load(weights_only=True)`
        # gating (CVE-2025-32434). t5-v1_1-base only ships pytorch_model.bin, no safetensors.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "timm",
        "sentence_transformers",
        "diffusers",
        "transformers",
        "bs4",
        "pytorch_fid",
        "pycocotools",
        "tiktoken",
        "protobuf",
        "blobfile",
        "sentencepiece==0.2.0",
        "accelerate",
        "comet_ml",
        "tensorboard",
        "pandas",
        "pyarrow",
        "boto3",
        "huggingface_hub",
        # Required by upstream's evaluation/common.py (transitively imported
        # from train.py via the prompt_handler module). We don't actually use
        # GPT-based eval, but the import is unconditional.
        "openai",
        "requests",
        "git+https://github.com/openai/CLIP.git",
    )
    # Drop the vendored LayouSyn tree into the image. This includes the
    # upstream Python package, train.py, demo.py, environment.yml, plus our
    # panel/ + page/ config folders and the MangaLayout dataset (already
    # registered in-tree).
    .add_local_dir(str(VENDORED_LAYOUSYN), REPO_DIR, copy=True)
)

app = modal.App(APP_NAME, image=image)

aws_secret = modal.Secret.from_name("lineart2-aws-s3")
# HF Lay-Your-Scene checkpoint repo is public, but using a token avoids
# anonymous-rate-limit issues. `hf-token` is the existing workspace secret.
hf_secret = modal.Secret.from_name("hf-token")
volume = modal.Volume.from_name("drawtoon-layousyn-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s3_client(*, region: str = "us-east-1"):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def _download_s3(uri: str, dest: Path) -> Path:
    bucket, key = _parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"S3 download: {uri} -> {dest}")
    _s3_client().download_file(bucket, key, str(dest))
    return dest


def _upload_s3(src: Path, uri: str) -> None:
    bucket, key = _parse_s3_uri(uri)
    print(f"S3 upload: {src} -> {uri}")
    _s3_client().upload_file(str(src), bucket, key)


def _download_warmstart() -> Path:
    dest = Path(WARMSTART_LOCAL)
    if dest.exists():
        print(f"Warmstart cached at {dest}")
        return dest
    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_WARMSTART_REPO}/{HF_WARMSTART_FILE} ...")
    local = hf_hub_download(
        repo_id=HF_WARMSTART_REPO,
        filename=HF_WARMSTART_FILE,
        token=os.environ.get("HF_TOKEN"),
        cache_dir="/cache/hf",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local, dest)
    return dest


def _write_run_config(
    *,
    template_config_path: Path,
    train_jsonl_path: Path,
    val_jsonl_path: Optional[Path],
    embed_dir: Path,
    result_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    ckpt_every: int,
    out_path: Path,
) -> Path:
    with open(template_config_path) as fh:
        config = json.load(fh)
    config["embed_dir"] = str(embed_dir)
    config["result_dir"] = str(result_dir)
    config["epochs"] = int(epochs)
    config["global_batch_size"] = int(batch_size)
    config["lr"] = float(lr)
    config["ckpt_every"] = int(ckpt_every)
    config["datasets"] = {
        "MangaLayout": {
            "jsonl_path": str(train_jsonl_path),
            "shuffle_bbox": False,
            "crop_augment": False,
        }
    }
    # drawtoon patch: end-of-epoch val-loss pass.
    config["val_jsonl_path"] = str(val_jsonl_path) if val_jsonl_path else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_path


def _sync_checkpoints_to_s3(
    *, experiment_dir: Path, s3_prefix: str, stop_flag: list[bool]
) -> None:
    uploaded: set[str] = set()

    def _scan_and_upload() -> None:
        if not experiment_dir.exists():
            return
        for path in experiment_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(experiment_dir).as_posix()
            if rel in uploaded:
                continue
            if path.suffix in (".pt", ".json", ".txt") or "tensorboard" in rel:
                try:
                    _upload_s3(path, f"{s3_prefix.rstrip('/')}/{rel}")
                    uploaded.add(rel)
                except Exception as exc:  # noqa: BLE001
                    print(f"Upload failed for {rel}: {exc}")

    while not stop_flag[0]:
        try:
            _scan_and_upload()
        except Exception as exc:  # noqa: BLE001
            print(f"Sync loop error: {exc}")
        time.sleep(60)
    _scan_and_upload()


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------


# GPU choice: H200 (141 GB VRAM) is the default — model is tiny so VRAM
# is not the constraint, but H200 has the highest throughput per dollar
# for this single-GPU fine-tune. Override via the LAYOUSYN_GPU env var on
# the local side, OR edit this line and re-deploy the app.
LAYOUSYN_GPU = os.environ.get("LAYOUSYN_GPU", "H200")


@app.function(
    gpu=LAYOUSYN_GPU,
    cpu=8,
    memory=32768,
    secrets=[aws_secret, hf_secret],
    volumes={"/cache": volume},
    timeout=60 * 60 * 12,
)
def train(
    variant: str = "panel",
    run_name: str = "layousyn-panel-v1",
    train_jsonl_s3: str = DEFAULT_TRAIN_JSONL_S3,
    val_jsonl_s3: str = DEFAULT_VAL_JSONL_S3,
    warmstart: bool = True,
    epochs: Optional[int] = None,
    batch_size: int = 256,
    lr: float = 1e-4,
    ckpt_every: int = 5000,
    output_s3_prefix: str = "",
    debug: bool = False,
) -> dict:
    if variant not in ("panel", "page"):
        raise ValueError(f"variant must be 'panel' or 'page', got {variant!r}")
    if not train_jsonl_s3:
        raise ValueError("train_jsonl_s3 is required")
    if not output_s3_prefix:
        output_s3_prefix = f"s3://{DEFAULT_S3_BUCKET}/{DEFAULT_OUTPUT_PREFIX}/{run_name}"

    workdir = Path(f"/cache/runs/{run_name}")
    data_dir = workdir / "data"
    config_dir = workdir / "configs"
    result_dir = workdir / "results"
    embed_dir = workdir / "embeds"
    for path in (workdir, data_dir, config_dir, result_dir, embed_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_jsonl_path = data_dir / "train.jsonl"
    if not train_jsonl_path.exists():
        _download_s3(train_jsonl_s3, train_jsonl_path)

    val_jsonl_path: Optional[Path] = None
    if val_jsonl_s3:
        val_jsonl_path = data_dir / "val.jsonl"
        if not val_jsonl_path.exists():
            try:
                _download_s3(val_jsonl_s3, val_jsonl_path)
            except Exception as exc:  # noqa: BLE001
                print(f"val.jsonl download skipped: {exc}")
                val_jsonl_path = None

    # Configs live in the vendored tree at panel/config.json and page/config.json.
    template = Path(REPO_DIR) / variant / "config.json"
    if not template.exists():
        raise FileNotFoundError(f"Missing template config: {template}")
    template_config = json.loads(template.read_text(encoding="utf-8"))
    effective_epochs = epochs if epochs is not None else int(template_config["epochs"])

    config_path = _write_run_config(
        template_config_path=template,
        train_jsonl_path=train_jsonl_path,
        val_jsonl_path=val_jsonl_path,
        embed_dir=embed_dir,
        result_dir=result_dir,
        epochs=effective_epochs,
        batch_size=batch_size,
        lr=lr,
        ckpt_every=ckpt_every,
        out_path=config_dir / "run_config.json",
    )

    warmstart_path: Optional[Path] = None
    if warmstart:
        warmstart_path = _download_warmstart()

    stop_flag = [False]
    sync_thread = threading.Thread(
        target=_sync_checkpoints_to_s3,
        kwargs={
            "experiment_dir": result_dir,
            "s3_prefix": output_s3_prefix,
            "stop_flag": stop_flag,
        },
        daemon=True,
    )
    sync_thread.start()

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "train.py",
        "--config",
        str(config_path),
    ]
    if warmstart_path is not None:
        cmd += ["--resume-ckpt", str(warmstart_path)]
    if debug:
        cmd += ["--debug"]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_DIR, env=env)
    stop_flag[0] = True
    sync_thread.join(timeout=300)

    if proc.returncode != 0:
        raise RuntimeError(f"torchrun exited {proc.returncode}")

    final_pt = next(result_dir.rglob("final.pt"), None)
    if final_pt is not None:
        _upload_s3(final_pt, f"{output_s3_prefix.rstrip('/')}/final.pt")
    _upload_s3(config_path, f"{output_s3_prefix.rstrip('/')}/config.json")

    return {
        "status": "ok",
        "run_name": run_name,
        "variant": variant,
        "output_s3_prefix": output_s3_prefix,
        "final_pt_uploaded": bool(final_pt),
    }


@app.local_entrypoint()
def main(
    variant: str = "panel",
    run_name: str = "layousyn-panel-v1",
    train_jsonl_s3: str = DEFAULT_TRAIN_JSONL_S3,
    val_jsonl_s3: str = DEFAULT_VAL_JSONL_S3,
    warmstart: bool = True,
    epochs: int = 0,
    batch_size: int = 256,
    lr: float = 1e-4,
    ckpt_every: int = 5000,
    output_s3_prefix: str = "",
    debug: bool = False,
) -> None:
    result = train.remote(
        variant=variant,
        run_name=run_name,
        train_jsonl_s3=train_jsonl_s3,
        val_jsonl_s3=val_jsonl_s3,
        warmstart=warmstart,
        epochs=epochs if epochs > 0 else None,
        batch_size=batch_size,
        lr=lr,
        ckpt_every=ckpt_every,
        output_s3_prefix=output_s3_prefix,
        debug=debug,
    )
    print(result)
