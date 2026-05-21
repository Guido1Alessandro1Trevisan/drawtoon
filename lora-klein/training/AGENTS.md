# Training Guide

## Scope

This file applies to `lora-klein/training/`.

Use it for Modal training launches, config selection, validation sync, and training-side troubleshooting.

## Active Pipeline

The standard path is:

1. Use Drawtoon canonical S3 data:
   - `datasets/pages/filtered`
   - `datasets/annotations/magi_v3`
   - `captions/<caption_run>`
2. Build a temporary Modal Volume cache with CPU workers before GPU allocation.
3. Train ai-toolkit from the generated local `manifest.jsonl`.
4. Mirror durable artifacts to `s3://drawtoon/models/<job_name>/`.

Do not reintroduce S3 `sample_records`, S3 `caption_groups`, or S3 `training_views` as active training inputs.

## Launch Rule

Default practice is:

```text
--preset <config-relative-path-without-.yaml>
```

The current Drawtoon preset is:

```text
haiku-4.5/panel_prediction_same_page_refs_native_pad16_lr28e7_ga8
```

## Directory Layout

- `run_modal.py`: active Modal launcher.
- `configs/`: active presets.
- `utils.py`: one Python helper entrypoint for launcher support and maintenance jobs.
- `sync_validate_images.sh`: local validation-image sync helper.

## Dataset Encoding

Keep ai-toolkit training on `type: manifest`. The launcher fills `manifest_path` after it builds the temporary Drawtoon cache.

Use upstream-style dataset controls:

- `buckets: false`
- `batch_size: 1`
- `cache_latents: false`
- `cache_latents_to_disk: false`

Target panels are padded to model divisibility by the cache builder before training.

## Storage Architecture

- S3 is the canonical source for pages, annotations, captions, and model artifacts.
- Modal volume `flux-dataset-cache` stores rebuildable training cache files.
- Modal volume `flux-lora-models` is older single-GPU resume state only.
- DDP checkpoints are written to local disk first, then mirrored to S3.

## On-demand checkpoint trigger

Touch a file inside the running container to force a normal checkpoint save at the next step boundary, no restart required. Implementation lives in `lora-klein/ai-toolkit/jobs/process/BaseSDTrainProcess.py` (`_external_checkpoint_requested` / `_consume_external_checkpoint_trigger`).

- Default trigger path: `<save_root>/.request_checkpoint` (i.e. inside the job's output dir under `training_folder`).
- Override the path via the env var `AITK_CHECKPOINT_TRIGGER_FILE=/some/abs/path` when launching the Modal function.
- Rank 0 checks the filesystem and broadcasts a 1-int decision to every DDP rank, so all ranks promote the same step to a save_step. The existing `accelerator.wait_for_everyone()` barriers handle synchronisation; the existing `self.save(step_num)` path handles weights, optimizer state, and S3 mirror. Trigger file is removed by rank 0 after the save completes.

Usage example:

```bash
# find the rank-0 container in the running app
modal container list <app-id>

# touch the trigger file inside the running container
modal container exec <container-id> -- bash -lc \
  'touch /root/lineart2/training_output/<job_name>/.request_checkpoint'

# the trainer will detect it on the next step (≤ one optimizer step of latency),
# run a normal save, then delete the trigger file
```

Safe to leave in place permanently; if the trigger file does not exist the check is a single `os.path.exists` on rank 0 and a 1-element NCCL broadcast — negligible overhead.

## EC2 launch playbook (capacity-block + Docker)

The first end-to-end EC2 run on B300 (2026-05-20, capacity reservation `cr-0d09325795399c31b`) hit several non-obvious failures before training kicked off. Document them here so the next launch can skip them. The general flow is: capacity-block purchase → minimal-bootstrap user-data → SSM-driven Docker build → `docker run train.sh` inside the container.

### Pre-flight checklist (do these before launching the instance)

1. **Tar the repo from inside its own directory.** `tar -C /path/to/parent drawtoon` puts a `drawtoon/` wrapper at the top of the archive; `train.sh:143` does `tar -xzf $TAR -C $WORK_ROOT/repo` expecting top-level files (`lora-klein/`, `workflows/`, etc.). Do this instead:
   ```bash
   tar -czf /tmp/repo.tar.gz \
     --exclude='.git' --exclude='__pycache__' --exclude='.aws-sam' \
     --exclude='artifacts' --exclude='*.safetensors' --exclude='*.bin' \
     -C /path/to/drawtoon .
   ```
2. **Cross-region IAM for HF token.** The `lineart2-p5en-ec2-role` inline policy `lineart2-hf-token-secret-read` must list **both** us-east-1 and us-west-2 secret ARNs. If the secret was created in only one region, copy it (`aws secretsmanager get-secret-value` → `aws secretsmanager create-secret`) AND patch the policy to include both ARNs. Otherwise the box fails to resolve the HF token mid-bootstrap and `train.sh` exits with Access Denied.
3. **`aws s3` region pin.** The drawtoon bucket lives in **us-east-1**. A us-west-2 EC2 instance running `aws s3 cp s3://drawtoon/...` without `--region us-east-1` gets `PermanentRedirect`. Either pin `--region us-east-1` on every `aws s3` call inside user-data OR set `AWS_DEFAULT_REGION=us-east-1` for those calls.
4. **Capacity-block launch flag.** `aws ec2 run-instances` must include both `--capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=cr-xxx}` AND `--instance-market-options MarketType=capacity-block`. Without the market option, EC2 rejects with "The market type (purchasing) option is not valid."
5. **Don't combine `set -o pipefail` with `| head -N`.** `set -euo pipefail` + `docker info | head -20` exits the script because docker info gets SIGPIPE → pipefail propagates → script death. Use `head` only without pipefail, or capture full output without head.

### Docker image strategy

Two options have been tried; the second is preferred for future runs.

**A) Container that re-runs `train.sh` (what 2026-05-20 used)**: Dockerfile pre-installs torch + ai-toolkit deps into `/usr/local/lib/python3.11/site-packages`, container CMD is `bash /workspace/drawtoon/lora-klein/training/ec2/train.sh`. **Wasted work** — `train.sh:146-149` creates its own venv at `$WORK_ROOT/venv` and pip-installs everything *again* inside the container (~15 min). The image's pre-install is dead weight.

**B) Container that calls `torchrun` directly (recommended)**: Dockerfile installs deps **and the venv goes where Python expects them**. Container CMD bypasses `train.sh` entirely and does the four phases inline (`utils.py build-ec2-cache`, `utils.py prepare-ec2-config`, `torchrun`, S3 sync). Saves ~15 min and removes a layer of indirection. Either patch `train.sh` to skip the venv create when `$VENV_DIR/bin/python` already points at a working interpreter, or write a new `container_train.sh` that mirrors `train.sh`'s phases without the venv step.

### Worker counts on B300

`p6-b300.48xlarge` has **192 vCPUs**, not 96. The default `--workers 96` in `utils.py:build_drawtoon_panel_cache` was tuned for older boxes; bump to 192 for B300 (already bumped on 2026-05-20). With 192 workers the manga+comic cache (137k captions, 128 shards) finishes in ~20 min instead of ~40 min.

### NVMe + Docker storage on p6-b300

p6-b300 ships with multiple instance-store NVMes (typically `/dev/nvme1n1` + others, ~3.8 TB usable). The bootstrap script must:
1. `mkfs.ext4 -F -E nodiscard /dev/nvme1n1 && mount /dev/nvme1n1 /mnt/local`
2. Set Docker `data-root` to `/mnt/local/docker` via `/etc/docker/daemon.json` so layer storage lands on NVMe (not the 500 GB EBS root, which would fill up).
3. Set `HF_HOME=/mnt/local/drawtoon/hf` so model downloads land on NVMe too.

### Critical pre-flight on Blackwell SXM (B100 / B200 / B300): nvidia-fabricmanager

**Symptom**: `torch.cuda.is_available()` returns `False` while `torch.cuda.device_count()` returns 8; `cudaGetDeviceCount` raises `Error 802: system not yet initialized`. The Docker container is correctly given `/dev/nvidia*` and `libcuda.so.580.x`, `nvidia-smi` works inside the container, but every CUDA call fails. Training then proceeds on CPU (each rank pegs ~87% on a single core), step counter stuck at 0, GPU mem stays at 0 MiB, and you eventually crash inside `optimizer_utils.copy_stochastic` with `AssertionError: Target is on cpu!` (because `param.device` is CPU since the model never moved).

**Cause**: NVIDIA Blackwell SXM GPUs require `nvidia-fabricmanager` running on the host to coordinate NVSwitch fabric initialization. Without it, CUDA cannot acquire the multi-GPU fabric state and every `cuInit()` returns Error 802. The 2026-05-19 build of DLAMI `ami-0f1875ad68367a3bb` (Deep Learning Base **OSS Nvidia Driver** GPU Ubuntu 22.04) **does not auto-enable fabric-manager** for B300. The package is preinstalled (`nvidia-fabricmanager-580` was already at version 580.159.03 on the box) but the systemd unit is **not enabled or started**. Modal images dodge this because they bundle and start it.

**Fix** (host-side, one-time per box):
```bash
apt-get install -y nvidia-fabricmanager-580       # noop if already installed
systemctl enable --now nvidia-fabricmanager
systemctl is-active nvidia-fabricmanager           # should print "active"
journalctl -u nvidia-fabricmanager --no-pager -n 5  # should show "state is `3` (configured)"
```

After that, inside a `docker run --gpus all ...` container:
```bash
python3 -c 'import torch; print(torch.cuda.is_available()); torch.zeros(1, device="cuda:0")'
# True
```

**Bootstrap requirement**: every B300 user-data / bootstrap script must `systemctl enable --now nvidia-fabricmanager` BEFORE any Docker container or `torchrun` is launched. Add this to `train.sh` near the apt block, OR add a separate pre-flight check that aborts with a clear message if `systemctl is-active nvidia-fabricmanager` doesn't return `active`.

**Related downstream symptoms that disappear once fabric-manager is up**:
- `Error running job: Target is on cpu!` from `stochastic_grad_accummulation` — was a downstream effect, not the real bug. Our device-aware patch in that function is still defensive and correct, but won't help if CUDA itself fails to init.
- The 30-min NCCL watchdog timeout: any DDP collective involving GPU 0 will silently fail to enqueue.

### Known torchrun crash: `omnigen2` triton autotuner @ import time

**Symptom**: every rank exits with `RuntimeError: Unexpected error from cudaGetDeviceCount(). Error 802: system not yet initialized`. Stack ends inside `lora-klein/ai-toolkit/extensions_built_in/diffusion_models/omnigen2/src/ops/triton/layer_norm.py:46` (`triton_autotune_configs`). torchrun reports "Root Cause (first observed failure): rank X exitcode 1".

**Cause**: stock ai-toolkit's `omnigen2` extension calls `cudaGetDeviceCount()` at module-import time via triton's autotuner. In a DDP subprocess spawned by torchrun, CUDA isn't yet initialized at the moment of import → error 802. We don't actually train OmniGen2 (we train FLUX-2 Klein 9B) but `extensions_built_in/diffusion_models/__init__.py` auto-imports every model class, including omnigen2.

**Fix** (apply to the repo before tarballing for the next launch):

```bash
# In lora-klein/ai-toolkit/extensions_built_in/diffusion_models/__init__.py
# delete the line  `from .omnigen2 import OmniGen2Model`
# delete the line  `    OmniGen2Model,`  (inside AI_TOOLKIT_MODELS = [...])
```

Either `sed -i` it pre-tar:
```bash
INIT=lora-klein/ai-toolkit/extensions_built_in/diffusion_models/__init__.py
sed -i '/^from \.omnigen2 /d; /^    OmniGen2Model,/d' "$INIT"
```

Or hot-patch a running EC2 box (host-side repo, then re-launch container with `-v $HOST_REPO:/workspace/drawtoon:ro` so the patched repo overrides the image's COPY'd version).

**Other auto-imported extensions that may bite the same way** (any extension whose `__init__.py` triggers a `torch.cuda.*` call before DDP setup): check `extensions_built_in/diffusion_models/__init__.py` against the actual models you train. If you only train FLUX-2 Klein 9B, the minimum set is `flux2` and (if you ever evaluate against them) `chroma`, `flux_kontext`, `hidream`. The rest can be deleted from the registry without consequence.

### Downstream symptom (not a bug): `AssertionError: Target is on cpu!` in stochastic grad accumulator

If you see the training crash at step 0 with `optimizer_utils.py:141: assert target.device.type != 'cpu', "Target is on cpu!"`, **the real cause is upstream — almost always the fabric-manager not running (Blackwell SXM). See the fabric-manager section above.** When CUDA fails to init, the model stays on CPU, the gradient accumulator gets created on CPU, and on the second micro-batch `copy_stochastic` asserts. The optimizer code is correct; **don't add a `.to(device)` workaround in `stochastic_grad_accummulation`** — it would mask the real bug (training would silently continue on CPU, ~100× slower).

History: on 2026-05-20 we briefly patched `optimizer_utils.py:218` to be device-aware as a workaround. After diagnosing that fabric-manager was missing, the patch was reverted — the function should NOT carry the workaround. If you see `Target is on cpu!`, fix the upstream CUDA init issue (fabric-manager, driver/runtime mismatch, missing /dev/nvidia*).

### Cache reuse on relaunch

`lora-klein/training/ec2/train.sh:213` passes `--overwrite` to `build-ec2-cache`, which **wipes the existing cache every relaunch**. If a previous run on the same instance already produced a complete cache (signalled by `<cache_root>/_SUCCESS` + `<cache_root>/manifest.jsonl` ≥ 1 row), drop the `--overwrite` flag so the second container start skips ~20–40 min of S3 GETs:

```bash
sed -i 's/  --overwrite | tee/  | tee/' lora-klein/training/ec2/train.sh
```

The cache fingerprint (in `_drawtoon_cache_key`) includes the regex, prefixes, caption_run, and `target_multiple`, so a config change correctly invalidates the cache automatically — `--overwrite` is only useful when you want to force a fresh build for some reason other than config diff.

### Verify before training starts

After `docker run` returns, verify:
- `docker ps` shows the container running
- `docker logs drawtoon-train | grep "Resolving HF token"` shows resolution succeeded (not blank or Access Denied)
- `nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader` jumps from 0 MiB → tens of GB when the first model load fires
- Cache build shard rate: should average ~3–6 shards/min once warmed; if a shard rate of ~1/min persists past minute 5, something is wrong with cross-region S3.

### Background S3 sync sidecar (added 2026-05-21)

`train.sh` now starts a background `aws s3 sync` daemon **right before `torchrun`** that mirrors `$save_root` → `s3://$S3_BUCKET/models/<job>/` every `SYNC_INTERVAL_SEC` seconds (default 30). It stops in the EXIT trap and again right after `torchrun` returns (so the post-`torchrun` `aws s3 sync` block does the final flush without racing the daemon).

**Why this exists:** the 2026-05-20 B300 run ended at step 8416/8518. `torchrun` exited but the wrapper never reached the post-`torchrun` `aws s3 sync` (no "Syncing artifacts" echo, S3 mtimes never advanced past my manual 10:48 push). The 30s sidecar means anything ai-toolkit writes to disk lands in S3 within 30s no matter how/when the wrapper or the box dies. Max data loss = `SYNC_INTERVAL_SEC` seconds.

To disable for debugging set `SYNC_INTERVAL_SEC=0`. The daemon log lives at `$LOG_DIR/sync_daemon.log` on the box (also synced because it's inside `$save_root`'s sibling, not inside `$save_root` — view it via SSM if you need to diagnose sync failures).

### `save_every` must align with end-of-epoch (added 2026-05-21)

ai-toolkit's `BaseSDTrainProcess` **does not auto-save at end-of-epoch** — only on `save_every` multiples. So `save_every: 3000` with an 8518-step epoch saves at 3000 and 6000, then the next save fires at 9000 which is past epoch end → final 2,518 steps (29%) of training are written nowhere. Pair `save_every` with the actual epoch length so a save always lands near the end:

- `save_every: 1000` for ~8.5k step epochs → last save at 8000, only 518 steps lost
- `save_every: 500` for tighter recovery (with `max_step_saves_to_keep: 4` to bound disk use)

This pairs with the sidecar above: `save_every` controls how often a checkpoint exists; the sidecar controls how fast it reaches S3.

### `AUTO_SHUTDOWN_ON_EXIT` — set to 0 for the next run

Default is `1`, which calls `shutdown -h now` from the EXIT trap on ANY exit (clean or signal). On the 2026-05-20 run this killed the box ~17 minutes before the capacity block boundary, taking `dmesg` / `docker logs` / NVMe with it. Set `AUTO_SHUTDOWN_ON_EXIT=0` in the SSM launch params for the next run so the box stays alive long enough to grep for the root cause; manually terminate when you're done. The capacity block charges for the full window either way.

## Fork Differences

- `lora-klein/` and `lora-klein/ai-toolkit/` are vendored in this repo.
- `run_modal.py` adds the local ai-toolkit fork into the Modal image.
- Manifest training consumes local cache paths generated from Drawtoon captions.
- Do not reintroduce regional-control/RAG generation hooks into the FLUX.2 model or pipeline.
