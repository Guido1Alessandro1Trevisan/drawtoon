# Drawtoon Remove Text — Modal

Qwen-Image-Edit-2511 manga text removal on Modal H100/H200. Same architecture as
`workflows/manga_annotate` (@app.cls + @modal.enter + HF cache volume + max
40 containers).

For true vanilla output, keep `vanilla` or `vanilla_40step_optimized`: 40
denoise steps, `true_cfg_scale=4.0`, no Lightning, no quantization. The current
production path prefers H200, enables CUDA backend flags, keeps the model on GPU
when VRAM allows, and warms the real prompt/CFG path before work starts.
Regional compile is available with `QWEN_40STEP_COMPILE_BLOCKS=1`, but it is
opt-in because the first compile warmup is expensive and needs a separate
amortized benchmark.

Production variants:

| Variant | Steps | true_cfg_scale | LoRA | Expected sec/image | $/image |
|---|---|---|---|---|---|
| `vanilla` / `vanilla_40step_optimized` | 40 | 4.0 | none | 39.055 s avg across 6 pages on H100; 38.022 s Vinland smoke on H200 full-GPU eager | ~$0.043 on H100; ~$0.063 on H200 |
| `lightning` | 4 | 1.0 | lightx2v/Qwen-Image-Edit-2511-Lightning 4-step | 2.293 s avg across 6 pages | ~$0.003 |
| `lightning_8step` | 8 | 1.0 | lightx2v/Qwen-Image-Edit-2511-Lightning 8-step | 4.267 s on Vinland smoke page | ~$0.005 |

For comparison: fal-ai/qwen-image-edit-2511 charges ~$0.031/page.

`lightning` is the throughput winner but can over-edit some pages. Use
`lightning_8step` when you want a slower middle ground before falling back to
`vanilla`.

Rejected fp8 optimization attempts:

| Variant | Result |
|---|---|
| `fp8_single` | Loads the lightx2v pre-calibrated fp8 single-file transformer, but Diffusers expands it to bf16 here; Vinland smoke page was 37.631 s, effectively vanilla speed. |
| `fp8_runtime_wo` | torchao fp8 weight-only quantization applied, but the Vinland smoke page took 59.179 s, slower than vanilla. |

Rejected or low-value 40-step ideas:

| Idea | Reason |
|---|---|
| KV cache | Not applicable to diffusion denoising; latents change each step, unlike autoregressive token generation. |
| `gpu_batch_size > 1` | Qwen Image Edit Plus treats `image=[...]` as conditioning images, not independent page batches; use Modal parallelism instead. |
| QKV projection fusion | Qwen uses a custom double-stream attention processor that reads separate projections; model-level fusion is not a reliable win. |
| attention slicing/offload | Preserves memory but slows the 40-step path; keep as emergency fallback only. |

## One-time setup

```bash
cd workflows/remove_text_modal
modal deploy modal_qwen.py
```

This deploys the production classes plus smoke/benchmark helpers. Weights live
on Modal volume `qwen-image-edit-hf-cache` and download on first container
start (~2-3 min).

## Compare quality on a handful of pages (writes PNGs to artifacts/)

```bash
modal run modal_qwen.py::compare_variants --pages 6
```

Saves originals + vanilla outputs + lightning outputs under
`artifacts/remove_text_modal_compare/{original,vanilla,lightning}/` plus a
`summary.json` with per-image timings. Open the PNGs side-by-side to judge
quality.

## Benchmark batch sizes

```bash
modal run modal_qwen.py::run_benchmark --variant lightning --n-images 8
modal run modal_qwen.py::run_benchmark --variant lightning_8step --n-images 8
modal run modal_qwen.py::run_benchmark --variant vanilla   --n-images 8
QWEN_GPU=H200 QWEN_40STEP_COMPILE_BLOCKS=0 \
  modal run modal_qwen.py::run_benchmark --variant vanilla_40step_optimized --n-images 8
```

Sweeps batch sizes [1, 2, 4] for the chosen variant on one GPU. For vanilla
production, keep `gpu_batch_size=1`; the sweep is only a diagnostic.

## Smoke

```bash
modal run modal_qwen.py::smoke_test --variant lightning
modal run modal_qwen.py::smoke_test --variant lightning_8step
```

## Bulk annotate

```bash
python3 start.py \
  --variant vanilla \
  --chapters vinland-saga \
  --output-prefix datasets/pages/text_removed/qwen2511_modal_vanilla_v1 \
  --skip-existing-prefix datasets/pages/text_removed/qwen2511_master_prompt_v1 \
  --gpu-batch-size 1 \
  --pages-per-shard 4 \
  --detach
```

For the remaining Vinland work, this selects only pages missing from the
canonical `qwen2511_master_prompt_v1` prefix while writing Modal vanilla outputs
to a separate prefix. A dry run on 2026-05-18 selected 4,501 pages and skipped
3,968 existing pages.

By default `lightning` writes to `s3://drawtoon/datasets/pages/text_removed/qwen2511_modal_lightning_v1/`
`lightning_8step` writes to `qwen2511_modal_lightning_8step_v1/`, and
`vanilla` writes to `qwen2511_modal_vanilla_v1/` — distinct from the fal output
prefix (`qwen2511_master_prompt_v1`) so they can be diffed.

The launcher intentionally accepts only `vanilla`, `lightning`, and
`lightning_8step`; fp8 variants are available only through explicit
`modal_qwen.py` smoke/benchmark entrypoints.

## Files

```
workflows/remove_text_modal/
├── README.md
├── requirements.txt
├── start.py
├── modal_qwen.py
└── prompts/master_prompt.md
```
