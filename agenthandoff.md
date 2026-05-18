# Agent Handoff — Drawtoon Annotation + Text Removal

Last updated: 2026-05-18. Session was building Magi v3 annotation and Qwen-Image-Edit
text removal for 6 manga franchises on Modal + AWS.

## TL;DR

1. **Magi v3 annotation is DONE** for all 6 target chapters (33,271 pages).
   Outputs at `s3://drawtoon/datasets/annotations/magi_v3/<chapter>/<page>.jsonl`.
2. **Text removal is PARTIAL**: 28,770/33,271 done via fal.ai. **4,501 vinland-saga
   pages still need text removal.** User stopped the fal run mid-vinland to
   evaluate a cheaper Modal-based alternative.
3. **A Modal Qwen-Image-Edit pipeline is built and deployed** at
   `workflows/remove_text_modal/`. Quality artifacts are saved under `artifacts/`.
   **The user has NOT yet picked vanilla vs Lightning vs 8-step for production.**
   That decision is the next major step.

## Target chapters

| Chapter | Pages | Magi v3 annotation | Text removal (fal) |
|---|---:|---|---|
| jujutsu-kaisen | 4,576 | ✅ | ✅ |
| monster | 3,497 | ✅ | ✅ |
| my-hero-academia | 6,716 | ✅ | ✅ |
| the-fragrant-flower-blooms-with-dignity | 3,463 | ✅ | ✅ |
| vagabond | 6,550 | ✅ | ✅ |
| vinland-saga | 8,469 | ✅ | **3,968 done, 4,501 remaining** |
| **Total** | **33,271** | **33,271 ✅** | **28,770 / 33,271** |

## Workflows

### `workflows/manga_annotate/` — Magi v3 annotation (COMPLETE)

- **Model**: `ragavsachdeva/magiv3` (Magi v3) on H100s via Modal
- **Pipeline architecture**: `@app.cls + @modal.enter(snap=True)` memory-snapshot cold start, `max_containers=40`
- **Files**: `modal_magi.py`, `start.py`, `README.md`
- **Deploy**: `modal deploy modal_magi.py`
- **Modal app**: `drawtoon-manga-annotate` (deployed in workspace `reinforcenow`)
- **Throughput**: ~50 pages/sec cluster after fixes; ~$30 total for 33k pages
- **Output schema**: matches existing `magi_v3_page_annotation` (`panels`, `characters`,
  `texts`, `tails`, `character_cluster_labels`, `text_character_associations`,
  `text_tail_associations`). DO NOT change — `manga_caption` consumes it.

### `workflows/remove_text/` — fal.ai Qwen-Image-Edit-2511 (PARTIAL)

- **Endpoint**: `fal-ai/qwen-image-edit-2511`, ~$0.031/page (charged per output MP)
- **Architecture**: Step Functions Distributed Map → Lambda → fal HTTP
- **SAM stack**: `drawtoon-remove-text` (deployed in account 274213480586, us-east-1)
- **Status**: PAUSED. SF execution `six-franchises-c80-20260517T155234Z` ABORTED.
- **Output**: `s3://drawtoon/datasets/pages/text_removed/qwen2511_master_prompt_v1/<chapter>/<page>.png`
- **AWS secrets**: `drawtoon-fal-key`, `drawtoon-modal-auth` (created)
- **AWS profile that has full perms**: `default` (= `guido-admin`). `lineart2-s3` is S3-read-only.

### `workflows/remove_text_modal/` — Modal Qwen-Image-Edit (BUILT, NOT in production)

- **Model**: `Qwen/Qwen-Image-Edit-2511` + optional Lightning LoRA
  `lightx2v/Qwen-Image-Edit-2511-Lightning`
- **Pipeline**: `QwenImageEditPlusPipeline` (the "Plus" variant from diffusers 0.36+)
- **Architecture**: H100 only, `max_containers=40`. **No memory snapshot** —
  bf16 model is ~57 GB, too big for Modal's snapshot
- **Files**: `modal_qwen.py`, `start.py`, `prompts/master_prompt.md`
- **Pinned versions** (load-bearing):
  - `diffusers==0.36.0` (Plus pipeline)
  - `transformers==4.57.0` (Qwen2.5-VL)
  - `peft==0.17.0` (diffusers 0.36's LoRA loader)
  - `torch==2.5.1`
  - `torchao==0.12.0` (last release supporting torch 2.5.x)
- **Modal volume**: `qwen-image-edit-hf-cache` (separate from `magi-hf-cache`)
- **Output prefix (proposed)**: `s3://drawtoon/datasets/pages/text_removed/qwen2511_modal_lightning_v1/`
  or `qwen2511_modal_vanilla_v1/` — keep distinct from the fal output prefix

## Pending production decision

The 4,501 vinland-saga pages need text removal. Options:

| Option | Per-image | Total cost | Wall clock | Quality |
|---|---|---|---|---|
| Resume fal | $0.031 | $139 | 3-12 hr (fal queue) | known good |
| Modal vanilla 40-step | $0.043 | $193 | ~73 min on 40 H100s | best, gold standard |
| Modal Lightning 4-step | $0.0025 | **$11** | **~5 min** on 40 H100s | **user flagged as over-editing on detailed pages** |
| Modal Lightning 8-step | $0.0047 | $21 | ~10 min on 40 H100s | likely better than 4-step; not yet judged by user |

**The user's last open question is whether Lightning 8-step preserves enough detail
on the vinland-saga problem page (`vinland-saga-chapter-0__006_6.jpg`).** See
`artifacts/remove_text_modal_compare/lightning_8step/` vs `vanilla/`.

## Quality experiments — artifacts to inspect

### `artifacts/remove_text_modal_compare/`

6 representative pages (one per chapter, including the user-flagged
`vinland-saga-chapter-0__006_6.jpg`) run through these variants:

```
artifacts/remove_text_modal_compare/
├── original/                       # 6 source jpgs
├── vanilla/                        # 40-step gold standard (39 s/img)
├── lightning/                      # 4-step Lightning, original master_prompt (2.29 s/img)
├── lightning_4step_minimal/        # 4-step + "Erase only the text and sfx..." prompt (2.26 s/img)
├── lightning_8step/                # 8-step Lightning, original master_prompt (4.26 s/img)
└── summary.json
```

### `artifacts/remove_text_modal_tweaks/`

The vinland problem page (only) run through 7 variants for prompt + step + cfg
ablation:

```
artifacts/remove_text_modal_tweaks/
├── original/                       # vinland page only
├── lightning_4step/
├── lightning_4step_cfg15/           # true_cfg=1.5 (enables 2nd forward pass)
├── lightning_4step_strongneg/       # strong negative prompt
├── lightning_4step_strict/          # stricter "preserve" positive prompt
├── lightning_8step/
├── vanilla_20step/                  # may be missing — OOM'd in compare_tweaks
├── vanilla_40step/
└── summary.json
```

## Modal best practices (apply rigorously)

These were established empirically — every one of them has burned us once.

1. **`@modal.enter(snap=True)` + `enable_memory_snapshot=True`** for fast cold start —
   ONLY when the model fits in a snapshot. Magi v3 (~1.5 GB) yes; Qwen-Image (~57 GB) no.
2. **`@modal.enter(snap=False)` warmup pass** with DUMMY inputs at the same batch
   size and resolution as production. Without this, the first real inference call
   pays ~30s of CUDA kernel JIT. We caught this with `magi_v3` after seeing
   per-shard inference times balloon for warm-but-uncompiled containers.
3. **NEVER `@modal.concurrent(max_inputs=N)` on GPU functions.** It serializes
   through the GPU; max_inputs=2 made magi 5× slower. This is the single
   biggest landmine.
4. **ALWAYS** wrap inference in `torch.inference_mode()`.
5. **ALWAYS** `torch.cuda.synchronize()` before timing measurements.
6. **`hf_transfer==0.1.8`** + `HF_HUB_ENABLE_HF_TRANSFER=1` env var for fast HF downloads.
7. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** reduces fragmentation —
   doesn't prevent all OOMs but helps.
8. **Volume `magi-hf-cache`** is shared across drawtoon and lineart2 — DO NOT
   delete or rename. Same for `qwen-image-edit-hf-cache`.
9. **AWS secret**: `lineart2-aws-s3` (shared). Modal workspace: `reinforcenow`.
10. **Modal CLI** lives at `/Library/Frameworks/Python.framework/Versions/3.12/bin/modal`
    on this machine. Works from interactive shell; watch out for subprocess
    PATH inheritance from older Python versions.
11. **`return_exceptions=True`** on `.map()` so one container's OOM doesn't kill
    the rest of the experiment.
12. **Per-shard `_failed/<run_id>/<sample_id>.json`** writes for hard-to-reproduce
    failures — let you trace which exact pages broke after a long run.

## Failed vanilla optimization experiments — DO NOT repeat without good reason

All four configs were measured to be SLOWER than the bf16 baseline. Root cause:
vanilla CFG=4 at 1024×1024 needs `attention_slicing` for memory, and slicing
breaks the kernel fusion that compile/fp8 depend on for speedup.

| Attempt | Outcome | Why |
|---|---|---|
| `torch.compile(mode="max-autotune-no-cudagraphs")` on whole transformer | `InternalTorchDynamoError` | Qwen pos_embed: `'int' object has no attribute 'pos_freqs'` |
| `torch.compile(mode="reduce-overhead")` on transformer_blocks | "CUDAGraphs that has been overwritten" | residual connections clobber cudagraph tensors |
| `torch.compile(mode="default")` on transformer_blocks + attention_slicing | 54.93 s/img (1.4× slower than 39 s baseline) | slicing breaks compile fusion |
| `torchao==0.7.0` fp8 | breaks env — silently downgrades transformers below Qwen2.5-VL | use 0.12.0 |
| `torchao==0.12.0` fp8 dyn-act + per-row + slicing | 51.86 s/img (1.3× slower) | dyn-act overhead exceeds savings without compile fusion |
| Stacked compile + fp8 + slicing | 229.75 s/img (5.9× slower) | compounded overheads + recompile per step |

### The untested option that should actually work

Use the pre-calibrated fp8 model file:
`lightx2v/Qwen-Image-Edit-2511-Lightning::qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors`

This is NOT runtime quantization — it's a separately-trained fp8 base model.
Load via `from_single_file()`. Should:
- Cut transformer VRAM to ~22 GB
- Eliminate need for `attention_slicing`
- Allow `torch.compile` to actually fuse kernels
- Quality is calibrated (per lightx2v docs, mitigates the fp8 conversion grid artifacts)

Estimated outcome if it works: ~15-20 s/img at vanilla quality.

## Costs cheat sheet

- Modal H100: $0.001097/sec = $3.95/hr
- Modal H200: $4.95/hr (141 GB VRAM, ~50% more bandwidth — would let us drop slicing)
- Modal B200: more $$ but 180 GB VRAM, biggest combined win likely
- fal-ai/qwen-image-edit-2511: ~$0.031/page (charges per output MP)
- Magi v3 33k pages: ~$30 total
- Text removal 28,770 pages via fal: ~$890 sunk

## Concrete next-step commands

### Resume vinland-saga text removal at Lightning 4-step (cheapest)

```bash
cd workflows/remove_text_modal
python3 start.py \
  --variant lightning \
  --chapters vinland-saga \
  --pages-per-shard 8 \
  --detach
```

### Resume at vanilla 40-step (highest quality but expensive)

```bash
cd workflows/remove_text_modal
python3 start.py \
  --variant vanilla \
  --chapters vinland-saga \
  --pages-per-shard 4 \
  --detach
```

(`pages_per_shard=4` for vanilla because per-image inference is 40 s, so
4 pages per shard ≈ 3 min per shard — keeps Modal container scaledown happy.)

### 2026-05-18 reoptimization follow-up

Implemented `lightning_8step` as a production variant:

- `workflows/remove_text_modal/modal_qwen.py`: added `QwenLightning8Step`
  and production routing for `annotate_manifest_local`.
- `workflows/remove_text_modal/start.py`: added `--variant lightning_8step`
  and default output prefix
  `datasets/pages/text_removed/qwen2511_modal_lightning_8step_v1`.
- Smoke on `vinland-saga-chapter-0__006_6.jpg`: 4.267 s/image, output at
  `artifacts/remove_text_modal_smoke/lightning_8step__vinland-saga-chapter-0__006_6.png`.

Tested the pre-calibrated fp8 model from
`lightx2v/Qwen-Image-Edit-2511-Lightning`:

- `fp8_single`: loaded
  `qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors` via
  `QwenImageTransformer2DModel.from_single_file()`, but Diffusers expanded it
  to bf16 in this loader path (`transformer_footprint` reported 20.43B bf16
  params, ~38.05 GiB). Vinland smoke page: 37.631 s, effectively vanilla speed.
- `fp8_runtime_wo`: torchao `Float8WeightOnlyConfig` applied successfully, but
  Vinland smoke page took 59.179 s, slower than vanilla.

Conclusion: keep `vanilla`, `lightning`, and `lightning_8step` as production
variants. fp8 paths are experiment-only smoke/benchmark variants and should not
be used for the remaining 4,501 Vinland pages.

## Things to NOT do

- Don't re-run the failed vanilla/fp8 optimization experiments without a
  fundamentally different memory strategy; both pre-calibrated fp8 single-file
  and torchao runtime fp8 weight-only were tested and rejected on 2026-05-18.
- Don't bump diffusers below 0.36 (Plus pipeline class doesn't exist).
- Don't bump transformers below 4.57 (Qwen2.5-VL `to_dict()` bug).
- Don't bump peft below 0.17 (diffusers 0.36 LoRA loader requires it).
- Don't bump torch above 2.5.x without also bumping torchao (compat matrix).
- Don't put `@modal.concurrent` on the `QwenVanilla` / `QwenLightning` / `MagiAnnotator` classes.
- Don't change the `magi_v3_page_annotation` output schema — manga_caption reads it.
- Don't delete or rename the Modal volumes `magi-hf-cache`, `qwen-image-edit-hf-cache`.

## File map

```
drawtoon/
├── agenthandoff.md             # THIS FILE
├── workflows/
│   ├── manga_annotate/         # Magi v3 annotation (COMPLETE, deployed)
│   │   ├── modal_magi.py       # @app.cls MagiAnnotator on H100s
│   │   ├── start.py            # CLI
│   │   └── README.md
│   ├── remove_text/            # fal-based text removal (PARTIAL, paused)
│   │   ├── src/handlers.py     # Lambda handlers
│   │   ├── statemachines/
│   │   ├── template.yaml       # SAM stack
│   │   ├── start.py
│   │   └── prompts/master_prompt.md
│   ├── remove_text_modal/      # Modal Qwen text removal (BUILT; production variants wired)
│   │   ├── modal_qwen.py       # QwenVanilla + QwenLightning + QwenLightning8Step + experiment helpers
│   │   ├── start.py            # production CLI
│   │   ├── README.md
│   │   └── prompts/master_prompt.md
│   ├── manga_caption/          # captioning (consumes magi_v3 annotations) — not touched
│   └── manga_filter/           # page filtering — not touched
└── artifacts/
    ├── remove_text_modal_compare/    # 6 pages × {original, vanilla, lightning, lightning_4step_minimal, lightning_8step}
    └── remove_text_modal_tweaks/     # 7 variants × vinland problem page
```
