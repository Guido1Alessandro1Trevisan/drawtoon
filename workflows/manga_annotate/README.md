# Drawtoon Manga Annotate

Distributed Magi v3 annotation for Drawtoon manga pages. 40 H200s in parallel,
~13 min wall-clock for 33k pages.

## Layout

```
s3://drawtoon/datasets/pages/filtered/<chapter>/<page_id>.jpg
   →   s3://drawtoon/datasets/annotations/magi_v3/<chapter>/<page_id>.jsonl
```

Existing annotations are kept by default. Pass `--overwrite` to re-annotate.
Per-page failures land at `_failed/<run_id>/<sample_id>.json` under the
output prefix so they're easy to find after the run.

## Output schema

Each `.jsonl` file is one JSON object compatible with the existing
`magi_v3_page_annotation` schema the manga_caption workflow reads:

```json
{
  "schema_name": "magi_v3_page_annotation",
  "model_repo": "ragavsachdeva/magiv3",
  "sample_id": "jujutsu-kaisen__0001",
  "source": {"bucket": "drawtoon", "key": "...", "s3_uri": "..."},
  "image_size": {"width": 711, "height": 1098},
  "tasks": ["detections"],
  "detections": {
    "panels": [{"bbox": [...], "score": 0.97, "panel_id": "..."}],
    "characters": [{"bbox": [...], "score": ..., "source_character_id": "0"}],
    "texts": [{"bbox": [...], "score": ...}],
    "tails": [...],
    "character_cluster_labels": ["0", "1", ...],
    "text_character_associations": [...],
    "text_tail_associations": [...]
  },
  "summary": {...},
  "run": {"run_id": "...", "git_sha": "...", "annotated_at": "..."}
}
```

`source_character_id` mirrors `character_cluster_labels[i]` so the caption
pipeline groups recurring characters correctly.

Every page is annotated in two mandatory stages:

1. Magi v3 detects panels, text, tails, and raw character boxes.
2. Gemini receives the clean page image plus the raw Magi character box coordinates,
   then returns a verified character label, drop/keep decision, and short visible
   evidence reason for each box.

Annotations always include a top-level `verification` object and `detections.characters[].source_character_id` is rewritten to the verified visual character label. Boxes marked `NoCharacter` are dropped. If Gemini fails on a page, that page is recorded under `datasets/annotations/magi_v3/_failed/<run_id>/<sample_id>.json` and not written to the output prefix.

## One-time setup

```bash
cd workflows/manga_annotate
modal deploy modal_magi.py
modal run modal_magi.py::smoke_test       # annotates one jujutsu-kaisen page
```

The Magi v3 weights live on the shared Modal volume `magi-hf-cache` —
already populated. AWS access uses Modal secret `lineart2-aws-s3`.

## Run

A single chapter (always runs Magi + Gemini):

```bash
python3 start.py --chapters jujutsu-kaisen --gpu-batch-size 8
```

The six target chapters:

```bash
python3 start.py \
  --chapters jujutsu-kaisen monster my-hero-academia \
             the-fragrant-flower-blooms-with-dignity vagabond vinland-saga \
  --gpu-batch-size 8 \
  --detach
```

Dry run — list pages and write the manifest without launching Modal:

```bash
python3 start.py --chapters jujutsu-kaisen --dry-run
```

Re-annotate (overwrite existing `.jsonl`):

```bash
python3 start.py --chapters jujutsu-kaisen --overwrite
```

## Tuning

- **`--gpu-batch-size 8`** — empirical sweet-spot on H200. Magi v3 hard-resizes
  every input to 768×768 (no width-dependent OOM), but the model is decoder-heavy
  so larger batches give diminishing returns. Run a sweep before pushing higher.
- **`--pages-per-shard 16`** — two GPU batches per shard. Lower for finer-grained
  retries, higher to amortize Modal cold-start.
- **`max_containers=40`** is set inside `modal_magi.py`. Raise both this and your
  Modal H200 quota together.

## Manwa mode — vertical webtoons / manhwa

Pass `--manwa` to annotate raw long-strip manhwa pages. Internally:

1. `start.py` lists raw source pages under `datasets/pages/single/<chapter>/`
   (override with `--manwa-source-prefix`).
2. For each chapter the launcher runs **Gemini-driven cut detection** locally
   (see `manwa_sheets.build_chapter_sheets`): consecutive pages are stitched
   into 2-page chunks, sent to `gemini-3-flash-preview` with `thinking_level=HIGH`,
   and the model returns whether each chunk is a title page plus the y-pixel
   coordinates of every safe inter-panel gutter on it. Coordinates come back
   normalized to 0-1000 per Gemini's image-understanding spec — we rescale to
   chunk pixel space and aggregate into global scroll y's.
3. The chapter scroll is then virtually cut at those y's into a list of
   **sheets**. Each sheet records its `slices[]` — the source-page bands that
   contribute to it, with both source-page y-ranges and sheet-local y-ranges.
4. The launcher emits a chapter-level manifest where each row carries one
   chapter's sheet list (no sheet JPEGs are written to S3; sheets exist only
   in memory during annotation and as slice plans in the manifest).
5. The Modal worker (`MagiAnnotator.annotate_manwa_chapter`) downloads each
   slice's source page, stitches the sheet back together in memory, runs the
   same Magi v3 + Gemini verifier path that manga pages use, and writes one
   annotation JSONL per sheet to
   `datasets/annotations/magi_v3/<chapter>/<chapter>__sheet-NNNN.jsonl`.

The output is **schema-compatible with manga page annotations** — every
field except `source` is identical. The `source` block for a sheet is:

```json
"source": {
  "type": "manwa_sheet",
  "sheet_id": "sheet_0003",
  "page_key": null,
  "slices": [
    {"source_page_key": "datasets/pages/single/<chapter>/<chapter>__page-0007.jpg",
     "source_y_start": 540, "source_y_end": 1280,
     "sheet_y_start": 0,    "sheet_y_end":   740},
    {"source_page_key": "datasets/pages/single/<chapter>/<chapter>__page-0008.jpg",
     "source_y_start": 0,   "source_y_end": 1100,
     "sheet_y_start": 740,  "sheet_y_end":  1840}
  ],
  "width": 800,
  "height": 1840
}
```

Bboxes in `detections.{panels,characters,texts,tails}[]` are in the sheet's
own coordinate space, so the trainer treats a sheet exactly like a page: it
calls `Image.open(asset)` and crops by bbox. The ai-toolkit's
`ManifestDataset` (in `lora-klein/ai-toolkit/toolkit/manifest_dataset.py`)
detects manifest control entries carrying `slices[]` instead of `image`
and stitches the sheet on demand into a per-plan cached JPEG. **Off-target
panel character sampling works without changes**: each sheet contains
multiple panels with multiple characters, and `_select_character_reference`
in `lora-klein/training/run_modal.py` already iterates panels-within-asset
for the off-target lookup.

### Run examples

```bash
# All chapters under datasets/pages/single/, manwa mode
python3 start.py --manwa --gpu-batch-size 8

# A single chapter
python3 start.py --manwa --chapters the-mafia-nanny_manwa

# Dry run (write chapter manifest, skip modal)
python3 start.py --manwa --chapters the-mafia-nanny_manwa --dry-run

# Re-annotate (clobber existing sheet jsonls)
python3 start.py --manwa --chapters the-mafia-nanny_manwa --overwrite
```

### Tuning (manwa-specific)

- `--manwa-pages-per-chunk 2` — number of consecutive pages stitched into
  one Gemini cut-detection request. 2 is the sweet spot: the page seam is
  visible inside the chunk so Gemini can place a cut at it.
- `--manwa-max-parallel 10` — max concurrent Gemini cut-detection requests
  per chapter. Increase for larger chapters; bound by your Gemini quota.

### Joint training (manwa + manga)

Because the per-sheet JSONLs share the manga `magi_v3_page_annotation` schema
and live under the same `datasets/annotations/magi_v3/<chapter>/` prefix,
training a LoRA on a mixture of manga and manwa is just a manifest of both
asset types. No new code paths in the trainer.

## Files

```
workflows/manga_annotate/
├── README.md
├── requirements.txt
├── start.py             # local CLI: list S3 → write manifest → modal run
├── modal_magi.py        # Modal app: H200 class with @modal.enter snapshotting
│                        # plus annotate_manwa_chapter + annotate_manwa_manifest_local
└── manwa_sheets.py      # Gemini-driven sheet builder (manwa mode only)
```
