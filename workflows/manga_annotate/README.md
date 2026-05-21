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

## Files

```
workflows/manga_annotate/
├── README.md
├── requirements.txt
├── start.py             # local CLI: list S3 → write manifest → modal run
└── modal_magi.py        # Modal app: H200 class with @modal.enter snapshotting
```
