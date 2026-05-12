# Workflows

Workflow-specific AWS orchestration code lives here.

- `manga_filter/` classifies images under `s3://drawtoon/datasets/pages/single/` with Claude Haiku and copies accepted manga pages to `s3://drawtoon/datasets/pages/filtered/`.
- `manga_caption/` captions filtered pages into `s3://drawtoon/captions/<caption_run>/<chapter>/` with Claude Haiku, using sequential chapter memory.
