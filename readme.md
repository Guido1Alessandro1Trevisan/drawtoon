# Drawtoon

Canonical dataset bucket:

```text
s3://drawtoon/datasets/pages/single/<manga_name>/<page_side>.jpg
```

Current workflow code:

- `workflows/manga_filter/` filters `datasets/pages/single/` with Claude Haiku and writes accepted pages to `datasets/pages/filtered/`.
