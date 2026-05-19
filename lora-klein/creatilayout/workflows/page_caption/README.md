# Whole-Page Caption

Distributed Map that produces one **10-20 word page-level caption** per manga
page, plus 1-4 controlled `page_tags`, for training the LayouSyn **page-layout**
model (see `creatilayout/HANDOFF.md` Model A and `creatilayout/vocabulary.md`).

One Gemini vision call per page. Panel count is **not** re-detected by the LLM
— it is read from the MAGI v3 annotation (if present) and passed into the
prompt as known structural context.

## Input

```text
s3://drawtoon/datasets/pages/text_removed/<chapter>/<page_id>.jpg
s3://drawtoon/datasets/annotations/magi_v3/<chapter>/<page_id>.jsonl   (optional)
```

The annotation is optional. When present it provides the pre-detected
`panel_count`. When missing the LLM is told `panel_count: 0` and writes a
caption without the count prefix.

By default `require_annotations=False` so pages without a MAGI annotation are
still captioned. Pass `--require-annotations` to skip them.

## Output

```text
s3://drawtoon/page_captions/<output_run>/<chapter>/<page_id>.json
```

```jsonc
{
  "schema_version": 1,
  "caption_type": "gemini_page_caption_v1",
  "output_run": "page_v1",
  "chapter": "20th-century-boys_mangazero",
  "page_id": "000__side_01",
  "page_size": {"width_px": 1080, "height_px": 1540},
  "panel_count": 6,
  "caption": "6-panel page of a dialogue scene with one wide establishing shot",
  "word_count": 13,
  "exceeded_word_cap": false,
  "page_tags": ["conversation", "establishing"],
  "image": {"image_format": "png", "image_bytes": 312441, "image_width": 1080, "image_height": 1540},
  "model": {"id": "gemini-3-flash-preview", "provider": "gemini"},
  "usage": {...}
}
```

`page_tags` come from the controlled vocabulary in
`creatilayout/vocabulary.md` (axis 3 — 21 tags across rhythm/function/mood/
dialogue/special). The LLM is constrained via Gemini's enum schema and may
pick 1-4 tags. Empty `page_tags` is allowed when none apply with confidence.

## Deploy

```bash
cd /Users/guidotrevisan/Desktop/drawtoon/lora-klein/creatilayout/workflows/page_caption
sam build
sam deploy \
  --stack-name drawtoon-page-caption \
  --region us-east-1 \
  --profile lineart2-s3 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides DatasetBucketName=drawtoon
```

## Dry run

```bash
python start.py \
  --stack-name drawtoon-page-caption \
  --profile lineart2-s3 \
  --dry-run \
  --max-concurrency 300 \
  caption-pages \
  --output-run page_v1 \
  --include-chapter-regex '_mangazero$'
```

Add `--require-annotations` to skip pages without MAGI v3 annotations.

## Notes

- Same page-input prefix as the existing `workflows/manga_caption/` workflow
  (`datasets/pages/text_removed/`), so the two can run side-by-side and produce
  complementary panel-level + page-level captions for the same corpus.
- Hard word cap is 20 words. The worker records `exceeded_word_cap: true` if
  the model overshoots; the LayouSyn data-prep should drop or re-truncate
  those rows.
- The caption format starts with `<N>-panel page of …` so the page-layout
  model gets the panel count both in the prompt and in the `items[]` list it
  will be supervised on.
- The MAGI v3 annotation is *only* used to count panels. We deliberately do
  not pass panel rects into this prompt — that's structural input to the
  layout model, not a caption.
