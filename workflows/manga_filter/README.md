# Drawtoon Manga Page Filter

This workflow filters the canonical single-page dataset:

```text
s3://drawtoon/datasets/pages/single/<manga_name>/<page_side>.jpg
```

It runs one Step Functions Distributed Map over the S3 prefix. Each worker calls Claude Haiku through Bedrock Converse, writes a per-page classification record, and copies accepted manga content to:

```text
s3://drawtoon/datasets/pages/filtered/<manga_name>/<page_side>.jpg
```

Operational outputs are written under:

```text
s3://drawtoon/datasets/pages/filtered/_status/
s3://drawtoon/datasets/pages/filtered/_jobs/
s3://drawtoon/datasets/_stepfunctions_audit/filter-manga-pages/
```

## Deploy

```bash
cd /Users/guidotrevisan/Desktop/drawtoon/workflows/manga_filter
sam build
sam deploy \
  --stack-name drawtoon-manga-filter \
  --region us-east-1 \
  --profile lineart2-s3 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides DatasetBucketName=drawtoon
```

## Start

`start.py` loads `/Users/guidotrevisan/Desktop/drawtoon/.env` automatically. You can omit `--profile` when using the AWS credentials from that file.

```bash
python start.py \
  --stack-name drawtoon-manga-filter \
  filter-pages
```

Use a small smoke run by pointing `--input-prefix` at a test prefix, or by temporarily copying a few images into a test prefix.

## Manhwa mode

Manhwa/webtoon pages can be much taller than manga pages, and adjacent pages can split one continuous panel. Use `--mode manhwa` to classify adjacent page pairs instead of independent pages:

```bash
python start.py \
  --stack-name drawtoon-manga-filter \
  --profile lineart2-s3 \
  --max-items-per-batch 2 \
  filter-pages \
  --mode manhwa \
  --include-relative-path-regex '_manwa/' \
  --max-output-tokens 220
```

Manhwa mode uses internal pair status files under:

```text
s3://drawtoon/datasets/pages/filtered/_status/<series>_manwa/<chapter>/_pairs/
```

Successful manhwa runs delete their `_jobs/<run_id>` and `_status/<series>` internals after finalization by default. Use `--manhwa-keep-internal-artifacts` only when debugging a run.

The usable chapter output is only one manifest plus strip images in the chapter folder:

```text
s3://drawtoon/datasets/pages/filtered/<series>_manwa/<chapter>/manifest.json
s3://drawtoon/datasets/pages/filtered/<series>_manwa/<chapter>/strip_0000_pages_0001-0003_len3.jpg
```

Each strip corresponds to one computed story component. Single story pages become one-page strips; connected chain pages are stitched vertically into one strip image. The chapter manifest includes the full `strips` list and each strip has `page_slices` with exact `x_start`, `x_end`, `y_start`, and `y_end` coordinates for every original page, so future text-removal jobs can reconstruct individual pages from the strip images. Failed pages such as title, credits, blank, or UI always break chains.
The summary includes `chain_length_distribution` and `strip_length_distribution`, so the run reports exactly how many 1-page, 2-page, 3-page, etc. chains/strips were found.

For each adjacent pair, the model returns only:

```text
page_1: Pass | Fail
page_2: Pass | Fail
is_chain: true | false
```

`Pass` means real story/comic page content. `Fail` means title, credits, cover, blank, UI, ads, or other non-story content. `is_chain` can only be true when both pages pass and the bottom/top boundary is the same split panel/artwork.

The pair classifier does not send full raw tall pages. It builds a compact diagnostic JPEG per adjacent pair with:

- full-page thumbnails for both pages
- the bottom crop of the left page
- the top crop of the right page
- red guide lines marking the exact bottom edge of the left page and top edge of the right page

Defaults are `896px` diagnostic width, `640px` full-overview height, `1536px` maximum boundary crop source height, and JPEG quality `82`.
Use `--manhwa-write-pair-artifacts` only for debug review sheets.
Strip JPEGs use quality `92` by default. Very tall strips that exceed JPEG dimension limits are written as PNG instead. Use `--no-manhwa-strips` to skip strip image/manifest output. Manhwa mode does not copy individual filtered pages by default; use `--manhwa-write-filtered-pages` if you also want those page files.

## Raw manwa mode

Use `--manwa` only for the raw manwa/webtoon path:

```bash
python start.py \
  --stack-name drawtoon-manga-filter \
  --profile lineart2-s3 \
  --max-items-per-batch 4 \
  filter-pages \
  --manwa \
  --include-relative-path-regex '_manwa/'
```

Without `--manwa`, the workflow behaves as before. With `--manwa`, the flow is:

1. Gemini classifies each single raw page with only `{"is_story_page": true|false}` and no thinking config.
2. Accepted pages are treated as one virtual chapter strip.
3. The finalizer cuts that virtual strip at horizontal uniform gutter bands.
4. The resulting segments are written as compact 2x2 sheet JPEGs directly in:

```text
s3://drawtoon/datasets/pages/filtered/<series>_manwa/<chapter>/
```

Each chapter folder contains only sheet images plus `manifest.json`. The manifest records pages, segments, sheet placements, and source slices so original page regions can be reconstructed later for text filtering.
