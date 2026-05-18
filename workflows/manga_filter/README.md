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

Manhwa mode writes pair status files under:

```text
s3://drawtoon/datasets/pages/filtered/_status/<series>_manwa/<chapter>/_pairs/
```

After the Distributed Map finishes, the finalizer writes one chapter manifest next to the filtered chapter:

```text
s3://drawtoon/datasets/pages/filtered/<series>_manwa/<chapter>/manifest.json
```

Each chapter manifest contains page labels, adjacent pair edges, and computed chains. A chain means page `i` and `i+1` were classified as continuing the same split panel/artwork; connected edges become multi-page chains. Non-story labels such as title, credits, blank, or UI always break chains.
The summary includes `chain_length_distribution`, so the run reports exactly how many 1-page, 2-page, 3-page, etc. chains were found.

Page labels:

```text
story
title_or_chapter
cover_or_illustration
credits_or_text
blank_or_low_content
screenshot_or_ui
other_non_story
uncertain
```

For each adjacent pair, the model returns only `continues_same_panel`, `chain_break`, `confidence`, and a short visual reason. A pair becomes a chain edge only when both pages are `story`, `continues_same_panel` is true, and confidence is at or above `0.75`.

The pair classifier does not send full raw tall pages. It builds a single diagnostic JPEG per adjacent pair with:

- full-page thumbnails for both pages
- the bottom crop of the left page
- the top crop of the right page
- red guide lines marking the exact bottom edge of the left page and top edge of the right page

Defaults are `896px` diagnostic width, `640px` full-overview height, `1536px` maximum boundary crop source height, JPEG quality `82`, and chain confidence `0.75`.
