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

Manwa / manhwa / manhua singles use the same filter — pass `--include-relative-path-regex '_manwa/'` (or `_manhwa/`, `_manhua/`) to scope a run. Recut singles look like normal manga pages, so the Haiku single-page classifier handles them without a special mode.
