# separata_manwa Handoff

Last updated: 2026-05-18 00:15 CEST

## Scope

This handoff is only for the missing/non-WEBTOON title lane under:

```text
artifacts/separata_manwa/
```

Do not use `artifacts/webtoon_manga` for this lane.

## Completed Work

Implemented and compiled:

```text
artifacts/separata_manwa/adapters/tapas_downloader.py
artifacts/separata_manwa/adapters/lezhin_lout_downloader.py
artifacts/separata_manwa/adapters/authorized_reader_downloader.py
artifacts/separata_manwa/aws_distributed/src/reader_handlers.py
artifacts/separata_manwa/aws_distributed/scripts/deploy_reader_download.py
artifacts/separata_manwa/clean_single_pages.py
artifacts/separata_manwa/aws_distributed/scripts/deploy_reader_clean.py
```

Updated:

```text
artifacts/separata_manwa/AGENTS.md
artifacts/separata_manwa/README.md
artifacts/separata_manwa/source_plan.md
artifacts/separata_manwa/site_investigation.md
artifacts/separata_manwa/target_titles.md
```

## S3 Results

Tapas run:

```text
run_id=tapas_public_20260517T171501Z
retry_run_id=tapas_retry_20260517T172547Z
S3 prefix=s3://drawtoon/datasets/pages/source/tapas/
series_count=7
episode_rows=1541
accessible_rows=33
locked_or_export_required_rows=1508
verified_pages=1668/1668
image_failures_after_retry=0
```

Verified per Tapas series:

```text
a-returners-magic-should-be-special: 4 episodes, 217 pages
lout-of-counts-family: 4 episodes, 209 pages
overgeared: 4 episodes, 282 pages
second-life-ranker: 5 episodes, 216 pages
solo-leveling-comic: 5 episodes, 171 pages
sss-class-revival-hunter: 6 episodes, 335 pages
the-archmage-returns-after-4000-years: 5 episodes, 238 pages
```

Lezhin run:

```text
run_id=lezhin_lout_public_20260517T173250Z
S3 prefix=s3://drawtoon/datasets/pages/source/lezhin/lout-of-counts-family/
episode_rows=172
accessible_rows=2
login_or_export_required_rows=170
verified_pages=93/93
image_failures=0
```

Verified Lezhin public-reader rows:

```text
episode-0001-p1: 52 pages
episode-0002-1: 41 pages
```

Total verified non-WEBTOON public/anonymous pages written or present in S3:

```text
1761 pages
```

Authorized reader-site distributed run:

```text
run_id=reader_distributed_20260517T2200
retry_run_id=reader_distributed_partial_retry_20260517T2203
S3 raw prefix=s3://drawtoon/datasets/pages/single/
Step Functions main=SUCCEEDED, 1079/1079 chapters, 0 state failures
Step Functions retry=SUCCEEDED, 60/60 partial chapters, 0 state failures
main audit image URLs=20584, written=18980, skipped=1509, internal image failures=95
retry audit image URLs=2951, written=94, skipped=2857, image failures=0
```

Verified raw reader-site page counts:

```text
solo-leveling_manwa: 8326
sss-class-suicide-hunter_manwa: 5407
second-life-ranker_manwa: 4060
a-returners-magic-should-be-special_manwa: 7890
the-great-mage-returns-after-4000-years_manwa: 2915
lout-of-counts-family_manwa: 313
total raw reader-site pages: 28911
```

Known reader-site rows with no exposed image URLs in this method:

```text
sss-class-suicide-hunter_manwa: 1 row
the-great-mage-returns-after-4000-years_manwa: 33 side-story rows, chapters 225-257
lout-of-counts-family_manwa: 143 preview rows before chapter 143
```

Relevant-page cleanup:

```text
run_id=reader_clean_relaxed_20260517T2213
S3 relevant prefix=s3://drawtoon/datasets/pages/single_relevant/
Step Functions=SUCCEEDED, 1110/1110 chapters, 0 failures
input raw images=28911
kept relevant images=27832
dropped likely cover/side/small/landscape images=1079
copy_errors=0
```

Relevant-page counts:

```text
solo-leveling_manwa: 7954
sss-class-suicide-hunter_manwa: 4922
second-life-ranker_manwa: 4047
a-returners-magic-should-be-special_manwa: 7718
the-great-mage-returns-after-4000-years_manwa: 2910
lout-of-counts-family_manwa: 281
total relevant reader-site pages: 27832
```

## Manifests

Local:

```text
artifacts/separata_manwa/manifests/tapas_public_20260517T171501Z_episodes.jsonl
artifacts/separata_manwa/manifests/tapas_public_20260517T171501Z_status.jsonl
artifacts/separata_manwa/manifests/tapas_retry_20260517T172547Z_status.jsonl
artifacts/separata_manwa/manifests/lezhin_lout_public_20260517T173250Z_episodes.jsonl
artifacts/separata_manwa/manifests/lezhin_lout_public_20260517T173250Z_status.jsonl
```

S3:

```text
s3://drawtoon/datasets/pages/source/tapas/_manifests/tapas_public_20260517T171501Z/
s3://drawtoon/datasets/pages/source/tapas/_manifests/tapas_retry_20260517T172547Z/
s3://drawtoon/datasets/pages/source/lezhin/_manifests/lezhin_lout_public_20260517T173250Z/
s3://drawtoon/datasets/pages/single/_distributed_runs/reader_distributed_20260517T2200/
s3://drawtoon/datasets/pages/single/_distributed_runs/reader_distributed_partial_retry_20260517T2203/
s3://drawtoon/datasets/pages/single_relevant/_distributed_runs/reader_clean_relaxed_20260517T2213/
```

## Current Blockers

The remaining rows are not public anonymous page-image sources:

- Tapas WUF/Ink/login/locked chapters.
- Lezhin login-required or coin chapters.
- Tappytoon/Yen/Ize/WebNovel catalog/preview surfaces without public full image manifests.
- Wind Breaker, whose official English WEBTOON pages were removed after the July 17, 2025 notice.

Use a licensor/platform export or explicitly authorized bulk delivery path for those. Do not use mirrors, archives, app caches, private APIs, CAPTCHA bypasses, paywall bypasses, or region/login evasion.

## Useful Verification Commands

```bash
python3 -m py_compile artifacts/separata_manwa/adapters/tapas_downloader.py
python3 -m py_compile artifacts/separata_manwa/adapters/lezhin_lout_downloader.py
aws s3 ls s3://drawtoon/datasets/pages/source/tapas/ --recursive --summarize | tail -n 4
aws s3 ls s3://drawtoon/datasets/pages/source/lezhin/lout-of-counts-family/ --recursive --summarize | tail -n 4
```
