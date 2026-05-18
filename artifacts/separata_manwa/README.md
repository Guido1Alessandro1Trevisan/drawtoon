# separata_manwa Source Dossier

This folder tracks platforms that need source-specific authorized ingestion
separate from the official WEBTOON public web downloader.

## Scope

- Keep all work in this folder. Do not create a separate `webtoon_manwa`,
  `wektoon_manga`, or Flutter workspace for this task.
- WEBTOON public-web titles are handled by the WEBTOON adapter/distributed
  downloader.
- Non-WEBTOON or locked/app-only titles need source-specific adapters or
  authorized source exports.
- Target title tracking lives in `target_titles.md`.

## Canonical S3 Format

Official/public platform adapters write raw page images in this format:

```text
s3://drawtoon/datasets/pages/source/<platform>/<series_slug>/<episode_or_chapter_slug>/page-####.<ext>
```

Examples:

```text
s3://drawtoon/datasets/pages/source/webtoon/tower-of-god/episode-000001/page-0001.jpg
s3://drawtoon/datasets/pages/source/tapas/solo-leveling/chapter-000001/page-0001.jpg
```

The user-provided authorized reader-site adapter is separate and writes directly to:

```text
s3://drawtoon/datasets/pages/single/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

The relevance cleanup pass copies likely story pages to:

```text
s3://drawtoon/datasets/pages/single_relevant/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

Do not delete the raw `single/` pages when producing filtered copies.

## Current Title Coverage

Covered by the current WEBTOON public-web adapter:

- Tower of God
- Bastard
- Omniscient Reader / Omniscient Reader's Viewpoint
- The God of High School
- Sweet Home
- The Horizon
- The Breaker: Eternal Force
- Noblesse
- Teenage Mercenary / Mercenary Enrollment
- Who Made Me a Princess
- Girls of the Wild's
- The Boxer
- Lookism
- Tomb Raider King
- Eleceed
- The Sound of Magic: Annarasumanara
- The Gamer

Still needing source-specific investigation or adapter work:

- Solo Leveling: Tapas public/free adapter implemented.
- A Returner's Magic Should Be Special: Tapas public/free adapter implemented.
- Ranker Who Lives a Second Time / Second Life Ranker: Tapas public/free adapter implemented.
- SSS-Class Suicide Hunter / SSS-Class Revival Hunter: Tapas public/free adapter implemented.
- Overgeared: Tapas public/free adapter implemented.
- The Great Mage Returns After 4,000 Years / The Archmage Returns After 4000 Years: Tapas public/free adapter implemented.
- Trash of the Count's Family / Lout of Count's Family: Tapas public/free adapter implemented; Lezhin public-reader adapter implemented.
- Wind Breaker: no current public official English source; requires WEBTOON/licensor export.

## Tapas Public/Free Adapter

Current adapter:

```text
adapters/tapas_downloader.py
```

It discovers episodes from official Tapas info pages and Tapas' public series episode endpoint, downloads only image URLs exposed in official public/free episode HTML, and records WUF/Ink/login/locked episodes as `locked_or_unavailable`.

Smoke command:

```bash
python3 -u artifacts/separata_manwa/adapters/tapas_downloader.py \
  --series solo-leveling-comic \
  --max-episodes-per-series 1 \
  --max-images-per-episode 1 \
  --download \
  --upload-manifest
```

Full public/free run:

```bash
python3 -u artifacts/separata_manwa/adapters/tapas_downloader.py \
  --download \
  --upload-manifest \
  --episode-workers 8 \
  --image-workers 12
```

Current verified Tapas result:

```text
run_id=tapas_public_20260517T171501Z plus retry tapas_retry_20260517T172547Z
accessible episodes=33
verified pages in S3=1668/1668
locked/export-required rows=1508
S3 prefix=s3://drawtoon/datasets/pages/source/tapas/
```

## Lezhin Lout Public Reader Adapter

Current adapter:

```text
adapters/lezhin_lout_downloader.py
```

It checks the official Lezhin Lout of Count's Family reader routes and downloads only pages where the official HTML embeds public reader cut metadata and signed CDN parameters. Routes that return `LOGIN_REQUIRED`, coin, or missing-reader content are recorded as `locked_or_unavailable`.

Run command:

```bash
python3 -u artifacts/separata_manwa/adapters/lezhin_lout_downloader.py \
  --download \
  --upload-manifest \
  --episode-workers 6 \
  --image-workers 6
```

Current verified Lezhin result:

```text
run_id=lezhin_lout_public_20260517T173250Z
accessible episodes=2
verified pages in S3=93/93
login/export-required rows=170
S3 prefix=s3://drawtoon/datasets/pages/source/lezhin/lout-of-counts-family/
```

## Authorized Reader-Site Adapter

Current raw adapter:

```text
adapters/authorized_reader_downloader.py
```

Current distributed worker/deploy scripts:

```text
aws_distributed/src/reader_handlers.py
aws_distributed/scripts/deploy_reader_download.py
```

Current verified raw result:

```text
run_id=reader_distributed_20260517T2200
retry_run_id=reader_distributed_partial_retry_20260517T2203
S3 prefix=s3://drawtoon/datasets/pages/single/
raw reader-site pages=28911
image failures after retry=0
```

Raw page counts:

```text
solo-leveling_manwa: 8326
sss-class-suicide-hunter_manwa: 5407
second-life-ranker_manwa: 4060
a-returners-magic-should-be-special_manwa: 7890
the-great-mage-returns-after-4000-years_manwa: 2915
lout-of-counts-family_manwa: 313
```

## Relevant Page Cleanup

Current cleaner:

```text
clean_single_pages.py
aws_distributed/scripts/deploy_reader_clean.py
```

Current verified relevant-page result:

```text
run_id=reader_clean_relaxed_20260517T2213
S3 prefix=s3://drawtoon/datasets/pages/single_relevant/
input raw pages=28911
kept relevant pages=27832
dropped cover/side/small/landscape pages=1079
copy_errors=0
```

Relevant page counts:

```text
solo-leveling_manwa: 7954
sss-class-suicide-hunter_manwa: 4922
second-life-ranker_manwa: 4047
a-returners-magic-should-be-special_manwa: 7718
the-great-mage-returns-after-4000-years_manwa: 2910
lout-of-counts-family_manwa: 281
```

## Constraints

- Use only authorized access, licensed exports, or web-visible pages that the
  account has rights to access.
- Do not bypass paywalls, app-only locks, CAPTCHA, DRM, or other access
  controls.
- Proxies may be used for routing isolation and stable network behavior, but
  not to evade platform limits or access restrictions.
- Store credentials only in environment variables or secrets, never in repo
  files.

## Outputs

- `AGENTS.md`: scoped instructions for future agents working in this folder.
- `target_titles.md`: requested title list, current coverage, and missing queue.
- `site_investigation.md`: platform findings and downloader recommendations.
- `source_plan.md`: staged implementation plan once platform findings are
  confirmed.
- `aws_distributed/`: AWS Step Functions Distributed Map downloader workspace.
- `adapters/`: source-specific downloader scripts for official non-WEBTOON
  sources.
- `clean_single_pages.py`: story-page relevance cleaner for raw reader-site
  pages under `datasets/pages/single/`.
- `manifests/`: local JSONL manifests/status summaries for completed runs.
