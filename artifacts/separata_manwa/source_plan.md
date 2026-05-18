# Separate Source Plan

## Current State

- The official WEBTOON public web downloader is separate from this folder and
  handles web-visible `webtoons.com/en` viewer pages.
- Non-WEBTOON platforms need their own adapters or licensed export ingestion.
- App-only, login-only, paid, or removed episodes require authorized platform
  access or licensor/source exports.
- The complete title queue and coverage status are tracked in `target_titles.md`.
- Do not create or use a separate `webtoon_manwa` workspace; this work belongs
  under `artifacts/separata_manwa/`.
- Tapas public/free adapter work is under `adapters/tapas_downloader.py`.
- Lezhin Lout public-reader adapter work is under
  `adapters/lezhin_lout_downloader.py`.
- Completed public/anonymous downloads are verified in S3:
  - Tapas: 33 accessible rows, 1,668/1,668 pages.
  - Lezhin Lout: 2 accessible rows, 93/93 pages.

## Adapter Shape

Each platform adapter should implement the same stages:

1. Discover authorized series metadata and episode URLs.
2. Verify access for each episode before image discovery.
3. Extract page image URLs or consume an authorized export/API.
4. Download through configured proxy routing with conservative concurrency.
5. Write raw pages to `s3://drawtoon/datasets/pages/source/<platform>/<series>/...`.
6. Run the canonical import/copy step into `datasets/pages/single/<series>_manwa/`.

Canonical raw page key shape:

```text
s3://drawtoon/datasets/pages/source/<platform>/<series_slug>/<episode_or_chapter_slug>/page-####.<ext>
```

## Missing Adapter Priority

Do not push these through the WEBTOON-only adapter unless the official source is
confirmed to be `webtoons.com` public-web:

1. Run the Tapas adapter for public/free episodes where platform acquisition is authorized:
   - Solo Leveling
   - A Returner's Magic Should Be Special
   - Ranker Who Lives a Second Time / Second Life Ranker
   - SSS-Class Suicide Hunter / SSS-Class Revival Hunter
   - Overgeared
   - The Great Mage Returns After 4,000 Years / The Archmage Returns After 4000 Years
   - Trash of the Count's Family / Lout of Count's Family
2. Lezhin public-reader adapter has been built for Lout of Count's Family; it
   downloads only Prologue and Episode 1 anonymously because later routes return
   `LOGIN_REQUIRED` without public image metadata.
3. Mark Wind Breaker as export-required: the official English WEBTOON public pages are gone after the July 17, 2025 removal notice.
4. For any locked/WUF/Ink/app-only/account-only title or episode, ingest from licensor/platform export packages, not consumer-page bypasses.

## Required Controls

- Proxy preflight must pass before downloads start.
- Empty-source runs must fail by default.
- Progress logs must include discovered episodes, images written, failures,
  rate, and ETA.
- Existing S3 objects should be skipped unless `--overwrite` is explicit.
- Manifest/status output must separate `accessible`, `locked_or_unavailable`,
  and `failed` records so downstream work can see exactly what remains blocked.
