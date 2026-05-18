# separata_manwa Agent Guide

## Scope

This guide applies to everything under `artifacts/separata_manwa/`.

Do not create a sibling `webtoon_manwa`, `wektoon_manga`, or Flutter workspace for this task. Keep manifests, AWS distributed code, notes, logs, and handoffs inside this folder.

## Objective

Ingest the user's authorized manhwa/webtoon page images into S3 without mixing lanes.

Official/public platform adapters keep their platform source format:

```text
s3://drawtoon/datasets/pages/source/<platform>/<series_slug>/<episode_or_chapter_slug>/page-####.<ext>
```

The user-provided authorized reader-site lane writes directly to:

```text
s3://drawtoon/datasets/pages/single/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

Filtered story-page copies, when produced, go to:

```text
s3://drawtoon/datasets/pages/single_relevant/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

Do not move or delete the raw `single/` pages during cleanup.

## Title Queue

The complete requested title list is tracked in:

```text
target_titles.md
```

Current priority is the titles not already covered by the WEBTOON public-web adapter:

- Solo Leveling
- A Returner's Magic Should Be Special
- Ranker Who Lives a Second Time
- SSS-Class Suicide Hunter
- Wind Breaker
- Overgeared
- The Great Mage Returns After 4,000 Years
- Trash of the Count's Family

## Source Rules

- Use official, authorized, or licensor-provided sources only.
- Use public web-visible pages only when they expose the content without bypassing access controls.
- Do not bypass paywalls, app-only locks, CAPTCHA, DRM, Fast Pass, signed entitlement checks, private app APIs, or account restrictions.
- If a title is locked/app-only, document the official platform and require an authorized export/session path instead of scraping around it.
- Proxies may be used for routing stability or account-approved access, not evasion.
- Do not store credentials in repo files. Use environment variables or AWS Secrets Manager.

## Distributed Download Rules

AWS work belongs under:

```text
aws_distributed/
```

Use direct AWS Lambda egress first. Fall back to Decodo proxies only if direct access fails and a proxy secret is configured.

Do not route S3 uploads through Decodo proxies. S3 writes should go direct from AWS to S3.

Keep concurrency explicit. Do not launch multiple downloaders against the same S3 prefix at the same time.

For the authorized reader-site adapter, direct HTTP is preferred first. Use a proxy pool only when direct access demonstrably fails and proxy credentials are supplied through environment variables, not repo files.

## Documentation Rules

When adding or changing a source adapter, update:

- `README.md`
- `target_titles.md`
- `source_plan.md`
- `HANDOFF.md` if a run is active or operational state changes
