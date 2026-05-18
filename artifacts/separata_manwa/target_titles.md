# separata_manwa Target Titles

The user wants authorized titles stored in stable S3 page formats.

Official/public platform adapters use:

```text
s3://drawtoon/datasets/pages/source/<platform>/<series_slug>/<episode_or_chapter_slug>/page-####.<ext>
```

User-provided authorized reader-site pages use:

```text
s3://drawtoon/datasets/pages/single/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

Filtered story-page copies use:

```text
s3://drawtoon/datasets/pages/single_relevant/<series_slug>_manwa/<chapter_slug>/page-####.<ext>
```

Use stable lowercase slugs. Do not force non-WEBTOON titles through the WEBTOON adapter.

## Already Covered By WEBTOON Public-Web Adapter

These titles are already present in the WEBTOON manifest/discovery list:

| Requested name | Manifest name | Platform prefix | Series slug |
| --- | --- | --- | --- |
| Tower of God | Tower of God | `webtoon` | `tower-of-god` |
| Bastard | Bastard | `webtoon` | `bastard` |
| Omniscient Reader's Viewpoint | Omniscient Reader | `webtoon` | `omniscient-reader` |
| God of Highschool | The God of High School | `webtoon` | `the-god-of-high-school` |
| Sweet Home | Sweet Home | `webtoon` | `sweet-home` |
| Horizon | The Horizon | `webtoon` | `the-horizon` |
| The Breaker | The Breaker: Eternal Force | `webtoon` | `the-breaker-eternal-force` |
| Noblesse | Noblesse | `webtoon` | `noblesse` |
| Teenage Mercenary / Mercenary Enrollment | Teenage Mercenary | `webtoon` | `teenage-mercenary` |
| Who Made Me a Princess | Who Made Me a Princess | `webtoon` | `who-made-me-a-princess` |
| Girls of the Wild | Girls of the Wild's | `webtoon` | `girls-of-the-wilds` |
| The Boxer | The Boxer | `webtoon` | `the-boxer` |
| Lookism | Lookism | `webtoon` | `lookism` |
| Tomb Raider King | Tomb Raider King | `webtoon` | `tomb-raider-king` |
| Eleceed | Eleceed | `webtoon` | `eleceed` |
| Sound of Magic | The Sound of Magic: Annarasumanara | `webtoon` | `the-sound-of-magic-annarasumanara` |
| The Gamer | The Gamer | `webtoon` | `the-gamer` |

## Missing Source Adapter Queue

These requested titles are not covered by the current WEBTOON public-web adapter and need source-specific investigation/adapters:

| Requested title | Working slug | Status | Required next step |
| --- | --- | --- | --- |
| Solo Leveling | `solo-leveling-comic` | Official Tapas source identified; public/free Tapas adapter implemented | Run `adapters/tapas_downloader.py` for public/free episodes; locked chapters require authorized Tapas/Tappytoon/WebNovel/Yen/Ize export. |
| A Returner's Magic Should Be Special | `a-returners-magic-should-be-special` | Official Tapas source identified; Tappytoon/Yen catalog-only alternatives found | Run Tapas adapter for public/free episodes; locked chapters require authorized Tapas/Tappytoon/Yen export. |
| Ranker Who Lives a Second Time | `second-life-ranker` | Official Tapas source identified | Official English title is Second Life Ranker; run Tapas adapter for public/free episodes or consume authorized export for locked chapters. |
| SSS-Class Suicide Hunter | `sss-class-revival-hunter` | Official Tapas source identified; Ize/Yen catalog/preview alternative found | Official English title is SSS-Class Revival Hunter; run Tapas adapter for public/free episodes or consume authorized Tapas/Ize export for locked chapters. |
| Wind Breaker | `wind-breaker` | Blocked for public web | Official WEBTOON English series was removed on July 17, 2025; use WEBTOON/licensor export only. |
| Overgeared | `overgeared` | Official Tapas source identified; Ize/Yen catalog/preview alternative found | Run Tapas adapter for public/free episodes or consume authorized Tapas/Ize export for locked chapters. |
| The Great Mage Returns After 4,000 Years | `the-archmage-returns-after-4000-years` | Official Tapas source identified | Official English title is The Archmage Returns After 4000 Years; run Tapas adapter for public/free episodes or consume authorized export for locked chapters. |
| Trash of the Count's Family | `lout-of-counts-family` | Official Tapas and Lezhin public-reader adapters implemented | Official English title is Lout of Count's Family; public/anonymous pages are downloaded from Tapas and Lezhin, while login/locked chapters require authorized export/session path. |

## Official Source Findings

| Title | Official source notes | Adapter path |
| --- | --- | --- |
| Solo Leveling | Tapas official web source: `https://tapas.io/series/solo-leveling-comic/info`. Also official on Tappytoon/WebNovel, but Tapas is the preferred web-visible adapter source. | `s3://drawtoon/datasets/pages/source/tapas/solo-leveling-comic/<episode>/page-####.<ext>` |
| Ranker Who Lives a Second Time | Official English title is Second Life Ranker on Tapas: `https://tapas.io/series/second-life-ranker/info`. | `s3://drawtoon/datasets/pages/source/tapas/second-life-ranker/<episode>/page-####.<ext>` |
| SSS-Class Suicide Hunter | Official English title is SSS-Class Revival Hunter on Tapas: `https://tapas.io/series/sss-class-revival-hunter/info`. | `s3://drawtoon/datasets/pages/source/tapas/sss-class-revival-hunter/<episode>/page-####.<ext>` |
| Overgeared | Tapas official web source: `https://tapas.io/series/overgeared/info`. | `s3://drawtoon/datasets/pages/source/tapas/overgeared/<episode>/page-####.<ext>` |
| A Returner's Magic Should Be Special | Tapas official web source: `https://tapas.io/series/a-returners-magic-should-be-special/info`. Tappytoon and Yen Press are official catalog/preview or export candidates, not public page-image sources. | `s3://drawtoon/datasets/pages/source/tapas/a-returners-magic-should-be-special/<episode>/page-####.<ext>` |
| The Great Mage Returns After 4,000 Years | Official English title is The Archmage Returns After 4000 Years on Tapas: `https://tapas.io/series/the-archmage-returns-after-4000-years/info`. | `s3://drawtoon/datasets/pages/source/tapas/the-archmage-returns-after-4000-years/<episode>/page-####.<ext>` |
| Trash of the Count's Family | Official English title is Lout of Count's Family on Tapas: `https://tapas.io/series/lout-of-counts-family/info`. Lezhin also has an official public-free candidate for some episodes. | `s3://drawtoon/datasets/pages/source/tapas/lout-of-counts-family/<episode>/page-####.<ext>` |
| Wind Breaker | Official English WEBTOON pages are no longer public after the July 17, 2025 removal notice. | `s3://drawtoon/datasets/pages/source/licensor/wind-breaker/<episode>/page-####.<ext>` |

Tapas public/free episodes can be downloaded by a Tapas-specific adapter. Locked/WUF/Ink/login chapters require a licensed export or explicitly permitted entitled access; do not bypass Tapas access controls.

Current Tapas adapter:

```text
artifacts/separata_manwa/adapters/tapas_downloader.py
```

Current Lezhin adapter for Lout of Count's Family:

```text
artifacts/separata_manwa/adapters/lezhin_lout_downloader.py
```

Verified completed public/anonymous source downloads:

| Platform | Title scope | Accessible episodes | Verified S3 pages | Blocked rows |
| --- | --- | ---: | ---: | ---: |
| Tapas | 7 official Tapas missing-title mappings | 33 | 1,668 / 1,668 | 1,508 locked/export-required |
| Lezhin | Lout of Count's Family | 2 | 93 / 93 | 170 login/export-required |

Verified completed user-authorized reader-site downloads:

| Series slug | Raw pages in `single/` | Relevant pages in `single_relevant/` |
| --- | ---: | ---: |
| `solo-leveling_manwa` | 8,326 | 7,954 |
| `sss-class-suicide-hunter_manwa` | 5,407 | 4,922 |
| `second-life-ranker_manwa` | 4,060 | 4,047 |
| `a-returners-magic-should-be-special_manwa` | 7,890 | 7,718 |
| `the-great-mage-returns-after-4000-years_manwa` | 2,915 | 2,910 |
| `lout-of-counts-family_manwa` | 313 | 281 |
| **Total** | **28,911** | **27,832** |

## Adapter Output Contract

Official/public platform adapters write raw page images like:

```text
s3://drawtoon/datasets/pages/source/tapas/solo-leveling/chapter-000001/page-0001.jpg
s3://drawtoon/datasets/pages/source/tappytoon/a-returners-magic-should-be-special/chapter-000001/page-0001.jpg
s3://drawtoon/datasets/pages/source/webtoon/wind-breaker/episode-000001/page-0001.jpg
```

Exact platform prefixes must match the official source used by the adapter.

The authorized reader-site adapter writes raw images like:

```text
s3://drawtoon/datasets/pages/single/solo-leveling_manwa/chapter-000001/page-0001.jpg
s3://drawtoon/datasets/pages/single/sss-class-suicide-hunter_manwa/chapter-000001/page-0001.jpg
```
