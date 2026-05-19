# Drawtoon

Canonical dataset bucket:

```text
s3://drawtoon/datasets/pages/single/<manga_name>/<page_side>.jpg
```

Current workflow code:

- `workflows/manga_filter/` filters `datasets/pages/single/` with Claude Haiku and writes accepted pages to `datasets/pages/filtered/`.

## Manwa / manhwa / manhua: how they were re-cut

Raw manhwa-family pages (`<series>_manwa`, `_manhwa`, `_manha`, `_manhua`)
arrive on S3 in wildly inconsistent shapes — most are 800×1280 but some
publishers export an entire chapter as one 5,000-to-16,000-pixel-tall jpeg.
Long-strip pages don't fit FLUX's 1024² training pixel budget and Gemini's
gutter detector becomes unreliable past ~5,000 px height.

To normalise the corpus we did a one-shot **stitch-by-episode + row-RGB-uniform
recut** instead of relying on Gemini for these:

1. Group every raw page under `datasets/pages/single/<series>/` by episode
   (subdirectory like `chapter-000001/`, or `<episode-id>__page-` filename
   prefix, or flat-series fallback).
2. Stitch all pages in an episode into one tall PIL strip (widths
   normalised to the max page width).
3. Run the same baseline row-uniform-gutter detector that lives in
   `workflows/manga_filter/src/manhwa_raw_sheets.py:_row_is_uniform_gutter`
   over the strip: per row, sample 96 columns; flag the row as gutter if
   horizontal RGB range ≤ 10 **and** luma ≥ 238 or luma ≤ 35 or
   saturation-proxy ≤ 18; require runs of ≥ 18 such rows to count as a
   gutter band; cut at each band's midpoint with a 320 px minimum between
   consecutive cuts.
4. Slice the stitched strip at every gutter midpoint. Drop any segment
   whose `width × height > 1024 × 1024 = 1,048,576` pixels (FLUX budget).
5. Write the surviving segments back to
   `s3://drawtoon/datasets/pages/single/<series>/<episode>__page-NNNN.jpg`
   in-place, then delete every original key that wasn't overwritten.

The whole job ran as a throwaway distributed Step-Functions workflow
under `artifacts/recut_stepfunctions/` (one Lambda per episode, ≤3,000
concurrency, ~3 min wall clock for 3,150 episodes). Naming, mime type,
and bucket layout match the original `single/` keys exactly so every
downstream workflow (`manga_filter`, `manga_annotate`/magi v3, captioner,
trainer) consumes the recut output without code changes.

**Resulting state of `datasets/pages/single/` for the manwa+manhua slice:**

- ~65,734 recut singles across 3,146 episodes / 34 series.
- Every surviving file fits the 1024² pixel budget; the dataset shrinks
  ~47% vs the original raw scrape (long-strip exports were not viable
  training samples).
- 4 episodes are empty due to corrupt source jpegs that PIL couldn't
  decode at recut time (37 pages, all already unreadable pre-recut).
- magi v3 annotation works on the recut output without modification —
  smoke-test on 200 pages returned 200 successful annotations
  (see `artifacts/recut_smoke_magi/` for sample layout overlays).

The implementation files (intentionally co-located in `artifacts/` since
this is a one-time normalisation, not an ongoing pipeline):

```
artifacts/recut_stepfunctions/
├── template.yaml                      # SAM stack
├── statemachines/recut_episodes.asl.json   # Distributed Map
├── src/handlers.py                    # row-RGB cutter + 1024² drop
├── start.py                           # local CLI: list → manifest → start
└── README.md                          # usage notes
```

If a downstream workflow needs the *original* long-strip pages back,
they were not preserved — re-scrape via `workflows/download_scraper/`.

## Free DC / Marvel Reading Shortlist

As of 2026-05-18, modern DC and Marvel copyrights and trademarks have not expired. Do not host or link pirated scans of modern books. Use official free pages, publisher apps, library access, or public-domain archives.

Legal/free entry points:

- Marvel official free comics: <https://www.marvel.com/comics?isFree=1>
- DC Universe Infinite free essentials: <https://www.dcuniverseinfinite.com/collections/dc-essential-reads-row>
- DC free-to-read collections generally require free DCUI registration.
- Public-domain Golden Age scans: <https://www.digitalcomicmuseum.com/> and <https://comicbookplus.com/>

Top 20 legal/free reads to check first:

| # | Title | Publisher | Why it is useful |
|---|---|---|---|
| 1 | Spectacular Spider-Man: Brand New Day (2026) #1 | Marvel | Current Spider-Man entry point. |
| 2 | Peter Parker, The Spectacular Spider-Man Facsimile Edition (2026) #1 | Marvel | Classic Spider-Man reference through an official facsimile. |
| 3 | Amazing Spider-Man/Venom: Death Spiral - Body Count (2026) #1 | Marvel | Spider-Man/Venom visual reference. |
| 4 | Iron Man (2026) #5 | Marvel | Current Iron Man armor/action reference. |
| 5 | Captain America (2025) #11 | Marvel | Current Captain America reference. |
| 6 | Wolverine (2024) #20 | Marvel | Current Wolverine reference. |
| 7 | Uncanny X-Men (2024) #28 | Marvel | Current X-Men team reference. |
| 8 | Mortal Thor (2025) #10 | Marvel | Current Thor/cosmic fantasy reference. |
| 9 | Doctor Strange (2025) #6 | Marvel | Current magic/cosmic Marvel reference. |
| 10 | Thanos: The Infinity Ending (2019) | Marvel | Thanos/cosmic Marvel reference from Marvel's free page. |
| 11 | All-Star Superman #1 | DC | Free DCUI Superman starter. |
| 12 | Superman (1939-) #1 | DC | Historic Superman issue available in DCUI's free Superman collection. |
| 13 | Action Comics (1938-) #775 | DC | Strong Superman ethics/heroism reference. |
| 14 | Batman (1940-) #404 | DC | Batman: Year One opener. |
| 15 | Batman (2010-) #608 | DC | Batman: Hush opener. |
| 16 | Batman (2011-) #1 | DC | New 52 Batman/Court of Owls starter. |
| 17 | Batman: The Long Halloween #1 | DC | Crime-noir Batman starter. |
| 18 | Batman: The Dark Knight Returns #1 | DC | Iconic older Batman visual language. |
| 19 | DC: The New Frontier #1 | DC | Broad retro DC universe reference. |
| 20 | Justice League (2011-) #1 | DC | Big-team Justice League starter. |

The larger modern canon list, such as *Infinity Gauntlet*, *Civil War*, *Weapon X*, *Hush*, *Kingdom Come*, and *Darkseid War*, is useful as a paid/library reading roadmap, but it should not be treated as free-to-host content unless the publisher or a library platform provides lawful access.

The current Marvel-only official free-page manifest is here:

```text
artifacts/marvel_official_free/top_official_free_links.jsonl
```

That file intentionally stores official issue links only. Marvel's issue pages
do not expose a public unauthenticated full-page image manifest, so full page
downloads should use a rights-holder export/API from Marvel rather than
scraping reader tokens or private reader endpoints.
