# Drawtoon

Canonical dataset bucket:

```text
s3://drawtoon/datasets/pages/single/<manga_name>/<page_side>.jpg
```

Current workflow code:

- `workflows/manga_filter/` filters `datasets/pages/single/` with Claude Haiku and writes accepted pages to `datasets/pages/filtered/`.

Manwa / manhwa / manhua singles (`<series>_manwa`, `_manhwa`, `_manha`,
`_manhua`) sit under `datasets/pages/single/` and were normalised to the
FLUX 1024² training pixel budget in a one-shot stitch-by-episode +
row-RGB-uniform recut. That recut tooling has been retired; the recut
output remains in place and is consumed by every downstream workflow
(`manga_filter`, `manga_annotate`/magi v3, captioner, trainer) without a
special path.

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
