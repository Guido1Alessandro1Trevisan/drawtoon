# Separate Manhwa Site Investigation

Generated: 2026-05-17.

This report covers non-WEBTOON-public ingestion routes and WEBTOON app-only
limitations. It assumes the user has rights to the content, but every adapter
must still fail closed unless the acquisition path is authorized for bulk
ingestion and downstream use.

## Current WEBTOON Public Downloader

The existing `artifacts/download_webtoon_sources.py` handles only
web-visible official `webtoons.com/en` viewer pages:

- list pages expose viewer URLs with `title_no` and `episode_no`;
- viewer pages expose ordered image URLs under `#_imageList img._images[data-url]`;
- app-only, Fast Pass, paid, region-locked, or removed content is not available
  through this public-web path.

## Tapas

Relevant titles:

- Solo Leveling
- A Returner's Magic Should Be Special
- Second Life Ranker
- SSS-Class Revival Hunter
- Overgeared
- The Archmage Returns After 4000 Years
- Lout of Count's Family

Findings:

- Public/free or already-unlocked comic episodes can expose lazy image
  `data-src` values under signed `https://us-a.tapas.io/pc/...` URLs.
- Signed image URLs are short-lived, so a downloader should fetch and store
  images immediately, while keeping durable provenance and hashes rather than
  signed URLs.
- Locked/WUF episodes do not expose image lists without authorized access.
- Series info pages expose initial episode rows; further metadata is available
  from a JSON endpoint shaped like
  `/series/{series_id}/episodes?page=N&sort=OLDEST&since={ms}&initLoad=0`.
- Public/no-login image access observed by the agent was limited to the first
  few episodes for the listed titles; full-series ingestion requires a formal
  rights-holder export or an explicitly authorized session/license.

Recommended adapter:

- Discover series id and episode metadata from the info page and paginated
  episode endpoint.
- Classify each episode as public/free, authenticated-unlocked, WUF/paid
  locked, or unavailable.
- Refuse locked episodes unless a contract/export or authorized session
  explicitly covers bulk retrieval and downstream use.
- Download image `data-src` values with browser-like headers and the episode
  referer; store checksums, dimensions, episode id, source URL, and acquisition
  method.
- Use proxies only as stable egress/rate-isolation routes. Do not rotate to
  evade limits, WUF timers, login limits, CAPTCHA, paywalls, or geography.

Risks:

- Tapas terms and virtual-currency access do not by themselves grant dataset or
  training rights.
- WUF/rental access can expire.
- Signed URLs expire quickly.
- HTML/API contracts may drift.

Implemented public/free adapter:

```text
artifacts/separata_manwa/adapters/tapas_downloader.py
```

The adapter downloads only image URLs exposed by official Tapas episode HTML for
episodes marked public/free by the public episode metadata. Episodes without
public content images are written to the status manifest as
`locked_or_unavailable`.

Useful references:

- Tapas Terms of Service: https://help.tapas.io/hc/en-us/articles/115005545248-Terms-of-Service
- Tapas robots.txt: https://tapas.io/robots.txt

## Tappytoon

Relevant titles:

- Solo Leveling
- A Returner's Magic Should Be Special
- Lout of Count's Family

Findings:

- Public HTML exposes metadata, thumbnails, free/paid flags, and partial
  chapter records, but not full page-image URLs.
- Reader pages are routable, but image manifests are dynamically fetched.
- API calls without authorization return `401`.
- Public/free chapters can work with an anonymous bearer/session; locked
  chapters require a legitimately entitled session.
- Manifest endpoint observed by the agent:
  `/content-delivery/contents/manifest` with chapter id, variant, locale, and
  signed-cookie support.
- Manifest responses include ordered files plus CloudFront signed cookies.
- Direct CDN image requests return `403` without those signed cookies.
- Chapter pagination uses a `Link` response header with an opaque cursor.

Recommended adapter:

- Maintain a session manager for a public anonymous session or authorized
  logged-in session stored in secrets.
- Parse catalog metadata and follow chapter API cursor pagination.
- Call an admission/entitlement check per chapter and fetch manifests only when
  access is admitted.
- Preserve CloudFront cookies in a cookie jar and refresh manifest after cookie
  expiry or one `403` retry.
- Download files sorted by manifest order with low concurrency and validate
  content type, content length, and checksums.
- Do not store bearer tokens or signed cookies in logs or repo files.

Risks:

- Dataset/training use needs explicit rights beyond consumer access.
- Account automation may violate terms unless whitelisted/authorized.
- Signed CDN cookies and API shapes can change.
- Rentals/WUF may expire and regional restrictions must be respected.

Useful reference:

- Tappytoon copyright terms: https://www.tappytoon.com/en/terms/copyright

## Lezhin

Relevant title:

- Lout of Count's Family

Findings:

- The official Lezhin page for Lout of Count's Family can expose public/free
  reader metadata during a limited free window.
- Paid/coin episodes remain entitlement-only.

Recommended adapter:

- Build only after confirming the free window is still active and that
  acquisition is authorized.
- Restrict the adapter to episodes marked free by Lezhin's official metadata.
- Stop on paid/coin/login/403/401/429 responses and record blocked rows rather
  than retrying around entitlement gates.

Implemented adapter:

```text
artifacts/separata_manwa/adapters/lezhin_lout_downloader.py
```

Observed on 2026-05-17: the official catalog lists many free rows, but anonymous
reader HTML only exposed public cut metadata for Prologue and Episode 1. Later
routes returned `LOGIN_REQUIRED` without reader images. The completed run wrote
93/93 exposed Lezhin pages to S3 and marked 170 rows as locked/login-required.

## KakaoPage / Kakao Webtoon

Relevant titles:

- Second Life Ranker
- Overgeared
- Korean-original pages related to licensed English editions

Findings:

- Public pages are catalog/consumer surfaces, not ingestion channels.
- Public pages expose limited metadata such as title, creators, format, genre,
  counters, ratings, schedules, thumbnails, and tabs.
- Episode/page assets are not exposed as a public export surface.
- Login, paid access, age, and regional constraints are normal parts of the
  service.
- The appropriate route is partner/licensor delivery rather than public-site
  extraction.

Recommended adapter:

- Fail closed unless there is a licensor delivery manifest and signed license
  reference.
- Request official export packages through Kakao/Kakao Entertainment, content
  provider, or the relevant licensor: original images/strips, episode order,
  metadata manifest, checksums, language/territory scope, and permitted uses.
- Ingest from licensor-controlled S3, secure file transfer, signed URLs, or an
  agreed B2B API, not from consumer viewer pages or app/session caches.
- Store provenance: licensor, contract id, source delivery id, territory,
  language, episode range, permitted uses, expiry, checksum, and takedown path.

Risks:

- Public thumbnails or consumer access are not enterprise ingestion licenses.
- Rights differ by title, territory, language, format, episode range, publisher,
  and downstream use.

Useful references:

- Second Life Ranker KakaoPage: https://page.kakao.com/content/52697368
- Second Life Ranker Kakao Webtoon: https://webtoon.kakao.com/content/%EB%91%90-%EB%B2%88-%EC%82%AC%EB%8A%94-%EB%9E%AD%EC%BB%A4/2335
- Overgeared Kakao Webtoon: https://webtoon.kakao.com/content/%ED%85%9C%EB%B9%A8/2379
- Overgeared KakaoPage: https://page.kakao.com/content/67010029?tab_type=about
- Kakao Webtoon terms: https://newsroom.kakaoent.com/terms-of-use/
- KakaoPage terms: https://page.kakao.com/policy/terms
- Kakao partnership page: https://with.kakao.com/webtoon

## Naver / WEBTOON App-Only / Removed Titles

Findings:

- Public WEBTOON scraping only covers episodes rendered as web viewer pages.
- App-only, Fast Pass, Title Purchase, removed, or region-locked episodes are
  not present as normal public viewer links.
- Public viewer pages expose ordered images under `div#_imageList` with lazy
  `img._images[data-url]`.
- App-only/mobile downloads are app-centered and temporary/personal; they are
  not bulk ingestion exports.
- Removed titles, such as Wind Breaker after the reported July 17, 2025
  removal, should be treated as no-public-source unless a licensor provides
  assets.

Recommended adapter:

- Split WEBTOON into `public-web-ingest` and `licensed-delivery-ingest`.
- Use public web only for web-visible viewer pages and only with authorization.
- For app-only, paid, removed, or region-locked episodes, require licensor
  delivery: S3/FTP/API/export bundle, manifest, checksums, territory/language
  scope, and permitted uses.
- Do not reverse engineer app APIs, bypass DRM, use app caches, or automate
  consumer entitlements for bulk extraction.

Useful references:

- WEBTOON terms: https://www.webtoons.com/en/terms
- Coins help: https://webtoon.zendesk.com/hc/en-us/articles/360051593511-How-do-I-purchase-Coins
- Title Purchase help: https://webtoon.zendesk.com/hc/en-us/articles/35121925979668-What-is-Title-Purchase
- Regional access help: https://webtoon.zendesk.com/hc/en-us/articles/25771121101332-Why-can-t-I-access-WEBTOON-in-my-country
- Wind Breaker removal notice: https://m.webtoons.com/app/notice/detail?noticeNo=3460
- WEBTOON contact/licensing: https://www.webtoons.com/en/contact
- WEBTOON robots.txt: https://www.webtoons.com/robots.txt
- Naver robots.txt: https://comic.naver.com/robots.txt

## Cross-Platform Proxy Policy

Use Decodo or other proxies only for:

- stable egress routing;
- per-platform workload isolation;
- rate accounting;
- allowlisted IP paths when the licensor requires them.

Do not use proxies for:

- paywall, WUF, login, CAPTCHA, region, or app-only bypass;
- multiplying account limits;
- hiding unauthorized collection;
- retrying through new identities after access denial.

Every platform adapter should stop on `401`, `403`, `429`, CAPTCHA, entitlement
failure, region denial, or manifest/license mismatch unless the authorized
integration explicitly says how to proceed.
