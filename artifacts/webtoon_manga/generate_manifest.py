#!/usr/bin/env python3
"""Generate a JSONL episode manifest for the webtoon_manga Distributed Map."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests


DEFAULT_OUT = "artifacts/webtoon_manga/manifest/webtoon_episodes.jsonl"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
EPISODE_LINK_RE = re.compile(r"<a\b[^>]+href=[\"']([^\"']*/viewer\?title_no=[^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def decodo_proxy_urls() -> list[str]:
    host = os.environ.get("DECODO_PROXY_HOST", "").strip()
    user = os.environ.get("DECODO_PROXY_USER", "").strip()
    password = os.environ.get("DECODO_PROXY_PASS", "")
    ports = [part.strip() for part in os.environ.get("DECODO_PROXY_PORTS", "").split(",") if part.strip()]
    if not (host and user and password and ports):
        return []
    safe_user = urllib.parse.quote(user, safe="")
    safe_password = urllib.parse.quote(password, safe="")
    return [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]


@dataclass(frozen=True)
class Series:
    name: str
    source_slug: str
    title_no: int
    list_url: str


SERIES = [
    Series("Tower of God", "tower-of-god", 95, "https://www.webtoons.com/en/fantasy/tower-of-god/list?title_no=95"),
    Series("Bastard", "bastard", 485, "https://www.webtoons.com/en/thriller/bastard/list?title_no=485"),
    Series("Omniscient Reader", "omniscient-reader", 2154, "https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"),
    Series("The God of High School", "the-god-of-high-school", 66, "https://www.webtoons.com/en/action/the-god-of-high-school/list?title_no=66"),
    Series("Sweet Home", "sweet-home", 1285, "https://www.webtoons.com/en/thriller/sweethome/list?title_no=1285"),
    Series("The Horizon", "the-horizon", 3141, "https://www.webtoons.com/en/drama/the-horizon/list?title_no=3141"),
    Series("The Breaker: Eternal Force", "the-breaker-eternal-force", 4501, "https://www.webtoons.com/en/action/the-breaker-eternal-force/list?title_no=4501"),
    Series("Noblesse", "noblesse", 87, "https://www.webtoons.com/en/action/noblesse/list?title_no=87"),
    Series("Teenage Mercenary", "teenage-mercenary", 2677, "https://www.webtoons.com/en/action/teenage-mercenary/list?title_no=2677"),
    Series("Who Made Me a Princess", "who-made-me-a-princess", 9475, "https://www.webtoons.com/en/romance/who-made-me-a-princess/list?title_no=9475"),
    Series("Girls of the Wild's", "girls-of-the-wilds", 93, "https://www.webtoons.com/en/action/girls-of-the-wilds/list?title_no=93"),
    Series("The Boxer", "the-boxer", 2027, "https://www.webtoons.com/en/sports/the-boxer/list?title_no=2027"),
    Series("Lookism", "lookism", 1049, "https://www.webtoons.com/en/drama/lookism/list?title_no=1049"),
    Series("Tomb Raider King", "tomb-raider-king", 10204, "https://www.webtoons.com/en/action/tomb-raider-king/list?title_no=10204"),
    Series("Eleceed", "eleceed", 1571, "https://www.webtoons.com/en/action/eleceed/list?title_no=1571"),
    Series("The Sound of Magic: Annarasumanara", "the-sound-of-magic-annarasumanara", 77, "https://www.webtoons.com/en/drama/the-sound-of-magic-annarasumanara/list?title_no=77"),
    Series("The Gamer", "the-gamer", 88, "https://www.webtoons.com/en/action/the-gamer/list?title_no=88"),
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "episode"


def fetch_text(url: str, *, referer: str | None = None, timeout: float = 45.0, retries: int = 4) -> str:
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(6.0, 0.5 * attempt))
    proxies = decodo_proxy_urls()
    for attempt in range(1, retries + 1):
        if not proxies:
            break
        proxy = proxies[(attempt - 1) % len(proxies)]
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                proxies={"http": proxy, "https": proxy},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(6.0, 0.5 * attempt))
    assert last_exc is not None
    raise last_exc


def text_from_anchor(anchor_html: str) -> str:
    text = TAG_RE.sub(" ", anchor_html)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def parse_episode_links(series: Series, html_text: str, base_url: str) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw_href, anchor_html in EPISODE_LINK_RE.findall(html_text):
        url = urllib.parse.urljoin(base_url, html.unescape(raw_href))
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        title_values = query.get("title_no") or []
        episode_values = query.get("episode_no") or []
        if not title_values or not episode_values:
            continue
        try:
            title_no = int(title_values[0])
            episode_no = int(episode_values[0])
        except ValueError:
            continue
        if title_no != series.title_no or episode_no in seen:
            continue
        seen.add(episode_no)
        label = text_from_anchor(anchor_html)
        parts = [part for part in Path(parsed.path).parts if part not in {"/", ""}]
        path_slug = slugify(parts[-2] if len(parts) >= 2 else label)
        episodes.append(
            {
                "series_name": series.name,
                "series_slug": series.source_slug,
                "title_no": series.title_no,
                "list_url": series.list_url,
                "episode_no": episode_no,
                "url": url,
                "slug": f"episode-{episode_no:06d}-{path_slug}",
                "label": label[:160],
            }
        )
    return episodes


def discover_series(series: Series, *, max_list_pages: int, max_episodes: int | None) -> list[dict[str, object]]:
    episodes_by_no: dict[int, dict[str, object]] = {}
    for page in range(1, max_list_pages + 1):
        separator = "&" if "?" in series.list_url else "?"
        url = f"{series.list_url}{separator}page={page}"
        html_text = fetch_text(url, referer=series.list_url)
        page_episodes = parse_episode_links(series, html_text, url)
        if not page_episodes:
            break
        before = len(episodes_by_no)
        for episode in page_episodes:
            episodes_by_no.setdefault(int(episode["episode_no"]), episode)
        new_count = len(episodes_by_no) - before
        print(f"[{series.source_slug}] page={page} new={new_count} total={len(episodes_by_no)}", flush=True)
        if new_count == 0:
            break
        if max_episodes and len(episodes_by_no) >= max_episodes:
            break
    episodes = [episodes_by_no[key] for key in sorted(episodes_by_no)]
    if max_episodes:
        episodes = episodes[:max_episodes]
    print(f"[{series.source_slug}] discovered={len(episodes)}", flush=True)
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-list-pages", type=int, default=200)
    parser.add_argument("--max-episodes-per-series", type=int)
    parser.add_argument("--series", action="append", default=[])
    args = parser.parse_args()

    wanted = {item.strip() for item in args.series if item.strip()}
    selected = [series for series in SERIES if not wanted or series.source_slug in wanted or series.name in wanted]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for series in selected:
            for episode in discover_series(
                series,
                max_list_pages=args.max_list_pages,
                max_episodes=args.max_episodes_per_series,
            ):
                handle.write(json.dumps({"episode": episode}, sort_keys=True) + "\n")
                total += 1
    print(f"manifest={out_path} episodes={total}", flush=True)
    return 0 if total else 2


if __name__ == "__main__":
    raise SystemExit(main())
