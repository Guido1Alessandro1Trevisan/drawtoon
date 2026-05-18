from __future__ import annotations

import concurrent.futures
import datetime as dt
import html
import json
import mimetypes
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.environ.get("DATASET_BUCKET_NAME", "drawtoon")
DEFAULT_SOURCE_PREFIX = "datasets/pages/source/webtoon"
DEFAULT_MANIFEST_PREFIX = "datasets/pages/source/webtoon/_distributed_runs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


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


def s3_client():
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=64,
            connect_timeout=15,
            read_timeout=120,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "episode"


def extension_for_url(url: str, content_type: str | None = None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed == ".jpeg":
            return ".jpg"
        if guessed in IMAGE_EXTENSIONS:
            return guessed
    return ".jpg"


def select_series(names: list[str]) -> list[Series]:
    if not names:
        return SERIES
    wanted = {name.strip() for name in names if name.strip()}
    selected = [series for series in SERIES if series.source_slug in wanted or series.name in wanted]
    missing = sorted(wanted - {series.source_slug for series in selected} - {series.name for series in selected})
    if missing:
        raise ValueError(f"unknown series: {', '.join(missing)}")
    return selected


def load_proxy_secret(secret_name: str) -> dict[str, Any]:
    if not secret_name:
        return {}
    raw = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)["SecretString"]
    data = json.loads(raw)
    host = str(data.get("host") or data.get("DECODO_PROXY_HOST") or "").strip()
    ports_value = data.get("ports") or data.get("DECODO_PROXY_PORTS") or []
    if isinstance(ports_value, str):
        ports = [part.strip() for part in ports_value.split(",") if part.strip()]
    else:
        ports = [str(part).strip() for part in ports_value if str(part).strip()]
    user = str(data.get("user") or data.get("username") or data.get("DECODO_PROXY_USER") or "").strip()
    password = str(data.get("password") or data.get("pass") or data.get("DECODO_PROXY_PASS") or "")
    if not (host and ports and user and password):
        raise ValueError("proxy secret must contain host, ports, user, and password")
    safe_user = quote(user, safe="")
    safe_password = quote(password, safe="")
    return {"urls": [f"http://{safe_user}:{safe_password}@{host}:{port}" for port in ports]}


def proxy_url(proxy_config: dict[str, Any]) -> str | None:
    urls = proxy_config.get("urls") or []
    if not urls:
        return None
    return str(random.choice(urls))


def request_bytes(
    url: str,
    *,
    referer: str | None,
    timeout: float,
    network_mode: str,
    proxy_config: dict[str, Any],
    retries: int = 4,
) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            if referer:
                headers["Referer"] = referer
            request = Request(url, headers=headers)
            if network_mode == "proxy":
                proxy = proxy_url(proxy_config)
                if not proxy:
                    raise RuntimeError("proxy mode selected but no proxy URL is configured")
                opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
                response = opener.open(request, timeout=timeout)
            else:
                response = urlopen(request, timeout=timeout)
            with response:
                return response.read(), {key.lower(): value for key, value in response.headers.items()}
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(10.0, 0.5 * attempt))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def request_text(url: str, *, referer: str | None, timeout: float, network_mode: str, proxy_config: dict[str, Any]) -> str:
    body, headers = request_bytes(
        url,
        referer=referer,
        timeout=timeout,
        network_mode=network_mode,
        proxy_config=proxy_config,
        retries=4,
    )
    content_type = headers.get("content-type", "")
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, re.I)
    if match:
        encoding = match.group(1).strip()
    return body.decode(encoding, errors="replace")


def parse_attrs(tag: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in re.finditer(r"""([a-zA-Z0-9_:\-]+)\s*=\s*(['"])(.*?)\2""", tag, re.S):
        attrs[match.group(1).lower()] = html.unescape(match.group(3))
    return attrs


def parse_episode_links(series: Series, page_html: str, base_url: str) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in re.finditer(r"""<a\b[^>]*href\s*=\s*(['"])(.*?)\1[^>]*>(.*?)</a>""", page_html, re.I | re.S):
        href = html.unescape(match.group(2))
        if "/viewer?title_no=" not in href:
            continue
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
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
        label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(3))).strip()
        path_parts = [part for part in Path(parsed.path).parts if part not in {"/", ""}]
        path_slug = slugify(path_parts[-2] if len(path_parts) >= 2 else label)
        episodes.append(
            {
                "series": {
                    "name": series.name,
                    "source_slug": series.source_slug,
                    "title_no": series.title_no,
                    "list_url": series.list_url,
                },
                "episode_no": episode_no,
                "url": url,
                "slug": f"episode-{episode_no:06d}-{path_slug}",
                "label": label[:160],
            }
        )
    return episodes


def parse_image_urls(page_html: str, episode_url: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for tag_match in re.finditer(r"<img\b[^>]*>", page_html, re.I | re.S):
        attrs = parse_attrs(tag_match.group(0))
        raw = attrs.get("data-url") or attrs.get("data-original") or attrs.get("src")
        if not raw:
            continue
        url = urljoin(episode_url, raw)
        if "webtoon-phinf.pstatic.net" not in urlparse(url).netloc:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def discover_series(
    series: Series,
    *,
    max_list_pages: int,
    max_episodes: int | None,
    network_mode: str,
    proxy_config: dict[str, Any],
) -> list[dict[str, Any]]:
    episodes_by_no: dict[int, dict[str, Any]] = {}
    page = 1
    while page <= max_list_pages:
        separator = "&" if "?" in series.list_url else "?"
        url = f"{series.list_url}{separator}page={page}"
        body = request_text(url, referer=series.list_url, timeout=45.0, network_mode=network_mode, proxy_config=proxy_config)
        page_episodes = parse_episode_links(series, body, url)
        if not page_episodes:
            break
        before_count = len(episodes_by_no)
        for episode in page_episodes:
            episodes_by_no.setdefault(int(episode["episode_no"]), episode)
        if len(episodes_by_no) == before_count:
            break
        if max_episodes and len(episodes_by_no) >= max_episodes:
            break
        page += 1
    episodes = [episodes_by_no[key] for key in sorted(episodes_by_no)]
    if max_episodes:
        episodes = episodes[:max_episodes]
    return episodes


def choose_network_mode(proxy_mode: str, proxy_secret_name: str) -> tuple[str, dict[str, Any]]:
    proxy_mode = (proxy_mode or "auto").strip().lower()
    if proxy_mode not in {"auto", "always", "never"}:
        raise ValueError("proxy_mode must be auto, always, or never")
    if proxy_mode == "never":
        return "direct", {}
    proxy_config = load_proxy_secret(proxy_secret_name) if proxy_secret_name else {}
    if proxy_mode == "always":
        if not proxy_config:
            raise ValueError("proxy_mode=always requires proxy_secret_name")
        return "proxy", proxy_config
    try:
        request_text(SERIES[0].list_url, referer=None, timeout=12.0, network_mode="direct", proxy_config={})
        return "direct", {}
    except Exception:
        if not proxy_config:
            raise
        request_text(SERIES[0].list_url, referer=None, timeout=20.0, network_mode="proxy", proxy_config=proxy_config)
        return "proxy", proxy_config


def put_jsonl(bucket: str, key: str, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    s3_client().put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/x-jsonlines")


def prepare_manifest(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    bucket = str(event.get("bucket") or DEFAULT_BUCKET)
    output_prefix = str(event.get("output_prefix") or DEFAULT_SOURCE_PREFIX).strip("/")
    run_id = str(event.get("run_id") or dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    manifest_prefix = str(event.get("manifest_prefix") or DEFAULT_MANIFEST_PREFIX).strip("/")
    max_list_pages = max(1, int(event.get("max_list_pages") or 200))
    max_episodes_value = int(event.get("max_episodes_per_series") or 0)
    max_episodes = max_episodes_value if max_episodes_value > 0 else None
    network_mode, proxy_config = choose_network_mode(str(event.get("proxy_mode") or "auto"), str(event.get("proxy_secret_name") or ""))
    selected = select_series(list(event.get("series") or []))
    print(
        f"prepare: run_id={run_id} network_mode={network_mode} series_count={len(selected)} "
        f"max_concurrency={event.get('max_concurrency')}",
        flush=True,
    )

    episodes: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(17, len(selected)))) as pool:
        future_map = {
            pool.submit(
                discover_series,
                series,
                max_list_pages=max_list_pages,
                max_episodes=max_episodes,
                network_mode=network_mode,
                proxy_config=proxy_config,
            ): series
            for series in selected
        }
        for future in concurrent.futures.as_completed(future_map):
            series = future_map[future]
            series_episodes = future.result()
            print(
                f"prepare: discovered series={series.source_slug} episodes={len(series_episodes)}",
                flush=True,
            )
            episodes.extend(series_episodes)

    manifest_key = f"{manifest_prefix}/{run_id}/episodes.jsonl"
    config_key = f"{manifest_prefix}/{run_id}/worker_config.json"
    audit_prefix = f"{manifest_prefix}/{run_id}/audit"
    worker_config = {
        "bucket": bucket,
        "output_prefix": output_prefix,
        "network_mode": network_mode,
        "proxy_secret_name": str(event.get("proxy_secret_name") or ""),
        "worker_image_concurrency": max(1, int(event.get("worker_image_concurrency") or 8)),
        "max_images_per_episode": max(0, int(event.get("max_images_per_episode") or 0)),
        "overwrite": bool(event.get("overwrite", False)),
    }
    client = s3_client()
    put_jsonl(bucket, manifest_key, episodes)
    client.put_object(
        Bucket=bucket,
        Key=config_key,
        Body=json.dumps(worker_config, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return {
        "run_id": run_id,
        "source": {"bucket": bucket, "manifest_key": manifest_key},
        "worker_config": {"bucket": bucket, "key": config_key},
        "audit": {"bucket": bucket, "prefix": audit_prefix},
        "batch": {"max_concurrency": max(1, int(event.get("max_concurrency") or 250))},
        "failure": {"tolerated_failure_count": max(0, int(event.get("tolerated_failure_count") or 100))},
        "discovery": {
            "network_mode": network_mode,
            "series_count": len(selected),
            "episode_count": len(episodes),
            "manifest_key": manifest_key,
            "config_key": config_key,
        },
    }


def s3_key_for_image(output_prefix: str, episode: dict[str, Any], image_url: str, index: int, content_type: str | None = None) -> str:
    series = episode["series"]["source_slug"]
    return f"{output_prefix.strip('/')}/{series}/{episode['slug']}/page-{index:04d}{extension_for_url(image_url, content_type)}"


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_one_image(
    *,
    client,
    bucket: str,
    output_prefix: str,
    episode: dict[str, Any],
    image_url: str,
    index: int,
    network_mode: str,
    proxy_config: dict[str, Any],
    overwrite: bool,
) -> str:
    provisional_key = s3_key_for_image(output_prefix, episode, image_url, index)
    if not overwrite and object_exists(client, bucket, provisional_key):
        return "skipped"
    body, headers = request_bytes(
        image_url,
        referer=str(episode["url"]),
        timeout=90.0,
        network_mode=network_mode,
        proxy_config=proxy_config,
        retries=5,
    )
    content_type = headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
    key = s3_key_for_image(output_prefix, episode, image_url, index, content_type)
    if key != provisional_key and not overwrite and object_exists(client, bucket, key):
        return "skipped"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata={
            "source-url": image_url,
            "episode-url": str(episode["url"]),
            "episode-no": str(episode["episode_no"]),
            "series": str(episode["series"]["source_slug"]),
        },
    )
    return "written"


def download_episode(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    episode = dict(event["episode"])
    config_ref = event["config_ref"]
    client = s3_client()
    config_obj = client.get_object(Bucket=config_ref["bucket"], Key=config_ref["key"])
    config = json.loads(config_obj["Body"].read().decode("utf-8"))
    bucket = str(config["bucket"])
    output_prefix = str(config["output_prefix"]).strip("/")
    network_mode = str(config.get("network_mode") or "direct")
    proxy_config = load_proxy_secret(str(config.get("proxy_secret_name") or "")) if network_mode == "proxy" else {}
    html_text = request_text(
        str(episode["url"]),
        referer=str(episode["series"]["list_url"]),
        timeout=45.0,
        network_mode=network_mode,
        proxy_config=proxy_config,
    )
    image_urls = parse_image_urls(html_text, str(episode["url"]))
    max_images = int(config.get("max_images_per_episode") or 0)
    if max_images > 0:
        image_urls = image_urls[:max_images]
    if not image_urls:
        raise RuntimeError(f"no images found for {episode['series']['source_slug']} episode={episode['episode_no']}")

    written = 0
    skipped = 0
    failures = 0
    failure_samples: list[str] = []
    worker_count = max(1, int(config.get("worker_image_concurrency") or 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                download_one_image,
                client=client,
                bucket=bucket,
                output_prefix=output_prefix,
                episode=episode,
                image_url=image_url,
                index=index,
                network_mode=network_mode,
                proxy_config=proxy_config,
                overwrite=bool(config.get("overwrite", False)),
            ): image_url
            for index, image_url in enumerate(image_urls, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                if len(failure_samples) < 5:
                    failure_samples.append(str(exc))
                continue
            if result == "written":
                written += 1
            else:
                skipped += 1

    if failures:
        raise RuntimeError(
            json.dumps(
                {
                    "message": "episode completed with image failures",
                    "series": episode["series"]["source_slug"],
                    "episode_no": episode["episode_no"],
                    "images": len(image_urls),
                    "written": written,
                    "skipped": skipped,
                    "failures": failures,
                    "failure_samples": failure_samples,
                },
                ensure_ascii=False,
            )
        )
    return {
        "series": episode["series"]["source_slug"],
        "episode_no": episode["episode_no"],
        "images": len(image_urls),
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "network_mode": network_mode,
    }
