#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def paged_url(seed_url: str, page_index: int) -> str:
    if page_index <= 1:
        return seed_url
    parsed = urlparse(seed_url)
    path = parsed.path
    if not path.endswith("/"):
        path += "/"
    path = f"{path}page/{page_index}/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def templated_url(template: str, page_index: int) -> str:
    return template.format(page=page_index, page_index=page_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RFCO listing-page rows for distributed catalog discovery.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-page", type=int, default=250)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Optional seed as publisher|seed_slug|seed_url. May be repeated. Defaults to Batman and Marvel seeds.",
    )
    parser.add_argument(
        "--seed-template",
        action="append",
        default=[],
        help="Optional seed as publisher|seed_slug|seed_url|listing_url_template, where the template uses {page}.",
    )
    args = parser.parse_args()

    if args.seed or args.seed_template:
        seeds = []
        for value in args.seed:
            parts = value.split("|", 2)
            if len(parts) != 3:
                raise ValueError("--seed must be publisher|seed_slug|seed_url")
            seeds.append({"publisher": parts[0], "seed_slug": parts[1], "seed_url": parts[2], "listing_url_template": ""})
        for value in args.seed_template:
            parts = value.split("|", 3)
            if len(parts) != 4:
                raise ValueError("--seed-template must be publisher|seed_slug|seed_url|listing_url_template")
            seeds.append({"publisher": parts[0], "seed_slug": parts[1], "seed_url": parts[2], "listing_url_template": parts[3]})
    else:
        seeds = [
            {
                "publisher": "dc",
                "seed_slug": "batman",
                "seed_url": "https://readfreecomicsonline.com/category/batman/",
                "listing_url_template": "",
            },
            {
                "publisher": "marvel",
                "seed_slug": "marvel",
                "seed_url": "https://readfreecomicsonline.com/read-free-marvel-comics-online/",
                "listing_url_template": "",
            },
        ]
    rows = []
    for seed in seeds:
        for page_index in range(max(1, args.start_page), max(1, args.max_page) + 1):
            rows.append(
                {
                    "task_type": "discover_listing_page",
                    "publisher": seed["publisher"],
                    "seed_slug": seed["seed_slug"],
                    "seed_url": seed["seed_url"],
                    "page_index": page_index,
                    "listing_url": templated_url(seed["listing_url_template"], page_index)
                    if seed.get("listing_url_template")
                    else paged_url(seed["seed_url"], page_index),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "start_page": args.start_page, "max_page": args.max_page}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
