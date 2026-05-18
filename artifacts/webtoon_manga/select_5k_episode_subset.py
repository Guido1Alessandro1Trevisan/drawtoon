#!/usr/bin/env python3
"""Select a balanced ~5k cleaned-image episode subset per WEBTOON series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DISTRIBUTION = ROOT / "episode_distribution_cleaned_dim_v2.csv"
DEFAULT_MANIFEST = ROOT / "manifest/webtoon_episodes.jsonl"
DEFAULT_OUTPUT_CSV = ROOT / "selected_5k_per_series_cleaned_dim_v2.csv"
DEFAULT_OUTPUT_JSONL = ROOT / "selected_5k_per_series_cleaned_dim_v2.jsonl"
DEFAULT_SUMMARY = ROOT / "selected_5k_per_series_cleaned_dim_v2_summary.json"
TARGET_SERIES = [
    "tower-of-god",
    "lookism",
    "omniscient-reader",
    "eleceed",
    "teenage-mercenary",
    "the-breaker-eternal-force",
]


def read_distribution(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            input_images = int(row["input_images"])
            kept_images = int(row["kept_images"])
            dropped_images = int(row["dropped_images"])
            rows.append(
                {
                    **row,
                    "episode_no": int(row["episode_no"]),
                    "input_images": input_images,
                    "kept_images": kept_images,
                    "dropped_images": dropped_images,
                    "dominant_width": int(row["dominant_width"]),
                    "story_start_position": int(row["story_start_position"] or 0),
                    "story_end_position": int(row["story_end_position"] or 0),
                    "dropout_rate": dropped_images / input_images if input_images else 1.0,
                }
            )
    return rows


def read_episode_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    episodes: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)["episode"]
            episodes[(episode["series_slug"], int(episode["episode_no"]))] = episode
    return episodes


def read_excluded_episode_keys(paths: list[str]) -> set[tuple[str, int]]:
    excluded: set[tuple[str, int]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                episode = payload.get("episode") or {}
                series_slug = str(episode.get("series_slug") or "").strip()
                episode_no = episode.get("episode_no")
                if series_slug and episode_no is not None:
                    excluded.add((series_slug, int(episode_no)))
    return excluded


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]


def split_bins(rows: list[dict[str, Any]], bin_count: int) -> dict[int, list[dict[str, Any]]]:
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        bin_index = min(bin_count - 1, int(index * bin_count / max(1, len(rows))))
        row["chronology_bin"] = bin_index
        bins[bin_index].append(row)
    return bins


def episode_score(row: dict[str, Any], median_kept: int) -> float:
    kept = max(1, int(row["kept_images"]))
    size_penalty = abs(kept - median_kept) / max(1, median_kept)
    drop_penalty = float(row["dropout_rate"])
    return drop_penalty * 10.0 + size_penalty


def choose_anchors(
    rows: list[dict[str, Any]],
    *,
    bin_count: int,
    max_dropout_rate: float,
) -> list[dict[str, Any]]:
    kept_values = [int(row["kept_images"]) for row in rows if int(row["kept_images"]) > 0]
    median_kept = quantile(kept_values, 0.5)
    bins = split_bins(rows, bin_count)
    anchors: list[dict[str, Any]] = []
    for bin_index in range(bin_count):
        bin_rows = bins.get(bin_index, [])
        if not bin_rows:
            continue
        candidates = [
            row
            for row in bin_rows
            if int(row["kept_images"]) > 0 and float(row["dropout_rate"]) <= max_dropout_rate
        ]
        if not candidates:
            candidates = [row for row in bin_rows if int(row["kept_images"]) > 0]
        if not candidates:
            continue
        anchors.append(min(candidates, key=lambda row: episode_score(row, median_kept)))
    return anchors


def choose_filler(
    rows: list[dict[str, Any]],
    *,
    selected_keys: set[tuple[str, int]],
    target_remaining: int,
    max_total: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (row["series_slug"], row["episode_no"]) not in selected_keys and int(row["kept_images"]) > 0
    ]
    median_kept = quantile([int(row["kept_images"]) for row in rows if int(row["kept_images"]) > 0], 0.5)
    cap = max(0, max_total)
    if target_remaining <= 0 or cap <= 0:
        return []

    # Sparse subset-sum DP. For each reachable total, keep the lower-penalty
    # episode list so ties prefer lower dropout and normal-sized episodes.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, row in enumerate(candidates):
        weight = int(row["kept_images"])
        if weight <= 0 or weight > cap:
            continue
        score = episode_score(row, median_kept)
        next_states = dict(states)
        for total, (current_score, picked) in states.items():
            new_total = total + weight
            if new_total > cap:
                continue
            new_score = current_score + score
            old = next_states.get(new_total)
            if old is None or new_score < old[0]:
                next_states[new_total] = (new_score, picked + (index,))
        states = next_states

    best_total = min(
        states,
        key=lambda total: (abs(total - target_remaining), states[total][0], abs(total - target_remaining - 1)),
    )
    return [candidates[index] for index in states[best_total][1]]


def select_series(
    rows: list[dict[str, Any]],
    *,
    target_images: int,
    bin_count: int,
    max_dropout_rate: float,
    max_over_target: int,
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: int(row["episode_no"]))
    anchors = choose_anchors(rows, bin_count=bin_count, max_dropout_rate=max_dropout_rate)
    selected: dict[tuple[str, int], dict[str, Any]] = {
        (row["series_slug"], int(row["episode_no"])): row for row in anchors
    }
    anchor_total = sum(int(row["kept_images"]) for row in selected.values())
    filler = choose_filler(
        rows,
        selected_keys=set(selected),
        target_remaining=max(0, target_images - anchor_total),
        max_total=max(0, target_images + max_over_target - anchor_total),
    )
    for row in filler:
        selected[(row["series_slug"], int(row["episode_no"]))] = row
    return sorted(selected.values(), key=lambda row: int(row["episode_no"]))


def select_series_random(
    rows: list[dict[str, Any]],
    *,
    target_images: int,
    min_images: int,
    max_images: int,
    max_dropout_rate: float,
    seed: int,
    attempts: int,
) -> list[dict[str, Any]]:
    rows = [row for row in rows if int(row["kept_images"]) > 0]
    preferred = [row for row in rows if float(row["dropout_rate"]) <= max_dropout_rate]
    candidates = preferred if preferred else rows
    if not candidates:
        return []

    best: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    rng = random.Random(seed)
    for _attempt in range(max(1, attempts)):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected: list[dict[str, Any]] = []
        total = 0
        for row in shuffled:
            kept = int(row["kept_images"])
            if total >= min_images:
                break
            if total + kept > max_images:
                remaining = [item for item in shuffled if item not in selected and total + int(item["kept_images"]) <= max_images]
                if remaining:
                    row = rng.choice(remaining)
                    kept = int(row["kept_images"])
                else:
                    continue
            selected.append(row)
            total += kept

        if total < min_images:
            remaining = [row for row in rows if row not in selected]
            rng.shuffle(remaining)
            for row in remaining:
                kept = int(row["kept_images"])
                if total + kept <= max_images:
                    selected.append(row)
                    total += kept
                if total >= min_images:
                    break

        if not selected:
            continue
        input_images = sum(int(row["input_images"]) for row in selected)
        dropped_images = sum(int(row["dropped_images"]) for row in selected)
        dropout = dropped_images / input_images if input_images else 1.0
        distance = abs(total - target_images)
        key = (distance, dropout, len(selected))
        if best_key is None or key < best_key:
            best = list(selected)
            best_key = key
    return sorted(best, key=lambda row: int(row["episode_no"]))


def summarize_selection(series_slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_images = sum(int(row["input_images"]) for row in rows)
    kept_images = sum(int(row["kept_images"]) for row in rows)
    dropped_images = sum(int(row["dropped_images"]) for row in rows)
    bins = sorted(set(int(row.get("chronology_bin", 0)) for row in rows))
    return {
        "series_slug": series_slug,
        "series_name": rows[0]["series_name"] if rows else "",
        "selected_episodes": len(rows),
        "episode_no_min": min((int(row["episode_no"]) for row in rows), default=0),
        "episode_no_max": max((int(row["episode_no"]) for row in rows), default=0),
        "selected_episode_numbers": [int(row["episode_no"]) for row in rows],
        "input_images": input_images,
        "kept_images": kept_images,
        "dropped_images": dropped_images,
        "weighted_dropout_rate": dropped_images / input_images if input_images else 1.0,
        "bins_covered": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", default=str(DEFAULT_DISTRIBUTION))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--target-images", type=int, default=5000)
    parser.add_argument("--min-images", type=int, default=4750)
    parser.add_argument("--max-images", type=int, default=5250)
    parser.add_argument("--max-over-target", type=int, default=250)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-dropout-rate", type=float, default=0.85)
    parser.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--random-attempts", type=int, default=5000)
    parser.add_argument("--exclude-jsonl", nargs="*", default=[])
    parser.add_argument("--series", nargs="*", default=TARGET_SERIES)
    args = parser.parse_args()

    distribution = read_distribution(Path(args.distribution))
    episode_manifest = read_episode_manifest(Path(args.manifest))
    excluded_keys = read_excluded_episode_keys(list(args.exclude_jsonl or []))
    selected_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in distribution:
        if row["series_slug"] in set(args.series) and (row["series_slug"], int(row["episode_no"])) not in excluded_keys:
            by_series[row["series_slug"]].append(row)

    for series_slug in args.series:
        rows = by_series.get(series_slug, [])
        if args.strategy == "random":
            selected = select_series_random(
                rows,
                target_images=int(args.target_images),
                min_images=int(args.min_images),
                max_images=int(args.max_images),
                max_dropout_rate=float(args.max_dropout_rate),
                seed=int(args.seed) + sum(ord(ch) for ch in series_slug),
                attempts=int(args.random_attempts),
            )
            for row in selected:
                row["chronology_bin"] = min(max(1, int(args.bins)) - 1, int((int(row["episode_no"]) - 1) * max(1, int(args.bins)) / max(1, len(rows))))
        else:
            selected = select_series(
                rows,
                target_images=int(args.target_images),
                bin_count=max(1, int(args.bins)),
                max_dropout_rate=float(args.max_dropout_rate),
                max_over_target=max(0, int(args.max_over_target)),
            )
        selected_rows.extend(selected)
        summaries.append(summarize_selection(series_slug, selected))

    selected_rows.sort(key=lambda row: (row["series_slug"], int(row["episode_no"])))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series_slug",
        "series_name",
        "episode_no",
        "episode_slug",
        "label",
        "input_images",
        "kept_images",
        "dropped_images",
        "dropout_rate",
        "dominant_width",
        "story_start_position",
        "story_end_position",
        "chronology_bin",
        "url",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    output_jsonl = Path(args.output_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            key = (row["series_slug"], int(row["episode_no"]))
            episode = episode_manifest.get(key)
            if not episode:
                raise KeyError(f"missing manifest episode for {key}")
            payload = {
                "episode": episode,
                "selection": {
                    "source_prefix": "datasets/pages/source/webtoon_cleaned_dim_v2",
                    "target_images_per_series": int(args.target_images),
                    "kept_images": int(row["kept_images"]),
                    "input_images": int(row["input_images"]),
                    "dropped_images": int(row["dropped_images"]),
                    "dropout_rate": float(row["dropout_rate"]),
                    "chronology_bin": int(row.get("chronology_bin", 0)),
                    "dominant_width": int(row["dominant_width"]),
                    "story_start_position": int(row["story_start_position"] or 0),
                    "story_end_position": int(row["story_end_position"] or 0),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "target_images_per_series": int(args.target_images),
        "min_images_per_series": int(args.min_images),
        "max_images_per_series": int(args.max_images),
        "max_over_target": int(args.max_over_target),
        "max_dropout_rate": float(args.max_dropout_rate),
        "strategy": str(args.strategy),
        "seed": int(args.seed),
        "excluded_episode_count": len(excluded_keys),
        "series": summaries,
        "total_selected_episodes": len(selected_rows),
        "total_kept_images": sum(int(row["kept_images"]) for row in selected_rows),
        "total_input_images": sum(int(row["input_images"]) for row in selected_rows),
        "total_dropped_images": sum(int(row["dropped_images"]) for row in selected_rows),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
