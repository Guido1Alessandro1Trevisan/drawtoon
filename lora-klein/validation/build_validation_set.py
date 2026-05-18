#!/usr/bin/env python3
"""Build a fixed 200-sample validation dataset from the compact training manifest."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import boto3
from PIL import Image


HERE = Path(__file__).parent.resolve()
DEFAULT_MANIFEST_CACHE = HERE / ".cache" / "training_manifest.jsonl"
DEFAULT_VOLUME = "flux-dataset-cache"
DEFAULT_VOLUME_MANIFEST = "drawtoon_panel/haiku-pass_179c4e5b4dedcff7/manifest.jsonl"
DEFAULT_OUTPUT = HERE / "datasets" / "generalist"
DEFAULT_COUNT = 200
DEFAULT_SEED = 20260515
REMOTE_VALIDATION_SET_ROOT = "/root/validation_datasets/generalist"


def load_validate_run_module():
    spec = importlib.util.spec_from_file_location("validate_run", HERE / "validate_run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def download_manifest_if_needed(manifest_path: Path, *, volume: str, volume_manifest: str, force: bool) -> None:
    if manifest_path.is_file() and not force:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = manifest_path.parent / ".modal_get"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "run",
            "--active",
            "--with",
            "modal",
            "python",
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            volume,
            volume_manifest,
            str(tmp_dir),
        ],
        check=True,
    )
    candidates = list(tmp_dir.rglob(Path(volume_manifest).name))
    if not candidates:
        raise FileNotFoundError(f"modal volume get did not produce {Path(volume_manifest).name}")
    shutil.move(str(candidates[0]), str(manifest_path))
    shutil.rmtree(tmp_dir)


def row_score(row: dict[str, Any], seed: int) -> int:
    sample_id = str(row.get("sample_id") or "")
    payload = json.dumps([seed, sample_id, row.get("target_panel")], sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest(), 16)


def eligible_row(row: dict[str, Any]) -> bool:
    controls = row.get("controls") if isinstance(row.get("controls"), dict) else {}
    return (
        row.get("sample_type") == "panel_prediction"
        and bool(str(row.get("caption") or "").strip())
        and bool(row.get("target_panel"))
        and bool(controls.get("layout_control") or controls.get("layout_control_path"))
        and bool(controls.get("character_ref_paths"))
    )


def sample_manifest_rows(manifest_path: Path, *, count: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    heap: list[tuple[int, int, dict[str, Any]]] = []
    eligible_count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not eligible_row(row):
                continue
            eligible_count += 1
            score = row_score(row, seed)
            item = (-score, line_number, row)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    selected = [item[2] for item in sorted(heap, reverse=True)]
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} eligible rows in {manifest_path}; need {count}")
    return selected, eligible_count


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def build_validation_set(
    *,
    manifest_path: Path,
    output_dir: Path,
    remote_root: str,
    count: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    validate_run = load_validate_run_module()
    s3_client = boto3.client("s3")
    asset_cache = output_dir / "_asset_cache"
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_cache.mkdir(parents=True, exist_ok=True)

    rows, eligible_count = sample_manifest_rows(manifest_path, count=count, seed=seed)
    manifest_rows: list[dict[str, Any]] = []
    local_manifest_rows: list[dict[str, Any]] = []

    for index, source_row in enumerate(rows):
        row = json.loads(json.dumps(source_row))
        item_name = f"img_{index:04d}"
        item_dir = output_dir / item_name
        if item_dir.exists():
            shutil.rmtree(item_dir)
        item_dir.mkdir(parents=True, exist_ok=True)

        target_path = validate_run.materialize_asset(s3_client, row["target_panel"], asset_root=asset_cache)
        target_image = Image.open(target_path).convert("RGB")
        target_out = item_dir / "target.png"
        target_image.save(target_out, format="PNG", optimize=True)
        width, height = target_image.size

        controls = row.get("controls") or {}
        layout_metadata = controls.get("layout_control") or {}
        layout_out = item_dir / "ctrl_img_1.png"
        if layout_metadata:
            validate_run.materialize_layout_control(layout_metadata, width=width, height=height).save(
                layout_out,
                format="PNG",
                optimize=True,
            )
        else:
            layout_path = validate_run.materialize_asset(s3_client, controls["layout_control_path"], asset_root=asset_cache)
            Image.open(layout_path).convert("RGB").save(layout_out, format="PNG", optimize=True)

        remote_refs: list[str] = []
        local_refs: list[str] = []
        for ref_index, ref in enumerate((controls.get("character_ref_paths") or [])[:6], start=2):
            ref_path = validate_run.materialize_asset(s3_client, ref, asset_root=asset_cache)
            ref_out = item_dir / f"ctrl_img_{ref_index}.png"
            Image.open(ref_path).convert("RGB").save(ref_out, format="PNG", optimize=True)
            remote_refs.append(f"{remote_root.rstrip('/')}/{item_name}/ctrl_img_{ref_index}.png")
            local_refs.append(f"{item_name}/ctrl_img_{ref_index}.png")

        caption = str(row.get("caption") or "").strip().rstrip(".") + "."
        sample_id = str(row.get("sample_id") or item_name)
        write_text(item_dir / "caption.txt", caption)
        write_text(item_dir / "sample_id.txt", sample_id)

        row["eval_index"] = index
        row["eval_seed"] = seed + index
        row["eval_stratum"] = "/".join(validate_run.row_stratum(row))
        row["validation_set_dir"] = item_name
        row["target_panel"] = f"{remote_root.rstrip('/')}/{item_name}/target.png"
        row["target_width"] = width
        row["target_height"] = height
        row["caption"] = caption
        row["character_count"] = len(remote_refs)
        row["controls"] = controls
        row["controls"]["layout_control"] = layout_metadata
        row["controls"]["layout_control_path"] = f"{remote_root.rstrip('/')}/{item_name}/ctrl_img_1.png"
        row["controls"]["character_ref_paths"] = remote_refs
        row["controls"]["has_previous_control"] = bool(row["controls"].get("has_previous_control", False))
        row["controls"]["character_ref_policy"] = "fixed_random_manifest_sample"
        manifest_rows.append(row)

        local_row = json.loads(json.dumps(row))
        local_row["target_panel"] = f"{item_name}/target.png"
        local_row["controls"]["layout_control_path"] = f"{item_name}/ctrl_img_1.png"
        local_row["controls"]["character_ref_paths"] = local_refs
        local_manifest_rows.append(local_row)
        (item_dir / "metadata.json").write_text(
            json.dumps({"source_row": source_row, "manifest_row": row}, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (output_dir / "manifest_local.jsonl").open("w", encoding="utf-8") as handle:
        for row in local_manifest_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "count": len(manifest_rows),
        "seed": seed,
        "source_manifest": str(manifest_path),
        "eligible_rows": eligible_count,
        "sampling": "deterministic_uniform_hash_sample_from_compact_training_manifest",
        "remote_root": remote_root.rstrip("/"),
        "strata": {},
    }
    for row in manifest_rows:
        summary["strata"][row["eval_stratum"]] = int(summary["strata"].get(row["eval_stratum"], 0)) + 1
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if asset_cache.exists():
        shutil.rmtree(asset_cache)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_CACHE)
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--volume-manifest", default=DEFAULT_VOLUME_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-root", default=REMOTE_VALIDATION_SET_ROOT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.no_download:
        download_manifest_if_needed(
            args.manifest_path,
            volume=args.volume,
            volume_manifest=args.volume_manifest,
            force=args.force,
        )
    summary = build_validation_set(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        remote_root=args.remote_root,
        count=args.count,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
