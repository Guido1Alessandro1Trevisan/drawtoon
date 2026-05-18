#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import ec2_aot_textcolor_ablation as runner


JOB_NAME = (
    "drawtoon_flux2_klein9b_attack-on-titan_mangazero_panel_prediction_native_pad16_"
    "haiku45_lora_r64_lr5e5_3500_b300_gb1_charbox_refborder"
)

DESCRIPTION = (
    "Attack on Titan control-encoding ablation. Same r64/LR 5e-5/global-batch-1/3500-step "
    "setup as the best AOT baseline, with text regions as filled type-color rectangles, "
    "characters as colored outline boxes with per-slot widths, and character reference crops "
    "wrapped in a 3px border matching the corresponding character control color."
)


runner.JOB_NAME = JOB_NAME


_original_write_config = runner.write_config


def write_config(repo_root: Path, output_config: Path) -> None:
    _original_write_config(repo_root, output_config)
    cfg = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    cfg.setdefault("meta", {})["description"] = DESCRIPTION
    output_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def summarize_layout_colors(manifest_path: Path, *, max_rows: int = 256) -> dict[str, Any]:
    checked = 0
    text_shapes: dict[str, int] = {}
    text_types: dict[str, int] = {}
    character_widths: dict[str, int] = {}
    ref_border_widths: dict[str, int] = {}
    ref_border_colors: dict[str, int] = {}
    for line in manifest_path.open("r", encoding="utf-8"):
        if checked >= max_rows:
            break
        row = json.loads(line)
        controls = row.get("controls") if isinstance(row.get("controls"), dict) else {}
        layout = controls.get("layout_control") if isinstance(controls.get("layout_control"), dict) else {}
        text = layout.get("text") if isinstance(layout.get("text"), dict) else {}
        for region in text.get("regions") or []:
            if not isinstance(region, dict) or not region.get("masked"):
                continue
            region_type = str(region.get("type") or "")
            shape = str(region.get("shape") or "")
            text_types[region_type] = text_types.get(region_type, 0) + 1
            text_shapes[shape] = text_shapes.get(shape, 0) + 1
        for character in layout.get("characters") or []:
            if not isinstance(character, dict):
                continue
            width = str(character.get("line_width") or "")
            character_widths[width] = character_widths.get(width, 0) + 1
        for ref in controls.get("character_ref_paths") or []:
            if not isinstance(ref, dict):
                continue
            border_width = str(ref.get("border_width") or "")
            ref_border_widths[border_width] = ref_border_widths.get(border_width, 0) + 1
            border_rgb = ref.get("border_rgb")
            if isinstance(border_rgb, list):
                color = ",".join(str(value) for value in border_rgb)
                ref_border_colors[color] = ref_border_colors.get(color, 0) + 1
        checked += 1
    return {
        "checked_manifest_rows": checked,
        "text_region_shapes": text_shapes,
        "text_region_types": text_types,
        "character_outline_widths": character_widths,
        "ref_border_widths": ref_border_widths,
        "ref_border_colors": ref_border_colors,
    }


runner.write_config = write_config
runner.summarize_layout_colors = summarize_layout_colors


if __name__ == "__main__":
    runner.main()
