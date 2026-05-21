#!/usr/bin/env python3
"""Convert ai-toolkit FLUX.2 LoRA keys to fal's PEFT-style prefix.

Usage:
    python scripts/remap_to_fal.py input.safetensors output_fal.safetensors
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


OLD_PREFIX = "diffusion_model."
NEW_PREFIX = "base_model.model."


def _read_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    metadata = header.get("__metadata__", {})
    return metadata if isinstance(metadata, dict) else {}


def remap_to_fal(src: str | Path, dst: str | Path) -> int:
    source_path = Path(src)
    destination_path = Path(dst)
    metadata = _read_metadata(source_path)
    tensors = {}
    converted = 0
    unexpected_source_keys: list[str] = []

    source_tensors = load_file(str(source_path), device="cpu")
    for key, tensor in source_tensors.items():
        if key.startswith(OLD_PREFIX):
            new_key = NEW_PREFIX + key[len(OLD_PREFIX) :]
            converted += 1
        else:
            unexpected_source_keys.append(key)
            new_key = key
        tensors[new_key] = tensor

    if converted == 0:
        raise RuntimeError(f"No {OLD_PREFIX!r} tensor keys found in {source_path}")
    if unexpected_source_keys:
        examples = ", ".join(unexpected_source_keys[:5])
        raise RuntimeError(
            f"Cannot remap {source_path} for fal: tensor keys without "
            f"{OLD_PREFIX!r} prefix: {examples}"
        )
    bad_output_keys = [key for key in tensors if not key.startswith(NEW_PREFIX)]
    if bad_output_keys:
        examples = ", ".join(bad_output_keys[:5])
        raise RuntimeError(
            f"Cannot remap {source_path} for fal: output keys without "
            f"{NEW_PREFIX!r} prefix: {examples}"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(f".{destination_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    save_file(tensors, str(temp_path), metadata=metadata or None)
    os.replace(temp_path, destination_path)
    return converted


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    converted = remap_to_fal(argv[1], argv[2])
    print(f"remapped {converted} tensors -> {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
