#!/usr/bin/env python3
"""Optional LoRA registry hook used by Modal training jobs.

The full fine-tune paths do not call this helper. It exists so the Modal image
mount remains stable when PostgreSQL-backed LoRA registration is disabled.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trigger-word", default="")
    parser.add_argument("--metadata", default="{}")
    args = parser.parse_args()

    # Validate metadata is at least parseable; actual registry integration is optional.
    json.loads(args.metadata)
    print(f"LoRA registry hook skipped for {args.name}: no registry backend configured")


if __name__ == "__main__":
    main()
