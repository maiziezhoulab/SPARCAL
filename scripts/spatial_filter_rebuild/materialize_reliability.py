#!/usr/bin/env python3
"""Materialize only reliability-v2 call sets under unique model names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.spatial_filter_rebuild import config
from scripts.spatial_filter_rebuild.materialize import build_viewer, materialize_matrix
from scripts.spatial_filter_rebuild.score_reliability import RELIABILITY_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    parser.add_argument("--viewer-root", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for model in RELIABILITY_MODELS:
        result = materialize_matrix(args.section, model, "matched", overwrite=False)
        results.append(result)
        if result["shape"][1] > 0:
            build_viewer(args.section, model, "matched", args.viewer_root, overwrite=False)
    manifest = config.section_dir(args.section) / "materialized/reliability_v2_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite: {manifest}")
    manifest.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
