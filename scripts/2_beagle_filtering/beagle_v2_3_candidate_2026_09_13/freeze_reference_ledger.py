#!/usr/bin/env python3
"""Freeze a completed panel-hash job into a bound immutable ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_sparcal_beagle_ledger_contracts", HERE / "contracts.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("could not load v2.3 contracts")
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)

COPIED = (
    "CODE_SHA256SUMS.txt",
    "FILE_METADATA.tsv",
    "MANIFEST.txt",
    "REFERENCE_PATHS.txt",
    "REFERENCE_SHA256SUMS.txt",
    "RUN_COMPLETE",
)


def freeze(source: Path, output: Path) -> Path:
    source = Path(source).resolve()
    output = Path(output).resolve()
    CONTRACTS.verify_reference_ledger(
        source, require_bound_completion=False
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)

    for name in COPIED:
        shutil.copy2(source / name, output / name)
    output_sums = output / "OUTPUT_SHA256SUMS.txt"
    with output_sums.open("x") as handle:
        for name in COPIED[:-1]:
            path = output / name
            handle.write(
                f"{CONTRACTS.sha256_file(path)}  {path.resolve()}\n"
            )

    frozen_manifest = output / "FROZEN_MANIFEST.json"
    payload = {
        "schema": "sparcal_beagle_reference_ledger_freeze_v1",
        "status": "outputs_ready_for_checksum",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "source_root": str(source),
        "source_artifacts": [
            CONTRACTS.file_identity(source / name)
            for name in (
                *COPIED,
                "OUTPUT_SHA256SUMS.txt",
            )
        ],
        "reference_artifact_count": 88,
        "reference_sha256sums_sha256": CONTRACTS.sha256_file(
            output / "REFERENCE_SHA256SUMS.txt"
        ),
    }
    with frozen_manifest.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    bundle_sums = CONTRACTS.write_sha256sums(
        output,
        [
            *(output / name for name in COPIED),
            output_sums,
            frozen_manifest,
        ],
        filename="BUNDLE_SHA256SUMS.txt",
    )
    completion = CONTRACTS.write_completion(
        output,
        manifest_path=frozen_manifest,
        checksum_path=bundle_sums,
    )
    CONTRACTS.verify_reference_ledger(output)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    completion = freeze(args.source_root, args.output_root)
    print(completion)


if __name__ == "__main__":
    main()
