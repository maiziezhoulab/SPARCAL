#!/usr/bin/env python3
"""Create an immutable-by-convention manifest for existing pipeline outputs.

This script never changes any file below a supplied root.  It records a
versioned inventory before a new analysis branch starts, so accidental changes
can be detected with a later re-run.  Small files are SHA-256 hashed; large
files retain exact size and nanosecond mtime unless ``--full-hash`` is requested.

The manifest is a freeze *contract*, not a filesystem permission change: shared
lab output trees remain writable for their owner.  New analyses must write into
new versioned directories and compare against this manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path,
                        help="Pipeline output root to freeze. Repeat as needed.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="New directory for this baseline manifest.")
    parser.add_argument("--label", required=True, help="Human-readable baseline identifier.")
    parser.add_argument("--exclude", action="append", default=[], metavar="RELATIVE_PATH_FRAGMENT",
                        help="Exclude files whose root-relative path contains this fragment.")
    parser.add_argument("--hash-max-mb", type=float, default=8.0,
                        help="Hash files no larger than this many MiB (default: 8).")
    parser.add_argument("--full-hash", action="store_true",
                        help="SHA-256 every file; potentially expensive for BAM/VCF trees.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(args: Iterable[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def should_exclude(relative: str, fragments: list[str]) -> bool:
    return any(fragment in relative for fragment in fragments)


def main() -> int:
    args = parse_args()
    if args.hash_max_mb < 0:
        raise ValueError("--hash-max-mb must be non-negative")
    roots = [root.resolve() for root in args.root]
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Freeze directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    hash_max_bytes = int(args.hash_max_mb * 1024 * 1024)
    manifest_path = out_dir / "file_manifest.tsv"
    root_summary = []
    total_files = total_bytes = hashed_files = 0
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["root", "relative_path", "size_bytes", "mtime_ns", "sha256", "hash_status"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for root in roots:
            root_files = root_bytes = root_hashed = 0
            # Stream the tree: materializing ``sorted(root.rglob('*'))`` can consume
            # unreasonable memory for the many per-spot files in a full pipeline run.
            for parent, dirnames, filenames in os.walk(root):
                dirnames.sort()
                for filename in sorted(filenames):
                    path = Path(parent) / filename
                    if path.is_symlink():
                        continue
                    relative = str(path.relative_to(root))
                    if should_exclude(relative, args.exclude):
                        continue
                    stat = path.stat()
                    do_hash = args.full_hash or stat.st_size <= hash_max_bytes
                    writer.writerow({
                        "root": str(root),
                        "relative_path": relative,
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256(path) if do_hash else "",
                        "hash_status": "sha256" if do_hash else "metadata_only",
                    })
                    root_files += 1
                    root_bytes += stat.st_size
                    root_hashed += int(do_hash)
            root_summary.append({"root": str(root), "files": root_files, "bytes": root_bytes,
                                 "sha256_files": root_hashed})
            total_files += root_files
            total_bytes += root_bytes
            hashed_files += root_hashed

    metadata = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": root_summary,
        "exclude_fragments": args.exclude,
        "hash_max_mb": args.hash_max_mb,
        "full_hash": args.full_hash,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "sha256_files": hashed_files,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_porcelain": git_value(["status", "--porcelain"]),
    }
    with (out_dir / "freeze_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (out_dir / "README.md").open("w") as handle:
        handle.write(
            f"# Frozen baseline: {args.label}\n\n"
            "The listed pipeline-output roots are not to be overwritten. New work must use a new "
            "versioned output directory. Re-run `freeze_baseline.py` with the same roots and excludes "
            "to detect changes using the inventory, metadata, and recorded checksums.\n"
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
