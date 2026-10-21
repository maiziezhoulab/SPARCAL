#!/usr/bin/env python3
"""Fail-closed integrity helpers for immutable Beagle selection bundles."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def verify_identity(
    record: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} is not an identity record")
    required = {"path", "size_bytes", "mtime_ns", "sha256"}
    if set(record) != required:
        raise ValueError(f"{label} identity fields are not exact")
    path = Path(str(record["path"])).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} points to the wrong path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = identity(path)
    for field in ("size_bytes", "mtime_ns", "sha256"):
        if record[field] != observed[field]:
            raise ValueError(f"{label} identity mismatch for {field}: {path}")
    return path


def parse_and_verify_ledger(
    ledger: Path,
    *,
    roots: Sequence[Path],
    label: str,
    require_contained_in: Path | None = None,
) -> dict[str, Path]:
    ledger = Path(ledger).resolve()
    if not ledger.is_file():
        raise FileNotFoundError(f"missing {label}: {ledger}")
    roots = tuple(Path(root).resolve() for root in roots)
    contained = (
        Path(require_contained_in).resolve()
        if require_contained_in is not None
        else None
    )
    records: dict[str, Path] = {}
    for line_number, line in enumerate(ledger.read_text().splitlines(), start=1):
        if not line or "  " not in line:
            raise ValueError(f"malformed {label} line {line_number}")
        digest, name = line.split("  ", 1)
        if not SHA256_RE.fullmatch(digest) or not name:
            raise ValueError(f"malformed {label} line {line_number}")
        if name in records:
            raise ValueError(f"duplicate {label} entry: {name}")
        source = Path(name)
        candidates = (
            (source.resolve(),)
            if source.is_absolute()
            else tuple((root / source).resolve() for root in roots)
        )
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                f"{label} entry does not resolve uniquely: {name}"
            )
        path = existing[0]
        if contained is not None:
            try:
                path.relative_to(contained)
            except ValueError as error:
                raise ValueError(f"{label} entry escapes its bundle: {name}") from error
        if sha256(path) != digest:
            raise ValueError(f"{label} checksum mismatch: {path}")
        records[name] = path
    if not records:
        raise ValueError(f"{label} is empty")
    return records


def verify_output_bundle(
    root: Path,
    completion: Mapping[str, object],
    *,
    expected_names: Sequence[str],
    label: str,
) -> list[Path]:
    root = Path(root).resolve()
    ledger = root / "OUTPUT_SHA256SUMS.txt"
    verify_identity(
        completion.get("output_sha256sums"),
        label=f"{label} output ledger",
        expected_path=ledger,
    )
    records = parse_and_verify_ledger(
        ledger,
        roots=(root,),
        label=f"{label} output ledger",
        require_contained_in=root,
    )
    observed = {
        str(path.relative_to(root)) for path in records.values()
    }
    expected = set(expected_names)
    if observed != expected or len(records) != len(expected):
        raise ValueError(
            f"{label} output ledger set is not exact: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return [root / name for name in expected_names]
