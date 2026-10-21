#!/usr/bin/env python3
"""Drop exact-allele hg19 1000 Genomes sites from presence matrices."""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# Membership code below is copied verbatim in semantics from:
# /data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/analysis/
# somatic_callset_validation_2026-09-07/run_analysis.py
# (variant_key, valid_key, query_panel, scan_reference_vcf).
# Panel header checked 2026-10-06: 1000 Genomes Phase 3 b37/hs37d5; b37
# autosomal coordinates and lengths match this task's hg19 reference.
PANEL = {
    "hg19": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/"
        "1000Genome_hg19/hg19_chr{chrom}.vcf.gz"
    )
}


def norm_chrom(value: str) -> str:
    value = str(value).strip()
    return value[3:] if value.lower().startswith("chr") else value


def variant_key(chrom: str, pos: str | int, ref: str, alt: str) -> tuple[str, int, str, str]:
    return norm_chrom(chrom), int(pos), ref.upper(), alt.upper()


def valid_key(key: tuple[str, int, str, str]) -> bool:
    return len(key[2]) == 1 and len(key[3]) == 1


def query_panel(build: str, query: set) -> set:
    jobs = [
        (Path(PANEL[build].format(chrom=chrom)), {key for key in query if key[0] == str(chrom)})
        for chrom in range(1, 23)
    ]
    hits = set()
    with ProcessPoolExecutor(max_workers=8) as executor:
        for result in executor.map(scan_reference_vcf, jobs):
            hits.update(result)
    return hits


def scan_reference_vcf(args: tuple[Path, set]) -> set:
    """Stream a reference once; faster than thousands of indexed random seeks."""
    path, query = args
    hits = set()
    proc = subprocess.Popen(
        ["zcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line[0] == "#":
            continue
        fields = line.split("\t", 8)
        if len(fields) < 5 or len(fields[3]) != 1:
            continue
        for alt in fields[4].split(","):
            key = variant_key(fields[0], fields[1], fields[3], alt)
            if valid_key(key) and key in query:
                hits.add(key)
    if proc.wait() != 0:
        raise RuntimeError(f"failed to stream {path}")
    return hits


def parse_column(column: str) -> tuple[str, int, str, str]:
    fields = column.rsplit("_", 3)
    if len(fields) != 4:
        raise ValueError(f"Expected chrom_pos_ref_alt column, got {column!r}")
    return variant_key(*fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", required=True,
                        help="Repeat for multiple callers so the 1KGP files are streamed once")
    args = parser.parse_args()

    frames = []
    union = set()
    for name in args.matrix:
        frame = pd.read_pickle(name)
        keys = {parse_column(column) for column in frame.columns}
        frames.append((Path(name), frame))
        union.update(keys)
    hits = query_panel("hg19", union)
    print(f"Exact-allele query={len(union):,} 1KGP_hits={len(hits):,}")
    for path, frame in frames:
        keep = [column for column in frame.columns if parse_column(column) not in hits]
        output = path.with_name(path.name.replace("_presence_matrix.pkl", "_no1kgp_presence_matrix.pkl"))
        out = frame.loc[:, keep].astype("uint8", copy=False)
        out.to_pickle(output)
        print(f"{path.name}: {frame.shape} -> {out.shape}; dropped={frame.shape[1] - out.shape[1]:,}; {output}")


if __name__ == "__main__":
    main()
