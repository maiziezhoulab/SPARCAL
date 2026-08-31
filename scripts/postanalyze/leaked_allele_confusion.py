#!/usr/bin/env python3
"""Classify common 1KGP alleles retained by SpatialSNV into SPARCAL outcomes.

This closes PAPER_WORK P1-6. The input is the allele-exact leaked-site table from
the corrected 2026-08-06 SpatialSNV callset-quality analysis. Every allele is
queried against the four mutually exclusive SPARCAL output classes from the same
tissue. Alleles absent from all four classes are reported as not detected.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")
BCFTOOLS = REPO / "apps/bcftools"
INPUT = (
    REPO
    / "data/spatialsnv_callset_quality_2026-08-06/germline_leaked_sites.csv"
)
DEFAULT_OUT = REPO / "data/leaked_allele_confusion_2026-08-23"

SAMPLE_ROOTS = {
    "P4": REPO / "data/P4_tumor/1",
    "P6": REPO / "data/P6_tumor/1",
    "DCIS1": REPO / "data/dcis1",
    "DCIS2": REPO / "data/dcis2",
}

CLASS_PATHS = {
    "germline": "spatial_filter_purity/baseQ0mapQ0/germline/defined/germline_defined.vcf.gz",
    "UPV": "spatial_filter_purity/baseQ0mapQ0/germline/denovo/germline_denovo.vcf.gz",
    "somatic": "spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
    "unresolved": "spatial_filter_purity/baseQ0mapQ0/ambiguous/denovo/ambiguous_denovo.vcf.gz",
}
CLASS_ORDER = ["not detected", "germline", "UPV", "somatic", "unresolved"]


def norm_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def allele_key(chrom: str, pos: str | int, ref: str, alt: str) -> str:
    return f"{norm_chrom(chrom)}_{int(pos)}_{str(ref).upper()}_{str(alt).upper()}"


def read_vcf(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    process = subprocess.Popen(
        [str(BCFTOOLS), "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    keys = set()
    assert process.stdout is not None
    for line in process.stdout:
        fields = line.rstrip().split("\t")
        if len(fields) != 4:
            continue
        chrom, pos, ref, alt_field = fields
        for alt in alt_field.split(","):
            if len(ref) == 1 and len(alt) == 1:
                keys.add(allele_key(chrom, pos, ref, alt))
    stderr = process.communicate()[1]
    if process.returncode:
        raise RuntimeError(f"bcftools failed for {path}: {stderr}")
    return keys


def write_results(
    outdir: Path, summary: pd.DataFrame, overlaps: int
) -> None:
    lines = [
        "# Leaked common-allele confusion table — 2026-08-23",
        "",
        "## Question",
        "",
        "SpatialSNV retained allele-exact 1KGP variants in its final somatic callset.",
        "For every such allele, this analysis asks whether SPARCAL did not detect it or",
        "assigned it to germline, UPV, retained somatic, or unresolved in the same tissue.",
        "",
        "## Results",
        "",
        "| sample | outcome | n | % of leaked alleles |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.sample} | {row.outcome} | {row.n} | {row.pct_of_leaked:.2f} |"
        )
    lines.extend([
        "",
        f"Class-overlap quality-control count: {overlaps}.",
        "",
        "The germline fraction is not an accuracy estimate by itself: these alleles were",
        "selected because they are in 1KGP, and SPARCAL routes panel alleles to germline",
        "by construction. The informative quantities are the full denominator, the",
        "not-detected fraction, and any allele assigned to a non-germline class.",
        "",
        "## Sources",
        "",
        "- Input: data/spatialsnv_callset_quality_2026-08-06/germline_leaked_sites.csv",
        "- SPARCAL class VCFs: each sample's spatial_filter_purity/baseQ0mapQ0 tree",
        "- Script: scripts/postanalyze/leaked_allele_confusion.py",
        "",
        "## Outputs",
        "",
        "- confusion_by_section.csv: plotted counts and percentages",
        "- confusion_sites.csv: allele-level classifications and class-membership QC",
    ])
    (outdir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    leaked = pd.read_csv(args.input, dtype={"chrom": str})
    leaked["allele_key"] = [
        allele_key(row.chrom, row.pos, row.ref, row.alt)
        for row in leaked.itertuples(index=False)
    ]
    if leaked.duplicated(["sample", "allele_key"]).any():
        raise ValueError("Input contains duplicate sample/allele rows")

    site_frames = []
    overlap_count = 0
    for sample, root in SAMPLE_ROOTS.items():
        sample_sites = leaked[leaked["sample"] == sample].copy()
        class_sets = {
            label: read_vcf(root / relative)
            for label, relative in CLASS_PATHS.items()
        }
        for label, keys in class_sets.items():
            sample_sites[f"in_{label.lower()}"] = sample_sites["allele_key"].isin(keys)
        membership_columns = [f"in_{label.lower()}" for label in CLASS_PATHS]
        sample_sites["n_sparcal_classes"] = sample_sites[membership_columns].sum(axis=1)
        overlap_count += int((sample_sites["n_sparcal_classes"] > 1).sum())

        def outcome(row) -> str:
            present = [
                label
                for label in CLASS_PATHS
                if bool(row[f"in_{label.lower()}"])
            ]
            if not present:
                return "not detected"
            if len(present) > 1:
                return "multiple classes"
            return present[0]

        sample_sites["outcome"] = sample_sites.apply(outcome, axis=1)
        site_frames.append(sample_sites)

    sites = pd.concat(site_frames, ignore_index=True)
    sites.to_csv(args.outdir / "confusion_sites.csv", index=False)
    if overlap_count:
        raise ValueError(
            f"{overlap_count} leaked alleles occur in multiple SPARCAL classes; "
            "inspect confusion_sites.csv before interpreting"
        )

    counts = (
        sites.groupby(["sample", "outcome"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [SAMPLE_ROOTS, CLASS_ORDER], names=["sample", "outcome"]
    ).to_frame(index=False)
    summary = complete.merge(counts, how="left").fillna({"n": 0})
    summary["n"] = summary["n"].astype(int)
    totals = sites.groupby("sample").size().rename("n_total_leaked")
    summary = summary.join(totals, on="sample")
    summary["pct_of_leaked"] = 100 * summary["n"] / summary["n_total_leaked"]
    detected = (
        sites.assign(detected=lambda frame: frame.outcome.ne("not detected"))
        .groupby("sample")["detected"]
        .sum()
        .rename("n_detected_by_sparcal")
    )
    summary = summary.join(detected, on="sample")
    summary["pct_detected_by_sparcal"] = (
        100 * summary["n_detected_by_sparcal"] / summary["n_total_leaked"]
    )

    summary.to_csv(args.outdir / "confusion_by_section.csv", index=False)
    write_results(args.outdir, summary, overlap_count)
    print(summary.to_string(index=False))
    print(f"\nWrote leaked-allele confusion package to {args.outdir}")


if __name__ == "__main__":
    main()
