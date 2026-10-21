#!/usr/bin/env python3
"""Collect read-level artifact features for candidate spatial-RNA SNVs.

This is deliberately a *feature collector*, not a new caller.  It preserves the
current SPARCAL/Monopogen-inspired calls as a baseline and extracts the evidence
needed for a replacement artifact likelihood:

* spot-by-site ref/alt counts and UMI support;
* alt-versus-ref base quality, mapping quality, read-end position and strand;
* duplicate/secondary/supplementary alignment and NM-tag summaries.

Input candidates are supplied as labelled VCFs, for example a current somatic
set, ambiguous set and/or the existing sequence-error set.  Labels are retained
as provenance only; they are *not* truth labels.  The BAM must be indexed and
have a Cell Ranger ``CB`` tag.  ``UB`` is used when it is present.

The output contains two gzip-compressed TSVs:

``spot_features.tsv.gz``
    One row per candidate and barcode with at least one ref/alt observation.
``site_features.tsv.gz``
    Pooled evidence and spatial-support summaries per candidate.

Use a small stable subsample first.  Do not run an unrestricted 60k-site P6 job
until the pilot feature table has been checked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from array import array
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple

import pysam


Key = Tuple[str, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        action="append",
        required=True,
        metavar="LABEL=VCF_GZ",
        help="Labelled candidate VCF. Repeat for multiple sets; labels are provenance only.",
    )
    parser.add_argument("--bam", required=True, help="Indexed CB-tagged BAM (normally possorted_genome_bam.dedup.bam).")
    parser.add_argument("--out-dir", required=True, help="New output directory; existing files are replaced only with --overwrite.")
    parser.add_argument(
        "--max-sites-per-source",
        type=int,
        default=2_000,
        help="Stable hash subsample per candidate source (default: 2,000; 0 means all sites).",
    )
    parser.add_argument("--seed", default="sparcal-artifact-v1", help="Seed used for deterministic source-wise subsampling.")
    parser.add_argument("--max-depth", type=int, default=100_000, help="Maximum pileup depth at a candidate site.")
    parser.add_argument("--read-end-bases", type=int, default=5, help="Bases at either read end counted as terminal.")
    parser.add_argument("--editing-bed", default=None,
                        help="Optional tabix-indexed BED of known RNA-editing positions.")
    parser.add_argument("--pon-bed", default=None,
                        help="Optional tabix-indexed BED of recurrent RNA PON positions.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing collector outputs in --out-dir.")
    return parser.parse_args()


def parse_source(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        raise ValueError(f"Candidate argument must be LABEL=VCF_GZ, got: {arg}")
    label, path = arg.split("=", 1)
    if not label or not path:
        raise ValueError(f"Candidate argument must be LABEL=VCF_GZ, got: {arg}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return label, path


def key_text(key: Key) -> str:
    return f"{key[0]}:{key[1]}:{key[2]}:{key[3]}"


def canonical_chrom(chrom: str) -> str:
    """Return UCSC-style chromosome names used by the public masks."""
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"M", "MT"}:
        return "chrM"
    return f"chr{chrom}"


class TabixCoordinateMask:
    """Memory-bounded lookup for a tabix-indexed BED mask.

    The bundled SComatic BED end coordinate equals the corresponding 1-based
    VCF position.  Cache one chromosome at a time rather than retaining the
    full multi-million-site resource in memory.
    """

    def __init__(self, path: str | None):
        self.tabix = pysam.TabixFile(path) if path else None
        self.current_chrom: str | None = None
        self.positions = array("I")

    def contains(self, chrom: str, pos: int) -> bool:
        if self.tabix is None:
            return False
        chrom = canonical_chrom(chrom)
        if chrom != self.current_chrom:
            self.current_chrom = chrom
            try:
                self.positions = array("I", (int(row.split("\t")[2]) for row in self.tabix.fetch(chrom)))
            except ValueError:
                self.positions = array("I")
        index = bisect_left(self.positions, pos)
        return index < len(self.positions) and self.positions[index] == pos


def stable_sample(keys: Iterable[Key], n: int, seed: str) -> Set[Key]:
    keys = set(keys)
    if n <= 0 or len(keys) <= n:
        return keys
    return set(
        sorted(
            keys,
            key=lambda key: hashlib.sha256(f"{seed}\0{key_text(key)}".encode()).digest(),
        )[:n]
    )


def read_candidate_vcf(path: str) -> Set[Key]:
    """Read unique biallelic SNP candidates without treating VCF FILTER as truth."""
    keys: Set[Key] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            chrom, pos_text, _, ref, alts = fields[:5]
            if len(ref) != 1:
                continue
            try:
                pos = int(pos_text)
            except ValueError:
                continue
            for alt in alts.split(","):
                if len(alt) == 1 and alt in "ACGT" and ref in "ACGT":
                    keys.add((chrom, pos, ref, alt))
    return keys


def load_candidates(args: argparse.Namespace) -> Tuple[Dict[Key, Set[str]], Dict[str, int], Dict[str, int]]:
    """Return candidate -> provenance labels plus source before/after counts."""
    by_key: DefaultDict[Key, Set[str]] = defaultdict(set)
    before: Dict[str, int] = {}
    after: Dict[str, int] = {}
    for source_arg in args.candidates:
        label, path = parse_source(source_arg)
        if label in before:
            raise ValueError(f"Candidate label was supplied more than once: {label}")
        source_keys = read_candidate_vcf(path)
        before[label] = len(source_keys)
        source_keys = stable_sample(source_keys, args.max_sites_per_source, args.seed)
        after[label] = len(source_keys)
        for key in source_keys:
            by_key[key].add(label)
    return dict(by_key), before, after


@dataclass
class AlleleStats:
    n: int = 0
    bq_sum: float = 0.0
    mq_sum: float = 0.0
    read_pos_sum: float = 0.0
    terminal_n: int = 0
    forward_n: int = 0
    reverse_n: int = 0
    duplicate_n: int = 0
    secondary_n: int = 0
    supplementary_n: int = 0
    mq0_n: int = 0
    nm_sum: float = 0.0
    nm_n: int = 0
    nh_sum: float = 0.0
    nh_n: int = 0
    multi_map_n: int = 0
    umis: Set[str] = field(default_factory=set)

    def add(self, aln: pysam.AlignedSegment, qpos: int, base_quality: int, read_end_bases: int) -> None:
        self.n += 1
        self.bq_sum += base_quality
        self.mq_sum += aln.mapping_quality
        read_length = max(1, aln.query_length or len(aln.query_sequence or ""))
        distance_to_end = min(qpos, max(0, read_length - 1 - qpos))
        self.read_pos_sum += distance_to_end / max(1, read_length - 1)
        self.terminal_n += int(distance_to_end < read_end_bases)
        self.forward_n += int(not aln.is_reverse)
        self.reverse_n += int(aln.is_reverse)
        self.duplicate_n += int(aln.is_duplicate)
        self.secondary_n += int(aln.is_secondary)
        self.supplementary_n += int(aln.is_supplementary)
        self.mq0_n += int(aln.mapping_quality == 0)
        if aln.has_tag("NM"):
            self.nm_sum += float(aln.get_tag("NM"))
            self.nm_n += 1
        if aln.has_tag("NH"):
            nh = float(aln.get_tag("NH"))
            self.nh_sum += nh
            self.nh_n += 1
            self.multi_map_n += int(nh > 1)
        if aln.has_tag("UB"):
            self.umis.add(str(aln.get_tag("UB")))

    def merge(self, other: "AlleleStats") -> None:
        self.n += other.n
        self.bq_sum += other.bq_sum
        self.mq_sum += other.mq_sum
        self.read_pos_sum += other.read_pos_sum
        self.terminal_n += other.terminal_n
        self.forward_n += other.forward_n
        self.reverse_n += other.reverse_n
        self.duplicate_n += other.duplicate_n
        self.secondary_n += other.secondary_n
        self.supplementary_n += other.supplementary_n
        self.mq0_n += other.mq0_n
        self.nm_sum += other.nm_sum
        self.nm_n += other.nm_n
        self.nh_sum += other.nh_sum
        self.nh_n += other.nh_n
        self.multi_map_n += other.multi_map_n
        # Across spots, a UMI string alone is not globally unique. Site summaries use
        # the sum of per-spot UMI counts rather than merging this set.


@dataclass
class SpotCounts:
    ref: AlleleStats = field(default_factory=AlleleStats)
    alt: AlleleStats = field(default_factory=AlleleStats)
    other_n: int = 0


def mean(total: float, n: int) -> float:
    return total / n if n else math.nan


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.6g}"


def quality_row(prefix: str, stats: AlleleStats) -> Dict[str, str]:
    return {
        f"{prefix}_n": str(stats.n),
        f"{prefix}_umi_n": str(len(stats.umis)),
        f"{prefix}_mean_bq": fmt(mean(stats.bq_sum, stats.n)),
        f"{prefix}_mean_mq": fmt(mean(stats.mq_sum, stats.n)),
        f"{prefix}_mean_read_pos_frac": fmt(mean(stats.read_pos_sum, stats.n)),
        f"{prefix}_terminal_frac": fmt(fraction(stats.terminal_n, stats.n)),
        f"{prefix}_forward_frac": fmt(fraction(stats.forward_n, stats.n)),
        f"{prefix}_duplicate_frac": fmt(fraction(stats.duplicate_n, stats.n)),
        f"{prefix}_secondary_frac": fmt(fraction(stats.secondary_n, stats.n)),
        f"{prefix}_supplementary_frac": fmt(fraction(stats.supplementary_n, stats.n)),
        f"{prefix}_mq0_frac": fmt(fraction(stats.mq0_n, stats.n)),
        f"{prefix}_mean_nm": fmt(mean(stats.nm_sum, stats.nm_n)),
        f"{prefix}_mean_nh": fmt(mean(stats.nh_sum, stats.nh_n)),
        f"{prefix}_multi_map_frac": fmt(fraction(stats.multi_map_n, stats.n)),
    }


def row_for_counts(
    key: Key,
    sources: Set[str],
    annotations: Mapping[str, str],
    barcode: str,
    counts: SpotCounts,
) -> Dict[str, str]:
    ref_n, alt_n = counts.ref.n, counts.alt.n
    depth = ref_n + alt_n
    row = {
        "candidate_id": key_text(key),
        "chrom": key[0],
        "pos": str(key[1]),
        "ref": key[2],
        "alt": key[3],
        "candidate_sources": ",".join(sorted(sources)),
        **annotations,
        "barcode": barcode,
        "depth_ref_alt": str(depth),
        "other_base_n": str(counts.other_n),
        "alt_vaf": fmt(fraction(alt_n, depth)),
        "alt_minus_ref_mean_bq": fmt(mean(counts.alt.bq_sum, alt_n) - mean(counts.ref.bq_sum, ref_n))
        if alt_n and ref_n
        else "NA",
        "alt_minus_ref_mean_mq": fmt(mean(counts.alt.mq_sum, alt_n) - mean(counts.ref.mq_sum, ref_n))
        if alt_n and ref_n
        else "NA",
        "alt_minus_ref_read_pos_frac": fmt(
            mean(counts.alt.read_pos_sum, alt_n) - mean(counts.ref.read_pos_sum, ref_n)
        )
        if alt_n and ref_n
        else "NA",
        "alt_minus_ref_terminal_frac": fmt(
            fraction(counts.alt.terminal_n, alt_n) - fraction(counts.ref.terminal_n, ref_n)
        )
        if alt_n and ref_n
        else "NA",
        "alt_minus_ref_forward_frac": fmt(
            fraction(counts.alt.forward_n, alt_n) - fraction(counts.ref.forward_n, ref_n)
        )
        if alt_n and ref_n
        else "NA",
    }
    row.update(quality_row("ref", counts.ref))
    row.update(quality_row("alt", counts.alt))
    return row


def output_columns() -> List[str]:
    base = [
        "candidate_id", "chrom", "pos", "ref", "alt", "candidate_sources",
        "known_rna_editing", "external_pon", "editing_compatible_substitution", "barcode",
        "depth_ref_alt", "other_base_n", "alt_vaf",
        "alt_minus_ref_mean_bq", "alt_minus_ref_mean_mq", "alt_minus_ref_read_pos_frac",
        "alt_minus_ref_terminal_frac", "alt_minus_ref_forward_frac",
    ]
    quality = []
    for prefix in ("ref", "alt"):
        quality.extend([
            f"{prefix}_n", f"{prefix}_umi_n", f"{prefix}_mean_bq", f"{prefix}_mean_mq",
            f"{prefix}_mean_read_pos_frac", f"{prefix}_terminal_frac", f"{prefix}_forward_frac",
            f"{prefix}_duplicate_frac", f"{prefix}_secondary_frac", f"{prefix}_supplementary_frac",
            f"{prefix}_mq0_frac", f"{prefix}_mean_nm", f"{prefix}_mean_nh", f"{prefix}_multi_map_frac",
        ])
    return base + quality


def resolve_contig(chrom: str, references: Set[str]) -> str | None:
    if chrom in references:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in references:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in references:
        return prefixed
    return None


def update_counts(
    counts_by_barcode: MutableMapping[str, SpotCounts],
    aln: pysam.AlignedSegment,
    qpos: int,
    ref: str,
    alt: str,
    read_end_bases: int,
) -> None:
    if not aln.has_tag("CB") or aln.query_sequence is None:
        return
    barcode = str(aln.get_tag("CB"))
    qualities = aln.query_qualities
    if qpos < 0 or qpos >= len(aln.query_sequence):
        return
    base = aln.query_sequence[qpos].upper()
    counts = counts_by_barcode.setdefault(barcode, SpotCounts())
    base_quality = int(qualities[qpos]) if qualities is not None and qpos < len(qualities) else 0
    if base == ref:
        counts.ref.add(aln, qpos, base_quality, read_end_bases)
    elif base == alt:
        counts.alt.add(aln, qpos, base_quality, read_end_bases)
    else:
        counts.other_n += 1


def pileup_one_site(
    bam: pysam.AlignmentFile,
    bam_chrom: str,
    pos: int,
    ref: str,
    alt: str,
    max_depth: int,
    read_end_bases: int,
) -> Tuple[Dict[str, SpotCounts], bool]:
    """Return barcode counts and whether pysam reached the depth cap."""
    counts_by_barcode: Dict[str, SpotCounts] = {}
    capped = False
    for column in bam.pileup(
        bam_chrom,
        pos - 1,
        pos,
        truncate=True,
        stepper="all",
        max_depth=max_depth,
        min_base_quality=0,
        min_mapping_quality=0,
    ):
        if column.reference_pos != pos - 1:
            continue
        capped = column.nsegments >= max_depth
        for pileup_read in column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip or pileup_read.query_position is None:
                continue
            update_counts(
                counts_by_barcode,
                pileup_read.alignment,
                pileup_read.query_position,
                ref,
                alt,
                read_end_bases,
            )
    return counts_by_barcode, capped


def build_site_row(
    key: Key,
    sources: Set[str],
    annotations: Mapping[str, str],
    spot_counts: Mapping[str, SpotCounts],
    pileup_capped: bool,
) -> Dict[str, str]:
    pooled = SpotCounts()
    ref_umi_sum = alt_umi_sum = 0
    n_alt_spots = n_ref_spots = 0
    for counts in spot_counts.values():
        pooled.ref.merge(counts.ref)
        pooled.alt.merge(counts.alt)
        pooled.other_n += counts.other_n
        ref_umi_sum += len(counts.ref.umis)
        alt_umi_sum += len(counts.alt.umis)
        n_alt_spots += int(counts.alt.n > 0)
        n_ref_spots += int(counts.ref.n > 0)
    row = row_for_counts(key, sources, annotations, "__POOLED__", pooled)
    row.pop("barcode")
    # UMI strings are only unique within a barcode; the pooled value is their sum.
    row["ref_umi_n"] = str(ref_umi_sum)
    row["alt_umi_n"] = str(alt_umi_sum)
    row["n_spots_depth"] = str(len(spot_counts))
    row["n_spots_alt"] = str(n_alt_spots)
    row["n_spots_ref"] = str(n_ref_spots)
    row["pileup_capped"] = str(int(pileup_capped))
    return row


def main() -> int:
    args = parse_args()
    if args.max_sites_per_source < 0:
        raise ValueError("--max-sites-per-source must be >= 0")
    if not os.path.isfile(args.bam):
        raise FileNotFoundError(args.bam)
    if not os.path.isfile(f"{args.bam}.bai") and not os.path.isfile(str(Path(args.bam).with_suffix(".bam.bai"))):
        raise FileNotFoundError(f"BAM index not found for {args.bam}")

    candidates, before, after = load_candidates(args)
    if not candidates:
        raise RuntimeError("No biallelic SNP candidates were found.")
    for mask in (args.editing_bed, args.pon_bed):
        if mask and not os.path.isfile(mask):
            raise FileNotFoundError(mask)
    editing_mask = TabixCoordinateMask(args.editing_bed)
    pon_mask = TabixCoordinateMask(args.pon_bed)
    annotations = {
        key: {
            "known_rna_editing": str(int(editing_mask.contains(key[0], key[1]))),
            "external_pon": str(int(pon_mask.contains(key[0], key[1]))),
            # Canonical A-to-I editing is observed as A>G; the reverse-strand
            # representation is T>C. This is annotation evidence, not a call.
            "editing_compatible_substitution": str(int((key[2], key[3]) in {("A", "G"), ("T", "C")})),
        }
        for key in candidates
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spot_path = out_dir / "spot_features.tsv.gz"
    site_path = out_dir / "site_features.tsv.gz"
    summary_path = out_dir / "summary.json"
    if not args.overwrite:
        existing = [path for path in (spot_path, site_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to replace: " + ", ".join(map(str, existing)))

    print(f"Candidates after source-wise sampling: {len(candidates):,}", flush=True)
    for label in sorted(before):
        print(f"  {label}: {before[label]:,} -> {after[label]:,}", flush=True)

    spot_columns = output_columns()
    site_columns = [column for column in spot_columns if column != "barcode"] + [
        "n_spots_depth", "n_spots_alt", "n_spots_ref", "pileup_capped"
    ]
    no_coverage: List[str] = []
    capped_sites = 0
    processed = 0

    with pysam.AlignmentFile(args.bam, "rb") as bam, \
         gzip.open(spot_path, "wt", newline="") as spot_handle, \
         gzip.open(site_path, "wt", newline="") as site_handle:
        references = set(bam.references)
        by_contig: DefaultDict[str, List[Key]] = defaultdict(list)
        unresolved: List[str] = []
        for key in candidates:
            bam_chrom = resolve_contig(key[0], references)
            if bam_chrom is None:
                unresolved.append(key_text(key))
            else:
                by_contig[bam_chrom].append(key)
        if unresolved:
            print(f"[WARN] {len(unresolved):,} candidates have no BAM contig and will be skipped.", file=sys.stderr, flush=True)

        spot_writer = csv.DictWriter(spot_handle, fieldnames=spot_columns, delimiter="\t", lineterminator="\n")
        site_writer = csv.DictWriter(site_handle, fieldnames=site_columns, delimiter="\t", lineterminator="\n")
        spot_writer.writeheader()
        site_writer.writeheader()

        total = sum(len(keys) for keys in by_contig.values())
        for bam_chrom in sorted(by_contig):
            for key in sorted(by_contig[bam_chrom], key=lambda row: row[1]):
                processed += 1
                _, pos, ref, alt = key
                spot_counts, pileup_capped = pileup_one_site(
                    bam, bam_chrom, pos, ref, alt, args.max_depth, args.read_end_bases
                )
                if pileup_capped:
                    capped_sites += 1
                if not spot_counts:
                    no_coverage.append(key_text(key))
                    continue
                for barcode in sorted(spot_counts):
                    spot_writer.writerow(
                        row_for_counts(key, candidates[key], annotations[key], barcode, spot_counts[barcode])
                    )
                site_writer.writerow(
                    build_site_row(key, candidates[key], annotations[key], spot_counts, pileup_capped)
                )
                if processed % 250 == 0 or processed == total:
                    print(f"Processed {processed:,}/{total:,} sites; covered={processed - len(no_coverage):,}", flush=True)

    no_coverage_path = out_dir / "no_coverage_candidates.txt"
    with open(no_coverage_path, "w") as handle:
        handle.write("candidate_id\n")
        handle.write("\n".join(no_coverage))
        if no_coverage:
            handle.write("\n")
    summary = {
        "bam": os.path.abspath(args.bam),
        "candidate_sources_before_sampling": before,
        "candidate_sources_after_sampling": after,
        "unique_candidates_after_sampling": len(candidates),
        "processed_candidates": processed,
        "covered_candidates": processed - len(no_coverage),
        "uncovered_candidates": len(no_coverage),
        "pileup_capped_candidates": capped_sites,
        "max_depth": args.max_depth,
        "read_end_bases": args.read_end_bases,
        "editing_bed": os.path.abspath(args.editing_bed) if args.editing_bed else None,
        "pon_bed": os.path.abspath(args.pon_bed) if args.pon_bed else None,
        "seed": args.seed,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {spot_path}\nWrote {site_path}\nWrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
