#!/usr/bin/env python3
"""Apply external RNA-artifact masks to existing SPARCAL novel-call outputs.

This is deliberately a *post-filter benchmark*.  It never edits the input
spatial-filter tree.  The output tree preserves the subset VCF layout expected
by ``sparcal_set_benchmark.py`` and rewrites the per-spot germline/somatic lists
so coverage-controlled burden analyses remain valid.

The 1KGP/Beagle-defined germline set is copied unchanged.  Only the three
novel-call sets are filtered: UPV (germline/denovo), somatic/denovo, and
ambiguous/denovo.  A site is removed if its chromosome and position occur in
either a known RNA-editing catalogue or a scRNA panel of normals (PON).

An optional third mask, ``--accessibility-bed``, removes sites that fall
OUTSIDE a reference-accessibility interval set (the 1000 Genomes strict mask).
Its polarity is inverted relative to the other two, and its rationale is
different: SPARCAL's novel-call criterion is "absent from the 1KGP panel", which
is satisfied for free wherever that panel had no power to call anything.
Measured 2026-09-07, 55-76% of the denovo classes sit in inaccessible territory
versus 26-30% of the genome, and all seven published matched-DNA-confirmed
SPARCAL somatic calls sit inside accessible territory (p~0.002).  Omitting the
flag preserves the original editing+PON-only behaviour exactly.

Masks live in resources/artifact_masks/ (SComatic editing + PON, 1000G strict).
Because every mask is applied to whatever tree ``--source-root`` names, this
composes with a re-run of step 7: point it at the new spatial_filter_purity tree
and it produces the masked sibling without touching either tree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path

import pysam


QF = "baseQ0mapQ0"
SUBSET_VCFS = {
    "defined": "germline/defined/germline_defined.vcf.gz",
    "upv": "germline/denovo/germline_denovo.vcf.gz",
    "somatic": "somatic/denovo/somatic_denovo.vcf.gz",
    "ambiguous": "ambiguous/denovo/ambiguous_denovo.vcf.gz",
}
NOVEL_SETS = ("upv", "somatic", "ambiguous")


def canonical_chrom(chrom: str) -> str:
    """Return UCSC-style contig names used by the supplied masks."""
    if chrom.startswith("chr"):
        return chrom
    if chrom in {"M", "MT"}:
        return "chrM"
    return f"chr{chrom}"


class MaskLookup:
    """Chromosome-at-a-time tabix lookup to avoid loading 22M loci in memory."""

    def __init__(self, editing_bed: Path, pon_bed: Path):
        self.editing = pysam.TabixFile(str(editing_bed))
        self.pon = pysam.TabixFile(str(pon_bed))
        self.current_chrom: str | None = None
        self.editing_positions = array("I")
        self.pon_positions = array("I")

    @staticmethod
    def _read_positions(tabix: pysam.TabixFile, chrom: str) -> array:
        """Read a chromosome as a compact, sorted integer array.

        A Python set for the large RNA-editing catalogue can exceed the memory
        available in a batch shell.  The BED files are coordinate-sorted, so
        binary search in an unsigned-int array is both exact and far leaner.
        """
        try:
            return array("I", (int(row.split("\t")[2]) for row in tabix.fetch(chrom)))
        except ValueError:
            return array("I")

    @staticmethod
    def _contains(positions: array, pos: int) -> bool:
        idx = bisect_left(positions, pos)
        return idx < len(positions) and positions[idx] == pos

    def _set_chrom(self, chrom: str) -> None:
        if chrom == self.current_chrom:
            return
        self.current_chrom = chrom
        self.editing_positions = self._read_positions(self.editing, chrom)
        self.pon_positions = self._read_positions(self.pon, chrom)

    def reasons(self, chrom: str, pos: int) -> tuple[bool, bool]:
        chrom = canonical_chrom(chrom)
        self._set_chrom(chrom)
        return self._contains(self.editing_positions, pos), self._contains(self.pon_positions, pos)


class AccessibilityLookup:
    """Interval containment against a 1000 Genomes-style accessibility BED.

    Polarity is the OPPOSITE of the artifact masks above: a site is removed when
    it falls **outside** the listed intervals, i.e. in territory the reference
    panel could not call.  This matters because SPARCAL's novel-call criterion is
    "absent from the 1KGP panel", which is satisfied for free wherever the panel
    had no power -- measured at 55-76% of the denovo classes across P4/P6/DCIS,
    versus 26-30% of the genome (see data/xmhc_artifact_mechanism_2026-08-31/).

    The BED is coordinate-sorted, so one chromosome is held at a time as parallel
    start/end arrays and containment is a single binary search.  BED intervals are
    0-based half-open; VCF positions are 1-based.
    """

    def __init__(self, bed: Path):
        self.tabix = pysam.TabixFile(str(bed))
        self.current_chrom: str | None = None
        self.starts = array("I")
        self.ends = array("I")

    def _set_chrom(self, chrom: str) -> None:
        if chrom == self.current_chrom:
            return
        self.current_chrom = chrom
        starts, ends = array("I"), array("I")
        try:
            for row in self.tabix.fetch(chrom):
                fields = row.split("\t")
                starts.append(int(fields[1]))
                ends.append(int(fields[2]))
        except ValueError:
            pass  # contig absent from the mask -> treated as inaccessible
        self.starts, self.ends = starts, ends

    def accessible(self, chrom: str, pos: int) -> bool:
        self._set_chrom(canonical_chrom(chrom))
        if not self.starts:
            return False
        idx = bisect_right(self.starts, pos - 1) - 1
        return idx >= 0 and self.ends[idx] > pos - 1


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_existing(src: Path, dst: Path) -> None:
    """Link unchanged VCFs without copying large files into every experiment."""
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)
    src_index = Path(f"{src}.tbi")
    dst_index = Path(f"{dst}.tbi")
    if src_index.exists():
        if dst_index.exists() or dst_index.is_symlink():
            dst_index.unlink()
        os.symlink(src_index.resolve(), dst_index)


def filter_vcf(
    src: Path,
    dst: Path,
    lookup: MaskLookup,
    access: "AccessibilityLookup | None" = None,
) -> Counter:
    """Write a bgzipped/indexed VCF after filtering sites in either mask.

    When ``access`` is supplied, sites outside the accessibility intervals are
    also removed.  Reasons are counted independently so the summary shows how
    much each mask contributes, but a site is removed if ANY reason fires.
    """
    ensure_parent(dst)
    stats: Counter = Counter()
    with gzip.open(src, "rt") as fin, pysam.BGZFile(str(dst), "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line.encode())
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos = fields[0], int(fields[1])
            editing, pon = lookup.reasons(chrom, pos)
            inaccessible = access is not None and not access.accessible(chrom, pos)
            stats["input"] += 1
            if editing or pon or inaccessible:
                stats["removed"] += 1
                stats["editing"] += int(editing)
                stats["pon"] += int(pon)
                stats["both"] += int(editing and pon)
                stats["inaccessible"] += int(inaccessible)
                stats["inaccessible_only"] += int(inaccessible and not (editing or pon))
                continue
            stats["kept"] += 1
            fout.write(line.encode())
    pysam.tabix_index(str(dst), preset="vcf", force=True)
    return stats


def vcf_keys(path: Path) -> set[tuple[str, int, str, str]]:
    keys = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            keys.add((canonical_chrom(fields[0]), int(fields[1]), fields[3], fields[4]))
    return keys


def removed_keys_by_set(source_root: Path, output_root: Path) -> dict[str, set[tuple[str, int, str, str]]]:
    """Identify the already-filtered loci without revisiting a large mask per spot."""
    source_base = source_root / "spatial_filter_purity" / QF
    output_base = output_root / "spatial_filter_purity" / QF
    result = {}
    for name in ("upv", "somatic"):
        src = source_base / SUBSET_VCFS[name]
        dst = output_base / SUBSET_VCFS[name]
        if not src.exists() or not dst.exists():
            raise FileNotFoundError(f"Missing source or filtered VCF for per-spot {name}: {src} / {dst}")
        result[name] = vcf_keys(src) - vcf_keys(dst)
    return result


def filter_spot_lists(
    source_root: Path,
    output_root: Path,
    removed_by_set: dict[str, set[tuple[str, int, str, str]]],
) -> Counter:
    """Filter only novel rows, retaining every barcode and defined-germline row."""
    stats: Counter = Counter()
    src_base = source_root / "spatial_filter_purity" / QF
    dst_base = output_root / "spatial_filter_purity" / QF
    for category in ("germline", "somatic"):
        src_dir = src_base / category
        dst_dir = dst_base / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.txt")):
            if src.name.endswith("_variants.txt"):
                continue
            dst = dst_dir / src.name
            with src.open() as fin, dst.open("w") as fout:
                header = fin.readline()
                fout.write(header)
                for line in fin:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 4:
                        fout.write(line)
                        continue
                    is_novel = category == "somatic" or (
                        len(fields) >= 5 and fields[-1] == "denovo"
                    )
                    if not is_novel:
                        fout.write(line)
                        stats[f"{category}_defined_rows"] += 1
                        continue
                    set_name = "somatic" if category == "somatic" else "upv"
                    key = (canonical_chrom(fields[0]), int(fields[1]), fields[2], fields[3])
                    stats[f"{category}_novel_input_rows"] += 1
                    if key in removed_by_set[set_name]:
                        stats[f"{category}_novel_removed_rows"] += 1
                        continue
                    fout.write(line)
                    stats[f"{category}_novel_kept_rows"] += 1
    return stats


def run(
    source_root: Path,
    output_root: Path,
    editing_bed: Path,
    pon_bed: Path,
    sets: tuple[str, ...],
    propagate_spot_lists: bool,
    resume: bool,
    accessibility_bed: Path | None = None,
) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if output_root.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if resume and not output_root.exists():
        raise FileNotFoundError(f"Cannot resume a missing output directory: {output_root}")
    required_masks = [editing_bed, pon_bed]
    if accessibility_bed is not None:
        required_masks.append(accessibility_bed)
    for mask in required_masks:
        if not mask.exists() or not Path(f"{mask}.tbi").exists():
            raise FileNotFoundError(f"Mask or tabix index is missing: {mask}")

    output_root.mkdir(parents=True, exist_ok=True)
    lookup = MaskLookup(editing_bed, pon_bed)
    access = AccessibilityLookup(accessibility_bed) if accessibility_bed else None
    base_src = source_root / "spatial_filter_purity" / QF
    base_dst = output_root / "spatial_filter_purity" / QF
    summary: list[dict[str, object]] = []

    # Preserve the defined set exactly; artifact masks are not germline truth.
    defined_src = base_src / SUBSET_VCFS["defined"]
    if defined_src.exists() and not (base_dst / SUBSET_VCFS["defined"]).exists():
        link_existing(defined_src, base_dst / SUBSET_VCFS["defined"])
    summary.append({"component": "defined", "input": "linked unchanged"})

    for name in sets:
        src = base_src / SUBSET_VCFS[name]
        if not src.exists():
            summary.append({"component": name, "input": "missing"})
            continue
        print(f"Filtering {name}: {src}", flush=True)
        stats = filter_vcf(src, base_dst / SUBSET_VCFS[name], lookup, access)
        print(f"  kept {stats['kept']:,}/{stats['input']:,}"
              + (f"  (inaccessible-only removals: {stats['inaccessible_only']:,})"
                 if access else ""), flush=True)
        summary.append({"component": name, **dict(stats)})

    if propagate_spot_lists:
        print("Propagating masks into per-spot burden lists", flush=True)
        removed = removed_keys_by_set(source_root, output_root)
        spot_stats = filter_spot_lists(source_root, output_root, removed)
        summary.extend(
            {"component": key, "input": value} for key, value in sorted(spot_stats.items())
        )
    summary_path = output_root / (
        "artifact_mask_spot_summary.csv" if propagate_spot_lists else "artifact_mask_vcf_summary.csv"
    )
    keys = sorted({key for row in summary for key in row})
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="Dataset tree containing spatial_filter_purity/baseQ0mapQ0")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="New, non-existent output directory")
    parser.add_argument("--editing-bed", required=True, type=Path,
                        help="Tabix-indexed BED of known RNA-editing loci")
    parser.add_argument("--pon-bed", required=True, type=Path,
                        help="Tabix-indexed BED of recurrent scRNA PON loci")
    parser.add_argument("--accessibility-bed", type=Path, default=None,
                        help="Optional tabix-indexed BED of accessible intervals "
                             "(e.g. the 1000 Genomes strict mask). Sites OUTSIDE "
                             "these intervals are removed. Omit to keep the "
                             "editing+PON-only behaviour unchanged.")
    parser.add_argument("--sets", default=",".join(NOVEL_SETS),
                        help="comma-separated subset of upv,somatic,ambiguous to filter; use 'none' to only propagate")
    parser.add_argument("--propagate-spot-lists", action="store_true",
                        help="write filtered per-spot lists after all VCF subsets are available")
    parser.add_argument("--resume", action="store_true",
                        help="add another stage to an already-created experiment directory")
    args = parser.parse_args()
    sets = () if args.sets == "none" else tuple(x for x in args.sets.split(",") if x)
    unknown = set(sets) - set(NOVEL_SETS)
    if unknown:
        parser.error(f"unknown --sets value(s): {sorted(unknown)}")
    if args.propagate_spot_lists and sets not in (NOVEL_SETS, ()):
        parser.error("--propagate-spot-lists requires all novel subsets or --sets none")
    print(run(args.source_root, args.output_root, args.editing_bed, args.pon_bed,
              sets, args.propagate_spot_lists, args.resume, args.accessibility_bed))


if __name__ == "__main__":
    main()
