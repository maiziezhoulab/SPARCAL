#!/usr/bin/env python3
"""Fail-fast validation for a two-section P4/P6 hg19 CalicoST run."""

import argparse
import csv
import gzip
import re
from pathlib import Path

import pysam


HG19_CHR1_LENGTH = 249_250_621


def config_value(path: Path, key: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", path.read_text(), re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"missing {key} in {path}")
    return match.group(1).strip('"\'')


def first_record_chrom(path: Path) -> str:
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                return line.split("\t", 1)[0]
    raise RuntimeError(f"no VCF records: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    args = parser.parse_args()
    sample_dir = args.sample_dir
    eagle = sample_dir / "config_eagle2.yaml"
    purity = sample_dir / "configuration_purity"
    cna = sample_dir / "configuration_cna"
    cna_unsup = sample_dir / "configuration_cna_unsup"

    combined = "".join(path.read_text() for path in (eagle, purity, cna, cna_unsup))
    if "hg38" in combined.lower() or "grch38" in combined.lower():
        raise RuntimeError("hg38/GRCh38 reference leaked into generated hg19 configs")
    if config_value(eagle, "samtools_sorting_mem") != "8G":
        raise RuntimeError("samtools_sorting_mem must be 8G for the merged BAM")
    if config_value(cna_unsup, "tumorprop_file") != "None":
        raise RuntimeError("unsupervised CNA config must have tumorprop_file: None")

    with (sample_dir / "input_filelist.tsv").open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) != 2 or any(len(fields) != 3 for fields in rows):
        raise RuntimeError("input_filelist.tsv must have exactly two three-column rows")
    sample_ids = [fields[1] for fields in rows]
    if len(set(sample_ids)) != 2 or any(not sample_id for sample_id in sample_ids):
        raise RuntimeError(f"sample IDs must be nonempty and distinct: {sample_ids}")

    for bam_text, sample_id, outs_text in rows:
        bam = Path(bam_text)
        outs = Path(outs_text)
        required = (
            bam,
            Path(str(bam) + ".bai"),
            outs / "filtered_feature_bc_matrix/barcodes.tsv.gz",
        )
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        with pysam.AlignmentFile(bam, "rb") as handle:
            lengths = dict(zip(handle.references, handle.lengths))
        if lengths.get("chr1") != HG19_CHR1_LENGTH:
            raise RuntimeError(
                f"BAM for {sample_id} is not chr-prefixed hg19: chr1={lengths.get('chr1')}"
            )

    panel = Path(config_value(eagle, "region_vcf"))
    if first_record_chrom(panel) != "chr1" or not Path(str(panel) + ".tbi").exists():
        raise RuntimeError("region VCF must be indexed and chr-prefixed hg19")
    eagle_map = Path(config_value(eagle, "eagle_genetic_map_file"))
    if "hg19" not in eagle_map.name or not eagle_map.exists():
        raise RuntimeError("Eagle is not configured with the hg19 genetic map")
    phasing_panel = Path(config_value(eagle, "phasing_panel"))
    for chrom in range(1, 23):
        for suffix in (".bcf", ".bcf.csi"):
            path = phasing_panel / f"chr{chrom}.genotypes{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
    with pysam.VariantFile(phasing_panel / "chr1.genotypes.bcf") as handle:
        if "chr1" not in handle.header.contigs:
            raise RuntimeError("Eagle hg19 reference BCF is not chr-prefixed")

    for config in (purity, cna, cna_unsup):
        with gzip.open(config_value(config, "geneticmap_file"), "rt") as handle:
            if handle.readline().rstrip() != "chrom\tpos\trecomb_rate\tpos_cm":
                raise RuntimeError(f"CalicoST hg19 genetic map has the wrong schema in {config}")
        hgtable = Path(config_value(config, "hgtable_file"))
        if hgtable.read_text().splitlines()[0] != "\tname2\tchrom\tcdsStart\tcdsEnd":
            raise RuntimeError(f"CalicoST hg19 gene table has the wrong schema in {config}")

    print(
        f"PASS: {sample_dir.name} has two distinct sections ({', '.join(sample_ids)}) "
        "and a consistent chr-prefixed hg19/GRCh37 reference stack"
    )


if __name__ == "__main__":
    main()
