#!/usr/bin/env python3
"""Run the non-destructive artifact-evidence workflow for one ST section.

The known P4/P6/DCIS layouts are resolved automatically. For a new cohort with
the same output convention, pass ``--section-root``, ``--bam`` and
``--genome-build`` rather than editing code. Every run requires a new label and
writes only below ``<section-root>/artifact_evidence/<label>/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
ST_ROOT = Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets")
QF_DEFAULT = "baseQ0mapQ0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="P4_TUMOR, P6_TUMOR, DCIS, or CUSTOM.")
    parser.add_argument("--section-id", required=True, help="e.g. 1, 2, dcis1, dcis2.")
    parser.add_argument("--run-label", required=True,
                        help="New name under artifact_evidence; existing labels are refused.")
    parser.add_argument("--stage", choices=("collect", "score", "all"), default="all")
    parser.add_argument("--quality-filter", default=QF_DEFAULT)
    parser.add_argument("--max-sites-per-source", type=int, default=2_000)
    parser.add_argument("--section-root", type=Path,
                        help="Override output root for a new section (must contain spatial_filter_purity).")
    parser.add_argument("--bam", type=Path, help="Override indexed CB/UB-tagged deduplicated BAM.")
    parser.add_argument("--genome-build", choices=("hg19", "hg38"),
                        help="Override mask build for a new section.")
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=VCF_GZ",
                        help="Optional candidate source; replaces the default four sources when supplied.")
    return parser.parse_args()


def resolve_known(dataset: str, section_id: str) -> tuple[Path, Path, str]:
    dataset = dataset.upper()
    if dataset in {"P4_TUMOR", "P6_TUMOR"}:
        patient = dataset.split("_")[0]
        section_root = PROJECT / "data" / f"{patient}_tumor" / section_id
        bam = (
            ST_ROOT / "STmut_Data" / f"{patient}_Visium"
            / f"spaceranger_align_rep{section_id}_hg19" / f"{patient}_Tumor_output" / "outs"
            / "possorted_genome_bam.dedup.bam"
        )
        return section_root, bam, "hg19"
    if dataset == "DCIS":
        normalized = section_id.lower()
        if normalized.isdigit():
            normalized = f"dcis{normalized}"
        if normalized not in {"dcis1", "dcis2"}:
            raise ValueError("Known DCIS sections are dcis1 and dcis2; use CUSTOM overrides for another section.")
        number = normalized.removeprefix("dcis")
        section_root = PROJECT / "data" / normalized
        bam = (
            ST_ROOT / "spatialSNV" / "10x-Visium" / f"DCIS{number}"
            / f"spaceranger_align_DCIS{number}_hg38" / f"DCIS{number}_output" / "outs"
            / "possorted_genome_bam.dedup.bam"
        )
        return section_root, bam, "hg38"
    raise ValueError(f"No automatic layout for {dataset}; provide all CUSTOM overrides.")


def default_candidates(section_root: Path, quality_filter: str) -> list[str]:
    base = section_root / "spatial_filter_purity" / quality_filter
    return [
        f"somatic={base / 'somatic/denovo/somatic_denovo.vcf.gz'}",
        f"ambiguous={base / 'ambiguous/denovo/ambiguous_denovo.vcf.gz'}",
        f"defined={base / 'germline/defined/germline_defined.vcf.gz'}",
        f"seq_error={section_root / 'output_VCFs/SeqErrModel' / quality_filter / 'sequence_error.vcf.gz'}",
    ]


def run(command: list[str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if args.max_sites_per_source < 0:
        raise ValueError("--max-sites-per-source must be >= 0")
    try:
        auto_root, auto_bam, auto_build = resolve_known(args.dataset, args.section_id)
    except ValueError:
        if args.dataset.upper() != "CUSTOM":
            raise
        auto_root = auto_bam = auto_build = None
    section_root = args.section_root.resolve() if args.section_root else auto_root
    bam = args.bam.resolve() if args.bam else auto_bam
    genome_build = args.genome_build or auto_build
    if section_root is None or bam is None or genome_build is None:
        raise ValueError("CUSTOM requires --section-root, --bam and --genome-build.")
    if not section_root.is_dir():
        raise NotADirectoryError(section_root)
    if not bam.is_file():
        raise FileNotFoundError(bam)

    run_root = section_root / "artifact_evidence" / args.run_label
    if run_root.exists():
        raise FileExistsError(f"Run label already exists; choose a new label: {run_root}")
    features = run_root / "features"
    evidence = run_root / "evidence"
    candidates = args.candidate or default_candidates(section_root, args.quality_filter)
    for candidate in candidates:
        if "=" not in candidate or not Path(candidate.split("=", 1)[1]).is_file():
            raise FileNotFoundError(f"Candidate source missing or malformed: {candidate}")

    masks = PROJECT / "resources" / "artifact_masks" / "SComatic"
    editing = masks / f"AllEditingSites.{genome_build}.bed.gz"
    pon = masks / f"PoN.scRNAseq.{genome_build}.bed.gz"
    if not editing.is_file() or not pon.is_file():
        raise FileNotFoundError(f"Missing masks for {genome_build}: {editing}, {pon}")
    run_root.mkdir(parents=True)
    config = {
        "dataset": args.dataset,
        "section_id": args.section_id,
        "section_root": str(section_root),
        "bam": str(bam),
        "genome_build": genome_build,
        "quality_filter": args.quality_filter,
        "candidates": candidates,
        "editing_bed": str(editing),
        "pon_bed": str(pon),
        "max_sites_per_source": args.max_sites_per_source,
        "baseline_contract": str(PROJECT / "data/_frozen_baselines/2026-07-15_current_pipeline"),
    }
    with (run_root / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    collector = PROJECT / "scripts/postanalyze/collect_spatial_artifact_features.py"
    scorer = PROJECT / "scripts/postanalyze/score_artifact_evidence.py"
    if args.stage in {"collect", "all"}:
        command = [sys.executable, str(collector), "--bam", str(bam), "--out-dir", str(features),
                   "--max-sites-per-source", str(args.max_sites_per_source), "--editing-bed", str(editing),
                   "--pon-bed", str(pon)]
        for candidate in candidates:
            command.extend(["--candidates", candidate])
        run(command)
    if args.stage in {"score", "all"}:
        if not (features / "site_features.tsv.gz").is_file():
            raise FileNotFoundError(f"Collect features first: {features / 'site_features.tsv.gz'}")
        run([sys.executable, str(scorer), "--input-dir", str(features), "--out-dir", str(evidence)])
    print(f"Completed non-destructive artifact-evidence run: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
