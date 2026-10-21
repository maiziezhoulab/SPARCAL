#!/usr/bin/env python3
"""Audit SPARCAL genotype-class predictions at non-1KGP loci against matched-normal WES.

This is deliberately a *genotyping* audit, not a somatic benchmark.  It retains high-confidence
normal-WES 0/1 and 1/1 SNVs, finds exact REF/ALT matches in the classifier prediction VCF, removes
sites in the SPARCAL defined/1KGP VCF, and compares both the raw RNA GT and the neural-network class.

The WES normal and spatial RNA are different assays; tumour LOH and RNA ASE can make a real
discordance.  Therefore this output is a calibration/sanity check and should be stratified by normal
spots or DNA/CNV state before being used as a headline accuracy estimate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


VariantKey = Tuple[str, str, str, str]
VALID_TRUTH_GTS = {"0/1", "1/1"}
CLASS_TO_GT = {"heterozygous": "0/1", "homozygous": "1/1", "no_variance": "0/0"}


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def parse_info(info: str, key: str) -> Optional[str]:
    prefix = f"{key}="
    for item in info.split(";"):
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def parse_format(format_field: str, sample_field: str) -> Dict[str, str]:
    keys = format_field.split(":")
    values = sample_field.split(":")
    return dict(zip(keys, values))


def parse_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ".", ""):
        return None
    try:
        return int(value.split(",")[0])
    except ValueError:
        return None


def iter_records(path: Path) -> Iterator[Tuple[VariantKey, list[str]]]:
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 10 and "," not in fields[4]:
                yield (fields[0], fields[1], fields[3], fields[4]), fields


def load_defined_keys(path: Path) -> set[VariantKey]:
    return {key for key, _ in iter_records(path)}


def load_truth(path: Path, min_dp: int, min_gq: int) -> Dict[VariantKey, str]:
    truth: Dict[VariantKey, str] = {}
    for key, fields in iter_records(path):
        if fields[6] not in {"PASS", "."}:
            continue
        fmt = parse_format(fields[8], fields[9])
        gt = fmt.get("GT", ".").replace("|", "/")
        if gt not in VALID_TRUTH_GTS:
            continue
        dp = parse_int(fmt.get("DP"))
        gq = parse_int(fmt.get("GQ"))
        if dp is None or gq is None or dp < min_dp or gq < min_gq:
            continue
        truth[key] = gt
    return truth


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["metric", "value"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-vcf", required=True, type=Path)
    parser.add_argument("--truth-vcf", required=True, type=Path,
                        help="Matched-normal WES VCF with GT, DP, and GQ.")
    parser.add_argument("--defined-vcf", required=True, type=Path,
                        help="SPARCAL germline_defined/1KGP VCF to exclude.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--class-info-key", default="NEURAL_NETWORK_CLASS")
    parser.add_argument("--min-dp", type=int, default=10)
    parser.add_argument("--min-gq", type=int, default=20)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    defined = load_defined_keys(args.defined_vcf)
    truth = load_truth(args.truth_vcf, args.min_dp, args.min_gq)

    seen = set()
    rows = []
    for key, fields in iter_records(args.prediction_vcf):
        if key in seen or key not in truth or key in defined:
            continue
        seen.add(key)
        fmt = parse_format(fields[8], fields[9])
        raw_gt = fmt.get("GT", ".").replace("|", "/")
        pred_class = parse_info(fields[7], args.class_info_key)
        pred_gt = CLASS_TO_GT.get(pred_class or "", "unknown")
        rows.append({
            "chrom": key[0], "pos": key[1], "ref": key[2], "alt": key[3],
            "normal_wes_gt": truth[key], "raw_rna_gt": raw_gt,
            "model_class": pred_class or "missing", "model_gt": pred_gt,
            "raw_exact": int(raw_gt == truth[key]),
            "model_exact": int(pred_gt == truth[key]),
        })

    per_locus = args.outdir / "per_locus.csv"
    with per_locus.open("w", newline="") as handle:
        fieldnames = [
            "chrom", "pos", "ref", "alt", "normal_wes_gt", "raw_rna_gt",
            "model_class", "model_gt", "raw_exact", "model_exact",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    raw_confusion = Counter((str(row["normal_wes_gt"]), str(row["raw_rna_gt"])) for row in rows)
    model_confusion = Counter((str(row["normal_wes_gt"]), str(row["model_class"])) for row in rows)
    n = len(rows)
    raw_exact = sum(int(row["raw_exact"]) for row in rows)
    model_exact = sum(int(row["model_exact"]) for row in rows)
    summary_rows: list[dict[str, object]] = [
        {"metric": "n_high_confidence_normal_wes_variants", "value": len(truth)},
        {"metric": "n_outside_defined_1kgp_matched_in_prediction_vcf", "value": n},
        {"metric": "raw_rna_gt_exact", "value": raw_exact},
        {"metric": "raw_rna_gt_exact_rate", "value": f"{raw_exact / n:.6f}" if n else "NA"},
        {"metric": "model_class_exact", "value": model_exact},
        {"metric": "model_class_exact_rate", "value": f"{model_exact / n:.6f}" if n else "NA"},
    ]
    summary_rows.extend(
        {"metric": f"raw_confusion:{truth_gt}->{called_gt}", "value": count}
        for (truth_gt, called_gt), count in sorted(raw_confusion.items())
    )
    summary_rows.extend(
        {"metric": f"model_confusion:{truth_gt}->{called_class}", "value": count}
        for (truth_gt, called_class), count in sorted(model_confusion.items())
    )
    write_summary(args.outdir / "summary.csv", summary_rows)

    print(f"High-confidence normal-WES variants: {len(truth):,}")
    print(f"Outside-defined/1KGP exact-allele matches: {n:,}")
    print(f"Raw RNA GT agreement: {raw_exact}/{n} ({raw_exact / n:.1%})" if n else "Raw RNA GT agreement: NA")
    print(f"Model-class agreement: {model_exact}/{n} ({model_exact / n:.1%})" if n else "Model-class agreement: NA")
    print(f"Wrote {per_locus} and {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
