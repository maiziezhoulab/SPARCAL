#!/usr/bin/env python3
"""Compare P4/P6 SPARCAL calls with and without CalicoST-derived evidence.

The full analysis uses every informative CalicoST input available for a section.
The ablation removes tumor purity, clone labels, CNV consistency, and the
purity-by-spatial proxy; only SPARCAL's spatial-clustering vote (zeta) remains.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr


PROJECT_ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
CALICOST_ROOT = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
FULL_TAG = "baseQ0mapQ0_full_calicost"
ABLATION_TAG = "baseQ0mapQ0_ablation_no_calicost"
SAMPLES = (
    ("P4", "1", "clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    ("P4", "2", "clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    ("P6", "1", "clone3_rectangle0_w1.0", "cnv_seglevel.tsv"),
    ("P6", "2", "clone3_rectangle0_w1.0", "cnv_diploid_seglevel.tsv"),
)


def variant_set(path: Path) -> Set[str]:
    frame = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    return {f"{str(chrom).removeprefix('chr')}_{int(pos)}"
            for chrom, pos in zip(frame["chrom"], frame["pos"])}


def barcode_counts(category_dir: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for path in category_dir.glob("*.txt"):
        if path.name.startswith(("germline_", "somatic_", "ambiguous_")):
            continue
        try:
            frame = pd.read_csv(path, sep="\t")
            result[path.stem] = len(frame)
        except pd.errors.EmptyDataError:
            result[path.stem] = 0
    return result


def cnv_index(path: Path) -> Dict[str, Tuple[list, list]]:
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [str(col).replace(" ", "_") for col in frame.columns]
    cn_cols = [
        col for col in frame.columns
        if col.startswith("clone") and col.endswith(("_A", "_B"))
        and not col.startswith("clone0_")
    ]
    if not cn_cols:
        cn_cols = [
            col for col in frame.columns
            if col.startswith("clone") and col.endswith(("_A", "_B"))
        ]
    index: Dict[str, Tuple[list, list]] = {}
    for chrom, group in frame.groupby(frame["CHR"].astype(str).str.removeprefix("chr")):
        records = []
        for row in group.sort_values("START").itertuples(index=False):
            values = [float(getattr(row, col)) for col in cn_cols]
            pairs = list(zip(values[0::2], values[1::2]))
            altered = any(a != 1 or b != 1 for a, b in pairs)
            loh = any((a == 0 or b == 0) and a + b > 0 for a, b in pairs)
            records.append((int(row.START), int(row.END), altered, loh))
        index[str(chrom)] = ([record[0] for record in records], records)
    return index


def cnv_status(variant: str, index: Dict[str, Tuple[list, list]]) -> Tuple[bool, bool, bool]:
    chrom, pos_text = variant.split("_", 1)
    if chrom not in index:
        return False, False, False
    starts, records = index[chrom]
    pos = int(pos_text)
    location = bisect.bisect_right(starts, pos) - 1
    if location < 0:
        return False, False, False
    start, end, altered, loh = records[location]
    if not (start <= pos <= end):
        return False, False, False
    return True, altered, loh


def safe_median(values: Iterable[float]) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("|", "\|")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "p4p6_calicost_ablation_20260907",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    concordance_rows = []
    partition_rows = []
    burden_rows = []
    burden_summary_rows = []

    for cohort, section, clone_dir, cnv_name in SAMPLES:
        label = f"{cohort}_rep{section}"
        section_root = PROJECT_ROOT / "data" / f"{cohort}_tumor" / section
        full_root = section_root / "spatial_filter_purity" / FULL_TAG
        ablation_root = section_root / "spatial_filter_purity" / ABLATION_TAG
        full_matrix = (
            section_root / "matrix"
            / f"{cohort}_TUMOR_{section}_SPARCAL_full_calicost_somatic_matrix.pkl"
        )
        cnv_path = (
            CALICOST_ROOT / f"{cohort}_sec{section}" / "calicost"
            / clone_dir / cnv_name
        )
        required = [
            full_root / "germline" / "germline_variants.txt",
            full_root / "somatic" / "somatic_variants.txt",
            ablation_root / "germline" / "germline_variants.txt",
            ablation_root / "somatic" / "somatic_variants.txt",
            full_root / "all_variant_scores.txt",
            full_matrix,
            cnv_path,
        ]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError("Missing ablation input(s):\n  " + "\n  ".join(missing))

        callsets = {}
        for classification in ("germline", "somatic"):
            full = variant_set(full_root / classification / f"{classification}_variants.txt")
            ablation = variant_set(
                ablation_root / classification / f"{classification}_variants.txt"
            )
            callsets[classification] = (full, ablation)
            intersection = full & ablation
            union = full | ablation
            concordance_rows.append({
                "sample": label,
                "classification": classification,
                "full_count": len(full),
                "ablation_count": len(ablation),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "full_only": len(full - ablation),
                "ablation_only": len(ablation - full),
                "full_retained_pct": 100 * len(intersection) / len(full) if full else 100.0,
            })

        full_somatic, ablation_somatic = callsets["somatic"]
        partitions = {
            "shared": full_somatic & ablation_somatic,
            "full_only": full_somatic - ablation_somatic,
            "ablation_only": ablation_somatic - full_somatic,
        }
        scores = pd.read_csv(full_root / "all_variant_scores.txt", sep="\t")
        scores["variant"] = (
            scores["variant"].astype(str).str.removeprefix("chr")
        )
        scores = scores.set_index("variant")
        intervals = cnv_index(cnv_path)

        for partition, variants in partitions.items():
            for variant in sorted(variants):
                covered, altered, loh = cnv_status(variant, intervals)
                score_row = scores.loc[variant] if variant in scores.index else None
                partition_rows.append({
                    "sample": label,
                    "partition": partition,
                    "variant": variant,
                    "cnv_covered": covered,
                    "cnv_altered": altered if covered else pd.NA,
                    "loh": loh if covered else pd.NA,
                    "purity_correlation": (
                        pd.to_numeric(score_row.get("f_purity_correlation"), errors="coerce")
                        if score_row is not None else np.nan
                    ),
                    "cnv_consistency": (
                        pd.to_numeric(score_row.get("f_cnv_consistency"), errors="coerce")
                        if score_row is not None else np.nan
                    ),
                    "spatial_clustering": (
                        pd.to_numeric(score_row.get("f_spatial_clustering"), errors="coerce")
                        if score_row is not None else np.nan
                    ),
                })

        full_counts = barcode_counts(full_root / "somatic")
        ablation_counts = barcode_counts(ablation_root / "somatic")
        # Include spots with zero somatic calls in both arms. Per-barcode text
        # outputs exist only for nonempty spots, whereas matrix rows retain the
        # complete spatial spot universe.
        matrix_barcodes = pd.read_pickle(full_matrix).index.astype(str)
        barcodes = sorted(set(matrix_barcodes) | set(full_counts) | set(ablation_counts))
        full_values = np.asarray([full_counts.get(bc, 0) for bc in barcodes])
        ablation_values = np.asarray([ablation_counts.get(bc, 0) for bc in barcodes])
        rho = spearmanr(full_values, ablation_values).statistic if barcodes else np.nan
        burden_summary_rows.append({
            "sample": label,
            "spots": len(barcodes),
            "full_median_somatic_burden": float(np.median(full_values)) if barcodes else np.nan,
            "ablation_median_somatic_burden": float(np.median(ablation_values)) if barcodes else np.nan,
            "median_delta_full_minus_ablation": (
                float(np.median(full_values - ablation_values)) if barcodes else np.nan
            ),
            "spearman_rho": rho,
        })
        burden_rows.extend({
            "sample": label,
            "barcode": barcode,
            "full_somatic_burden": int(full_counts.get(barcode, 0)),
            "ablation_somatic_burden": int(ablation_counts.get(barcode, 0)),
            "delta_full_minus_ablation": int(
                full_counts.get(barcode, 0) - ablation_counts.get(barcode, 0)
            ),
        } for barcode in barcodes)

    concordance = pd.DataFrame(concordance_rows)
    partitions = pd.DataFrame(partition_rows)
    burdens = pd.DataFrame(burden_rows)
    burden_summary = pd.DataFrame(burden_summary_rows)

    partition_summary_rows = []
    for (sample, partition), group in partitions.groupby(["sample", "partition"]):
        covered = group[group["cnv_covered"]]
        partition_summary_rows.append({
            "sample": sample,
            "partition": partition,
            "variants": len(group),
            "cnv_covered": len(covered),
            "cnv_altered_fraction": (
                float(pd.to_numeric(covered["cnv_altered"]).mean()) if len(covered) else np.nan
            ),
            "loh_fraction": (
                float(pd.to_numeric(covered["loh"]).mean()) if len(covered) else np.nan
            ),
            "median_purity_correlation": safe_median(group["purity_correlation"]),
            "median_cnv_consistency": safe_median(group["cnv_consistency"]),
            "median_spatial_clustering": safe_median(group["spatial_clustering"]),
        })
    partition_summary = pd.DataFrame(partition_summary_rows)

    contrast_rows = []
    for sample, sample_frame in partitions.groupby("sample"):
        for outcome in ("cnv_altered", "loh"):
            groups = {}
            for partition in ("full_only", "ablation_only"):
                values = sample_frame.loc[
                    sample_frame["partition"] == partition, outcome
                ].dropna().astype(bool)
                groups[partition] = (int(values.sum()), int((~values).sum()))
            table = [groups["full_only"], groups["ablation_only"]]
            odds_ratio, pvalue = fisher_exact(table)
            contrast_rows.append({
                "sample": sample,
                "outcome": outcome,
                "full_only_positive": table[0][0],
                "full_only_negative": table[0][1],
                "ablation_only_positive": table[1][0],
                "ablation_only_negative": table[1][1],
                "odds_ratio_full_vs_ablation_only": odds_ratio,
                "fisher_pvalue": pvalue,
            })
    contrasts = pd.DataFrame(contrast_rows)

    concordance.to_csv(args.output_dir / "callset_concordance.tsv", sep="\t", index=False)
    partitions.to_csv(args.output_dir / "somatic_partition_cnv.tsv", sep="\t", index=False)
    partition_summary.to_csv(
        args.output_dir / "somatic_partition_summary.tsv", sep="\t", index=False
    )
    contrasts.to_csv(
        args.output_dir / "somatic_cnv_fisher_contrasts.tsv", sep="\t", index=False
    )
    burdens.to_csv(args.output_dir / "spot_burden_comparison.tsv", sep="\t", index=False)
    burden_summary.to_csv(
        args.output_dir / "spot_burden_summary.tsv", sep="\t", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    somatic = concordance[concordance["classification"] == "somatic"]
    axes[0, 0].bar(somatic["sample"], somatic["jaccard"], color="#4472C4")
    axes[0, 0].set(title="Somatic-call Jaccard", ylim=(0, 1), ylabel="Jaccard")
    axes[0, 0].tick_params(axis="x", rotation=25)

    axes[0, 1].bar(
        somatic["sample"],
        100 * somatic["full_only"] / somatic["full_count"].replace(0, np.nan),
        color="#C55A11",
    )
    axes[0, 1].set(title="Full calls removed by ablation", ylabel="% of full somatic calls")
    axes[0, 1].tick_params(axis="x", rotation=25)

    pivot = partition_summary.pivot(
        index="sample", columns="partition", values="cnv_altered_fraction"
    )
    pivot.reindex(columns=["shared", "full_only", "ablation_only"]).plot(
        kind="bar", ax=axes[1, 0], color=["#70AD47", "#C55A11", "#A5A5A5"]
    )
    axes[1, 0].set(title="Fraction in altered CalicoST CNV segments", ylabel="Fraction")
    axes[1, 0].legend(title="Call partition", fontsize=8)

    axes[1, 1].bar(
        burden_summary["sample"], burden_summary["spearman_rho"], color="#5B9BD5"
    )
    axes[1, 1].set(title="Per-spot somatic-burden concordance", ylabel="Spearman rho", ylim=(0, 1))
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(args.output_dir / "calicost_ablation_summary.pdf")
    fig.savefig(args.output_dir / "calicost_ablation_summary.png", dpi=200)
    plt.close(fig)

    report_lines = [
        "# P4/P6 CalicoST evidence ablation",
        "",
        "Full mode uses informative tumor-purity correlation/proxy plus spatial clustering "
        "and clone-resolved CNV consistency. The ablation deliberately uses spatial "
        "clustering (zeta) only.",
        "",
        "P4 rep2 has no informative CalicoST purity values, so full mode correctly "
        "uses spatial clustering plus CNV consistency for that section.",
        "",
        "## Somatic callset comparison",
        "",
        markdown_table(concordance[concordance["classification"] == "somatic"]),
        "",
        "## Per-spot burden comparison",
        "",
        markdown_table(burden_summary),
        "",
        "## CNV enrichment by call partition",
        "",
        markdown_table(partition_summary),
        "",
        "## Full-only versus ablation-only CNV/LOH contrasts",
        "",
        markdown_table(contrasts),
        "",
        "Diagnostic: the current theta score fixes copy-neutral segments at 0.5, while "
        "altered segments begin at within-clone variant prevalence. Sparse variants can "
        "therefore score below the copy-neutral baseline. A full-only odds ratio below 1 "
        "is evidence of this scoring-direction bias, not evidence that CNV and SNVs are "
        "biologically incompatible.",
        "",
        "Any full-only enrichment in altered/LOH segments is internal supporting "
        "evidence, not independent validation, because the full caller used CalicoST CNV.",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report_lines))
    manifest = {
        "full_tag": FULL_TAG,
        "ablation_tag": ABLATION_TAG,
        "samples": [f"{cohort}_rep{section}" for cohort, section, _, _ in SAMPLES],
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(concordance.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
