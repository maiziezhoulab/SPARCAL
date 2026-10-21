#!/usr/bin/env python3
"""Aggregate the four low-BAF UPV read-level purity-tertile experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
INPUT = PROJECT / "data/upv_delta_baf_2026-08-23"
GMM = {
    "P4": PROJECT / "data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/denovo/gmm_subfilter/upv_gmm_classification.tsv",
    "P6": PROJECT / "data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/denovo/gmm_subfilter/upv_gmm_classification.tsv",
    "DCIS1": PROJECT / "data/dcis1/spatial_filter_purity/baseQ0mapQ0/germline/denovo/gmm_subfilter/upv_gmm_classification.tsv",
    "DCIS2": PROJECT / "data/dcis2/spatial_filter_purity/baseQ0mapQ0/germline/denovo/gmm_subfilter/upv_gmm_classification.tsv",
}


def upv_metrics_path():
    final = PROJECT / "data/upv_baf_assessment_2026-08-23/upv_variant_metrics.csv.gz"
    interim = PROJECT / "data/upv_baf_assessment_mid_2026-08-23/upv_variant_metrics.csv.gz"
    return final if final.exists() else interim


def summarize(sample, frame, n_scanned, class_name):
    group = frame if class_name == "all" else frame[frame.current_class.eq(class_name)]
    x = group.delta_baf_exact
    min_group_dp = group[["t_dp", "n_dp"]].min(axis=1)
    def rank_corr(a, b):
        return (
            float(spearmanr(a, b).statistic)
            if len(a) > 2 and pd.Series(a).nunique() > 1 and pd.Series(b).nunique() > 1
            else np.nan
        )
    return {
        "sample": sample,
        "current_class": class_name,
        "n_scanned_low_baf": n_scanned,
        "n_evaluable": len(group),
        "fraction_evaluable": len(group) / n_scanned if n_scanned else np.nan,
        "delta_baf_mean": x.mean(),
        "delta_baf_q05": x.quantile(0.05),
        "delta_baf_q25": x.quantile(0.25),
        "delta_baf_median": x.median(),
        "delta_baf_q75": x.quantile(0.75),
        "delta_baf_q95": x.quantile(0.95),
        "n_negative_delta_lt_minus_005": int((x < -0.05).sum()),
        "n_near_zero_minus_005_to_005": int(x.between(-0.05, 0.05).sum()),
        "n_positive_delta_gt_005": int((x > 0.05).sum()),
        "fraction_positive_delta_gt_005": float((x > 0.05).mean()) if len(x) else np.nan,
        "spearman_delta_vs_pseudobulk_baf": (
            rank_corr(x, group.baf)
        ),
        "spearman_delta_vs_min_group_dp": (
            rank_corr(x, min_group_dp)
        ),
    }


def main():
    combined, summary = [], []
    for sample in ["P4", "P6", "DCIS1", "DCIS2"]:
        frame = pd.read_csv(INPUT / sample / "upv_delta_baf.tsv", sep="\t")
        # Recompute from integer counts so results are independent of the text
        # precision used by the exploratory runs.
        frame["delta_baf_exact"] = (
            frame.t_alt / frame.t_dp - frame.n_alt / frame.n_dp
        )
        frame["delta_category"] = np.select(
            [frame.delta_baf_exact < -0.05, frame.delta_baf_exact > 0.05],
            ["negative", "positive"], default="near_zero",
        )
        frame["sample"] = sample
        gmm = pd.read_csv(GMM[sample], sep="\t")
        n_scanned = int((gmm.baf <= 0.35).sum())
        for class_name in ["all"] + sorted(frame.current_class.unique()):
            summary.append(summarize(sample, frame, n_scanned, class_name))
        combined.append(frame)

    combined = pd.concat(combined, ignore_index=True)
    summary = pd.DataFrame(summary)
    metrics = pd.read_csv(upv_metrics_path())
    evidence_columns = [
        "sample", "chrom_pos_ref_alt", "artifact_masked", "editing_compatible",
        "purity_corr_burden_adjusted", "purity_adjusted_q",
        "normal_wes_callable", "normal_wes_alt_evidence",
    ]
    combined = combined.merge(
        metrics[evidence_columns], on=["sample", "chrom_pos_ref_alt"],
        how="left", validate="one_to_one",
    )
    if combined.artifact_masked.isna().any():
        raise RuntimeError("delta-BAF records failed to match the UPV evidence table")
    evidence = combined.groupby(["sample", "delta_category"], as_index=False).agg(
        n=("chrom_pos_ref_alt", "size"),
        artifact_mask_fraction=("artifact_masked", "mean"),
        editing_compatible_fraction=("editing_compatible", "mean"),
        median_burden_adjusted_purity_corr=("purity_corr_burden_adjusted", "median"),
        adjusted_q_lt_005_n=("purity_adjusted_q", lambda x: int((x < 0.05).sum())),
        normal_wes_callable_n=("normal_wes_callable", lambda x: int((x == True).sum())),
        normal_wes_alt_evidence_n=("normal_wes_alt_evidence", lambda x: int((x == True).sum())),
    )
    combined.to_csv(INPUT / "upv_delta_baf_all_variants.csv.gz", index=False, compression="gzip")
    summary.to_csv(INPUT / "upv_delta_baf_summary.csv", index=False)
    evidence.to_csv(INPUT / "upv_delta_baf_evidence_by_category.csv", index=False)

    all_rows = summary[summary.current_class.eq("all")]
    lines = [
        "# Direct read-level UPV delta-BAF assessment", "",
        "Low-pseudobulk-BAF UPVs (BAF <= 0.35) were tested by pooling CB-tagged "
        "reference and alternate reads separately across the lowest and highest "
        "CalicoST purity tertiles. Delta-BAF is high-purity VAF minus low-purity VAF. "
        "Mapping quality >=20, base quality >=13, and at least five ref+alt reads in "
        "each tertile were required.", "",
        "| sample | scanned | evaluable | median delta-BAF | negative (<-0.05) | near zero | positive (>0.05) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows.itertuples():
        lines.append(
            f"| {row.sample} | {row.n_scanned_low_baf:,} | {row.n_evaluable:,} | "
            f"{row.delta_baf_median:+.4f} | {row.n_negative_delta_lt_minus_005:,} | "
            f"{row.n_near_zero_minus_005_to_005:,} | {row.n_positive_delta_gt_005:,} |"
        )
    lines += [
        "", "## Interpretation", "",
        "The read-level result does not reveal a reproducible tumor-enriched UPV class. "
        "P4, DCIS1 and DCIS2 have medians near zero; P6 is positive but has only 28 "
        "evaluable sites. In DCIS1, 1,744 of 1,757 evaluable sites carry the optional "
        "GMM somatic-candidate label, yet those 1,744 sites divide into 539 negative, "
        "584 near-zero and 621 positive values (median +0.0196). The GMM label is thus "
        "principally a low-pseudobulk-BAF label, not a direct tumor-enrichment result.", "",
        "This is an exploratory magnitude diagnostic, not a hypothesis test: reads "
        "within a spot and spots within a tissue are not independent, and CalicoST "
        "purity was estimated from the same RNA assay. Allele-specific expression, "
        "copy number, mapping bias and unequal tertile coverage can change pooled VAF. "
        "The analysis therefore complements, but does not replace, matched-normal WES "
        "and spot-level burden-adjusted association analyses.", "",
        "## Files", "",
        "- `upv_delta_baf_all_variants.csv.gz`: combined exact count-derived values.",
        "- `upv_delta_baf_summary.csv`: sample/class summaries.",
        "- `upv_delta_baf_evidence_by_category.csv`: artifact, editing, purity-association and WES evidence by delta direction.",
        "- `{sample}/upv_delta_baf.tsv`: raw per-sample counts and pooled VAFs.",
        f"- Generator: `{Path(__file__).resolve()}`.",
        f"- Read scanner: `{PROJECT / 'scripts/6_spatial_filter/upv_delta_baf_experiment.py'}`.",
    ]
    (INPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
