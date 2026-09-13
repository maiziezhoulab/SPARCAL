#!/usr/bin/env python3
"""Expanded, post-freeze evaluation for the spatial-filter sidecar.

This module is deliberately separate from :mod:`evaluate`: it writes a new
output namespace and never changes the frozen v1 evaluation.  WES and
pathology are read only here, after scores and selections have been written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom

from scripts.spatial_filter_rebuild import config
from scripts.spatial_filter_rebuild import evaluate as frozen_eval


DEFAULT_MODELS = ("baseline", *config.MODEL_NAMES)

# Absolute non-inferiority margins fixed before this expanded summary was run.
# Strict comparisons (margin zero) are also emitted beside these practical
# comparisons so the conclusion never depends on a single tolerance choice.
PRACTICAL_MARGINS = {
    "silver_eligible_recall": 0.0,
    "editing_fraction": 0.005,
    "xmhc_fraction": 0.001,
    "singleton_fraction": 0.010,
    "low_depth_fraction": 0.020,
    "dense_10bp_fraction": 0.010,
    "abs_burden_umi_spearman": 0.050,
    "replicate_jaccard": 0.001,
    "pathology_f1": 0.020,
}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval without a statsmodels dependency."""

    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def selection_path(section: str, model: str) -> Path:
    return config.score_dir(section) / f"selected__{model}__matched.tsv.gz"


def selected_frame(section: str, model: str) -> pd.DataFrame:
    return pd.read_csv(selection_path(section, model), sep="\t", low_memory=False)


def candidate_universe(features: pd.DataFrame) -> pd.Series:
    return (
        features["race"].astype(str).eq("denovo")
        & features["snp"].fillna(False).astype(bool)
        & pd.to_numeric(features["total_alt"], errors="coerce").fillna(0).gt(0)
    )


def intrinsic_row(section: str, model: str, calls: pd.DataFrame) -> dict[str, object]:
    return {
        "section": section,
        "patient": str(config.SECTIONS[section]["patient"]),
        "model": model,
        "n_selected": len(calls),
        "editing_fraction": calls["rna_editing_change"].fillna(False).astype(bool).mean(),
        "xmhc_fraction": calls["xmhc"].fillna(False).astype(bool).mean(),
        "singleton_fraction": calls["positive_spots"].eq(1).mean(),
        "low_depth_fraction": calls["total_depth_cache"].lt(3).mean(),
        "dense_10bp_fraction": calls["local_density_10bp"].gt(1).mean(),
        "median_total_depth": calls["total_depth_cache"].median(),
        "median_total_alt": calls["total_alt"].median(),
        "median_positive_spots": calls["positive_spots"].median(),
    }


def evaluate_callsets(
    sections: list[str], models: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truths = frozen_eval.truth_sets()
    silver_rows: list[dict[str, object]] = []
    locus_rows: list[dict[str, object]] = []
    intrinsic_rows: list[dict[str, object]] = []
    selected: dict[tuple[str, str], set[str]] = {}
    observed_by_section: dict[str, set[str]] = {}
    eligible_by_section: dict[str, set[str]] = {}
    universe_by_section: dict[str, set[str]] = {}

    for section in sections:
        patient = str(config.SECTIONS[section]["patient"])
        features = pd.read_csv(config.score_dir(section) / "features.tsv.gz", sep="\t", low_memory=False)
        features["variant_key"] = features["variant_key"].astype(str)
        observed = set(features.loc[pd.to_numeric(features["total_alt"], errors="coerce").fillna(0).gt(0), "variant_key"])
        universe = set(features.loc[candidate_universe(features), "variant_key"])
        observed_truth = truths[patient] & observed
        eligible_truth = truths[patient] & universe
        observed_by_section[section] = observed_truth
        eligible_by_section[section] = eligible_truth
        universe_by_section[section] = universe

        memberships: dict[str, set[str]] = {}
        for model in models:
            calls = selected_frame(section, model)
            calls["variant_key"] = calls["variant_key"].astype(str)
            keys = set(calls["variant_key"])
            selected[(section, model)] = keys
            memberships[model] = keys
            hits_observed = keys & observed_truth
            hits_eligible = keys & eligible_truth
            ci_low, ci_high = wilson_interval(len(hits_eligible), len(eligible_truth))
            selection_fraction = len(keys & universe) / len(universe) if universe else np.nan
            enrichment = (
                (len(hits_eligible) / len(eligible_truth)) / selection_fraction
                if eligible_truth and selection_fraction > 0 else np.nan
            )
            p_enrich = (
                hypergeom.sf(len(hits_eligible) - 1, len(universe), len(eligible_truth), len(keys & universe))
                if universe and eligible_truth else np.nan
            )
            silver_rows.append({
                "section": section,
                "patient": patient,
                "model": model,
                "n_wes_truth": len(truths[patient]),
                "n_rna_observed_truth": len(observed_truth),
                "n_eligible_callable_truth": len(eligible_truth),
                "n_candidate_universe": len(universe),
                "n_selected": len(keys),
                "observed_wes_overlap": len(hits_observed),
                "eligible_wes_overlap": len(hits_eligible),
                "silver_observed_conditional_recall": len(hits_observed) / len(observed_truth) if observed_truth else np.nan,
                "silver_eligible_recall": len(hits_eligible) / len(eligible_truth) if eligible_truth else np.nan,
                "silver_eligible_recall_ci_low": ci_low,
                "silver_eligible_recall_ci_high": ci_high,
                "selected_universe_fraction": selection_fraction,
                "silver_enrichment_over_quota": enrichment,
                "hypergeom_enrichment_p": p_enrich,
                "gt_used_for_scoring": False,
            })
            intrinsic_rows.append(intrinsic_row(section, model, calls))

        baseline_keys = memberships["baseline"]
        for model in models:
            model_only = len((memberships[model] - baseline_keys) & eligible_truth)
            baseline_only = len((baseline_keys - memberships[model]) & eligible_truth)
            discordant = model_only + baseline_only
            p_mcnemar = binomtest(model_only, discordant, 0.5).pvalue if discordant else 1.0
            silver_rows[-len(models) + models.index(model)]["mcnemar_model_only_hits"] = model_only
            silver_rows[-len(models) + models.index(model)]["mcnemar_baseline_only_hits"] = baseline_only
            silver_rows[-len(models) + models.index(model)]["mcnemar_exact_p"] = p_mcnemar

        truth_features = features.loc[features["variant_key"].isin(observed_truth)].copy()
        keep = [
            "variant_key", "variant", "total_alt", "total_depth_cache", "positive_spots",
            "source_classification", "race", "snp", "rna_editing_change", "xmhc",
            "local_density_10bp", "vaf_log_bf", "vaf_information", "spatial_information",
            "clone_information",
        ]
        for row in truth_features[[column for column in keep if column in truth_features]].to_dict("records"):
            key = str(row["variant_key"])
            locus_rows.append({
                "section": section,
                "patient": patient,
                **row,
                "eligible_for_matched_selection": key in universe,
                **{f"selected__{model}": key in memberships[model] for model in models},
            })

    replicate_rows = []
    for patient in ("P4", "P6"):
        left, right = f"{patient}_rep1", f"{patient}_rep2"
        if left not in sections or right not in sections:
            continue
        for model in models:
            a, b = selected[(left, model)], selected[(right, model)]
            replicate_rows.append({
                "patient": patient,
                "model": model,
                "intersection": len(a & b),
                "union": len(a | b),
                "replicate_jaccard": len(a & b) / len(a | b) if a | b else np.nan,
                "shared_silver_hits": len(a & b & truths[patient]),
                "either_rep_silver_hits": len((a | b) & truths[patient]),
            })

    patient_rows = []
    for patient in ("P4", "P6"):
        patient_sections = [section for section in sections if str(config.SECTIONS[section]["patient"]) == patient]
        if not patient_sections:
            continue
        observed_truth = set().union(*(observed_by_section[section] for section in patient_sections))
        eligible_truth = set().union(*(eligible_by_section[section] for section in patient_sections))
        universe = set().union(*(universe_by_section[section] for section in patient_sections))
        for model in models:
            calls = set().union(*(selected[(section, model)] for section in patient_sections))
            hits = calls & eligible_truth
            ci_low, ci_high = wilson_interval(len(hits), len(eligible_truth))
            patient_rows.append({
                "patient": patient,
                "model": model,
                "n_sections": len(patient_sections),
                "n_unique_observed_truth": len(observed_truth),
                "n_unique_eligible_truth": len(eligible_truth),
                "n_unique_universe": len(universe),
                "n_unique_selected": len(calls),
                "unique_eligible_wes_overlap": len(hits),
                "unique_eligible_recall": len(hits) / len(eligible_truth) if eligible_truth else np.nan,
                "unique_eligible_recall_ci_low": ci_low,
                "unique_eligible_recall_ci_high": ci_high,
            })

    return (
        pd.DataFrame(silver_rows), pd.DataFrame(locus_rows), pd.DataFrame(intrinsic_rows),
        pd.DataFrame(replicate_rows), pd.DataFrame(patient_rows),
    )


def comparison_table(
    silver: pd.DataFrame, intrinsic: pd.DataFrame, viewer: pd.DataFrame, replicate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        (silver, ["section"], "silver_eligible_recall", 1),
        (intrinsic, ["section"], "editing_fraction", -1),
        (intrinsic, ["section"], "xmhc_fraction", -1),
        (intrinsic, ["section"], "singleton_fraction", -1),
        (intrinsic, ["section"], "low_depth_fraction", -1),
        (intrinsic, ["section"], "dense_10bp_fraction", -1),
        (
            viewer.loc[viewer["detector"].eq("fixed_raw")].assign(
                abs_burden_umi_spearman=lambda x: x["burden_umi_spearman"].abs()
            ),
            ["section"], "abs_burden_umi_spearman", -1,
        ),
        (replicate, ["patient"], "replicate_jaccard", 1),
        (
            viewer.loc[viewer["section"].eq("P6_rep1")].rename(columns={"f1": "pathology_f1"}),
            ["section", "detector"], "pathology_f1", 1,
        ),
    ]
    rows: list[dict[str, object]] = []
    for frame, unit_columns, metric, direction in frames:
        baseline = frame.loc[frame["model"].eq("baseline"), unit_columns + [metric]].rename(columns={metric: "baseline"})
        compared = frame.loc[~frame["model"].eq("baseline")].merge(baseline, on=unit_columns, how="inner")
        for row in compared.itertuples(index=False):
            value = float(getattr(row, metric))
            base = float(row.baseline)
            favorable_delta = direction * (value - base)
            margin = PRACTICAL_MARGINS[metric]
            rows.append({
                **{column: getattr(row, column) for column in unit_columns},
                "model": row.model,
                "metric": metric,
                "direction": "higher" if direction > 0 else "lower",
                "value": value,
                "baseline": base,
                "delta_favorable": favorable_delta,
                "practical_margin": margin,
                "strict_noninferior": favorable_delta >= -1e-12,
                "practical_noninferior": favorable_delta >= -margin - 1e-12,
                "strict_improvement": favorable_delta > 1e-12,
            })
    comparison = pd.DataFrame(rows)
    summary = (
        comparison.groupby("model")
        .agg(
            evaluated_cases=("metric", "size"),
            strict_noninferior_cases=("strict_noninferior", "sum"),
            strict_noninferior_fraction=("strict_noninferior", "mean"),
            practical_noninferior_cases=("practical_noninferior", "sum"),
            practical_noninferior_fraction=("practical_noninferior", "mean"),
            strict_improvement_cases=("strict_improvement", "sum"),
        )
        .reset_index()
        .sort_values(["practical_noninferior_fraction", "strict_noninferior_fraction"], ascending=False)
    )
    return comparison, summary


def boxplots(
    silver: pd.DataFrame, intrinsic: pd.DataFrame, viewer: pd.DataFrame,
    replicate: pd.DataFrame, models: list[str], output: Path,
) -> None:
    merged = intrinsic.merge(
        silver[["section", "model", "silver_eligible_recall"]], on=["section", "model"]
    ).merge(
        viewer.loc[viewer["detector"].eq("fixed_raw"), ["section", "model", "burden_umi_spearman"]],
        on=["section", "model"], how="left",
    )
    panels = [
        (merged, "silver_eligible_recall", "Eligible WES-silver recall"),
        (intrinsic, "editing_fraction", "RNA-editing fraction"),
        (intrinsic, "xmhc_fraction", "xMHC fraction"),
        (intrinsic, "low_depth_fraction", "Low-depth fraction (DP < 3)"),
        (intrinsic, "singleton_fraction", "Single-spot fraction"),
        (intrinsic, "dense_10bp_fraction", "Dense mismatch fraction"),
        (merged.assign(abs_burden_umi_spearman=merged["burden_umi_spearman"].abs()), "abs_burden_umi_spearman", "|burden-UMI Spearman|"),
        (replicate, "replicate_jaccard", "Replicate call-set Jaccard"),
        (viewer.loc[viewer["section"].eq("P6_rep1")].rename(columns={"f1": "pathology_f1"}), "pathology_f1", "P6 pathology F1"),
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(models)))
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for axis, (frame, column, title) in zip(axes.ravel(), panels):
        values = [pd.to_numeric(frame.loc[frame["model"].eq(model), column], errors="coerce").dropna() for model in models]
        artists = axis.boxplot(values, tick_labels=models, patch_artist=True, showmeans=True)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        for index, vals in enumerate(values, 1):
            axis.scatter(np.full(len(vals), index), vals, color="black", s=14, alpha=0.70, zorder=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=42)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle("Spatial-filter redesign: expanded frozen-baseline comparison")
    fig.tight_layout()
    fig.savefig(output / "model_comparison_full_metrics_boxplots.png", dpi=220)
    fig.savefig(output / "model_comparison_full_metrics_boxplots.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", default=list(config.SECTIONS), choices=sorted(config.SECTIONS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path, default=config.OUTPUT_ROOT / "evaluation_v2_full_metrics")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {args.output}")
    missing = [str(selection_path(section, model)) for section in args.sections for model in args.models if not selection_path(section, model).exists()]
    if missing:
        raise FileNotFoundError("Missing matched selections:\n" + "\n".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)

    silver, loci, intrinsic, replicate, patient = evaluate_callsets(args.sections, args.models)
    viewer = pd.DataFrame([
        row for section in args.sections for model in args.models for row in frozen_eval.viewer_rows(section, model)
    ])
    comparison, summary = comparison_table(silver, intrinsic, viewer, replicate)
    outputs = {
        "silver_recovery_with_uncertainty": silver,
        "silver_locus_audit": loci,
        "silver_patient_union": patient,
        "intrinsic_metrics": intrinsic,
        "viewer_tumor_clustering": viewer,
        "replicate_consistency": replicate,
        "full_metric_comparison": comparison,
        "replacement_summary": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    boxplots(silver, intrinsic, viewer, replicate, args.models, args.output)
    (args.output / "evaluation_manifest.json").write_text(json.dumps({
        "run_label": config.RUN_LABEL,
        "evaluation_version": "v2_full_metrics",
        "models": args.models,
        "sections": args.sections,
        "scoring_used_wes_or_pathology": False,
        "evaluation_uses_wes": True,
        "evaluation_uses_pathology": ["P6_rep1"],
        "wes_statistics": ["Wilson 95% interval", "hypergeometric quota enrichment", "exact paired McNemar/binomial"],
        "practical_noninferiority_margins": PRACTICAL_MARGINS,
        "output_collision_policy": "refuse non-empty output directory",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
