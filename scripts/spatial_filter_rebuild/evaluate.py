#!/usr/bin/env python3
"""Independent post-freeze evaluation and comparison boxplots.

WES and pathology are loaded only here, after score/selection files exist.
Neither source can affect model scores, thresholds, or matched set sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from scripts.spatial_filter_rebuild import config
from scripts.spatial_filter_rebuild.materialize import compatible_matrix_path


MODELS = ("baseline", *config.MODEL_NAMES)


def variant_key(chromosome, position, ref, alt) -> str:
    chromosome = str(chromosome).removeprefix("chr")
    return f"{chromosome}_{int(position)}_{str(ref).upper()}_{str(alt).upper()}"


def truth_sets() -> dict[str, set[str]]:
    truth = pd.read_csv(config.WES_TRUTH, sep="\t")
    truth["variant_key"] = [
        variant_key(*row) for row in truth[["chrom_hg19", "pos_hg19", "ref", "alt"]].itertuples(index=False, name=None)
    ]
    return {
        patient: set(truth.loc[truth["sample"].eq(patient), "variant_key"])
        for patient in sorted(truth["sample"].unique())
    }


def baseline_viewer_config(section: str) -> Path:
    cfg = config.SECTIONS[section]
    study = f"{cfg['patient']}_rep{cfg['rep']}_SPARCAL_somatic_ad_vaf_shadow_v1"
    return (
        config.BENCHMARKING / "viewer/dist/studies/ad_vaf_shadow_v1"
        / study / f"{study}.config"
    )


def load_study(section: str):
    sys.path.insert(0, str(config.BENCHMARKING / "viewer"))
    from sparcal_viewer.config import load_config  # pylint: disable=import-outside-toplevel
    from sparcal_viewer.data import StudyData  # pylint: disable=import-outside-toplevel

    return StudyData(load_config(str(baseline_viewer_config(section))))


def detection_metrics(predicted: set[str], truth: set[str], universe: list[str]) -> dict[str, float]:
    p = predicted & set(universe)
    t = truth & set(universe)
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "binary_ari": adjusted_rand_score(
            [barcode in t for barcode in universe], [barcode in p for barcode in universe]
        ),
    }


def viewer_rows(section: str, model: str) -> list[dict[str, object]]:
    matrix = pd.read_pickle(compatible_matrix_path(section, model, "matched"))
    study = load_study(section)
    matrix = matrix.reindex(index=study.positions.index, fill_value=0)
    study.matrix = matrix
    study._total_counts = (matrix > 0).sum(axis=0)
    study._adj = None
    burden = study.per_spot_burden().astype(float)
    coverage = study.coverage.reindex(burden.index) if study.coverage is not None else pd.Series(np.nan, index=burden.index)
    finite = coverage.notna() & np.isfinite(burden)
    rho = spearmanr(burden[finite], coverage[finite]).statistic if finite.sum() >= 3 else np.nan
    pathology = set()
    if section == "P6_rep1":
        groups = pd.read_csv(config.P6_PATHOLOGY_STUDY / "tumor_groups.csv", dtype=str)
        normalized = groups["profile"].str.replace(" ", "", regex=False).str.lower()
        pathology = set(groups.loc[normalized.eq("groundtruth"), "barcode"])

    rows = []
    for mode, normalize in (("fixed_raw", False), ("fixed_coverage_residual", True)):
        result = study.auto_tumor_regions(
            seed_pct=90, grow_pct=80, min_size=5, normalize=normalize, split_depth=0.40
        )
        predicted = {barcode for region in result["regions"] for barcode in region}
        row: dict[str, object] = {
            "section": section,
            "model": model,
            "detector": mode,
            "n_spots": len(study.positions),
            "n_variants": matrix.shape[1],
            "n_regions": len(result["regions"]),
            "n_predicted_tumor_spots": len(predicted),
            "predicted_tumor_fraction": len(predicted) / len(study.positions),
            "burden_umi_spearman": rho,
            "parameters_frozen": True,
        }
        if pathology:
            row.update(detection_metrics(predicted, pathology, list(study.positions.index)))
        rows.append(row)
    return rows


def selected_frame(section: str, model: str) -> pd.DataFrame:
    return pd.read_csv(
        config.score_dir(section) / f"selected__{model}__matched.tsv.gz", sep="\t", low_memory=False
    )


def evaluate_sets(sections: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truths = truth_sets()
    silver_rows = []
    intrinsic_rows = []
    selected: dict[tuple[str, str], set[str]] = {}
    for section in sections:
        patient = str(config.SECTIONS[section]["patient"])
        features = pd.read_csv(
            config.score_dir(section) / "features.tsv.gz",
            sep="\t", usecols=["variant_key", "total_alt"], low_memory=False,
        )
        observed = set(features.loc[features["total_alt"] > 0, "variant_key"].astype(str))
        callable_truth = truths[patient] & observed
        for model in MODELS:
            calls = selected_frame(section, model)
            keys = set(calls["variant_key"].astype(str))
            selected[(section, model)] = keys
            hits = keys & truths[patient]
            conditional_hits = keys & callable_truth
            silver_rows.append({
                "section": section,
                "patient": patient,
                "model": model,
                "n_wes_truth": len(truths[patient]),
                "n_rna_observed_truth": len(callable_truth),
                "wes_overlap": len(hits),
                "callable_wes_overlap": len(conditional_hits),
                "silver_conditional_recall": len(conditional_hits) / len(callable_truth) if callable_truth else np.nan,
                "gt_used_for_scoring": False,
            })
            intrinsic_rows.append({
                "section": section,
                "patient": patient,
                "model": model,
                "n_selected": len(calls),
                "editing_fraction": calls["rna_editing_change"].fillna(False).astype(bool).mean(),
                "xmhc_fraction": calls["xmhc"].fillna(False).astype(bool).mean(),
                "singleton_fraction": calls["positive_spots"].eq(1).mean(),
                "low_depth_fraction": calls["total_depth_cache"].lt(3).mean(),
                "dense_10bp_fraction": calls["local_density_10bp"].gt(1).mean(),
                "median_positive_spots": calls["positive_spots"].median(),
            })

    replicate_rows = []
    for patient in ("P4", "P6"):
        left, right = f"{patient}_rep1", f"{patient}_rep2"
        if left not in sections or right not in sections:
            continue
        for model in MODELS:
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
    return pd.DataFrame(silver_rows), pd.DataFrame(intrinsic_rows), pd.DataFrame(replicate_rows)


def boxplots(silver: pd.DataFrame, intrinsic: pd.DataFrame, viewer: pd.DataFrame, replicate: pd.DataFrame, output: Path) -> None:
    order = list(MODELS)
    merged = intrinsic.merge(
        silver[["section", "model", "silver_conditional_recall"]], on=["section", "model"]
    ).merge(
        viewer.loc[viewer["detector"].eq("fixed_raw"), ["section", "model", "burden_umi_spearman", "predicted_tumor_fraction"]],
        on=["section", "model"], how="left",
    )
    panels = [
        (merged, "silver_conditional_recall", "WES silver callable recall", True),
        (merged, "editing_fraction", "RNA-editing fraction", False),
        (merged, "singleton_fraction", "Single-spot fraction", False),
        (merged, "dense_10bp_fraction", "Dense mismatch fraction", False),
        (merged.assign(abs_burden_umi=merged["burden_umi_spearman"].abs()), "abs_burden_umi", "|burden-UMI Spearman|", False),
        (replicate, "replicate_jaccard", "Replicate call-set Jaccard", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = ["#777777", "#4C78A8", "#72B7B2", "#F2CF5B", "#E45756", "#B279A2"]
    for axis, (frame, column, title, _) in zip(axes.ravel(), panels):
        values = [pd.to_numeric(frame.loc[frame["model"].eq(model), column], errors="coerce").dropna() for model in order]
        artists = axis.boxplot(values, labels=order, patch_artist=True, showmeans=True)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for index, vals in enumerate(values, 1):
            axis.scatter(np.full(len(vals), index), vals, color="black", s=14, alpha=0.7, zorder=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Spatial-filter redesign: frozen baseline vs training-free models")
    fig.tight_layout()
    fig.savefig(output / "model_comparison_boxplots.png", dpi=220)
    fig.savefig(output / "model_comparison_boxplots.pdf")
    plt.close(fig)


def dominance_table(silver: pd.DataFrame, intrinsic: pd.DataFrame, viewer: pd.DataFrame, replicate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        (silver, ["section"], "silver_conditional_recall", 1),
        (intrinsic, ["section"], "editing_fraction", -1),
        (intrinsic, ["section"], "singleton_fraction", -1),
        (intrinsic, ["section"], "dense_10bp_fraction", -1),
        (replicate, ["patient"], "replicate_jaccard", 1),
        (viewer[viewer["section"].eq("P6_rep1") & viewer["detector"].eq("fixed_raw")], ["section"], "f1", 1),
    ]
    for frame, unit_columns, metric, direction in metrics:
        if metric not in frame:
            continue
        baseline = frame[frame["model"].eq("baseline")][unit_columns + [metric]].rename(columns={metric: "baseline"})
        compared = frame[~frame["model"].eq("baseline")].merge(baseline, on=unit_columns, how="inner")
        for row in compared.itertuples(index=False):
            value = getattr(row, metric)
            base = row.baseline
            rows.append({
                **{column: getattr(row, column) for column in unit_columns},
                "model": row.model,
                "metric": metric,
                "value": value,
                "baseline": base,
                "delta_favorable": direction * (value - base),
                "noninferior_to_baseline": bool(direction * (value - base) >= -1e-12),
            })
    table = pd.DataFrame(rows)
    summary = table.groupby("model")["noninferior_to_baseline"].agg(["sum", "count", "mean"]).reset_index()
    summary = summary.rename(columns={"sum": "noninferior_cases", "count": "evaluated_cases", "mean": "noninferior_fraction"})
    return table.merge(summary, on="model", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", default=list(config.SECTIONS), choices=sorted(config.SECTIONS))
    parser.add_argument("--output", type=Path, default=config.OUTPUT_ROOT / "evaluation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    silver, intrinsic, replicate = evaluate_sets(args.sections)
    viewer = pd.DataFrame([
        row for section in args.sections for model in MODELS for row in viewer_rows(section, model)
    ])
    dominance = dominance_table(silver, intrinsic, viewer, replicate)
    for name, frame in (
        ("silver_recovery", silver), ("intrinsic_metrics", intrinsic),
        ("viewer_tumor_clustering", viewer), ("replicate_consistency", replicate),
        ("baseline_dominance", dominance),
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False)
    boxplots(silver, intrinsic, viewer, replicate, args.output)
    summary = (
        dominance[["model", "noninferior_cases", "evaluated_cases", "noninferior_fraction"]]
        .drop_duplicates().sort_values("noninferior_fraction", ascending=False)
    )
    summary.to_csv(args.output / "replacement_summary.csv", index=False)
    (args.output / "evaluation_manifest.json").write_text(json.dumps({
        "run_label": config.RUN_LABEL,
        "models": list(MODELS),
        "sections": args.sections,
        "scoring_used_wes_or_pathology": False,
        "evaluation_uses_wes": True,
        "evaluation_uses_pathology": ["P6_rep1"],
        "viewer_detector": {"seed_pct": 90, "grow_pct": 80, "min_size": 5, "split_depth": 0.40},
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
