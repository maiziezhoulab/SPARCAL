#!/usr/bin/env python3
"""Additive v2 reliability ranking for the frozen multistate posteriors.

The v1 scorer shrank a five-state somatic posterior toward 0.5 when evidence
was weak.  That is neutral for a binary classifier, but it is not the somatic
prior of this five-state model (0.10 for denovo candidates), and therefore made
very low-information loci rank above well-measured non-somatic loci during a
large matched-quota fill.

This follow-up does not alter v1 scores or selections.  It ranks candidates by
``P(somatic | evidence) * observation_information`` and writes new, uniquely
named matched sets.  The product has a direct interpretation: expected somatic
probability discounted by how much usable depth/spot support exists.  No WES,
pathology, production classification, or fitted coefficient is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.spatial_filter_rebuild import config


BASE_MODELS = ("molecular", "spatial", "calicost", "integrated")
RELIABILITY_MODELS = tuple(f"{name}_reliability" for name in BASE_MODELS) + ("consensus_reliability",)


def information_column(frame: pd.DataFrame, model: str) -> str:
    # The frozen table contains both the spatial feature information and the
    # posterior observation information.  pandas disambiguates the latter.
    if model == "spatial" and "spatial_information.1" in frame:
        return "spatial_information.1"
    return f"{model}_information"


def reliability_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Pure label-free transformation of frozen posterior components."""

    result = pd.DataFrame(index=frame.index)
    component_scores = []
    for model in BASE_MODELS:
        somatic = (
            pd.to_numeric(frame[f"{model}_p_somatic_subclonal"], errors="coerce").fillna(0)
            + pd.to_numeric(frame[f"{model}_p_somatic_clonal"], errors="coerce").fillna(0)
        ).clip(0, 1)
        information = pd.to_numeric(frame[information_column(frame, model)], errors="coerce").fillna(0).clip(0, 1)
        score = somatic * information
        name = f"{model}_reliability"
        result[f"{name}_p_somatic"] = somatic
        result[f"{name}_information"] = information
        result[f"{name}_score"] = score
        result[f"{name}_natural"] = (
            somatic.ge(0.80)
            & information.ge(0.20)
            & pd.to_numeric(frame["total_alt"], errors="coerce").fillna(0).ge(2)
            & pd.to_numeric(frame["positive_spots"], errors="coerce").fillna(0).ge(2)
        )
        component_scores.append(score)

    ranks = np.column_stack([score.rank(method="average", pct=True).to_numpy() for score in component_scores])
    consensus = 0.5 * np.median(ranks, axis=1) + 0.5 * np.min(ranks, axis=1)
    result["consensus_reliability_score"] = consensus
    result["consensus_reliability_information"] = np.column_stack([
        result[f"{model}_reliability_information"].to_numpy() for model in BASE_MODELS
    ]).min(axis=1)
    result["consensus_reliability_p_somatic"] = np.column_stack([
        result[f"{model}_reliability_p_somatic"].to_numpy() for model in BASE_MODELS
    ]).min(axis=1)
    result["consensus_reliability_natural"] = (
        result["consensus_reliability_p_somatic"].ge(0.80)
        & result["consensus_reliability_information"].ge(0.20)
        & pd.to_numeric(frame["total_alt"], errors="coerce").fillna(0).ge(2)
        & pd.to_numeric(frame["positive_spots"], errors="coerce").fillna(0).ge(2)
    )
    return result


def score_section(section: str, overwrite: bool = False) -> dict[str, object]:
    score_dir = config.score_dir(section)
    output = score_dir / "reliability_v2_scores.tsv.gz"
    targets = [score_dir / f"selected__{model}__matched.tsv.gz" for model in RELIABILITY_MODELS]
    collisions = [path for path in [output, *targets] if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError("Refusing to overwrite reliability-v2 outputs:\n" + "\n".join(map(str, collisions)))

    frame = pd.read_csv(score_dir / "all_model_scores.tsv.gz", sep="\t", low_memory=False)
    scored = reliability_scores(frame)
    manifest_v1 = json.loads((score_dir / "selection_manifest.json").read_text())
    baseline_n = int(manifest_v1["baseline_n"])
    universe = (
        frame["race"].astype(str).eq("denovo")
        & frame["ref"].astype(str).str.len().eq(1)
        & frame["alt"].astype(str).str.len().eq(1)
        & pd.to_numeric(frame["total_alt"], errors="coerce").fillna(0).gt(0)
    )

    identity = [
        "variant", "variant_key", "ref", "alt", "race", "source_classification",
        "total_alt", "total_depth_cache", "positive_spots", "covered_spots_cache",
        "coverage_prevalence", "spatial_log_enrichment", "clone_log_odds",
        "clone_information", "local_density_10bp", "rna_editing_change", "xmhc",
        "vaf_log_bf", "vaf_information", "vaf_context_reliability",
        "patient_rep_count", "cross_patient_count",
    ]
    combined = pd.concat([frame[identity].reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    combined.to_csv(output, sep="\t", index=False)

    summary: dict[str, object] = {
        "section": section,
        "source_scores": str(score_dir / "all_model_scores.tsv.gz"),
        "score_table": str(output),
        "n_universe": int(universe.sum()),
        "baseline_matched_n": baseline_n,
        "ranking_rule": "P(somatic|evidence) * observation_information",
        "consensus_rule": "0.5 * median percentile rank + 0.5 * minimum percentile rank",
        "gt_used": False,
        "source_classification_used_for_scoring_or_selection": False,
        "models": {},
    }
    for model in RELIABILITY_MODELS:
        score_col = f"{model}_score"
        eligible = combined.loc[universe].copy()
        eligible["_key"] = eligible["variant_key"].astype(str)
        selected = eligible.sort_values([score_col, "_key"], ascending=[False, True], kind="stable").head(baseline_n)
        selected.drop(columns="_key").to_csv(
            score_dir / f"selected__{model}__matched.tsv.gz", sep="\t", index=False
        )
        summary["models"][model] = {
            "matched_n": len(selected),
            "natural_in_universe": int((universe & scored[f"{model}_natural"]).sum()),
            "low_depth_fraction": float(pd.to_numeric(selected["total_depth_cache"], errors="coerce").lt(3).mean()),
            "median_depth": float(pd.to_numeric(selected["total_depth_cache"], errors="coerce").median()),
            "median_positive_spots": float(pd.to_numeric(selected["positive_spots"], errors="coerce").median()),
        }

    manifest = score_dir / "reliability_v2_selection_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(score_section(args.section, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
