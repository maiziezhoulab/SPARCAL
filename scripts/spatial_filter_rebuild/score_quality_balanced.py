#!/usr/bin/env python3
"""Build additive quality-balanced hybrids without external truth labels.

Each hybrid preserves as much as possible of a frozen v1 spatial/CNV ranking,
but imposes two transparent deployment constraints:

1. no xMHC loci (the production baseline already treats this region as an
   artifact-prone hard veto), and
2. no more DP<3 loci than the frozen baseline of the same section.

Removed loci are filled by the matching reliability-v2 score.  WES,
pathology, production classification, and fitted coefficients are absent from
the selection function.  Baseline membership is read only to define the set
size and the low-depth quality budget.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts.spatial_filter_rebuild import config


HYBRIDS = {
    "calicost_quality_balanced": ("calicost", "calicost_reliability_score"),
    "integrated_quality_balanced": ("integrated", "integrated_reliability_score"),
    "consensus_quality_balanced": ("consensus", "consensus_reliability_score"),
}


def constrained_selection(
    frame: pd.DataFrame,
    base_keys: set[str],
    base_score_col: str,
    fill_score_col: str,
    target_n: int,
    low_depth_budget: int,
) -> pd.DataFrame:
    """Maximally retain a base set under xMHC and low-depth constraints."""

    data = frame.copy()
    data["variant_key"] = data["variant_key"].astype(str)
    data["_key"] = data["variant_key"]
    data["_low_depth"] = pd.to_numeric(data["total_depth_cache"], errors="coerce").fillna(0).lt(3)
    universe = (
        data["race"].astype(str).eq("denovo")
        & data["ref"].astype(str).str.len().eq(1)
        & data["alt"].astype(str).str.len().eq(1)
        & pd.to_numeric(data["total_alt"], errors="coerce").fillna(0).gt(0)
        & ~data["xmhc"].fillna(False).astype(bool)
    )
    eligible = data.loc[universe].copy()
    in_base = eligible["variant_key"].isin(base_keys)
    base = eligible.loc[in_base].sort_values([base_score_col, "_key"], ascending=[False, True], kind="stable")
    retained_high = base.loc[~base["_low_depth"]]
    retained_low = base.loc[base["_low_depth"]].head(low_depth_budget)
    selected = pd.concat([retained_high, retained_low], ignore_index=False)

    missing = target_n - len(selected)
    already = set(selected["variant_key"])
    fill = eligible.loc[~eligible["variant_key"].isin(already)].sort_values(
        [fill_score_col, "_key"], ascending=[False, True], kind="stable"
    )
    fill_high = fill.loc[~fill["_low_depth"]].head(max(0, missing))
    selected = pd.concat([selected, fill_high], ignore_index=False)
    missing = target_n - len(selected)
    remaining_low_budget = low_depth_budget - int(selected["_low_depth"].sum())
    if missing > 0 and remaining_low_budget > 0:
        already = set(selected["variant_key"])
        fill_low = fill.loc[
            fill["_low_depth"] & ~fill["variant_key"].isin(already)
        ].head(min(missing, remaining_low_budget))
        selected = pd.concat([selected, fill_low], ignore_index=False)
    if len(selected) != target_n:
        raise ValueError(
            f"Could select only {len(selected):,}/{target_n:,} variants under declared constraints"
        )
    selected = selected.sort_values([base_score_col, "_key"], ascending=[False, True], kind="stable")
    return selected.drop(columns=["_key", "_low_depth"])


def score_section(section: str) -> dict[str, object]:
    score_dir = config.score_dir(section)
    reliability = pd.read_csv(score_dir / "reliability_v2_scores.tsv.gz", sep="\t", low_memory=False)
    frozen = pd.read_csv(score_dir / "all_model_scores.tsv.gz", sep="\t", low_memory=False)
    score_columns = ["variant_key", "calicost_score", "integrated_score", "consensus_score"]
    frame = reliability.merge(frozen[score_columns], on="variant_key", how="inner", validate="one_to_one")
    baseline = pd.read_csv(score_dir / "selected__baseline__matched.tsv.gz", sep="\t", low_memory=False)
    target_n = len(baseline)
    low_depth_budget = int(pd.to_numeric(baseline["total_depth_cache"], errors="coerce").fillna(0).lt(3).sum())

    outputs = [score_dir / f"selected__{model}__matched.tsv.gz" for model in HYBRIDS]
    manifest_path = score_dir / "quality_balanced_v3_selection_manifest.json"
    collisions = [path for path in [*outputs, manifest_path] if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite quality-balanced outputs:\n" + "\n".join(map(str, collisions)))

    summary: dict[str, object] = {
        "section": section,
        "target_n": target_n,
        "low_depth_budget": low_depth_budget,
        "xmhc_veto": True,
        "gt_used": False,
        "source_classification_used_for_scoring_or_selection": False,
        "baseline_used_only_for": ["target set size", "DP<3 count ceiling"],
        "models": {},
    }
    for model, (base_model, fill_score) in HYBRIDS.items():
        base = pd.read_csv(
            score_dir / f"selected__{base_model}__matched.tsv.gz", sep="\t", usecols=["variant_key"]
        )
        base_keys = set(base["variant_key"].astype(str))
        selected = constrained_selection(
            frame, base_keys, f"{base_model}_score", fill_score, target_n, low_depth_budget
        )
        selected.to_csv(score_dir / f"selected__{model}__matched.tsv.gz", sep="\t", index=False)
        keys = set(selected["variant_key"].astype(str))
        summary["models"][model] = {
            "base_model": base_model,
            "fill_score": fill_score,
            "matched_n": len(selected),
            "retained_from_base": len(keys & base_keys),
            "retained_fraction": len(keys & base_keys) / target_n if target_n else 1.0,
            "low_depth_n": int(pd.to_numeric(selected["total_depth_cache"], errors="coerce").fillna(0).lt(3).sum()),
            "low_depth_fraction": float(pd.to_numeric(selected["total_depth_cache"], errors="coerce").fillna(0).lt(3).mean()),
            "xmhc_n": int(selected["xmhc"].fillna(False).astype(bool).sum()),
        }
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    args = parser.parse_args()
    print(json.dumps(score_section(args.section), indent=2))


if __name__ == "__main__":
    main()
