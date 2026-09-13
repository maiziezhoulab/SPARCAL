#!/usr/bin/env python3
"""Matched no-CalicoST controls for the quality-balanced v3 experiment."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts.spatial_filter_rebuild import config
from scripts.spatial_filter_rebuild.score_quality_balanced import constrained_selection


ABLATIONS = {
    "molecular_quality_balanced": ("molecular", "molecular_reliability_score"),
    "spatial_quality_balanced": ("spatial", "spatial_reliability_score"),
}


def score_section(section: str) -> dict[str, object]:
    score_dir = config.score_dir(section)
    reliability = pd.read_csv(score_dir / "reliability_v2_scores.tsv.gz", sep="\t", low_memory=False)
    frozen = pd.read_csv(score_dir / "all_model_scores.tsv.gz", sep="\t", low_memory=False)
    frame = reliability.merge(
        frozen[["variant_key", "molecular_score", "spatial_score"]],
        on="variant_key", how="inner", validate="one_to_one",
    )
    baseline = pd.read_csv(score_dir / "selected__baseline__matched.tsv.gz", sep="\t", low_memory=False)
    target_n = len(baseline)
    low_depth_budget = int(pd.to_numeric(baseline["total_depth_cache"], errors="coerce").fillna(0).lt(3).sum())
    outputs = [score_dir / f"selected__{model}__matched.tsv.gz" for model in ABLATIONS]
    manifest_path = score_dir / "quality_balanced_v3_ablation_manifest.json"
    collisions = [path for path in [*outputs, manifest_path] if path.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite quality-balanced ablations:\n" + "\n".join(map(str, collisions)))

    summary: dict[str, object] = {
        "section": section,
        "target_n": target_n,
        "low_depth_budget": low_depth_budget,
        "xmhc_veto": True,
        "gt_used": False,
        "source_classification_used_for_scoring_or_selection": False,
        "models": {},
    }
    for model, (base_model, fill_score) in ABLATIONS.items():
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
