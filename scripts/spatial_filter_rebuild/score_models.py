#!/usr/bin/env python3
"""Score all classes under pre-registered, training-free evidence combinations.

The production class is retained only to reproduce the frozen baseline and its
set size.  It is never read by ``score_frame``.  WES/pathology inputs are absent
from this module by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.spatial_filter_rebuild import config
from scripts.spatial_filter_rebuild.statistics import (
    STATE_NAMES,
    bounded_evidence,
    effective_somatic_probability,
    entropy,
    posterior_from_logits,
    state_log_priors,
    vaf_state_logliks,
)


def numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def molecular_quality(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (real-variant evidence, somatic-artifact risk), both bounded."""

    total_alt = numeric(frame, "total_alt", 0)
    positive_spots = numeric(frame, "positive_spots", 0)
    support = 0.35 * np.log1p(total_alt) + 0.25 * np.log1p(positive_spots)

    baseq = numeric(frame, "alt_mean_baseq")
    mapq = numeric(frame, "alt_mean_mapq")
    baseq_evidence = np.where(np.isfinite(baseq), np.clip((baseq - 20) / 15, -1, 1), 0)
    mapq_evidence = np.where(np.isfinite(mapq), np.clip((mapq - 20) / 20, -1, 1), 0)

    strand_count = numeric(frame, "alt_forward", 0) + numeric(frame, "alt_reverse", 0)
    strand_balance = numeric(frame, "alt_strand_balance")
    strand_evidence = np.where(
        strand_count >= 4,
        np.clip((np.nan_to_num(strand_balance, nan=0) - 0.05) / 0.20, -1, 1),
        0,
    )
    bias_penalty = np.zeros(len(frame), dtype=float)
    for name in ("info_mqb", "info_bqb", "info_rpb"):
        value = numeric(frame, name)
        bias_penalty += np.where(np.isfinite(value) & (value < 1e-3), 0.25, 0)

    nn_novar = numeric(frame, "nn_novar")
    nn_real = np.where(
        np.isfinite(nn_novar),
        np.clip(np.log((1 - np.clip(nn_novar, 1e-4, 1 - 1e-4)) / np.clip(nn_novar, 1e-4, 1)), -2, 2),
        0,
    )
    real_evidence = support + 0.35 * baseq_evidence + 0.25 * mapq_evidence + 0.20 * strand_evidence + 0.20 * nn_real - bias_penalty

    density = numeric(frame, "local_density_10bp", 1)
    editing = frame.get("rna_editing_change", pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy()
    xmhc = frame.get("xmhc", pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy()
    cross_patient = numeric(frame, "cross_patient_count", 1)
    patient_reps = numeric(frame, "patient_rep_count", 1)
    risk = (
        0.55 * np.log1p(np.maximum(density - 1, 0))
        + 0.85 * editing
        + 0.65 * xmhc
        + 0.45 * (cross_patient > 1)
        - 0.55 * (patient_reps > 1)
    )
    return np.clip(real_evidence, -3, 4), np.clip(risk, -1, 5)


def apply_real_evidence(logits: np.ndarray, real: np.ndarray, risk: np.ndarray) -> np.ndarray:
    result = logits.copy()
    result[:, 0] -= real
    result[:, 1:] += 0.35 * real[:, None]
    result[:, 0] += risk
    result[:, 3:] -= risk[:, None]
    return result


def apply_somatic_shift(logits: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    result = logits.copy()
    result[:, 1:3] -= 0.5 * evidence[:, None]
    result[:, 3:] += evidence[:, None]
    return result


def spatial_evidence(frame: pd.DataFrame) -> np.ndarray:
    information = np.nan_to_num(numeric(frame, "spatial_information"), nan=0)
    enrichment = np.nan_to_num(numeric(frame, "spatial_log_enrichment"), nan=0)
    precision = np.nan_to_num(numeric(frame, "spatial_neighbor_precision"), nan=0)
    return information * (1.25 * bounded_evidence(enrichment, 1.0) + 0.5 * (precision - 0.5))


def calicost_evidence(frame: pd.DataFrame) -> np.ndarray:
    log_bf = np.nan_to_num(numeric(frame, "vaf_log_bf"), nan=0)
    vaf_info = np.nan_to_num(numeric(frame, "vaf_information"), nan=0)
    context = np.nan_to_num(numeric(frame, "vaf_context_reliability"), nan=0)
    mechanistic = 2.0 * bounded_evidence(log_bf, 5.0) * vaf_info * context
    clone = (
        bounded_evidence(np.nan_to_num(numeric(frame, "clone_log_odds"), nan=0), 2.0)
        * np.nan_to_num(numeric(frame, "clone_information"), nan=0)
    )
    purity = np.nan_to_num(numeric(frame, "f_purity_correlation"), nan=0)
    cnv = np.nan_to_num(numeric(frame, "f_cnv_consistency", 0.5), nan=0.5) - 0.5
    return mechanistic + 0.65 * clone + 0.25 * purity + 0.20 * cnv


def state_logits(frame: pd.DataFrame, use_context: bool) -> np.ndarray:
    total_alt = numeric(frame, "total_alt", 0)
    total_depth = numeric(frame, "total_depth_cache", 0)
    if use_context:
        slo = numeric(frame, "vaf_somatic_low")
        shi = numeric(frame, "vaf_somatic_high")
        glo = numeric(frame, "vaf_germline_low")
        ghi = numeric(frame, "vaf_germline_high")
    else:
        slo = shi = glo = ghi = np.full(len(frame), np.nan)
    likelihood = vaf_state_logliks(total_alt, total_depth, slo, shi, glo, ghi)
    panel = frame["race"].astype(str).eq("defined").to_numpy()
    real, risk = molecular_quality(frame)
    return apply_real_evidence(likelihood + state_log_priors(panel), real, risk)


def summarize_model(frame: pd.DataFrame, logits: np.ndarray, name: str) -> pd.DataFrame:
    posterior = posterior_from_logits(logits)
    effective, information = effective_somatic_probability(
        posterior, numeric(frame, "total_depth_cache", 0), numeric(frame, "positive_spots", 0)
    )
    output = pd.DataFrame(index=frame.index)
    for index, state in enumerate(STATE_NAMES):
        output[f"{name}_p_{state}"] = posterior[:, index]
    state = np.asarray(STATE_NAMES)[posterior.argmax(axis=1)]
    state_entropy = entropy(posterior)
    output[f"{name}_state"] = state
    output[f"{name}_entropy"] = state_entropy
    output[f"{name}_information"] = information
    output[f"{name}_score"] = effective
    output[f"{name}_decision"] = np.where(
        (information < 0.20) | (state_entropy > 0.85), "unresolved", state
    )
    output[f"{name}_natural"] = (
        (posterior[:, 3] + posterior[:, 4] >= 0.80)
        & (information >= 0.20)
        & (numeric(frame, "total_alt", 0) >= 2)
        & (numeric(frame, "positive_spots", 0) >= 2)
    )
    return output


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Pure scorer: deliberately never accesses ``source_classification``."""

    base = state_logits(frame, use_context=False)
    contextual = state_logits(frame, use_context=True)
    spatial = spatial_evidence(frame)
    calicost = calicost_evidence(frame)
    model_logits = {
        "molecular": base,
        "spatial": apply_somatic_shift(base, spatial),
        "calicost": apply_somatic_shift(contextual, calicost),
        "integrated": apply_somatic_shift(apply_somatic_shift(contextual, spatial), calicost),
    }
    outputs = [summarize_model(frame, logits, name) for name, logits in model_logits.items()]
    result = pd.concat(outputs, axis=1)
    ranks = np.column_stack([
        result[f"{name}_score"].rank(method="average", pct=True).to_numpy()
        for name in ("molecular", "spatial", "calicost", "integrated")
    ])
    consensus_score = 0.5 * np.median(ranks, axis=1) + 0.5 * np.min(ranks, axis=1)
    result["consensus_score"] = consensus_score
    result["consensus_information"] = result[
        [f"{name}_information" for name in ("molecular", "spatial", "calicost", "integrated")]
    ].min(axis=1)
    result["consensus_entropy"] = result[
        [f"{name}_entropy" for name in ("molecular", "spatial", "calicost", "integrated")]
    ].max(axis=1)
    best_state = result["integrated_state"].astype(str)
    result["consensus_state"] = best_state
    result["consensus_natural"] = (
        (consensus_score >= 0.90)
        & result["integrated_state"].isin(["somatic_subclonal", "somatic_clonal"])
        & (result["integrated_information"] >= 0.20)
        & (numeric(frame, "total_alt", 0) >= 2)
        & (numeric(frame, "positive_spots", 0) >= 2)
    )
    return result


def add_recurrence(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for section, frame in frames.items():
        observed = frame.loc[numeric(frame, "total_alt", 0) > 0, ["variant_key"]].copy()
        observed["section"] = section
        observed["patient"] = str(config.SECTIONS[section]["patient"])
        rows.append(observed)
    all_observed = pd.concat(rows, ignore_index=True).drop_duplicates(["section", "variant_key"])
    patient_count = all_observed.groupby(["patient", "variant_key"])["section"].nunique()
    cross_count = all_observed.groupby("variant_key")["patient"].nunique()
    for section, frame in frames.items():
        patient = str(config.SECTIONS[section]["patient"])
        keys = pd.MultiIndex.from_arrays([[patient] * len(frame), frame["variant_key"]])
        frame["patient_rep_count"] = patient_count.reindex(keys, fill_value=0).to_numpy()
        frame["cross_patient_count"] = cross_count.reindex(frame["variant_key"], fill_value=0).to_numpy()


def write_section(section: str, frame: pd.DataFrame, scored: pd.DataFrame, overwrite: bool) -> dict[str, object]:
    outdir = config.score_dir(section)
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / "all_model_scores.tsv.gz"
    if score_path.exists() and not overwrite:
        raise FileExistsError(score_path)
    identity = [
        "variant", "variant_key", "ref", "alt", "race", "source_classification",
        "total_alt", "total_depth_cache", "positive_spots", "covered_spots_cache",
        "coverage_prevalence", "spatial_log_enrichment", "spatial_information",
        "clone_log_odds", "clone_information", "local_density_10bp",
        "rna_editing_change", "xmhc", "vaf_log_bf", "vaf_information",
        "vaf_context_reliability", "patient_rep_count", "cross_patient_count",
    ]
    identity = [column for column in identity if column in frame]
    combined = pd.concat([frame[identity].reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    combined.to_csv(score_path, sep="\t", index=False)

    universe = (
        frame["race"].astype(str).eq("denovo")
        & frame["snp"].fillna(False).astype(bool)
        & (numeric(frame, "total_alt", 0) > 0)
    )
    baseline = universe & frame["source_classification"].astype(str).eq("somatic")
    baseline_n = int(baseline.sum())
    selection_columns = identity + [column for column in scored if column.endswith(("_score", "_state", "_entropy", "_information"))]
    selected_summary: dict[str, object] = {
        "section": section,
        "n_candidates": len(frame),
        "n_universe": int(universe.sum()),
        "baseline_n": baseline_n,
        "gt_used": False,
        "models": {},
    }
    baseline_frame = combined.loc[baseline, selection_columns]
    baseline_frame.to_csv(outdir / "selected__baseline__matched.tsv.gz", sep="\t", index=False)

    stable_key = frame["variant_key"].astype(str)
    for model in config.MODEL_NAMES:
        score = scored[f"{model}_score"]
        eligible = combined.loc[universe].copy()
        eligible["_score"] = score.loc[universe].to_numpy()
        eligible["_key"] = stable_key.loc[universe].to_numpy()
        matched = eligible.sort_values(["_score", "_key"], ascending=[False, True], kind="stable").head(baseline_n)
        matched.drop(columns=["_score", "_key"]).to_csv(
            outdir / f"selected__{model}__matched.tsv.gz", sep="\t", index=False
        )
        natural_mask = universe & scored[f"{model}_natural"].astype(bool)
        combined.loc[natural_mask, selection_columns].to_csv(
            outdir / f"selected__{model}__natural.tsv.gz", sep="\t", index=False
        )
        selected_summary["models"][model] = {
            "matched_n": int(len(matched)),
            "natural_n": int(natural_mask.sum()),
            "matched_changed_from_baseline": int(len(set(matched["variant_key"]) ^ set(baseline_frame["variant_key"]))),
        }
    (outdir / "selection_manifest.json").write_text(json.dumps(selected_summary, indent=2) + "\n")
    return selected_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", default=list(config.SECTIONS), choices=sorted(config.SECTIONS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    frames = {
        section: pd.read_csv(config.score_dir(section) / "features.tsv.gz", sep="\t", low_memory=False)
        for section in args.sections
    }
    add_recurrence(frames)
    summaries = []
    for section, frame in frames.items():
        scored = score_frame(frame)
        summaries.append(write_section(section, frame, scored, args.overwrite))
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_ROOT / "scoring_manifest.json").write_text(json.dumps({
        "run_label": config.RUN_LABEL,
        "sections": summaries,
        "models": list(config.MODEL_NAMES),
        "selection_rules": {
            "matched": "top N among all denovo SNPs with ALT support; N=frozen baseline count",
            "natural": "somatic posterior>=0.8, information>=0.2, ALT>=2, positive spots>=2",
        },
        "source_classification_used_only_for": ["baseline membership", "baseline matched N"],
        "gt_used": False,
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
