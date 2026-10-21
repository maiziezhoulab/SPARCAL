#!/usr/bin/env python3
"""Quality gates for section-specific CalicoST purity and clone/CNV evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

MIN_CLONE_SPOT_FRACTION = 0.05
MIN_UMI_AUC = 0.20
MAX_UMI_AUC = 0.80
MAX_REFERENCE_ALTERED_FRACTION = 0.01
MIN_TUMOR_ALTERED_FRACTION = 0.05
MIN_MEDIAN_MAX_POSTERIOR = 0.80
MIN_PURITY_FINITE_FRACTION = 0.50
MIN_PURITY_ABOVE_MIN_FRACTION = 0.20
MAX_ABS_PURITY_UMI_SPEARMAN = 0.50
MIN_PURITY_CLONE_AUC = 0.70

THRESHOLDS = {
    "min_clone_spot_fraction": MIN_CLONE_SPOT_FRACTION,
    "min_umi_auc": MIN_UMI_AUC,
    "max_umi_auc": MAX_UMI_AUC,
    "max_reference_altered_fraction": MAX_REFERENCE_ALTERED_FRACTION,
    "min_tumor_altered_fraction": MIN_TUMOR_ALTERED_FRACTION,
    "min_median_max_posterior": MIN_MEDIAN_MAX_POSTERIOR,
    "min_purity_finite_fraction": MIN_PURITY_FINITE_FRACTION,
    "min_purity_above_min_fraction": MIN_PURITY_ABOVE_MIN_FRACTION,
    "max_abs_purity_umi_spearman": MAX_ABS_PURITY_UMI_SPEARMAN,
    "min_purity_clone_auc": MIN_PURITY_CLONE_AUC,
}


def _as_python(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def section_rows(
    frame: pd.DataFrame,
    sample_id: Optional[str],
    barcode_column: str = "BARCODES",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select one joint-run section and strip exactly its terminal suffix."""
    if barcode_column not in frame.columns:
        raise ValueError(f"missing required barcode column {barcode_column!r}")
    barcodes = frame[barcode_column].astype(str)
    if sample_id == "NONE":
        suffixed = [barcode for barcode in barcodes if "_" in barcode]
        if suffixed:
            raise ValueError(
                "--sample_id NONE requires unsuffixed single-section labels; "
                f"found suffix-like barcode {suffixed[0]!r}"
            )
        mask = np.ones(len(frame), dtype=bool)
        selected = frame.copy()
    elif sample_id:
        suffix = f"_{sample_id}"
        mask = barcodes.str.endswith(suffix).to_numpy()
        if not mask.any():
            raise ValueError(f"no barcodes end in requested CalicoST sample suffix {suffix}")
        selected = frame.loc[mask].copy()
        selected[barcode_column] = selected[barcode_column].astype(str).str.slice(
            stop=-len(suffix)
        )
    else:
        suffixes = {
            barcode.rsplit("_", 1)[1]
            for barcode in barcodes
            if "_" in barcode
        }
        if len(suffixes) > 1:
            raise ValueError(
                "joint CalicoST input contains more than one sample suffix; "
                "pass --calicost_sample_id/--sample_id"
            )
        mask = np.ones(len(frame), dtype=bool)
        selected = frame.copy()
        selected[barcode_column] = [
            barcode.split("_")[0] if "_" in barcode else barcode
            for barcode in barcodes
        ]
    duplicates = selected[barcode_column][selected[barcode_column].duplicated()].unique()
    if len(duplicates):
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"duplicate barcodes after suffix stripping: {preview}")
    return selected.reset_index(drop=True), mask


def _clone_columns(cnv_segments: pd.DataFrame) -> Dict[int, Tuple[str, str]]:
    normalized = {str(column).replace(" ", "_"): column for column in cnv_segments.columns}
    pairs: Dict[int, Tuple[str, str]] = {}
    for normalized_name in normalized:
        if not normalized_name.startswith("clone") or not normalized_name.endswith("_A"):
            continue
        middle = normalized_name[len("clone") : -len("_A")]
        if not middle.isdigit():
            continue
        b_name = f"clone{middle}_B"
        if b_name not in normalized:
            raise ValueError(f"CNV segments have {normalized_name!r} but no matching B column")
        pairs[int(middle)] = (normalized[normalized_name], normalized[b_name])
    if not pairs:
        raise ValueError("CNV segments contain no cloneN A/cloneN B column pairs")
    return pairs


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(labels) < 2 or np.unique(labels).size != 2:
        return np.nan
    return float(roc_auc_score(labels, values))


def evaluate_gate(
    clone_labels: pd.DataFrame,
    cnv_segments: pd.DataFrame,
    posterior: np.ndarray,
    purity_or_None: Optional[pd.DataFrame],
    sample_id: Optional[str],
    spot_umi: pd.Series,
    thresholds: Optional[Mapping[str, float]] = None,
) -> dict:
    """Evaluate the fixed CNV and purity gates for one spatial section."""
    limits = dict(THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"unknown gate thresholds: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in thresholds.items()})

    clone_labels = clone_labels.copy()
    posterior = np.asarray(posterior)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be a 2D array, got shape {posterior.shape}")
    if len(clone_labels) != posterior.shape[0]:
        raise ValueError(
            "posterior row count does not equal clone_labels.tsv row count: "
            f"{posterior.shape[0]} != {len(clone_labels)}"
        )
    section_labels, section_mask = section_rows(clone_labels, sample_id)
    if "clone_label" not in section_labels.columns:
        raise ValueError("clone labels are missing clone_label")
    section_labels["clone_label"] = pd.to_numeric(
        section_labels["clone_label"], errors="raise"
    ).astype(int)
    section_posterior = posterior[section_mask]
    if posterior.shape[1] <= int(section_labels["clone_label"].max()):
        raise ValueError("posterior has fewer columns than the largest clone label")

    if not isinstance(spot_umi, pd.Series):
        spot_umi = pd.Series(spot_umi)
    spot_umi = pd.to_numeric(spot_umi, errors="coerce")
    spot_umi.index = spot_umi.index.astype(str)
    if spot_umi.index.has_duplicates:
        raise ValueError("spot UMI input contains duplicate barcodes")

    barcodes = section_labels["BARCODES"].astype(str).to_numpy()
    labels = section_labels["clone_label"].to_numpy(dtype=int)
    umi = spot_umi.reindex(barcodes).to_numpy(dtype=float)
    log_umi = np.where(np.isfinite(umi) & (umi >= 0), np.log10(umi + 1.0), np.nan)

    pairs = _clone_columns(cnv_segments)
    starts = pd.to_numeric(cnv_segments["START"], errors="raise").to_numpy(dtype=float)
    ends = pd.to_numeric(cnv_segments["END"], errors="raise").to_numpy(dtype=float)
    widths = ends - starts + 1.0
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("CNV segments contain invalid START/END coordinates")

    clone_metrics: Dict[str, dict] = {}
    dropped: Dict[str, list] = {}
    total_spots = len(labels)
    observed_clones = {int(value) for value in labels}
    for clone_id in sorted(observed_clones | set(pairs)):
        clone_mask = labels == clone_id
        n_spots = int(clone_mask.sum())
        spot_fraction = n_spots / total_spots if total_spots else np.nan
        auc = _safe_auc(clone_mask.astype(int), log_umi)
        if clone_id in pairs:
            col_a, col_b = pairs[clone_id]
            allele_a = pd.to_numeric(cnv_segments[col_a], errors="coerce").to_numpy(float)
            allele_b = pd.to_numeric(cnv_segments[col_b], errors="coerce").to_numpy(float)
            valid = np.isfinite(allele_a) & np.isfinite(allele_b)
            denom = float(widths[valid].sum())
            altered = valid & ((allele_a != 1) | (allele_b != 1))
            loh = valid & ((allele_a == 0) | (allele_b == 0))
            altered_fraction = float(widths[altered].sum() / denom) if denom else np.nan
            loh_fraction = float(widths[loh].sum() / denom) if denom else np.nan
        else:
            altered_fraction = np.nan
            loh_fraction = np.nan
        reasons = []
        if not np.isfinite(spot_fraction) or spot_fraction < limits["min_clone_spot_fraction"]:
            reasons.append("spot_fraction_below_min")
        if np.isfinite(auc) and (
            auc > limits["max_umi_auc"] or auc < limits["min_umi_auc"]
        ):
            reasons.append("umi_auc_outside_range")
        if reasons:
            dropped[str(clone_id)] = reasons
        clone_metrics[str(clone_id)] = {
            "altered_genome_fraction": _as_python(altered_fraction),
            "loh_fraction": _as_python(loh_fraction),
            "n_spots": n_spots,
            "spot_fraction": _as_python(spot_fraction),
            "log_umi_auc_clone_vs_rest": _as_python(auc),
        }

    surviving = sorted(observed_clones - {int(clone_id) for clone_id in dropped})
    reference_candidates = [
        clone_id for clone_id in surviving
        if clone_metrics[str(clone_id)]["altered_genome_fraction"] is not None
        and clone_metrics[str(clone_id)]["altered_genome_fraction"]
        <= limits["max_reference_altered_fraction"]
    ]
    reference_clone = None
    if reference_candidates:
        reference_clone = max(
            reference_candidates,
            key=lambda clone_id: (clone_metrics[str(clone_id)]["n_spots"], -clone_id),
        )
    tumor_clones = [
        clone_id for clone_id in surviving
        if clone_metrics[str(clone_id)]["altered_genome_fraction"] is not None
        and clone_metrics[str(clone_id)]["altered_genome_fraction"]
        >= limits["min_tumor_altered_fraction"]
        and clone_id != reference_clone
    ]
    c1 = reference_clone is not None and bool(tumor_clones)

    surviving_mask = np.isin(labels, surviving)
    maximum_posterior = np.max(section_posterior[surviving_mask], axis=1)
    median_max_posterior = (
        float(np.nanmedian(maximum_posterior)) if maximum_posterior.size else np.nan
    )
    c3 = bool(
        np.isfinite(median_max_posterior)
        and median_max_posterior >= limits["min_median_max_posterior"]
    )
    cnv_pass = bool(c1 and c3)
    effective_labels = {
        barcode: (int(label) if int(label) in surviving else None)
        for barcode, label in zip(barcodes, labels)
    }

    purity_metrics: Dict[str, Any] = {
        "provided": purity_or_None is not None,
        "finite_fraction": 0.0,
        "above_min_fraction": 0.0,
        "iqr": None,
        "spearman_with_log_umi": None,
        "tumor_vs_reference_auc": None,
    }
    purity_checks: Dict[str, Optional[bool]] = {
        "P1_finite_fraction": False,
        "P2_dynamic_range": False,
        "P3_umi_independence": False,
        "P4_clone_discrimination": None,
    }
    if purity_or_None is not None:
        section_purity, _ = section_rows(purity_or_None.copy(), sample_id)
        if "Tumor" not in section_purity.columns:
            raise ValueError("purity input is missing Tumor")
        purity_series = pd.Series(
            pd.to_numeric(section_purity["Tumor"], errors="coerce").to_numpy(float),
            index=section_purity["BARCODES"].astype(str),
        )
        purity = purity_series.reindex(barcodes).to_numpy(dtype=float)
        finite = np.isfinite(purity)
        finite_fraction = float(finite.mean()) if len(finite) else 0.0
        finite_values = purity[finite]
        if finite_values.size:
            above_min_fraction = float(np.mean(finite_values > np.min(finite_values)))
            iqr = float(np.percentile(finite_values, 75) - np.percentile(finite_values, 25))
        else:
            above_min_fraction, iqr = 0.0, np.nan
        corr_mask = finite & np.isfinite(log_umi)
        correlation = np.nan
        if corr_mask.sum() >= 2 and np.ptp(purity[corr_mask]) > 0 and np.ptp(log_umi[corr_mask]) > 0:
            correlation = float(spearmanr(purity[corr_mask], log_umi[corr_mask]).statistic)
        p1 = finite_fraction >= limits["min_purity_finite_fraction"]
        p2 = (
            above_min_fraction >= limits["min_purity_above_min_fraction"]
            and np.isfinite(iqr) and iqr > 0
        )
        p3 = bool(
            np.isfinite(correlation)
            and abs(correlation) < limits["max_abs_purity_umi_spearman"]
        )
        purity_checks.update({
            "P1_finite_fraction": bool(p1),
            "P2_dynamic_range": bool(p2),
            "P3_umi_independence": bool(p3),
        })
        clone_auc = np.nan
        if cnv_pass:
            group_mask = finite & np.isin(labels, [reference_clone] + tumor_clones)
            binary = np.isin(labels[group_mask], tumor_clones).astype(int)
            clone_auc = _safe_auc(binary, purity[group_mask])
            purity_checks["P4_clone_discrimination"] = bool(
                np.isfinite(clone_auc) and clone_auc >= limits["min_purity_clone_auc"]
            )
        purity_metrics.update({
            "finite_fraction": finite_fraction,
            "above_min_fraction": above_min_fraction,
            "iqr": _as_python(iqr),
            "spearman_with_log_umi": _as_python(correlation),
            "tumor_vs_reference_auc": _as_python(clone_auc),
        })

    applicable_purity = [value for value in purity_checks.values() if value is not None]
    purity_pass = bool(purity_or_None is not None and all(applicable_purity))
    return {
        "schema": "sparcal_calicost_gate_v1",
        "sample_id": sample_id,
        "thresholds": limits,
        "n_section_spots": int(total_spots),
        "n_spots_with_finite_umi": int(np.isfinite(umi).sum()),
        "clone_metrics": clone_metrics,
        "dropped_clones": dropped,
        "surviving_clones": surviving,
        "reference_clone": reference_clone,
        "tumor_clones": tumor_clones,
        "effective_labels": effective_labels,
        "cnv_checks": {
            "C1_reference_and_tumor_profiles": bool(c1),
            "C2_clone_filters_applied": True,
            "C3_posterior_confidence": bool(c3),
        },
        "median_max_posterior": _as_python(median_max_posterior),
        "cnv_pass": cnv_pass,
        "purity_metrics": purity_metrics,
        "purity_checks": purity_checks,
        "purity_pass": purity_pass,
    }


def evaluate_gate_files(
    clone_labels_path: str,
    cnv_segments_path: str,
    posterior_path: str,
    purity_path: Optional[str],
    sample_id: Optional[str],
    spot_umi_path: str,
    thresholds: Optional[Mapping[str, float]] = None,
) -> dict:
    clone_labels = pd.read_csv(clone_labels_path, sep="\t")
    cnv_segments = pd.read_csv(cnv_segments_path, sep="\t")
    posterior = np.load(posterior_path, allow_pickle=False)
    purity = pd.read_csv(purity_path, sep="\t") if purity_path else None
    umi_frame = pd.read_csv(spot_umi_path, sep="\t")
    if not {"barcode", "total_umi"}.issubset(umi_frame.columns):
        raise ValueError("spot UMI TSV must contain barcode and total_umi columns")
    spot_umi = pd.Series(
        pd.to_numeric(umi_frame["total_umi"], errors="coerce").to_numpy(float),
        index=umi_frame["barcode"].astype(str),
    )
    return evaluate_gate(
        clone_labels, cnv_segments, posterior, purity, sample_id, spot_umi, thresholds
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone_labels", required=True)
    parser.add_argument("--cnv_segments", required=True)
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--purity")
    parser.add_argument(
        "--sample_id", required=True,
        help="Joint-run suffix, or NONE for explicitly unsuffixed single-section labels",
    )
    parser.add_argument("--spot_umi_tsv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate_gate_files(
        args.clone_labels,
        args.cnv_segments,
        args.posterior,
        args.purity,
        args.sample_id,
        args.spot_umi_tsv,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"{args.sample_id}: CNV {'PASS' if result['cnv_pass'] else 'FAIL'}; "
        f"reference={result['reference_clone']}; tumors={result['tumor_clones']}; "
        f"dropped={sorted(map(int, result['dropped_clones']))}; "
        f"purity {'PASS' if result['purity_pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
