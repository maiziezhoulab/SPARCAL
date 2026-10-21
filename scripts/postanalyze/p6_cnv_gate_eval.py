#!/usr/bin/env python3
"""Evaluate the paired P6 spatial-only and gated-CNV SPARCAL arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
ARM_CONFIG = {
    "A_spatial_only": {
        "tag": "baseQ0mapQ0_cnvgate_spatial_only",
        "model": "SPARCAL_cnvgate_spatial_only",
    },
    "B_gated_cnv": {
        "tag": "baseQ0mapQ0_cnvgate_cnv",
        "model": "SPARCAL_cnvgate_cnv",
    },
}
TRUTH_PATH = PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv"
PATHOLOGY_ROOT = (
    PROJECT / "data/spatialsnv_reanalysis_2026-07-17/p6_gt/extracted"
    / "P6_rep1_SPARCAL_somatic"
)
DEFAULT_OUT = PROJECT / "data/p6_cnv_gate_eval_2026-10-19"


def variant_key(chromosome, position, ref, alt) -> str:
    chromosome = str(chromosome).removeprefix("chr")
    return f"{chromosome}_{int(position)}_{str(ref).upper()}_{str(alt).upper()}"


def load_call_table(path: Path) -> Tuple[pd.DataFrame, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype={"chrom": str, "ref": str, "alt": str})
    required = {"chrom", "pos", "ref", "alt"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required variant columns")
    keys = {
        variant_key(*row)
        for row in frame[["chrom", "pos", "ref", "alt"]].itertuples(index=False, name=None)
    }
    if len(keys) != len(frame):
        raise ValueError(f"{path} contains duplicate allele-exact variant keys")
    return frame, keys


def arm_paths(section: int, arm: str) -> Dict[str, Path]:
    cfg = ARM_CONFIG[arm]
    section_root = PROJECT / f"data/P6_tumor/{section}"
    output = section_root / "spatial_filter_purity" / cfg["tag"]
    matrix = section_root / "matrix" / f"P6_TUMOR_{section}_{cfg['model']}_somatic_matrix.pkl"
    return {
        "output": output,
        "complete": output / "RUN_COMPLETE",
        "calls": output / "somatic/somatic_variants.txt",
        "matrix": matrix,
        "matrix_complete": Path(str(matrix) + ".complete"),
        "gate": output / "calicost_gate.json",
    }


def preflight() -> Tuple[dict, dict]:
    """Verify/load both unlabeled arm outputs before opening truth or pathology."""
    paths: dict = {}
    calls: dict = {}
    missing = []
    for section in (1, 2):
        paths[section] = {}
        calls[section] = {}
        for arm in ARM_CONFIG:
            item = arm_paths(section, arm)
            paths[section][arm] = item
            required = [item["complete"], item["calls"], item["matrix"], item["matrix_complete"]]
            missing.extend(str(path) for path in required if not path.is_file())
            if all(path.is_file() for path in required):
                calls[section][arm] = load_call_table(item["calls"])
    if missing:
        raise FileNotFoundError(
            "both arms must exist for both sections before labels are read; missing:\n  "
            + "\n  ".join(missing)
        )
    for section in (1, 2):
        sizes = {arm: len(calls[section][arm][1]) for arm in ARM_CONFIG}
        if len(set(sizes.values())) != 1:
            raise AssertionError(f"section {section} arm set sizes differ: {sizes}")
    return paths, calls


def safe_auc(labels: Iterable[bool], values: Iterable[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2 or np.unique(labels[finite]).size != 2:
        return np.nan
    return float(roc_auc_score(labels[finite], values[finite]))


def exact_mcnemar(a: set[str], b: set[str], truth: set[str]) -> dict:
    a_only = len((a & truth) - b)
    b_only = len((b & truth) - a)
    discordant = a_only + b_only
    pvalue = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {"a_only_truth_hits": a_only, "b_only_truth_hits": b_only,
            "discordant": discordant, "exact_pvalue": pvalue}


def load_positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"barcode", "in_tissue", "array_row", "array_col"}
    if not expected.issubset(frame.columns):
        frame = pd.read_csv(path, header=None)
        frame = frame.rename(columns={0: "barcode", 1: "in_tissue", 2: "array_row", 3: "array_col"})
    return frame.loc[frame["in_tissue"].astype(int).eq(1)].copy()


def residualize_log_burden(burden: pd.Series, umi: pd.Series) -> pd.Series:
    aligned = pd.concat([burden.rename("burden"), umi.rename("umi")], axis=1)
    valid = aligned.notna().all(axis=1) & aligned["umi"].ge(0) & aligned["burden"].ge(0)
    result = pd.Series(np.nan, index=aligned.index, dtype=float)
    x = np.log1p(aligned.loc[valid, "umi"].to_numpy(float))
    y = np.log1p(aligned.loc[valid, "burden"].to_numpy(float))
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    result.loc[valid] = y - design @ beta
    return result


def block_bootstrap(
    labels: pd.Series,
    score_a: pd.Series,
    score_b: pd.Series,
    positions: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict:
    table = positions.set_index("barcode")[["array_row", "array_col"]].copy()
    table["label"] = labels.reindex(table.index)
    table["a"] = score_a.reindex(table.index)
    table["b"] = score_b.reindex(table.index)
    table = table.dropna()
    table["block"] = (
        (table["array_row"].astype(int) // 8).astype(str) + "_"
        + (table["array_col"].astype(int) // 8).astype(str)
    )
    blocks = [group.index.to_numpy() for _, group in table.groupby("block", sort=True)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        index = np.concatenate([blocks[value] for value in sampled])
        y = table.loc[index, "label"].to_numpy(int)
        if np.unique(y).size != 2:
            continue
        deltas.append(
            roc_auc_score(y, table.loc[index, "b"])
            - roc_auc_score(y, table.loc[index, "a"])
        )
    values = np.asarray(deltas, dtype=float)
    return {
        "n_bootstrap_requested": reps,
        "n_bootstrap_valid": len(values),
        "delta_b_minus_a": safe_auc(table["label"], table["b"]) - safe_auc(table["label"], table["a"]),
        "ci_low": float(np.percentile(values, 2.5)) if len(values) else np.nan,
        "ci_high": float(np.percentile(values, 97.5)) if len(values) else np.nan,
    }


def detect_panel_annotation(frame: pd.DataFrame):
    candidates = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("1000g", "panel_member", "germline_panel"))
    ]
    if candidates:
        return candidates[0]
    # The production text output documents race=defined as panel-concordant.
    return "race" if "race" in frame.columns else None


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    display = frame.copy().replace({np.nan: "NA"})
    columns = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    render = lambda row: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    return "\n".join([
        render(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
        *(render(row) for row in rows),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.out}")

    paths, calls = preflight()
    # Labels are intentionally opened only after preflight succeeds.
    truth_frame = pd.read_csv(TRUTH_PATH, sep="\t")
    truth_frame = truth_frame.loc[truth_frame["sample"].eq("P6")].copy()
    truth_frame["variant"] = [
        variant_key(*row)
        for row in truth_frame[["chrom_hg19", "pos_hg19", "ref", "alt"]].itertuples(index=False, name=None)
    ]
    truth_all = set(truth_frame["variant"])

    call_rows = []
    mcnemar_rows = []
    for section in (1, 2):
        callable_col_ref = f"st_rep{section}_ref"
        callable_col_alt = f"st_rep{section}_alt"
        callable_truth = None
        if {callable_col_ref, callable_col_alt}.issubset(truth_frame.columns):
            depth = (
                pd.to_numeric(truth_frame[callable_col_ref], errors="coerce").fillna(0)
                + pd.to_numeric(truth_frame[callable_col_alt], errors="coerce").fillna(0)
            )
            callable_truth = set(truth_frame.loc[depth.gt(0), "variant"])
        for arm in ARM_CONFIG:
            callset = calls[section][arm][1]
            call_rows.append({
                "section": section,
                "arm": arm,
                "n_calls": len(callset),
                "wes_truth_total": len(truth_all),
                "wes_hits": len(callset & truth_all),
                "callable_denominator": len(callable_truth) if callable_truth is not None else np.nan,
                "callable_hits": len(callset & callable_truth) if callable_truth is not None else np.nan,
            })
        paired = exact_mcnemar(
            calls[section]["A_spatial_only"][1], calls[section]["B_gated_cnv"][1], truth_all
        )
        mcnemar_rows.append({"section": section, **paired})
    pooled_a = calls[1]["A_spatial_only"][1] | calls[2]["A_spatial_only"][1]
    pooled_b = calls[1]["B_gated_cnv"][1] | calls[2]["B_gated_cnv"][1]
    mcnemar_rows.append({"section": "pooled_union", **exact_mcnemar(pooled_a, pooled_b, truth_all)})

    replicate_rows = []
    for arm in ARM_CONFIG:
        left, right = calls[1][arm][1], calls[2][arm][1]
        union = left | right
        replicate_rows.append({
            "arm": arm, "rep1_n": len(left), "rep2_n": len(right),
            "intersection": len(left & right), "union": len(union),
            "jaccard": len(left & right) / len(union) if union else np.nan,
        })

    leakage_rows = []
    leakage_note = "Skipped: no 1kG/germline panel-membership annotation exists in somatic variant outputs."
    for section in (1, 2):
        for arm in ARM_CONFIG:
            frame = calls[section][arm][0]
            column = detect_panel_annotation(frame)
            if column:
                normalized = frame[column].astype(str).str.lower()
                if column == "race":
                    values = normalized.isin({"defined", "panel_concordant", "1000g"})
                else:
                    values = normalized.isin({"1", "true", "yes", "y"})
                leakage_rows.append({"section": section, "arm": arm, "annotation": column,
                                     "panel_hits": int(values.sum()), "fraction": float(values.mean())})
    if leakage_rows:
        leakage_note = "Panel-membership leakage was evaluated from existing variant annotations."

    groups = pd.read_csv(PATHOLOGY_ROOT / "tumor_groups.csv", dtype=str)
    truth_spots = set(groups.loc[groups["profile"].eq("GroundTruth"), "barcode"])
    positions = load_positions(PATHOLOGY_ROOT / "tissue_positions.csv")
    tissue = pd.Index(positions["barcode"].astype(str), name="barcode")
    coverage_frame = pd.read_csv(PATHOLOGY_ROOT / "spot_coverage.csv")
    umi = pd.Series(
        pd.to_numeric(coverage_frame["total_umi"], errors="coerce").to_numpy(float),
        index=coverage_frame["barcode"].astype(str), name="umi",
    ).reindex(tissue)
    labels = pd.Series(tissue.isin(truth_spots).astype(int), index=tissue, name="gt")

    burdens = {}
    residuals = {}
    pathology_rows = []
    for arm in ARM_CONFIG:
        matrix = pd.read_pickle(paths[1][arm]["matrix"])
        burden = (matrix > 0).sum(axis=1).astype(float).reindex(tissue, fill_value=0.0)
        residual = residualize_log_burden(burden, umi)
        burdens[arm] = burden
        residuals[arm] = residual
        pathology_rows.append({"arm": arm, "metric": "raw_burden", "umi_quartile": "all",
                               "n_spots": len(tissue), "auc": safe_auc(labels, burden)})
        pathology_rows.append({"arm": arm, "metric": "coverage_residual", "umi_quartile": "all",
                               "n_spots": int(residual.notna().sum()), "auc": safe_auc(labels, residual)})
        valid_umi = umi.notna()
        quartiles = pd.qcut(umi[valid_umi].rank(method="first"), 4, labels=False)
        for quartile in range(4):
            members = quartiles.index[quartiles.eq(quartile)]
            pathology_rows.append({
                "arm": arm, "metric": "raw_burden", "umi_quartile": quartile + 1,
                "n_spots": len(members), "auc": safe_auc(labels.reindex(members), burden.reindex(members)),
            })

    bootstrap_rows = []
    for metric, scores_a, scores_b in (
        ("raw_burden", burdens["A_spatial_only"], burdens["B_gated_cnv"]),
        ("coverage_residual", residuals["A_spatial_only"], residuals["B_gated_cnv"]),
    ):
        bootstrap_rows.append({
            "metric": metric,
            **block_bootstrap(labels, scores_a, scores_b, positions, args.bootstrap_reps, args.seed),
        })

    gate = json.loads(paths[1]["B_gated_cnv"]["gate"].read_text())
    tumor_spots = {
        barcode for barcode, clone in gate["effective_labels"].items()
        if clone in gate["tumor_clones"]
    }
    gate_score = pd.Series(tissue.isin(tumor_spots).astype(float), index=tissue)
    gate_baseline = pd.DataFrame([{
        "section": 1, "reference_clone": gate["reference_clone"],
        "tumor_clones": ",".join(map(str, gate["tumor_clones"])),
        "n_gate_tumor_spots": int(gate_score.sum()),
        "auc_gate_tumor_clone_vs_gt": safe_auc(labels, gate_score),
    }])

    args.out.mkdir(parents=True, exist_ok=True)
    tables = {
        "callset_metrics.csv": pd.DataFrame(call_rows),
        "mcnemar.csv": pd.DataFrame(mcnemar_rows),
        "replicate_concordance.csv": pd.DataFrame(replicate_rows),
        "pathology_auc.csv": pd.DataFrame(pathology_rows),
        "pathology_block_bootstrap.csv": pd.DataFrame(bootstrap_rows),
        "gate_clone_baseline.csv": gate_baseline,
        "germline_panel_leakage.csv": pd.DataFrame(leakage_rows),
    }
    for name, table in tables.items():
        table.to_csv(args.out / name, index=False, mode="x")

    call_table = tables["callset_metrics.csv"]
    path_table = tables["pathology_auc.csv"]
    leakage_report = leakage_note
    if leakage_rows:
        leakage_report += "\n\n" + markdown_table(tables["germline_panel_leakage.csv"])
    report = [
        "# P6 CalicoST CNV-gate evaluation", "",
        "Both arms were required for both sections before any truth/pathology labels were read.", "",
        "## Call-set size and WES recovery", "",
        markdown_table(call_table), "",
        "## Exact paired McNemar tests", "",
        markdown_table(tables["mcnemar.csv"]), "",
        "## Replicate concordance", "",
        markdown_table(tables["replicate_concordance.csv"]), "",
        "## 1kG/germline leakage proxy", "", leakage_report, "",
        "## P6 replicate-1 pathology", "",
        markdown_table(path_table), "",
        "### Paired 8x8 spatial-block bootstrap (B - A)", "",
        markdown_table(tables["pathology_block_bootstrap.csv"]), "",
        "### Trivial gate-clone baseline", "",
        markdown_table(gate_baseline), "",
        f"Bootstrap: {args.bootstrap_reps} replicates; fixed seed {args.seed}.", "",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(report))
    print(f"Wrote evaluation to {args.out}")


if __name__ == "__main__":
    main()
