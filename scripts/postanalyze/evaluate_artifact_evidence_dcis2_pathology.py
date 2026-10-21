#!/usr/bin/env python3
"""Evaluate fixed DCIS2 artifact evidence against pathology, with and without UMI.

This is an endpoint-first, descriptive ablation of the fixed 500-site somatic
pilot.  It never selects a threshold or changes a VCF.  It reports all of:

* raw candidate alt-UMI evidence (which intentionally retains coverage signal),
* evidence per 10,000 total UMIs,
* total UMI alone, and
* joint total-UMI plus candidate-evidence models.

The joint models answer whether combining UMI with evidence is useful; the
normalized models are a diagnostic of what remains after accounting for UMI.
Both views are reported because coverage can itself be biologically useful.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
VIEWER = Path("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL")
PILOT = PROJECT / "data/dcis2/artifact_evidence/v2_pilot_2026-07-16_dcis2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=PILOT)
    parser.add_argument("--viewer-root", type=Path, default=VIEWER)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def clip_barcodes(viewer_root: Path) -> tuple[list[str], set[str]]:
    pos = pd.read_csv(viewer_root / "tissue_positions.csv")
    pos = pos[pos["in_tissue"] == 1].copy()
    pos["x"] = pos["pxl_col_in_fullres"]
    pos["y"] = pos["pxl_row_in_fullres"]
    x_cut = pos["x"].min() + 0.4 * (pos["x"].max() - pos["x"].min())
    y_cut = pos["y"].min() + 0.4 * (pos["y"].max() - pos["y"].min())
    clip = set(pos.loc[(pos["x"] >= x_cut) & (pos["y"] >= y_cut), "barcode"])
    groups = pd.read_csv(viewer_root / "tumor_groups.csv")
    tumor = set(groups.loc[groups["profile"] == "Ground Truth", "barcode"])
    coverage = pd.read_csv(viewer_root / "spot_coverage.csv").set_index("barcode")["total_umi"]
    return sorted(clip.intersection(coverage.index)), tumor


def spot_scores(pilot_root: Path) -> pd.DataFrame:
    evidence = pd.read_csv(
        pilot_root / "evidence/site_artifact_evidence.tsv.gz",
        sep="\t",
        usecols=["candidate_id", "candidate_sources", "artifact_evidence_status"],
    )
    spots = pd.read_csv(
        pilot_root / "features/spot_features.tsv.gz",
        sep="\t",
        usecols=["candidate_id", "barcode", "alt_n", "alt_umi_n"],
    )
    merged = spots.merge(evidence, on="candidate_id", how="inner", validate="many_to_one")
    somatic = merged[merged["candidate_sources"] == "somatic"].copy()
    if somatic.empty:
        raise ValueError("No somatic observations in the fixed pilot.")
    all_score = somatic.groupby("barcode")["alt_umi_n"].sum().rename("raw_somatic_alt_umi")
    pass_score = (
        somatic.loc[somatic["artifact_evidence_status"] == "read_evidence_pass"]
        .groupby("barcode")["alt_umi_n"].sum().rename("pass_somatic_alt_umi")
    )
    all_sites = somatic.groupby("barcode")["candidate_id"].nunique().rename("raw_somatic_sites")
    pass_sites = (
        somatic.loc[somatic["artifact_evidence_status"] == "read_evidence_pass"]
        .groupby("barcode")["candidate_id"].nunique().rename("pass_somatic_sites")
    )
    return pd.concat([all_score, pass_score, all_sites, pass_sites], axis=1).fillna(0)


def mean_cv_auc(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> tuple[float, float]:
    splitter = RepeatedStratifiedKFold(
        n_splits=args.splits, n_repeats=args.repeats, random_state=args.seed
    )
    values: list[float] = []
    for train, test in splitter.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(X.iloc[train], y.iloc[train])
        values.append(roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring pandas' tabulate extra."""
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.splits < 2 or args.repeats < 1:
        raise ValueError("--splits must be >=2 and --repeats must be >=1.")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace an existing endpoint result: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    barcodes, tumor_set = clip_barcodes(args.viewer_root)
    coverage = pd.read_csv(args.viewer_root / "spot_coverage.csv").set_index("barcode")["total_umi"]
    scores = spot_scores(args.pilot_root)
    table = pd.DataFrame(index=barcodes)
    table["tumor"] = [int(barcode in tumor_set) for barcode in table.index]
    table["total_umi"] = coverage.reindex(table.index).astype(float)
    table = table.join(scores, how="left").fillna(0)
    if (table["total_umi"] <= 0).any():
        raise ValueError("Clip contains non-positive total UMI values.")
    for prefix in ("raw", "pass"):
        table[f"{prefix}_alt_umi_per10k"] = 1e4 * table[f"{prefix}_somatic_alt_umi"] / table["total_umi"]
    table["log_total_umi"] = np.log1p(table["total_umi"])

    # The raw and normalized scores use the same fixed candidate sample.  No
    # thresholds are fitted or selected using pathology labels.
    features = {
        "UMI_only": ["log_total_umi"],
        "raw_all_somatic_alt_UMI": ["raw_somatic_alt_umi"],
        "raw_pass_somatic_alt_UMI": ["pass_somatic_alt_umi"],
        "UMI_normalized_all_somatic_alt_UMI": ["raw_alt_umi_per10k"],
        "UMI_normalized_pass_somatic_alt_UMI": ["pass_alt_umi_per10k"],
        "UMI_plus_raw_all_somatic_alt_UMI": ["log_total_umi", "raw_somatic_alt_umi"],
        "UMI_plus_raw_pass_somatic_alt_UMI": ["log_total_umi", "pass_somatic_alt_umi"],
    }
    y = table["tumor"].astype(int)
    rows: list[dict[str, object]] = []
    for model_name, columns in features.items():
        X = np.log1p(table[columns])
        mean_auc, sd_auc = mean_cv_auc(X, y, args)
        full_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        full_model.fit(X, y)
        in_sample_auc = roc_auc_score(y, full_model.predict_proba(X)[:, 1])
        rows.append(
            {
                "model": model_name,
                "features": "+".join(columns),
                "uses_total_umi": "total_umi" in "+".join(columns),
                "cv_mean_auc": round(mean_auc, 4),
                "cv_sd_auc": round(sd_auc, 4),
                "in_sample_auc_diagnostic_only": round(float(in_sample_auc), 4),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "model_auc.csv", index=False)
    table.reset_index(names="barcode").to_csv(args.out_dir / "spot_scores.csv", index=False)

    with (args.out_dir / "README.md").open("w") as handle:
        handle.write("# DCIS2 pathology endpoint: artifact-evidence ablation\n\n")
        handle.write(
            "Fixed 500-site somatic pilot; no VCF was filtered and no score threshold was chosen using pathology. "
            "The primary metric is repeated stratified cross-validated AUC.\n\n"
        )
        handle.write(
            f"Analysis window: {len(table)} spots; {int(y.sum())} pathology-tumor and "
            f"{len(table) - int(y.sum())} pathology-normal spots.\n\n"
        )
        handle.write(markdown_table(results))
        handle.write("\n\nRaw models retain UMI-associated signal; normalized models are diagnostic only. "
                     "Joint models preserve UMI and test the utility of adding fixed candidate evidence.\n")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
