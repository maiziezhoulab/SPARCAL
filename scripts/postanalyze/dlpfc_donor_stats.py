#!/usr/bin/env python3
"""Donor-aware reanalysis of DLPFC comparisons used in the manuscript.

The 12 spatialLIBD sections are four serial sections from each of three donors.
The script reports section-level paired tests for reproducibility, exact tests on
three donor-mean differences, and a random-intercept mixed-model sensitivity
analysis on section-level paired differences.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
import statsmodels.api as sm


REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")
DEFAULT_OUT = REPO / "data/dlpfc_donor_stats_2026-08-23"

DONOR_BY_SECTION = {
    "151507": "Donor 1", "151508": "Donor 1",
    "151509": "Donor 1", "151510": "Donor 1",
    "151669": "Donor 2", "151670": "Donor 2",
    "151671": "Donor 2", "151672": "Donor 2",
    "151673": "Donor 3", "151674": "Donor 3",
    "151675": "Donor 3", "151676": "Donor 3",
}

PRETTY = {
    "somtop25_bin250kb": "SPARCAL spatially augmented, 250 kb",
    "defined_bin250kb": "SPARCAL 1KGP-only, 250 kb",
    "defined1000G": "SPARCAL 1KGP-only, unbinned",
    "gatk_bin250kb": "GATK, 250 kb",
    "strelka2_bin250kb": "Strelka2, 250 kb",
    "gatk": "GATK, unbinned",
    "strelka2": "Strelka2, unbinned",
    "gene_expr": "Gene expression",
}

# A is the focal representation and effect = A - B.
COMPARISONS = [
    ("knn_spatial_aug_vs_gene", "knn", "somtop25_bin250kb", "gene_expr", "main Fig. 2"),
    ("knn_spatial_aug_vs_gatk250", "knn", "somtop25_bin250kb", "gatk_bin250kb", "main Fig. 2"),
    ("knn_spatial_aug_vs_strelka250", "knn", "somtop25_bin250kb", "strelka2_bin250kb", "main Fig. 2"),
    ("ari_spatial_aug_vs_gatk250", "ari", "somtop25_bin250kb", "gatk_bin250kb", "main Fig. 2"),
    ("ari_spatial_aug_vs_gene", "ari", "somtop25_bin250kb", "gene_expr", "main Fig. 2"),
    ("ari_spatial_aug_vs_defined250", "ari", "somtop25_bin250kb", "defined_bin250kb", "main Results"),
    ("ari_defined250_vs_gatk250", "ari", "defined_bin250kb", "gatk_bin250kb", "SI Fig. S2"),
    ("ari_defined250_vs_strelka250", "ari", "defined_bin250kb", "strelka2_bin250kb", "SI Fig. S2"),
    ("ari_gatk250_vs_strelka250", "ari", "gatk_bin250kb", "strelka2_bin250kb", "SI Fig. S2"),
    ("ari_defined250_vs_defined_unbinned", "ari", "defined_bin250kb", "defined1000G", "SI Fig. S2"),
    ("ari_defined_unbinned_vs_gatk", "ari", "defined1000G", "gatk", "SI Fig. S1"),
    ("ari_defined_unbinned_vs_strelka", "ari", "defined1000G", "strelka2", "SI Fig. S1"),
]


def exact_wilcoxon(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x) or np.allclose(x, 0):
        return math.nan, math.nan
    result = wilcoxon(
        x, alternative="two-sided", zero_method="wilcox", method="auto"
    )
    return float(result.statistic), float(result.pvalue)


def mixed_intercept(differences: np.ndarray, donors: np.ndarray) -> dict:
    """Fit difference ~ 1 + (1|donor), trying stable optimizers in order."""
    differences = np.asarray(differences, dtype=float)
    donors = np.asarray(donors, dtype=str)
    keep = np.isfinite(differences)
    y, groups = differences[keep], donors[keep]
    exog = np.ones((len(y), 1), dtype=float)
    last_error = ""
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = sm.MixedLM(y, exog, groups=groups).fit(
                    reml=True, method=method, disp=False, maxiter=2000
                )
            effect = float(fit.fe_params[0])
            se = float(fit.bse_fe[0])
            z_value = effect / se if se > 0 else math.nan
            p_value = (
                float(2 * norm.sf(abs(z_value)))
                if np.isfinite(z_value) else math.nan
            )
            donor_var = float(np.asarray(fit.cov_re)[0, 0])
            status = [f"optimizer={method}", f"converged={bool(fit.converged)}"]
            if caught:
                status.append(
                    "warnings=" + " | ".join(str(w.message) for w in caught)
                )
            return {
                "lmm_effect": effect,
                "lmm_se": se,
                "lmm_ci_low": effect - 1.96 * se,
                "lmm_ci_high": effect + 1.96 * se,
                "lmm_p": p_value,
                "lmm_donor_variance": donor_var,
                "lmm_status": "; ".join(status),
            }
        except Exception as exc:
            last_error = f"{method}: {exc}"
    return {
        "lmm_effect": math.nan,
        "lmm_se": math.nan,
        "lmm_ci_low": math.nan,
        "lmm_ci_high": math.nan,
        "lmm_p": math.nan,
        "lmm_donor_variance": math.nan,
        "lmm_status": "failed; " + last_error,
    }


def load_values() -> pd.DataFrame:
    selected = pd.read_csv(
        REPO
        / "data/dlpfc/clustering_benchmark/somtop25_selected_performance_source.csv",
        dtype={"section": str},
    )[["section", "modality", "metric", "value", "n_runs"]]

    ari = pd.read_csv(
        REPO / "data/dlpfc/clustering_benchmark/ari_matrix_mean.csv",
        dtype={"section": str},
    ).melt(id_vars="section", var_name="modality", value_name="value")
    ari["metric"] = "ari"
    ari["n_runs"] = 10

    # Prefer the exact source snapshot used for main Fig. 2 where duplicated.
    all_values = pd.concat([selected, ari], ignore_index=True)
    all_values["priority"] = np.where(all_values.index < len(selected), 0, 1)
    all_values = (
        all_values.sort_values("priority")
        .drop_duplicates(["section", "modality", "metric"], keep="first")
        .drop(columns="priority")
    )
    all_values["donor"] = all_values["section"].map(DONOR_BY_SECTION)
    if all_values["donor"].isna().any():
        missing = sorted(
            all_values.loc[all_values["donor"].isna(), "section"].unique()
        )
        raise ValueError(f"No donor mapping for sections: {missing}")
    return all_values


def analyse(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    test_rows = []
    for comparison, metric, a, b, manuscript_location in COMPARISONS:
        subset = values[
            (values.metric == metric) & values.modality.isin([a, b])
        ]
        pivot = subset.pivot(
            index=["section", "donor"], columns="modality", values="value"
        )
        pivot = pivot.dropna(subset=[a, b]).reset_index()
        pivot["difference"] = pivot[a] - pivot[b]
        pivot["comparison"] = comparison
        pivot["metric"] = metric
        pivot["modality_a"] = a
        pivot["modality_b"] = b
        pivot["label_a"] = PRETTY[a]
        pivot["label_b"] = PRETTY[b]
        detail_rows.extend(pivot.to_dict("records"))

        section_stat, section_p = exact_wilcoxon(
            pivot["difference"].to_numpy()
        )
        donor_means = pivot.groupby("donor", sort=True)["difference"].mean()
        donor_stat, donor_p = exact_wilcoxon(donor_means.to_numpy())
        lmm = mixed_intercept(
            pivot["difference"].to_numpy(), pivot["donor"].to_numpy()
        )
        test_rows.append({
            "comparison": comparison,
            "manuscript_location": manuscript_location,
            "metric": metric,
            "modality_a": a,
            "modality_b": b,
            "label_a": PRETTY[a],
            "label_b": PRETTY[b],
            "effect_definition": "A minus B",
            "n_sections": int(len(pivot)),
            "n_donors": int(len(donor_means)),
            "section_mean_effect": float(pivot["difference"].mean()),
            "section_median_effect": float(pivot["difference"].median()),
            "section_wilcoxon_stat": section_stat,
            "section_wilcoxon_p": section_p,
            "donor_mean_effect": float(donor_means.mean()),
            "donor_effect_min": float(donor_means.min()),
            "donor_effect_max": float(donor_means.max()),
            "donors_positive": int((donor_means > 0).sum()),
            "donor_wilcoxon_stat": donor_stat,
            "donor_wilcoxon_p": donor_p,
            **lmm,
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(test_rows)


def write_results(outdir: Path, tests: pd.DataFrame) -> None:
    columns = [
        "comparison", "section_mean_effect", "section_wilcoxon_p",
        "donors_positive", "donor_wilcoxon_p", "lmm_effect", "lmm_p",
    ]
    display = tests[columns].copy()
    numeric = [
        "section_mean_effect", "section_wilcoxon_p", "donor_wilcoxon_p",
        "lmm_effect", "lmm_p",
    ]
    for column in numeric:
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4g}"
        )
    headers = list(display.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        markdown_rows.append("| " + " | ".join(map(str, row)) + " |")
    table = "\n".join(markdown_rows)
    positive = tests.loc[tests.donors_positive == 3, "comparison"].tolist()
    not_all = tests.loc[tests.donors_positive < 3, "comparison"].tolist()

    text = f"""# Donor-aware DLPFC reanalysis — 2026-08-23

## Question

The manuscript treated 12 spatialLIBD sections as independent replicates. They are
four serial sections from each of three donors. This analysis retains section-level
effect sizes but recalculates inference using donor as the replication unit.

## Donor mapping

- Donor 1: 151507, 151508, 151509, 151510
- Donor 2: 151669, 151670, 151671, 151672
- Donor 3: 151673, 151674, 151675, 151676

## Results

{table}

All three donor means are positive for: {', '.join(positive) if positive else 'none'}.

At least one donor mean is non-positive for: {', '.join(not_all) if not_all else 'none'}.

With only three donors, the smallest attainable two-sided exact Wilcoxon P value is
0.25. Consequently, no donor-mean test can reach 0.05 regardless of effect
consistency. The mixed model is a sensitivity analysis, not a way to manufacture
precision from three biological units. Manuscript claims should lead with effect
sizes and the three donor means; section-level P values are descriptive analyses of
the 12 sections.

## Methods

For every comparison, the effect is modality A minus modality B. Section-level P
values reproduce the existing paired two-sided Wilcoxon analysis. Donor-level P
values use the same test on the three donor-mean paired differences. The mixed model
fits the section-level paired difference as \`difference ~ 1 + (1 | donor)\` by REML.
This is a random-intercept model with sections nested within donor through the
residual term. Optimizer and convergence information are retained in
\`donor_level_tests.csv\`.

## Sources

- \`data/dlpfc/clustering_benchmark/somtop25_selected_performance_source.csv\`
- \`data/dlpfc/clustering_benchmark/ari_matrix_mean.csv\`
- Script: \`scripts/postanalyze/dlpfc_donor_stats.py\`
"""
    (outdir / "RESULTS.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    values = load_values()
    detail, tests = analyse(values)
    detail.to_csv(args.outdir / "comparison_values.csv", index=False)
    tests.to_csv(args.outdir / "donor_level_tests.csv", index=False)
    write_results(args.outdir, tests)
    print(tests[[
        "comparison", "section_mean_effect", "section_wilcoxon_p",
        "donors_positive", "donor_wilcoxon_p", "lmm_p",
    ]].to_string(index=False))
    print(f"\nWrote donor-aware results to {args.outdir}")


if __name__ == "__main__":
    main()
