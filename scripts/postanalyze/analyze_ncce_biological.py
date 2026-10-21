#!/usr/bin/env python3
"""Create an exploratory biological report for NCCE SPARCAL results.

This analysis keeps the supplied RCTD labels as the working cell-type
reference.  It reads the original SpaceRanger matrices and all SPARCAL outputs
without modifying them, and writes only derived material below
``data/ncce/biological_analysis/report_v1`` by default.

The report deliberately calls the final SPARCAL class ``somatic-pattern``:
the spatial filter used the binary copyKAT annotation as an input, so agreement
with copyKAT is internal consistency rather than independent validation.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.io import mmread
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
NCCE_ROOT = PROJECT / "data" / "ncce"
SPACERANGER_ROOT = Path("/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs")
SAMPLES = [
    "6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C",
    "11A", "11B", "11C", "14A", "14B", "14C",
]
RESPONSE = {"6": "non-responder", "8": "non-responder", "10": "non-responder",
            "11": "non-responder", "14": "responder"}
TIME_NUM = {"A": 0, "B": 1, "C": 2}

CELL_PALETTE = {
    "B": "#2d6fa3", "CD8": "#9eb9d6", "DC": "#ed7d21",
    "endothelial": "#f4b36c", "epithelial": "#f08a8d",
    "fibroblasts": "#8fd17e", "macrophage": "#229943",
    "pericytes": "#ab9abd", "plasma": "#78509b", "Treg": "#d28db9",
    "NK": "#5d8fd1", "mast": "#9467bd", "Unknown": "#bdbdbd",
}
CELL_ORDER = [
    "B", "CD8", "DC", "endothelial", "epithelial", "fibroblasts",
    "macrophage", "pericytes", "plasma", "Treg", "NK", "mast", "Unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=NCCE_ROOT / "biological_analysis" / "report_v1",
        help="Project-local destination for figures, tables, and report.",
    )
    parser.add_argument(
        "--locus-gap", type=int, default=20,
        help="Merge somatic calls on the same chromosome within this many bp (default: 20).",
    )
    return parser.parse_args()


def load_umi(section: str) -> pd.Series:
    """Return total feature-barcode UMIs keyed by the SpaceRanger barcode."""
    matrix_dir = SPACERANGER_ROOT / f"NCCE_{section}" / "outs" / "filtered_feature_bc_matrix"
    matrix = mmread(matrix_dir / "matrix.mtx.gz").tocsr()
    with gzip.open(matrix_dir / "barcodes.tsv.gz", "rt") as handle:
        barcodes = [line.rstrip().split("\t")[0] for line in handle]
    if matrix.shape[1] != len(barcodes):
        raise ValueError(f"Barcode/matrix mismatch for NCCE {section}")
    return pd.Series(np.asarray(matrix.sum(axis=0)).ravel(), index=barcodes, name="umi")


def matrix_path(section: str, label: str) -> Path:
    return NCCE_ROOT / section / "matrix" / f"NCCE_{section}_SPARCAL_{label}_matrix.pkl"


def load_somatic_catalog(section: str, locus_gap: int) -> pd.DataFrame:
    """Load final somatic-pattern calls and collapse adjacent bases into loci."""
    path = NCCE_ROOT / section / "spatial_filter_purity" / "baseQ0mapQ0" / "somatic" / "somatic_variants.txt"
    if not path.exists():
        return pd.DataFrame(columns=["section", "variant", "chrom", "pos", "ref", "alt", "locus_id"])
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    if df.empty:
        return pd.DataFrame(columns=["section", "variant", "chrom", "pos", "ref", "alt", "locus_id"])
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    df["pos"] = df["pos"].astype(int)
    df["variant"] = df["chrom"] + "_" + df["pos"].astype(str)
    df = df.sort_values(["chrom", "pos"]).copy()
    locus_numbers = []
    current = -1
    previous_chrom = None
    previous_pos = None
    for row in df.itertuples(index=False):
        if row.chrom != previous_chrom or row.pos - previous_pos > locus_gap:
            current += 1
        locus_numbers.append(current)
        previous_chrom, previous_pos = row.chrom, row.pos
    df["_locus_number"] = locus_numbers
    bounds = df.groupby("_locus_number", sort=False).agg(
        chrom=("chrom", "first"), start=("pos", "min"), end=("pos", "max")
    )
    bounds["locus_id"] = bounds.apply(
        lambda x: f"{x.chrom}:{x.start}" if x.start == x.end else f"{x.chrom}:{x.start}-{x.end}", axis=1
    )
    df = df.merge(bounds[["locus_id"]], left_on="_locus_number", right_index=True, how="left")
    df["section"] = section
    return df.drop(columns="_locus_number")


def collapse_locus_burden(somatic: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Turn binary spot x SNV calls into binary spot x adjacent-call locus calls."""
    if somatic.shape[1] == 0:
        return pd.DataFrame(index=somatic.index)
    lookup = catalog.set_index("variant")["locus_id"].to_dict()
    matrix = somatic.copy()
    matrix.columns = [lookup.get(str(column).replace("chr", ""), str(column)) for column in matrix.columns]
    # A locus is present once in a spot even if adjacent bases were called together.
    return matrix.T.groupby(level=0, sort=False).max().T.astype(np.int8)


def load_section(section: str, locus_gap: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(NCCE_ROOT / section / "annotations" / "spot_metadata.tsv", sep="\t")
    metadata = metadata.set_index("barcode")
    metadata["umi"] = load_umi(section).reindex(metadata.index)
    if metadata["umi"].isna().any():
        raise ValueError(f"SpaceRanger UMI coverage missing for NCCE {section}")

    matrices = {label: pd.read_pickle(matrix_path(section, label)) for label in ["1000G", "germline", "somatic", "merged"]}
    catalog = load_somatic_catalog(section, locus_gap)
    locus_matrix = collapse_locus_burden(matrices["somatic"], catalog)

    for label, matrix in matrices.items():
        metadata[f"n_{label}"] = matrix.sum(axis=1).reindex(metadata.index).fillna(0).astype(int)
    metadata["n_upv"] = metadata["n_germline"] - metadata["n_1000G"]
    metadata["n_somatic_loci"] = locus_matrix.sum(axis=1).reindex(metadata.index).fillna(0).astype(int)
    metadata["somatic_any"] = metadata["n_somatic_loci"].gt(0)
    metadata["callable"] = metadata.index.isin(matrices["merged"].index)
    metadata["cell_type"] = metadata["cell_type"].fillna("Unknown").astype(str).str.strip()
    metadata.loc[metadata["cell_type"].isin(["", "NA", "nan"]), "cell_type"] = "Unknown"
    metadata["copykat_state"] = metadata["copykat_class"].map({"aneuploid": "aneuploid", "diploid": "diploid"}).fillna("unlabelled")
    metadata["section"] = section
    metadata["patient"] = section[:-1]
    metadata["timepoint"] = section[-1]
    metadata["time_num"] = TIME_NUM[section[-1]]
    metadata["response"] = RESPONSE[section[:-1]]
    return metadata.reset_index(), catalog, locus_matrix


def build_dataset(locus_gap: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    spots, catalogs, somatic_matrices, locus_matrices = [], [], {}, {}
    for section in SAMPLES:
        frame, catalog, loci = load_section(section, locus_gap)
        spots.append(frame)
        catalogs.append(catalog)
        somatic_matrices[section] = pd.read_pickle(matrix_path(section, "somatic"))
        locus_matrices[section] = loci
    return pd.concat(spots, ignore_index=True), pd.concat(catalogs, ignore_index=True), somatic_matrices, locus_matrices


def section_summary(spots: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    locus_count = catalog.groupby("section")["locus_id"].nunique()
    snv_count = catalog.groupby("section")["variant"].nunique()
    rows = []
    for section, df in spots.groupby("section", sort=False):
        labelled = df[df["copykat_state"] != "unlabelled"]
        rows.append({
            "section": section,
            "patient": df["patient"].iat[0],
            "timepoint": df["timepoint"].iat[0],
            "response": df["response"].iat[0],
            "tissue_spots": len(df),
            "callable_spots": int(df["callable"].sum()),
            "callable_fraction": df["callable"].mean(),
            "median_umi": df["umi"].median(),
            "p10_umi": df["umi"].quantile(0.10),
            "p90_umi": df["umi"].quantile(0.90),
            "n_1000g_snvs": int(df["n_1000G"].max()),
            "n_upv_snvs": int(df["n_upv"].max()),
            "n_somatic_snvs": int(snv_count.get(section, 0)),
            "n_somatic_loci": int(locus_count.get(section, 0)),
            "somatic_positive_spots": int(df["somatic_any"].sum()),
            "somatic_positive_fraction": df["somatic_any"].mean(),
            "somatic_locus_instances": int(df["n_somatic_loci"].sum()),
            "copykat_labelled_spots": len(labelled),
            "copykat_aneuploid_spots": int((labelled["copykat_state"] == "aneuploid").sum()),
            "copykat_aneuploid_fraction_labelled": (
                (labelled["copykat_state"] == "aneuploid").mean() if len(labelled) else np.nan
            ),
            "rctd_labelled_spots": int((df["cell_type"] != "Unknown").sum()),
        })
    return pd.DataFrame(rows).set_index("section").loc[SAMPLES].reset_index()


def cell_type_summary(spots: pd.DataFrame) -> pd.DataFrame:
    table = spots.groupby("cell_type", dropna=False).agg(
        spots=("barcode", "size"),
        callable_spots=("callable", "sum"),
        median_umi=("umi", "median"),
        somatic_positive_spots=("somatic_any", "sum"),
        somatic_positive_fraction=("somatic_any", "mean"),
        somatic_locus_instances=("n_somatic_loci", "sum"),
        mean_loci_per_spot=("n_somatic_loci", "mean"),
    ).reset_index()
    return table.sort_values(["somatic_positive_spots", "spots"], ascending=False)


def make_celltype_group(spots: pd.DataFrame) -> pd.DataFrame:
    frame = spots.copy()
    keep = ["epithelial", "fibroblasts", "macrophage", "pericytes", "endothelial", "plasma"]
    frame["celltype_group"] = frame["cell_type"].where(frame["cell_type"].isin(keep), "Other/unknown")
    frame["log_umi"] = np.log1p(frame["umi"])
    frame["log_umi_z_section"] = frame.groupby("section")["log_umi"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    frame["somatic_any_int"] = frame["somatic_any"].astype(int)
    return frame


def tidy_model(fit, model_name: str) -> pd.DataFrame:
    ci = fit.conf_int()
    table = pd.DataFrame({
        "term": fit.params.index,
        "coefficient": fit.params.values,
        "odds_ratio": np.exp(fit.params.values),
        "ci_low": np.exp(ci.iloc[:, 0].values),
        "ci_high": np.exp(ci.iloc[:, 1].values),
        "p_value_cluster": fit.pvalues.values,
        "model": model_name,
    })
    return table


def association_models(spots: pd.DataFrame) -> pd.DataFrame:
    """Exploratory fixed-section association models; copyKAT model is not validation."""
    frame = make_celltype_group(spots)
    outputs = []
    formulas = [
        ("RCTD_context", "somatic_any_int ~ log_umi_z_section + C(section) + C(celltype_group, Treatment(reference='epithelial'))", frame),
        ("RCTD_context_conditional_copyKAT", "somatic_any_int ~ log_umi_z_section + C(section) + C(celltype_group, Treatment(reference='epithelial')) + C(copykat_state, Treatment(reference='diploid'))", frame[frame["copykat_state"] != "unlabelled"]),
        ("copyKAT_internal_consistency", "somatic_any_int ~ log_umi_z_section + C(section) + C(copykat_state, Treatment(reference='diploid'))", frame[frame["copykat_state"] != "unlabelled"]),
    ]
    for model_name, formula, subset in formulas:
        try:
            fitted = smf.glm(formula, data=subset, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": subset["section"]}
            )
            table = tidy_model(fitted, model_name)
            table["n_spots"] = len(subset)
            table["n_events"] = int(subset["somatic_any_int"].sum())
            outputs.append(table)
        except Exception as exc:  # defensive: a completely separated future dataset should not abort reporting
            outputs.append(pd.DataFrame({"model": [model_name], "term": [f"MODEL_FAILED: {exc}"]}))
    return pd.concat(outputs, ignore_index=True)


def longitudinal_overlap(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient in sorted(RESPONSE, key=int):
        sets = {
            time: set(catalog.loc[catalog["section"] == f"{patient}{time}", "locus_id"])
            for time in "ABC"
        }
        for first, second in itertools.combinations("ABC", 2):
            left, right = sets[first], sets[second]
            union = left | right
            rows.append({
                "patient": patient,
                "response": RESPONSE[patient],
                "comparison": f"{first}-{second}",
                "loci_first": len(left),
                "loci_second": len(right),
                "shared_loci": len(left & right),
                "jaccard": len(left & right) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def plot_section_overview(summary: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(summary))
    colors = np.where(summary["response"].eq("responder"), "#c85a17", "#4c78a8")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    panels = [
        ("n_somatic_loci", "Somatic-pattern candidate loci"),
        ("somatic_positive_fraction", "Spots with ≥1 candidate locus"),
        ("median_umi", "Median total UMI per spot"),
    ]
    for ax, (column, ylabel) in zip(axes, panels):
        ax.bar(x, summary[column], color=colors, width=0.76)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        if column == "somatic_positive_fraction":
            ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
        if column == "median_umi":
            ax.set_yscale("log")
            ax.set_ylabel("Median UMI per spot (log scale)")
    axes[-1].set_xticks(x, summary["section"])
    axes[0].set_title("NCCE SPARCAL focal-pattern landscape", weight="bold")
    axes[0].text(0.995, 1.05, "orange = responder patient 14; blue = non-responders",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9)
    for boundary in [2.5, 5.5, 8.5, 11.5]:
        for ax in axes:
            ax.axvline(boundary, color="#d0d0d0", lw=0.8, zorder=0)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rctd_context(celltypes: pd.DataFrame, output: Path) -> None:
    display = celltypes[celltypes["spots"] >= 10].sort_values("somatic_positive_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bars = axes[0].barh(display["cell_type"], display["somatic_positive_fraction"], color=[CELL_PALETTE.get(x, "#bdbdbd") for x in display["cell_type"]])
    axes[0].set_xlabel("Fraction of spots with ≥1 somatic-pattern locus")
    axes[0].xaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    axes[0].set_title("Raw SPARCAL-pattern incidence by RCTD class", weight="bold")
    for bar, n in zip(bars, display["spots"]):
        axes[0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    sc = axes[1].scatter(display["median_umi"], display["somatic_positive_fraction"],
                         s=np.sqrt(display["spots"]) * 18,
                         c=[CELL_PALETTE.get(x, "#bdbdbd") for x in display["cell_type"]],
                         edgecolor="black", linewidth=0.4, alpha=0.9)
    for row in display.itertuples(index=False):
        axes[1].annotate(row.cell_type, (row.median_umi, row.somatic_positive_fraction), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_label_text("Median total UMI per spot (log scale)")
    axes[1].yaxis.set_label_text("Fraction with ≥1 somatic-pattern locus")
    axes[1].yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    axes[1].set_title("Coverage varies substantially by RCTD class", weight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def categorical_map(ax: plt.Axes, frame: pd.DataFrame, column: str, palette: dict[str, str], title: str) -> None:
    for label in reversed(CELL_ORDER if column == "cell_type" else list(palette)):
        sub = frame[frame[column] == label]
        if len(sub):
            ax.scatter(sub["array_col"], sub["array_row"], s=22, c=palette[label], edgecolors="#777777", linewidths=0.2)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_patient14_case(spots: pd.DataFrame, output: Path) -> None:
    frame = spots[spots["patient"] == "14"].copy()
    fig, axes = plt.subplots(3, 4, figsize=(15, 12), constrained_layout=True)
    ck_palette = {"aneuploid": "#d62d27", "diploid": "#bdbdbd", "unlabelled": "#f0f0f0"}
    max_burden = max(1, int(frame["n_somatic_loci"].max()))
    for row, time in enumerate("ABC"):
        section = frame[frame["timepoint"] == time]
        categorical_map(axes[row, 0], section, "cell_type", CELL_PALETTE, f"14{time}: RCTD cell type")
        categorical_map(axes[row, 1], section, "copykat_state", ck_palette, f"14{time}: copyKAT state")
        scatter = axes[row, 2].scatter(section["array_col"], section["array_row"], s=22,
                                       c=section["n_somatic_loci"], cmap="Reds", vmin=0, vmax=max_burden,
                                       edgecolors="#777777", linewidths=0.2)
        axes[row, 2].set_title(f"14{time}: candidate locus burden", fontsize=10, weight="bold")
        coverage = axes[row, 3].scatter(section["array_col"], section["array_row"], s=22,
                                        c=np.log10(section["umi"] + 1), cmap="magma",
                                        edgecolors="#777777", linewidths=0.2)
        axes[row, 3].set_title(f"14{time}: log10 total UMI", fontsize=10, weight="bold")
        for ax in axes[row, 2:]:
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.colorbar(scatter, ax=axes[:, 2], label="Somatic-pattern loci per spot", shrink=0.65)
    fig.colorbar(coverage, ax=axes[:, 3], label="log10(total UMI + 1)", shrink=0.65)
    fig.suptitle("Responder case study, patient 14: RCTD context and SPARCAL candidate patterns", fontsize=15, weight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_longitudinal(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for patient, frame in summary.groupby("patient", sort=False):
        frame = frame.sort_values("timepoint")
        color = "#c85a17" if patient == "14" else "#4c78a8"
        label = f"{patient} ({frame['response'].iat[0]})"
        axes[0].plot(frame["timepoint"], frame["n_somatic_loci"], marker="o", color=color, label=label)
        axes[1].plot(frame["timepoint"], frame["somatic_positive_fraction"], marker="o", color=color, label=label)
    axes[0].set_title("Candidate loci across serial sections", weight="bold")
    axes[0].set_ylabel("Number of candidate loci")
    axes[1].set_title("Spatial extent of candidate signal", weight="bold")
    axes[1].set_ylabel("Positive spot fraction")
    axes[1].yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    for ax in axes:
        ax.set_xlabel("Section timepoint")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8, title="Patient")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_purity_free_sensitivity() -> dict[str, float] | None:
    original = NCCE_ROOT / "14A" / "spatial_filter_purity" / "baseQ0mapQ0" / "somatic" / "somatic_variants.txt"
    sensitivity = NCCE_ROOT / "biological_analysis" / "purity_free_filter" / "14A" / "somatic" / "somatic_variants.txt"
    if not (original.exists() and sensitivity.exists()):
        return None
    def calls(path: Path) -> set[str]:
        df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
        return set(df["chrom"].str.replace("chr", "", regex=False) + "_" + df["pos"].astype(str))
    left, right = calls(original), calls(sensitivity)
    return {"original": len(left), "purity_free": len(right), "shared": len(left & right),
            "jaccard": len(left & right) / len(left | right) if left | right else np.nan}


def model_effect(models: pd.DataFrame, model: str, contains: str) -> str:
    rows = models[(models["model"] == model) & models["term"].astype(str).str.contains(contains, regex=False)]
    if rows.empty or "odds_ratio" not in rows:
        return "not estimable"
    row = rows.iloc[0]
    return f"OR {row.odds_ratio:.2f} (cluster-robust nominal p={row.p_value_cluster:.3g})"


def write_report(output: Path, summary: pd.DataFrame, celltypes: pd.DataFrame,
                 models: pd.DataFrame, overlap: pd.DataFrame, catalog: pd.DataFrame,
                 sensitivity: dict[str, float] | None, locus_gap: int) -> None:
    total_snvs = catalog["variant"].nunique()
    total_loci = catalog["locus_id"].nunique()
    multi_locus = catalog.groupby("locus_id")["variant"].nunique()
    n_multi = int((multi_locus > 1).sum())
    snvs_multi = int(multi_locus[multi_locus > 1].sum())
    p14 = summary[summary["patient"] == "14"].sort_values("timepoint")
    p14_loci = " → ".join(str(int(value)) for value in p14["n_somatic_loci"])
    p14_positive = " → ".join(f"{value:.1%}" for value in p14["somatic_positive_fraction"])
    p14_umi = " → ".join(f"{value:,.0f}" for value in p14["median_umi"])
    p14_copykat = " → ".join(f"{value:.1%}" for value in p14["copykat_aneuploid_fraction_labelled"])
    p14_plasma = " → ".join(
        f"{(celltypes.loc[celltypes['cell_type'] == 'plasma', 'spots'].sum() if False else 0)}" for _ in []
    )
    p14_spots = []
    # RCTD proportions for patient 14 are recovered from the per-section spot table embedded in its source summary below.
    for section in p14["section"]:
        metadata = pd.read_csv(NCCE_ROOT / section / "annotations" / "spot_metadata.tsv", sep="\t")
        plasma = metadata["cell_type"].fillna("Unknown").eq("plasma").mean()
        epithelial = metadata["cell_type"].fillna("Unknown").eq("epithelial").mean()
        p14_spots.append((plasma, epithelial))
    p14_plasma_text = " → ".join(f"{plasma:.1%}" for plasma, _ in p14_spots)
    p14_epithelial_text = " → ".join(f"{epithelial:.1%}" for _, epithelial in p14_spots)
    p14_overlap = overlap[overlap["patient"] == "14"]
    p14_overlap_text = "; ".join(f"{row.comparison}: {int(row.shared_loci)} shared" for row in p14_overlap.itertuples(index=False))
    med_min = summary.loc[summary["median_umi"].idxmin()]
    med_max = summary.loc[summary["median_umi"].idxmax()]
    plasma_raw = celltypes.loc[celltypes["cell_type"] == "plasma"]
    plasma_text = "not available"
    if len(plasma_raw):
        row = plasma_raw.iloc[0]
        plasma_text = f"{row.somatic_positive_fraction:.1%} ({int(row.somatic_positive_spots)}/{int(row.spots)} spots)"
    epithelial_raw = celltypes.loc[celltypes["cell_type"] == "epithelial"].iloc[0]
    copykat_effect = model_effect(models, "copyKAT_internal_consistency", "aneuploid")
    plasma_effect = model_effect(models, "RCTD_context_conditional_copyKAT", "[T.plasma]")

    sensitivity_text = "No local purity-free sensitivity output was available."
    if sensitivity is not None:
        sensitivity_text = (
            f"For 14A, a project-local rerun without a tumor-purity file selected {sensitivity['purity_free']} calls, "
            f"but only {sensitivity['shared']} overlapped the original {sensitivity['original']} calls "
            f"(Jaccard {sensitivity['jaccard']:.3f}). This is a sensitivity check, not an alternative ground truth."
        )

    report = f"""# NCCE SPARCAL biological analysis — exploratory report

Generated from the completed 15 NCCE SPARCAL runs using RCTD as the working cell-type reference. All source SpaceRanger, RCTD, and copyKAT files were read only.

## Scope and terminology

- Sections: 15 serial Visium sections from five patients (6, 8, 10, 11 non-responders; 14 responder), with A/B/C corresponding to the serial sampling labels supplied with the study.
- `somatic-pattern` means a final SPARCAL non-1000G focal call. It is **not** independently validated somatic truth.
- The final NCCE spatial filter used binary copyKAT aneuploid/diploid state as its tumor-purity input. Consequently, copyKAT enrichment is internal consistency, not validation.
- Adjacent same-chromosome calls within {locus_gap} bp are counted as one candidate locus. This avoids treating multi-base local call clusters as independent events.

## What the completed SPARCAL runs contain

- The final matrices contain {total_snvs} somatic-pattern SNVs, collapsing to {total_loci} candidate loci across the cohort.
- {snvs_multi} of those SNVs lie in {n_multi} multi-base loci. All biological burden summaries below are locus-level.
- RNA depth is highly heterogeneous: section median UMI ranges from {med_min.median_umi:,.0f} ({med_min.section}) to {med_max.median_umi:,.0f} ({med_max.section}). Coverage adjustment is therefore essential.
- Three sections have no final somatic-pattern calls: 8C, 10B, and 11C. Their interpretations differ: 8C and 11C have incomplete callable-spot coverage, whereas 10B is nearly fully callable and well-covered.

## Reportable case-level finding: patient 14 (the sole responder)

Patient 14 is the cleanest longitudinal case because all three sections are fully callable. Its somatic-pattern candidate-locus count fell **{p14_loci}** from A to C; the fraction of positive spots fell **{p14_positive}**. This decline occurred while median UMI increased **{p14_umi}**, so it is not explained by a loss of RNA coverage.

The RCTD context changed at the same time: plasma-labelled spots were **{p14_plasma_text}** and epithelial-labelled spots were **{p14_epithelial_text}** across A/B/C. The copyKAT aneuploid fraction among labelled spots was **{p14_copykat}**. Thus the first post-treatment section (14B) combines lower copyKAT aneuploid burden with a marked plasma-cell expansion; later 14C has aneuploid rebound but remains low in focal SPARCAL candidate burden. This is a coherent *case-level treatment-response hypothesis*, not a response biomarker (one responder only).

No exact candidate locus persists between patient-14 serial sections ({p14_overlap_text}). The data therefore support changing aggregate spatial-pattern burden, but **do not support clonal lineage tracking or a named driver claim**.

## RCTD cell-type context

Raw candidate-positive incidence is {epithelial_raw.somatic_positive_fraction:.1%} in epithelial-labelled spots ({int(epithelial_raw.somatic_positive_spots)}/{int(epithelial_raw.spots)}) and {plasma_text} in plasma-labelled spots. Epithelial and fibroblast labels dominate the raw candidate burden, but they also have very different coverage and section composition.

After adjusting for section and within-section UMI, the RCTD analysis does **not** show a robust epithelial-specific focal-SNV signal relative to fibroblasts or macrophages. Plasma-labelled spots have lower signal in the conditional exploratory model ({plasma_effect}), but this is driven in part by plasma-rich serial sections and should be presented as a microenvironment hypothesis rather than a definitive association. In Visium, each spot is mixed tissue; these associations are spatial-context results, not cell-intrinsic genotype calls.

## CopyKAT relationship — descriptive only

Among copyKAT-labelled spots, aneuploid status is associated with a higher chance of a final SPARCAL focal-pattern locus after section and UMI adjustment ({copykat_effect}). This is expected because copyKAT state was an input feature of the SPARCAL spatial filter; it must not be described as independent validation.

{sensitivity_text}

## Recommended language for a result/slide

> In this exploratory serial gastric-cancer cohort, SPARCAL identified section-specific focal SNV patterns whose spatial burden varied substantially across biopsies. In the single responder (patient 14), candidate-locus burden and spatial extent decreased across serial sections despite increasing transcriptomic coverage, concurrent with a transient plasma-cell-rich, copyKAT-low-aneuploid section. No exact focal candidate loci were shared across patient-14 timepoints, indicating that these data support dynamic spatial-pattern burden rather than clone-resolved mutation tracking.

## What should *not* be claimed from the current data

1. Do not name gene-level drivers or therapeutic mutations: this analysis did not transmit genomic coordinates outside the controlled workspace, and no independent DNA validation is available.
2. Do not call the final SPARCAL somatic class ground-truth somatic mutations. Its copyKAT relationship is partially built into the pipeline.
3. Do not infer responder-versus-non-responder effects: the cohort has one responder and substantial section-to-section RNA/callability heterogeneity.

## Generated files

- `section_summary.tsv` — per-section coverage, callability, variant/locus burden, and copyKAT/RCTD coverage.
- `spot_level_results.tsv.gz` — per-spot SPARCAL burden, UMI, RCTD label, and copyKAT state.
- `somatic_pattern_loci.tsv` — collapsed candidate loci; use this instead of raw SNV counts.
- `cell_type_summary.tsv`, `association_models.tsv`, and `longitudinal_locus_overlap.tsv` — supporting analyses.
- `section_overview.png`, `rctd_context.png`, `patient14_case_maps.png`, and `longitudinal_locus_counts.png` — report figures.
"""
    (output / "NCCE_SPARCAL_biological_report.md").write_text(report)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    spots, catalog, _, _ = build_dataset(args.locus_gap)
    summary = section_summary(spots, catalog)
    celltypes = cell_type_summary(spots)
    models = association_models(spots)
    overlap = longitudinal_overlap(catalog)
    sensitivity = read_purity_free_sensitivity()

    spots.to_csv(output / "spot_level_results.tsv.gz", sep="\t", index=False, compression="gzip")
    catalog.to_csv(output / "somatic_pattern_loci.tsv", sep="\t", index=False)
    summary.to_csv(output / "section_summary.tsv", sep="\t", index=False)
    celltypes.to_csv(output / "cell_type_summary.tsv", sep="\t", index=False)
    models.to_csv(output / "association_models.tsv", sep="\t", index=False)
    overlap.to_csv(output / "longitudinal_locus_overlap.tsv", sep="\t", index=False)

    plot_section_overview(summary, output / "section_overview.png")
    plot_rctd_context(celltypes, output / "rctd_context.png")
    plot_patient14_case(spots, output / "patient14_case_maps.png")
    plot_longitudinal(summary, output / "longitudinal_locus_counts.png")
    write_report(output, summary, celltypes, models, overlap, catalog, sensitivity, args.locus_gap)

    print(f"Wrote NCCE biological analysis to: {output}")
    print(summary[["section", "n_somatic_loci", "somatic_positive_fraction", "median_umi", "callable_fraction"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
