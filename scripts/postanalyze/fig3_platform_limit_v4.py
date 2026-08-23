#!/usr/bin/env python
"""Figure 3 (v4, 2026-08-22) -- spatial-RNA evidence limits across three callers.

Rewritten from `fig2_platform_limit.py` after the 2026-08-22 advisor revision:

  (a) per-call ALTERNATE-READ EVIDENCE (in how many spots the alternate allele
      is actually observed) for SPARCAL somatic, Monopogen somatic and
      SpatialSNV, with each caller's callset size annotated on its bar, split
      into a cSCC block (P4/P6) and a DCIS block (DCIS1/DCIS2). The old panel a
      characterised only SpatialSNV.
  (b) allele-exact spatial-RNA observation of matched-WES Mutect2 SNP alleles,
      IN and OUT of the Beagle/1KGP panel, drawn as TWO axes on ABSOLUTE counts
      (the two strata differ ~11x in size, so one shared percentage axis hid the
      in-panel stratum) and with an explicit legend for every colour.
  (c) position-level overlap Venn per section: 4 sets for P4/P6 (SPARCAL
      somatic, SpatialSNV, Monopogen somatic, matched-WES Mutect2) and 3 sets
      for DCIS1/DCIS2, which have NO matched WES.
  (d) removed at the advisor's request.

SET DEFINITIONS -- one definition per callset, used by BOTH panel a and panel c
so the figure is internally consistent:
  SPARCAL somatic  : current post-dedup somatic class, `data/<s>/.../matrix/
                     *_SPARCAL_somatic_matrix.pkl` (spots x variants, presence).
  Monopogen somatic: `Monopogen_<S>/out/classified/<S>.somatic.csv`; spot support
                     is Monopogen's own cellScan output (cell_alt + cell_ref_alt),
                     never a re-derived presence rule of ours.
  SpatialSNV       : released CallBack matrix filtered to PASS n SNV,
                     `SpatialSNV/results/<s>/matrix/*_presence_filtered_matrix.pkl`.
                     Barcodes are left exactly as released (they include
                     non-tissue spots), which is the generous choice for
                     SpatialSNV in the recurrence comparison.
  WES              : matched tumour/normal Mutect2 exome SNP alleles (P4/P6 only).

WHY PANEL a DOES NOT USE THE SPARCAL MATRIX  -- read before changing it.
  The SPARCAL spot x variant matrix records the pipeline's per-spot CALLS after
  the spatial filter, not per-spot alternate-read evidence. Measured against the
  deduplicated BAM on the same sites, a P4 somatic call whose alternate allele is
  seen in a median of 1 spot is "present" in a median of 8 matrix spots; the
  matrix therefore reports 84-100% of somatic calls in >=2 spots where the read
  evidence supports 8-23%. The spatial filter requires neighbour support, so
  multi-spot presence is enforced by construction and comparing that number
  against SpatialSNV's or Monopogen's pileup-derived support would be circular
  and would flatter SPARCAL. Panel a therefore uses, for all three callers, the
  number of spots in which the ALTERNATE ALLELE IS OBSERVED:
    SPARCAL   `data/<s>/artifact_evidence/<pilot>/features/site_features.tsv.gz`,
              column `n_spots_alt`, from a direct CB-tagged pileup of the
              deduplicated BAM. This is a deterministic 500-site sample per
              class, so SPARCAL's bars carry a binomial 95% CI and the panel says
              so; the other two callers are complete callsets.
    Monopogen `cell_alt + cell_ref_alt` from cellScan.
    SpatialSNV nonzero entries of the released CallBack ALT matrix.
  The matrix-derived presence counts are still written to the derived CSV as
  `n_ge2_spots_matrix_presence` for provenance, and are NOT plotted.

Panel c is position-level (`chrom_pos`, no `chr` prefix) because the SPARCAL
bundle matrix carries no alleles.

Run (env snv_caller, CPU, ~1 min):
  python scripts/postanalyze/fig3_platform_limit_v4.py
Final Arial export:
  SPARCAL_ARIAL_FONT=/path/to/Arial.ttf python scripts/postanalyze/fig3_platform_limit_v4.py

Outputs
  data/paper_figs_2026-08-22/fig3a_callset_support.csv
  data/paper_figs_2026-08-22/fig3b_beagle_rna_coverage.csv
  data/paper_figs_2026-08-22/fig3c_venn_compartments.csv
  SPARCAL_pnas_2026/figs/v4_2026-08-22/fig3_platform_limit[_preview].{png,pdf}
"""
import itertools
import os
import pickle

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
    """Use real Arial when available; otherwise produce a clearly named preview."""
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local_dir = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/fonts/arial"
    local = font_manager.findSystemFonts(fontpaths=[local_dir]) if os.path.isdir(local_dir) else []
    explicit_family = (font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
                       if explicit and os.path.dirname(explicit) else [])
    candidates = ([explicit] + explicit_family) if explicit else local + font_manager.findSystemFonts()
    matches = []
    for path in candidates:
        if not path or path in matches or not os.path.exists(path):
            continue
        try:
            family = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if family.casefold() == "arial":
            matches.append(path)
    if matches:
        for path in matches:
            font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Arial"
        return True, matches[0]
    plt.rcParams["font.family"] = "Nimbus Sans"
    return False, None


HAS_ARIAL, ARIAL_PATH = configure_required_font()

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
EVID_DIR = f"{PROJECT}/data/somatic_evidence_2026-07-28"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-08-22"
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v4_2026-08-22"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
CSCC, DCIS = ["P4", "P6"], ["DCIS1", "DCIS2"]

SPARCAL_MAT = {
    "P4": f"{PROJECT}/data/P4_tumor/1/matrix/P4_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    "P6": f"{PROJECT}/data/P6_tumor/1/matrix/P6_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    "DCIS1": f"{PROJECT}/data/dcis1/matrix/DCIS_dcis1_SPARCAL_somatic_matrix.pkl",
    "DCIS2": f"{PROJECT}/data/dcis2/matrix/DCIS_dcis2_SPARCAL_somatic_matrix.pkl",
}
SPARCAL_EVIDENCE = {
    "P4": f"{PROJECT}/data/P4_tumor/1/artifact_evidence/v2_pilot_2026-07-15_p4_batch",
    "P6": f"{PROJECT}/data/P6_tumor/1/artifact_evidence/v2_pilot_2026-07-16_p6",
    "DCIS1": f"{PROJECT}/data/dcis1/artifact_evidence/v2_pilot_2026-07-16_dcis1",
    "DCIS2": f"{PROJECT}/data/dcis2/artifact_evidence/v2_pilot_2026-07-16_dcis2",
}
MONO_CSV = {
    "P4": f"{PROJECT}/Monopogen_P4_rep1/out/classified/P4_rep1.somatic.csv",
    "P6": f"{PROJECT}/Monopogen_P6_rep1/out/classified/P6_rep1.somatic.csv",
    "DCIS1": f"{PROJECT}/Monopogen_DCIS1/out/classified/DCIS1.somatic.csv",
    "DCIS2": f"{PROJECT}/Monopogen_DCIS2/out/classified/DCIS2.somatic.csv",
}
SSNV_MAT = {s: f"{PROJECT}/SpatialSNV/results/{t}/matrix/{t}_spatialsnv_presence_filtered_matrix.pkl"
            for s, t in [("P4", "p4"), ("P6", "p6"), ("DCIS1", "dcis1"), ("DCIS2", "dcis2")]}

METHODS = ["SPARCAL somatic", "Monopogen somatic", "SpatialSNV"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"
IN_C, IN_L = "#2a78d6", "#c9dcf4"
OUT_C, OUT_L = "#eb6834", "#f8d6c5"
METHOD_C = {"SPARCAL somatic": (SPARCAL_C, SPARCAL_L),
            "Monopogen somatic": (MONO_C, MONO_L),
            "SpatialSNV": (SSNV_C, SSNV_L)}


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _strip(chrom):
    c = str(chrom)
    return c[3:] if c.startswith("chr") else c


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_callsets():
    """Position sets (chrom_pos, no chr prefix) and per-call spot recurrence."""
    positions, support = {}, []
    for s in SAMPLES:
        mat = pickle.load(open(SPARCAL_MAT[s], "rb"))
        ns = (mat.values > 0).sum(axis=0)
        positions[(s, "SPARCAL somatic")] = {
            f"{_strip(k.split('_')[0])}_{k.split('_')[1]}" for k in mat.columns}
        # Alternate-read evidence from the deduplicated BAM (sampled; see header).
        sf = pd.read_csv(f"{SPARCAL_EVIDENCE[s]}/features/site_features.tsv.gz", sep="\t")
        sf = sf[sf.candidate_sources.astype(str) == "somatic"]
        support.append(dict(sample=s, method="SPARCAL somatic", n_calls=int(mat.shape[1]),
                            n_spots_in_matrix=int(mat.shape[0]),
                            n_evaluated=int(len(sf)), sampled=True,
                            n_ge2_spots_alt=int((sf.n_spots_alt >= 2).sum()),
                            median_spots_alt=float(sf.n_spots_alt.median()),
                            n_ge2_spots_matrix_presence=int((ns >= 2).sum())))

        mono = pd.read_csv(MONO_CSV[s])
        ns = (mono["cell_alt"].fillna(0) + mono["cell_ref_alt"].fillna(0)).astype(int).values
        positions[(s, "Monopogen somatic")] = {
            f"{_strip(c)}_{int(p)}" for c, p in zip(mono["chr"], mono["pos"])}
        support.append(dict(sample=s, method="Monopogen somatic", n_calls=int(len(mono)),
                            n_spots_in_matrix=np.nan,
                            n_evaluated=int(len(mono)), sampled=False,
                            n_ge2_spots_alt=int((ns >= 2).sum()),
                            median_spots_alt=float(np.median(ns)),
                            n_ge2_spots_matrix_presence=np.nan))

        mat = pickle.load(open(SSNV_MAT[s], "rb"))
        ns = (mat.values > 0).sum(axis=0)
        positions[(s, "SpatialSNV")] = {
            f"{_strip(k.split('_')[0])}_{k.split('_')[1]}" for k in mat.columns}
        support.append(dict(sample=s, method="SpatialSNV", n_calls=int(mat.shape[1]),
                            n_spots_in_matrix=int(mat.shape[0]),
                            n_evaluated=int(mat.shape[1]), sampled=False,
                            n_ge2_spots_alt=int((ns >= 2).sum()),
                            median_spots_alt=float(np.median(ns)),
                            n_ge2_spots_matrix_presence=np.nan))

    wes = pd.read_csv(f"{EVID_DIR}/wes_leakage_af_stratified.csv")
    for s in CSCC:
        sub = wes[wes["sample"] == s]
        positions[(s, "WES")] = {f"{_strip(c)}_{int(p)}" for c, p in zip(sub.chrom, sub.pos)}

    df = pd.DataFrame(support)
    df["pct_ge2_spots_alt"] = 100 * df.n_ge2_spots_alt / df.n_evaluated
    # Binomial 95% CI -- meaningful only for the sampled SPARCAL rows.
    p_hat = df.pct_ge2_spots_alt / 100
    df["ci95"] = np.where(df.sampled,
                          100 * 1.96 * np.sqrt(p_hat * (1 - p_hat) / df.n_evaluated), 0.0)
    df.to_csv(f"{DERIVED_DIR}/fig3a_callset_support.csv", index=False)
    print(f"[fig3a] wrote {DERIVED_DIR}/fig3a_callset_support.csv")
    return positions, df


def beagle_rna_coverage():
    """Allele-exact WES/RNA coverage stratified by 1KGP-panel membership."""
    accounting = pd.read_csv(f"{EVID_DIR}/wes_confirmed_full_accounting.csv")
    leakage = pd.read_csv(f"{EVID_DIR}/wes_leakage_af_stratified.csv")
    keys = ["sample", "chrom", "pos", "ref", "alt"]
    merged = leakage.merge(accounting[keys + ["rna_covered"]], on=keys, how="left",
                           validate="one_to_one")
    if merged["rna_covered"].isna().any():
        raise ValueError("WES leakage rows did not all match the RNA-coverage accounting table")
    merged["beagle_group"] = np.where(merged["category"].eq("leaked_exact"),
                                      "In Beagle/1KGP panel", "Out of Beagle/1KGP panel")
    out = (merged.groupby(["sample", "beagle_group"], as_index=False)
           .agg(n_wes=("pos", "size"), n_rna_covered=("rna_covered", "sum")))
    out["n_not_covered"] = out.n_wes - out.n_rna_covered
    out["pct_rna_covered"] = 100 * out.n_rna_covered / out.n_wes
    out.to_csv(f"{DERIVED_DIR}/fig3b_beagle_rna_coverage.csv", index=False)
    print(f"[fig3b] wrote {DERIVED_DIR}/fig3b_beagle_rna_coverage.csv")
    return out


def venn_compartments(positions):
    """Exclusive-compartment counts for every section's Venn."""
    rows = []
    for s in SAMPLES:
        names = METHODS + (["WES"] if (s, "WES") in positions else [])
        sets = [positions[(s, n)] for n in names]
        k = len(names)
        for bits in itertools.product([0, 1], repeat=k):
            if not any(bits):
                continue
            inside = [sets[i] for i, b in enumerate(bits) if b]
            outside = [sets[i] for i, b in enumerate(bits) if not b]
            comp = set.intersection(*inside)
            for o in outside:
                comp = comp - o
            rows.append(dict(sample=s, n_sets=k, code="".join(map(str, bits)),
                             members="+".join(n for n, b in zip(names, bits) if b),
                             n=len(comp)))
        for n, st in zip(names, sets):
            rows.append(dict(sample=s, n_sets=k, code="TOTAL", members=n, n=len(st)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{DERIVED_DIR}/fig3c_venn_compartments.csv", index=False)
    print(f"[fig3c] wrote {DERIVED_DIR}/fig3c_venn_compartments.csv")
    return df


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _fmt(n):
    return f"{int(n):,}"


def panel_a(ax, support, samples, block_title, show_ylabel, show_legend):
    x = np.arange(len(samples))
    width = 0.26
    for i, method in enumerate(METHODS):
        sub = support[support.method.eq(method)].set_index("sample").loc[samples]
        xp = x + (i - 1) * width
        dark, _ = METHOD_C[method]
        ax.bar(xp, sub.pct_ge2_spots_alt, width=width * 0.88, color=dark, edgecolor=INK,
               linewidth=0.6, zorder=3,
               yerr=sub.ci95.values, error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2.4))
        for xi, row in zip(xp, sub.itertuples()):
            top = row.pct_ge2_spots_alt + row.ci95
            ax.text(xi, top + 1.4, f"{row.pct_ge2_spots_alt:.0f}%", ha="center",
                    va="bottom", fontsize=7.4, color=dark, fontweight="bold")
            ax.text(xi, top + 7.2, f"n = {_fmt(row.n_calls)}", ha="center", va="bottom",
                    fontsize=6.8, color=MUTED, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, fontsize=9.5)
    ax.set_xlim(-0.62, len(samples) - 0.38)
    ax.set_ylim(0, 88)
    if show_ylabel:
        ax.set_ylabel("% of calls whose alternate allele\nis observed in ≥2 spots",
                      fontsize=9.0, color=INK, linespacing=1.4)
    ax.set_title(block_title, fontsize=9.5, color=MUTED, loc="center", pad=3)  # data-subset label, not prose
    if show_legend:
        handles = [Patch(facecolor=METHOD_C[m][0], edgecolor=INK, label=m) for m in METHODS]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(-0.08, -0.115), ncol=3,
                  frameon=False, fontsize=8.2, handlelength=1.3, columnspacing=1.6)
        ax.text(-0.08, -0.215, "bar label n = calls reported · SPARCAL is a 500-site "
                               "sample per section (95% CI shown); the other two are "
                               "complete callsets",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color=MUTED)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def panel_b(ax, coverage, group, show_ylabel):
    dark, light = (IN_C, IN_L) if group.startswith("In") else (OUT_C, OUT_L)
    sub = coverage[coverage.beagle_group.eq(group)].set_index("sample").loc[CSCC]
    x = np.arange(len(CSCC))
    ax.bar(x, sub.n_wes, width=0.46, color=light, edgecolor=INK, linewidth=0.8, zorder=2)
    ax.bar(x, sub.n_rna_covered, width=0.46, color=dark, edgecolor=INK, linewidth=0.8, zorder=3)
    for xi, row in zip(x, sub.itertuples()):
        ax.text(xi, row.n_wes * 1.03, _fmt(row.n_wes), ha="center", va="bottom",
                fontsize=8.2, color=INK)
        ax.text(xi, row.n_wes * 0.135,
                f"{int(row.n_rna_covered)} observed\n({row.pct_rna_covered:.2f}%)",
                ha="center", va="bottom", fontsize=8.0, color=dark, fontweight="bold",
                linespacing=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CSCC, fontsize=9.5)
    ax.set_xlim(-0.62, len(CSCC) - 0.38)
    ax.set_ylim(0, sub.n_wes.max() * 1.22)
    if show_ylabel:
        ax.set_ylabel("WES Mutect2 SNP alleles", fontsize=9.5, color=INK)
    ax.set_title(group, fontsize=9.5, color=MUTED, loc="center", pad=3)  # stratum label, not prose
    ax.legend(handles=[Patch(facecolor=light, edgecolor=INK, label="not observed in spatial RNA"),
                       Patch(facecolor=dark, edgecolor=INK, label="observed, allele-exact")],
              loc="upper left", frameon=False, fontsize=7.6, handlelength=1.2,
              borderaxespad=0.15, labelspacing=0.3)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


VENN4_NAMES = ["SPARCAL somatic", "SpatialSNV", "Monopogen somatic", "WES"]
VENN4_SHORT = ["SPARCAL", "SpatialSNV", "Monopogen", "WES"]
VENN4_COLORS = [SPARCAL_C, SSNV_C, MONO_C, WES_C]
VENN4_ELLIPSES = [(0.350, 0.400, 0.72, 0.45, 140.0),
                  (0.450, 0.500, 0.72, 0.45, 140.0),
                  (0.544, 0.500, 0.72, 0.45, 40.0),
                  (0.644, 0.400, 0.72, 0.45, 40.0)]
VENN4_LABEL_XY = {
    "0001": (0.85, 0.42), "0010": (0.68, 0.72), "0011": (0.77, 0.59), "0100": (0.32, 0.72),
    "0101": (0.71, 0.30), "0110": (0.50, 0.66), "0111": (0.65, 0.50), "1000": (0.14, 0.42),
    "1001": (0.50, 0.17), "1010": (0.29, 0.30), "1011": (0.39, 0.24), "1100": (0.23, 0.59),
    "1101": (0.61, 0.24), "1110": (0.35, 0.50), "1111": (0.50, 0.38),
}
VENN4_NAME_XY = [(0.02, 0.10, "left", "bottom"), (0.02, 0.98, "left", "top"),
                 (0.98, 0.98, "right", "top"), (0.98, 0.10, "right", "bottom")]

VENN3_NAMES = ["SPARCAL somatic", "SpatialSNV", "Monopogen somatic"]
VENN3_COLORS = [SPARCAL_C, SSNV_C, MONO_C]


def panel_c_venn4(ax, comp, sample):
    lookup = {r.code: r.n for r in comp[comp["sample"].eq(sample)].itertuples()}
    totals = {r.members: r.n for r in
              comp[comp["sample"].eq(sample) & comp.code.eq("TOTAL")].itertuples()}
    for (cx, cy, w, h, ang), color in zip(VENN4_ELLIPSES, VENN4_COLORS):
        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, facecolor=color, edgecolor=INK,
                             linewidth=0.9, alpha=0.24))
    for code, (xp, yp) in VENN4_LABEL_XY.items():
        n = lookup.get(code, 0)
        deep = code.count("1") >= 3
        ax.text(xp, yp, _fmt(n), ha="center", va="center",
                fontsize=7.4 if deep else 8.0,
                fontweight="bold" if deep else "normal", color=INK)
    for name, short, color, (xp, yp, ha, va) in zip(VENN4_NAMES, VENN4_SHORT,
                                                     VENN4_COLORS, VENN4_NAME_XY):
        ax.text(xp, yp, f"{short}\n{_fmt(totals[name])}", ha=ha, va=va, fontsize=8.0,
                color=color, fontweight="bold", linespacing=1.2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.02, 1.06); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(sample, fontsize=9.5, color=MUTED, loc="center", pad=0)


def panel_c_venn3(ax, comp, sample):
    lookup = {r.code: r.n for r in comp[comp["sample"].eq(sample)].itertuples()}
    totals = {r.members: r.n for r in
              comp[comp["sample"].eq(sample) & comp.code.eq("TOTAL")].itertuples()}
    circles = [((0.38, 0.60), 0.27), ((0.62, 0.60), 0.27), ((0.50, 0.39), 0.27)]
    for (center, radius), color in zip(circles, VENN3_COLORS):
        ax.add_patch(Circle(center, radius, facecolor=color, edgecolor=INK,
                            linewidth=0.9, alpha=0.24))
    pos = {"100": (0.24, 0.68), "010": (0.76, 0.68), "001": (0.50, 0.23),
           "110": (0.50, 0.71), "101": (0.36, 0.44), "011": (0.64, 0.44), "111": (0.50, 0.535)}
    for code, (xp, yp) in pos.items():
        ax.text(xp, yp, _fmt(lookup.get(code, 0)), ha="center", va="center",
                fontsize=7.4 if code == "111" else 8.0,
                fontweight="bold" if code == "111" else "normal", color=INK)
    ax.text(0.02, 0.98, f"SPARCAL\n{_fmt(totals['SPARCAL somatic'])}", ha="left",
            va="top", fontsize=8.0, color=SPARCAL_C, fontweight="bold", linespacing=1.2)
    ax.text(0.98, 0.98, f"SpatialSNV\n{_fmt(totals['SpatialSNV'])}", ha="right", va="top",
            fontsize=8.0, color=SSNV_C, fontweight="bold", linespacing=1.2)
    ax.text(0.02, 0.10, f"Monopogen\n{_fmt(totals['Monopogen somatic'])}",
            ha="left", va="bottom", fontsize=8.0, color=MONO_C, fontweight="bold",
            linespacing=1.2)
    ax.text(0.98, 0.10, "no matched\nWES", ha="right", va="bottom", fontsize=8.0, color=MUTED,
            style="italic", linespacing=1.2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.02, 1.06); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(sample, fontsize=9.5, color=MUTED, loc="center", pad=0)


def main():
    positions, support = build_callsets()
    coverage = beagle_rna_coverage()
    comp = venn_compartments(positions)

    fig = plt.figure(figsize=(11.6, 7.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.30, 1.0], hspace=0.40, wspace=0.34,
                          left=0.055, right=0.985, top=0.905, bottom=0.035)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b1 = fig.add_subplot(gs[0, 2])
    ax_b2 = fig.add_subplot(gs[0, 3])
    ax_c = [fig.add_subplot(gs[1, i]) for i in range(4)]

    panel_a(ax_a1, support, CSCC, "cSCC", True, False)
    panel_a(ax_a2, support, DCIS, "DCIS", False, True)
    panel_b(ax_b1, coverage, "In Beagle/1KGP panel", True)
    panel_b(ax_b2, coverage, "Out of Beagle/1KGP panel", False)
    panel_c_venn4(ax_c[0], comp, "P4")
    panel_c_venn4(ax_c[1], comp, "P6")
    panel_c_venn3(ax_c[2], comp, "DCIS1")
    panel_c_venn3(ax_c[3], comp, "DCIS2")

    # Bare panel letters only. Descriptive headings and the figure-level title live
    # in the LaTeX caption (CLAUDE.md "Figures -- the standard", No titles in the
    # artwork). The letters must stay or the caption's (a)/(b)/(c) reference nothing.
    ax_a1.text(-0.21, 1.13, "a", transform=ax_a1.transAxes, fontsize=13,
               fontweight="bold", color=INK, va="top", ha="left")
    ax_b1.text(-0.21, 1.13, "b", transform=ax_b1.transAxes, fontsize=13,
               fontweight="bold", color=INK, va="top", ha="left")
    ax_c[0].text(-0.03, 1.13, "c", transform=ax_c[0].transAxes, fontsize=13,
                 fontweight="bold", color=INK, va="top", ha="left")

    stem = "fig3_platform_limit" if HAS_ARIAL else "fig3_platform_limit_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans preview only.")
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"[fig3] wrote {path}")


if __name__ == "__main__":
    main()
