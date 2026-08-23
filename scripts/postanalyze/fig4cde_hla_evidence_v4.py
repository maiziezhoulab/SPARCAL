#!/usr/bin/env python
"""Figure 4c/d/e (v4, 2026-08-22) -- HLA dominance, per-patient genes, evidence.

Rewritten from `fig7_hla_wes_evidence.py` after the 2026-08-22 advisor revision.

Panel (c)  unchanged in content: HLA share of each sample's somatic-class COSMIC
    hits, all hits vs protein-altering hits.

Panel (d)  REPLACES the old "P4 n P6 shared somatic genes" bar chart.
    Two changes, both requested:
      1. P4 and P6 are DIFFERENT PATIENTS, so an intersection is not a
         meaningful unit. The intersection is no longer computed anywhere in
         this figure; each sample is shown on its own axis (a union view over
         four independent panels).
      2. The old axis listed every gene in the intersection but drew
         `n_protein_altering`, so 15 of 24 genes had a zero-length bar and
         looked like entries with no support. They were not unsupported -- each
         had >=1 catalogue hit in both patients -- but the plotted quantity was
         a different, stricter one. Here the LIST CRITERION AND THE PLOTTED
         QUANTITY ARE THE SAME: a gene appears only if it has >=1
         protein-altering COSMIC hit in that sample, so no empty rows exist.
    DCIS1 and DCIS2 get the same treatment, which is the requested DCIS panel.

Panel (e)  cSCC half unchanged: the 14 WES-confirmed SPARCAL somatic variants.
    DCIS half is NEW and deliberately on a DIFFERENT evidence axis, because
    DCIS1/DCIS2 have NO matched WES (confirmed: only P4/P6 have cSCC/Normal WES
    under STmut_Data). The DCIS panel therefore shows CATALOGUE corroboration --
    protein-altering COSMIC hits with their spatial burden and classifier score,
    Cancer Gene Census genes starred. This is weaker evidence than orthogonal
    DNA and is labelled as such on the panel; the two halves must never be read
    as the same claim.

INPUTS
  data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv
  data/somatic_evidence_2026-07-28/wes_confirmed_somatic_annotated.csv
  cosmic_amb/{dcis1,dcis2}_somatic_nochr.vcf.gz   (NS = spots per call)
  data/{dcis1,dcis2}/matrix/*_SPARCAL_somatic_matrix.pkl  (in-tissue spot count)
  Cosmic_CancerGeneCensus_v103_GRCh37.tsv.gz

OUTPUTS
  data/paper_figs_2026-08-22/fig4c_hla_share.csv
  data/paper_figs_2026-08-22/fig4d_per_sample_genes.csv
  data/paper_figs_2026-08-22/fig4e_dcis_cosmic_evidence.csv
  SPARCAL_pnas_2026/figs/v4_2026-08-22/fig4cde_hla_evidence[_preview].{png,pdf}

Run (env snv_caller, CPU, ~1 min):
  python scripts/postanalyze/fig4cde_hla_evidence_v4.py
"""
import io
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
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
COSMIC_DIR = os.path.join(PROJECT, "data", "somatic_hits_2026-07-28")
EVID_DIR = os.path.join(PROJECT, "data", "somatic_evidence_2026-07-28")
AMB_DIR = os.path.join(PROJECT, "cosmic_amb")
BCFTOOLS = os.path.join(PROJECT, "apps", "bcftools")
CGC_TSV = "/data/maiziezhou_lab/leiy4/Cosmic_CancerGeneCensus_v103_GRCh37.tsv.gz"
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-08-22")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v4_2026-08-22"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
IN_TISSUE_SPOTS = {"P4": 744, "P6": 3650}   # data/somatic_evidence_2026-07-28/SUMMARY.md
DCIS_MATRIX = {
    "DCIS1": os.path.join(PROJECT, "data/dcis1/matrix/DCIS_dcis1_SPARCAL_somatic_matrix.pkl"),
    "DCIS2": os.path.join(PROJECT, "data/dcis2/matrix/DCIS_dcis2_SPARCAL_somatic_matrix.pkl"),
}
DCIS_TOKEN = {"DCIS1": "dcis1", "DCIS2": "dcis2"}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
HLA_COLOR = "#eb6834"
NONHLA_COLOR = "#c9c8c2"
SOMATIC_RED = "#e34948"
CGC_GOLD = "#eda100"
NON_PA = ["non_coding_or_unannotated", "synonymous"]
TOP_N_GENES = 10
TOP_N_DCIS_ROWS = 8


def cgc_genes():
    cgc = pd.read_csv(CGC_TSV, sep="\t")
    return set(cgc["GENE_SYMBOL"].dropna())


def build_fig4c():
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    som = hits[hits.class_code == "somatic"].copy()
    som["is_hla"] = som.gene.astype(str).str.startswith("HLA-")
    som["protein_altering"] = ~som.consequence.isin(NON_PA)
    rows = []
    for sk in SAMPLES:
        sub = som[som["sample"] == sk]
        pa = sub[sub.protein_altering]
        rows.append(dict(sample=sk, n_all_hits=len(sub), n_hla_all_hits=int(sub.is_hla.sum()),
                         pct_hla_all_hits=100 * sub.is_hla.sum() / len(sub),
                         n_protein_altering=len(pa), n_hla_protein_altering=int(pa.is_hla.sum()),
                         pct_hla_protein_altering=100 * pa.is_hla.sum() / len(pa) if len(pa) else np.nan))
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig4c_hla_share.csv")
    df.to_csv(out, index=False)
    print(f"[fig4c] wrote {out}")
    return df


def build_fig4d(cgc):
    """Per-sample protein-altering COSMIC-hit genes. No intersection is taken."""
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    som = hits[(hits.class_code == "somatic") & hits.gene.notna()].copy()
    som["protein_altering"] = ~som.consequence.isin(NON_PA)
    rows = []
    for sk in SAMPLES:
        sub = som[(som["sample"] == sk) & som.protein_altering]
        for gene, grp in sub.groupby("gene"):
            rows.append(dict(sample=sk, gene=gene, n_protein_altering=len(grp),
                             n_all_hits=int(((som["sample"] == sk) & (som.gene == gene)).sum()),
                             is_hla=gene.startswith("HLA-"), in_cgc=gene in cgc,
                             example_aa="; ".join(sorted(set(grp.aa.dropna()))[:3])))
    df = pd.DataFrame(rows).sort_values(["sample", "n_protein_altering", "gene"],
                                        ascending=[True, False, True])
    out = os.path.join(DERIVED_DIR, "fig4d_per_sample_genes.csv")
    df.to_csv(out, index=False)
    for sk in SAMPLES:
        s = df[df["sample"] == sk]
        print(f"[fig4d] {sk}: {len(s)} genes with >=1 protein-altering hit, "
              f"{int(s.is_hla.sum())} HLA carrying {int(s[s.is_hla].n_protein_altering.sum())}"
              f"/{int(s.n_protein_altering.sum())} hits, {int(s.in_cgc.sum())} Cancer Gene Census")
    print(f"[fig4d] wrote {out}")
    return df


def build_fig4e_dcis(cgc):
    """DCIS catalogue-corroboration table: protein-altering COSMIC hits + burden."""
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    out_rows = []
    for sk in ["DCIS1", "DCIS2"]:
        proc = subprocess.run(
            [BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%INFO/NS\t%INFO/AF\t%INFO/SCORE\n",
             os.path.join(AMB_DIR, f"{DCIS_TOKEN[sk]}_somatic_nochr.vcf.gz")],
            capture_output=True, text=True, check=True)
        vcf = pd.read_csv(io.StringIO(proc.stdout), sep="\t", header=None,
                          names=["chrom", "pos", "NS", "AF", "SCORE"])
        n_spots = pickle.load(open(DCIS_MATRIX[sk], "rb")).shape[0]
        sub = hits[(hits["sample"] == sk) & (hits.class_code == "somatic")
                   & hits.gene.notna() & ~hits.consequence.isin(NON_PA)].copy()
        merged = sub.merge(vcf, on=["chrom", "pos"], how="left", validate="many_to_one")
        if merged.NS.isna().any():
            raise ValueError(f"{sk}: COSMIC hits without an NS value in the somatic VCF")
        merged["pct_spots"] = 100 * merged.NS / n_spots
        merged["in_cgc"] = merged.gene.isin(cgc)
        merged["is_hla"] = merged.gene.str.startswith("HLA-")
        merged["n_in_tissue_spots"] = n_spots
        out_rows.append(merged)
    df = pd.concat(out_rows, ignore_index=True)
    out = os.path.join(DERIVED_DIR, "fig4e_dcis_cosmic_evidence.csv")
    df.to_csv(out, index=False)
    print(f"[fig4e] wrote {out}  ({len(df)} protein-altering COSMIC hits, "
          f"{int(df.in_cgc.sum())} in the Cancer Gene Census)")
    return df


def build_fig4e_cscc():
    wes = pd.read_csv(os.path.join(EVID_DIR, "wes_confirmed_somatic_annotated.csv"))
    wes = wes[wes.class_label == "somatic"].copy()
    wes["gene_plot_label"] = wes.gene.fillna("—")
    wes.loc[wes.gene.eq("GJB2"), "gene_plot_label"] = "—"
    wes["pct_spots"] = [100 * r.n_spots_calling / IN_TISSUE_SPOTS[r["sample"]]
                        for _, r in wes.iterrows()]
    wes["variant_label"] = wes.chrom.astype(str) + ":" + wes.pos.map("{:,}".format)
    wes["has_gene"] = wes.gene.notna()
    wes["has_cosmic"] = wes.cosmic_id.notna()
    return wes


def _select_dcis_rows(df, sk):
    """All Cancer Gene Census hits, then the highest-burden remainder."""
    sub = df[df["sample"] == sk].copy()
    cgc_rows = sub[sub.in_cgc].sort_values("NS", ascending=False)
    rest = sub[~sub.in_cgc].sort_values("NS", ascending=False)
    keep = pd.concat([cgc_rows, rest.head(max(TOP_N_DCIS_ROWS - len(cgc_rows), 0))])
    return keep.sort_values("NS", ascending=True).reset_index(drop=True)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_axisbelow(True)


def main():
    cgc = cgc_genes()
    df_c = build_fig4c()
    df_d = build_fig4d(cgc)
    df_e_dcis = build_fig4e_dcis(cgc)
    df_e_cscc = build_fig4e_cscc()

    fig = plt.figure(figsize=(11.0, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.60, 1.30, 1.34], hspace=0.62, wspace=0.55,
                          left=0.105, right=0.985, top=0.945, bottom=0.055)
    ax_c = fig.add_subplot(gs[0, 0:4])
    ax_d = [fig.add_subplot(gs[1, i]) for i in range(4)]
    ax_e1 = fig.add_subplot(gs[2, 0:2])
    ax_e2 = fig.add_subplot(gs[2, 2:4])

    # ---------------- Panel c ----------------
    x = np.arange(len(SAMPLES))
    w = 0.36
    d = df_c.set_index("sample").loc[SAMPLES]
    b1 = ax_c.bar(x - w / 2, d.pct_hla_all_hits, width=w, color=HLA_COLOR, alpha=0.5,
                  edgecolor=HLA_COLOR, linewidth=0.8, label="all COSMIC hits", zorder=3)
    b2 = ax_c.bar(x + w / 2, d.pct_hla_protein_altering, width=w, color=HLA_COLOR,
                  edgecolor=HLA_COLOR, linewidth=0.8, label="protein-altering hits", zorder=3)
    for bars, col in ((b1, "n_hla_all_hits"), (b2, "n_hla_protein_altering")):
        den = "n_all_hits" if col == "n_hla_all_hits" else "n_protein_altering"
        for b, (_, r) in zip(bars, d.iterrows()):
            ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                      f"{int(r[col])}/{int(r[den])}", ha="center", va="bottom",
                      fontsize=6.2, color=INK)
    ax_c.set_xticks(x); ax_c.set_xticklabels(SAMPLES, fontsize=8.2)
    ax_c.set_ylabel("% of somatic-class\nCOSMIC hits in HLA genes", fontsize=7.6)
    ax_c.set_ylim(0, 58)
    ax_c.tick_params(axis="y", labelsize=7.0)
    ax_c.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    style(ax_c)
    ax_c.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=1.1,
                handletextpad=0.4, borderaxespad=0.1)

    # Bare panel letters only; every descriptive heading now lives in the LaTeX
    # caption (CLAUDE.md "No titles in the artwork"). The letters must stay or the
    # caption's (c)/(d)/(e) reference nothing.
    fig.text(0.012, 0.975, "c", fontsize=11, fontweight="bold", color=INK,
             va="top", ha="left")

    # ---------------- Panel d ----------------
    for i, sk in enumerate(SAMPLES):
        ax = ax_d[i]
        sub = (df_d[df_d["sample"] == sk]
               .sort_values(["n_protein_altering", "gene"], ascending=[False, True])
               .head(TOP_N_GENES).iloc[::-1].reset_index(drop=True))
        y = np.arange(len(sub))
        colors = [HLA_COLOR if h else NONHLA_COLOR for h in sub.is_hla]
        ax.barh(y, sub.n_protein_altering, color=colors, edgecolor="white", linewidth=0.4,
                height=0.72, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{g} *" if c else g for g, c in zip(sub.gene, sub.in_cgc)],
                           fontsize=6.6, fontstyle="italic")
        for lbl, r in zip(ax.get_yticklabels(), sub.itertuples()):
            lbl.set_color(HLA_COLOR if r.is_hla else (CGC_GOLD if r.in_cgc else MUTED))
            if r.is_hla or r.in_cgc:
                lbl.set_fontweight("bold")
        ax.set_xlim(0, max(sub.n_protein_altering.max() * 1.18, 2))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
        ax.tick_params(axis="x", labelsize=6.6)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_axisbelow(True)
        allg = df_d[df_d["sample"] == sk]
        n_hla = int(allg.is_hla.sum())
        hla_pa = int(allg[allg.is_hla].n_protein_altering.sum())
        tot_pa = int(allg.n_protein_altering.sum())
        ax.set_title(sk, fontsize=9.0, loc="center", pad=4, color=INK)  # section label
        ax.text(0.97, 0.03,
                f"{len(allg)} genes · {int(allg.in_cgc.sum())} CGC\n"
                f"HLA {hla_pa}/{tot_pa} hits ({100*hla_pa/tot_pa:.0f}%)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
                color=MUTED, linespacing=1.4)
        if i == 0:
            ax.set_xlabel("protein-altering COSMIC hits", fontsize=7.2)
    fig.text(0.012, 0.700, "d", fontsize=11, fontweight="bold", color=INK,
             va="top", ha="left")

    # ---------------- Panel e, cSCC half ----------------
    order = df_e_cscc.sort_values(["sample", "pct_spots"], ascending=[True, False]).reset_index(drop=True)
    yc = np.arange(len(order))[::-1]
    for yy, r in zip(yc, order.itertuples()):
        ax_e1.scatter([r.pct_spots], [yy], s=max(r.wes_tumor_vaf * 900, 18),
                      facecolor=SOMATIC_RED, edgecolor=SOMATIC_RED, linewidth=1.3, zorder=3)
        if r.possible_rna_editing_A2G_T2C:
            ax_e1.scatter([r.pct_spots], [yy], marker="^", s=24, facecolor=INK,
                          edgecolor="white", linewidth=0.45, zorder=4)
        if r.has_cosmic:
            ax_e1.scatter([r.pct_spots], [yy], marker="*", s=90, facecolor=CGC_GOLD,
                          edgecolor="black", linewidth=0.4, zorder=5)
    ax_e1.set_yticks(yc)
    ax_e1.set_yticklabels([f"{r.gene_plot_label}   {r.variant_label}" for r in order.itertuples()],
                          fontsize=6.4)
    for lbl, r in zip(ax_e1.get_yticklabels(), order.itertuples()):
        lbl.set_color(INK if r.has_gene else MUTED)
    ax_e1.set_xlabel("% of in-tissue spots calling the variant", fontsize=7.4)
    ax_e1.set_xlim(-0.3, 6.4)
    ax_e1.tick_params(axis="x", labelsize=6.8)
    ax_e1.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    style(ax_e1)
    n_p4 = int((order["sample"] == "P4").sum())
    ax_e1.axhline(yc[n_p4 - 1] - 0.5, color=GRID, linewidth=0.8, zorder=1)
    ax_e1.text(6.3, yc[:n_p4].mean(), "P4", fontsize=7.4, color=MUTED, ha="right",
               va="center", fontweight="bold")
    ax_e1.text(6.3, yc[n_p4:].mean(), "P6", fontsize=7.4, color=MUTED, ha="right",
               va="center", fontweight="bold")
    ax_e1.set_title("cSCC · WES-confirmed", fontsize=9.0, loc="center", pad=4, color=INK)
    fig.text(0.012, 0.345, "e", fontsize=11, fontweight="bold", color=INK,
             va="top", ha="left")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.5, markerfacecolor=SOMATIC_RED,
               markeredgecolor=SOMATIC_RED, label="dot area: WES tumour VAF"),
        Line2D([], [], marker="^", linestyle="none", markersize=5.5, markerfacecolor=INK,
               markeredgecolor="white", label="possible RNA editing (A>G/T>C), 2/14"),
        Line2D([], [], marker="*", linestyle="none", markersize=9, markerfacecolor=CGC_GOLD,
               markeredgecolor="black", label="COSMIC hit, 1/14"),
    ]
    ax_e1.legend(handles=handles, loc="lower right", fontsize=6.0, frameon=False,
                 handletextpad=0.5, labelspacing=0.5, borderaxespad=0.3)

    # ---------------- Panel e, DCIS half ----------------
    rows, ylabels, colors_lbl, boundary = [], [], [], None
    for sk in ["DCIS1", "DCIS2"]:
        sel = _select_dcis_rows(df_e_dcis, sk)
        if boundary is None and sk == "DCIS1":
            boundary = len(sel)
        rows.append(sel)
    sel_all = pd.concat(rows, ignore_index=True)
    yd = np.arange(len(sel_all))[::-1]
    smax = sel_all.SCORE.max()
    for yy, r in zip(yd, sel_all.itertuples()):
        color = HLA_COLOR if r.is_hla else SOMATIC_RED
        ax_e2.scatter([r.pct_spots], [yy], s=max(28 + 120 * (r.SCORE / smax), 18),
                      facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.9, zorder=3)
        if r.in_cgc:
            ax_e2.scatter([r.pct_spots], [yy], marker="*", s=90, facecolor=CGC_GOLD,
                          edgecolor="black", linewidth=0.4, zorder=5)
        ylabels.append(f"{r.gene} {r.aa}")
        colors_lbl.append(HLA_COLOR if r.is_hla else (CGC_GOLD if r.in_cgc else MUTED))
    ax_e2.set_yticks(yd)
    ax_e2.set_yticklabels(ylabels, fontsize=6.4, fontstyle="italic")
    for lbl, c in zip(ax_e2.get_yticklabels(), colors_lbl):
        lbl.set_color(c)
        if c != MUTED:
            lbl.set_fontweight("bold")
    ax_e2.set_xlabel("% of in-tissue spots calling the variant", fontsize=7.4)
    ax_e2.set_xlim(0, sel_all.pct_spots.max() * 1.55)
    ax_e2.tick_params(axis="x", labelsize=6.8)
    ax_e2.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    style(ax_e2)
    ax_e2.axhline(yd[boundary - 1] - 0.5, color=GRID, linewidth=0.8, zorder=1)
    xr = ax_e2.get_xlim()[1]
    ax_e2.text(xr * 0.985, yd[:boundary].mean(), "DCIS1", fontsize=7.4, color=MUTED,
               ha="right", va="center", fontweight="bold")
    ax_e2.text(xr * 0.985, yd[boundary:].mean(), "DCIS2", fontsize=7.4, color=MUTED,
               ha="right", va="center", fontweight="bold")
    ax_e2.set_title("DCIS · COSMIC-hit", fontsize=9.0, loc="center", pad=4, color=INK)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.5, markerfacecolor=SOMATIC_RED,
               markeredgecolor=SOMATIC_RED, label="dot area: SPARCAL classifier score"),
        Line2D([], [], marker="o", linestyle="none", markersize=6.5, markerfacecolor=HLA_COLOR,
               markeredgecolor=HLA_COLOR, label="HLA gene"),
        Line2D([], [], marker="*", linestyle="none", markersize=9, markerfacecolor=CGC_GOLD,
               markeredgecolor="black", label="Cancer Gene Census gene"),
    ]
    ax_e2.legend(handles=handles, loc="lower left", fontsize=6.0, frameon=False,
                 handletextpad=0.5, labelspacing=0.5, borderaxespad=0.3)

    stem = "fig4cde_hla_evidence" if HAS_ARIAL else "fig4cde_hla_evidence_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans preview only.")
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"[fig4cde] wrote {path}")


if __name__ == "__main__":
    main()
