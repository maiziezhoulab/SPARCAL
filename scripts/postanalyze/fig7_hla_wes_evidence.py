#!/usr/bin/env python
"""Fig 7 -- HLA/MHC recurrence caution + DNA-confirmed somatic calls.
(PAPER_PLAN.md Story C5, Open Decision D7 option (b): reframe Fig 7 around the
HLA/MHC recurrence caution, the paper's most generalizable somatic-side finding.)

Panel (a): HLA share of somatic-class COSMIC hits per sample, all hits vs
           protein-altering hits.
Panel (b): P4 (inverted) P6 shared somatic genes, sorted by protein-altering
           hit count, HLA genes highlighted -- makes the HLA dominance (4 of 24
           shared genes carry 24 of 30 shared protein-altering hits) the visual
           point, with 0 Cancer Gene Census genes shared.
Panel (c): the 14 WES-confirmed somatic variants (7 P4, 7 P6) as an annotated
           dot plot -- gene where known, % of in-tissue spots calling it, WES
           tumor VAF (dot size). The plotted calls are identified explicitly as
           SPARCAL somatic results. Possible RNA-editing candidates retain their
           filled somatic circle and add a triangle; the single COSMIC hit is
           marked with a star. The GJB2 gene text is omitted from the left-side
           row label per the 2026-08 advisor revision, while the coordinate and
           COSMIC marker remain.

CRITICAL FRAMING (PAPER_PLAN.md Sec 3 Story C5, Sec 5 Decision D7, Sec 6):
  - Panel (c) is NOT driver discovery: 0 of 14 WES-confirmed variants are
    Cancer Gene Census hits (1/14 is in COSMIC at all -- GJB2 p.F141=, a
    synonymous change in a non-Census gene).
  - Panel (b): 0 of 24 P4 boolean P6 shared somatic genes are Cancer Gene
    Census genes; what recurs is the HLA region, not a driver signature. The
    manuscript's prior "KRT6B/SPINK5 shared cSCC signature" claim (built on an
    8-variant stale P4 callset) does NOT reproduce on the current sets and
    must not be repeated.
  - This independently corroborates the xMHC-exclusion finding in Fig 5b from
    a completely different direction (gene identity here, vs positional
    exclusion there).

VERIFY-BEFORE-PLOT DISAGREEMENT FOUND (reported prominently, not silently
trusted) -- see SUMMARY.md / README.md in the figure output dir:
  The task brief stated "11 of 14 [WES-confirmed somatic variants] have no gene
  annotation." (PAPER_PLAN.md Story C5 item 16 was corrected to 13 on 2026-07-29;
  Decision D7 kept quoting 11 until it was corrected on 2026-08-20.) Recomputed directly
  from data/somatic_evidence_2026-07-28/wes_confirmed_somatic_annotated.csv
  (the dedicated deliverable script's own output, and its SUMMARY.md, which
  independently reports "13 unannotated"): only 1 of 14 (GJB2) carries ANY gene
  annotation -- the other 13 have an empty `gene` field. The correct number is
  13/14 unannotated, not 11/14. This script uses the CSV-derived 13/14 and
  flags the discrepancy in its printed output and in the derived CSV's comment.

INPUT (committed CSVs):
  data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv
  data/somatic_hits_2026-07-28/cscc_shared_p4_p6.csv
  data/somatic_evidence_2026-07-28/wes_confirmed_somatic_annotated.csv

OUTPUT:
  data/paper_figs_2026-07-29/fig7a_hla_share.csv
  data/paper_figs_2026-07-29/fig7b_shared_genes.csv
  data/paper_figs_2026-07-29/fig7c_wes_confirmed.csv
  With Arial available:
    SPARCAL_pnas_2026/figs/v2_2026-07-29/fig7_hla_wes_evidence.{png,pdf}
  Without Arial (partial-redraw preview; panels a/b still await clarification):
    SPARCAL_pnas_2026/figs/v2_2026-07-29/fig7_hla_wes_evidence_preview.{png,pdf}

Run (env snv_caller, CPU, seconds): python scripts/postanalyze/fig7_hla_wes_evidence.py
To supply Arial explicitly:
  SPARCAL_ARIAL_FONT=/path/to/Arial.ttf python scripts/postanalyze/fig7_hla_wes_evidence.py
"""
import os

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
    """Use real Arial when available; otherwise write a named preview only."""
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
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-07-29")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v2_2026-07-29"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
HLA_COLOR = "#eb6834"       # orange -- HLA highlight (distinct from the locked germline/UPV/somatic hues)
NONHLA_COLOR = "#c9c8c2"    # light neutral gray
SOMATIC_RED = "#e34948"     # locked somatic color, used for panel (c) dots

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
IN_TISSUE_SPOTS = {"P4": 744, "P6": 3650}   # from data/somatic_evidence_2026-07-28/SUMMARY.md


# ---------------------------------------------------------------------------
# Panel (a): HLA share of somatic COSMIC hits
# ---------------------------------------------------------------------------
def build_fig7a():
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    som = hits[hits.class_code == "somatic"].copy()
    som["is_hla"] = som.gene.astype(str).str.startswith("HLA-")
    som["protein_altering"] = ~som.consequence.isin(["non_coding_or_unannotated", "synonymous"])

    rows = []
    for sk in SAMPLES:
        sub = som[som["sample"] == sk]
        n_all = len(sub)
        n_hla_all = int(sub.is_hla.sum())
        pa = sub[sub.protein_altering]
        n_pa = len(pa)
        n_hla_pa = int(pa.is_hla.sum())
        rows.append(dict(sample=sk, n_all_hits=n_all, n_hla_all_hits=n_hla_all,
                          pct_hla_all_hits=100 * n_hla_all / n_all,
                          n_protein_altering=n_pa, n_hla_protein_altering=n_hla_pa,
                          pct_hla_protein_altering=100 * n_hla_pa / n_pa if n_pa else np.nan))
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig7a_hla_share.csv")
    df.to_csv(out, index=False)
    print(f"[fig7a] wrote {out}")
    return df


# ---------------------------------------------------------------------------
# Panel (b): P4 x P6 shared somatic genes
# ---------------------------------------------------------------------------
def build_fig7b():
    cscc = pd.read_csv(os.path.join(COSMIC_DIR, "cscc_shared_p4_p6.csv"))
    cscc["is_hla"] = cscc.gene.astype(str).str.startswith("HLA-")
    cscc = cscc.sort_values(["n_protein_altering", "is_hla"], ascending=[False, False]).reset_index(drop=True)
    out = os.path.join(DERIVED_DIR, "fig7b_shared_genes.csv")
    cscc.to_csv(out, index=False)
    print(f"[fig7b] wrote {out}  "
          f"({len(cscc)} genes, {int(cscc.in_cgc.sum())} CGC, {int(cscc.is_hla.sum())} HLA, "
          f"{int(cscc.n_protein_altering.sum())} total protein-altering hits, "
          f"{int(cscc[cscc.is_hla].n_protein_altering.sum())} of them in HLA genes)")
    return cscc


# ---------------------------------------------------------------------------
# Panel (c): 14 WES-confirmed somatic variants
# ---------------------------------------------------------------------------
def build_fig7c():
    wes = pd.read_csv(os.path.join(EVID_DIR, "wes_confirmed_somatic_annotated.csv"))
    wes = wes[wes.class_label == "somatic"].copy()
    wes["gene_display"] = wes.gene.fillna("—")   # em dash for "no annotation"
    # The GJB2 call remains in the analysis and retains its COSMIC star, but the
    # advisor requested that its gene name not appear in the left-side row label.
    wes["gene_plot_label"] = wes["gene_display"]
    wes.loc[wes.gene.eq("GJB2"), "gene_plot_label"] = "—"
    wes["pct_spots"] = [100 * r.n_spots_calling / IN_TISSUE_SPOTS[r["sample"]] for _, r in wes.iterrows()]
    wes["variant_label"] = wes["sample"] + "  " + wes.chrom.astype(str) + ":" + wes.pos.map("{:,}".format)
    wes["has_gene"] = wes.gene.notna()
    wes["has_cosmic"] = wes.cosmic_id.notna()

    n_no_gene = int((~wes.has_gene).sum())
    n_cosmic = int(wes.has_cosmic.sum())
    n_edit = int(wes.possible_rna_editing_A2G_T2C.sum())
    n_mhc = int(wes.in_mhc_xMHC_chr6_28_34Mb.sum())
    print(f"[fig7c] {len(wes)} WES-confirmed somatic variants "
          f"({(wes['sample']=='P4').sum()} P4 + {(wes['sample']=='P6').sum()} P6)")
    print(f"[fig7c] no gene annotation: {n_no_gene}/{len(wes)}  "
          f"(** task brief / PAPER_PLAN.md quote 11/14 here -- DISAGREES with the "
          f"CSV; the deliverable script's own SUMMARY.md independently reports 13 "
          f"unannotated. Using the CSV-verified {n_no_gene}/{len(wes)}. **)")
    print(f"[fig7c] COSMIC hits: {n_cosmic}/{len(wes)}  possible RNA-editing (A>G/T>C): "
          f"{n_edit}/{len(wes)}  in xMHC: {n_mhc}/{len(wes)}")

    out = os.path.join(DERIVED_DIR, "fig7c_wes_confirmed.csv")
    wes.to_csv(out, index=False)
    print(f"[fig7c] wrote {out}")
    return wes


def main():
    df_a = build_fig7a()
    df_b = build_fig7b()
    df_c = build_fig7c()

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.45, 1.55], hspace=0.62, wspace=0.5)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0:2, 1])
    ax_c = fig.add_subplot(gs[1:, 0])
    ax_legend = fig.add_subplot(gs[2, 1])
    ax_legend.axis("off")

    # ---- Panel (a): HLA share, all hits vs protein-altering ----
    x = np.arange(len(SAMPLES))
    width = 0.36
    b1 = ax_a.bar(x - width / 2, df_a.set_index("sample").loc[SAMPLES].pct_hla_all_hits,
                   width=width, color=HLA_COLOR, alpha=0.55, edgecolor=HLA_COLOR,
                   linewidth=0.8, label="all COSMIC hits", zorder=3)
    b2 = ax_a.bar(x + width / 2, df_a.set_index("sample").loc[SAMPLES].pct_hla_protein_altering,
                   width=width, color=HLA_COLOR, edgecolor=HLA_COLOR, linewidth=0.8,
                   label="protein-altering hits", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{b.get_height():.1f}",
                       ha="center", va="bottom", fontsize=6.0, color=INK)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SAMPLES, fontsize=7.5)
    ax_a.set_ylabel("% of somatic-class\nCOSMIC hits that are HLA", fontsize=7.2)
    ax_a.set_ylim(0, 52)
    ax_a.tick_params(axis="y", labelsize=6.6)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_color(MUTED)
    ax_a.spines["bottom"].set_color(MUTED)
    ax_a.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.1,
                handletextpad=0.4, borderaxespad=0.1)
    ax_a.set_title("c", fontsize=10, fontweight="bold", loc="left", x=-0.24, y=1.03)

    # ---- Panel (b): P4-P6 shared somatic genes, HLA highlighted ----
    genes = df_b.gene.tolist()
    y = np.arange(len(genes))[::-1]
    colors = [HLA_COLOR if h else NONHLA_COLOR for h in df_b.is_hla]
    ax_b.barh(y, df_b.n_protein_altering, color=colors, edgecolor="white", linewidth=0.4,
              height=0.72, zorder=3)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(genes, fontsize=5.9,
                          fontstyle="italic")
    for lbl, is_hla in zip(ax_b.get_yticklabels(), df_b.is_hla):
        if is_hla:
            lbl.set_color(HLA_COLOR)
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(MUTED)
    ax_b.set_xlabel("protein-altering hits\nshared by P4 and P6", fontsize=7.2)
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.spines["bottom"].set_color(MUTED)
    ax_b.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    n_hla = int(df_b.is_hla.sum())
    hla_pa = int(df_b[df_b.is_hla].n_protein_altering.sum())
    tot_pa = int(df_b.n_protein_altering.sum())
    ax_b.text(0.98, 0.03,
              f"{n_hla} HLA genes carry {hla_pa}/{tot_pa}\nshared protein-altering hits "
              f"({100*hla_pa/tot_pa:.0f}%)\n0/{len(df_b)} genes are Cancer Gene Census",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.0,
              color=INK, bbox=dict(facecolor="white", edgecolor=GRID, pad=3))
    ax_b.set_title("P4 ∩ P6 shared somatic genes", fontsize=8.0, loc="center", pad=4, color=INK)
    ax_b.set_title("d", fontsize=10, fontweight="bold", loc="left", x=-0.30, y=1.02)

    # ---- Panel (c): 14 WES-confirmed somatic variants, dot plot ----
    order = df_c.sort_values(["sample", "pct_spots"], ascending=[True, False]).reset_index(drop=True)
    yc = np.arange(len(order))[::-1]
    vaf_scale = 900
    for i, (yy, r) in enumerate(zip(yc, order.itertuples())):
        edge = SOMATIC_RED
        ax_c.scatter([r.pct_spots], [yy], s=max(r.wes_tumor_vaf * vaf_scale, 18),
                     facecolor=SOMATIC_RED, edgecolor=edge, linewidth=1.3, zorder=3)
        if r.possible_rna_editing_A2G_T2C:
            ax_c.scatter([r.pct_spots], [yy], marker="^", s=24,
                         facecolor=INK, edgecolor="white", linewidth=0.45, zorder=4)
        if r.has_cosmic:
            ax_c.scatter([r.pct_spots], [yy], marker="*", s=90, facecolor="#eda100",
                         edgecolor="black", linewidth=0.4, zorder=5)
    ax_c.set_yticks(yc)
    ylabels = [f"{r.gene_plot_label}   {r.variant_label}" for r in order.itertuples()]
    ax_c.set_yticklabels(ylabels, fontsize=6.3)
    for lbl, r in zip(ax_c.get_yticklabels(), order.itertuples()):
        lbl.set_color(INK if r.has_gene else MUTED)
    ax_c.set_xlabel("% of in-tissue spots calling the variant", fontsize=7.2)
    ax_c.set_xlim(-0.3, 6.2)
    ax_c.tick_params(axis="x", labelsize=6.6)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color(MUTED)
    ax_c.spines["bottom"].set_color(MUTED)
    ax_c.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_c.set_axisbelow(True)
    # sample group separator
    n_p4 = int((order["sample"] == "P4").sum())
    ax_c.axhline(yc[n_p4 - 1] - 0.5, color=GRID, linewidth=0.8, zorder=1)
    ax_c.text(6.15, yc[:n_p4].mean(), "P4", fontsize=7, color=MUTED, ha="right", va="center",
              fontweight="bold")
    ax_c.text(6.15, yc[n_p4:].mean(), "P6", fontsize=7, color=MUTED, ha="right", va="center",
              fontweight="bold")
    ax_c.set_title("WES-confirmed SPARCAL somatic calls (n=14; 0 Cancer Gene Census)",
                    fontsize=7.6, loc="center", pad=4, color=INK)
    ax_c.set_title("e", fontsize=10, fontweight="bold", loc="left", x=-0.26, y=1.015)

    # ---- shared legend for panel (c) ----
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=SOMATIC_RED,
               markeredgecolor=SOMATIC_RED,
               label="WES-confirmed SPARCAL somatic call (dot size ∝ WES tumor VAF)"),
        Line2D([], [], marker="^", linestyle="none", markersize=6, markerfacecolor=INK,
               markeredgecolor="white",
               label="triangle = possible RNA editing (A>G/T>C), 2/14"),
        Line2D([], [], marker="*", linestyle="none", markersize=10, markerfacecolor="#eda100",
               markeredgecolor="black", label="COSMIC hit (GJB2 p.F141=, synonymous, non-CGC), 1/14"),
        Line2D([], [], color="none", label=f"gray label = no gene annotation, "
                                            f"{int((~order.has_gene).sum())}/14"),
    ]
    ax_legend.legend(handles=handles, loc="center left", fontsize=6.4, frameon=False,
                      handletextpad=0.6, labelspacing=1.1, borderaxespad=0)

    stem = "fig7_hla_wes_evidence" if HAS_ARIAL else "fig7_hla_wes_evidence_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial is unavailable; writing partial-redraw preview only. "
              "The manuscript asset is not being overwritten.")
    png = os.path.join(FIG_DIR, f"{stem}.png")
    pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[fig7] wrote {png}")
    print(f"[fig7] wrote {pdf}")


if __name__ == "__main__":
    main()
