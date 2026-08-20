#!/usr/bin/env python
"""Fig 5 -- COSMIC separation of SPARCAL somatic and unresolved results.

Panel (a): grouped bars, COSMIC hit rate (%) for the SPARCAL somatic and
           unresolved results. Germline and UPV are intentionally excluded from
           the main figure per the 2026-08 advisor revision.
Panel (b): the somatic-vs-unresolved contrast as a ratio with 95% CI, RAW and
           xMHC-excluded (chr6:28-34 Mb) side by side per sample, with Fisher's
           exact p-values marked honestly (including the non-significant ones).

CRITICAL FRAMING (locked Decision D2, PAPER_PLAN.md Sec 5 / Sec 6):
This figure shows CLASS SEPARATION only. It is NOT evidence that retained calls
are cancer-driving -- COSMIC catalogue membership is depth-associated for every
class alike (logistic cosmic_hit ~ is_somatic + log10(DP): OR 1.04, p=0.56 in
P4; OR 1.00, p=0.95 in P6 -- see PAPER_PLAN_DEPRECATED.md Sec 3.1). Do not caption
this figure as validating somatic calls as cancer-relevant.

INPUTS (recomputed here, not hardcoded):
  data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv
      -- one row per COSMIC hit x sample x class (from cosmic_somatic_gene_annotation.py)
  cosmic_amb/{sample}_{class}_nochr.vcf.gz
      -- the full per-class call sets (for xMHC total-count denominators; these
         totals are NOT present in any existing CSV, so this script reads the
         bgzipped VCFs directly via bcftools -- see "VERIFY BEFORE YOU PLOT" note
         in the source task; there is no way to avoid this without a new CSV)
  cosmic_amb/isec_{sample}_{class}/0003.vcf
      -- the COSMIC-hit subset of the same VCFs (for xMHC hit-count numerators)

Set sizes (denominators for panel a) are the same hardcoded SET_TOTALS used by
cosmic_somatic_gene_annotation.py (sourced from
COSMIC/somatic_vs_ambiguous_rates_2026-07-13.csv) -- reproduced here, not
re-derived, because no VCF-count round-trip is needed for panel (a).

OUTPUT:
  data/paper_figs_2026-07-29/fig5a_cosmic_hit_rates.csv     (derived, committed)
  data/paper_figs_2026-07-29/fig5b_somatic_vs_unresolved_ratio.csv  (derived, committed)
  With Arial available:
    SPARCAL_pnas_2026/figs/v2_2026-07-29/fig5_cosmic_cascade.{png,pdf}
  Without Arial (content/layout preview only; never replaces the paper asset):
    SPARCAL_pnas_2026/figs/v2_2026-07-29/fig5_cosmic_cascade_preview.{png,pdf}

Run (env snv_caller, CPU, seconds): python scripts/postanalyze/fig5_cosmic_cascade.py
To supply Arial explicitly:
  SPARCAL_ARIAL_FONT=/path/to/Arial.ttf python scripts/postanalyze/fig5_cosmic_cascade.py
"""
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
    """Use real Arial when available; otherwise produce a clearly named preview.

    Matplotlib silently substitutes another sans-serif when Arial is absent.
    That behavior previously made it too easy to overwrite a paper asset that
    did not meet the advisor's font requirement.
    """
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
AMB_DIR = os.path.join(PROJECT, "cosmic_amb")
BCFTOOLS = os.path.join(PROJECT, "apps", "bcftools")
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-07-29")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v2_2026-07-29"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]           # locked sample order
SAMPLE_TOKEN = {"P4": "p4_tumor", "P6": "p6_tumor", "DCIS1": "dcis1", "DCIS2": "dcis2"}
CLASS_ORDER = ["somatic", "ambiguous"]
CLASS_LABEL = {"defined": "germline (1kG)", "upv": "UPV", "somatic": "somatic",
               "ambiguous": "unresolved"}

# Same SET_TOTALS as cosmic_somatic_gene_annotation.py (source of truth for
# per-class set sizes on the CURRENT/post-dedup sets, PAPER_PLAN Decision D1).
SET_TOTALS = {
    ("P4", "defined"): 42375, ("P4", "upv"): 9491,
    ("P4", "somatic"): 19523, ("P4", "ambiguous"): 175711,
    ("P6", "defined"): 125269, ("P6", "upv"): 1744,
    ("P6", "somatic"): 65655, ("P6", "ambiguous"): 590897,
    ("DCIS1", "defined"): 140773, ("DCIS1", "upv"): 53120,
    ("DCIS1", "somatic"): 18536, ("DCIS1", "ambiguous"): 166825,
    ("DCIS2", "defined"): 129764, ("DCIS2", "upv"): 36323,
    ("DCIS2", "somatic"): 25154, ("DCIS2", "ambiguous"): 226386,
}

MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
CLASS_COLOR = {
    "defined": "#2a78d6",     # germline -- locked blue
    "upv": "#4a3aa7",         # UPV -- locked purple
    "somatic": "#e34948",     # somatic -- locked red
    "ambiguous": "#898781",   # unresolved -- muted gray (no locked color)
}


# ---------------------------------------------------------------------------
# Panel (a): COSMIC hit rate per class per sample
# ---------------------------------------------------------------------------
def build_fig5a():
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    rows = []
    for sk in SAMPLES:
        for cls in CLASS_ORDER:
            sub = hits[(hits["sample"] == sk) & (hits.class_code == cls)]
            n_hits = len(sub)
            n_total = SET_TOTALS[(sk, cls)]
            # xMHC share of this class's catalogue hits. chr6:28-34 Mb is 0.19% of
            # the genome, so a large share here is the concentration panel b tests.
            in_mhc = ((sub.chrom.astype(str) == MHC_CHR)
                      & (sub.pos.astype(int).between(MHC_LO, MHC_HI)))
            n_hits_mhc = int(in_mhc.sum())
            rows.append({
                "sample": sk, "class_code": cls, "class_label": CLASS_LABEL[cls],
                "n_hits": n_hits, "n_total": n_total,
                "hit_rate_pct": 100 * n_hits / n_total,
                "n_hits_mhc": n_hits_mhc,
                "hit_rate_mhc_pct": 100 * n_hits_mhc / n_total,
                "pct_hits_from_mhc": 100 * n_hits_mhc / n_hits if n_hits else float("nan"),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig5a_cosmic_hit_rates.csv")
    df.to_csv(out, index=False)
    print(f"[fig5a] wrote {out}")
    return df


# ---------------------------------------------------------------------------
# Panel (b): somatic vs unresolved ratio, raw and xMHC-excluded, with 95% CI
# ---------------------------------------------------------------------------
def _count_vcf_total_and_mhc(path):
    p = subprocess.run([BCFTOOLS, "view", "-H", path], capture_output=True, text=True, check=True)
    tot = mhc = 0
    for line in p.stdout.splitlines():
        if not line:
            continue
        c = line.split("\t", 2)
        chrom, pos = c[0], int(c[1])
        tot += 1
        if chrom == MHC_CHR and MHC_LO <= pos <= MHC_HI:
            mhc += 1
    return tot, mhc


def _count_isec_total_and_mhc(path):
    if not os.path.exists(path):
        return 0, 0
    tot = mhc = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t", 2)
            chrom, pos = c[0], int(c[1])
            tot += 1
            if chrom == MHC_CHR and MHC_LO <= pos <= MHC_HI:
                mhc += 1
    return tot, mhc


def _ratio_ci(h1, n1, h2, n2):
    """95% CI on rate1/rate2 via the log-ratio (Katz) normal approximation."""
    p1, p2 = h1 / n1, h2 / n2
    logr = np.log(p1 / p2)
    se = np.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2))
    return np.exp(logr - 1.96 * se), np.exp(logr + 1.96 * se)


def build_fig5b():
    rows = []
    for sk in SAMPLES:
        tok = SAMPLE_TOKEN[sk]
        counts = {}
        for cls in ["somatic", "ambiguous"]:
            vcf_path = os.path.join(AMB_DIR, f"{tok}_{cls}_nochr.vcf.gz")
            hit_path = os.path.join(AMB_DIR, f"isec_{tok}_{cls}", "0003.vcf")
            tot, tot_mhc = _count_vcf_total_and_mhc(vcf_path)
            hit, hit_mhc = _count_isec_total_and_mhc(hit_path)
            counts[cls] = dict(tot=tot, tot_mhc=tot_mhc, hit=hit, hit_mhc=hit_mhc)

        s, a = counts["somatic"], counts["ambiguous"]

        for variant, (sh, st, ah, at) in {
            "raw": (s["hit"], s["tot"], a["hit"], a["tot"]),
            "xMHC_excluded": (s["hit"] - s["hit_mhc"], s["tot"] - s["tot_mhc"],
                               a["hit"] - a["hit_mhc"], a["tot"] - a["tot_mhc"]),
            "xMHC_only": (s["hit_mhc"], s["tot_mhc"], a["hit_mhc"], a["tot_mhc"]),
        }.items():
            ratio = (sh / st) / (ah / at)
            _, p = fisher_exact([[sh, st - sh], [ah, at - ah]], alternative="greater")
            ci_lo, ci_hi = _ratio_ci(sh, st, ah, at)
            rows.append(dict(sample=sk, variant=variant, ratio=ratio, ci_lo=ci_lo, ci_hi=ci_hi,
                              p_value=p, somatic_hit=sh, somatic_tot=st,
                              unresolved_hit=ah, unresolved_tot=at))
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig5b_somatic_vs_unresolved_ratio.csv")
    df.to_csv(out, index=False)
    print(f"[fig5b] wrote {out}")
    return df


def sig_label(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    df_a = build_fig5a()
    df_b = build_fig5b()

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.45))

    # ---- Panel (a): grouped bars, hit rate % ----
    ax = axes[0]
    n_class = len(CLASS_ORDER)
    x = np.arange(len(SAMPLES))
    width = 0.8 / n_class
    for i, cls in enumerate(CLASS_ORDER):
        sub = df_a[df_a.class_code == cls].set_index("sample").loc[SAMPLES]
        xpos = x + (i - (n_class - 1) / 2) * width
        bars = ax.bar(xpos, sub.hit_rate_pct, width=width * 0.92,
                       color=CLASS_COLOR[cls], label=f"SPARCAL {CLASS_LABEL[cls]}",
                       edgecolor=INK, linewidth=0.65, zorder=3)
        # Overlay the xMHC-derived part of the same bar. It is drawn on top of the
        # bar it belongs to, so bar height still reads as the total hit rate.
        ax.bar(xpos, sub.hit_rate_mhc_pct, width=width * 0.92,
               bottom=sub.hit_rate_pct - sub.hit_rate_mhc_pct,
               color="none", edgecolor=INK, linewidth=0.65, hatch="////", zorder=4)
        # Annotate the raw fraction behind each bar rather than the percentage the
        # bar already encodes, so the reader sees the magnitude of each set (the
        # unresolved sets are ~10x the somatic sets and that is invisible in a rate).
        for b, (_, r) in zip(bars, sub.iterrows()):
            ax.text(b.get_x() + b.get_width() / 2, r.hit_rate_pct + 0.035,
                    f"{int(r.n_hits):,}/{int(r.n_total):,}",
                    ha="center", va="bottom", fontsize=5.5, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8)
    ax.set_ylabel("COSMIC hit rate (%)", fontsize=8)
    ax.set_ylim(0, df_a.hit_rate_pct.max() * 1.55)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    h_mhc_hatch = mpatches.Patch(facecolor="white", edgecolor=INK, linewidth=0.65,
                                 hatch="////", label="of which xMHC (chr6:28–34 Mb)")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles=handles + [h_mhc_hatch], labels=labels + [h_mhc_hatch.get_label()],
                    fontsize=6.2, frameon=False, loc="upper right", ncol=1,
                    handlelength=1.1, handletextpad=0.5, borderaxespad=0.1,
                    title="bar labels: COSMIC hits / variants in set")
    leg.get_title().set_fontsize(5.8)
    leg.get_title().set_color(MUTED)
    ax.set_title("a  SPARCAL somatic and unresolved results",
                 fontsize=8.2, fontweight="bold", loc="left", x=-0.16, y=1.02)

    # ---- Panel (b): forest-style ratio + 95% CI, raw vs xMHC-excluded ----
    ax = axes[1]
    variant_style = {
        "raw": dict(marker="o", facecolor="#e34948", edgecolor="#e34948", dy=0.26, label="raw"),
        "xMHC_excluded": dict(marker="o", facecolor="white", edgecolor="#e34948", dy=0.0,
                               label="xMHC-excluded (chr6:28–34 Mb)"),
        "xMHC_only": dict(marker="D", facecolor="#8c2d2c", edgecolor="#8c2d2c", dy=-0.26,
                           label="inside xMHC only"),
    }
    ylabels = []
    yticks = []
    for row_i, sk in enumerate(SAMPLES):
        y0 = len(SAMPLES) - 1 - row_i
        ylabels.append(sk)
        yticks.append(y0)
        for variant, style in variant_style.items():
            r = df_b[(df_b["sample"] == sk) & (df_b.variant == variant)].iloc[0]
            y = y0 + style["dy"]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=style["edgecolor"], lw=1.1, zorder=2)
            ax.scatter([r.ratio], [y], s=26, marker=style["marker"],
                       facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                       linewidth=1.1, zorder=3)
            lab = sig_label(r.p_value)
            ax.text(r.ci_hi + 0.045, y, f"{r.ratio:.2f}× ({lab})",
                     fontsize=5.6, va="center", ha="left", color=INK)
    ax.axvline(1.0, color=MUTED, lw=0.9, linestyle="--", zorder=1)
    ax.text(1.0, len(SAMPLES) - 0.53, "no enrichment", fontsize=5.4, color=MUTED,
             ha="center", va="bottom")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("somatic / unresolved COSMIC-hit-rate ratio", fontsize=7.5)
    ax.set_xlim(0.55, 5.6)
    ax.set_ylim(-0.72, len(SAMPLES) - 0.22)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    # manual legend (marker fill only; classes not reused here)
    h_raw = plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                        markerfacecolor="#e34948", markeredgecolor="#e34948", label="raw")
    h_mhc = plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                        markerfacecolor="white", markeredgecolor="#e34948",
                        label="xMHC-excluded")
    h_only = plt.Line2D([], [], marker="D", linestyle="none", markersize=4.4,
                         markerfacecolor="#8c2d2c", markeredgecolor="#8c2d2c",
                         label="inside xMHC only")
    ax.legend(handles=[h_raw, h_mhc, h_only], fontsize=6.2, frameon=False, loc="lower right",
              handletextpad=0.4, borderaxespad=0.1)
    ax.set_title("b  Somatic/unresolved ratio inside vs outside the xMHC",
                 fontsize=8.2, fontweight="bold", loc="left", x=-0.22, y=1.02)

    fig.suptitle("SPARCAL results separate somatic and unresolved candidates",
                 fontsize=10.2, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(wspace=0.55)

    stem = "fig5_cosmic_cascade" if HAS_ARIAL else "fig5_cosmic_cascade_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial is unavailable; writing Nimbus Sans preview only. "
              "The manuscript asset is not being overwritten.")
    png = os.path.join(FIG_DIR, f"{stem}.png")
    pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[fig5] wrote {png}")
    print(f"[fig5] wrote {pdf}")


if __name__ == "__main__":
    main()
