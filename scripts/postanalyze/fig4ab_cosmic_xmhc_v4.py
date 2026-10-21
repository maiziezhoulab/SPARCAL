#!/usr/bin/env python
"""Figure 4a/b (v4, 2026-08-22) -- COSMIC separation and the xMHC contribution.

Rewritten from `fig5_cosmic_cascade.py` after the 2026-08-22 advisor revision.

Panel (a): two bars per sample -- SPARCAL somatic (red) and SPARCAL unresolved
    (green). Bar height is the catalogue hit rate, COSMIC hits / all variants
    called in that class. The revision asks for BOTH ratios to be readable on
    the bar itself, so:
      * INSIDE each bar  -> the hit-rate fraction that sets the bar height,
                            e.g. P4 somatic 243/19,523.
      * ABOVE each bar   -> how many of those catalogue hits fall in the
                            extended MHC, e.g. 31/243 for P4 somatic.
    The old hatched overlay encoded the second ratio geometrically and was hard
    to read off; it is replaced by the explicit numerator/denominator labels.

Panel (b): the somatic/unresolved hit-rate ratio with its 95% CI, INSIDE the
    extended MHC only. The raw and xMHC-excluded series were dropped at the
    advisor's request. See RATIO NOTE below.

RATIO NOTE (what panel b is and why it has an interval)
    ratio = (somatic hits / somatic variants) / (unresolved hits / unresolved
    variants), both restricted to chr6:28-34 Mb. It is a relative risk: how many
    times more often a retained somatic candidate is in COSMIC than an
    unresolved one. 1.0 = no separation. Each rate is a proportion estimated
    from a finite count (P4: 31/326 vs 68/1,141), so it carries binomial
    sampling error; the 95% CI is the Katz log-ratio normal approximation
    exp(log(p1/p2) +/- 1.96*SE), SE = sqrt((1-p1)/(p1*n1) + (1-p2)/(p2*n2)).
    The interval is sampling uncertainty in the two rates, NOT biological
    variation between patients. The p-value is a one-sided Fisher exact test.

CRITICAL FRAMING (unchanged, locked Decision D2):
    This is CLASS SEPARATION only, not evidence that retained calls are cancer
    drivers. COSMIC membership is a same-basis catalogue comparison, and the
    catalogue itself carries common germline polymorphism
    (data/cosmic_xmhc_2026-08-20/RESULT.md).

Run (env snv_caller, CPU, ~1 min):
  python scripts/postanalyze/fig4ab_cosmic_xmhc_v4.py

Outputs
  data/paper_figs_2026-08-22/fig4a_cosmic_hit_rates.csv
  data/paper_figs_2026-08-22/fig4b_xmhc_only_ratio.csv
  SPARCAL_pnas_2026/figs/v4_2026-08-22/fig4ab_cosmic_xmhc[_preview].{png,pdf}
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
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-08-22")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v4_2026-08-22"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
SAMPLE_TOKEN = {"P4": "p4_tumor", "P6": "p6_tumor", "DCIS1": "dcis1", "DCIS2": "dcis2"}
CLASS_ORDER = ["somatic", "ambiguous"]
CLASS_LABEL = {"somatic": "somatic", "ambiguous": "unresolved"}

# Per-class set sizes on the CURRENT post-dedup sets (PAPER_PLAN Decision D1),
# identical to cosmic_somatic_gene_annotation.py and the superseded fig5 script.
SET_TOTALS = {
    ("P4", "somatic"): 19523, ("P4", "ambiguous"): 175711,
    ("P6", "somatic"): 65655, ("P6", "ambiguous"): 590897,
    ("DCIS1", "somatic"): 18536, ("DCIS1", "ambiguous"): 166825,
    ("DCIS2", "somatic"): 25154, ("DCIS2", "ambiguous"): 226386,
}

MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SOMATIC_C = "#e34948"      # locked somatic red
UNRESOLVED_C = "#3f9b5c"   # unresolved green (2026-08-22 revision)
CLASS_COLOR = {"somatic": SOMATIC_C, "ambiguous": UNRESOLVED_C}


def build_fig4a():
    hits = pd.read_csv(os.path.join(COSMIC_DIR, "cosmic_hits_annotated.csv"))
    rows = []
    for sk in SAMPLES:
        for cls in CLASS_ORDER:
            sub = hits[(hits["sample"] == sk) & (hits.class_code == cls)]
            n_hits = len(sub)
            n_total = SET_TOTALS[(sk, cls)]
            in_mhc = ((sub.chrom.astype(str) == MHC_CHR)
                      & (sub.pos.astype(int).between(MHC_LO, MHC_HI)))
            n_hits_mhc = int(in_mhc.sum())
            rows.append({
                "sample": sk, "class_code": cls, "class_label": CLASS_LABEL[cls],
                "n_hits": n_hits, "n_total": n_total,
                "hit_rate_pct": 100 * n_hits / n_total,
                "n_hits_mhc": n_hits_mhc,
                "pct_hits_from_mhc": 100 * n_hits_mhc / n_hits if n_hits else float("nan"),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig4a_cosmic_hit_rates.csv")
    df.to_csv(out, index=False)
    print(f"[fig4a] wrote {out}")
    return df


def _count_vcf_total_and_mhc(path):
    p = subprocess.run([BCFTOOLS, "view", "-H", path], capture_output=True, text=True, check=True)
    tot = mhc = 0
    for line in p.stdout.splitlines():
        if not line:
            continue
        c = line.split("\t", 2)
        tot += 1
        if c[0] == MHC_CHR and MHC_LO <= int(c[1]) <= MHC_HI:
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
            tot += 1
            if c[0] == MHC_CHR and MHC_LO <= int(c[1]) <= MHC_HI:
                mhc += 1
    return tot, mhc


def _ratio_ci(h1, n1, h2, n2):
    """95% CI on rate1/rate2 via the Katz log-ratio normal approximation."""
    p1, p2 = h1 / n1, h2 / n2
    se = np.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2))
    logr = np.log(p1 / p2)
    return np.exp(logr - 1.96 * se), np.exp(logr + 1.96 * se)


def build_fig4b():
    rows = []
    for sk in SAMPLES:
        tok = SAMPLE_TOKEN[sk]
        counts = {}
        for cls in CLASS_ORDER:
            tot, tot_mhc = _count_vcf_total_and_mhc(
                os.path.join(AMB_DIR, f"{tok}_{cls}_nochr.vcf.gz"))
            hit, hit_mhc = _count_isec_total_and_mhc(
                os.path.join(AMB_DIR, f"isec_{tok}_{cls}", "0003.vcf"))
            counts[cls] = dict(tot=tot, tot_mhc=tot_mhc, hit=hit, hit_mhc=hit_mhc)
        s, a = counts["somatic"], counts["ambiguous"]
        sh, st, ah, at = s["hit_mhc"], s["tot_mhc"], a["hit_mhc"], a["tot_mhc"]
        ratio = (sh / st) / (ah / at)
        _, p = fisher_exact([[sh, st - sh], [ah, at - ah]], alternative="greater")
        ci_lo, ci_hi = _ratio_ci(sh, st, ah, at)
        rows.append(dict(sample=sk, variant="xMHC_only", ratio=ratio, ci_lo=ci_lo, ci_hi=ci_hi,
                         p_value=p, somatic_hit=sh, somatic_tot=st,
                         unresolved_hit=ah, unresolved_tot=at))
    df = pd.DataFrame(rows)
    out = os.path.join(DERIVED_DIR, "fig4b_xmhc_only_ratio.csv")
    df.to_csv(out, index=False)
    print(f"[fig4b] wrote {out}")
    return df


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    df_a = build_fig4a()
    df_b = build_fig4b()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), gridspec_kw=dict(width_ratios=[1.32, 1.0]))

    # ---- Panel (a) ----
    ax = axes[0]
    x = np.arange(len(SAMPLES))
    width = 0.36
    for i, cls in enumerate(CLASS_ORDER):
        sub = df_a[df_a.class_code == cls].set_index("sample").loc[SAMPLES]
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, sub.hit_rate_pct, width=width * 0.9, color=CLASS_COLOR[cls],
               label=f"SPARCAL {CLASS_LABEL[cls]}", edgecolor=INK, linewidth=0.65, zorder=3)
        for xi, r in zip(xpos, sub.itertuples()):
            # inside the bar: the fraction that sets the bar height
            # Keep the in-bar fraction inside the bar: the longest label
            # ("3,226/590,897") has to fit the shortest bar (P6 unresolved, 0.55%).
            ax.text(xi, r.hit_rate_pct * 0.5, f"{int(r.n_hits):,}/{int(r.n_total):,}",
                    ha="center", va="center", fontsize=5.6, color="white",
                    fontweight="bold", rotation=90, zorder=5)
            # above the bar: how many of those hits came from the xMHC
            ax.text(xi, r.hit_rate_pct + 0.035,
                    f"{int(r.n_hits_mhc)}/{int(r.n_hits)}",
                    ha="center", va="bottom", fontsize=6.6, color=CLASS_COLOR[cls],
                    fontweight="bold", rotation=90, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8.5)
    ax.set_ylabel("COSMIC hit rate (%)", fontsize=8.5)
    ax.set_ylim(0, 1.85)
    ax.tick_params(axis="y", labelsize=7.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(facecolor=CLASS_COLOR[c], edgecolor=INK, linewidth=0.65,
                              label=f"SPARCAL {CLASS_LABEL[c]}") for c in CLASS_ORDER]
    leg = ax.legend(handles=handles, fontsize=6.8, frameon=False, loc="upper right",
                    ncol=1, handlelength=1.1, handletextpad=0.5, borderaxespad=0.1)
    # Bare panel letter only; what the two printed fractions mean is stated in the
    # LaTeX caption (CLAUDE.md "No titles in the artwork").
    ax.set_title("a", fontsize=11, fontweight="bold", loc="left", x=-0.13, y=1.02)

    # ---- Panel (b) ----
    ax = axes[1]
    for row_i, sk in enumerate(SAMPLES):
        y = len(SAMPLES) - 1 - row_i
        r = df_b[df_b["sample"] == sk].iloc[0]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=SOMATIC_C, lw=1.4, zorder=2)
        for end in (r.ci_lo, r.ci_hi):
            ax.plot([end, end], [y - 0.09, y + 0.09], color=SOMATIC_C, lw=1.4, zorder=2)
        ax.scatter([r.ratio], [y], s=42, marker="D", facecolor=SOMATIC_C,
                   edgecolor=INK, linewidth=0.7, zorder=3)
        ax.text(r.ci_hi + 0.12, y + 0.16, f"{r.ratio:.2f}×  ({sig_label(r.p_value)})",
                fontsize=6.9, va="bottom", ha="left", color=INK, fontweight="bold")
        ax.text(r.ci_hi + 0.12, y - 0.16,
                f"{int(r.somatic_hit)}/{int(r.somatic_tot)} vs "
                f"{int(r.unresolved_hit)}/{int(r.unresolved_tot)}",
                fontsize=6.0, va="top", ha="left", color=MUTED)
    ax.axvline(1.0, color=MUTED, lw=0.9, linestyle="--", zorder=1)
    ax.text(1.02, -0.60, "no separation", fontsize=6.2, color=MUTED,
            ha="left", va="bottom")
    ax.set_yticks(range(len(SAMPLES)))
    ax.set_yticklabels(SAMPLES[::-1], fontsize=8.5)
    ax.set_xlabel("somatic / unresolved COSMIC hit-rate ratio\ninside the xMHC (95% CI)",
                  fontsize=7.6, linespacing=1.4)
    ax.set_xlim(0.7, 6.6)
    ax.set_ylim(-0.7, len(SAMPLES) - 0.25)
    ax.tick_params(axis="x", labelsize=7.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("b", fontsize=11, fontweight="bold", loc="left", x=-0.20, y=1.02)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(wspace=0.44)

    stem = "fig4ab_cosmic_xmhc" if HAS_ARIAL else "fig4ab_cosmic_xmhc_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans preview only.")
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[fig4ab] wrote {path}")


if __name__ == "__main__":
    main()
