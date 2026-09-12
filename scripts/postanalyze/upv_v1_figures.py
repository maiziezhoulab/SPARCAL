#!/usr/bin/env python
"""Figures for doc/UPV_design_v1.md (advisor discussion, 2026-09-12).

Six figures, each backed by a derived source table written next to it, so every
plotted number can be traced without rerunning the heavy stages.

  fig1  allelic dropout: VAF of known germline converges only above ADP~50
  fig2  gnomAD corroboration: exact vs permuted-ALT null, and AF composition
  fig3  expected-carriage correction of the gnomAD germline estimate
  fig4  substitution spectrum: (C>T|G>A)/Tv meter and 6-channel profiles
  fig5  DLPFC clustering at 250 kb bin: SPARCAL vs GATK vs Strelka2
  fig6  UPV v1 tier composition

House style follows SPARCAL_pnas_2026/CLAUDE.md: Arial (licensed files under
fonts/arial), no titles in the artwork (bare panel letters only), dpi 300,
pdf+png, top/right spines off, horizontal grid only.

Colours. Variant classes use the manuscript convention (germline blue, UPV violet,
somatic red; unresolved classes neutral gray) and validate all-pairs for CVD.
Callers keep the project-wide colours (SPARCAL red, GATK green, Strelka2 blue) for
cross-figure consistency; SPARCAL-red vs GATK-green FAILS deutan separation
(dE 4.8), so in fig5 every caller is also identified by x-position and tick label
and colour never carries identity alone.

Run (env snv_caller):
  python scripts/postanalyze/upv_v1_figures.py            # compute + plot
  python scripts/postanalyze/upv_v1_figures.py --plot-only
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
DNAFREE = f"{PROJECT}/data/dnafree_2026-09-08"
OUT = f"{PROJECT}/doc/figures"
SRC = f"{OUT}/source"
SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
ROOT = {"P4": "data/P4_tumor/1", "P6": "data/P6_tumor/1",
        "DCIS1": "data/dcis1", "DCIS2": "data/dcis2"}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#ffffff"
GERM_C, UPV_C, SOM_C = "#2a78d6", "#4a3aa7", "#e34948"
AMB_C, UNC_C = "#898781", "#c9c7c0"
CLASS_C = {"panel": GERM_C, "upv": UPV_C, "ambiguous": AMB_C,
           "somatic": SOM_C, "uncalled": UNC_C}
CLASS_LABEL = {"panel": "1KGP panel germline", "upv": "UPV", "ambiguous": "ambiguous",
               "somatic": "somatic", "uncalled": "uncalled"}
SPARCAL_C, GATK_C, STRELKA_C = "#e34948", "#45a66a", "#2a78d6"
AF_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
TIER_C = {"confident": UPV_C, "editing": "#eb6834", "low_depth": "#c9c7c0"}
WES_UPV = {"P4": 45 / 2910, "P6": 12 / 618}      # UPV_P1_1_REPORT_2026-08-23.md, exact-or-ALT

DEPTH_BINS = [(1, 3), (4, 9), (10, 19), (20, 49), (50, 199), (200, 10**9)]
DEPTH_LAB = ["1–3", "4–9", "10–19", "20–49", "50–199", "200+"]
TI = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
CHANNELS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def configure_required_font():
    """Use licensed Arial when present; otherwise fall back and say so."""
    local_dir = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/fonts/arial"
    paths = font_manager.findSystemFonts(fontpaths=[local_dir]) if os.path.isdir(local_dir) else []
    arial = [p for p in paths if font_manager.FontProperties(fname=p).get_name().casefold() == "arial"]
    for p in arial:
        font_manager.fontManager.addfont(p)
    plt.rcParams["font.family"] = "Arial" if arial else "Nimbus Sans"
    return bool(arial)


HAS_ARIAL = configure_required_font()
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "legend.frameon": False, "legend.fontsize": 7.5,
})


def style(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def letter(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=11, color=INK)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    base = name if HAS_ARIAL else f"{name}_preview"
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{base}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {OUT}/{base}.{{pdf,png}}")


def label_ends(ax, items, x, min_gap):
    """Direct labels at line ends, nudged apart so they never collide."""
    items = sorted(items, key=lambda t: t[0])
    placed = []
    for y, text, color in items:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text, color))
    for y, text, color in placed:
        ax.text(x, y, text, color=INK, fontsize=7, va="center", ha="left")
        ax.plot([x - 0.12, x - 0.02], [y, y], color=color, lw=2, solid_capstyle="butt",
                clip_on=False)


# ─────────────────────────── compute ───────────────────────────
def compute_features():
    depth_rows, spec_rows = [], []
    for s in SAMPLES:
        f = pd.read_csv(f"{DNAFREE}/track_c_somatic/out/{s}.features.tsv.gz", sep="\t",
                        usecols=["cls", "ref", "alt", "ref_fwd", "ref_rev",
                                 "alt_fwd", "alt_rev", "is_edit"], low_memory=False)
        f = f[f.ref.isin(list("ACGT")) & f.alt.isin(list("ACGT")) & (f.ref != f.alt)]
        adp = f.ref_fwd + f.ref_rev + f.alt_fwd + f.alt_rev
        f = f.assign(adp=adp, vaf=(f.alt_fwd + f.alt_rev) / adp.replace(0, np.nan))
        for cls, g in f[f.adp > 0].groupby("cls"):
            for (lo, hi), lab in zip(DEPTH_BINS, DEPTH_LAB):
                x = g[(g.adp >= lo) & (g.adp <= hi)]
                if len(x) == 0:
                    continue
                v = x.vaf.values
                depth_rows.append({"sample": s, "cls": cls, "adp_bin": lab, "n": len(x),
                                   "het_like": float(np.mean(np.abs(v - 0.5) < 0.15)),
                                   "hom_like": float(np.mean(v > 0.9)),
                                   "edit_frac": float(x.is_edit.mean())})
        # pyrimidine-normalised 6-channel spectrum
        ref = f.ref.values
        alt = f.alt.values
        pur = np.isin(ref, ["A", "G"])
        r6 = np.where(pur, pd.Series(ref).map(COMP).values, ref)
        a6 = np.where(pur, pd.Series(alt).map(COMP).values, alt)
        f = f.assign(ch=pd.Series(r6).str.cat(pd.Series(a6), sep=">").values,
                     ti=[(r, a) in TI for r, a in zip(ref, alt)])
        for cls, g in f.groupby("cls"):
            n = len(g)
            ct = (g.ch == "C>T").sum()
            tv = (~g.ti).sum()
            row = {"sample": s, "cls": cls, "n": n, "ti_tv": g.ti.sum() / max(tv, 1),
                   "ct_over_tv": ct / max(tv, 1), "edit_frac": float(g.is_edit.mean())}
            vc = g.ch.value_counts()
            for c in CHANNELS:
                row[c] = vc.get(c, 0) / n
            spec_rows.append(row)
        print(f"  features {s}: {len(f):,}")
    pd.DataFrame(depth_rows).to_csv(f"{SRC}/fig1_vaf_by_depth.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(f"{SRC}/fig4_spectrum.csv", index=False)


def compute_gnomad():
    rows, afrows = [], []
    for s in SAMPLES:
        g = pd.concat([pd.read_csv(x, sep="\t", low_memory=False)
                       for x in sorted(glob.glob(f"{DNAFREE}/track_b/gnomad/{s}.chr*.tsv.gz"))],
                      ignore_index=True).drop_duplicates(["chrom", "pos", "ref", "alt"])
        g["afv"] = pd.to_numeric(g.af, errors="coerce")
        stat = {}
        for cls, flag in [("panel", "in_panel"), ("upv", "in_upv"),
                          ("ambiguous", "in_ambiguous"), ("somatic", "in_somatic")]:
            x = g[g[flag] == 1]
            h = x[x.allele_exact == 1]
            af = h.afv.fillna(0.0).clip(0, 1)
            stat[cls] = dict(n=len(x), hits=len(h), exact=len(h) / max(len(x), 1),
                             null=x.n_null_alts.sum() / max(x.n_null_possible.sum(), 1),
                             e_carried=float((1 - (1 - af) ** 2).sum()),
                             pass_hits=int((h.is_pass == 1).sum()))
        anchor = stat["panel"]["exact"]
        r_panel = stat["panel"]["hits"] / max(stat["panel"]["e_carried"], 1)
        for cls, d in stat.items():
            rep = (max(0.0, (d["exact"] - d["null"]) / (anchor - d["null"]))
                   if cls != "panel" else np.nan)
            corr = (d["e_carried"] * r_panel / max(d["n"], 1)) if cls != "panel" else np.nan
            rows.append({"sample": s, "cls": cls, **d,
                         "hits_over_e": d["hits"] / max(d["e_carried"], 1),
                         "panel_ratio": r_panel,
                         "f_reported": rep, "f_corrected": corr,
                         "wes_support": WES_UPV.get(s, np.nan) if cls == "upv" else np.nan})
        af = g[(g.in_upv == 1) & (g.allele_exact == 1)].afv.dropna()
        afrows.append({"sample": s, "n_hits_with_af": len(af),
                       "AF<1e-4": float(np.mean(af < 1e-4)),
                       "1e-4–1e-3": float(np.mean((af >= 1e-4) & (af < 1e-3))),
                       "1e-3–1e-2": float(np.mean((af >= 1e-3) & (af < 1e-2))),
                       "AF≥1e-2": float(np.mean(af >= 1e-2))})
        print(f"  gnomAD {s}: panel ratio {r_panel:.2f}")
    pd.DataFrame(rows).to_csv(f"{SRC}/fig2_fig3_gnomad.csv", index=False)
    pd.DataFrame(afrows).to_csv(f"{SRC}/fig2b_upv_hit_af.csv", index=False)


def compute_small():
    a = pd.read_csv(f"{PROJECT}/data/dlpfc/clustering_benchmark/ari_matrix_mean.csv")
    a[["section", "gene_expr", "defined_bin250kb", "germline_bin250kb",
       "gatk_bin250kb", "strelka2_bin250kb"]].to_csv(f"{SRC}/fig5_ari_250kb.csv", index=False)
    t = []
    for s in SAMPLES:
        d = pd.read_csv(f"{PROJECT}/{ROOT[s]}/spatial_filter_purity/baseQ0mapQ0/germline/denovo/"
                        "upv_v1_summary.tsv", sep="\t")
        for _, r in d.iterrows():
            t.append({"sample": s, "tier": r["tier"], "n": int(r["n"]), "fraction": r["fraction"]})
    pd.DataFrame(t).to_csv(f"{SRC}/fig6_upv_tiers.csv", index=False)


# ─────────────────────────── plot ───────────────────────────
def fig1():
    d = pd.read_csv(f"{SRC}/fig1_vaf_by_depth.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, s, lt in zip(axes, ["DCIS1", "P4"], "ab"):
        style(ax)
        ends = []
        for cls, col, key, ls, lab in [("panel", GERM_C, "het_like", "-", "germline, het-like"),
                                       ("panel", GERM_C, "hom_like", "--", "germline, hom-like"),
                                       ("upv", UPV_C, "het_like", "-", "UPV, het-like"),
                                       ("ambiguous", AMB_C, "het_like", "-", "ambiguous, het-like")]:
            x = d[(d["sample"] == s) & (d.cls == cls) & (d.n >= 50)].set_index("adp_bin")
            xs = [i for i, b in enumerate(DEPTH_LAB) if b in x.index]
            ys = [100 * x.loc[DEPTH_LAB[i], key] for i in xs]
            ax.plot(xs, ys, color=col, lw=2, ls=ls, marker="o", ms=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ends.append((ys[-1], lab, col))
        ax.axvline(3.5, color=INK, lw=0.8, ls=":")
        ax.text(3.55, 97, "ADP 50", fontsize=7, color=INK, va="top")
        ax.set_xticks(range(len(DEPTH_LAB)))
        ax.set_xticklabels(DEPTH_LAB)
        ax.set_xlabel("allelic depth (ADP, reads)")
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.3, 5.3)
        ax.text(0.02, 0.98, s, transform=ax.transAxes, fontsize=8, color=INK, va="top")
        letter(ax, lt)
    axes[0].set_ylabel("sites (%)")
    handles = [Line2D([], [], color=GERM_C, lw=2, label="1KGP panel germline — het-like"),
               Line2D([], [], color=GERM_C, lw=2, ls="--", label="1KGP panel germline — hom-like"),
               Line2D([], [], color=UPV_C, lw=2, label="UPV — het-like"),
               Line2D([], [], color=AMB_C, lw=2, label="ambiguous — het-like")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    save(fig, "upv_v1_fig1_allelic_dropout")


def fig2():
    d = pd.read_csv(f"{SRC}/fig2_fig3_gnomad.csv")
    af = pd.read_csv(f"{SRC}/fig2b_upv_hit_af.csv").set_index("sample")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    style(ax)
    classes = ["upv", "ambiguous", "somatic"]
    off = {"upv": -0.22, "ambiguous": 0.0, "somatic": 0.22}
    for i, s in enumerate(SAMPLES):
        for cls in classes:
            r = d[(d["sample"] == s) & (d.cls == cls)].iloc[0]
            x = i + off[cls]
            ax.plot([x, x], [100 * r.null, 100 * r.exact], color=GRID, lw=2, zorder=1)
            ax.scatter([x], [100 * r.null], s=26, facecolor=SURFACE, edgecolor=MUTED, lw=1.2, zorder=3)
            ax.scatter([x], [100 * r.exact], s=34, facecolor=CLASS_C[cls], edgecolor=SURFACE,
                       lw=1.2, zorder=4)
        r = d[(d["sample"] == s) & (d.cls == "upv")].iloc[0]
        ax.text(i - 0.22, 100 * r.exact + 1.4, f"{r.exact / r.null:.2f}×", ha="center",
                fontsize=6.5, color=INK)
    ax.set_xticks(range(len(SAMPLES)))
    ax.set_xticklabels(SAMPLES)
    ax.set_ylabel("gnomAD allele-exact rate (%)")
    ax.set_ylim(0, 36)
    handles = [Line2D([], [], marker="o", ls="none", color=CLASS_C[c], markeredgecolor=SURFACE,
                      ms=6, label=CLASS_LABEL[c]) for c in classes]
    handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
                          markeredgecolor=MUTED, ms=5.5, label="permuted-ALT null"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4,
              fontsize=6.5, handletextpad=0.2, columnspacing=0.9)
    letter(ax, "a")

    style(bx, grid=False)
    bx.xaxis.grid(True, color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)
    bins = ["AF<1e-4", "1e-4–1e-3", "1e-3–1e-2", "AF≥1e-2"]
    for j, s in enumerate(SAMPLES):
        left = 0.0
        for b, col in zip(bins, AF_RAMP):
            w = 100 * af.loc[s, b]
            bx.barh(j, w, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62)
            if w >= 9:
                bx.text(left + w / 2, j, f"{w:.0f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if col in AF_RAMP[1:] else INK)
            left += w
    bx.set_yticks(range(len(SAMPLES)))
    bx.set_yticklabels(SAMPLES)
    bx.invert_yaxis()
    bx.set_xlim(0, 100)
    bx.set_xlabel("UPV gnomAD hits by allele frequency (%)")
    bx.legend(handles=[Patch(facecolor=c, label=b) for b, c in zip(bins, AF_RAMP)],
              loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=6.5,
              handlelength=1.0, columnspacing=0.8)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "upv_v1_fig2_gnomad_corroboration")


def fig3():
    d = pd.read_csv(f"{SRC}/fig2_fig3_gnomad.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    style(ax, grid=False)
    ax.grid(True, color=GRID, linewidth=0.5, which="major")
    ax.set_axisbelow(True)
    marker = {"P4": "o", "P6": "s", "DCIS1": "^", "DCIS2": "D"}
    lo, hi = 10, 3e5
    xx = np.array([lo, hi])
    rmin, rmax = d.panel_ratio.min(), d.panel_ratio.max()
    ax.fill_between(xx, xx * rmin, xx * rmax, color=GERM_C, alpha=0.12, lw=0, zorder=1)
    ax.plot(xx, xx, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    place = {"DCIS1": (0.78, 1.32, "right"), "DCIS2": (1.28, 0.78, "left"),
             "P4": (1.3, 1.0, "left"), "P6": (1.3, 1.0, "left")}
    for _, r in d.iterrows():
        ax.scatter(r.e_carried, r.hits, marker=marker[r["sample"]],
                   s=40 if r.cls == "upv" else 30,
                   facecolor=CLASS_C[r.cls], edgecolor=SURFACE, lw=1.0,
                   # DCIS1/DCIS2 UPV sit 0.14 decades apart; keep the triangle visible
                   zorder=(6 if r["sample"] == "DCIS1" else 5) if r.cls == "upv" else 3)
        if r.cls == "upv":
            fx, fy, ha = place[r["sample"]]
            ax.text(r.e_carried * fx, r.hits * fy, f"{r['sample']} UPV {r.hits_over_e:.1f}×",
                    fontsize=6.5, color=INK, va="center", ha=ha)
    ax.set_xlabel("expected carried by one individual, Σ(1−(1−f)²)")
    ax.set_ylabel("gnomAD allele-exact hits")
    h1 = [Line2D([], [], marker="o", ls="none", color=CLASS_C[c], markeredgecolor=SURFACE,
                 ms=6, label=CLASS_LABEL[c]) for c in ["panel", "upv", "ambiguous", "somatic"]]
    h2 = [Line2D([], [], marker=marker[s], ls="none", color=MUTED, markeredgecolor=SURFACE,
                 ms=6, label=s) for s in SAMPLES]
    fig.legend(handles=h1 + h2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               bbox_transform=ax.transAxes, fontsize=6.5, ncol=4, handletextpad=0.2,
               columnspacing=0.8)
    ax.legend(handles=[Patch(facecolor=GERM_C, alpha=0.25, edgecolor="none",
                             label=f"germline calibration\n(hits/E = {rmin:.1f}–{rmax:.1f})"),
                       Line2D([], [], color=MUTED, lw=0.8, ls=":", label="hits = E")],
              loc="lower right", fontsize=6.5, handlelength=1.4)
    ax.text(0, 1.17, "a", transform=ax.transAxes, fontweight="bold", fontsize=11, color=INK)

    style(bx)
    u = d[d.cls == "upv"].set_index("sample")
    for i, s in enumerate(SAMPLES):
        r = u.loc[s]
        bx.plot([i, i], [100 * r.f_corrected, 100 * r.f_reported], color=GRID, lw=2, zorder=1)
        bx.scatter([i], [100 * r.f_reported], s=34, facecolor=SURFACE, edgecolor=MUTED, lw=1.2, zorder=3)
        bx.scatter([i], [100 * r.f_corrected], s=40, facecolor=UPV_C, edgecolor=SURFACE, lw=1.2, zorder=4)
        bx.text(i + 0.14, 100 * r.f_corrected, f"{100 * r.f_corrected:.1f}%", fontsize=6.5,
                color=INK, va="center")
        bx.text(i + 0.14, 100 * r.f_reported, f"{100 * r.f_reported:.1f}%", fontsize=6.5,
                color=MUTED, va="center")
        if not np.isnan(r.wes_support):
            bx.scatter([i - 0.16], [100 * r.wes_support], marker="D", s=28, facecolor=INK,
                       edgecolor=SURFACE, lw=1.0, zorder=5)
    bx.set_xticks(range(len(SAMPLES)))
    bx.set_xticklabels(SAMPLES)
    bx.set_xlim(-0.5, 3.7)
    bx.set_ylim(0, 22)
    bx.set_ylabel("UPV estimated as germline (%)")
    bx.legend(handles=[
        Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE, markeredgecolor=MUTED,
               ms=6, label="chance-corrected (uncorrected for carriage)"),
        Line2D([], [], marker="o", ls="none", color=UPV_C, markeredgecolor=SURFACE, ms=6,
               label="carriage-corrected"),
        Line2D([], [], marker="D", ls="none", color=INK, markeredgecolor=SURFACE, ms=5,
               label="matched-normal WES support")], loc="upper left", fontsize=6.5)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "upv_v1_fig3_carriage_correction")


def fig4():
    d = pd.read_csv(f"{SRC}/fig4_spectrum.csv")
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    classes = ["panel", "upv", "ambiguous", "somatic", "uncalled"]
    offs = dict(zip(classes, np.linspace(-0.28, 0.28, 5)))
    mk = {"panel": "o", "upv": "o", "ambiguous": "o", "somatic": "o", "uncalled": "s"}
    ref = d[d.cls == "panel"].ct_over_tv.mean()
    ax.axhline(ref, color=GERM_C, lw=0.8, ls="--")
    ax.axhline(0.25, color=MUTED, lw=0.8, ls=":")
    ax.text(3.45, ref + 0.03, f"true germline {ref:.2f}", fontsize=6.5, color=GERM_C, ha="right")
    ax.text(3.45, 0.25 + 0.03, "uniform random error 0.25", fontsize=6.5, color=MUTED, ha="right")
    for i, s in enumerate(SAMPLES):
        for cls in classes:
            r = d[(d["sample"] == s) & (d.cls == cls)]
            if r.empty:
                continue
            ax.scatter([i + offs[cls]], [r.ct_over_tv.iloc[0]], marker=mk[cls], s=34,
                       facecolor=CLASS_C[cls], edgecolor=SURFACE, lw=1.0, zorder=3)
    ax.set_xticks(range(len(SAMPLES)))
    ax.set_xticklabels(SAMPLES)
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("(C>T | G>A) / transversions")
    ax.legend(handles=[Line2D([], [], marker=mk[c], ls="none", color=CLASS_C[c],
                              markeredgecolor=SURFACE, ms=6, label=CLASS_LABEL[c]) for c in classes],
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=5, fontsize=7)
    letter(ax, "a")
    for k, (s, lt) in enumerate([("P4", "b"), ("DCIS1", "c")]):
        bx = fig.add_subplot(gs[1, k])
        style(bx)
        w = 0.38
        for j, (cls, dx) in enumerate([("panel", -w / 2), ("upv", w / 2)]):
            r = d[(d["sample"] == s) & (d.cls == cls)].iloc[0]
            bx.bar(np.arange(6) + dx, [100 * r[c] for c in CHANNELS], width=w,
                   color=CLASS_C[cls], edgecolor=SURFACE, linewidth=2)
        bx.set_xticks(range(6))
        bx.set_xticklabels(CHANNELS)
        bx.set_ylim(0, 55)
        bx.text(0.02, 0.97, s, transform=bx.transAxes, fontsize=8, color=INK, va="top")
        if k == 0:
            bx.set_ylabel("substitutions (%)")
            bx.legend(handles=[Patch(facecolor=GERM_C, label="1KGP panel germline"),
                               Patch(facecolor=UPV_C, label="UPV")],
                      loc="upper right", fontsize=7)
        letter(bx, lt)
    save(fig, "upv_v1_fig4_substitution_spectrum")


def fig5():
    from scipy import stats
    a = pd.read_csv(f"{SRC}/fig5_ari_250kb.csv")
    cols = ["gene_expr", "defined_bin250kb", "germline_bin250kb", "gatk_bin250kb", "strelka2_bin250kb"]
    labs = ["gene\nexpression", "SPARCAL\n1kG-defined", "SPARCAL\n1kG+UPV", "GATK", "Strelka2"]
    cols_c = [MUTED, SPARCAL_C, SPARCAL_C, GATK_C, STRELKA_C]
    hollow = [False, False, True, False, False]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    style(ax)
    rng = np.random.default_rng(0)
    for i, (c, col, ho) in enumerate(zip(cols, cols_c, hollow)):
        v = a[c].dropna().values
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=MUTED, lw=0.8),
                   medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED, lw=0.8),
                   capprops=dict(color=MUTED, lw=0.8))
        jit = rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(i + jit, v, s=20, facecolor=SURFACE if ho else col, edgecolor=col,
                   lw=1.3 if ho else 0.8, zorder=3)
        ax.text(i, 0.63, f"{v.mean():.3f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labs)
    ax.set_ylim(0.1, 0.66)
    ax.set_ylabel("ARI (250 kb bin)")
    letter(ax, "a")

    style(bx)
    g = a["gatk_bin250kb"]
    pairs = [("defined_bin250kb", "SPARCAL\n1kG-defined", False),
             ("germline_bin250kb", "SPARCAL\n1kG+UPV", True)]
    bx.axhline(0, color=INK, lw=0.8)
    for i, (c, lab, ho) in enumerate(pairs):
        dv = (a[c] - g).dropna().values
        _, p = stats.wilcoxon(a[c], g)
        jit = rng.uniform(-0.12, 0.12, len(dv))
        bx.scatter(i + jit, dv, s=20, facecolor=SURFACE if ho else SPARCAL_C, edgecolor=SPARCAL_C,
                   lw=1.3 if ho else 0.8, zorder=3)
        bx.plot([i - 0.22, i + 0.22], [dv.mean()] * 2, color=INK, lw=1.6, zorder=4)
        bx.text(i, 0.155, f"P = {p:.3f}\n{(dv > 0).sum()}/{len(dv)} sections", ha="center",
                fontsize=6.5, color=INK)
    bx.set_xticks([0, 1])
    bx.set_xticklabels([p[1] for p in pairs])
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(-0.22, 0.2)
    bx.set_ylabel("ΔARI vs GATK (paired)")
    letter(bx, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "upv_v1_fig5_clustering_250kb")


def fig6():
    t = pd.read_csv(f"{SRC}/fig6_upv_tiers.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    style(ax, grid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    tiers = ["confident", "editing", "low_depth"]
    labels = {"confident": "confident (ADP≥50, non-editing)", "editing": "editing (catalogue hit)",
              "low_depth": "low_depth (ADP<50)"}
    for j, s in enumerate(SAMPLES):
        x = t[t["sample"] == s].set_index("tier")
        left = 0.0
        for tr in tiers:
            w = 100 * x.loc[tr, "fraction"]
            ax.barh(j, w, left=left, color=TIER_C[tr], edgecolor=SURFACE, linewidth=2, height=0.62)
            if w >= 4.5:
                ax.text(left + w / 2, j, f"{int(x.loc[tr, 'n']):,}", ha="center", va="center",
                        fontsize=6.5, color=SURFACE if tr != "low_depth" else INK)
            left += w
        ax.text(101, j, f"n = {int(x.loc['total', 'n']):,}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(SAMPLES)))
    ax.set_yticklabels(SAMPLES)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("UPV calls (%)")
    ax.legend(handles=[Patch(facecolor=TIER_C[k], label=labels[k]) for k in tiers],
              loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=7)
    save(fig, "upv_v1_fig6_tiers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(SRC, exist_ok=True)
    print(f"Arial: {'licensed' if HAS_ARIAL else 'MISSING -> _preview basenames'}")
    if not a.plot_only:
        compute_features()
        compute_gnomad()
    compute_small()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()


if __name__ == "__main__":
    main()
