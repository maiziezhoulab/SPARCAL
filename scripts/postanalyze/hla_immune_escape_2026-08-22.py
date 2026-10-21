#!/usr/bin/env python
"""HLA / xMHC deep dive -- what HLA mutations each sample carries, where they sit
in the tissue, and whether the HLA locus is copy-number altered (2026-08-22).

This is an ANALYSIS PACKAGE, not a manuscript figure. It answers three questions
the advisor raised, in order:

  1. WHICH HLA loci are mutated in each sample, split into class I (HLA-A/B/C,
     plus the non-classical E/F/G) and class II (HLA-DP/DQ/DR, plus DM/DO)?
  2. WHERE in the tissue are the spots that carry those mutations -- i.e. is
     there a spatially restricted, potentially immune-escaped compartment?
  3. Is the HLA region copy-number altered, in particular LOH, in the matched
     CalicoST allele-specific CNV profile?

METHOD AND ITS LIMITS -- read before quoting anything from this package.

  * HLA gene footprints are derived per build from the build-matched COSMIC
    Genome Screens Mutant VCF, restricted to canonical transcripts, as the
    min/max annotated position per gene. The hg19 footprints reproduce the
    GENCODE v37lift37 gene bounds (HLA-A 29,909,108-29,913,495 here vs
    29,909,037-29,913,661 in GENCODE), which cross-validates the construction.
    HLA-DPA1 and HLA-DPB1 genuinely overlap; variants inside the shared span are
    flagged `ambiguous_locus` and counted once, under the first gene by
    coordinate.
  * A "call in an HLA gene" is a SPARCAL somatic-class call whose position falls
    in that footprint. Positions are NOT HLA-typed: standard short-read RNA
    alignment against a single reference collapses the true polymorphic
    haplotypes, so these are reference-relative variants, not allele-typed HLA
    mutations. Real HLA typing needs a dedicated caller (e.g. OptiType/arcasHLA)
    on the same BAMs. Every count here inherits that caveat.
  * The HLA region is the single most polymorphic locus in the genome and is a
    known mapping-artifact hotspot. Sequence divergence from the reference
    produces reference-mismatch pileups that any caller will report as variants.
    A high HLA burden is therefore NOT by itself evidence of somatic mutation,
    and this package does not claim it is.
  * Spot burden is confounded by per-spot coverage. Every burden-vs-tumour
    association below is reported raw AND after conditioning on the spot's total
    SPARCAL somatic burden, which is the repo's standing requirement.

INPUTS
  data/<sample>/.../matrix/*_SPARCAL_somatic_matrix.pkl   (spots x variants)
  data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv  (COSMIC gene/AA)
  COSMIC/Cosmic_GenomeScreensMutant_v103_{GRCh37,GRCh38}.vcf.gz  (footprints)
  spaceranger / viewer tissue_positions.csv               (spatial coordinates)
  CalicoST/<run>/calicost/clone3_rectangle0_w1.0/{cnv_genelevel,cnv_seglevel,
      clone_labels}.tsv                                   (CNV, LOH, purity)

OUTPUTS  (data/hla_immune_escape_2026-08-22/)
  hla_gene_footprints.csv      per build, the footprint actually used
  hla_variant_inventory.csv    one row per HLA-region SPARCAL somatic call
  hla_locus_summary.csv        per sample x HLA gene rollup
  hla_spot_burden.csv          per sample x spot: class I / II burden + purity
  hla_cnv_status.csv           per sample x clone x HLA gene CalicoST A/B copies
  hla_purity_association.csv   burden vs tumour proportion, raw and conditional
  fig_hla_locus_inventory.{png,pdf}
  fig_hla_spatial.{png,pdf}
  fig_hla_cnv.{png,pdf}

Run (env snv_caller, CPU, ~3 min):
  python scripts/postanalyze/hla_immune_escape_2026-08-22.py
"""
import io
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
for fam in ("Arial", "Nimbus Sans", "DejaVu Sans"):
    try:
        font_manager.FontProperties(family=fam)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
COSMIC_ROOT = "/data/maiziezhou_lab/leiy4/COSMIC"
CALICO = "/data/maiziezhou_lab/leiy4/CalicoST"
VIEWER = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer"
STMUT = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/"
         "ST_datasets/STmut_Data")
BCFTOOLS = os.path.join(PROJECT, "apps", "bcftools")
OUT = os.path.join(PROJECT, "data", "hla_immune_escape_2026-08-22")
os.makedirs(OUT, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
BUILD = {"P4": "GRCh37", "P6": "GRCh37", "DCIS1": "GRCh38", "DCIS2": "GRCh38"}
MATRIX = {
    "P4": f"{PROJECT}/data/P4_tumor/1/matrix/P4_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    "P6": f"{PROJECT}/data/P6_tumor/1/matrix/P6_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    "DCIS1": f"{PROJECT}/data/dcis1/matrix/DCIS_dcis1_SPARCAL_somatic_matrix.pkl",
    "DCIS2": f"{PROJECT}/data/dcis2/matrix/DCIS_dcis2_SPARCAL_somatic_matrix.pkl",
}
POSITIONS = {
    "P4": f"{STMUT}/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/spatial/tissue_positions.csv",
    "P6": f"{STMUT}/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/spatial/tissue_positions.csv",
    "DCIS1": f"{VIEWER}/DCIS_1_SPARCAL_1000G_umi_dedup/tissue_positions.csv",
    "DCIS2": f"{VIEWER}/DCIS_2_SPARCAL/tissue_positions.csv",
}
CALICO_RUN = {"P4": "P4_sec1", "P6": "P6_sec1", "DCIS1": "DCIS1", "DCIS2": "DCIS2"}
CALICO_SUB = "calicost/clone3_rectangle0_w1.0"

CLASS_I_CLASSICAL = ["HLA-A", "HLA-B", "HLA-C"]
CLASS_I_NONCLASSICAL = ["HLA-E", "HLA-F", "HLA-G"]
CLASS_II_CLASSICAL = ["HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DQA1", "HLA-DQA2",
                      "HLA-DQB1", "HLA-DQB2", "HLA-DPA1", "HLA-DPB1"]
CLASS_II_NONCLASSICAL = ["HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB"]
CLASS_OF = ({g: "class I (classical)" for g in CLASS_I_CLASSICAL}
            | {g: "class I (non-classical)" for g in CLASS_I_NONCLASSICAL}
            | {g: "class II (DP/DQ/DR)" for g in CLASS_II_CLASSICAL}
            | {g: "class II (DM/DO)" for g in CLASS_II_NONCLASSICAL})
CLASS_ORDER = ["class I (classical)", "class I (non-classical)",
               "class II (DP/DQ/DR)", "class II (DM/DO)"]
CLASS_COLOR = {"class I (classical)": "#d1495b", "class I (non-classical)": "#edae49",
               "class II (DP/DQ/DR)": "#2a78d6", "class II (DM/DO)": "#8fb8de"}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
MHC_LO, MHC_HI = 28_000_000, 34_000_000


# ---------------------------------------------------------------------------
# 1. HLA gene footprints, per build, from the build-matched COSMIC catalogue
# ---------------------------------------------------------------------------
def hla_footprints():
    rows = []
    for build in ("GRCh37", "GRCh38"):
        vcf = f"{COSMIC_ROOT}/Cosmic_GenomeScreensMutant_v103_{build}.vcf.gz"
        proc = subprocess.run(
            [BCFTOOLS, "query", "-r", f"6:{MHC_LO}-{MHC_HI}",
             "-f", "%POS\t%INFO/GENE\t%INFO/IS_CANONICAL\n", vcf],
            capture_output=True, text=True, check=True)
        df = pd.read_csv(io.StringIO(proc.stdout), sep="\t", header=None,
                         names=["pos", "gene", "canonical"])
        df = df[df.gene.isin(CLASS_OF) & df.canonical.eq("y")]
        for gene, grp in df.groupby("gene"):
            rows.append(dict(build=build, gene=gene, chrom="6",
                             start=int(grp.pos.min()), end=int(grp.pos.max()),
                             n_catalogue_records=len(grp),
                             hla_class=CLASS_OF[gene]))
    fp = pd.DataFrame(rows).sort_values(["build", "start"]).reset_index(drop=True)
    fp.to_csv(f"{OUT}/hla_gene_footprints.csv", index=False)
    print(f"[footprints] wrote {OUT}/hla_gene_footprints.csv "
          f"({fp.gene.nunique()} genes x 2 builds)")
    return fp


def assign_locus(fp_build, pos):
    """Return (gene, ambiguous) for a position, or (None, False)."""
    hits = fp_build[(fp_build.start <= pos) & (pos <= fp_build.end)]
    if hits.empty:
        return None, False
    return hits.iloc[0].gene, len(hits) > 1


# ---------------------------------------------------------------------------
# 2. Per-sample HLA variant inventory and per-spot burden
# ---------------------------------------------------------------------------
def build_inventory(fp):
    cosmic = pd.read_csv(f"{PROJECT}/data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv")
    cosmic = cosmic[cosmic.class_code.eq("somatic")]
    cosmic_key = {(r["sample"], str(r.chrom), int(r.pos)): r
                  for _, r in cosmic.iterrows()}

    inventory, spot_rows = [], []
    for sk in SAMPLES:
        fpb = fp[fp.build.eq(BUILD[sk])]
        mat = pickle.load(open(MATRIX[sk], "rb"))
        chroms = np.array([c.split("_")[0] for c in mat.columns])
        poss = np.array([int(c.split("_")[1]) for c in mat.columns])
        in_mhc = (chroms == "6") & (poss >= MHC_LO) & (poss <= MHC_HI)

        hla_cols, hla_genes, hla_amb = [], [], []
        for col, pos in zip(mat.columns[in_mhc], poss[in_mhc]):
            gene, amb = assign_locus(fpb, pos)
            if gene is None:
                continue
            hla_cols.append(col)
            hla_genes.append(gene)
            hla_amb.append(amb)

        sub = mat[hla_cols]
        n_spots_total = mat.shape[0]
        total_burden = (mat.values > 0).sum(axis=1)

        for col, gene, amb in zip(hla_cols, hla_genes, hla_amb):
            pos = int(col.split("_")[1])
            ns = int((sub[col].values > 0).sum())
            c = cosmic_key.get((sk, "6", pos))
            inventory.append(dict(
                sample=sk, build=BUILD[sk], chrom="6", pos=pos, gene=gene,
                hla_class=CLASS_OF[gene], ambiguous_locus=amb,
                n_spots=ns, pct_spots=100 * ns / n_spots_total,
                cosmic_hit=c is not None,
                cosmic_id=(c.cosmic_id if c is not None else None),
                aa=(c.aa if c is not None else None),
                consequence=(c.consequence if c is not None else None)))

        cls_of_col = np.array([CLASS_OF[g] for g in hla_genes])
        present = sub.values > 0
        per_spot = {cls: present[:, cls_of_col == cls].sum(axis=1) for cls in CLASS_ORDER}
        for i, bc in enumerate(mat.index):
            spot_rows.append(dict(
                sample=sk, barcode=bc,
                n_class_I=int(per_spot["class I (classical)"][i]),
                n_class_I_nonclassical=int(per_spot["class I (non-classical)"][i]),
                n_class_II=int(per_spot["class II (DP/DQ/DR)"][i]),
                n_class_II_dm_do=int(per_spot["class II (DM/DO)"][i]),
                n_hla_total=int(present[i].sum()),
                n_somatic_total=int(total_burden[i])))
        print(f"[inventory] {sk}: {len(hla_cols)} SPARCAL somatic calls inside HLA "
              f"footprints ({int(in_mhc.sum())} in the xMHC overall)")

    inv = pd.DataFrame(inventory)
    inv.to_csv(f"{OUT}/hla_variant_inventory.csv", index=False)
    spots = pd.DataFrame(spot_rows)
    print(f"[inventory] wrote {OUT}/hla_variant_inventory.csv ({len(inv)} calls)")
    return inv, spots


def locus_summary(inv, spots):
    rows = []
    for sk in SAMPLES:
        sub = inv[inv["sample"] == sk]
        n_spots_total = int((spots["sample"] == sk).sum())
        for gene, grp in sub.groupby("gene"):
            rows.append(dict(sample=sk, gene=gene, hla_class=CLASS_OF[gene],
                             n_calls=len(grp), n_cosmic_hits=int(grp.cosmic_hit.sum()),
                             max_pct_spots=grp.pct_spots.max(),
                             median_pct_spots=grp.pct_spots.median(),
                             n_spots_total=n_spots_total))
    df = pd.DataFrame(rows).sort_values(["sample", "n_calls"], ascending=[True, False])
    df.to_csv(f"{OUT}/hla_locus_summary.csv", index=False)
    print(f"[locus] wrote {OUT}/hla_locus_summary.csv")
    return df


# ---------------------------------------------------------------------------
# 3. Spatial join: coordinates + CalicoST clone / tumour proportion
# ---------------------------------------------------------------------------
def spatial_join(spots):
    out = []
    for sk in SAMPLES:
        sub = spots[spots["sample"] == sk].copy()
        pos = pd.read_csv(POSITIONS[sk])
        sub = sub.merge(pos[["barcode", "array_row", "array_col",
                             "pxl_row_in_fullres", "pxl_col_in_fullres"]],
                        on="barcode", how="left", validate="one_to_one")
        if sub.pxl_row_in_fullres.isna().any():
            raise ValueError(f"{sk}: matrix barcodes missing from tissue_positions")
        cl_path = f"{CALICO}/{CALICO_RUN[sk]}/{CALICO_SUB}/clone_labels.tsv"
        if os.path.exists(cl_path):
            cl = pd.read_csv(cl_path, sep="\t")
            cl["barcode"] = cl.BARCODES.str.rsplit("_", n=1).str[0]
            sub = sub.merge(cl[["barcode", "clone_label", "tumor_proportion"]],
                            on="barcode", how="left")
        else:
            sub["clone_label"] = np.nan
            sub["tumor_proportion"] = np.nan
        out.append(sub)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(f"{OUT}/hla_spot_burden.csv", index=False)
    print(f"[spatial] wrote {OUT}/hla_spot_burden.csv ({len(df)} spots)")
    return df


# ---------------------------------------------------------------------------
# 4. CalicoST copy number / LOH at the HLA loci
# ---------------------------------------------------------------------------
def cnv_status():
    rows = []
    for sk in SAMPLES:
        path = f"{CALICO}/{CALICO_RUN[sk]}/{CALICO_SUB}/cnv_genelevel.tsv"
        if not os.path.exists(path):
            print(f"[cnv] {sk}: no CalicoST gene-level output")
            continue
        gl = pd.read_csv(path, sep="\t").set_index("gene")
        clones = sorted({c.rsplit(" ", 1)[0] for c in gl.columns})
        for gene in CLASS_OF:
            if gene not in gl.index:
                continue
            for clone in clones:
                a, b = gl.loc[gene, f"{clone} A"], gl.loc[gene, f"{clone} B"]
                rows.append(dict(sample=sk, clone=clone, gene=gene,
                                 hla_class=CLASS_OF[gene], cn_A=int(a), cn_B=int(b),
                                 total_cn=int(a) + int(b),
                                 loh=(int(a) == 0 or int(b) == 0),
                                 balanced=(int(a) == int(b))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/hla_cnv_status.csv", index=False)
    n_loh = int(df.loh.sum()) if len(df) else 0
    print(f"[cnv] wrote {OUT}/hla_cnv_status.csv "
          f"({len(df)} sample x clone x gene rows, {n_loh} with LOH)")
    return df


def genome_loh_context():
    """How much of each sample's genome IS in LOH, so the HLA result has a scale."""
    rows = []
    for sk in SAMPLES:
        path = f"{CALICO}/{CALICO_RUN[sk]}/{CALICO_SUB}/cnv_genelevel.tsv"
        if not os.path.exists(path):
            continue
        gl = pd.read_csv(path, sep="\t").set_index("gene")
        clones = sorted({c.rsplit(" ", 1)[0] for c in gl.columns})
        for clone in clones:
            a = gl[f"{clone} A"].astype(int)
            b = gl[f"{clone} B"].astype(int)
            rows.append(dict(sample=sk, clone=clone, n_genes=len(gl),
                             n_loh=int(((a == 0) | (b == 0)).sum()),
                             pct_loh=100 * ((a == 0) | (b == 0)).mean(),
                             n_imbalanced=int((a != b).sum()),
                             pct_imbalanced=100 * (a != b).mean()))
    ctx = pd.DataFrame(rows)

    # Local context: the CalicoST SEGMENTS that cover the xMHC. The HLA genes sit
    # in one small window on chr6p21, so "19 balanced genes" is not 19 independent
    # observations -- the honest unit is the segment.
    seg_rows = []
    for sk in SAMPLES:
        path = f"{CALICO}/{CALICO_RUN[sk]}/{CALICO_SUB}/cnv_seglevel.tsv"
        if not os.path.exists(path):
            continue
        seg = pd.read_csv(path, sep="\t")
        seg["CHR"] = seg.CHR.astype(str).str.replace("chr", "", regex=False)
        chr6 = seg[seg.CHR == "6"]
        xmhc = chr6[(chr6.END >= MHC_LO) & (chr6.START <= MHC_HI)]
        clones = sorted({c.rsplit(" ", 1)[0] for c in seg.columns if c.endswith((" A", " B"))})
        for clone in clones:
            a6, b6 = chr6[f"{clone} A"].astype(int), chr6[f"{clone} B"].astype(int)
            ax, bx = xmhc[f"{clone} A"].astype(int), xmhc[f"{clone} B"].astype(int)
            seg_rows.append(dict(
                sample=sk, clone=clone,
                n_chr6_segments=len(chr6),
                pct_chr6_segments_loh=100 * ((a6 == 0) | (b6 == 0)).mean() if len(chr6) else np.nan,
                n_xmhc_segments=len(xmhc),
                n_xmhc_segments_loh=int(((ax == 0) | (bx == 0)).sum()),
                n_xmhc_segments_imbalanced=int((ax != bx).sum())))
    seg_ctx = pd.DataFrame(seg_rows)
    ctx = ctx.merge(seg_ctx, on=["sample", "clone"], how="left")
    ctx.to_csv(f"{OUT}/hla_cnv_context.csv", index=False)
    print(f"[cnv] wrote {OUT}/hla_cnv_context.csv")
    return ctx


# ---------------------------------------------------------------------------
# 5. Burden vs tumour proportion, raw and coverage-conditional
# ---------------------------------------------------------------------------
def purity_association(spatial):
    rows = []
    for sk in SAMPLES:
        sub = spatial[(spatial["sample"] == sk) & spatial.tumor_proportion.notna()]
        if len(sub) < 30:
            print(f"[assoc] {sk}: only {len(sub)} spots with a tumour proportion, skipped")
            continue
        # Condition on the NON-HLA somatic burden. Conditioning on the TOTAL would
        # be conditioning on a sum that contains the variable of interest, which
        # induces a spurious negative association on its own.
        non_hla = (sub.n_somatic_total - sub.n_hla_total).values.astype(float)
        for label, col in [("class I (classical)", "n_class_I"),
                           ("class II (DP/DQ/DR)", "n_class_II"),
                           ("all HLA", "n_hla_total"),
                           ("all SPARCAL somatic", "n_somatic_total")]:
            x, y = sub[col].values.astype(float), sub.tumor_proportion.values
            if np.ptp(x) == 0:
                rows.append(dict(sample=sk, measure=label, n_spots=len(sub),
                                 spearman_rho=np.nan, spearman_p=np.nan,
                                 partial_rho_given_nonhla_burden=np.nan, partial_p=np.nan,
                                 mean_burden=float(np.mean(x)),
                                 note="no variation (0 calls in every spot)"))
                continue
            rho, p = stats.spearmanr(x, y)
            if col == "n_somatic_total":
                prho, pp = np.nan, np.nan
            else:
                # partial Spearman: rank-residualise both on the non-HLA burden
                z = stats.rankdata(non_hla)
                rx = stats.rankdata(x) - np.polyval(np.polyfit(z, stats.rankdata(x), 1), z)
                ry = stats.rankdata(y) - np.polyval(np.polyfit(z, stats.rankdata(y), 1), z)
                prho, pp = stats.pearsonr(rx, ry)
            rows.append(dict(sample=sk, measure=label, n_spots=len(sub),
                             spearman_rho=rho, spearman_p=p,
                             partial_rho_given_nonhla_burden=prho, partial_p=pp,
                             mean_burden=float(np.mean(x)), note=""))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/hla_purity_association.csv", index=False)
    print(f"[assoc] wrote {OUT}/hla_purity_association.csv")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_axisbelow(True)


def fig_locus_inventory(loci):
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.4))
    gene_order = CLASS_I_CLASSICAL + CLASS_I_NONCLASSICAL + CLASS_II_CLASSICAL + CLASS_II_NONCLASSICAL
    for ax, sk in zip(axes, SAMPLES):
        sub = loci[loci["sample"] == sk].set_index("gene").reindex(gene_order).fillna(0)
        y = np.arange(len(gene_order))[::-1]
        colors = [CLASS_COLOR[CLASS_OF[g]] for g in gene_order]
        ax.barh(y, sub.n_calls, color=colors, edgecolor="white", linewidth=0.5,
                height=0.74, zorder=3)
        for yy, n, nc in zip(y, sub.n_calls, sub.n_cosmic_hits):
            if n > 0:
                ax.text(n * 1.03, yy, f"{int(n)}" + (f"  ({int(nc)} in COSMIC)" if nc else ""),
                        va="center", ha="left", fontsize=7.0, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(gene_order, fontsize=7.6, fontstyle="italic")
        for lbl, g in zip(ax.get_yticklabels(), gene_order):
            lbl.set_color(CLASS_COLOR[CLASS_OF[g]])
        ax.set_xlim(0, max(sub.n_calls.max() * 1.42, 2))
        ax.tick_params(axis="x", labelsize=7.2)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{sk}  ({int(sub.n_calls.sum())} calls in HLA genes)",
                     fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlabel("SPARCAL somatic calls", fontsize=8.2)
    axes[0].legend(handles=[Patch(facecolor=CLASS_COLOR[c], edgecolor="white", label=c)
                            for c in CLASS_ORDER],
                   loc="lower right", fontsize=7.0, frameon=False, handlelength=1.2)
    fig.suptitle("SPARCAL somatic calls per HLA locus, by sample and HLA class",
                 fontsize=12, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_hla_locus_inventory.{ext}", dpi=250)
    print(f"[fig] wrote {OUT}/fig_hla_locus_inventory.{{png,pdf}}")


def fig_spatial(spatial):
    panels = [("n_class_I", "HLA class I (A/B/C)\nmutant calls per spot", "Reds"),
              ("n_class_II", "HLA class II (DP/DQ/DR)\nmutant calls per spot", "Blues"),
              ("tumor_proportion", "CalicoST tumour proportion", "viridis")]
    fig, axes = plt.subplots(len(panels), 4, figsize=(14.4, 10.2))
    for j, sk in enumerate(SAMPLES):
        sub = spatial[spatial["sample"] == sk]
        x = sub.pxl_col_in_fullres.values
        y = -sub.pxl_row_in_fullres.values
        for i, (col, title, cmap) in enumerate(panels):
            ax = axes[i, j]
            v = sub[col].values.astype(float)
            ok = ~np.isnan(v)
            ax.scatter(x[~ok], y[~ok], s=5.5, color="#eeeeea", edgecolor="none", zorder=2)
            top = np.nanmax(v[ok]) if ok.sum() else 0
            vmax = np.nanpercentile(v[ok], 99) if (ok.sum() and top > 0) else 1
            sc = ax.scatter(x[ok], y[ok], c=v[ok], s=5.5, cmap=cmap, edgecolor="none",
                            zorder=3, vmin=0, vmax=max(vmax, 1e-9))
            ax.set_aspect("equal")
            ax.axis("off")
            cb = plt.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
            cb.ax.tick_params(labelsize=6.5, colors=MUTED)
            cb.outline.set_visible(False)
            if col != "tumor_proportion" and ok.sum() and top == 0:
                ax.text(0.5, 0.5, "no calls in this class", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=MUTED, style="italic")
            if j == 0:
                ax.text(-0.10, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.6, color=INK,
                        fontweight="bold", linespacing=1.4)
            if col != "tumor_proportion":
                frac = 100 * (v[ok] > 0).mean() if ok.sum() else 0
                ax.text(0.5, -0.03, f"{frac:.1f}% of spots carry ≥1 call",
                        transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                        color=MUTED)
    fig.suptitle("Where the HLA-region SPARCAL somatic calls sit in the tissue",
                 fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=[0.015, 0, 1, 0.945])
    for j, sk in enumerate(SAMPLES):
        box = axes[0, j].get_position()
        fig.text(box.x0 + box.width / 2, 0.945, sk, fontsize=12, fontweight="bold",
                 color=INK, ha="center", va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_hla_spatial.{ext}", dpi=220)
    print(f"[fig] wrote {OUT}/fig_hla_spatial.{{png,pdf}}")


def fig_cnv(cnv, ctx):
    genes = [g for g in (CLASS_I_CLASSICAL + CLASS_I_NONCLASSICAL
                         + CLASS_II_CLASSICAL + CLASS_II_NONCLASSICAL)]
    keys = [(r.sample, r.clone) for r in cnv[["sample", "clone"]].drop_duplicates().itertuples()]
    fig, ax = plt.subplots(figsize=(12.0, 0.62 * len(keys) + 2.6))
    grid = np.full((len(keys), len(genes)), np.nan)
    text = np.empty((len(keys), len(genes)), dtype=object)
    for i, (sk, clone) in enumerate(keys):
        for j, g in enumerate(genes):
            r = cnv[(cnv["sample"] == sk) & (cnv.clone == clone) & (cnv.gene == g)]
            if r.empty:
                text[i, j] = ""
                continue
            r = r.iloc[0]
            grid[i, j] = 2 if r.loh else (1 if not r.balanced else 0)
            text[i, j] = f"{r.cn_A}|{r.cn_B}"
    cmap = matplotlib.colors.ListedColormap(["#eef2ee", "#f6d9a0", "#d1495b"])
    ax.imshow(np.nan_to_num(grid, nan=0), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(genes)):
            if text[i, j]:
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=7.6,
                        color=INK if grid[i, j] != 2 else "white",
                        fontweight="bold" if grid[i, j] == 2 else "normal")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7.6, color=MUTED)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8.0, fontstyle="italic")
    for lbl, g in zip(ax.get_xticklabels(), genes):
        lbl.set_color(CLASS_COLOR[CLASS_OF[g]])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{s}  {c}" for s, c in keys], fontsize=8.6)
    ax.set_xticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(facecolor="#eef2ee", edgecolor=GRID, label="balanced (A = B)"),
                       Patch(facecolor="#f6d9a0", edgecolor=GRID, label="allelic imbalance"),
                       Patch(facecolor="#d1495b", edgecolor=GRID, label="LOH (one allele lost)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False,
              fontsize=8.0)
    ctx_txt = "\n".join(
        f"{r.sample} {r.clone}: {r.pct_loh:.0f}% of genes in LOH genome-wide, "
        f"{r.pct_chr6_segments_loh:.0f}% of chr6 segments in LOH, "
        f"{r.n_xmhc_segments_loh}/{int(r.n_xmhc_segments)} xMHC segments in LOH"
        for r in ctx.itertuples())
    ax.set_title("CalicoST allele-specific copy number at the HLA loci "
                 "(cell text = A|B copies)\n" + ctx_txt,
                 fontsize=10.0, fontweight="bold", color=INK, pad=12, linespacing=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_hla_cnv.{ext}", dpi=250, bbox_inches="tight")
    print(f"[fig] wrote {OUT}/fig_hla_cnv.{{png,pdf}}")


def main():
    fp = hla_footprints()
    inv, spots = build_inventory(fp)
    loci = locus_summary(inv, spots)
    spatial = spatial_join(spots)
    cnv = cnv_status()
    ctx = genome_loh_context()
    assoc = purity_association(spatial)

    fig_locus_inventory(loci)
    fig_spatial(spatial)
    if len(cnv):
        fig_cnv(cnv, ctx)

    print("\n================ HEADLINE NUMBERS ================")
    print(inv.groupby(["sample", "hla_class"]).size().unstack(fill_value=0).to_string())
    print("\nLOH at HLA loci:", int(cnv.loh.sum()), "of", len(cnv), "sample x clone x gene rows")
    print("\nGenome-wide LOH context:\n", ctx.to_string(index=False))
    print("\nBurden vs tumour proportion:\n", assoc.to_string(index=False))


if __name__ == "__main__":
    main()
