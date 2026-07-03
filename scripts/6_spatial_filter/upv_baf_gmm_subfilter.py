#!/usr/bin/env python3
"""
upv_baf_gmm_subfilter.py  —  Pipeline step 7c (post-processing)

Refine the UPV set (Ubiquitous Private Variants; code token `germline_denovo`)
into two sub-classes using a 2-D Gaussian mixture on [BAF, PURITY_CORR]:

    UPV-germline-like      (germline het/hom, incl. ASE-skewed het)
    UPV-somatic-candidate  (low BAF AND tumor-purity-associated)

Why 2-D: in spatial *transcriptomics* BAF comes from RNA reads, so allele-
specific expression (ASE) pulls true germline-het BAF below 0.5 and overlaps the
somatic low-VAF range. PURITY_CORR (presence-vs-tumor-purity correlation) is an
orthogonal axis: somatic ⇒ PURITY_CORR > 0, germline (incl. ASE) ⇒ ≈ 0. A
component is only called somatic if its centroid has low BAF *and* positive
PURITY_CORR — that is the step that defeats the ASE confounder.

Inputs (both precomputed by the pipeline — no BAM rescan):
  UPV VCF : data/{section}/spatial_filter_purity/{qf}/germline/denovo/germline_denovo.vcf.gz
            (carries INFO PURITY_CORR, AF, NS, RACE)
  BAF VCF : data/{section}/output_VCFs/mpileup_multi_bam/{qf}/merged_sorted_gt.vcf.gz
            (carries FORMAT BAF computed from I16; single pseudobulk sample)

Outputs: data/{section}/spatial_filter_purity/{qf}/germline/denovo/gmm_subfilter/
  upv_gmm_classification.tsv        per-variant: baf, dp, purity_corr, af, prob, class
  upv_germline_like.vcf[.gz]        UPV records called germline-like
  upv_somatic_candidate.vcf[.gz]    UPV records called somatic-candidate
  upv_baf_gmm.png                   BAF histogram + [BAF×PURITY_CORR] scatter, by class
  upv_gmm_summary.txt               component means/weights + counts + params

Usage:
  python upv_baf_gmm_subfilter.py --dataset DCIS --section_id dcis1 --quality_filter baseQ0mapQ0
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/data/maiziezhou_lab/leiy4/snv_calling"
APPS = os.path.join(PROJECT_ROOT, "apps")
DATASET_CONFIGS = {
    "DCIS":     {"base_path": os.path.join(PROJECT_ROOT, "data")},
    "P4_TUMOR": {"base_path": os.path.join(PROJECT_ROOT, "data", "P4_tumor")},
    "P6_TUMOR": {"base_path": os.path.join(PROJECT_ROOT, "data", "P6_tumor")},
    "DLPFC":    {"base_path": os.path.join(PROJECT_ROOT, "data", "dlpfc")},
}


def vkey(chrom, pos, ref, alt):
    """Contig-agnostic key (UPV VCF is chr-prefixed, merged VCF is bare)."""
    return (chrom.replace("chr", ""), pos, ref, alt)


def read_upv(upv_vcf):
    """Return (header_lines, records). records: list of dicts with raw line + parsed INFO."""
    header, records = [], []
    with gzip.open(upv_vcf, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                header.append(ln.rstrip("\n"))
                continue
            x = ln.rstrip("\n").split("\t")
            chrom, pos, _id, ref, alt = x[:5]
            info = dict(
                kv.split("=", 1) if "=" in kv else (kv, "")
                for kv in x[7].split(";")
            )
            records.append({
                "line": ln.rstrip("\n"),
                "key": vkey(chrom, pos, ref, alt),
                "purity_corr": float(info.get("PURITY_CORR", "nan")),
                "af": float(info.get("AF", "nan")),
                "ns": int(info["NS"]) if "NS" in info and info["NS"] else 0,
            })
    return header, records


def lookup_baf(merged_vcf, keys):
    """Stream merged single-sample VCF; return {key: (baf, dp)} for requested keys.

    BAF is RECOMPUTED as alt/(ref+alt) from the I16 base counts
    (I16[0:4] = ref_fwd, ref_rev, alt_fwd, alt_rev). The FORMAT `BAF` field in
    merged_sorted_gt is unreliable — it is mis-stored as 0 for homozygous-ALT
    sites (e.g. chr1:632560: GT=1/1, true alt-frac 0.999, stored BAF=0). Using
    I16 gives the correct alt-allele fraction: hom-ALT≈1.0, het≈0.5, somatic<0.5.
    Falls back to the FORMAT BAF field only if I16 is absent.
    """
    want = set(keys)
    out = {}
    with gzip.open(merged_vcf, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            x = ln.rstrip("\n").split("\t")
            if len(x) < 10:
                continue
            k = vkey(x[0], x[1], x[3], x[4])
            if k not in want:
                continue
            info = {}
            for fld in x[7].split(";"):
                if "=" in fld:
                    a, b = fld.split("=", 1)
                    info[a] = b
            baf = float("nan")
            if "I16" in info:
                try:
                    i16 = [float(v) for v in info["I16"].split(",")]
                    ref = i16[0] + i16[1]
                    alt = i16[2] + i16[3]
                    if ref + alt > 0:
                        baf = alt / (ref + alt)
                except (ValueError, IndexError):
                    baf = float("nan")
            if not np.isfinite(baf):  # fallback to FORMAT BAF field
                d = dict(zip(x[8].split(":"), x[9].split(":")))
                try:
                    baf = float(d.get("BAF", "nan"))
                except ValueError:
                    baf = float("nan")
            try:
                dp = int(info.get("DP", "0"))
            except ValueError:
                dp = 0
            out[k] = (baf, dp)
    return out


def bgzip_tabix(vcf_path):
    """bgzip+tabix using bundled apps/ binaries; no-op (leave plain) if unavailable."""
    bgzip = os.path.join(APPS, "bgzip")
    tabix = os.path.join(APPS, "tabix")
    if not (os.path.exists(bgzip) and os.path.exists(tabix)):
        return vcf_path
    try:
        subprocess.run([bgzip, "-f", vcf_path], check=True)
        subprocess.run([tabix, "-p", "vcf", vcf_path + ".gz"], check=True)
        return vcf_path + ".gz"
    except subprocess.CalledProcessError:
        return vcf_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument("--section_id", required=True,
                    help="e.g. dcis1 / dcis2 (DCIS) or 1 (P4/P6)")
    ap.add_argument("--quality_filter", default="baseQ0mapQ0")
    ap.add_argument("--min-dp", type=int, default=5,
                    help="min read depth for a variant to enter the GMM fit (default 5)")
    ap.add_argument("--n-components", type=int, default=3,
                    help="GMM components (default 3: somatic-low / het / hom)")
    ap.add_argument("--baf-cut", type=float, default=0.40,
                    help="a component is somatic-like only if mean BAF < this (default 0.40)")
    ap.add_argument("--pur-cut", type=float, default=0.10,
                    help="...AND mean PURITY_CORR > this (default 0.10)")
    ap.add_argument("--somatic-baf-max", type=float, default=0.35,
                    help="HARD per-variant ceiling: a variant can only be called "
                         "somatic if its own BAF < this (default 0.35). Excludes the "
                         "germline-het mode at BAF~0.5, which the soft GMM otherwise "
                         "spills into. Set >=1 to disable the hard gate.")
    ap.add_argument("--prob-threshold", type=float, default=0.5,
                    help="posterior over somatic-like comps to call somatic (default 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = DATASET_CONFIGS[args.dataset]["base_path"]
    sec_dir = os.path.join(base, args.section_id)
    qf = args.quality_filter
    upv_vcf = os.path.join(sec_dir, "spatial_filter_purity", qf,
                           "germline", "denovo", "germline_denovo.vcf.gz")
    merged_vcf = os.path.join(sec_dir, "output_VCFs", "mpileup_multi_bam", qf,
                              "merged_sorted_gt.vcf.gz")
    out_dir = os.path.join(sec_dir, "spatial_filter_purity", qf,
                           "germline", "denovo", "gmm_subfilter")
    os.makedirs(out_dir, exist_ok=True)

    for p in (upv_vcf, merged_vcf):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing input: {p}")

    print(f"[7c] {args.dataset}/{args.section_id} ({qf})")
    header, records = read_upv(upv_vcf)
    print(f"  UPV variants: {len(records)}")
    baf_map = lookup_baf(merged_vcf, [r["key"] for r in records])

    # attach BAF/DP; keep variants with finite BAF + finite PURITY_CORR
    feats = []
    for r in records:
        baf, dp = baf_map.get(r["key"], (float("nan"), 0))
        r["baf"], r["dp"] = baf, dp
        if np.isfinite(baf) and np.isfinite(r["purity_corr"]):
            feats.append(r)
    X = np.array([[r["baf"], r["purity_corr"]] for r in feats])
    DP = np.array([r["dp"] for r in feats])
    print(f"  with BAF+PURITY_CORR: {len(feats)} ({100*len(feats)/max(1,len(records)):.1f}%)")

    # ── fit 2-D GMM on depth-supported sites, score all ──
    from sklearn.mixture import GaussianMixture
    fit_mask = DP >= args.min_dp
    if fit_mask.sum() < args.n_components:
        sys.exit(f"ERROR: only {fit_mask.sum()} sites with DP>={args.min_dp}; "
                 f"need >= n_components ({args.n_components}).")
    gmm = GaussianMixture(n_components=args.n_components, covariance_type="full",
                          n_init=5, random_state=args.seed).fit(X[fit_mask])
    means = gmm.means_                       # (K, 2): [BAF, PURITY_CORR]
    weights = gmm.weights_
    post = gmm.predict_proba(X)              # (N, K) for ALL variants

    # ── label somatic-like components by centroid rule (defeats ASE) ──
    somatic_comps = [k for k in range(args.n_components)
                     if means[k, 0] < args.baf_cut and means[k, 1] > args.pur_cut]
    somatic_prob = post[:, somatic_comps].sum(axis=1) if somatic_comps \
        else np.zeros(len(feats))
    # HARD per-variant BAF ceiling: the soft GMM component spills past 0.5 and
    # engulfs the germline-het mode; require each variant's own BAF below the cap.
    baf_arr = X[:, 0]
    is_somatic = (somatic_prob > args.prob_threshold) & (baf_arr < args.somatic_baf_max)
    n_gmm_only = int((somatic_prob > args.prob_threshold).sum())

    n_som = int(is_somatic.sum())
    print(f"  GMM components (sorted by BAF):")
    for k in np.argsort(means[:, 0]):
        tag = " <-- somatic-like" if k in somatic_comps else ""
        print(f"    comp{k}: BAF={means[k,0]:.3f} PURITY_CORR={means[k,1]:.3f} "
              f"w={weights[k]:.3f}{tag}")
    print(f"  GMM somatic-like: {n_gmm_only}  ->  after hard BAF<{args.somatic_baf_max} gate: {n_som}")
    print(f"  -> UPV-somatic-candidate: {n_som} ; UPV-germline-like: {len(feats)-n_som}")

    # ── write classification TSV ──
    tsv = os.path.join(out_dir, "upv_gmm_classification.tsv")
    with open(tsv, "w") as f:
        f.write("chrom_pos_ref_alt\tbaf\tdp\tpurity_corr\taf\tsomatic_prob\tclass\n")
        for r, sp, som in zip(feats, somatic_prob, is_somatic):
            c, p, ref, alt = r["key"]
            cls = "somatic_candidate" if som else "germline_like"
            f.write(f"{c}_{p}_{ref}_{alt}\t{r['baf']:.4f}\t{r['dp']}\t"
                    f"{r['purity_corr']:.4f}\t{r['af']:.4f}\t{sp:.4f}\t{cls}\n")

    # ── write the two sub-class VCFs (UPV records + new INFO) ──
    extra_hdr = [
        '##INFO=<ID=BAF,Number=1,Type=Float,Description="Pseudobulk B-allele frequency (merged_sorted_gt)">',
        '##INFO=<ID=DP_BAF,Number=1,Type=Integer,Description="Read depth at BAF site">',
        '##INFO=<ID=UPV_SOMATIC_PROB,Number=1,Type=Float,Description="GMM posterior over somatic-like components">',
        '##INFO=<ID=UPV_CLASS,Number=1,Type=String,Description="UPV sub-class: germline_like|somatic_candidate">',
    ]
    col_line = next(h for h in header if h.startswith("#CHROM"))
    meta = [h for h in header if h.startswith("##")]

    def write_vcf(path, want_somatic):
        with open(path, "w") as f:
            f.write("\n".join(meta + extra_hdr + [col_line]) + "\n")
            for r, sp, som in zip(feats, somatic_prob, is_somatic):
                if bool(som) != want_somatic:
                    continue
                x = r["line"].split("\t")
                cls = "somatic_candidate" if som else "germline_like"
                x[7] = (x[7] + f";BAF={r['baf']:.4f};DP_BAF={r['dp']}"
                        f";UPV_SOMATIC_PROB={sp:.4f};UPV_CLASS={cls}")
                f.write("\t".join(x) + "\n")

    gl = os.path.join(out_dir, "upv_germline_like.vcf")
    sm = os.path.join(out_dir, "upv_somatic_candidate.vcf")
    write_vcf(gl, want_somatic=False)
    write_vcf(sm, want_somatic=True)
    gl_out, sm_out = bgzip_tabix(gl), bgzip_tabix(sm)

    # ── plot: BAF histogram by class + [BAF × PURITY_CORR] scatter ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    baf = X[:, 0]
    pur = X[:, 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 41)
    ax1.hist(baf[~is_somatic], bins=bins, color="purple", alpha=0.6,
             label=f"germline-like (n={(~is_somatic).sum()})")
    ax1.hist(baf[is_somatic], bins=bins, color="red", alpha=0.6,
             label=f"somatic-candidate (n={n_som})")
    ax1.axvline(args.somatic_baf_max, ls="--", c="black", lw=1.2,
                label=f"hard somatic cap (BAF<{args.somatic_baf_max})")
    ax1.set_xlabel("BAF"); ax1.set_ylabel("UPV variants"); ax1.legend()
    ax1.set_title(f"UPV BAF distribution — {args.dataset} {args.section_id}")

    ax2.scatter(baf[~is_somatic], pur[~is_somatic], s=6, c="purple", alpha=0.4,
                label="germline-like")
    ax2.scatter(baf[is_somatic], pur[is_somatic], s=6, c="red", alpha=0.5,
                label="somatic-candidate")
    ax2.scatter(means[:, 0], means[:, 1], marker="X", s=160, c="black",
                edgecolor="white", linewidth=1.5, label="GMM centroids", zorder=5)
    ax2.axvline(args.somatic_baf_max, ls="--", c="black", lw=1.2)
    ax2.axhline(args.pur_cut, ls="--", c="grey", lw=1)
    ax2.set_xlabel("BAF"); ax2.set_ylabel("PURITY_CORR")
    ax2.set_title("2-D GMM feature space"); ax2.legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "upv_baf_gmm.png")
    fig.savefig(png, dpi=150); plt.close(fig)

    # ── summary ──
    with open(os.path.join(out_dir, "upv_gmm_summary.txt"), "w") as f:
        f.write(f"UPV BAF-GMM sub-filter (step 7c)\n")
        f.write(f"dataset={args.dataset} section={args.section_id} qf={qf}\n")
        f.write(f"params: min_dp={args.min_dp} n_components={args.n_components} "
                f"baf_cut={args.baf_cut} pur_cut={args.pur_cut} "
                f"somatic_baf_max={args.somatic_baf_max} "
                f"prob_threshold={args.prob_threshold}\n")
        f.write(f"somatic = (GMM somatic-like posterior>{args.prob_threshold}) "
                f"AND (BAF < {args.somatic_baf_max} hard gate)\n\n")
        f.write(f"UPV total: {len(records)}\n")
        f.write(f"scored (finite BAF+PURITY_CORR): {len(feats)}\n")
        f.write(f"fit sites (DP>={args.min_dp}): {int(fit_mask.sum())}\n\n")
        f.write("GMM components [BAF, PURITY_CORR] (weight):\n")
        for k in np.argsort(means[:, 0]):
            tag = "  <-- somatic-like" if k in somatic_comps else ""
            f.write(f"  comp{k}: BAF={means[k,0]:.4f} PURITY_CORR={means[k,1]:.4f} "
                    f"w={weights[k]:.4f}{tag}\n")
        f.write(f"\nUPV-somatic-candidate: {n_som}\n")
        f.write(f"UPV-germline-like:     {len(feats)-n_som}\n")

    print(f"  outputs -> {out_dir}")
    print(f"    {os.path.basename(gl_out)}, {os.path.basename(sm_out)}, "
          f"upv_gmm_classification.tsv, upv_baf_gmm.png, upv_gmm_summary.txt")


if __name__ == "__main__":
    main()
