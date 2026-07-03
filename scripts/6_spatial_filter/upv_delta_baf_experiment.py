#!/usr/bin/env python3
"""
upv_delta_baf_experiment.py  —  EXPERIMENT (dcis1)

Test whether a ΔBAF (tumor-spot VAF − normal-spot VAF) separates the UPV set
better than PURITY_CORR. ΔBAF uses read-level VAF magnitude (not binary presence),
so it survives the spatial-ubiquity saturation that kills PURITY_CORR.

Per-spot read counts are NOT in any VCF (vcf_by_spot encodes presence/absence
only; its INFO is the merged pseudobulk value). So we scan the CB-tagged
possorted_genome_bam.bam directly: at each UPV position, pool reads by their CB
(cell-barcode) tag into TUMOR vs NORMAL spot groups (CalicoST tumor_proportion
tertiles — continuous, avoids the noisy discrete clone labels), then

  ΔBAF(variant) = altreads/(ref+alt) over TUMOR-spot reads
                − altreads/(ref+alt) over NORMAL-spot reads      (pooled, depth-weighted)

Outputs (alongside the gmm_subfilter dir):
  upv_delta_baf.tsv     per-variant: baf, delta_baf, t_alt/t_dp, n_alt/n_dp, current_class
  upv_delta_baf.png     BAF×ΔBAF scatter + ΔBAF distributions

Usage:
  python upv_delta_baf_experiment.py --dataset DCIS --section_id dcis1 \
      --quality_filter baseQ0mapQ0 \
      --bam /lfs/.../DCIS1/.../outs/possorted_genome_bam.bam \
      --clone_labels /data/maiziezhou_lab/leiy4/CalicoST/DCIS1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv
"""
import argparse, gzip, os, sys
import numpy as np
import pysam

PROJECT_ROOT = "/data/maiziezhou_lab/leiy4/snv_calling"
BASE = {"DCIS": os.path.join(PROJECT_ROOT, "data"),
        "P4_TUMOR": os.path.join(PROJECT_ROOT, "data", "P4_tumor"),
        "P6_TUMOR": os.path.join(PROJECT_ROOT, "data", "P6_tumor")}


def vkey(c, p, r, a):
    return (c.replace("chr", ""), p, r, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--section_id", required=True)
    ap.add_argument("--quality_filter", default="baseQ0mapQ0")
    ap.add_argument("--bam", required=True, help="CB-tagged possorted_genome_bam.bam")
    ap.add_argument("--clone_labels", required=True,
                    help="CalicoST clone_labels.tsv (cols: BARCODES, clone_label, tumor_proportion)")
    ap.add_argument("--min-group-dp", type=int, default=5,
                    help="min pooled ref+alt reads in EACH of tumor/normal to keep a variant (default 5)")
    ap.add_argument("--max-depth", type=int, default=200000)
    args = ap.parse_args()

    sec = os.path.join(BASE[args.dataset], args.section_id)
    qf = args.quality_filter
    upv_vcf = os.path.join(sec, "spatial_filter_purity", qf, "germline", "denovo",
                           "germline_denovo.vcf.gz")
    gmm_tsv = os.path.join(sec, "spatial_filter_purity", qf, "germline", "denovo",
                           "gmm_subfilter", "upv_gmm_classification.tsv")
    out_dir = os.path.dirname(gmm_tsv)
    os.makedirs(out_dir, exist_ok=True)

    # ── UPV variants: chrom(bare) -> {pos1: [(ref,alt), ...]} ── + 7c BAF/class ──
    upv_at = {}
    baf = {}; cls = {}
    with gzip.open(upv_vcf, "rt") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            c, p, _i, r, a = ln.split("\t")[:5]
            cb = c.replace("chr", "")
            upv_at.setdefault(cb, {}).setdefault(int(p), []).append((r, a))
    n_upv = sum(len(v) for d in upv_at.values() for v in d.values())
    print(f"UPV variants: {n_upv}")
    if os.path.exists(gmm_tsv):
        for l in open(gmm_tsv).read().splitlines()[1:]:
            x = l.split("\t"); c, p, r, a = x[0].split("_")
            baf[(c, p, r, a)] = float(x[1]); cls[(c, p, r, a)] = x[6]

    # ── tumor/normal spot groups by tumor_proportion tertiles ──
    tp = {}
    with open(args.clone_labels) as f:
        next(f)
        for ln in f:
            x = ln.rstrip().split("\t")
            if len(x) < 3:
                continue
            try:
                tp[x[0].rsplit("_", 1)[0]] = float(x[2])
            except ValueError:
                pass
    vals = np.array(list(tp.values()))
    lo, hi = np.quantile(vals, [1/3, 2/3])
    group = {bc: ("normal" if v <= lo else "tumor" if v >= hi else "mid")
             for bc, v in tp.items()}
    print(f"purity tertiles: normal<= {lo:.3f} (n={sum(g=='normal' for g in group.values())}), "
          f"tumor>= {hi:.3f} (n={sum(g=='tumor' for g in group.values())})")

    # ── BAM pileup: pool ref/alt reads per variant per group via CB tag ──
    cnt = {}  # key -> [t_ref, t_alt, n_ref, n_alt]
    bam = pysam.AlignmentFile(args.bam, "rb")
    bam_contigs = set(bam.references)
    done = 0
    for cb_chrom, posmap in upv_at.items():
        bam_chrom = cb_chrom if cb_chrom in bam_contigs else ("chr" + cb_chrom)
        if bam_chrom not in bam_contigs:
            continue
        for pos1 in sorted(posmap):
            variants = posmap[pos1]
            for pcol in bam.pileup(bam_chrom, pos1 - 1, pos1, truncate=True,
                                   max_depth=args.max_depth, min_base_quality=0):
                for pr in pcol.pileups:
                    if pr.is_del or pr.is_refskip or pr.query_position is None:
                        continue
                    aln = pr.alignment
                    if not aln.has_tag("CB"):
                        continue
                    g = group.get(aln.get_tag("CB"))
                    if g not in ("tumor", "normal"):
                        continue
                    base = aln.query_sequence[pr.query_position]
                    for (ref, alt) in variants:
                        k = (cb_chrom, str(pos1), ref, alt)
                        c = cnt.setdefault(k, [0, 0, 0, 0])
                        if base == alt:
                            c[1 if g == "tumor" else 3] += 1
                        elif base == ref:
                            c[0 if g == "tumor" else 2] += 1
            done += 1
            if done % 500 == 0:
                print(f"  scanned {done}/{n_upv} positions")

    # ── ΔBAF per variant (pooled, depth-weighted) ──
    rows = []
    for k, (tr, ta, nr, na) in cnt.items():
        tdp, ndp = tr + ta, nr + na
        if tdp >= args.min_group_dp and ndp >= args.min_group_dp:
            dbaf = ta / tdp - na / ndp
            rows.append((k, baf.get(k, float("nan")), dbaf, ta, tdp, na, ndp,
                         cls.get(k, "?")))
    print(f"variants with ΔBAF (>= {args.min_group_dp} reads each side): {len(rows)}")
    if not rows:
        sys.exit("No variants had enough pooled reads in both groups.")

    with open(os.path.join(out_dir, "upv_delta_baf.tsv"), "w") as f:
        f.write("chrom_pos_ref_alt\tbaf\tdelta_baf\tt_alt\tt_dp\tn_alt\tn_dp\t"
                "vaf_tumor\tvaf_normal\tcurrent_class\n")
        for k, b, db, ta, tdp, na, ndp, cc in rows:
            f.write(f"{'_'.join(k)}\t{b:.4f}\t{db:.4f}\t{ta}\t{tdp}\t{na}\t{ndp}\t"
                    f"{ta/tdp:.4f}\t{na/ndp:.4f}\t{cc}\n")

    # ── report + plot ──
    dbaf = np.array([r[2] for r in rows])
    bafs = np.array([r[1] for r in rows])
    issom = np.array([r[7] == "somatic_candidate" for r in rows])
    print(f"\nΔBAF quantiles [5,25,50,75,95]: {np.round(np.quantile(dbaf,[.05,.25,.5,.75,.95]),3)}")
    print(f"ΔBAF mean | current somatic_candidate: {dbaf[issom].mean():+.3f}  "
          f"germline_like: {dbaf[~issom].mean():+.3f}")
    print(f"ΔBAF > 0.05 (tumor-enriched): {(dbaf>0.05).sum()}  "
          f"| of current somatic set: {(dbaf[issom]>0.05).sum()}/{issom.sum()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.scatter(bafs[~issom], dbaf[~issom], s=7, c="purple", alpha=0.4, label="germline-like (7c)")
    a1.scatter(bafs[issom], dbaf[issom], s=7, c="red", alpha=0.5, label="somatic-cand (7c)")
    a1.axhline(0, c="grey", lw=1); a1.axhline(0.05, ls="--", c="black", lw=1)
    a1.set_xlabel("BAF (pseudobulk)"); a1.set_ylabel("ΔBAF (tumor − normal VAF)")
    a1.set_title(f"UPV  BAF × ΔBAF — {args.dataset} {args.section_id}"); a1.legend()
    bins = np.linspace(min(-0.3, dbaf.min()), max(0.3, dbaf.max()), 41)
    a2.hist(dbaf[~issom], bins=bins, color="purple", alpha=0.6, label="germline-like (7c)")
    a2.hist(dbaf[issom], bins=bins, color="red", alpha=0.6, label="somatic-cand (7c)")
    a2.axvline(0, c="grey", lw=1); a2.axvline(0.05, ls="--", c="black", lw=1)
    a2.set_xlabel("ΔBAF (tumor − normal VAF)"); a2.set_ylabel("UPV variants")
    a2.set_title("ΔBAF distribution"); a2.legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "upv_delta_baf.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"\noutputs -> {out_dir}/upv_delta_baf.{{tsv,png}}")


if __name__ == "__main__":
    main()
