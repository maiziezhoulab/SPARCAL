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
import argparse, gzip, os, subprocess, sys, tempfile
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
    ap.add_argument("--min-base-quality", type=int, default=13)
    ap.add_argument("--min-mapping-quality", type=int, default=20)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: the production gmm_subfilter directory)")
    ap.add_argument("--baf-max", type=float, default=None,
                    help="restrict the direct BAM scan to UPVs at or below this pseudobulk BAF")
    args = ap.parse_args()

    sec = os.path.join(BASE[args.dataset], args.section_id)
    qf = args.quality_filter
    upv_vcf = os.path.join(sec, "spatial_filter_purity", qf, "germline", "denovo",
                           "germline_denovo.vcf.gz")
    gmm_tsv = os.path.join(sec, "spatial_filter_purity", qf, "germline", "denovo",
                           "gmm_subfilter", "upv_gmm_classification.tsv")
    out_dir = args.out_dir or os.path.dirname(gmm_tsv)
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
    if args.baf_max is not None:
        restricted = {}
        for chrom, posmap in upv_at.items():
            for pos1, alleles in posmap.items():
                kept = [
                    (ref, alt) for ref, alt in alleles
                    if baf.get((chrom, str(pos1), ref, alt), float("inf")) <= args.baf_max
                ]
                if kept:
                    restricted.setdefault(chrom, {})[pos1] = kept
        upv_at = restricted
        n_upv = sum(len(v) for d in upv_at.values() for v in d.values())
        print(f"restricted to pseudobulk BAF <= {args.baf_max}: {n_upv} variants")

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

    # ── BAM scan: pool ref/alt reads per variant per group via CB tag ──
    # Stream every alignment overlapping any target exactly once.  This avoids
    # thousands of indexed one-base pileup calls, which were prohibitively slow
    # on the 16-GB DCIS BAMs.
    cnt = {}  # key -> [t_ref, t_alt, n_ref, n_alt]
    bam = pysam.AlignmentFile(args.bam, "rb")
    bam_contigs = set(bam.references)
    target_map = {}
    bed_lines = []
    for cb_chrom, posmap in upv_at.items():
        bam_chrom = cb_chrom if cb_chrom in bam_contigs else ("chr" + cb_chrom)
        if bam_chrom not in bam_contigs:
            continue
        target_map[bam_chrom] = posmap
        bed_lines.extend(f"{bam_chrom}\t{pos1 - 1}\t{pos1}\n" for pos1 in sorted(posmap))
    bam.close()
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as bed:
        bed.writelines(bed_lines)
        bed_path = bed.name
    samtools = os.path.join(PROJECT_ROOT, "apps", "samtools")
    proc = subprocess.Popen(
        [samtools, "view", "-u", "-L", bed_path, "-q", str(args.min_mapping_quality), args.bam],
        stdout=subprocess.PIPE,
    )
    try:
        selected = pysam.AlignmentFile(proc.stdout, "rb")
        n_alignments = 0
        for aln in selected.fetch(until_eof=True):
            n_alignments += 1
            if not aln.has_tag("CB"):
                continue
            g = group.get(aln.get_tag("CB"))
            if g not in ("tumor", "normal"):
                continue
            posmap = target_map.get(aln.reference_name, {})
            sequence = aln.query_sequence
            qualities = aln.query_qualities
            for qpos, rpos in aln.get_aligned_pairs(matches_only=True):
                pos1 = rpos + 1
                variants = posmap.get(pos1)
                if not variants or qpos is None:
                    continue
                if qualities is not None and qualities[qpos] < args.min_base_quality:
                    continue
                base = sequence[qpos].upper()
                for ref, alt in variants:
                    k = (canon := aln.reference_name.removeprefix("chr"), str(pos1), ref, alt)
                    c = cnt.setdefault(k, [0, 0, 0, 0])
                    if base == alt:
                        c[1 if g == "tumor" else 3] += 1
                    elif base == ref:
                        c[0 if g == "tumor" else 2] += 1
            if n_alignments % 1000000 == 0:
                print(f"  streamed {n_alignments:,} target-overlapping alignments")
        selected.close()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    finally:
        os.unlink(bed_path)

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
            f.write(f"{'_'.join(k)}\t{b:.8f}\t{db:.8f}\t{ta}\t{tdp}\t{na}\t{ndp}\t"
                    f"{ta/tdp:.8f}\t{na/ndp:.8f}\t{cc}\n")
    with open(os.path.join(out_dir, "PARAMETERS.txt"), "w") as f:
        f.write(f"bam={args.bam}\nclone_labels={args.clone_labels}\n")
        f.write(f"min_group_dp={args.min_group_dp}\nmin_base_quality={args.min_base_quality}\n")
        f.write(f"min_mapping_quality={args.min_mapping_quality}\nmax_depth={args.max_depth}\n")
        f.write(f"baf_max={args.baf_max}\n")
        f.write(f"n_scanned_upv={n_upv}\n")

    # ── report + plot ──
    dbaf = np.array([r[2] for r in rows])
    bafs = np.array([r[1] for r in rows])
    issom = np.array([r[7] == "somatic_candidate" for r in rows])
    print(f"\nΔBAF quantiles [5,25,50,75,95]: {np.round(np.quantile(dbaf,[.05,.25,.5,.75,.95]),3)}")
    somatic_mean = dbaf[issom].mean() if issom.any() else float("nan")
    germline_mean = dbaf[~issom].mean() if (~issom).any() else float("nan")
    print(f"ΔBAF mean | current somatic_candidate: {somatic_mean:+.3f}  "
          f"germline_like: {germline_mean:+.3f}")
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
