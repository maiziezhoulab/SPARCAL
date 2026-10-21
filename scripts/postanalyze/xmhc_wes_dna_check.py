#!/usr/bin/env python
"""xMHC somatic-class calls vs matched tumour/normal WES DNA (P4, P6).

Companion to scripts/postanalyze/xmhc_artifact_mechanism.py.  That script asked
whether xMHC "somatic"-class calls look like reference-divergence artifacts from
READ evidence.  This one asks the orthogonal DNA question, using the matched
exomes that exist for P4 and P6 only (hg19, chr-prefixed):

  Q1  Does the exome capture kit even cover xMHC?  (If not, any DNA negative in
      the MHC is uninterpretable.)
  Q2  Do the WES matched tumour/normal somatic callsets contain ANY xMHC calls?
  Q3  Per SPARCAL somatic-class call, is the ALT allele present in the tumour
      exome, and -- decisively -- is it ALSO present in the patient's own NORMAL
      exome?  A genuine somatic variant must be absent from the matched normal;
      a germline / haplotype-divergence artifact will sit there at germline VAF.

Q3 is UNDERPOWERED by construction: only ~7-23% of RNA call positions are
covered at DP>=10 in both exomes, and of those only a few percent carry any ALT
read at all.  It is reported with Fisher tests so the reader can see that
directly.  Note also the standing caveat from
data/somatic_evidence_2026-07-28/SUMMARY.md: bulk WES misses spatially
restricted subclones, so DNA absence is NOT proof an RNA call is wrong.

DCIS1/DCIS2 have no matched exome and are therefore out of scope here.

Read-only on all BAMs/VCFs; writes only under --out-dir.

Run (env snv_caller):
    python scripts/postanalyze/xmhc_wes_dna_check.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import subprocess
import sys

import pysam
from scipy.stats import fisher_exact

STMUT = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/"
         "ST_datasets/STmut_Data")
LO, HI = 28_000_000, 34_000_000          # xMHC, chr6, both builds (see caveat)
MIN_DP = 10                              # "covered" in an exome
ALT_MIN_READS, ALT_MIN_VAF = 2, 0.05     # "ALT present" in that exome
N_CTRL, SEED = 2000, 42

SAMPLES = {
    "P4": dict(
        tumour=f"{STMUT}/P4_cSCC_WES/P4_cSCC_WES.bam",
        normal=f"{STMUT}/P4_Normal_WES/P4_Normal_WES.bam",
        wes_somatic=f"{STMUT}/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz",
        wes_somatic_full=f"{STMUT}/P4_Somatic_Mutect2/P4_somatic.vcf.gz",
        calls="data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
    ),
    "P6": dict(
        tumour=f"{STMUT}/P6_cSCC_WES/P6_cSCC_WES.bam",
        normal=f"{STMUT}/P6_Normal_WES/P6_Normal_WES.bam",
        wes_somatic=f"{STMUT}/P6_Somatic_Mutect2/P6_somatic_exome_snps_chr1_22.vcf.gz",
        wes_somatic_full=f"{STMUT}/P6_Somatic_Mutect2/P6_somatic.vcf.gz",
        calls="data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
    ),
}
BCFTOOLS = "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools"


def in_xmhc(chrom, pos):
    return chrom == "6" and LO <= pos <= HI


def load_calls(path):
    out = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            q = line.rstrip("\n").split("\t")
            if len(q) >= 4:
                out.append((q[0], int(q[1]), q[2], q[3]))
    return out


def count_at(bam, chrom, pos, alt):
    """1-based pos -> (depth, alt_reads); None if the contig is absent."""
    try:
        cols = bam.pileup("chr" + chrom, pos - 1, pos, truncate=True,
                          max_depth=8000, stepper="samtools",
                          min_base_quality=13, ignore_overlaps=False)
    except ValueError:
        return None
    for col in cols:
        if col.reference_pos != pos - 1:
            continue
        dp = na = 0
        for pr in col.pileups:
            if pr.is_del or pr.is_refskip:
                continue
            dp += 1
            if pr.alignment.query_sequence[pr.query_position].upper() == alt.upper():
                na += 1
        return dp, na
    return 0, 0


def vcf_counts(path):
    """(total records, xMHC records, xMHC PASS records) via bcftools."""
    if not os.path.exists(path):
        return None
    reg = f"chr6:{LO}-{HI},6:{LO}-{HI}"

    def n(args):
        r = subprocess.run([BCFTOOLS, "view", "-H", *args, path],
                           capture_output=True, text=True)
        return sum(1 for line in r.stdout.splitlines() if line.strip())

    return n([]), n(["-r", reg]), n(["-f", "PASS", "-r", reg])


def profile(bam_t, bam_n, sites):
    rows, cov = [], 0
    alt_t = alt_n = both = 0
    vt_all, vn_all = [], []
    for chrom, pos, ref, alt in sites:
        rt = count_at(bam_t, chrom, pos, alt)
        rn = count_at(bam_n, chrom, pos, alt)
        if rt is None or rn is None:
            continue
        dt, at = rt
        dn, an = rn
        if dt < MIN_DP or dn < MIN_DP:
            continue
        cov += 1
        vt, vn = at / dt, an / dn
        vt_all.append(vt)
        vn_all.append(vn)
        pt = at >= ALT_MIN_READS and vt >= ALT_MIN_VAF
        pn = an >= ALT_MIN_READS and vn >= ALT_MIN_VAF
        alt_t += pt
        alt_n += pn
        both += pt and pn
        rows.append(dict(chrom=chrom, pos=pos, ref=ref, alt=alt,
                         tumour_dp=dt, tumour_alt=at, tumour_vaf=round(vt, 4),
                         normal_dp=dn, normal_alt=an, normal_vaf=round(vn, 4),
                         alt_in_tumour=pt, alt_in_normal=pn,
                         germline_not_somatic=bool(pt and pn)))
    return dict(n=len(sites), covered=cov, alt_tumour=alt_t, alt_normal=alt_n,
                both=both,
                median_vaf_tumour=round(st.median(vt_all), 4) if vt_all else None,
                median_vaf_normal=round(st.median(vn_all), 4) if vn_all else None), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-control", type=int, default=N_CTRL)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed)

    summary, per_site = [], []
    print(f"xMHC = chr6:{LO:,}-{HI:,}.  'covered' = DP>={MIN_DP} in BOTH exomes.")
    print(f"'ALT present' = >={ALT_MIN_READS} alt reads AND VAF>={ALT_MIN_VAF}.\n")

    for samp, cfg in SAMPLES.items():
        missing = [k for k in ("tumour", "normal", "calls") if not os.path.exists(cfg[k])]
        if missing:
            print(f"{samp}: SKIPPED, missing {missing}", file=sys.stderr)
            continue

        print(f"===== {samp} =====")
        for tag in ("wes_somatic", "wes_somatic_full"):
            c = vcf_counts(cfg[tag])
            if c:
                tot, xm, xp = c
                print(f"  Q2 {os.path.basename(cfg[tag]):42s} total={tot:>6}  "
                      f"xMHC={xm:>3}  xMHC PASS={xp:>3}")
                summary.append(dict(sample=samp, stratum=f"WES_VCF:{tag}",
                                    n=tot, covered="", alt_tumour=xm,
                                    alt_normal="", both=xp,
                                    median_vaf_tumour="", median_vaf_normal="",
                                    note="xMHC / xMHC-PASS record counts"))

        sites = load_calls(cfg["calls"])
        xm = [s for s in sites if in_xmhc(s[0], s[1])]
        non = [s for s in sites if not in_xmhc(s[0], s[1])]
        ctrl = random.sample(non, min(a.n_control, len(non)))
        print(f"  SPARCAL somatic calls: {len(sites)} ({len(xm)} xMHC; "
              f"control = {len(ctrl)} of {len(non)} non-xMHC, seed {a.seed})")

        T = pysam.AlignmentFile(cfg["tumour"], "rb")
        N = pysam.AlignmentFile(cfg["normal"], "rb")
        res = {}
        for label, ss in (("xMHC", xm), ("non_xMHC_control", ctrl)):
            r, rows = profile(T, N, ss)
            res[label] = r
            r2 = dict(sample=samp, stratum=label, **r, note="")
            summary.append(r2)
            for d in rows:
                per_site.append(dict(sample=samp, stratum=label, **d))
            pc = (100 * r["covered"] / r["n"]) if r["n"] else 0
            print(f"  Q3 {label:18s} n={r['n']:>5}  covered={r['covered']:>4} ({pc:5.1f}%)  "
                  f"ALT-in-tumour={r['alt_tumour']:>3}  ALT-in-normal={r['alt_normal']:>3}  "
                  f"in-BOTH(germline)={r['both']:>3}")
        T.close()
        N.close()

        x, c = res["xMHC"], res["non_xMHC_control"]
        if x["covered"] and c["covered"]:
            orr, p = fisher_exact([[x["alt_tumour"], x["covered"] - x["alt_tumour"]],
                                   [c["alt_tumour"], c["covered"] - c["alt_tumour"]]])
            print(f"     tumour-ALT rate xMHC vs control: OR={orr:.2f} p={p:.3g}"
                  f"{'  (ns)' if p >= 0.05 else ''}")
            orr2, p2 = fisher_exact([[x["covered"], x["n"] - x["covered"]],
                                     [c["covered"], c["n"] - c["covered"]]])
            print(f"     WES coverage  xMHC vs control: OR={orr2:.2f} p={p2:.3g}"
                  f"  ({'xMHC better covered' if orr2 > 1 else 'xMHC worse covered'})")
            summary.append(dict(sample=samp, stratum="fisher", n="", covered="",
                                alt_tumour="", alt_normal="", both="",
                                median_vaf_tumour="", median_vaf_normal="",
                                note=(f"tumourALT OR={orr:.3f} p={p:.4g}; "
                                      f"coverage OR={orr2:.3f} p={p2:.4g}")))
        print()

    s_path = os.path.join(a.out_dir, "07_wes_dna_confirmation_summary.csv")
    p_path = os.path.join(a.out_dir, "07b_wes_dna_confirmation_per_site.csv")
    with open(s_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open(p_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_site[0].keys()))
        w.writeheader(); w.writerows(per_site)
    print(f"wrote {s_path}\nwrote {p_path}")


if __name__ == "__main__":
    main()
