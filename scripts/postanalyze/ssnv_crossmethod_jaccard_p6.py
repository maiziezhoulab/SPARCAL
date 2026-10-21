#!/usr/bin/env python
"""Variant-level cross-method overlap on P6: SPARCAL somatic vs SpatialSNV vs the
WES-somatic (Mutect2) truth set. Position-level (chrom,pos) comparison -- SPARCAL's
bundle-matrix column key carries no allele, so allele-aware matching is not possible
for that set; SpatialSNV and WES are allele-aware internally but are also reduced to
position-level here for a like-for-like three-way comparison.

READ-ONLY. Run (env snv_caller, NOT sbatch): python scripts/postanalyze/ssnv_crossmethod_jaccard_p6.py
Output: data/spatialsnv_reanalysis_2026-07-17/p6_crossmethod_jaccard.csv
"""
import gzip
import pandas as pd
import pysam

GT_BUNDLE = ("/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/"
             "p6_gt/extracted/P6_rep1_SPARCAL_somatic/snv_matrix.pkl")
SSNV_FEAT = "/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/results/p6/matrix/p6_alt/features.tsv.gz"
WES_VCF = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
           "P6_Somatic_Mutect2/P6_somatic_snp_chr1_22.vcf.gz")
OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"


def strip_chr(c):
    return c[3:] if c.startswith("chr") else c


def load_sparcal():
    M = pd.read_pickle(GT_BUNDLE)
    out = set()
    for col in M.columns:
        chrom, pos = col.split("_", 1)
        out.add((strip_chr(chrom), int(pos)))
    return out


def load_spatialsnv():
    out = set()
    with gzip.open(SSNV_FEAT, "rt") as fh:
        for line in fh:
            key = line.split("\t", 1)[0]           # "chr11_74456788:G>A"
            chrom_pos, _allele = key.split(":", 1)
            chrom, pos = chrom_pos.rsplit("_", 1)
            out.add((strip_chr(chrom), int(pos)))
    return out


def load_wes():
    out = set()
    with pysam.VariantFile(WES_VCF) as vf:
        for r in vf:
            if r.alts and len(r.ref) == 1 and len(r.alts[0]) == 1:
                out.add((strip_chr(r.chrom), r.pos))
    return out


def jac(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter, union, inter / union if union else float("nan")


if __name__ == "__main__":
    sparcal = load_sparcal()
    ssnv = load_spatialsnv()
    wes = load_wes()
    print(f"SPARCAL P6 somatic: {len(sparcal):,} positions")
    print(f"SpatialSNV P6:      {len(ssnv):,} positions")
    print(f"WES-somatic P6:     {len(wes):,} positions")

    rows = []
    pairs = [("SPARCAL", sparcal, "SpatialSNV", ssnv),
             ("SPARCAL", sparcal, "WES_somatic", wes),
             ("SpatialSNV", ssnv, "WES_somatic", wes)]
    for na, a, nb, b in pairs:
        inter, union, j = jac(a, b)
        row = {"set_A": na, "n_A": len(a), "set_B": nb, "n_B": len(b),
               "intersection": inter, "union": union, "jaccard": round(j, 5),
               "overlap_coef_minsize": round(inter / min(len(a), len(b)), 5),
               "pct_of_A_in_B": round(100 * inter / len(a), 3),
               "pct_of_B_in_A": round(100 * inter / len(b), 3)}
        rows.append(row)
        print(f"{na} vs {nb}: intersection={inter}  jaccard={j:.5f}  "
              f"{100 * inter / len(a):.3f}% of {na} in {nb}, {100 * inter / len(b):.3f}% of {nb} in {na}")

    triple = sparcal & ssnv & wes
    print(f"\nThree-way SPARCAL ∩ SpatialSNV ∩ WES_somatic: {len(triple)}")
    rows.append({"set_A": "SPARCAL", "n_A": len(sparcal), "set_B": "SpatialSNV∩WES_somatic (3-way)",
                 "n_B": len(ssnv & wes), "intersection": len(triple), "union": float("nan"),
                 "jaccard": float("nan"), "overlap_coef_minsize": float("nan"),
                 "pct_of_A_in_B": float("nan"), "pct_of_B_in_A": float("nan")})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/p6_crossmethod_jaccard.csv", index=False)
    print(f"\nWROTE: {OUT}/p6_crossmethod_jaccard.csv")
