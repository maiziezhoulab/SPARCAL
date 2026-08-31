#!/usr/bin/env python3
"""
compute_engine_concordance.py — 4.1-vs-5.4 inter-tool phase concordance.

For every (sample, chrom) where BOTH engines produced output, load the
phased GT at every heterozygous site present (same POS,REF,ALT) in both
Beagle 4.1's (niter5_imputeF_gt) and Beagle 5.4's (beagle54_imputeF_gt)
output. For each shared het site, record which haplotype index (0 or 1)
carries ALT under each engine. A whole-chromosome-block flip (engine B's
haplotype-0/1 labeling swapped relative to engine A, which is expected and
NOT a disagreement -- Beagle does not guarantee which arbitrary haplotype
index it calls "0" vs "1", only internal consistency within one run) is
handled by taking whichever global orientation (as-is or flipped) gives
higher agreement, per chromosome:

    n_match_as_is  = count(hapA_i == hapB_i)
    n_match_flip   = count(hapA_i != hapB_i)   # equivalent to hapA_i == (1-hapB_i)
    agreement_rate = max(n_match_as_is, n_match_flip) / n_shared_het_sites

This is a genuine switch-error-like proxy: individual sites that disagree
with the chromosome's dominant orientation are real discordances (possible
switch errors in one engine or the other), not just an arbitrary labeling
difference.

Never fabricates a value: chromosomes with zero shared het sites are
reported as such (agreement_rate=None), never silently dropped or assumed
concordant.
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
STAGE1_ROOT = os.path.join(REPO, "data/confident_set_phasing_2026-08-24/genomewide_beagle_gt")
QUALITY_FILTER = "baseQ0mapQ0"
RUN_TAG_41 = "niter5_imputeF_gt"
RUN_TAG_54 = "beagle54_imputeF_gt"

SAMPLE_INFO = {
    "P4": "P4_tumor/1", "P6": "P6_tumor/1", "DCIS1": "dcis1", "DCIS2": "dcis2",
}
CHROMS = [f"chr{i}" for i in range(1, 23)]


def beagle_path(sample_rel, chrom, run_tag):
    return os.path.join(STAGE1_ROOT, sample_rel, chrom, QUALITY_FILTER, run_tag, f"{chrom}.beagle_raw.vcf.gz")


def load_het_haplotypes(vcf_path):
    """Return dict (pos,ref,alt) -> hap_alt_index (0 or 1) for phased het sites."""
    out = {}
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            pos, ref, alt = int(f[1]), f[3], f[4]
            fmt = f[8].split(":")
            if "GT" not in fmt:
                continue
            gt = f[9].split(":")[fmt.index("GT")]
            if "|" not in gt:
                continue
            alleles = gt.split("|")
            if len(set(alleles)) != 2:
                continue  # homozygous, not a het site
            hap_alt = 0 if alleles[0] == "1" else 1
            out[(pos, ref, alt)] = hap_alt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "data/confident_set_phasing_2026-08-24",
                                                    "engine_concordance.json"))
    args = ap.parse_args()

    per_chrom_rows = []
    per_sample = defaultdict(lambda: {"n_shared": 0, "n_match": 0, "chroms_compared": 0, "chroms_no_54": 0,
                                       "chroms_no_41": 0, "chroms_zero_shared": 0})

    for label, sample_rel in SAMPLE_INFO.items():
        for chrom in CHROMS:
            p41 = beagle_path(sample_rel, chrom, RUN_TAG_41)
            p54 = beagle_path(sample_rel, chrom, RUN_TAG_54)
            has41, has54 = os.path.exists(p41), os.path.exists(p54)
            row = {"sample": label, "chrom": chrom, "has_4.1": has41, "has_5.4": has54}
            if not has41:
                per_sample[label]["chroms_no_41"] += 1
                row["status"] = "no_4.1_output"
                per_chrom_rows.append(row)
                continue
            if not has54:
                per_sample[label]["chroms_no_54"] += 1
                row["status"] = "no_5.4_output"
                per_chrom_rows.append(row)
                continue

            hap41 = load_het_haplotypes(p41)
            hap54 = load_het_haplotypes(p54)
            shared = set(hap41) & set(hap54)
            n_shared = len(shared)
            if n_shared == 0:
                per_sample[label]["chroms_zero_shared"] += 1
                row["status"] = "zero_shared_het_sites"
                row["n_shared_het_sites"] = 0
                row["agreement_rate"] = None
                per_chrom_rows.append(row)
                continue

            n_match_as_is = sum(1 for k in shared if hap41[k] == hap54[k])
            n_match_flip = n_shared - n_match_as_is
            n_match = max(n_match_as_is, n_match_flip)
            orientation = "as_is" if n_match_as_is >= n_match_flip else "flipped"
            agreement_rate = n_match / n_shared

            row.update({
                "status": "compared",
                "n_shared_het_sites": n_shared,
                "n_match_as_is": n_match_as_is,
                "n_match_flipped": n_match_flip,
                "orientation_used": orientation,
                "agreement_rate": agreement_rate,
            })
            per_chrom_rows.append(row)

            per_sample[label]["n_shared"] += n_shared
            per_sample[label]["n_match"] += n_match
            per_sample[label]["chroms_compared"] += 1

    per_sample_summary = {}
    for label, d in per_sample.items():
        rate = (d["n_match"] / d["n_shared"]) if d["n_shared"] > 0 else None
        per_sample_summary[label] = {
            "chroms_compared": d["chroms_compared"],
            "chroms_no_5.4_output": d["chroms_no_54"],
            "chroms_no_4.1_output": d["chroms_no_41"],
            "chroms_zero_shared_het_sites": d["chroms_zero_shared"],
            "n_shared_het_sites_total": d["n_shared"],
            "n_match_total": d["n_match"],
            "overall_agreement_rate": rate,
        }

    overall_shared = sum(d["n_shared_het_sites_total"] for d in per_sample_summary.values())
    overall_match = sum(d["n_match_total"] for d in per_sample_summary.values())
    overall_rate = (overall_match / overall_shared) if overall_shared > 0 else None

    result = {
        "per_chromosome": per_chrom_rows,
        "per_sample": per_sample_summary,
        "overall": {
            "n_shared_het_sites_total": overall_shared,
            "n_match_total": overall_match,
            "agreement_rate": overall_rate,
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[compute_engine_concordance] wrote {args.out}")
    print(json.dumps(result["per_sample"], indent=2, default=str))
    print(json.dumps(result["overall"], indent=2, default=str))


if __name__ == "__main__":
    main()
