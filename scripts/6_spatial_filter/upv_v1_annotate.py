#!/usr/bin/env python
"""UPV v1 — the default post-processing route for the UPV class.

WHAT THIS DOES AND WHY IT DOES SO LITTLE
----------------------------------------
UPV membership is left EXACTLY as the Stage-1 gate defines it (spatial
uniformity alpha > T_alpha AND global prevalence beta > T_beta). We tested
replacing it with a learned classifier trained on gnomAD-corroborated positives
(data/germline_model_2026-09-12/). On the confident set the held-out AUC was only
0.665-0.678, and the training label is ~95% positional coincidence by construction
(30,353 PASS gnomAD hits in DCIS1 against a summed carriage expectation of 1,432).
There is therefore no evidence that would justify changing which variants are
called UPV, and v1 does not change it.

What v1 does add is the stratification we DID validate, as annotation:

  ADP       allelic depth recomputed from the I16 pileup counts (ref+alt, both
            strands). The VCF DP field is not the quantity that governs whether
            VAF is interpretable.
  VAFR      recomputed ALT fraction = (alt_fwd+alt_rev)/ADP. The stored FORMAT
            BAF is unreliable for homozygous-ALT records.
  IS_EDIT   exact hit in the SComatic RNA-editing catalogue.
  UPV_TIER  confident | low_depth | editing

WHY THE TIERS ARE WHERE THEY ARE
--------------------------------
Known-true germline (1KGP panel) reads as 91% "hom-like" at ADP 1-3 purely from
allelic dropout, and only resolves into the het/hom bimodal shape above ADP~50
(het-like 4.7% at ADP 1-3 -> 51.1% at ADP 200+). Below ADP 50 an allelic
statement about a variant is not interpretable, so those calls are emitted but
not asserted.

Editing is split out BEFORE any allelic judgement because RNA editing counterfeits
het-like: DCIS ambiguous at ADP>=100 shows het-like 62.4%, above the panel's
47.6%, while 80.7% of those sites are catalogue hits. Applying an allelic test
before removing editing gives the wrong answer.

Nothing is dropped. Per the 2026-09-12 direction: the confident tier is what gets
reported, compared and calibrated; the rest is called but carries no claim.

USAGE
  python scripts/6_spatial_filter/upv_v1_annotate.py \
      --root data/dcis1 --build hg38 [--qf baseQ0mapQ0]
Writes <root>/spatial_filter_purity/<qf>/germline/denovo/upv_v1.vcf.gz
and upv_v1_summary.tsv next to it. Never overwrites germline_denovo.vcf.gz.
"""
import argparse, bisect, gzip, os, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDIT_BED = {
    "hg19": os.path.join(REPO, "resources/artifact_masks/SComatic/AllEditingSites.hg19.bed.gz"),
    "hg38": os.path.join(REPO, "resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz"),
}
CONFIDENT_ADP = 50


def load_bed(path):
    iv = defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as h:
        for line in h:
            if line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            if len(f) < 3:
                continue
            c = f[0] if f[0].startswith("chr") else "chr" + f[0]
            try:
                iv[c].append((int(f[1]), int(f[2])))
            except ValueError:
                continue
    merged = {}
    for c, v in iv.items():
        v.sort()
        out = []
        for s, e in v:
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        merged[c] = ([x[0] for x in out], [x[1] for x in out])
    return merged


def in_bed(bed, chrom, pos0):
    if chrom not in bed:
        return False
    st, en = bed[chrom]
    i = bisect.bisect_right(st, pos0) - 1
    return i >= 0 and pos0 < en[i]


def canon(c):
    return c if c.startswith("chr") else "chr" + c


def parse_info(s):
    d = {}
    for kv in s.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def load_i16(paths):
    """(chrom,pos,ref,alt) -> (ref_fwd, ref_rev, alt_fwd, alt_rev)."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt") as h:
            for line in h:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8 or len(f[3]) != 1 or "," in f[4] or len(f[4]) != 1:
                    continue
                i16 = parse_info(f[7]).get("I16")
                if not i16:
                    continue
                try:
                    v = [int(float(x)) for x in i16.split(",")[:4]]
                except ValueError:
                    continue
                out[(canon(f[0]), int(f[1]), f[3].upper(), f[4].upper())] = tuple(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. data/dcis1")
    ap.add_argument("--build", required=True, choices=["hg19", "hg38"])
    ap.add_argument("--qf", default="baseQ0mapQ0")
    ap.add_argument("--confident-adp", type=int, default=CONFIDENT_ADP)
    a = ap.parse_args()

    base = os.path.join(a.root, "spatial_filter_purity", a.qf)
    src = os.path.join(base, "germline", "denovo", "germline_denovo.vcf.gz")
    if not os.path.exists(src):
        sys.exit(f"missing UPV VCF: {src}")
    dst = os.path.join(base, "germline", "denovo", "upv_v1.vcf.gz")
    summ = os.path.join(base, "germline", "denovo", "upv_v1_summary.tsv")

    bed = load_bed(EDIT_BED[a.build])
    i16 = load_i16([
        os.path.join(a.root, "output_VCFs", "SeqErrModel", a.qf, "sequence_no_error.vcf.gz"),
        os.path.join(a.root, "output_VCFs", "SeqErrModel", a.qf, "sequence_error.vcf.gz"),
    ])
    print(f"editing catalogue: {sum(len(v[0]) for v in bed.values()):,} intervals")
    print(f"I16 pileup records: {len(i16):,}")

    hdr_extra = [
        '##INFO=<ID=ADP,Number=1,Type=Integer,Description="Allelic depth from I16 (ref+alt, both strands)">',
        '##INFO=<ID=VAFR,Number=1,Type=Float,Description="Recomputed ALT fraction (alt_fwd+alt_rev)/ADP">',
        '##INFO=<ID=IS_EDIT,Number=1,Type=Integer,Description="Exact hit in SComatic RNA-editing catalogue">',
        '##INFO=<ID=UPV_TIER,Number=1,Type=String,Description="UPV v1 tier: confident (ADP>=%d, non-editing) | editing | low_depth">' % a.confident_adp,
        '##sparcal_upv_version=v1',
        '##sparcal_upv_note=Membership unchanged from the Stage-1 gate; v1 adds validated stratification only.',
    ]
    counts = defaultdict(int)
    n_tot = 0
    with gzip.open(src, "rt") as fin, gzip.open(dst, "wt") as fout:
        wrote_extra = False
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue
            if line.startswith("#CHROM"):
                if not wrote_extra:
                    fout.write("\n".join(hdr_extra) + "\n")
                    wrote_extra = True
                fout.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            c, pos, ref, alt = canon(f[0]), int(f[1]), f[3].upper(), f[4].upper()
            n_tot += 1
            edit = 1 if in_bed(bed, c, pos - 1) else 0
            cnt = i16.get((c, pos, ref, alt))
            if cnt:
                adp = cnt[0] + cnt[1] + cnt[2] + cnt[3]
                vafr = (cnt[2] + cnt[3]) / adp if adp else float("nan")
            else:
                adp, vafr = 0, float("nan")
            if edit:
                tier = "editing"
            elif adp >= a.confident_adp:
                tier = "confident"
            else:
                tier = "low_depth"
            counts[tier] += 1
            f[7] = (f[7] + f";ADP={adp};VAFR="
                    + ("%.4f" % vafr if vafr == vafr else ".")
                    + f";IS_EDIT={edit};UPV_TIER={tier}")
            fout.write("\t".join(f) + "\n")

    with open(summ, "w") as h:
        h.write("tier\tn\tfraction\n")
        for t in ("confident", "editing", "low_depth"):
            h.write(f"{t}\t{counts[t]}\t{counts[t]/max(n_tot,1):.4f}\n")
        h.write(f"total\t{n_tot}\t1.0000\n")
    print(f"\nUPV v1 -> {dst}")
    for t in ("confident", "editing", "low_depth"):
        print(f"  {t:10} {counts[t]:>8,}  ({100*counts[t]/max(n_tot,1):.1f}%)")
    print(f"  {'total':10} {n_tot:>8,}")


if __name__ == "__main__":
    main()
