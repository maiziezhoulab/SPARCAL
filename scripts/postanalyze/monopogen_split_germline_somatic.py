#!/usr/bin/env python
"""
Split a Monopogen run's calls into germline / somatic / unresolved.

WHY THIS IS POSSIBLE
    Monopogen's two modules already carry the distinction; it is just not
    collected anywhere:

      * `out/germline/{chr}.phased.vcf.gz` -- Beagle-phased genotypes against the
        1KGP panel. This IS the germline callset.
      * `out/somatic/{chr}.allSNVs.csv` -- every locus the somatic module scored.
        Its `genotype` column carries the phased germline GT for that locus
        (`0|0`, `0|1`, `1|0`, `1|1`), or `.|.` when the locus has no germline
        genotype.
      * `out/somatic/{chr}.putativeSNVs.csv` -- the putative somatic calls.

    Verified on DCIS2 (chr1-22): of 345,929 scored candidates, 31,588 carry a
    phased germline genotype and 314,341 are `.|.`; all 42,657 putative somatic
    calls fall in the `.|.` group, and the intersection of putative-somatic with
    germline-genotyped is EXACTLY 0. The classes are disjoint by construction.

OUTPUTS (under <run-dir>/out/classified/)
    {S}.germline.csv            full germline callset, from the phased VCFs
                                (chrom,pos,ref,alt,GT) -- the germline MODULE's
                                output, not restricted to somatic candidates
    {S}.germline_scored.csv     the germline loci that the somatic module also
                                scored, with its depth/quality columns
    {S}.somatic.csv             putativeSNVs, with scores
    {S}.unresolved.csv          scored, no germline genotype, not called somatic
    {S}.classes.txt             counts + the disjointness check

The germline callset is deliberately reported both ways: `{S}.germline.csv` is
much larger than `{S}.germline_scored.csv` because the somatic module only scores
loci that clear its depth floor (>=4 ref and >=4 alt high-quality bases).

USAGE
    python monopogen_split_germline_somatic.py --run-dir <dir> --sample DCIS2
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys

NO_GT = {".|.", "./.", "", "NA", "."}


def read_phased(ger_dir: str):
    """Full germline callset from the germline module's phased VCFs."""
    out = []
    for i in range(1, 23):
        p = os.path.join(ger_dir, f"chr{i}.phased.vcf.gz")
        if not os.path.exists(p):
            sys.stderr.write(f"[warn] missing {p}\n")
            continue
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1:
                    continue          # SNVs only, matching every other callset here
                gt = f[9].split(":", 1)[0]
                out.append((f[0], f[1], f[3], f[4], gt))
    return out


def read_somatic_tables(som_dir: str):
    all_rows, put_keys, hdr_all, hdr_put = [], set(), None, None
    for i in range(1, 23):
        p = os.path.join(som_dir, f"chr{i}.allSNVs.csv")
        if os.path.exists(p):
            with open(p) as fh:
                r = csv.DictReader(fh)
                hdr_all = hdr_all or r.fieldnames
                all_rows.extend(list(r))
        p = os.path.join(som_dir, f"chr{i}.putativeSNVs.csv")
        if os.path.exists(p):
            with open(p) as fh:
                r = csv.DictReader(fh)
                hdr_put = hdr_put or r.fieldnames
                for row in r:
                    put_keys.add((row["chr"], row["pos"],
                                  row["Ref_allele"], row["Alt_allele"]))
    return all_rows, put_keys, hdr_all, hdr_put


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    args = ap.parse_args()

    ger = os.path.join(args.run_dir, "out", "germline")
    som = os.path.join(args.run_dir, "out", "somatic")
    out = os.path.join(args.run_dir, "out", "classified")
    os.makedirs(out, exist_ok=True)
    S = args.sample

    phased = read_phased(ger)
    with open(os.path.join(out, f"{S}.germline.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chr", "pos", "ref", "alt", "GT"])
        w.writerows(phased)

    all_rows, put_keys, hdr_all, hdr_put = read_somatic_tables(som)
    if not all_rows:
        sys.exit(f"no allSNVs.csv found under {som}")

    germ_scored, unresolved, somatic = [], [], []
    for r in all_rows:
        k = (r["chr"], r["pos"], r["ref"], r["alt"])
        has_gt = r.get("genotype", "") not in NO_GT
        if k in put_keys:
            somatic.append(r)
        elif has_gt:
            germ_scored.append(r)
        else:
            unresolved.append(r)

    def dump(name, rows):
        p = os.path.join(out, name)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr_all)
            w.writeheader()
            w.writerows(rows)
        return p

    dump(f"{S}.germline_scored.csv", germ_scored)
    dump(f"{S}.somatic.csv", somatic)
    dump(f"{S}.unresolved.csv", unresolved)

    # disjointness check: a putative somatic call must never carry a germline GT
    bad = [r for r in somatic if r.get("genotype", "") not in NO_GT]

    lines = [
        f"Monopogen call classes -- {S}",
        f"  germline (phased VCF, chr1-22 SNVs) : {len(phased):,}",
        f"  scored candidates (allSNVs)         : {len(all_rows):,}",
        f"    - germline-genotyped              : {len(germ_scored):,}",
        f"    - somatic (putativeSNVs)          : {len(somatic):,}",
        f"    - unresolved (no GT, not somatic) : {len(unresolved):,}",
        f"  somatic calls carrying a germline GT: {len(bad)}  "
        f"({'OK - classes are disjoint' if not bad else '*** OVERLAP, INVESTIGATE ***'})",
    ]
    txt = "\n".join(lines)
    open(os.path.join(out, f"{S}.classes.txt"), "w").write(txt + "\n")
    print(txt)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
