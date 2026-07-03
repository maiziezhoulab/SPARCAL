#!/usr/bin/env python3
"""
COSMIC_validation.py
Intersects somatic_denovo and germline_denovo VCFs against COSMIC GenomeScreenMutants.
Outputs per-sample hit TSVs and a combined summary table.

Usage:
    python COSMIC_validation.py \
        --dataset P4_tumor \
        --sections 1 2 \
        --genome GRCh37 \
        --outdir /data/maiziezhou_lab/leiy4/COSMIC/validation
"""

import argparse
import os
import subprocess
import tempfile
import shutil
import sys

# ── Constants ─────────────────────────────────────────────────────────────────
BCFTOOLS     = "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools"
DATA_ROOT    = "/data/maiziezhou_lab/leiy4/snv_calling/data"
COSMIC_DIR   = "/data/maiziezhou_lab/leiy4/COSMIC"
COSMIC_VCFS  = {
    "GRCh37": os.path.join(COSMIC_DIR, "Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz"),
    "GRCh38": os.path.join(COSMIC_DIR, "Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz"),
}

VARIANT_TYPES = ["somatic", "germline"]


# ── Path helpers ──────────────────────────────────────────────────────────────
def get_denovo_vcf(dataset, section, vtype):
    """Return path to somatic_denovo or germline_denovo VCF."""
    return os.path.join(
        DATA_ROOT, dataset, str(section),
        "spatial_filter_purity", "baseQ0mapQ0",
        vtype, "denovo", f"{vtype}_denovo.vcf.gz"
    )


# ── VCF preparation: strip chr prefix, bgzip, tabix ──────────────────────────
def prepare_vcf(input_vcf, outdir, label):
    """
    Strip chr prefix from both header contig IDs and data lines.
    Returns path to prepared .vcf.gz with .tbi index.
    """
    out_vcf = os.path.join(outdir, f"{label}.vcf.gz")
    print(f"  [prepare] {os.path.basename(input_vcf)} -> {os.path.basename(out_vcf)}")

    view_cmd  = [BCFTOOLS, "view", input_vcf]
    bgzip_cmd = ["bgzip", "-c"]

    view_proc = subprocess.Popen(view_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sed_proc  = subprocess.Popen(
        ["sed", "s/^chr//; s/ID=chr/ID=/"],
        stdin=view_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    view_proc.stdout.close()

    with open(out_vcf, "wb") as fout:
        bgzip_proc = subprocess.Popen(bgzip_cmd, stdin=sed_proc.stdout, stdout=fout)
        sed_proc.stdout.close()
        bgzip_proc.wait()

    sed_proc.wait()
    view_proc.wait()

    subprocess.run(["tabix", "-p", "vcf", out_vcf], check=True)
    return out_vcf


# ── Ensure COSMIC VCF is indexed ─────────────────────────────────────────────
def ensure_cosmic_indexed(cosmic_vcf):
    tbi = cosmic_vcf + ".tbi"
    if not os.path.exists(tbi):
        print(f"  [index] Indexing COSMIC VCF: {os.path.basename(cosmic_vcf)}")
        subprocess.run(["tabix", "-p", "vcf", cosmic_vcf], check=True)
    return cosmic_vcf


# ── bcftools isec ─────────────────────────────────────────────────────────────
def run_isec(query_vcf, cosmic_vcf, isec_dir):
    """
    Run bcftools isec. Output structure:
      0000.vcf  query only (not in COSMIC)
      0001.vcf  COSMIC only
      0002.vcf  query variants ALSO in COSMIC  <- hits we want
      0003.vcf  COSMIC variants ALSO in query
    """
    os.makedirs(isec_dir, exist_ok=True)
    subprocess.run(
        [BCFTOOLS, "isec", "-p", isec_dir, query_vcf, cosmic_vcf],
        check=True,
        stderr=subprocess.PIPE
    )
    return os.path.join(isec_dir, "0002.vcf")


# ── Parse hits VCF → TSV with COSMIC INFO fields ─────────────────────────────
def parse_hits(hits_vcf, dataset, section, vtype):
    """
    Parse 0002.vcf and extract GENE, TIER, GENOME_SCREEN_SAMPLE_COUNT from INFO.
    Returns list of dicts.
    """
    records = []
    if not os.path.exists(hits_vcf):
        return records

    with open(hits_vcf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, pos, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            info_str = parts[7] if len(parts) > 7 else "."
            info = {}
            for kv in info_str.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    info[k] = v
            records.append({
                "dataset":    dataset,
                "section":    section,
                "type":       vtype,
                "CHROM":      chrom,
                "POS":        pos,
                "ID":         vid,
                "REF":        ref,
                "ALT":        alt,
                "GENE":       info.get("GENE", "."),
                "TIER":       info.get("TIER", "."),
                "SAMPLE_CNT": info.get("GENOME_SCREEN_SAMPLE_COUNT", "."),
                "SO_TERM":    info.get("SO_TERM", "."),
            })
    return records


# ── Count variants in a VCF ───────────────────────────────────────────────────
def count_vcf(vcf_path):
    result = subprocess.run(
        [BCFTOOLS, "view", "-H", vcf_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return len([l for l in result.stdout.decode().splitlines() if l and not l.startswith("#")])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="COSMIC validation of somatic/germline denovo VCFs")
    parser.add_argument("--dataset",  required=True, help="e.g. P4_tumor, P6_tumor, DCIS")
    parser.add_argument("--sections", nargs="+", required=True, help="section IDs e.g. 1 2 or dcis1 dcis2")
    parser.add_argument("--genome",   required=True, choices=["GRCh37", "GRCh38"])
    parser.add_argument("--outdir",   required=True, help="Output directory")
    args = parser.parse_args()

    cosmic_vcf = ensure_cosmic_indexed(COSMIC_VCFS[args.genome])
    os.makedirs(args.outdir, exist_ok=True)

    all_hits    = []
    summary_rows = []

    for section in args.sections:
        for vtype in VARIANT_TYPES:
            tag = f"{args.dataset}_sec{section}_{vtype}"
            print(f"\n[INFO] Processing {tag}")

            input_vcf = get_denovo_vcf(args.dataset, section, vtype)
            if not os.path.exists(input_vcf):
                print(f"  [WARN] Not found, skipping: {input_vcf}")
                summary_rows.append({
                    "dataset": args.dataset, "section": section, "type": vtype,
                    "total": "N/A", "cosmic_hits": "N/A", "hit_rate": "N/A"
                })
                continue

            # Working dir for this sample
            work_dir = os.path.join(args.outdir, tag)
            os.makedirs(work_dir, exist_ok=True)

            # Step 1: prepare (strip chr, bgzip, index)
            prepared_vcf = prepare_vcf(input_vcf, work_dir, f"{tag}_prepared")

            # Step 2: isec vs COSMIC
            isec_dir  = os.path.join(work_dir, "isec")
            hits_vcf  = run_isec(prepared_vcf, cosmic_vcf, isec_dir)

            # Step 3: parse hits
            hits = parse_hits(hits_vcf, args.dataset, section, vtype)
            all_hits.extend(hits)

            # Step 4: save per-sample hits TSV
            hits_tsv = os.path.join(work_dir, f"{tag}_cosmic_hits.tsv")
            tsv_cols  = ["dataset", "section", "type", "CHROM", "POS", "REF", "ALT",
                         "GENE", "TIER", "SAMPLE_CNT", "SO_TERM"]
            with open(hits_tsv, "w") as f:
                f.write("\t".join(tsv_cols) + "\n")
                for r in hits:
                    f.write("\t".join(str(r[c]) for c in tsv_cols) + "\n")
            print(f"  [output] {hits_tsv}")

            # Step 5: count for summary
            total = count_vcf(prepared_vcf)
            nhits = len(hits)
            rate  = f"{nhits/total*100:.1f}%" if total > 0 else "N/A"
            summary_rows.append({
                "dataset": args.dataset, "section": section, "type": vtype,
                "total": total, "cosmic_hits": nhits, "hit_rate": rate
            })

    # ── Write combined hits TSV ───────────────────────────────────────────────
    combined_tsv = os.path.join(args.outdir, f"{args.dataset}_all_cosmic_hits.tsv")
    tsv_cols = ["dataset", "section", "type", "CHROM", "POS", "REF", "ALT",
                "GENE", "TIER", "SAMPLE_CNT", "SO_TERM"]
    with open(combined_tsv, "w") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for r in all_hits:
            f.write("\t".join(str(r[c]) for c in tsv_cols) + "\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  COSMIC Validation Summary — {args.dataset} ({args.genome})")
    print("="*65)
    print(f"  {'Section':<10} {'Type':<10} {'Total':>8} {'COSMIC Hits':>12} {'Hit Rate':>10}")
    print("-"*65)
    for r in summary_rows:
        print(f"  {str(r['section']):<10} {r['type']:<10} {str(r['total']):>8} "
              f"{str(r['cosmic_hits']):>12} {str(r['hit_rate']):>10}")
    print("="*65)
    print(f"\n  Combined hits TSV : {combined_tsv}")
    print(f"  Per-sample TSVs   : {args.outdir}/<sample>/")


if __name__ == "__main__":
    main()