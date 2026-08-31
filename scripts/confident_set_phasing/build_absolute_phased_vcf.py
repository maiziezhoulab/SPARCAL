#!/usr/bin/env python3
"""
build_absolute_phased_vcf.py — Stage 2 (chain relative->absolute) + Stage 3
(honest VCF output) of the confident-set absolute-phasing task (2026-08-26).

INPUT:
  - data/anchored_phase_model_2026-08-24/labels_relative_phase.csv (READ-ONLY)
    filtered to confident==True (897 rows: n_total>=2 AND majority_frac>=0.8).
  - data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/<sample_rel>/
    <chrom>/baseQ0mapQ0/niter5_imputeF_gt/<chrom>.beagle_raw.vcf.gz
    (Stage 1 output, produced by run_stage1_genomewide.py / run_stage1_driver.py).
  - data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/_manifest.json
    (Stage 1 driver manifest, for chromosome-level success/failure + timing).

LOGIC (per candidate v with anchor het h):
  1. Look up h at (anchor_pos_1based, anchor_ref, anchor_alt) in Stage 1's
     phased output for v's sample+chrom.
  2. FAIL (v emitted UNPHASED) if any of:
       - Stage 1 has no output at all for that (sample, chrom) [job never
         run / still running / crashed]
       - h's exact (POS,REF,ALT) is not present in the phased output
         (not a panel-intersection site, or an allele mismatch)
       - h's GT in the phased output is not phased (no "|") -- should not
         happen given Stage 1's 100% het-phased result on every chromosome
         tested so far, but is checked, never assumed
       - h's GT in the phased output is homozygous (0|0 or 1|1) -- would
         mean Beagle's gt=-mode genotype refinement disagreed with the het
         call that produced this anchor in the first place; a real data
         inconsistency, never silently resolved
  3. SUCCESS: anchor_hap_alt = 1 if GT=="0|1" else 0 (GT=="1|0"). Then
       v_hap_alt = anchor_hap_alt         if label == "same"
                 = 1 - anchor_hap_alt     if label == "opposite"
     v's absolute phased GT = "0|1" if v_hap_alt==1 else "1|0".
     (Ties/"other" labels are never in the confident set: majority_frac>=0.8
     with n_total>=2 categorically excludes n_same==n_opposite.)

OUTPUT: one bgzipped VCF per sample under
  data/confident_set_phasing_2026-08-24/phased_vcfs/{SAMPLE}_confident_phased.vcf.gz
containing ALL 897 confident candidates for that sample (phased ones with
"|" GT, failed-anchor ones with "/" GT) -- never omitted, so every one of the
897 is traceable and accounted for. INFO fields carry every piece of
supporting evidence requested: PS (chromosome-scoped phase-set id, since
Beagle 4.1 has no PS tag and phases one implicit block per chromosome, NOT
genome-wide), ANCHOR (chrom:pos), ANCHOR_REF, ANCHOR_ALT, ANCHOR_GT,
DIST_TO_ANCHOR, N_SAME, N_OPPOSITE, N_TOTAL, MAJORITY_FRAC, LABEL,
PHASE_STATUS (fine-grained, per-record): phased / anchor_chromosome_missing /
anchor_not_found / anchor_unphased / anchor_homozygous / label_not_same_or_opposite.
These roll up into exactly two reportable failure buckets, never conflated:
  - anchor_chromosome_missing: Stage 1 has no (successful) output for that
    (sample, chrom) at all -- a COVERAGE GAP. Must be zero once Stage 1 has
    finished all chr1-22 for all four samples.
  - anchor_not_phased: Stage 1 DID run on that chromosome, but this specific
    anchor did not come out phased (not in the panel-intersection output,
    left unphased, or came out homozygous) -- a REAL, reportable finding,
    not a gap.

Never writes into data_sidecar_phased/, data/anchored_phase_model_2026-08-24/,
or any data/<sample>/output_VCFs/ tree. Never edits scripts/1_calling..
scripts/6_spatial_filter.
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
LABELS_CSV = os.path.join(REPO, "data/anchored_phase_model_2026-08-24/labels_relative_phase.csv")
STAGE1_ROOT = os.path.join(REPO, "data/confident_set_phasing_2026-08-24/genomewide_beagle_gt")
STAGE1_MANIFEST_41 = os.path.join(STAGE1_ROOT, "_manifest.json")
STAGE1_MANIFEST_54 = os.path.join(STAGE1_ROOT, "_manifest_beagle54.json")
OUT_DIR = os.path.join(REPO, "data/confident_set_phasing_2026-08-24/phased_vcfs")
# Engine preference order: Beagle 5.4 is PRIMARY (complete, uniform 22-chrom
# coverage per sample, and does not carry 4.1's inconsistent-markers crash
# class). Beagle 4.1 output is used ONLY as a per-chromosome fallback for the
# handful of DCIS chromosomes that were never re-run under 5.4 (there should
# be none after the full run -- this fallback exists purely as a safety net,
# never as the intended primary source). Every emitted record carries which
# engine actually produced its anchor's phase (ENGINE_USED INFO field).
RUN_TAG_54 = "beagle54_imputeF_gt"
RUN_TAG_41 = "niter5_imputeF_gt"
QUALITY_FILTER = "baseQ0mapQ0"

# section label (as it appears in labels_relative_phase.csv "section" column)
# -> (sample_rel used under STAGE1_ROOT, genome build, contig-length source .fai)
SAMPLE_INFO = {
    "P4": dict(sample_rel="P4_tumor/1", build="GRCh37/hg19",
               fai="/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa.fai",
               chr_prefix_in_fai=True, vcf_sample_name=None),
    "P6": dict(sample_rel="P6_tumor/1", build="GRCh37/hg19",
               fai="/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa.fai",
               chr_prefix_in_fai=True, vcf_sample_name=None),
    "DCIS1": dict(sample_rel="dcis1", build="GRCh38",
                  fai="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa.fai",
                  chr_prefix_in_fai=False, vcf_sample_name=None),
    "DCIS2": dict(sample_rel="dcis2", build="GRCh38",
                  fai="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa.fai",
                  chr_prefix_in_fai=False, vcf_sample_name=None),
}


def log(msg):
    print(f"[build_absolute_phased_vcf] {msg}", flush=True)


def load_confident_candidates(labels_csv):
    rows = []
    with open(labels_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["confident"] == "True":
                rows.append(row)
    return rows


def load_chrom_lengths(fai_path, chr_prefix_in_fai):
    lengths = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            name, length = parts[0], int(parts[1])
            bare = name.removeprefix("chr") if chr_prefix_in_fai else name
            if bare.isdigit():
                lengths[bare] = length
    return lengths


def stage1_beagle_path(sample_rel, chrom, run_tag):
    return os.path.join(STAGE1_ROOT, sample_rel, chrom, QUALITY_FILTER, run_tag,
                         f"{chrom}.beagle_raw.vcf.gz")


def resolve_engine_for_chrom(sample, chrom, sample_rel, manifest54_by_key, manifest41_by_key):
    """Beagle 5.4 is PRIMARY. Returns (engine_name_or_None, beagle_vcf_path_or_None).
    Falls back to 4.1 only if no successful 5.4 output exists for this
    (sample, chrom) -- expected to never trigger after the full 5.4 run, but
    kept as an explicit, logged safety net rather than a silent gap."""
    m54 = manifest54_by_key.get((sample, chrom))
    p54 = stage1_beagle_path(sample_rel, chrom, RUN_TAG_54)
    if os.path.exists(p54) and (m54 is None or m54.get("returncode") == 0):
        return "beagle5.4", p54

    m41 = manifest41_by_key.get((sample, chrom))
    p41 = stage1_beagle_path(sample_rel, chrom, RUN_TAG_41)
    if os.path.exists(p41) and (m41 is None or m41.get("returncode") == 0):
        return "beagle4.1", p41

    return None, None


def load_anchor_index(vcf_path):
    """Return dict (pos:int, ref:str, alt:str) -> GT string, for a chromosome's
    Stage 1 phased output. Small file (thousands of records), load fully."""
    index = {}
    with gzip.open(vcf_path, "rt") as fh:
        sample_col = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue
            fields = line.rstrip("\n").split("\t")
            pos = int(fields[1])
            ref, alt = fields[3], fields[4]
            format_keys = fields[8].split(":")
            gt_idx = format_keys.index("GT") if "GT" in format_keys else None
            if gt_idx is None:
                continue
            gt = fields[9].split(":")[gt_idx]
            index[(pos, ref, alt)] = gt
    return index


def resolve_anchor(anchor_index, anchor_pos, anchor_ref, anchor_alt):
    """Returns (status, anchor_hap_alt_or_None, gt_or_None).
    status in {"phased", "not_found", "unphased", "homozygous"}."""
    gt = anchor_index.get((anchor_pos, anchor_ref, anchor_alt))
    if gt is None:
        return "not_found", None, None
    if "|" not in gt:
        return "unphased", None, gt
    alleles = gt.split("|")
    if len(set(alleles)) == 1:
        return "homozygous", None, gt
    # het, phased: alleles[0]/alleles[1] in {"0","1"}
    anchor_hap_alt = 0 if alleles[0] == "1" else 1  # index of haplotype (0 or 1) carrying ALT
    return "phased", anchor_hap_alt, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-csv", default=LABELS_CSV)
    ap.add_argument("--stage1-root", default=STAGE1_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    manifest54 = []
    if os.path.exists(STAGE1_MANIFEST_54):
        with open(STAGE1_MANIFEST_54) as f:
            manifest54 = json.load(f)
    manifest54_by_key = {}
    for m in manifest54:
        manifest54_by_key[(m.get("label"), m.get("chrom"))] = m  # last entry wins

    manifest41 = []
    if os.path.exists(STAGE1_MANIFEST_41):
        with open(STAGE1_MANIFEST_41) as f:
            manifest41 = json.load(f)
    manifest41_by_key = {}
    for m in manifest41:
        manifest41_by_key[(m.get("label"), m.get("chrom"))] = m  # last entry wins

    log(f"Beagle 5.4 manifest entries: {len(manifest54)}; Beagle 4.1 manifest entries: {len(manifest41)}")

    candidates = load_confident_candidates(args.labels_csv)
    log(f"Loaded {len(candidates)} confident candidates from {args.labels_csv}")

    by_sample = defaultdict(list)
    for c in candidates:
        by_sample[c["section"]].append(c)

    overall_summary = {}
    same_opposite_split = {"same": 0, "opposite": 0}
    all_status_counts = defaultdict(int)

    for sample, rows in by_sample.items():
        info = SAMPLE_INFO[sample]
        sample_rel = info["sample_rel"]
        lengths = load_chrom_lengths(info["fai"], info["chr_prefix_in_fai"])

        # group rows by chrom so we load each chromosome's anchor index once
        rows_by_chrom = defaultdict(list)
        for row in rows:
            rows_by_chrom[row["chrom"]].append(row)

        out_records = []  # (chrom_int, pos, record_dict)
        status_counts = defaultdict(int)
        vcf_sample_name = None

        for chrom_bare, chrom_rows in rows_by_chrom.items():
            chrom = f"chr{chrom_bare}"
            engine_used, beagle_vcf = resolve_engine_for_chrom(
                sample, chrom, sample_rel, manifest54_by_key, manifest41_by_key
            )
            stage1_ok = engine_used is not None
            if stage1_ok and engine_used == "beagle4.1":
                log(f"NOTE: {sample} {chrom} has no successful Beagle 5.4 output -- "
                    f"falling back to Beagle 4.1 (should not happen after the full "
                    f"5.4 run; investigate if seen).")

            anchor_index = None
            if stage1_ok:
                try:
                    anchor_index = load_anchor_index(beagle_vcf)
                    if vcf_sample_name is None:
                        with gzip.open(beagle_vcf, "rt") as fh:
                            for line in fh:
                                if line.startswith("#CHROM"):
                                    vcf_sample_name = line.rstrip("\n").split("\t")[9]
                                    break
                except Exception as e:
                    log(f"WARNING: failed to load Stage1 output for {sample} {chrom}: {e}")
                    anchor_index = None
                    stage1_ok = False
                    engine_used = None

            for row in chrom_rows:
                anchor_pos = int(row["anchor_pos_1based"])
                anchor_ref = row["anchor_ref"]
                anchor_alt = row["anchor_alt"]
                v_pos = int(row["pos_1based"])
                v_ref = row["ref"]
                v_alt = row["alt"]
                label = row["label"]

                if not stage1_ok:
                    status = "anchor_chromosome_missing"
                    gt_out = "0/1"
                    ps = "."
                    row_engine = "none"
                else:
                    row_engine = engine_used
                    res_status, anchor_hap_alt, anchor_gt = resolve_anchor(
                        anchor_index, anchor_pos, anchor_ref, anchor_alt
                    )
                    if res_status != "phased":
                        status = f"anchor_{res_status}"
                        gt_out = "0/1"
                        ps = "."
                    else:
                        if label == "same":
                            v_hap_alt = anchor_hap_alt
                        elif label == "opposite":
                            v_hap_alt = 1 - anchor_hap_alt
                        else:
                            # confident set excludes ties by construction (majority_frac>=0.8
                            # is mathematically impossible for n_same==n_opposite); guard anyway
                            status = "label_not_same_or_opposite"
                            gt_out = "0/1"
                            ps = "."
                            status_counts[status] += 1
                            all_status_counts[status] += 1
                            out_records.append((int(chrom_bare), v_pos, {
                                "chrom": chrom, "pos": v_pos, "ref": v_ref, "alt": v_alt,
                                "gt": gt_out, "ps": ps, "row": row, "status": status,
                                "engine": row_engine,
                            }))
                            continue
                        gt_out = "0|1" if v_hap_alt == 1 else "1|0"
                        status = "phased"
                        ps = chrom_bare  # chromosome-scoped phase-set id (int-as-string)
                        same_opposite_split[label] += 1

                status_counts[status] += 1
                all_status_counts[status] += 1
                out_records.append((int(chrom_bare), v_pos, {
                    "chrom": chrom, "pos": v_pos, "ref": v_ref, "alt": v_alt,
                    "gt": gt_out, "ps": ps, "row": row, "status": status,
                    "engine": row_engine,
                }))

        out_records.sort(key=lambda t: (t[0], t[1]))

        # ---- write VCF ----
        vcf_path = os.path.join(args.out_dir, f"{sample}_confident_phased.vcf.gz")
        raw_path = vcf_path[:-3]  # strip .gz, write plain then bgzip
        sample_col_name = vcf_sample_name or f"{sample}_merged"

        with open(raw_path, "w") as out:
            out.write("##fileformat=VCFv4.2\n")
            out.write(f"##source=build_absolute_phased_vcf.py (confident_set_phasing_2026-08-24)\n")
            out.write(f"##reference_build={info['build']}\n")
            out.write('##INFO=<ID=PS_BLOCK,Number=1,Type=String,Description="Chromosome-scoped phase-set id (Beagle 4.1 phases one implicit block per chromosome, no genome-wide PS tag; equals chrom_bare)">\n')
            out.write('##INFO=<ID=ANCHOR,Number=1,Type=String,Description="Anchor het CHROM:POS used for phasing">\n')
            out.write('##INFO=<ID=ANCHOR_REF,Number=1,Type=String,Description="Anchor het REF allele">\n')
            out.write('##INFO=<ID=ANCHOR_ALT,Number=1,Type=String,Description="Anchor het ALT allele">\n')
            out.write('##INFO=<ID=DIST_TO_ANCHOR,Number=1,Type=Integer,Description="Distance in bp from candidate to anchor het">\n')
            out.write('##INFO=<ID=N_SAME,Number=1,Type=Integer,Description="Number of supporting molecules voting same-haplotype">\n')
            out.write('##INFO=<ID=N_OPPOSITE,Number=1,Type=Integer,Description="Number of supporting molecules voting opposite-haplotype">\n')
            out.write('##INFO=<ID=N_TOTAL,Number=1,Type=Integer,Description="Total supporting molecules (n_same+n_opposite+n_other)">\n')
            out.write('##INFO=<ID=MAJORITY_FRAC,Number=1,Type=Float,Description="max(n_same,n_opposite)/n_total">\n')
            out.write('##INFO=<ID=REL_LABEL,Number=1,Type=String,Description="Relative-phase label from labels_relative_phase.csv: same/opposite">\n')
            out.write('##INFO=<ID=PHASE_STATUS,Number=1,Type=String,Description="phased | anchor_chromosome_missing | anchor_not_found | anchor_unphased | anchor_homozygous">\n')
            out.write('##INFO=<ID=ENGINE_USED,Number=1,Type=String,Description="Phasing engine that produced the anchor GT this record is chained from: beagle5.4 (primary) | beagle4.1 (fallback) | none (anchor_chromosome_missing)">\n')
            out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype (phased 0|1 / 1|0, or unphased 0/1 if anchor failed to phase -- never guessed)">\n')
            out.write('##FORMAT=<ID=PS,Number=1,Type=String,Description="Phase set id (chromosome-scoped); \\".\\" if unphased">\n')
            for chrom_bare in sorted(lengths.keys(), key=int):
                length = lengths[chrom_bare]
                out.write(f"##contig=<ID=chr{chrom_bare},length={length}>\n")
            out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample_col_name + "\n")
            for chrom_bare, pos, rec in out_records:
                row = rec["row"]
                info_str = ";".join([
                    f"PS_BLOCK={rec['ps']}",
                    f"ANCHOR=chr{row['chrom']}:{row['anchor_pos_1based']}",
                    f"ANCHOR_REF={row['anchor_ref']}",
                    f"ANCHOR_ALT={row['anchor_alt']}",
                    f"DIST_TO_ANCHOR={row['dist_to_anchor_bp']}",
                    f"N_SAME={row['n_same']}",
                    f"N_OPPOSITE={row['n_opposite']}",
                    f"N_TOTAL={row['n_total']}",
                    f"MAJORITY_FRAC={row['majority_frac']}",
                    f"REL_LABEL={row['label']}",
                    f"PHASE_STATUS={rec['status']}",
                    f"ENGINE_USED={rec['engine']}",
                ])
                out.write(
                    f"{rec['chrom']}\t{rec['pos']}\t.\t{rec['ref']}\t{rec['alt']}\t.\tPASS\t"
                    f"{info_str}\tGT:PS\t{rec['gt']}:{rec['ps']}\n"
                )

        subprocess.run(["bgzip", "-f", raw_path], check=True, cwd=REPO,
                        env={**os.environ, "PATH": f"{REPO}/apps:" + os.environ.get("PATH", "")})
        subprocess.run(["tabix", "-f", "-p", "vcf", vcf_path], check=True,
                        env={**os.environ, "PATH": f"{REPO}/apps:" + os.environ.get("PATH", "")})

        # Rollup into the two reportable failure buckets requested (never conflate
        # "Stage 1 hasn't covered this chromosome yet" with "Stage 1 covered it but
        # this anchor didn't phase"):
        n_chrom_missing = status_counts.get("anchor_chromosome_missing", 0)
        n_not_phased = (status_counts.get("anchor_not_found", 0)
                        + status_counts.get("anchor_unphased", 0)
                        + status_counts.get("anchor_homozygous", 0))
        n_phased = status_counts.get("phased", 0)

        log(f"{sample}: wrote {len(out_records)} records -> {vcf_path}")
        log(f"{sample}: status breakdown: {dict(status_counts)}")
        log(f"{sample}: rollup -- phased={n_phased} anchor_not_phased={n_not_phased} "
            f"anchor_chromosome_missing={n_chrom_missing}")
        overall_summary[sample] = {
            "n_confident_total": len(rows),
            "n_records_written": len(out_records),
            "status_counts": dict(status_counts),
            "n_phased": n_phased,
            "n_anchor_not_phased": n_not_phased,
            "n_anchor_chromosome_missing": n_chrom_missing,
            "vcf_path": vcf_path,
        }

    total_chrom_missing = sum(v["n_anchor_chromosome_missing"] for v in overall_summary.values())
    total_phased = sum(v["n_phased"] for v in overall_summary.values())
    total_not_phased = sum(v["n_anchor_not_phased"] for v in overall_summary.values())
    total_confident = sum(v["n_confident_total"] for v in overall_summary.values())

    if total_chrom_missing > 0:
        log(f"WARNING: {total_chrom_missing} candidates have anchor_chromosome_missing "
            f"(Stage 1 coverage gap) -- this run does NOT reflect a complete Stage 1. "
            f"Do not report these numbers as final.")
    else:
        log(f"Stage 1 coverage check PASSED: 0 candidates with anchor_chromosome_missing "
            f"across all samples/chromosomes.")

    summary_path = os.path.join(args.out_dir, "_stage2_3_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "per_sample": overall_summary,
            "same_opposite_split_in_final_phased_set": same_opposite_split,
            "status_counts_all_samples": dict(all_status_counts),
            "totals": {
                "n_confident_total": total_confident,
                "n_phased": total_phased,
                "n_anchor_not_phased": total_not_phased,
                "n_anchor_chromosome_missing": total_chrom_missing,
                "stage1_coverage_complete": total_chrom_missing == 0,
            },
        }, f, indent=2, default=str)
    log(f"Summary -> {summary_path}")
    print(json.dumps(overall_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
