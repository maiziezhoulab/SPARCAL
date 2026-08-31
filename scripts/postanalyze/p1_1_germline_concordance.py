#!/usr/bin/env python3
"""Callable-site matched-normal WES validation of SPARCAL germline calls.

Sensitivity is evaluated at every high-confidence matched-normal WES SNV after
targeted RNA pileup, including SPARCAL misses. A precision-like WES-supported
fraction is evaluated after targeted normal-WES pileup of every final SPARCAL
germline allele. The source WES VCF is variant-only, not a reference-confidence
gVCF, so that second quantity is explicitly qualified in the report.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
SAMTOOLS = PROJECT / "apps/samtools"
BCFTOOLS = PROJECT / "apps/bcftools"
REFERENCE = Path("/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa")
PANEL_DIR = Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19")
WES_ROOT = Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data")
QF = "baseQ0mapQ0"
SAMPLES = {
    "P4": {
        "root": PROJECT / "data/P4_tumor/1",
        "rna_bam": WES_ROOT / "P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam",
        "wes_bam": WES_ROOT / "P4_Normal_WES/P4_Normal_WES.bam",
        "truth": WES_ROOT / "P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz",
    },
    "P6": {
        "root": PROJECT / "data/P6_tumor/1",
        "rna_bam": WES_ROOT / "P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam",
        "wes_bam": WES_ROOT / "P6_Normal_WES/P6_Normal_WES.bam",
        "truth": WES_ROOT / "P6_Normal_WES/P6_Normal_WES_gatk_snp_chr1_22.vcf.gz",
    },
}
GT_FROM_CLASS = {"heterozygous": "0/1", "homozygous": "1/1", "no_variance": "0/0"}
DEPTH_ORDER = ["0", "1-3", "4-9", "10-29", "30+"]
TRUTH_PANELS = ["all", "defined_1kgp", "non_1kgp"]
CALL_PANELS = ["all", "defined_1kgp", "upv"]


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def canon_chrom(chrom):
    chrom = chrom[3:] if chrom.startswith("chr") else chrom
    return "chr" + chrom


def vkey(chrom, pos, ref, alt):
    return canon_chrom(chrom), int(pos), ref.upper(), alt.upper()


def parse_format(fmt, sample):
    return dict(zip(fmt.split(":"), sample.split(":")))


def as_int(value):
    if value in {None, "", "."}:
        return None
    try:
        return int(str(value).split(",")[0])
    except ValueError:
        return None


def load_truth(path, min_dp, min_gq):
    truth = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or "," in f[4]:
                continue
            if f[6] not in {"PASS", "."}:
                continue
            d = parse_format(f[8], f[9])
            gt = d.get("GT", ".").replace("|", "/")
            dp, gq = as_int(d.get("DP")), as_int(d.get("GQ"))
            if gt not in {"0/1", "1/1"} or dp is None or gq is None:
                continue
            if dp >= min_dp and gq >= min_gq:
                truth[vkey(f[0], f[1], f[3], f[4])] = {
                    "wes_gt": gt, "wes_vcf_dp": dp, "wes_gq": gq
                }
    return truth


def load_final_germline(root):
    base = root / ("spatial_filter_purity/%s/germline" % QF)
    result = {}
    for panel, rel in [
        ("defined_1kgp", "defined/germline_defined.vcf.gz"),
        ("upv", "denovo/germline_denovo.vcf.gz"),
    ]:
        with gzip.open(base / rel, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                f = line.rstrip().split("\t")
                if len(f[3]) == 1 and len(f[4]) == 1 and "," not in f[4]:
                    result[vkey(f[0], f[1], f[3], f[4])] = panel
    return result


def load_predictions(root):
    """Load raw GTs for all candidates, then overlay SparcalNet classes where available.

    Final class-specific VCFs intentionally omit FORMAT/sample columns.  The upstream
    ``temp_gt_inferred.vcf`` retains the original GT, whereas SparcalNet's prediction
    VCF contains only the subset evaluated by that model.  Keeping the two sources
    separate prevents panel-defined alleles from disappearing from raw-GT accuracy.
    """
    result = {}
    raw_path = root / ("output_VCFs/mpileup_multi_bam/%s/temp_gt_inferred.vcf" % QF)
    with open_text(raw_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or "," in f[4]:
                continue
            fmt = parse_format(f[8], f[9])
            result[vkey(f[0], f[1], f[3], f[4])] = {
                "raw_gt": fmt.get("GT", ".").replace("|", "/"),
                "model_class": "not_evaluated",
                "model_gt": "not_evaluated",
            }

    model_path = root / ("output_VCFs/Classifier/%s/results/neural_network_predictions.vcf.gz" % QF)
    with gzip.open(model_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or "," in f[4]:
                continue
            info = {}
            for item in f[7].split(";"):
                if "=" in item:
                    a, b = item.split("=", 1)
                    info[a] = b
            fmt = parse_format(f[8], f[9])
            cls = info.get("NEURAL_NETWORK_CLASS", "missing")
            key = vkey(f[0], f[1], f[3], f[4])
            rec = result.setdefault(key, {})
            rec.setdefault("raw_gt", fmt.get("GT", ".").replace("|", "/"))
            rec["model_class"] = cls
            rec["model_gt"] = GT_FROM_CLASS.get(cls, "unknown")
    return result


def write_positions(keys, path):
    def rank(item):
        chrom, pos = item
        token = chrom[3:]
        return (int(token) if token.isdigit() else 99, pos)
    positions = sorted({(k[0], k[1]) for k in keys}, key=rank)
    with open(path, "w") as out:
        for chrom, pos in positions:
            out.write("%s\t%d\n" % (chrom, pos))


def run_pileup(bam, positions, output, force):
    if output.exists() and output.stat().st_size > 0 and not force:
        return
    cmd = [
        str(SAMTOOLS), "mpileup", "-f", str(REFERENCE), "-l", str(positions),
        "-q", "20", "-Q", "13", "-d", "100000", "-o", str(output), str(bam),
    ]
    subprocess.run(cmd, check=True)


def load_panel_keys(positions):
    """Return exact REF/ALT 1KGP alleles at the requested truth positions."""
    result = set()
    by_chrom = {"chr%d" % i: [] for i in range(1, 23)}
    with open(positions) as handle:
        for line in handle:
            chrom = line.split("\t", 1)[0]
            if chrom in by_chrom:
                by_chrom[chrom].append(line)
    for chrom, lines in by_chrom.items():
        if not lines:
            continue
        region_file = positions.with_name(positions.stem + "." + chrom + ".tsv")
        region_file.write_text("".join(lines))
        panel = PANEL_DIR / ("hg19_%s.vcf.gz" % chrom)
        try:
            proc = subprocess.run(
                [str(BCFTOOLS), "query", "-R", str(region_file),
                 "-f", "%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\n", str(panel)],
                check=True, text=True, capture_output=True,
            )
        finally:
            region_file.unlink(missing_ok=True)
        for line in proc.stdout.splitlines():
            f = line.split("\t")
            if len(f) < 5 or len(f[3]) != 1:
                continue
            for alt in f[4].split(","):
                if len(alt) == 1:
                    result.add(vkey(f[0], f[1], f[3], alt))
    return result


def parse_bases(bases, ref):
    counts = Counter()
    i = 0
    while i < len(bases):
        char = bases[i]
        if char == "^":
            i += 2
            continue
        if char == "$":
            i += 1
            continue
        if char in "+-":
            match = re.match(r"(\d+)", bases[i + 1:])
            if not match:
                i += 1
                continue
            i += 1 + len(match.group(1)) + int(match.group(1))
            continue
        if char in ".,":
            counts[ref.upper()] += 1
        elif char.upper() in {"A", "C", "G", "T"}:
            counts[char.upper()] += 1
        i += 1
    return counts


def load_pileup(path):
    result = {}
    with open(path) as handle:
        for line in handle:
            f = line.rstrip().split("\t")
            if len(f) < 5:
                continue
            chrom, pos, ref = canon_chrom(f[0]), int(f[1]), f[2].upper()
            counts = parse_bases(f[4], ref)
            result[(chrom, pos)] = {
                "counts": counts,
                "usable_dp": sum(counts.values()),
                "mpileup_dp": int(f[3]),
            }
    return result


def allele_counts(pile, key):
    rec = pile.get((key[0], key[1]), {"counts": Counter(), "usable_dp": 0})
    return int(rec["counts"][key[2]]), int(rec["counts"][key[3]]), int(rec["usable_dp"])


def depth_bin(dp):
    if dp == 0:
        return "0"
    if dp <= 3:
        return "1-3"
    if dp <= 9:
        return "4-9"
    if dp <= 29:
        return "10-29"
    return "30+"


def wilson(k, n):
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def aggregate_depth(rows):
    output = []
    expanded = pd.concat([rows, rows.assign(panel="all")], ignore_index=True)
    for keys, group in expanded.groupby(["sample", "panel", "rna_depth_bin"], dropna=False):
        sample, panel, dbin = keys
        for truth_gt, g in [("all", group), ("0/1", group[group.wes_gt == "0/1"]),
                            ("1/1", group[group.wes_gt == "1/1"])]:
            n, detected = len(g), int(g.detected.sum())
            lo, hi = wilson(detected, n)
            dg = g[g.detected == 1]
            raw = dg[dg.raw_gt.isin(["0/1", "1/1"])]
            model = dg[dg.model_gt.isin(["0/1", "1/1"])]
            output.append({
                "sample": sample, "panel": panel, "rna_depth_bin": dbin,
                "wes_truth_gt": truth_gt,
                "n_wes_truth": n, "n_sparcal_germline_detected": detected,
                "sensitivity": detected / n if n else math.nan,
                "sensitivity_ci_low": lo, "sensitivity_ci_high": hi,
                "n_raw_gt_evaluable": len(raw),
                "raw_gt_accuracy": (raw.raw_gt == raw.wes_gt).mean() if len(raw) else math.nan,
                "n_model_gt_evaluable": len(model),
                "model_gt_accuracy": (model.model_gt == model.wes_gt).mean() if len(model) else math.nan,
            })
    out = pd.DataFrame(output)
    out["rna_depth_bin"] = pd.Categorical(out.rna_depth_bin, DEPTH_ORDER, ordered=True)
    return out.sort_values(["sample", "panel", "rna_depth_bin", "wes_truth_gt"])


def add_summary(output, analysis, sample, panel, threshold, frame, numerator, note):
    n, k = len(frame), int(numerator(frame).sum()) if len(frame) else 0
    lo, hi = wilson(k, n)
    output.append({
        "analysis": analysis, "sample": sample, "panel": panel,
        "threshold": threshold, "n": n, "numerator": k,
        "estimate": k / n if n else math.nan, "ci_low": lo, "ci_high": hi,
        "note": note,
    })


def build_summary(truth_rows, call_rows):
    output = []
    for sample in ["P4", "P6"]:
        sdf = truth_rows[truth_rows["sample"] == sample]
        for panel in TRUTH_PANELS:
            g = sdf if panel == "all" else sdf[sdf.panel == panel]
            for minimum in [1, 4, 10, 30]:
                q = g[g.rna_dp >= minimum]
                add_summary(output, "truth_sensitivity", sample, panel,
                            "RNA_DP>=%d" % minimum, q, lambda x: x.detected,
                            "Exact REF/ALT SPARCAL germline detection among high-confidence WES SNVs")
                dg = q[q.detected == 1]
                raw = dg[dg.raw_gt.isin(["0/1", "1/1"])]
                model = dg[dg.model_gt.isin(["0/1", "1/1"])]
                add_summary(output, "raw_gt_accuracy", sample, panel,
                            "RNA_DP>=%d" % minimum, raw,
                            lambda x: x.raw_gt == x.wes_gt,
                            "Among exact-allele SPARCAL germline detections")
                add_summary(output, "model_gt_accuracy", sample, panel,
                            "RNA_DP>=%d" % minimum, model,
                            lambda x: x.model_gt == x.wes_gt,
                            "Among exact-allele SPARCAL germline detections")
        sdf = call_rows[call_rows["sample"] == sample]
        for panel in CALL_PANELS:
            g = sdf if panel == "all" else sdf[sdf.panel == panel]
            for minimum in [10, 20, 30]:
                q = g[g.wes_pileup_dp >= minimum]
                add_summary(output, "wes_supported_fraction", sample, panel,
                            "normal_WES_pileup_DP>=%d" % minimum, q,
                            lambda x: x.high_conf_wes_exact,
                            "Precision-like; WES source is variant-only, not a reference-confidence gVCF")
                add_summary(output, "wes_alt_evidence_fraction", sample, panel,
                            "normal_WES_pileup_DP>=%d" % minimum, q,
                            lambda x: x.high_conf_wes_exact.astype(bool) | (
                                (x.normal_wes_alt_count >= 3) & (x.wes_pileup_vaf >= 0.10)
                            ),
                            "Exact WES-VCF support or >=3 ALT reads and VAF>=0.10; supportive bound, not calibrated precision")
    return pd.DataFrame(output)


def make_report(outdir, summary, truth_rows, call_rows, min_dp, min_gq):
    lines = [
        "# P1-1 matched-normal WES germline validation", "", "## Design", "",
        "Truth variants are biallelic PASS autosomal SNVs in the matched-normal GATK VCF "
        "with DP>=%d and GQ>=%d. RNA depth is measured independently at every truth site "
        "by targeted mpileup of the whole Visium BAM (mapping quality >=20, base quality "
        ">=13). Final SPARCAL germline calls are the union of the defined/1KGP and UPV VCFs. "
        "Exact REF/ALT matching is required." % (min_dp, min_gq), "",
        "For WES-truth sensitivity, `non_1kgp` means absent from the queried 1KGP "
        "panel at that exact allele; it does not assume that SPARCAL detected or "
        "classified the site as UPV. For final-call support, `upv` refers to the "
        "actual final de-novo germline/UPV VCF.", "",
        "The WES-supported fraction is precision-like rather than calibrated precision: "
        "the available WES file is variant-only, so an adequately covered absent allele "
        "does not have an explicit homozygous-reference genotype quality. A second, looser "
        "bound counts an exact WES-VCF match or at least three normal-WES ALT reads at "
        "VAF>=0.10; this is reported separately and is not called precision.", "",
        "## Results at RNA depth >=10", "",
        "| sample | panel | truth sites | detected | sensitivity | raw-GT accuracy | model-class accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in ["P4", "P6"]:
        for panel in TRUTH_PANELS:
            def one(analysis):
                x = summary[(summary.analysis == analysis) & (summary["sample"] == sample)
                            & (summary.panel == panel) & (summary.threshold == "RNA_DP>=10")]
                return x.iloc[0] if len(x) else None
            sens, raw, model = one("truth_sensitivity"), one("raw_gt_accuracy"), one("model_gt_accuracy")
            if sens is None:
                continue
            def fmt(row):
                return "NA" if row is None or pd.isna(row.estimate) else "%.3f (%d)" % (row.estimate, row.n)
            lines.append("| %s | %s | %d | %d | %.3f | %s | %s |" % (
                sample, panel, sens.n, sens.numerator, sens.estimate, fmt(raw), fmt(model)))
    lines += [
        "", "## Interpretation", "",
        "- Sensitivity is conditional on RNA depth; no RNA method can recover a normal-WES "
        "SNV with no overlapping transcript reads.",
        "- Genotype agreement is secondary because tumor RNA can differ from normal DNA through "
        "allele-specific expression, copy-number change and LOH. SparcalNet selects a class but "
        "does not rewrite the original VCF GT field.",
        "- The prior 51-56% audit was restricted to non-1KGP sites already present in the "
        "prediction VCF. It remains a model diagnostic, not the P1-1 headline.",
        "- No current P4/P6 RNA-derived GATK or Strelka2 callsets exist in the workspace. A "
        "three-way comparison requires new harmonized RNA-caller runs; tumor-WES or DLPFC "
        "callsets are not substituted.", "", "## Output files", "",
        "- concordance_by_depth.csv: sensitivity and genotype agreement by sample, panel and RNA depth.",
        "- concordance_summary.csv: thresholded estimates with Wilson intervals.",
        "- truth_site_details.csv.gz: one row per high-confidence normal-WES truth allele.",
        "- sparcal_call_details.csv.gz: one row per final SPARCAL germline allele.",
        "- sample target and mpileup TSVs: exact callable-site inputs and cached counts.", "",
        "## Source paths", "",
    ]
    for sample, cfg in SAMPLES.items():
        lines += [
            "- %s normal-WES VCF: %s" % (sample, cfg["truth"]),
            "- %s normal-WES BAM: %s" % (sample, cfg["wes_bam"]),
            "- %s whole Visium BAM: %s" % (sample, cfg["rna_bam"]),
            "- %s SPARCAL root: %s" % (sample, cfg["root"]),
        ]
    lines += ["- Generator: %s" % Path(__file__).resolve(), "",
              "Truth rows: %d; SPARCAL-call rows: %d." % (len(truth_rows), len(call_rows))]
    (outdir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def write_results(outdir, truth_df, calls_df, min_dp, min_gq):
    by_depth = aggregate_depth(truth_df)
    summary = build_summary(truth_df, calls_df)
    by_depth.to_csv(outdir / "concordance_by_depth.csv", index=False)
    summary.to_csv(outdir / "concordance_summary.csv", index=False)
    truth_df.to_csv(outdir / "truth_site_details.csv.gz", index=False, compression="gzip")
    calls_df.to_csv(outdir / "sparcal_call_details.csv.gz", index=False, compression="gzip")
    make_report(outdir, summary, truth_df, calls_df, min_dp, min_gq)
    print(summary.to_string(index=False))
    print("Wrote P1-1 package to %s" % outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT / "data/germline_concordance_2026-08-23")
    parser.add_argument("--min-wes-dp", type=int, default=10)
    parser.add_argument("--min-wes-gq", type=int, default=20)
    parser.add_argument("--force-pileup", action="store_true")
    parser.add_argument(
        "--reaggregate-existing", action="store_true",
        help="reuse completed detail tables and refresh GT/model metrics without rerunning panel queries or pileups",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.reaggregate_existing:
        truth_path = args.outdir / "truth_site_details.csv.gz"
        calls_path = args.outdir / "sparcal_call_details.csv.gz"
        if not truth_path.exists() or not calls_path.exists():
            raise FileNotFoundError("reaggregation requires completed detail tables")
        truth_df = pd.read_csv(truth_path)
        calls_df = pd.read_csv(calls_path)
        # A panel-absent WES truth allele is not an UPV unless SPARCAL detects
        # and classifies it. Preserve actual final-call UPVs in calls_df.
        truth_df.loc[truth_df.panel.eq("upv"), "panel"] = "non_1kgp"
        for sample, cfg in SAMPLES.items():
            predictions = load_predictions(cfg["root"])
            mask = truth_df["sample"] == sample
            for idx, row in truth_df.loc[mask].iterrows():
                pred = predictions.get(vkey(row.chrom, row.pos, row.ref, row.alt), {})
                truth_df.at[idx, "raw_gt"] = pred.get("raw_gt", "missing")
                truth_df.at[idx, "model_class"] = pred.get("model_class", "missing")
                truth_df.at[idx, "model_gt"] = pred.get("model_gt", "missing")
        write_results(args.outdir, truth_df, calls_df, args.min_wes_dp, args.min_wes_gq)
        return
    truth_rows, call_rows = [], []
    for sample, cfg in SAMPLES.items():
        truth = load_truth(cfg["truth"], args.min_wes_dp, args.min_wes_gq)
        calls = load_final_germline(cfg["root"])
        predictions = load_predictions(cfg["root"])
        target_path = args.outdir / ("%s_targets.tsv" % sample)
        write_positions(set(truth) | set(calls), target_path)
        truth_target_path = args.outdir / ("%s_truth_targets.tsv" % sample)
        write_positions(set(truth), truth_target_path)
        panel_keys = load_panel_keys(truth_target_path)
        rna_path = args.outdir / ("%s_rna.mpileup.tsv" % sample)
        wes_path = args.outdir / ("%s_normal_wes.mpileup.tsv" % sample)
        run_pileup(cfg["rna_bam"], target_path, rna_path, args.force_pileup)
        run_pileup(cfg["wes_bam"], target_path, wes_path, args.force_pileup)
        rna_pile, wes_pile = load_pileup(rna_path), load_pileup(wes_path)
        for key, t in truth.items():
            rr, ra, rdp = allele_counts(rna_pile, key)
            pred = predictions.get(key, {})
            panel = "defined_1kgp" if key in panel_keys else "non_1kgp"
            detected_panel = calls.get(key, "not_detected")
            truth_rows.append({
                "sample": sample, "chrom": key[0], "pos": key[1], "ref": key[2], "alt": key[3],
                **t, "panel": panel, "detected_panel": detected_panel,
                "detected": int(detected_panel != "not_detected"),
                "rna_ref_count": rr, "rna_alt_count": ra, "rna_dp": rdp,
                "rna_vaf": ra / rdp if rdp else math.nan, "rna_depth_bin": depth_bin(rdp),
                "raw_gt": pred.get("raw_gt", "missing"),
                "model_class": pred.get("model_class", "missing"),
                "model_gt": pred.get("model_gt", "missing"),
            })
        for key, panel in calls.items():
            wr, wa, wdp = allele_counts(wes_pile, key)
            rr, ra, rdp = allele_counts(rna_pile, key)
            t = truth.get(key)
            call_rows.append({
                "sample": sample, "chrom": key[0], "pos": key[1], "ref": key[2], "alt": key[3],
                "panel": panel, "normal_wes_ref_count": wr, "normal_wes_alt_count": wa,
                "wes_pileup_dp": wdp, "wes_pileup_vaf": wa / wdp if wdp else math.nan,
                "high_conf_wes_exact": int(t is not None),
                "wes_gt": t["wes_gt"] if t else "absent_from_variant_vcf",
                "wes_vcf_dp": t["wes_vcf_dp"] if t else math.nan,
                "wes_gq": t["wes_gq"] if t else math.nan,
                "rna_ref_count": rr, "rna_alt_count": ra, "rna_dp": rdp,
                "rna_vaf": ra / rdp if rdp else math.nan, "rna_depth_bin": depth_bin(rdp),
            })
    truth_df, calls_df = pd.DataFrame(truth_rows), pd.DataFrame(call_rows)
    write_results(args.outdir, truth_df, calls_df, args.min_wes_dp, args.min_wes_gq)


if __name__ == "__main__":
    main()
