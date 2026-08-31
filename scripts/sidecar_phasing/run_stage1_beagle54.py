#!/usr/bin/env python3
"""
run_stage1_beagle54.py — Beagle 5.4 side-car for the confident-set absolute
phasing task (2026-08-27). NON-DESTRUCTIVE, coexists with the Beagle 4.1
output produced by run_stage1_genomewide.py under the SAME
data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/ tree, in a
sibling run_tag directory (beagle54_imputeF_gt) so neither engine's output
is ever deleted or overwritten by the other.

WHY THIS EXISTS: Beagle 4.1 (gt= mode) crashes on most DCIS chromosomes at
output time with `java.lang.IllegalArgumentException: inconsistent markers`
in `main.ConstrainedAlleleProbs.<init>` / `main.Main.printOutput` -- a
multi-window bug with the dense GRCh38 30x reference panel (same failure
family as the earlier impute=true crash). Beagle 5.4's gt=-only input mode
structurally avoids the code path that crashes (verified: DCIS1 chr1, the
single worst 4.1 failure, phases 100% of het sites cleanly under 5.4).
Verified to also work on hg19 (P4 chr10 spot check) before being run on all
of P4/P6.

Marker-overlap pre-check is INTENTIONALLY SKIPPED here (unlike
run_stage1_genomewide.py): every (sample, chrom) this script is run on has
already been confirmed to have reference-panel overlap by the 4.1 pipeline
(either it succeeded there, or -- for the 29 DCIS failures -- it got PAST
the marker-overlap check and crashed later, at Beagle's own printOutput
step). Re-running the same bcftools-view-over-full-panel check would only
add ~2-10 minutes of redundant I/O per chromosome for no new information.

No `map=` is supplied (matches the 4.1 runs, keeps the two engines
comparable, avoids introducing a new dependency mid-stream) -- Beagle 5.4
falls back to its own default `1 cM = 1 Mb`. This is a recorded limitation,
not an oversight: reference-panel phasing is less map-sensitive than de
novo phasing, but not zero-sensitive.

Never touches scripts/1_calling../6_spatial_filter or data/<sample>/output_VCFs.
"""
import argparse
import gzip
import json
import os
import re
import statistics
import subprocess
import time
from typing import Dict, List, Optional, Tuple

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
JAVA17 = "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/java/17.0.6/bin/java"
BEAGLE54_JAR = "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/beagle/5.4-240301/beagle.01Mar24.d36.jar"
ENGINE_NAME = "beagle5.4"
RUN_TAG = "beagle54_imputeF_gt"

DATASET_CONFIGS = {
    "P4_TUMOR": {"output_dir": "data/P4_tumor/{section_id}", "section_ids": ["1", "2"], "reference": "TUMOR"},
    "P6_TUMOR": {"output_dir": "data/P6_tumor/{section_id}", "section_ids": ["1", "2"], "reference": "TUMOR"},
    "DCIS": {"output_dir": "data/dcis{section_id}", "section_ids": ["1", "2"], "reference": "DLPFC"},
}

THOUSAND_GENOME_CONFIGS = {
    "GRCh38": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "pattern": "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    "hg19": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19/",
        "pattern": "hg19_chr{chrom}.vcf.gz"
    }
}

SIDECAR_OUTPUT_ROOT = os.path.join(REPO, "data", "confident_set_phasing_2026-08-24", "genomewide_beagle_gt")


def log(msg: str) -> None:
    print(f"[stage1_beagle54] {msg}", flush=True)


def get_1000genome_reference(dataset_name: str, chromosome: str) -> str:
    dataset_config = DATASET_CONFIGS[dataset_name]
    reference_name = dataset_config['reference']
    genome_build = "hg19" if reference_name == "TUMOR" else "GRCh38"
    genome_config = THOUSAND_GENOME_CONFIGS[genome_build]
    chrom = chromosome.replace('chr', '') if genome_build == "hg19" else chromosome
    reference_pattern = os.path.join(genome_config["base_path"], genome_config["pattern"].format(chrom=chrom))
    if not os.path.exists(reference_pattern):
        raise FileNotFoundError(f"1000 Genome reference not found: {reference_pattern}")
    return reference_pattern


def resolve_paths(dataset_name, section_id, quality_filter, chromosome):
    dataset_config = DATASET_CONFIGS[dataset_name]
    sample_rel = dataset_config["output_dir"].format(section_id=section_id)
    if sample_rel.startswith("data/"):
        sample_rel = sample_rel[len("data/"):]
    input_vcf = os.path.join(REPO, "data", sample_rel, "output_VCFs", "mpileup_multi_bam",
                              quality_filter, "merged_sorted_gt.vcf.gz")
    run_dir = os.path.join(SIDECAR_OUTPUT_ROOT, sample_rel, chromosome, quality_filter, RUN_TAG)
    return {
        "sample_rel": sample_rel,
        "input_vcf": input_vcf,
        "run_dir": run_dir,
        "beagle_raw_prefix": os.path.join(run_dir, f"{chromosome}.beagle_raw"),
        "beagle_raw_vcf": os.path.join(run_dir, f"{chromosome}.beagle_raw.vcf.gz"),
        "beagle_log": os.path.join(run_dir, f"{chromosome}.beagle_raw.log"),
        "stats_json": os.path.join(run_dir, f"{chromosome}.stats.json"),
        "run_meta_json": os.path.join(run_dir, "run_meta.json"),
    }


def run_beagle54(dataset_name, chromosome, input_vcf, beagle_raw_prefix, threads, log_file):
    reference_panel = get_1000genome_reference(dataset_name, chromosome)
    cmd = [JAVA17, "-Xmx10g", "-jar", BEAGLE54_JAR,
           f"gt={input_vcf}", f"ref={reference_panel}", f"chrom={chromosome}",
           f"out={beagle_raw_prefix}", "impute=false", f"nthreads={threads}"]
    start = time.time()
    with open(log_file, 'w') as lf:
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return time.time() - start


_AR2_RE = re.compile(r'(?:^|;)DR2=([^;]+)')


def compute_phasing_stats(vcf_path: str) -> Dict:
    n_records = n_gt_present = n_phased = n_het = n_het_phased = 0
    sample_names: List[str] = []
    with gzip.open(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                sample_names = line.rstrip('\n').split('\t')[9:]
                continue
            fields = line.rstrip('\n').split('\t')
            n_records += 1
            format_keys = fields[8].split(':')
            if 'GT' not in format_keys:
                continue
            gt_idx = format_keys.index('GT')
            sample_field = fields[9]
            sub = sample_field.split(':')
            if gt_idx >= len(sub):
                continue
            gt = sub[gt_idx]
            if gt in ('.', './.', '.|.', ''):
                continue
            n_gt_present += 1
            phased = '|' in gt
            if phased:
                n_phased += 1
            alleles = re.split(r'[|/]', gt)
            if '.' in alleles or len(alleles) < 2:
                continue
            if len(set(alleles)) > 1:
                n_het += 1
                if phased:
                    n_het_phased += 1
    return {
        "vcf_path": vcf_path, "n_records": n_records, "n_samples": len(sample_names) if sample_names else 1,
        "sample_names": sample_names, "n_gt_present": n_gt_present, "n_phased": n_phased,
        "pct_gt_phased": (100.0 * n_phased / n_gt_present) if n_gt_present else None,
        "n_het": n_het, "n_het_phased": n_het_phased,
        "pct_het_phased": (100.0 * n_het_phased / n_het) if n_het else None,
    }


def run_one(dataset_name, section_id, quality_filter, chromosome, threads, force=False):
    paths = resolve_paths(dataset_name, section_id, quality_filter, chromosome)
    os.makedirs(paths["run_dir"], exist_ok=True)

    if not os.path.exists(paths["input_vcf"]):
        raise FileNotFoundError(f"Input VCF not found: {paths['input_vcf']}")

    if os.path.exists(paths["beagle_raw_vcf"]) and os.path.exists(paths["stats_json"]) and not force:
        log(f"[{RUN_TAG}] output already present at {paths['run_dir']} -- skipping (idempotent).")
        with open(paths["stats_json"]) as f:
            stats = json.load(f)
        with open(paths["run_meta_json"]) as f:
            meta = json.load(f)
        meta["reused_existing_run"] = True
        return {**meta, "stats": stats}

    total_start = time.time()
    log(f"[{RUN_TAG}] launching Beagle 5.4: impute=false map=unset(1cM=1Mb default) "
        f"input_field=gt beagle_input={paths['input_vcf']}")
    beagle_seconds = run_beagle54(dataset_name, chromosome, paths["input_vcf"],
                                   paths["beagle_raw_prefix"], threads, paths["beagle_log"])
    if not os.path.exists(paths["beagle_raw_vcf"]):
        raise FileNotFoundError(f"Beagle 5.4 reported success but expected output missing: {paths['beagle_raw_vcf']}")
    log(f"[{RUN_TAG}] Beagle 5.4 finished in {beagle_seconds:.1f}s -> {paths['beagle_raw_vcf']}")

    subprocess.run(['tabix', '-f', '-p', 'vcf', paths["beagle_raw_vcf"]], check=True)

    stats = compute_phasing_stats(paths["beagle_raw_vcf"])
    with open(paths["stats_json"], 'w') as f:
        json.dump(stats, f, indent=2)

    total_seconds = time.time() - total_start
    meta = {
        "engine": ENGINE_NAME, "dataset": dataset_name, "section_id": section_id, "chromosome": chromosome,
        "quality_filter": quality_filter, "run_tag": RUN_TAG,
        "params": {"impute": False, "map": None, "nthreads": threads},
        "input_vcf": paths["input_vcf"], "beagle_raw_vcf": paths["beagle_raw_vcf"],
        "beagle_wall_seconds": beagle_seconds, "total_wall_seconds": total_seconds,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "reused_existing_run": False,
    }
    with open(paths["run_meta_json"], 'w') as f:
        json.dump(meta, f, indent=2)
    return {**meta, "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    ap.add_argument("--section_id", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--quality-filter", default="baseQ0mapQ0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    result = run_one(args.dataset, args.section_id, args.quality_filter, args.chrom, args.threads, force=args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
