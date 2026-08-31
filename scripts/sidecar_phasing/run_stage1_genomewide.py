#!/usr/bin/env python3
"""
run_stage1_genomewide.py — Stage 1 of the confident-set absolute-phasing task
(2026-08-26): genome-wide (chr1-22) `gt=` Beagle phasing for all four samples
(P4, P6, DCIS1, DCIS2), NON-DESTRUCTIVE.

THIS IS A COPY of scripts/sidecar_phasing/run_beagle_phased.py, not an import
and not an edit of that file. Only one thing is deliberately changed:
SIDECAR_OUTPUT_ROOT (see "WHY THE OUTPUT ROOT DIFFERS" below). Every other
function/dict/default is retyped verbatim from run_beagle_phased.py as of
2026-08-26, which itself copied its reference-panel/marker-overlap logic
(verbatim, re-typed) from BeaglePipeline in scripts/2_beagle_filtering/run_beagle.py.
scripts/1_calling/, scripts/2_beagle_filtering/, scripts/3_classifier_prep/,
scripts/4_classifier/, scripts/5_refilter_bam/, scripts/6_spatial_filter/ are
never touched, and nothing is ever written under data/<sample>/output_VCFs/.

WHY THE OUTPUT ROOT DIFFERS FROM run_beagle_phased.py
------------------------------------------------------
The task instructions for this run say to retain output "in the parallel
data_sidecar_phased/ tree" (matching run_beagle_phased.py's own
SIDECAR_OUTPUT_ROOT). But the SAME instructions' absolute rules separately
and explicitly list `data_sidecar_phased/` as READ-ONLY ("Write nothing into
them"), alongside other dated, already-finalized probe-result directories.
That is a direct conflict inside one instruction set: write here / never
write here, both said explicitly about the same path.

Resolution taken: the rule numbered "ABSOLUTE" and phrased as a hard
constraint ("Write nothing into them") is treated as authoritative over the
task prose, because (a) it is explicit, unconditional, and lists this exact
directory by its exact name, not by inference, and (b) `data_sidecar_phased/`
already holds prior, reviewed probe results (P4 chr10, 8 run_tags) that a
silent Stage-1 write could be mistaken for extending/superseding. Rather than
block on this, the genome-wide run was redirected one level: same relative
layout (`<dataset_output_dir>/<chrom>/<quality_filter>/<run_tag>/`), rooted
under this task's own deliverable directory instead:

    data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/

`data_sidecar_phased/` itself is left completely untouched by this script.
This substitution is recorded in RESULTS.md as an explicit deviation from the
literal task wording, for the user to confirm or override.

Everything else below (params, marker-overlap guard, Beagle invocation,
stats) is unchanged from run_beagle_phased.py.
"""

import os
import re
import sys
import json
import time
import gzip
import argparse
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Copied from scripts/2_beagle_filtering/run_beagle.py (read-only reference,
# retyped via run_beagle_phased.py as of 2026-08-26). Do NOT import from that
# module -- copy only, per the non-destructive/no-shared-state requirement.
# ---------------------------------------------------------------------------

REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",
        "regions": [f"chr{i}" for i in range(1, 23)]
    },
    "TUMOR": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",
        "regions": [f"chr{i}" for i in range(1, 23)]
    }
}

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC"
    },
    "OVAR_P5": {
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "DLPFC"
    },
    "NCCE": {
        "base_path": "/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs",
        "output_dir": "data/ncce/{section_id}",
        "has_sections": True,
        "section_ids": ["6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C",
                         "11A", "11B", "11C", "14A", "14B", "14C"],
        "reference": "DLPFC"
    }
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

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "BEAGLE_JAR": "beagle.27Jul16.86a.jar",
    "JAVA": "src/jdk-11.0.2/bin/java",
    "THOUSAND_GENOME_DIR": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix"
}

# Stage 1 production defaults: gt= is the ONLY input field verified to phase
# (see scripts/sidecar_phasing/README.md Stage-2 result); niterations=5 and
# impute=false are the verified-working combination (niterations was shown
# not to matter, 0 and 5 both phase; impute=true crashes under gt= too on
# this jar -- ArrayIndexOutOfBoundsException in ImputationData.refHapSegs).
STAGE1_DEFAULT_PARAMS = {
    "THREADS": 4,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 5,
    "IMPUTE": False,
    "GPROBS": True,
    "INPUT_FIELD": "gt",
}

# See module docstring "WHY THE OUTPUT ROOT DIFFERS" -- this is the only
# substantive change from run_beagle_phased.py.
SIDECAR_OUTPUT_ROOT = os.path.join(
    PATH_CONFIG["PROJECT_DIR"], "data", "confident_set_phasing_2026-08-24", "genomewide_beagle_gt"
)


def log(msg: str) -> None:
    print(f"[stage1_genomewide] {msg}", flush=True)


def get_1000genome_reference(dataset_name: str, chromosome: str) -> str:
    dataset_config = DATASET_CONFIGS[dataset_name]
    reference_name = dataset_config['reference']
    genome_build = "hg19" if reference_name == "TUMOR" else "GRCh38"
    genome_config = THOUSAND_GENOME_CONFIGS[genome_build]
    if genome_build == "hg19":
        chrom = chromosome.replace('chr', '')
    else:
        chrom = chromosome
    reference_pattern = os.path.join(
        genome_config["base_path"],
        genome_config["pattern"].format(chrom=chrom)
    )
    if not os.path.exists(reference_pattern):
        raise FileNotFoundError(
            f"1000 Genome reference not found: {reference_pattern}\n"
            f"Genome build: {genome_build}, Chromosome: {chromosome}"
        )
    return reference_pattern


def chromosome_has_reference_marker(
    dataset_name: str, input_vcf: str, chromosome: str, scratch_dir: str
) -> Tuple[bool, int]:
    target_variants = set()
    normalized_chromosome = chromosome.removeprefix('chr')
    with gzip.open(input_vcf, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if fields[0].removeprefix('chr') == normalized_chromosome:
                target_variants.add((fields[1], fields[3], fields[4]))

    if not target_variants:
        return False, 0

    reference_panel = get_1000genome_reference(dataset_name, chromosome)
    region_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'{chromosome}_',
            suffix='.regions',
            dir=scratch_dir,
            delete=False,
        ) as regions:
            region_file = regions.name
            for position in sorted({variant[0] for variant in target_variants}, key=int):
                regions.write(f'{chromosome}\t{position}\t{position}\n')

        reference_records = subprocess.run(
            ['bcftools', 'view', '-R', region_file, '-H', reference_panel],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in reference_records.splitlines():
            fields = line.split('\t')
            if (fields[1], fields[3], fields[4]) in target_variants:
                return True, len(target_variants)
        return False, len(target_variants)
    finally:
        if region_file and os.path.exists(region_file):
            os.remove(region_file)


def merge_vcf_fields(original_vcf: str, beagle_vcf: str, output_vcf: str) -> None:
    for vcf in [original_vcf, beagle_vcf]:
        if not os.path.exists(vcf + '.tbi'):
            subprocess.run(['tabix', '-p', 'vcf', vcf], check=True)

    cmd = [
        'bcftools', 'annotate',
        '-a', original_vcf,
        '-c', 'INFO/DP,INFO/I16,INFO/QS,INFO/SGB,INFO/RPB,INFO/MQB,INFO/MQSB,INFO/BQB,INFO/MQ0F,FORMAT/GQ,FORMAT/BAF,FORMAT/PL',
        '-O', 'z',
        '-o', output_vcf,
        beagle_vcf
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(['tabix', '-p', 'vcf', output_vcf], check=True)


def resolve_paths(dataset_name: str, section_id: Optional[str], quality_filter: str,
                   chromosome: str, params: Dict) -> Dict[str, str]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choices: {list(DATASET_CONFIGS)}")
    dataset_config = DATASET_CONFIGS[dataset_name]

    if dataset_config["has_sections"]:
        if not section_id:
            raise ValueError(f"Dataset {dataset_name} requires --section_id")
        if "section_ids" in dataset_config and section_id not in dataset_config["section_ids"]:
            raise ValueError(
                f"Invalid section_id for {dataset_name}. Valid: {dataset_config['section_ids']}"
            )
        sample_rel = dataset_config["output_dir"].format(section_id=section_id)
    else:
        sample_rel = dataset_config["output_dir"]

    if sample_rel.startswith("data/"):
        sample_rel = sample_rel[len("data/"):]

    input_vcf = os.path.join(
        PATH_CONFIG["PROJECT_DIR"], "data", sample_rel,
        "output_VCFs", "mpileup_multi_bam", quality_filter,
        "merged_sorted_gt.vcf.gz",
    )

    run_tag = f"niter{params['ITERATIONS']}_impute{'T' if params['IMPUTE'] else 'F'}_{params.get('INPUT_FIELD', 'gt')}"
    run_dir = os.path.join(
        SIDECAR_OUTPUT_ROOT, sample_rel, chromosome, quality_filter, run_tag
    )

    return {
        "dataset_config": dataset_config,
        "sample_rel": sample_rel,
        "input_vcf": input_vcf,
        "beagle_input_vcf": input_vcf,
        "run_dir": run_dir,
        "run_tag": run_tag,
        "beagle_raw_prefix": os.path.join(run_dir, f"{chromosome}.beagle_raw"),
        "beagle_raw_vcf": os.path.join(run_dir, f"{chromosome}.beagle_raw.vcf.gz"),
        "beagle_log": os.path.join(run_dir, f"{chromosome}.beagle_raw.log"),
        "merged_vcf": os.path.join(run_dir, f"{chromosome}.merged.vcf.gz"),
        "stats_json": os.path.join(run_dir, f"{chromosome}.stats.json"),
        "run_meta_json": os.path.join(run_dir, "run_meta.json"),
    }


def run_beagle_command(dataset_name: str, chromosome: str, input_vcf: str,
                        beagle_raw_prefix: str, params: Dict, log_file: str) -> float:
    reference_panel = get_1000genome_reference(dataset_name, chromosome)
    input_field = params.get('INPUT_FIELD', 'gt')

    cmd = [
        os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["JAVA"]),
        f"-Xmx{params['MEMORY']}",
        "-jar",
        os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["BEAGLE_JAR"]),
        f"{input_field}={input_vcf}",
        f"ref={reference_panel}",
        f"chrom={chromosome}",
        f"out={beagle_raw_prefix}",
        f"impute={'true' if params['IMPUTE'] else 'false'}",
        f"modelscale={params['MODEL_SCALE']}",
        f"nthreads={params['THREADS']}",
        f"gprobs={'true' if params['GPROBS'] else 'false'}",
        f"niterations={params['ITERATIONS']}",
    ]

    start = time.time()
    with open(log_file, 'w') as lf:
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return time.time() - start


_AR2_RE = re.compile(r'(?:^|;)AR2=([^;]+)')
_DR2_RE = re.compile(r'(?:^|;)DR2=([^;]+)')


def compute_phasing_stats(vcf_path: str) -> Dict:
    opener = gzip.open if vcf_path.endswith('.gz') else open
    n_records = 0
    n_gt_present = 0
    n_phased = 0
    n_het = 0
    n_het_phased = 0
    ar2_vals: List[float] = []
    dr2_vals: List[float] = []
    has_ps_tag_header = False
    sample_names: List[str] = []

    with opener(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('##'):
                if line.startswith('##FORMAT=<ID=PS,'):
                    has_ps_tag_header = True
                continue
            if line.startswith('#CHROM'):
                header_fields = line.rstrip('\n').split('\t')
                sample_names = header_fields[9:]
                continue

            fields = line.rstrip('\n').split('\t')
            n_records += 1
            info = fields[7]
            m = _AR2_RE.search(info)
            if m and m.group(1) not in ('.', ''):
                try:
                    ar2_vals.append(float(m.group(1)))
                except ValueError:
                    pass
            m = _DR2_RE.search(info)
            if m and m.group(1) not in ('.', ''):
                try:
                    dr2_vals.append(float(m.group(1)))
                except ValueError:
                    pass

            format_keys = fields[8].split(':')
            if 'GT' not in format_keys:
                continue
            gt_idx = format_keys.index('GT')

            for sample_field in fields[9:9 + max(1, len(sample_names))]:
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
                is_het = len(set(alleles)) > 1
                if is_het:
                    n_het += 1
                    if phased:
                        n_het_phased += 1
                break

    def _mean(v):
        return statistics.mean(v) if v else None

    def _median(v):
        return statistics.median(v) if v else None

    return {
        "vcf_path": vcf_path,
        "n_records": n_records,
        "n_samples": len(sample_names) if sample_names else 1,
        "sample_names": sample_names,
        "n_gt_present": n_gt_present,
        "n_phased": n_phased,
        "pct_gt_phased": (100.0 * n_phased / n_gt_present) if n_gt_present else None,
        "n_het": n_het,
        "n_het_phased": n_het_phased,
        "pct_het_phased": (100.0 * n_het_phased / n_het) if n_het else None,
        "ar2_n": len(ar2_vals),
        "ar2_mean": _mean(ar2_vals),
        "ar2_median": _median(ar2_vals),
        "dr2_n": len(dr2_vals),
        "dr2_mean": _mean(dr2_vals),
        "dr2_median": _median(dr2_vals),
        "has_ps_tag_declared_in_header": has_ps_tag_header,
    }


def run_one(dataset_name: str, section_id: Optional[str], quality_filter: str,
            chromosome: str, params: Dict, force: bool = False) -> Dict:
    paths = resolve_paths(dataset_name, section_id, quality_filter, chromosome, params)
    run_dir = paths["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    if not os.path.exists(paths["input_vcf"]):
        raise FileNotFoundError(
            f"Input VCF not found (shipped pre-Beagle mpileup VCF, read-only): {paths['input_vcf']}"
        )

    already_done = (
        os.path.exists(paths["beagle_raw_vcf"])
        and os.path.exists(paths["stats_json"])
        and not force
    )
    if already_done:
        log(f"[{paths['run_tag']}] output already present at {run_dir} -- skipping "
            f"(idempotent; pass --force to rerun).")
        with open(paths["stats_json"]) as f:
            stats = json.load(f)
        with open(paths["run_meta_json"]) as f:
            meta = json.load(f)
        meta["reused_existing_run"] = True
        return {**meta, "stats": stats}

    total_start = time.time()

    marker_start = time.time()
    has_marker, raw_marker_count = chromosome_has_reference_marker(
        dataset_name, paths["beagle_input_vcf"], chromosome, run_dir
    )
    marker_check_seconds = time.time() - marker_start
    if not has_marker:
        raise RuntimeError(
            f"No target markers on {chromosome} overlap the 1000 Genomes reference "
            f"({raw_marker_count} raw calls) -- Beagle would exit without output."
        )
    log(f"[{paths['run_tag']}] {raw_marker_count} raw calls on {chromosome}; "
        f"reference-marker overlap confirmed ({marker_check_seconds:.1f}s check).")

    log(f"[{paths['run_tag']}] launching Beagle: niterations={params['ITERATIONS']} "
        f"impute={params['IMPUTE']} gprobs={params['GPROBS']} modelscale={params['MODEL_SCALE']} "
        f"input_field={params.get('INPUT_FIELD', 'gt')} beagle_input={paths['beagle_input_vcf']}")
    beagle_seconds = run_beagle_command(
        dataset_name, chromosome, paths["beagle_input_vcf"], paths["beagle_raw_prefix"],
        params, paths["beagle_log"]
    )
    if not os.path.exists(paths["beagle_raw_vcf"]):
        raise FileNotFoundError(
            f"Beagle reported success but expected output missing: {paths['beagle_raw_vcf']}"
        )
    log(f"[{paths['run_tag']}] Beagle finished in {beagle_seconds:.1f}s "
        f"-> {paths['beagle_raw_vcf']} (RETAINED, not deleted).")

    subprocess.run(['tabix', '-f', '-p', 'vcf', paths["beagle_raw_vcf"]], check=True)

    merge_start = time.time()
    merge_seconds = None
    merge_error = None
    try:
        merge_vcf_fields(paths["input_vcf"], paths["beagle_raw_vcf"], paths["merged_vcf"])
        merge_seconds = time.time() - merge_start
        log(f"[{paths['run_tag']}] merged-fields parity output -> {paths['merged_vcf']} "
            f"({merge_seconds:.1f}s).")
    except subprocess.CalledProcessError as e:
        merge_seconds = time.time() - merge_start
        merge_error = str(e)
        log(f"[{paths['run_tag']}] WARNING: merge-fields parity step failed ({merge_error}); "
            f"continuing.")

    stats = compute_phasing_stats(paths["beagle_raw_vcf"])
    with open(paths["stats_json"], 'w') as f:
        json.dump(stats, f, indent=2)

    total_seconds = time.time() - total_start
    meta = {
        "dataset": dataset_name,
        "section_id": section_id,
        "chromosome": chromosome,
        "quality_filter": quality_filter,
        "params": params,
        "run_tag": paths["run_tag"],
        "input_vcf": paths["input_vcf"],
        "beagle_input_vcf": paths["beagle_input_vcf"],
        "beagle_raw_vcf": paths["beagle_raw_vcf"],
        "merged_vcf": paths["merged_vcf"] if merge_seconds is not None and merge_error is None else None,
        "merge_error": merge_error,
        "raw_marker_count": raw_marker_count,
        "marker_check_seconds": marker_check_seconds,
        "beagle_wall_seconds": beagle_seconds,
        "merge_wall_seconds": merge_seconds,
        "total_wall_seconds": total_seconds,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reused_existing_run": False,
    }
    with open(paths["run_meta_json"], 'w') as f:
        json.dump(meta, f, indent=2)

    return {**meta, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--chrom", required=True, help="e.g. chr10")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    parser.add_argument("--niterations", type=int, default=STAGE1_DEFAULT_PARAMS["ITERATIONS"])
    parser.add_argument("--impute", action="store_true", default=STAGE1_DEFAULT_PARAMS["IMPUTE"])
    parser.add_argument("--no-gprobs", dest="gprobs", action="store_false",
                         default=STAGE1_DEFAULT_PARAMS["GPROBS"])
    parser.add_argument("--model-scale", type=float, default=STAGE1_DEFAULT_PARAMS["MODEL_SCALE"])
    parser.add_argument("--threads", type=int, default=STAGE1_DEFAULT_PARAMS["THREADS"])
    parser.add_argument("--memory", default=STAGE1_DEFAULT_PARAMS["MEMORY"])
    parser.add_argument("--input-field", choices=["gl", "gt", "gtgl"],
                         default=STAGE1_DEFAULT_PARAMS["INPUT_FIELD"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    params = {
        "THREADS": args.threads,
        "MEMORY": args.memory,
        "MODEL_SCALE": args.model_scale,
        "ITERATIONS": args.niterations,
        "IMPUTE": args.impute,
        "GPROBS": args.gprobs,
        "INPUT_FIELD": args.input_field,
    }

    result = run_one(args.dataset, args.section_id, args.quality_filter,
                      args.chrom, params, force=args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
