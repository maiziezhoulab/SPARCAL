#!/usr/bin/env python3
"""
run_beagle_phased.py — NON-DESTRUCTIVE side-car Beagle phasing probe.

This script does NOT modify, import from, or get imported by
scripts/2_beagle_filtering/run_beagle.py. It exists to answer one question:
does Beagle 4.1 emit usable phase on this project's data when `niterations` is
set to a value greater than 0, and at what rate?

WHY THIS EXISTS
----------------
scripts/2_beagle_filtering/run_beagle.py runs Beagle 4.1 with
DEFAULT_PARAMS["ITERATIONS"] = 0, which disables the Li-Stephens phasing
iterations entirely. Every record it has ever produced carries AR2=0;DR2=0 and
an unphased GT ("/", never "|"). This script reruns Beagle on one chromosome
of one sample with niterations set to Beagle's own factory default (5) and to
10, WITHOUT touching the shipped pipeline, and retains Beagle's raw output
(the shipped script deletes it) so the phased result can actually be
inspected.

WHAT IS COPIED FROM run_beagle.py (verbatim logic, re-typed here on purpose)
-----------------------------------------------------------------------
- REFERENCE_CONFIGS, DATASET_CONFIGS, THOUSAND_GENOME_CONFIGS, PATH_CONFIG
  (same dicts, same values, as of 2026-08-24)
- get_1000genome_reference()      <- BeaglePipeline.get_1000genome_reference
- chromosome_has_reference_marker() <- BeaglePipeline.chromosome_has_reference_marker
- the Beagle command-line construction <- BeaglePipeline.run_beagle_command
- the bcftools annotate merge recipe <- BeaglePipeline.merge_vcf_fields

Everything else (output paths, retention of raw output, configurable
niterations/impute, idempotency, the stats function) is new for this side-car.

OUTPUT LAYOUT (parallel tree — never touches data/<sample>/output_VCFs/)
-------------------------------------------------------------------
data_sidecar_phased/<dataset_output_dir>/<chrom>/<quality_filter>/<run_tag>/
    <chrom>.beagle_raw.vcf.gz(.tbi)   Beagle's own output. RETAINED, never deleted.
    <chrom>.beagle_raw.log            Beagle's stdout/stderr (command line + run stats)
    <chrom>.merged.vcf.gz(.tbi)       original INFO/FORMAT fields re-annotated onto
                                       the raw phased calls (same recipe as
                                       merge_vcf_fields in run_beagle.py) — NOT
                                       required for the phasing question, produced
                                       for parity with the shipped pipeline's final
                                       product.
    <chrom>.stats.json                phasing/quality metrics computed from
                                       <chrom>.beagle_raw.vcf.gz (see
                                       compute_phasing_stats below)
    run_meta.json                     exact command line, params, wall time,
                                       timestamps

run_tag = f"niter{ITERATIONS}_impute{'T' if IMPUTE else 'F'}"

USAGE
-----
    conda activate snv_caller
    python scripts/sidecar_phasing/run_beagle_phased.py \\
        --dataset P4_TUMOR --section_id 1 --chrom chr10 \\
        --niterations 5 --quality-filter baseQ0mapQ0

    # analyze an arbitrary VCF (e.g. the shipped niterations=0 output) with the
    # exact same stats function used for the side-car runs, no Beagle run:
    python scripts/sidecar_phasing/run_beagle_phased.py --analyze-only \\
        --vcf-path data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/chr10.vcf.gz

Idempotent: if <chrom>.merged.vcf.gz and <chrom>.stats.json already exist for
a given run_tag, the run is skipped unless --force is given.
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
# 2026-08-24). Do NOT import from that module — copy only, per the
# non-destructive/no-shared-state requirement for this side-car.
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

# Shipped pipeline's defaults, reproduced here ONLY for documentation/diffing.
# The side-car's own defaults are SIDECAR_DEFAULT_PARAMS below, and they
# deliberately differ (ITERATIONS 0 -> 5).
SHIPPED_DEFAULT_PARAMS = {
    "THREADS": 24,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 0,
    "IMPUTE": False,
    "GPROBS": True,
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

# ---------------------------------------------------------------------------
# Side-car specific configuration (new for this script)
# ---------------------------------------------------------------------------

SIDECAR_DEFAULT_PARAMS = {
    "THREADS": 24,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 5,      # Beagle 4.1's own factory default (confirmed via
                           # `java -jar beagle.jar` usage banner, 2026-08-24:
                           # "niterations=<...> (default=5)")
    "IMPUTE": False,       # matches shipped pipeline's setting by default so a
                           # single-parameter (niterations) diff is isolated;
                           # override with --impute to test imputation too.
    "GPROBS": True,        # per task spec: gprobs=true always.
    "INPUT_FIELD": "gl",   # matches shipped pipeline (gl=<VCF: use GL/PL field>).
                           # DIAGNOSTIC FINDING 2026-08-24: with impute=true and
                           # gl=, Beagle prints "WARNING: Imputation of
                           # ungenotyped markers will not be performed. Imputation
                           # requires the "gt=" argument and called genotypes."
                           # --input-field {gt,gtgl} lets the probe test whether
                           # gl= itself (not just niterations) is gating phased
                           # output, since the input VCF's GT field already
                           # carries called genotypes (bcftools mpileup/call),
                           # not just PL likelihoods.
}

SIDECAR_OUTPUT_ROOT = os.path.join(PATH_CONFIG["PROJECT_DIR"], "data_sidecar_phased")


def log(msg: str) -> None:
    print(f"[run_beagle_phased] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Copied logic from BeaglePipeline (run_beagle.py) — reference-panel
# resolution and the marker-overlap guard. Adapted only to use this script's
# own PATH_CONFIG/output dirs (no shared state with the original class).
# ---------------------------------------------------------------------------

def get_1000genome_reference(dataset_name: str, chromosome: str) -> str:
    """Get the appropriate 1000 Genome reference file for a chromosome.

    Copied from BeaglePipeline.get_1000genome_reference (run_beagle.py).
    """
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
    """Return whether a contig has a target marker shared with 1000 Genomes.

    Copied from BeaglePipeline.chromosome_has_reference_marker (run_beagle.py).
    Beagle 4.1 exits unsuccessfully without producing a VCF if all raw calls
    on a contig are absent from the reference panel. Query only the target
    positions from the indexed reference and require an exact
    CHROM/POS/REF/ALT match before launching Beagle.

    `scratch_dir` replaces the original's use of self.output_dirs['log_dir']
    — points at this side-car's own run directory, never the shipped tree.
    """
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
    """Merge the FORMAT/INFO fields from the original VCF onto Beagle's output.

    Copied verbatim (recipe) from BeaglePipeline.merge_vcf_fields
    (run_beagle.py). GT is NOT in the -c list, so whatever GT Beagle produced
    (phased or not) survives untouched into output_vcf — this is the same
    recipe used by the shipped pipeline and confirmed not to be the cause of
    the missing-phase bug.
    """
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


# ---------------------------------------------------------------------------
# New logic for this side-car
# ---------------------------------------------------------------------------

def resolve_paths(dataset_name: str, section_id: Optional[str], quality_filter: str,
                   chromosome: str, params: Dict, custom_input_vcf: Optional[str] = None,
                   run_label: Optional[str] = None) -> Dict[str, str]:
    """Compute all side-car input/output paths for one (dataset, section,
    chrom, params) combination. Mirrors the shipped pipeline's path layout
    under data/<...>/output_VCFs/... for the INPUT only; all OUTPUT paths are
    rooted at data_sidecar_phased/ instead of data/.

    `custom_input_vcf`, if given, overrides what actually gets fed to Beagle
    as gl=/gt=/gtgl= (Stage 2: two-pass phasing, where pass 2's input is
    pass 1's retained raw output, not the original mpileup VCF). The TRUE
    original mpileup VCF (`input_vcf` below) is still resolved and still used
    as the merge step's annotation source either way, since that is the only
    file that carries the real DP/I16/QS/... pileup fields.
    """
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

    # sample_rel looks like "data/P4_tumor/1" -- strip the leading "data/" so
    # the side-car tree reads as data_sidecar_phased/P4_tumor/1/... (a true
    # parallel of data/P4_tumor/1/...).
    if sample_rel.startswith("data/"):
        sample_rel = sample_rel[len("data/"):]

    # INPUT: the real, existing, READ-ONLY shipped pre-Beagle VCF. This is
    # the exact file run_beagle.py's process_chromosome() feeds to Beagle via
    # gl={input_vcf} (see BeaglePipeline.setup_paths -> input_vcf_dir, and
    # BeaglePipeline.process_chromosome -> input_vcf). NOT
    # all_filtered_in.vcf.gz, which is a POST-Beagle product.
    input_vcf = os.path.join(
        PATH_CONFIG["PROJECT_DIR"], "data", sample_rel,
        "output_VCFs", "mpileup_multi_bam", quality_filter,
        "merged_sorted_gt.vcf.gz",
    )

    run_tag = f"niter{params['ITERATIONS']}_impute{'T' if params['IMPUTE'] else 'F'}"
    input_field = params.get('INPUT_FIELD', 'gl')
    if input_field != 'gl':
        # Only suffix the tag for non-default input fields, so the three
        # primary gl= runs already on disk (niter5_imputeF, niter10_imputeF,
        # niter5_imputeT) keep their existing directories/idempotency.
        run_tag = f"{run_tag}_{input_field}"
    if run_label:
        run_tag = f"{run_tag}_{run_label}"
    run_dir = os.path.join(
        SIDECAR_OUTPUT_ROOT, sample_rel, chromosome, quality_filter, run_tag
    )

    beagle_input_vcf = custom_input_vcf if custom_input_vcf else input_vcf

    return {
        "dataset_config": dataset_config,
        "sample_rel": sample_rel,
        "input_vcf": input_vcf,
        "beagle_input_vcf": beagle_input_vcf,
        "is_custom_input": bool(custom_input_vcf),
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
    """Invoke Beagle 4.1 directly (structurally identical to
    BeaglePipeline.run_beagle_command in run_beagle.py, with two changes:
    niterations/impute are caller-supplied, and the raw output is never
    deleted -- the caller is responsible for retention). Returns wall time in
    seconds for the Beagle subprocess itself.
    """
    reference_panel = get_1000genome_reference(dataset_name, chromosome)
    input_field = params.get('INPUT_FIELD', 'gl')

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


# ---------------------------------------------------------------------------
# Phasing statistics — new for this side-car
# ---------------------------------------------------------------------------

_AR2_RE = re.compile(r'(?:^|;)AR2=([^;]+)')
_DR2_RE = re.compile(r'(?:^|;)DR2=([^;]+)')


def compute_phasing_stats(vcf_path: str) -> Dict:
    """Compute phasing / imputation-quality metrics directly from a
    (possibly bgzipped) VCF. Assumes a single sample column, which holds for
    every VCF in this pipeline (one merged-multi-bam sample per section).

    Returns a dict with: n_records, n_gt_present, n_phased, pct_gt_phased,
    n_het, n_het_phased, pct_het_phased, ar2_mean/median/n, dr2_mean/median/n,
    has_ps_tag (declared in header), n_samples, sample_names.
    """
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

            # Single-sample assumption (col 9, 0-indexed -> fields[9]).
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
                # single-sample file: only one sample column present, so stop
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_one(dataset_name: str, section_id: Optional[str], quality_filter: str,
            chromosome: str, params: Dict, force: bool = False,
            custom_input_vcf: Optional[str] = None, run_label: Optional[str] = None) -> Dict:
    """Run (or reuse, if idempotent-complete and not --force) one Beagle
    side-car configuration on one chromosome of one sample. Returns the
    combined run_meta + stats dict, and always leaves <chrom>.stats.json and
    run_meta.json written to disk under the side-car tree.

    `custom_input_vcf`/`run_label`: Stage-2 two-pass support. When given,
    Beagle is fed `custom_input_vcf` (e.g. a prior pass's retained
    beagle_raw.vcf.gz) instead of the resolved original mpileup VCF; the
    original mpileup VCF is still resolved and still used as the merge step's
    annotation source (see resolve_paths docstring).
    """
    paths = resolve_paths(dataset_name, section_id, quality_filter, chromosome, params,
                           custom_input_vcf=custom_input_vcf, run_label=run_label)
    run_dir = paths["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    if not os.path.exists(paths["input_vcf"]):
        raise FileNotFoundError(
            f"Input VCF not found (this is the shipped pre-Beagle mpileup VCF, "
            f"read-only, produced by scripts/1_calling/mpileup_pipeline.py via "
            f"run_beagle.py's own input_vcf_dir convention): {paths['input_vcf']}"
        )
    if not os.path.exists(paths["beagle_input_vcf"]):
        raise FileNotFoundError(
            f"Beagle input VCF not found: {paths['beagle_input_vcf']}"
            + (" (--custom-input-vcf path)" if paths["is_custom_input"] else "")
        )

    already_done = (
        os.path.exists(paths["beagle_raw_vcf"])
        and os.path.exists(paths["stats_json"])
        and not force
    )
    if already_done:
        log(f"[{paths['run_tag']}] output already present at {run_dir} — skipping "
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
            f"({raw_marker_count} raw calls) — Beagle would exit without output. "
            f"Cannot probe phasing on this chromosome/sample as configured."
        )
    log(f"[{paths['run_tag']}] {raw_marker_count} raw calls on {chromosome}; "
        f"reference-marker overlap confirmed ({marker_check_seconds:.1f}s check).")

    log(f"[{paths['run_tag']}] launching Beagle: niterations={params['ITERATIONS']} "
        f"impute={params['IMPUTE']} gprobs={params['GPROBS']} modelscale={params['MODEL_SCALE']} "
        f"input_field={params.get('INPUT_FIELD', 'gl')} "
        f"beagle_input={paths['beagle_input_vcf']}{' (CUSTOM/two-pass)' if paths['is_custom_input'] else ''}")
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

    # Index the raw output (Beagle does not tabix its own output).
    subprocess.run(['tabix', '-f', '-p', 'vcf', paths["beagle_raw_vcf"]], check=True)

    # Parity artifact: same field-merge recipe as the shipped pipeline, for
    # anyone who wants to see what the eventual pipeline product would look
    # like. Not required to answer the phasing question (that's read directly
    # off beagle_raw_vcf, which is untouched by this step: GT/DS/GP/AR2/DR2
    # are not in the merge's -c list). Always annotate from the TRUE original
    # mpileup VCF (paths["input_vcf"]), never from a custom/two-pass Beagle
    # input, since only the true original carries real DP/I16/QS/... fields.
    # Best-effort: never let this parity step fail the whole run -- stats are
    # computed from beagle_raw_vcf regardless of whether this succeeds.
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
            f"continuing -- stats are computed from beagle_raw_vcf, not the merged file.")

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
        "is_custom_input": paths["is_custom_input"],
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
    parser = argparse.ArgumentParser(
        description="NON-DESTRUCTIVE side-car: run Beagle 4.1 with configurable "
                    "niterations/impute, retain raw output, write to data_sidecar_phased/. "
                    "Does not modify scripts/2_beagle_filtering/run_beagle.py or its outputs.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="P4_TUMOR")
    parser.add_argument("--section_id", default="1")
    parser.add_argument("--chrom", default="chr10", help="Single chromosome to probe (e.g. chr10).")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    parser.add_argument("--niterations", type=int, default=SIDECAR_DEFAULT_PARAMS["ITERATIONS"])
    parser.add_argument("--impute", action="store_true", default=SIDECAR_DEFAULT_PARAMS["IMPUTE"])
    parser.add_argument("--no-gprobs", dest="gprobs", action="store_false",
                         default=SIDECAR_DEFAULT_PARAMS["GPROBS"])
    parser.add_argument("--model-scale", type=float, default=SIDECAR_DEFAULT_PARAMS["MODEL_SCALE"])
    parser.add_argument("--threads", type=int, default=SIDECAR_DEFAULT_PARAMS["THREADS"])
    parser.add_argument("--memory", default=SIDECAR_DEFAULT_PARAMS["MEMORY"])
    parser.add_argument("--input-field", choices=["gl", "gt", "gtgl"],
                         default=SIDECAR_DEFAULT_PARAMS["INPUT_FIELD"],
                         help="Which Beagle input parameter to feed the (same, unmodified) "
                              "input VCF through. Shipped pipeline uses gl= (GL/PL field). "
                              "gt=/gtgl= use the VCF's already-called GT field instead -- "
                              "diagnostic for whether gl= itself (not niterations) gates "
                              "phased output.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run Beagle even if this run_tag's output already exists.")
    parser.add_argument("--custom-input-vcf", default=None,
                         help="Stage-2 two-pass support: feed Beagle THIS VCF (via --input-field, "
                              "typically gt=) instead of the resolved original mpileup VCF -- e.g. "
                              "point this at a prior gl= run's retained beagle_raw.vcf.gz for the "
                              "standard 'pass 1 calls genotypes, pass 2 phases them' pattern. Must "
                              "already be bgzip+tabix indexed. The TRUE original mpileup VCF is "
                              "still used as the merge-parity step's annotation source regardless.")
    parser.add_argument("--run-label", default=None,
                         help="Extra suffix appended to run_tag (e.g. 'twopass'), to keep "
                              "--custom-input-vcf runs from colliding with the standard run_tag "
                              "for the same niterations/impute/input-field.")

    parser.add_argument("--analyze-only", action="store_true",
                         help="Skip Beagle entirely; just compute phasing stats for --vcf-path "
                              "(e.g. the shipped niterations=0 output) using the same "
                              "compute_phasing_stats() used for side-car runs, and print JSON.")
    parser.add_argument("--vcf-path", help="VCF to analyze when --analyze-only is given.")

    args = parser.parse_args()

    if args.analyze_only:
        if not args.vcf_path:
            parser.error("--analyze-only requires --vcf-path")
        stats = compute_phasing_stats(args.vcf_path)
        print(json.dumps(stats, indent=2))
        return

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
                      args.chrom, params, force=args.force,
                      custom_input_vcf=args.custom_input_vcf, run_label=args.run_label)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
