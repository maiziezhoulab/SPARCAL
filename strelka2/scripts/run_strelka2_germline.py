#!/usr/bin/env python
"""
run_strelka2_germline.py
------------------------
Germline SNV calling with Strelka2 for a single DLPFC section.

Workflow per section:
  1. Wipe any pre-existing strelka2 output directory.
  2. Merge all per-cell BAMs -> one section-level BAM (samtools merge).
     (skipped if merged BAM + index already exist)
  3. Index the merged BAM (samtools index).
  4. Configure Strelka2 germline workflow.
  5. Execute the workflow locally.

Usage:
    python run_strelka2_germline.py --section_id 151507 [--threads 8]
"""

import os
import glob
import shutil
import argparse
import subprocess
import logging
import sys

# ---[ Paths ]------------------------------------------------------------------

PROJECT_DIR  = "/data/maiziezhou_lab/leiy4/snv_calling"
SAMTOOLS     = "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools"

# Strelka configure script. The prebuilt CentOS6 binary
# (strelka-2.9.2.centos6_x86_64) is BROKEN on the RHEL9 compute nodes: its
# statically-linked old boost::filesystem::canonical() throws inside
# GetSequenceErrorCounts regardless of path form (see DEBUGGING.md Bug 5).
# We now use the bioconda strelka 2.9.10 rebuild instead. Resolution order:
#   1. $STRELKA_CONFIG env var (set by the SLURM script after `source activate strelka`)
#   2. configureStrelkaGermlineWorkflow.py found on PATH (the activated conda env)
#   3. legacy CentOS6 path (broken on compute nodes; kept only as last resort)
_LEGACY_STRELKA2_BIN = (
    "/data/maiziezhou_lab/leiy4/snv_calling/"
    "strelka-2.9.2.centos6_x86_64/bin/configureStrelkaGermlineWorkflow.py"
)

def _resolve_strelka_config():
    env_path = os.environ.get("STRELKA_CONFIG")
    if env_path and os.path.exists(env_path):
        return env_path
    on_path = shutil.which("configureStrelkaGermlineWorkflow.py") if hasattr(shutil, "which") else None
    if on_path:
        return on_path
    log.warning("Falling back to legacy CentOS6 strelka binary (known broken on "
                "RHEL9 compute nodes): %s", _LEGACY_STRELKA2_BIN)
    return _LEGACY_STRELKA2_BIN

DLPFC_BASE   = "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD"
REFERENCE_FA = "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"

# ---[ Logging ]----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---[ Helpers ]----------------------------------------------------------------

def run(cmd, step, stream=False):
    log.info("[%s] Running: %s", step, " ".join(cmd))
    if stream:
        # Don't capture output - let it flow directly to SLURM log so errors are visible
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit("Step '%s' failed (exit %d)" % (step, rc))
        return
    result = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = result.communicate()
    if result.returncode != 0:
        log.error("[%s] STDOUT:\n%s", step, stdout)
        log.error("[%s] STDERR:\n%s", step, stderr)
        sys.exit("Step '%s' failed (exit %d)" % (step, result.returncode))
    if stdout.strip():
        log.info("[%s] STDOUT:\n%s", step, stdout.strip())

def join_path(*parts):
    return os.path.join(*parts)

# ---[ Main ]-------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Strelka2 germline calling for DLPFC.")
    parser.add_argument("--section_id", required=True,
                        help="DLPFC section ID, e.g. 151507")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for Strelka2 workflow (default: 8)")
    parser.add_argument("--call_regions", default=None,
                        help="Path to bgzipped+tabix-indexed BED file for --callRegions "
                             "(restricts Strelka2 to specified regions, e.g. main chromosomes only)")
    args = parser.parse_args()

    section_id   = args.section_id
    threads      = args.threads
    call_regions = args.call_regions

    # --- Resolve reference path -----------------------------------------------
    # Use realpath to get the canonical /panfs/accrepfs.vampire/... form, which
    # boost::filesystem::canonical() in the strelka2 C++ binary can resolve.
    reference_fa_real = os.path.realpath(REFERENCE_FA)
    log.info("Reference FASTA (canonical): %s", reference_fa_real)

    # --- Directory setup ------------------------------------------------------

    section_out = join_path(PROJECT_DIR, "data", "dlpfc", section_id)
    strelka_dir = join_path(section_out, "strelka2")
    merged_bam  = join_path(section_out, section_id + "_merged.bam")
    merged_bai  = merged_bam + ".bai"

    # Wipe strelka2 subfolder if it exists
    if os.path.exists(strelka_dir):
        log.info("Removing existing strelka2 directory: %s", strelka_dir)
        shutil.rmtree(strelka_dir)

    if not os.path.exists(section_out):
        os.makedirs(section_out)

    # --- Check if merged BAM already exists -----------------------------------

    if os.path.exists(merged_bam) and os.path.exists(merged_bai):
        log.info("Merged BAM and index already exist, skipping merge+index steps.")
        log.info("  BAM : %s", merged_bam)
        log.info("  BAI : %s", merged_bai)
    else:
        # --- Step 1: Discover per-cell BAMs -----------------------------------

        bam_glob  = join_path(DLPFC_BASE, section_id, "bam_bycell", "*.bam")
        bam_files = sorted(glob.glob(bam_glob))

        if not bam_files:
            sys.exit("No BAM files found matching pattern: %s" % bam_glob)

        log.info("Found %d per-cell BAMs for section %s.", len(bam_files), section_id)

        # --- Step 2: Merge BAMs -----------------------------------------------

        log.info("Merging BAMs -> %s", merged_bam)
        merge_cmd = [SAMTOOLS, "merge", "-f", "-@", str(threads), merged_bam] + bam_files
        run(merge_cmd, "samtools merge")

        # --- Step 3: Index merged BAM -----------------------------------------
        # Note: -@ (threading) is NOT supported by samtools index in older versions.

        log.info("Indexing merged BAM: %s", merged_bam)
        run([SAMTOOLS, "index", merged_bam], "samtools index")

    # --- Step 4: Configure Strelka2 germline workflow -------------------------

    # boost::filesystem::canonical() in the strelka2 C++ binary (CentOS6 build)
    # cannot resolve panfs symlink paths (/data/maiziezhou_lab -> /panfs/...)
    # when called as a subprocess on compute nodes, but it CAN resolve the
    # canonical /panfs/accrepfs.vampire/... form directly. Use os.path.realpath()
    # here so pyflow records canonical paths in all task commands (--align-file,
    # --chrom-depth-file, output dirs), not the symlink form.
    merged_bam_real  = os.path.realpath(merged_bam)
    strelka_dir_real = os.path.realpath(strelka_dir)

    strelka_config = _resolve_strelka_config()
    log.info("Configuring Strelka2 germline workflow in: %s", strelka_dir_real)
    log.info("  configure script: %s", strelka_config)
    log.info("  BAM (canonical): %s", merged_bam_real)
    configure_cmd = [
        "python", strelka_config,
        "--bam",            merged_bam_real,
        "--referenceFasta", reference_fa_real,
        "--runDir",         strelka_dir_real,
    ]
    if call_regions:
        if not os.path.exists(call_regions):
            sys.exit("callRegions file not found: %s" % call_regions)
        configure_cmd += ["--callRegions", os.path.realpath(call_regions)]
        log.info("Restricting calls to regions in: %s", os.path.realpath(call_regions))
    run(configure_cmd, "strelka2 configure")

    # --- Step 5: Run Strelka2 workflow ----------------------------------------

    run_script = join_path(strelka_dir_real, "runWorkflow.py")
    if not os.path.exists(run_script):
        sys.exit("runWorkflow.py not found at %s -- configure step may have failed." % run_script)

    log.info("Running Strelka2 workflow with %d threads...", threads)
    run(["python", run_script, "-m", "local", "-j", str(threads)],
        "strelka2 runWorkflow", stream=True)

    # --- Done -----------------------------------------------------------------

    results_dir = join_path(strelka_dir_real, "results", "variants")
    log.info("Strelka2 germline complete for section %s.", section_id)
    log.info("Output VCFs are in: %s", results_dir)
    log.info("  genome.vcf.gz     -- all variant sites")
    log.info("  genome.S1.vcf.gz  -- per-sample calls")


if __name__ == "__main__":
    main()