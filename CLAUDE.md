# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ongoing Tasks

See [On_going.md](On_going.md) for the live task list (active SLURM jobs, expected outputs, completed work).
When the user says **"check ongoing tasks"**, read that file, run `squeue` to check job status, inspect output files, and update the file in place.

For the **paper figure design and manuscript wording** (abstract/methods/results,
variant-category definitions), see [pipeline_intro.md](pipeline_intro.md).

## Variant Categories & Naming (paper)

The spatial filter (step 7) partitions variants into **three classes by spatial
behavior**, not by assumed origin. Paper/figure names (locked 2026-06-02) and
their current code tokens:

| Paper label (color) | Code token (current) | Definition |
|---------------------|----------------------|------------|
| **germline** (blue) | `germline_defined` | present in the 1000 Genomes panel |
| **UPV** = Ubiquitous Private Variants (purple) | `germline_denovo` | absent from 1KGP **and** spatially ubiquitous/uniform; a mix of rare-germline + de-novo + early/truncal somatic (≈ germline in normal tissue) |
| **somatic** (red) | `somatic_denovo` | absent from 1KGP **and** spatially focal, tumor-clone/CNV-consistent |

The two model names: **SPARCAL** = overall method; **SparcalNet** = the NN classifier.
Code rename (`germline_defined→germline`, `germline_denovo→upv`, `somatic_denovo→somatic`)
is **deferred** — see the TODO in [On_going.md](On_going.md).

## Project Overview

SNV (Single Nucleotide Variant) calling pipeline for spatial transcriptomics data, specifically 10x Visium per-barcode BAM files. The pipeline detects germline and somatic SNVs across tissue spots and generates spot×SNV binary matrices for downstream spatial analysis.

## Environment Setup

This runs on the ACCRE HPC cluster at Vanderbilt (SLURM scheduler, account `maiziezhou_lab_phd_int`, partition `interactive`, qos `maiziezhou_lab_phd_int`)

```bash
# In Slurm script, activate the current environment
source activate snv_caller         # active name used in SLURM scripts. No need to `module load Anaconda3`
```

The `apps/` directory contains bundled binaries (samtools, bcftools, bgzip, tabix, vcftools, beagle.jar, picard.jar, java) used directly by the pipeline scripts instead of system-installed versions.

## Running the Pipeline

Each pipeline step has a numbered SLURM script under `run_slurm/{dataset}/`. The canonical dataset with working examples is `dlpfc/`. Submit with `sbatch`:

```bash
sbatch run_slurm/dlpfc/1_mpileup_pipeline.sh
sbatch run_slurm/dlpfc/2_beagle_pipeline.sh
sbatch run_slurm/dlpfc/3_beagle_genotype_shifting.sh
sbatch run_slurm/dlpfc/4_sequence_error_model.sh
sbatch run_slurm/dlpfc/5_classifier.sh
sbatch run_slurm/dlpfc/6_single_bam_snp_filter.sh
sbatch run_slurm/dlpfc/7_spatial_filter_n_matrix.sh
```

Each SLURM script calls the corresponding Python script with `--dataset` and `--section_id` arguments. The quality filter string (e.g., `baseQ0mapQ0`, `baseQ13mapQ20`) propagates through all steps and determines output subdirectory naming.

To run a single step manually:
```bash
conda activate snv_caller_new
python scripts/1_calling/mpileup_pipeline.py --dataset DLPFC --section_id 151507 --base_quality 0 --mapping_quality 0
python scripts/2_beagle_filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ0mapQ0 --section_id 151507
python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset DLPFC --section_id 151507 --quality_filter baseQ0mapQ0
python scripts/3_classifier_prep/run_sequence_error_model.py --dataset DLPFC --section_id 151507 --quality_filter baseQ0mapQ0
python scripts/4_classifier/run_supplimentary_models.py --dataset DLPFC --section_id 151507 --quality-filter baseQ0mapQ0 --model-type neural_network --max-training-samples 90000
python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset DLPFC --section_id 151507 --quality_filter baseQ0mapQ0
python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py --dataset dlpfc --section_id 151507 --quality_filter baseQ0mapQ0 --min_neighbours 1
python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151507 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs --output-name normal
```

## Standard run pipeline (canonical — use this) — updated 2026-07-02

Each study now has ONE self-contained SLURM script that runs **steps 1–7 then the
matrix step (step 8)** in a single job, with a resumable `START_STEP` argument.
Prefer these over the individual numbered `1_…`–`8_…` scripts (kept only for
one-off reruns). Submit from the project root:

```bash
sbatch run_slurm/P4_tumor/run_pipeline_P4.sh          # P4 rep1, all steps
sbatch run_slurm/P6_tumor/run_pipeline_P6.sh          # P6 rep1, all steps
sbatch run_slurm/DCIS/run_pipeline_DCIS.sh            # DCIS, array 1-2 (dcis1+dcis2)
bash   run_slurm/dlpfc/run_pipeline_DLPFC.sh          # DLPFC wrapper (submits an array of 12)

# Resume from a given step (earlier steps skipped). START_STEP is arg $1:
sbatch run_slurm/P4_tumor/run_pipeline_P4.sh 5        # resume P4 from the NN classifier
sbatch run_slurm/DCIS/run_pipeline_DCIS.sh 8          # only (re)build DCIS matrices
sbatch --array=1 run_slurm/DCIS/run_pipeline_DCIS.sh  # dcis1 only
```

Steps: `1` mpileup · `2` beagle · `3` genotype-shift · `4` seq-error ·
`5` NN classifier (`run_supplimentary_models.py`, **not** `run_sparcal_net.py` —
that has the `no_variance` label-encoder bug) · `6` single-BAM filter ·
`7` spatial filter (+ best-effort visualization) · `8` 4-class SPARCAL matrices.
Each script aborts on the first failing step (no more silent `exit 0` chaining).

**DCIS dual section-id (important):** steps 1–6 take the NUMERIC id (`1`/`2`;
their `output_dir` template `data/dcis{section_id}` formats to `data/dcis1`/
`data/dcis2`), steps 7–8 take the PREFIXED id (`dcis1`/`dcis2`; the spatial
filter's `output_base=data` joins it directly). Both resolve to the same
`data/dcis{1,2}` tree — DCIS output lives ONLY there (the old stray
`data/dcis/dcis{1,2}` and literal `data/dcis{section_id}` dirs were removed
2026-07-02).

### Matrix generation — `scripts/6_spatial_filter/generate_sparcal_matrices.py`

The canonical SPARCAL matrix builder (step 8). Replaces the fragile
`run_generate_matrix.py` / `final_snv_mat.py` path juggling that failed
repeatedly (stale `--filter-subdir`, case-sensitive `data/P6_tumor`, UPV-matrix
corruption from wrong input). It reads the step-7 per-barcode lists
(`spatial_filter_purity/{qf}/{germline,somatic}/{barcode}.txt`, which carry a
`race` column = `defined`/`denovo`) and writes **four** binary int8 spot×SNV
matrices that share one row index (union of all barcodes):

| class | definition |
|-------|------------|
| `1000G`    | germline rows with `race==defined` (present in 1000G panel) |
| `germline` | all germline rows (`1000G` + UPV; == the old `normal` matrix) |
| `normal`   | alias of `germline` (for normal-tissue datasets, e.g. DLPFC) |
| `somatic`  | all somatic rows |
| `merged`   | germline ∪ somatic |

Output name: **`{STUDY}_{section}_SPARCAL_{class}_matrix.pkl`** (model token is
always `SPARCAL`, **no** `_6` grouping) under `data/{study}/{section}/matrix/`.
Columns keyed `chrom_pos`. `--classes` picks the subset (tumor default = the four
above; DLPFC uses `--classes normal`). Run directly (fast, safe, no sbatch needed):

```bash
python scripts/6_spatial_filter/generate_sparcal_matrices.py --dataset P6_TUMOR --section_id 1
python scripts/6_spatial_filter/generate_sparcal_matrices.py --dataset DCIS --section_id dcis1
python scripts/6_spatial_filter/generate_sparcal_matrices.py --dataset DLPFC --section_id 151507 --classes normal
```

DLPFC's pipeline emits **`DLPFC_{section}_SPARCAL_normal_matrix.pkl`** (replaces
the old `bcftools_normal_6`). The clustering benchmark
(`clustering_benchmark/`, formerly "SPATIAL_SNV") was updated to match: the
`sparcal` modality is now `{caller: SPARCAL, filter: normal, grouping: ""}` and
both `SPARCAL_clustering.py` and `st_loading_utils.py` drop the grouping token
when empty. strelka2/gatk matrices keep the legacy `_{caller}_{filter}_6` name.

### DLPFC UMI dedup (whole pipeline) — step 0

DLPFC now runs through the same UMI-dedup ablation as P4/P6/DCIS. Its source has
**no possorted BAM**, only read-only per-cell `bam_bycell/*.bam` (with CB+UB
tags), so step 0 **merges** them, `umi_tools dedup --per-cell`, and splits back:

```bash
sbatch run_slurm/dlpfc/0_umidedup_split_DLPFC.sh   # array 0-11, ~5-12h/section
# -> data/dlpfc/{section}/bam_bycell_dedup/{barcode}.bam
bash run_slurm/dlpfc/run_pipeline_DLPFC.sh          # then the full pipeline
```

The DLPFC config in `mpileup_pipeline.py` and `run_filter_bams_by_snv_pools.py`
reads BAMs from `data/dlpfc/{s}/bam_bycell_dedup/` via a new optional
**`bam_base_path`** field (repoints ONLY the BAM glob; `base_path` still serves
the read-only spatial/position files). Rollback = drop `bam_base_path` and set
`bam_pattern` back to `{section_id}/bam_bycell/*.bam`. **Step 1 fails with "No BAM
files found" until step 0 has produced the deduped BAMs.**

## Architecture

### Pipeline Steps and Their Scripts

| Step | SLURM script | Python script | Input → Output |
|------|-------------|---------------|----------------|
| 0 | `0_split_bam.sh` | `scripts/0_split_bam/split_bam_dcis.sh` | whole BAM → per-barcode BAMs |
| 1 | `1_mpileup_pipeline.sh` | `scripts/1_calling/mpileup_pipeline.py` | per-barcode BAMs → `output_VCFs/mpileup_multi_bam/{quality_filter}/chr*.vcf.gz` |
| 2 | `2_beagle_pipeline.sh` | `scripts/2_beagle_filtering/run_beagle.py` | mpileup VCFs → `output_VCFs/beagle/{quality_filter}/chr*.vcf.gz` |
| 3 | `3_beagle_genotype_shifting.sh` | `scripts/2_beagle_filtering/run_beagle_genotype_shifting.py` | beagle VCFs → `all_filtered_in.vcf.gz` / `all_filtered_out.vcf.gz` |
| 4 | `4_sequence_error_model.sh` | `scripts/3_classifier_prep/run_sequence_error_model.py` | filtered VCFs → error model features |
| 5 | `5_classifier.sh` | `scripts/4_classifier/run_supplimentary_models.py` | features → classified VCFs (true variants vs artifacts) |
| 6 | `6_single_bam_snp_filter.sh` | `scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py` | classified VCFs + BAMs → `BAM_filtered/{quality_filter}/{barcode}.bam` + `snv_positions/{barcode}.txt` |
| 7 | `7_spatial_filter_n_matrix.sh` | `scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py` + `run_generate_matrix.py` | snv_positions TXTs + spatial positions → spot×SNV matrix (`.pkl`) |

### Dataset Configurations

Dataset configs (base paths, BAM patterns, reference genome, spatial file locations) are duplicated in each pipeline script's `DATASET_CONFIGS` dict. When adding a new dataset, update every script. Supported datasets:

- **DLPFC** – GRCh38 (no `chr` prefix in chromosomes), sections 151507–151510 and 151669–151676
- **P4_TUMOR / P6_TUMOR** – hg19 (`chr` prefix), Visium breast tumor data
- **DCIS** – GRCh38 (`chr` prefix), ductal carcinoma in situ
<!-- - **FFPE_VISIUM / 10X_BC_*** – GRCh38 (`chr` prefix) -->

The `chr` prefix convention differs by dataset. Chromosome naming mismatch between the BAM and reference is a common failure mode.

### Output Directory Structure

All pipeline outputs are rooted at `data/{dataset_lower}/{section_id}/`:
```
output_VCFs/
  mpileup_multi_bam/{quality_filter}/chr*.vcf.gz
  beagle/{quality_filter}/chr*.vcf.gz
  beagle/{quality_filter}/all_filtered_in.vcf.gz
  BAM_filtered/{quality_filter}/{barcode}.bam
  BAM_filtered/{quality_filter}/snv_positions/{barcode}.txt
spatial_analysis/{quality_filter}/filtered_snvs/{barcode}.txt
spatial_analysis/{quality_filter}/all_filtered_variants.vcf.gz
{section_id}_*_matrix.pkl                  (spot × SNV binary matrix)
```

### Utility Tools (`scripts/tools/`)

- `draw_vcf_statistics.py` — plot VCF feature distributions across multiple VCF files for comparison
- `visualize_snv_spatial_distribution.py` / `run_visualize_spatial_snv.sh` — plot per-SNV spatial maps
- `calculate_overlap_vcfs.py` — compute overlap between VCF sets
- `Beagle_1kG_validation.py` — validate genotype calls against 1000 Genomes

Convert chromosome naming (integer → `chr` prefix):
```bash
zcat file.vcf.gz | awk 'BEGIN{OFS="\t"} /^#/{print; next} {$1="chr"$1; print}' | \
  bgzip > file_chr.vcf.gz && tabix -p vcf file_chr.vcf.gz
```

## Strelka2 Germline Calling (benchmark caller)

Strelka2 is run as an independent germline caller on the **merged per-section BAM** to benchmark
against the in-house pipeline. Everything lives under `strelka2/`.

**IMPORTANT — do not use the prebuilt binary.** `strelka-2.9.2.centos6_x86_64/` is broken against
the current ACCRE OS: `GetSequenceErrorCounts` dies with `can't resolve reference path` (old static
`boost::filesystem::canonical`). It fails even on the login node on a 5 KB demo file — it is NOT a
path/symlink issue and NOT compute-node-specific. Use the bioconda rebuild instead. Full history:
`strelka2/DEBUGGING.md` (2026-06-01 "RESOLVED" entry).

### One-time setup

```bash
# Creates conda env `strelka` (bioconda strelka 2.9.10) and re-applies the SMTP-timeout
# patch to the env's pyflow.py/makeRunScript.py (else runWorkflow.py hangs on compute nodes).
bash strelka2/scripts/install_strelka_conda.sh
```

### Run germline calling (all 12 DLPFC sections)

```bash
cd /data/maiziezhou_lab/leiy4/snv_calling/strelka2
sbatch --array=0-11 run_slurm/strelka2_germline_dlpfc.sh    # ~11 min/section, ~6.4 GB RSS
```
The SLURM script activates env `strelka`, exports `STRELKA_CONFIG`, and calls
`scripts/run_strelka2_germline.py` (which resolves the configure script via `$STRELKA_CONFIG` →
PATH → legacy fallback). Merged BAMs `data/dlpfc/{section}/{section}_merged.bam` are reused if
present (no re-merge). Output: `data/dlpfc/{section}/strelka2/results/variants/{genome,variants}.vcf.gz`.

Validate all sections before downstream use:
```bash
bash strelka2/scripts/validate_strelka2_outputs.sh   # integrity + record/PASS/SNV counts per section
```

### Generate the strelka2 spot×SNV matrix (for comparison)

Strelka2's merged-BAM VCF has **no per-spot resolution**, so it cannot feed `run_generate_matrix.py`
directly. The projection scans the merged BAM (which carries `CB:Z:` tags) at strelka2 PASS-SNV
positions and writes one `<barcode>.txt` per in-tissue spot — allele-aware (a spot "has" the SNV
if ≥ `--min-alt-reads` reads carry the ALT base), PASS SNVs only. This is the same CB-tag scan as
`generate_original_snp_profile.py`. The matrix is then built with the canonical builder so it is
directly comparable to the pipeline matrices (binary int8, rows=barcodes, cols=`chrom_pos`).

```bash
cd /data/maiziezhou_lab/leiy4/snv_calling
sbatch --array=0-11 scripts/tools/strelka2_spot_matrix_dlpfc.sh    # ~35–50 min/section, ~6–7 GB RSS
```
Per section this runs two steps:
```bash
# 1. project strelka2 SNVs onto spots  (set MIN_ALT_READS in the .sh; default 1)
python scripts/tools/strelka2_to_spot_snvs.py --section_id 151507 --min-alt-reads 1 --max-workers 22
#    -> data/dlpfc/151507/strelka2/spot_snvs/{barcode}.txt
# 2. build the matrix
python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151507 \
    --quality-filter baseQ0mapQ0 --input-dir data/dlpfc/151507/strelka2/spot_snvs \
    --caller strelka2 --output-name germline
#    -> data/dlpfc/151507/matrix/DLPFC_151507_strelka2_germline_6_matrix.pkl
```
Run under env `snv_caller` (has pysam + tqdm). Note: `conda run -n snv_caller` is flaky here —
use `source activate snv_caller` (the SLURM script does this).

### Generate the GATK spot×SNV matrix (for comparison)

GATK (run by collaborators, source: `/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{section}/gatk/output_VCFs/`)
already produces **one VCF per barcode**, so no BAM scan is needed — just parse each VCF.

GATK filter levels (cascade): `unfiltered/0` (raw, ~561k SNVs/section — avoid) →
`filtered_by_1000Genome/0` (drops 1000 Genomes panel variants; **default**) →
`filtered_by_1000Genome_by_neighbor_1/0` (also requires the variant in ≥1 of the 6 neighbor spots;
~18× smaller). The trailing `0`/`6` subdir: only `0` is populated. Set the level via `GATK_SUBDIR`
in the SLURM script.

```bash
cd /data/maiziezhou_lab/leiy4/snv_calling
sbatch --array=0-11 scripts/tools/gatk_spot_matrix_dlpfc.sh    # ~2–3 min/section, 4 CPU / 24 GB
```
Per section this runs two steps:
```bash
# 1. per-barcode GATK VCF -> per-spot SNV .txt  (in-tissue only, SNV-only, GT!=0/0)
python scripts/tools/gatk_to_spot_snvs.py --section_id 151507 --gatk-subdir filtered_by_1000Genome/0
#    -> data/dlpfc/151507/gatk/spot_snvs/{barcode}.txt
# 2. build the matrix
python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151507 \
    --quality-filter baseQ0mapQ0 --input-dir data/dlpfc/151507/gatk/spot_snvs \
    --caller gatk --output-name germline
#    -> data/dlpfc/151507/matrix/DLPFC_151507_gatk_germline_6_matrix.pkl
```

Perf note: `run_generate_matrix.py:create_snv_matrix` was fixed (2026-06-01) to use an O(1)
`{snv: j}` dict instead of `snv_list.index()` (O(N)) — essential at GATK's column counts; build
~7 s instead of hours. Output identical.

### Benchmark matrices — three callers (comparison inputs)

All three are spot×SNV binary `int8` matrices, **rows = in-tissue barcodes** (same set per section),
**cols = `chrom_pos`** — directly comparable. All 12 DLPFC sections done (`baseQ0mapQ0`), one `.pkl`
each under `data/dlpfc/{section}/matrix/`:

| Caller | Matrix file | 151507 shape | Generated by | Filtering |
|--------|-------------|--------------|--------------|-----------|
| **Pipeline** (in-house / SPARCAL) | `DLPFC_{section}_SPARCAL_normal_matrix.pkl` | 4226 × 49,602 | `run_slurm/dlpfc/run_pipeline_DLPFC.sh` (step 8) | full pipeline (beagle + classifier + spatial) |
| **Strelka2** | `DLPFC_{section}_strelka2_germline_6_matrix.pkl` | 4226 × 58,979 | `scripts/tools/strelka2_spot_matrix_dlpfc.sh` | strelka PASS SNVs, allele-aware ≥1 ALT read |
| **GATK** | `DLPFC_{section}_gatk_germline_6_matrix.pkl` | 4226 × 51,553 | `scripts/tools/gatk_spot_matrix_dlpfc.sh` | 1000G-filtered (no neighbor) |

**Pipeline matrix provenance (important):** the in-house matrix is produced by
`rerun_stage7b_DLPFC.sh`, which runs `run_generate_matrix.py --input-dir
data/dlpfc/{section}/spatial_filter_purity/{quality_filter}/germline --output-name normal`.
Do NOT regenerate it via `run_pipeline_DLPFC.sh` stage 7b — that uses `--filter-subdir
filtered_snvs`, but `spatial_analysis/{qf}/filtered_snvs/` holds no per-barcode `.txt` (only a
merged VCF), so it produces nothing. The per-spot `.txt` inputs live in
`spatial_filter_purity/{qf}/germline/` (4,227 files for 151507).

Row note: pipeline has 4,227 spots vs 4,226 for strelka2/GATK; all three share the same 4,226
in-tissue barcodes (pipeline has 1 extra). Intersect to the shared spot set before comparing.

Caveat: filtering differs per caller — only GATK has the 1000 Genomes filter applied; strelka2 and
the pipeline still include common SNPs. Account for this when comparing.

**Clustering comparison — IMPLEMENTED** as the post-processing step in `SPATIAL_SNV/`
(see "Clustering benchmark" under the SPATIAL_SNV section below). Clusters each caller's
matrix (+ a gene-expression baseline) with STAGATE→mclust and scores spatial domains vs
DLPFC layers (ARI). First full section (151507): sparcal 0.304 > gatk 0.184 > strelka2 0.128.
Still TODO: caller-vs-caller ARI/NMI and SNV-column Jaccard.

## External benchmark tools — SpatialSNV & SpaceTracer (in progress, 2026-06-13)

Two published spatial-SNV callers are being benchmarked against SPARCAL on **P4, P6,
DCIS1, DCIS2**. Each lives in its own subdir with a dedicated `CLAUDE.md` (and, for
SpatialSNV, an `On_going.md` job tracker) — **refer to those for full setup, gotchas,
resources, and live status.**

- **SpatialSNV** (`SpatialSNV/`, env `spatialsnv`) — **FUNCTIONAL.** Mutect2-based caller
  (github.com/YoungLi88/SpatialSNV). chr22 smoke test passed; **4 whole-genome runs
  launched 2026-06-13** (jobs 11869087 dcis1 / 11869088 dcis2 / 11869090 p4 / 11869091 p6).
  Outputs = 10x MTX triplets (alt/depth/ref, SNV×spot). A binarize+transpose adapter to the
  spots×SNVs benchmark contract is the remaining step. Details: `SpatialSNV/CLAUDE.md`,
  status: `SpatialSNV/On_going.md`.
- **SpaceTracer** (`/data/maiziezhou_lab/leiy4/SpaceTracer/`, **outside** this repo) —
  **TIME-BOXED, THEN CITE-ONLY (decision 2026-07-06).** Snakemake-based; env + resources +
  configs are built and it runs *partway*, but ~2 weeks stagnated on a steady stream of
  brittle-assumption bugs (`get_features`, `get_ASE.py` biallelic allele-split, hg19
  gnomAD/dbSNP build mismatch wedging P6, downstream `identifier` KeyError). Plan: **one
  time-boxed last attempt on a single sample (DCIS)**; if it doesn't complete end-to-end,
  **leave as-is and cite the preprint only** (bioRxiv + Nature submission). It's an
  *unaccepted preprint*, so the "why not compare" reviewer risk is weak/pushable — cover it
  with an honest Methods/Limitations sentence. Do NOT resume open-ended debugging. Panel that
  stands without it: Strelka2, GATK, Monopogen, SpatialSNV. (A second, partial v2.0.0-CLI
  clone also exists at `snv_calling/SpaceTracer/` — pick one.) Full rationale + status:
  `/data/maiziezhou_lab/leiy4/SpaceTracer/CLAUDE.md` (top DECISION block).

## Benchmark Results

### Original SNP Profile (`scripts/tools/generate_original_snp_profile.py`)

Pre-filter per-spot SNV counts from pysam CB-tag BAM scan (`baseQ0mapQ0`, run 2026-05-08).
All datasets use single-sample merged VCFs (mpileup on merged SpaceRanger BAM); the BAM scan
is the only valid method — VCF split produces 0 barcode matches.

| Dataset | Section | In-tissue spots | Spots matched | Total SNVs | Median / spot | Max / spot | Scan time |
|---------|---------|----------------|---------------|------------|---------------|------------|-----------|
| P4_TUMOR | 1 | 744 | 743 | 2,889,030 | 31 | 16,364 | ~13 min |
| P6_TUMOR | 1 | 3,650 | 3,650 | 5,485,324 | 555 | 94,615 | ~13 min |
| DCIS | 1 | 1,454 | 1,454 | 12,915,992 | 90 | 117,655 | — |
| DCIS | 2 | 1,807 | 1,807 | 15,329,749 | 599 | 150,526 | — |

Note: DCIS BAM uses bare contig names (`1`, `2`, …) while the merged VCF has `chr`-prefixed names.
`_build_chrom_map()` in the script auto-detects and remaps this mismatch at runtime.

Visualization outputs: `data/{dataset_lower}/{section_id}/original_snp_profile/plots/spot_snv_counts.png`
Per-barcode counts CSV: `data/{dataset_lower}/{section_id}/original_snp_profile/counts.csv`

## Known Issues

- **`run_supplimentary_models.py`** (`scripts/4_classifier/`): `REFERENCE_CONFIGS` uses key `"CHR_PREFIX"` instead of `"FFPE_VISIUM"` — marked `# BUGS HERE!!!` in the file.
- **BAF=0 for high-depth variants (`parse_i16` bug, FIXED 2026-06-02 for future runs).**
  `scripts/1_calling/mpileup_pipeline.py:parse_i16` used `int(x)` over all 16 I16
  values; at high depth bcftools writes later entries in scientific notation
  (`1.38392e+07`), `int()` throws, the `except` silently returned `[0]*16` → the
  `BAF` FORMAT field = 0 (and a false `DiscordantBAF` flag) for every high-depth
  site (~0.1% of variants, but the best-covered ones). Fixed to `int(float(x))`.
  **Existing `merged_sorted_gt.vcf.gz` (all datasets) still carry the wrong BAF**
  until step 1 is re-run; the seq-error model consumed it. Same latent bug in
  `mpileup_pipeline_old.py` and `all_caller_pipeline.py` (not fixed). The 7c
  sub-filter already recomputes BAF from I16, so it is unaffected.
- Hardcoded absolute paths throughout (`/data/maiziezhou_lab/leiy4/snv_calling`, `/data/maiziezhou_lab/Softwares/`, `/lfs/archer.accre.vu/...`). When migrating, update `PATH_CONFIG`, `REFERENCE_CONFIGS`, and `DATASET_CONFIGS` at the top of each pipeline script.
- The `readme.md` tracks a prior path migration (`/lio/lfs` → `lfs/archer.accre.vu`, `yuqi` → `leiy4`) — search for residual old paths if scripts fail with file-not-found errors.
- Steps 3 and 4 may need dataset-specific adaptation (see `readme.md` TODO).

## Planned Method Additions (TODO — not yet implemented)

Tracked in [On_going.md](On_going.md); design + paper wording in [pipeline_intro.md](pipeline_intro.md) §7–§8.

- **BAF-GMM sub-filter inside the UPV set — step 7c DRAFTED (2026-06-02).**
  `scripts/6_spatial_filter/upv_baf_gmm_subfilter.py`; runners
  `run_slurm/{DCIS,P4_tumor,P6_tumor}/7c_upv_baf_gmm.sh`; also wired into the
  enhanced filter via `--run_baf_gmm_subfilter`. Non-destructive: joins UPV
  `germline_denovo.vcf.gz` (PURITY_CORR) with `merged_sorted_gt.vcf.gz`, **recomputes
  BAF = alt/(ref+alt) from I16** (the FORMAT BAF field is buggy — see Known Issues),
  fits a 2-D GMM on `[BAF, PURITY_CORR]` + a **hard BAF ceiling**
  (`somatic ⇔ GMM-somatic-posterior>0.5 AND BAF<0.35`, `--somatic-baf-max`, to keep
  germline-het BAF≈0.5 out of the somatic set), emits `germline/denovo/gmm_subfilter/`
  (two sub-class VCFs + TSV + plot + summary). Ran all 4: dcis1 739 somatic-cand
  (BAF 0.11–0.35), dcis2/P4/P6 → 0. **Caveat:** PURITY_CORR is empirically uninformative within UPV
  (clipped ≥0 and flat because UPV are ubiquitous) → effectively BAF-only, and
  dcis1's low-BAF mode is likely ASE-skewed germline het, not somatic. Exploratory,
  not a reportable result. **Planned upgrade:** replace PURITY_CORR with a per-clone
  BAF contrast (ΔBAF tumor−normal spots) from `spotprofiles/{qf}/vcf_by_spot/` +
  CalicoST `clone_labels.tsv`. See On_going.md. Eventually fold into Stage 1 of
  `run_spatial_snv_filter_enhanced.py`. (Seq-error model already removes BAF<0.05.)
- **CHIP rule-out on the final somatic set** — drop variants in common CHIP genes
  (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, …) as a **post-processing** filter on
  the somatic VCF (gene-list intersection), not in the core cascade.
- **Category code rename** (deferred): `germline_defined→germline`,
  `germline_denovo→upv`, `somatic_denovo→somatic` in `run_spatial_snv_filter_enhanced.py`
  + `final_snv_mat.py` (keep the 3-way merged output).

## SPATIAL_SNV — SNV-based spatial clustering (active benchmark project)

`SPATIAL_SNV/` at `/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SPATIAL_SNV/`
is the **active project** that adapts the STAGATE graph-attention autoencoder to use binary
spot×SNV matrices as input instead of gene expression. This is the benchmark comparison tool:
run STAGATE clustering on each caller's matrix (SPARCAL / Strelka2 / GATK), compare the
resulting spatial domains against DLPFC ground-truth layer labels (ARI), and compare callers
against each other.

**Environment:** `snv_clustering` (conda, created 2026-06-09). Contains torch 2.2.2+cu121,
torch_geometric 2.5.2, torch_sparse 0.6.18+pt22cu121, scanpy 1.11.4, rpy2 3.6.4, R mclust 6.1.2.
`source activate snv_clustering` in SLURM scripts.

### Key files

| File | Purpose |
|------|---------|
| `classificationSNP.ipynb` | **Main SNV notebook** — loads pkl matrices, runs STAGATE, clusters with mclust, reports ARI |
| `st_loading_utils.py` | All data-loading functions; `load_DLPFC_SNV_from_Original` is the SNV entry point |
| `classification_codepart.py` | Script extraction of the gene-expression STAGATE workflow (reference baseline only — uses `load_DLPFC`, NOT SNV matrices) |
| `classification.ipynb` | Original gene-expression STAGATE (reference baseline) |
| `snvdata_analysis.ipynb` | SNV heatmaps, Jaccard spot-similarity, characteristic-SNV selection |

### Data loading — `load_DLPFC_SNV_from_Original`

```python
from st_loading_utils import load_DLPFC_SNV_from_Original

adata_dict = load_DLPFC_SNV_from_Original(
    root_dir  = '/data/maiziezhou_lab/leiy4/DLPFC12',   # SpaceRanger folders (for images + GT labels)
    section_list = ['151507', '151508', ...],
    snv_root  = '<flat dir containing pkl files>',       # see path note below
    caller    = 'bcftools',   # or 'strelka2' / 'gatk'
    filter    = 'normal',     # or 'germline'
    grouping  = '6',
)
```

The function constructs: `{snv_root}/DLPFC_{section_id}_{caller}_{filter}_{grouping}_matrix.pkl`

**Path mismatch — must resolve before running:** our matrices live at
`data/dlpfc/{section_id}/matrix/DLPFC_{section_id}_{caller}_{filter}_{grouping}_matrix.pkl`
(per-section subdirs), but the function expects a single flat directory. Fix options:
- (A) Pass `snv_root` as a per-section template and update the function (one-line change to
  `snv_file_path = os.path.join(snv_root, section_id, 'matrix', f'DLPFC_...')`).
- (B) Create a flat symlink directory pointing all pkl files into one place.

**Correct `caller/filter/grouping` for each matrix:**

| Matrix | caller | filter | grouping |
|--------|--------|--------|----------|
| SPARCAL (pipeline) | `SPARCAL` | `normal` | `` (empty — no grouping token) |
| Strelka2 | `strelka2` | `germline` | `6` |
| GATK | `gatk` | `germline` | `6` |

The function returns a dict `{section_id: AnnData}` where each AnnData has:
- `X` — sparse CSR binary matrix (spots × SNVs)
- `obs` — barcode metadata incl. `original_clusters` (ground-truth layer label)
- `obsm['spatial']` — pixel coordinates for `sc.pl.spatial`
- `uns['spatial']` — tissue images (copied from SpaceRanger h5)
- `var` — SNV metadata: `snp_id`, `chromosome`, `position`

**Data locations (ACCRE):**
- Gene expression + GT labels + images: `/data/maiziezhou_lab/leiy4/DLPFC12/{section_id}/`
  - GT file: `gt/tissue_positions_list_GTs.txt` (col 6 = layer label 1–7)
  - h5: `{section_id}_filtered_feature_bc_matrix.h5`
- SNV pkl matrices: `data/dlpfc/{section_id}/matrix/DLPFC_{section_id}_{caller}_{filter}_{grouping}_matrix.pkl`
  - All 36 files (12 sections × 3 callers) exist as of 2026-06-09.

### Pipeline inside the notebook

```
load pkl matrix → AnnData (spots × SNVs, binary)
  ↓ sc.pp.normalize_total + sc.pp.log1p   (or binning / characteristic-SNV selection)
  ↓ Cal_Spatial_Net(rad_cutoff=150)        → Spatial_Net in adata.uns
  ↓ train_STAGATE / train_STAGATE_select  → adata.obsm['STAGATE'] (30-d embedding)
  ↓ sc.pp.neighbors + sc.tl.umap
  ↓ mclust_R(num_cluster=7)               → adata.obs['mclust']
  ↓ ARI vs adata.obs['original_clusters']
```

Three preprocessing choices in `classificationSNP.ipynb`:
1. **Full matrix** — normalize + log (same as gene expression pipeline)
2. **Binned** — `bin_snvs_and_count_frequencies(adata, bin_size=1_000_000)` → 1 Mb bins
3. **Characteristic SNVs** — `identify_characteristic_snvs(threshold=0.2, p_value=0.01)`
   (Fisher's exact test per SNV; keeps only discriminative variants)

### Known issues & applied fixes

- **`reveal_importance()` crash** — `STAGATE_select.reveal_importance()` returns
  `self.adaptive_weights` which is commented out in `__init__`. `SPARCAL_clustering.py`
  uses `train_STAGATE` (unsupervised) to avoid this entirely.
- **Stale hardcoded paths** — `snv_root='/home/leiy4/DISCGATE/st_snv'` and
  `root_dir='/home/leiy4/DLPFC12'` in notebook cells; `SPARCAL_clustering.py` reads
  from `clustering_config.json` instead.
- **`gat_conv.py` PyG 2.x fix (2026-06-09)** — removed `NoneType` from
  `torch_geometric.typing` import; added `NoneType = type(None)` locally.
- **`utils.py` Transfer_pytorch_Data fix (2026-06-09)** — `torch.Tensor(lbl)` crashed
  when `original_clusters` is a string Series; fixed to LabelEncode before tensor
  construction (same pattern as `classification_codepart.py`).
- **NumPy cross-version pkl shim** — pkl matrices were serialized with NumPy 2.x
  (`snv_caller` env); `snv_clustering` uses NumPy 1.24.4 (required by torch 2.2.2).
  `SPARCAL_clustering.py` registers `numpy._core` as an alias for `numpy.core` before
  unpickling so that torch and pickle can co-exist.
- **`STAGATE_pyG` not installable** — `setup.py` references a missing `README.rst`;
  import via `PYTHONPATH` (set in `run_clustering.slurm`) instead.

### Clustering benchmark — IMPLEMENTED (post-processing step, 2026-06-09)

The notebook workflow above is now a runnable post-processing pipeline. It clusters each
caller's matrix with STAGATE→mclust and scores spatial domains vs DLPFC layers (ARI).
Live status/results: [On_going.md](On_going.md) "SPATIAL_SNV benchmark clustering".

Scripts in `SPATIAL_SNV/` (env `snv_clustering`):
- `SPARCAL_clustering.py` — per section×modality: load matrix → `Cal_Spatial_Net` →
  `train_STAGATE` (GPU auto) → `mclust_R` → ARI. Saves embedding.npy, cluster_labels.csv,
  ari.txt, and `umap`/`spatial` plots as **PNG + editable PDF** (`pdf.fonttype=42`).
- `run_clustering.slurm` — array job (`--array=0-11`), **a6000 GPU** (`batch_gpu`/
  `maiziezhou_lab_acc`); ~5 min/section for all 4 modalities. Note: a6000 cap = **2
  concurrent** (only authorized GPU with live nodes), so 12 sections run ~2 at a time.
- `make_ari_boxplot.py` — aggregates every `summary.csv` → 4-modality ARI box plot
  (PNG+PDF) + `ari_table.csv`/`ari_matrix.csv` under `data/dlpfc/clustering_benchmark/`.
- `make_combined_figure.py --section_id <s>` — concatenated per-section figure: row 1 =
  spatial domains (GT + each modality, predicted colors Hungarian-matched to GT layers),
  row 2 = per-modality UMAP colored by true layer. PNG+PDF.

**Four modalities** in `clustering_config.json`: `sparcal` (SPARCAL/normal), `strelka2`,
`gatk`, **`gene_expr`** (Visium baseline). gene_expr has no pkl — `load_gene_expr_section`
reads the SpaceRanger `{section}_filtered_feature_bc_matrix.h5` directly with the same
`normalize_total+log1p` (no HVG) as the SNV path → identical STAGATE preprocessing across
all 4. The `gene_expr` modality is dispatched in `run_one` by `modality_cfg.caller == 'gene_expr'`.

**mclust_R note (rpy2 3.6.x):** must push the embedding into R's global env under a
`localconverter` context and call `Mclust(emb_mat, G=k, modelNames="EEE")` from an R
string. `numpy2ri.activate()` is removed (raises) and positional array passing throws a
`dimnames` error in rpy2 3.6.4 / mclust 6.1.2. (Older scripts/notebooks still use the old
calling convention and will fail the same way on this env.)

**Outputs:** `data/dlpfc/{section}/clustering/{modality}/` (per-modality), `.../clustering/summary.csv`
(per-section ARI), `.../clustering/combined_{section}.{png,pdf}`, and
`data/dlpfc/clustering_benchmark/` (cross-section boxplot + tables).

**151507 (first full run):** ARI **sparcal 0.304 > gatk 0.184 > strelka2 0.128** (gene_expr
pending the 12-section re-run). Caveat: filtering differs per caller (only GATK has the
1000G filter) — note in methods. Caller-vs-caller ARI / SNV-column Jaccard still TODO.

---

## DISCGATE_imputation (gene-expression reference — superseded by SPATIAL_SNV)

`DISCGATE_imputation/` is the original gene-expression STAGATE experiment (copied from
`yuqi@10.8.128.208:/home/yuqi/DISCGATE_imputation`). **For SNV-based clustering use
`SPATIAL_SNV/` above.** This directory is kept as the gene-expression baseline and houses
the `STAGATE_pyG/` package (shared with SPATIAL_SNV).

**Environment:** `snv_clustering` (same as SPATIAL_SNV).
Data path: `/data/maiziezhou_lab/leiy4/DLPFC12/` (update from the stale `/home/yunfei/...` in notebooks).

`classification.ipynb` runs STAGATE on gene expression (`load_DLPFC` → h5 counts).
`classification_codepart.py` is the script version of the same (gene expression, not SNV).

**⚠️ Known broken state:** `STAGATE_select.reveal_importance()` returns
`self.adaptive_weights` which is commented out — raises `AttributeError`. Model trains
fine; only the feature-importance return is broken.

`gat_conv.py` **fix applied 2026-06-09:** removed `NoneType` from `torch_geometric.typing`
import (incompatible with PyG 2.x); added `NoneType = type(None)` locally.
