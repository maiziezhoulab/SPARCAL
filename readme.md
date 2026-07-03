# SNV Calling Pipeline — Walkthrough (DCIS example)

SNV (Single Nucleotide Variant) calling for 10x Visium spatial transcriptomics.
The pipeline detects **germline** and **somatic** SNVs across tissue spots, then
produces spot × SNV matrices for downstream spatial analysis.

This README walks through the full pipeline using **DCIS** as the worked example.
All SLURM scripts live under [run_slurm/DCIS/](run_slurm/DCIS/) and are submitted
with `sbatch`. Each script activates the conda env and calls the corresponding
Python script — see [CLAUDE.md](CLAUDE.md) for the script/path reference table.
For paper-figure design and manuscript wording, see [pipeline_intro.md](pipeline_intro.md).

**Variant categories.** The spatial filter (step 7) partitions variants into three
spatially-defined classes (paper names → current code tokens):
**germline** (`germline_defined`, in 1000G), **UPV / Ubiquitous Private Variants**
(`germline_denovo`, not in 1000G but spatially ubiquitous — a germline/early-somatic
mix), and **somatic** (`somatic_denovo`, not in 1000G and spatially focal,
tumor-clone/CNV-consistent). The model is **SPARCAL** (classifier: **SparcalNet**).

```bash
source activate snv_caller        # env used by all DCIS SLURM scripts
```

DCIS specifics:
- Reference: **GRCh38** with `chr`-prefixed contigs.
- Two sections, run as a SLURM array `--array=1-2` (replicates `1`, `2`).
- Section IDs are `1`/`2` for calling steps, but resolve to `data/dcis1/`,
  `data/dcis2/` on disk for the later (spatial/matrix) steps, which use
  `dcis1`/`dcis2`.

---

## Pipeline Overview

```
0. split_bam            whole BAM ─────────────────► per-barcode BAMs
1. mpileup              per-barcode BAMs ──────────► chr*.vcf.gz  (raw calls)
2. beagle               raw VCFs ──────────────────► imputed/phased genotypes
3. genotype shifting    beagle VCFs ───────────────► all_filtered_in/out.vcf.gz
4. sequence error model filtered VCFs ─────────────► error-model features
5. neural-net classifier features ─────────────────► true variants vs artifacts
6. single-BAM filter    classified VCFs + BAMs ────► per-barcode SNV positions
7. spatial filter       SNV positions + CalicoST ──► germline/somatic spot SNVs
8. final SNV matrix     spatial categories ─────────► merged SNV × spot matrix
```

---

## Step 0 — Split BAM by barcode

[0_split_bam.sh](run_slurm/DCIS/0_split_bam.sh) — splits the SpaceRanger
`possorted_genome_bam.bam` into one BAM per cell barcode (`CB` tag), using the
filtered barcode list. Runs as `--array=1-2`.

```bash
sbatch run_slurm/DCIS/0_split_bam.sh
```

Output: `…/DCIS{N}_output/outs/split_BAM/{barcode}.bam`

---

## Step 1 — mpileup calling

[1_mpileup_pipeline.sh](run_slurm/DCIS/1_mpileup_pipeline.sh) — `samtools mpileup`
+ `bcftools` over all per-barcode BAMs (multi-sample call mode), restricted to
in-tissue spots.

```bash
sbatch run_slurm/DCIS/1_mpileup_pipeline.sh
# python scripts/1_calling/mpileup_pipeline.py --dataset DCIS --section_id 1 \
#     --base_quality 0 --mapping_quality 0 --call_mode multi --threads 30 --filter_out_tissue
```

The base/mapping-quality pair sets the **quality filter** string
(`baseQ0mapQ0`) that names the output subdirectory and propagates through every
later step.

Output: `data/dcis{N}/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/chr*.vcf.gz`

---

## Step 2 — Beagle genotyping

[2_beagle.sh](run_slurm/DCIS/2_beagle.sh) — Beagle imputation/phasing to refine
genotypes.

```bash
sbatch run_slurm/DCIS/2_beagle.sh
# python scripts/2_beagle_filtering/run_beagle.py --dataset DCIS --section_id 1 \
#     --quality-filter baseQ0mapQ0 --threads 30 --memory 200g
```

Output: `data/dcis{N}/output_VCFs/beagle/baseQ0mapQ0/chr*.vcf.gz`

---

## Step 3 — Genotype shifting

[3_genotype_shifting.sh](run_slurm/DCIS/3_genotype_shifting.sh) — compares
pre/post-Beagle genotypes, computes metrics/plots, and partitions variants.

```bash
sbatch run_slurm/DCIS/3_genotype_shifting.sh
# python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
#     --dataset DCIS --section_id 1 --quality_filter baseQ0mapQ0
```

Output (under `data/dcis{N}/output_VCFs/beagle/baseQ0mapQ0/`):
- `all_filtered_in.vcf.gz`  — variants passing filters
- `all_filtered_out.vcf.gz` — variants failing filters

---

## Step 4 — Sequence error model

[4_sequence_error_model.sh](run_slurm/DCIS/4_sequence_error_model.sh) — builds a
per-site sequencing-error model used as features for the classifier.

```bash
sbatch run_slurm/DCIS/4_sequence_error_model.sh
# python scripts/3_classifier_prep/run_sequence_error_model.py \
#     --dataset DCIS --section_id 1 --quality_filter baseQ0mapQ0
```

---

## Step 5 — Neural-network classifier

[5_neural_network.sh](run_slurm/DCIS/5_neural_network.sh) — classifies each
candidate as a true variant vs an artifact.

```bash
sbatch run_slurm/DCIS/5_neural_network.sh
# python scripts/4_classifier/run_supplimentary_models.py \
#     --dataset DCIS --section_id 1 --quality-filter baseQ0mapQ0 \
#     --model-type neural_network --max-training-samples 90000
```

---

## Step 6 — Single-BAM SNP filter

[6_single_bam_snp_filter.sh](run_slurm/DCIS/6_single_bam_snp_filter.sh) — loops
over both replicates. For each barcode BAM, keeps reads covering classified
SNVs and writes the SNV positions detected in that spot.

```bash
sbatch run_slurm/DCIS/6_single_bam_snp_filter.sh
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
#     --dataset DCIS --section-id 1 --quality-filter baseQ0mapQ0 \
#     --max-workers 30 --classifier neural_network
```

Output:
- `data/dcis{N}/output_VCFs/BAM_filtered/baseQ0mapQ0/{barcode}.bam`
- `data/dcis{N}/output_VCFs/BAM_filtered/baseQ0mapQ0/snv_positions/{barcode}.txt`
  (`chrom<TAB>pos<TAB>ref<TAB>alt`, one variant per line)

---

## Step 7 — Spatial filter (germline + somatic, CalicoST-aware)

[7_spatial_filter_n_matrix.sh](run_slurm/DCIS/7_spatial_filter_n_matrix.sh) —
the spatial step. It loops over `dcis1 dcis2` and integrates **CalicoST**
tumor-purity, clone-label, and CNV-segment files to separate germline from
somatic calls and apply spatial-neighborhood support thresholds. A
visualization pass always follows (set `VIZ_ONLY=1` to re-plot from existing
outputs without re-filtering).

```bash
sbatch run_slurm/DCIS/7_spatial_filter_n_matrix.sh
# python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
#     --dataset dcis --section_id dcis1 --quality_filter baseQ0mapQ0 \
#     --tumor_purity_file  <CalicoST>/DCIS1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \
#     --clone_labels       <CalicoST>/DCIS1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv \
#     --cnv_segments       <CalicoST>/DCIS1/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv \
#     --exclude_vcf  data/dcis1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz \
#     --kept_variants data/dcis1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz \
#     --min_expression_germline 2 --min_expression_somatic 1 \
#     --neighbor_distance 2.0 --germline_threshold 0.5 --somatic_threshold 0.2
# Then: python scripts/6_spatial_filter/visualize_spatial_filter.py \
#     --dataset dcis --section_id dcis1 --quality_filter baseQ0mapQ0
```

CalicoST inputs live under `/data/maiziezhou_lab/leiy4/CalicoST/DCIS{N}/`
(note the **upper-case** `DCIS1`/`DCIS2` directory names).

Output: per-spot SNV `.txt` files split into spatial-filter categories under
`data/dcis{N}/spatial_filter_purity/baseQ0mapQ0/{germline,somatic,…}/`, plus
visualization plots.

---

## Step 8 — Final SNV matrix

[8_final_snv_mat_dcis1.sh](run_slurm/DCIS/8_final_snv_mat_dcis1.sh) /
[8_final_snv_mat_dcis2.sh](run_slurm/DCIS/8_final_snv_mat_dcis2.sh) — merges the
three spatial-filter categories — **germline** (`germline_defined`), **UPV**
(`germline_denovo`), **somatic** (`somatic_denovo`) — into the final outputs.

```bash
sbatch run_slurm/DCIS/8_final_snv_mat_dcis1.sh
sbatch run_slurm/DCIS/8_final_snv_mat_dcis2.sh
# python scripts/postprocess/final_snv_mat.py \
#     --dataset DCIS --section_id dcis1 --quality_filter baseQ0mapQ0
```

Output root: `data/dcis{N}/final_matrices/baseQ0mapQ0/`
- `merged_snv_spot_matrix.vcf.gz` — one row per unique SNV, one sample column
  per barcode. INFO carries all pipeline features + a `CATEGORY` tag; FORMAT is
  `GT` only (`1/1` present, `0/0` absent).
- `per_barcode/{barcode}.vcf.gz` — one VCF per spot listing every SNV found in
  that barcode (union of all categories).

---

## Utility tools

VCF feature statistics across multiple VCFs:
```bash
python scripts/tools/draw_vcf_statistics.py \
    -i somatic.vcf.gz germline.vcf.gz \
    -n Somatic Germline \
    -o comparison_out
```

Convert integer contigs → `chr`-prefixed:
```bash
zcat in.vcf.gz | awk 'BEGIN{OFS="\t"} /^#/{print; next} {$1="chr"$1; print}' | \
  bgzip > in_chr.vcf.gz && tabix -p vcf in_chr.vcf.gz
```

Plot per-SNV and overall spatial distributions:
```bash
scripts/tools/run_visualize_spatial_snv.sh
```

---

## Migration notes

This project was relocated; if scripts fail with file-not-found, check for
residual old paths and update them:
- `/lio/lfs` → `lfs/archer.accre.vu`
- `yuqi` → `leiy4`

SLURM account/partition for interactive steps:
```
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
```

Known caveats:
- Steps 3 & 4 may need dataset-specific adaptation — watch for
  `"Unknown dataset format"`.
- `run_supplimentary_models.py` has a `REFERENCE_CONFIGS` key bug (`"CHR_PREFIX"`
  vs `"FFPE_VISIUM"`) marked `# BUGS HERE!!!`.
- Hardcoded absolute paths are scattered throughout; see [CLAUDE.md](CLAUDE.md)
  for the `PATH_CONFIG` / `REFERENCE_CONFIGS` / `DATASET_CONFIGS` blocks to edit
  when migrating.
</content>
</invoke>
