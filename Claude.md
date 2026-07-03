# Project: ST-SNV Calling (Spatial Transcriptomics SNV)

Working directory: `/data/maiziezhou_lab/yuqi/snv_calling`

---

## Datasets

### DLPFC (Dorsolateral Prefrontal Cortex)
- Source: DLPFC_spatialLIBD / DLPFC12
- Sections: 151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676
- Raw BAMs: `/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section}/bam_bycell/`
- Gene expression + spatial metadata: `/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12/{section}/`
- Ground truth cluster labels: `DLPFC12/{section}/gt/tissue_positions_list_GTs.txt`
- Reference genome: GRCh38-3.0.0 (no `chr` prefix on chromosomes)

### Tumor datasets
- P4_TUMOR: `/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/`
- P6_TUMOR: `/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/`

---

## SNV Calling Pipeline (yuqi's pipeline)

All scripts live under `snv_calling/scripts/`, SLURM jobs under `snv_calling/run_slurm/dlpfc/`.
Conda env: `snv_caller_new`

### 7-Step Workflow

| Step | SLURM script | Python script | Description |
|------|-------------|---------------|-------------|
| 1 | `1_mpileup_pipeline.sh` | `scripts/1_calling/mpileup_pipeline.py` | bcftools mpileup per-spot BAM → VCF |
| 2 | `2_beagle_pipeline.sh` | `scripts/2_beagle_filtering/run_beagle.py` | Beagle imputation/filtering |
| 3 | `3_beagle_genotype_shifting.sh` | `scripts/2_beagle_filtering/run_beagle_genotype_shifting.py` | Identify shifted vs stable genotypes |
| 4 | `4_sequence_error_model.sh` | `scripts/3_classifier_prep/run_sequence_error_model.py` | Build sequence error profile |
| 5 | `5_classifier.sh` | `scripts/4_classifier/run_supplimentary_models.py` | Train NN/SVM classifiers (max 90k samples) |
| 6 | `6_single_bam_snp_filter.sh` | — | Apply classifiers to individual BAMs |
| 7 | `7_spatial_filter_n_matrix.sh` | `scripts/6_spatial_filter/run_spatial_snv_filter.py` + `run_generate_matrix.py` | Spatial neighbor filter → binary matrix |

Step 7 SLURM job loops over sections `{151508..151510} {151669..151672} {151673..151676}` — **section 151507 is absent from this loop**.

### Quality filter used for DLPFC
`baseQ0mapQ0` (base quality 0, mapping quality 0 — no quality filtering at mpileup stage).

---

## Spatial SNV Filter Design

Script: `scripts/6_spatial_filter/run_spatial_snv_filter.py`
Class: `SpatialSNVFilter`

### Core logic
An SNV in a spot is **kept only if ≥ N neighboring spots also carry the same SNV**.
Rationale: true somatic variants appear in spatially adjacent (clonally related) spots; sequencing artifacts are isolated.

### Neighbor graph construction (`build_spatial_graph`)
1. Load pixel coordinates from `tissue_positions_list.csv` (in-tissue spots only)
2. Read `spot_diameter_fullres` from `scalefactors_json.json`, multiply by **6** (safety factor)
3. Run `sklearn.NearestNeighbors` with `k = MAX_NEIGHBORS + 1 = 13`
4. Threshold: `neighbor_distance × spot_diameter = 1.5 × (spot_diameter_fullres × 6)`
5. Spots within threshold → neighbors (typically ~6 on hexagonal Visium grid)

### Key parameters (DLPFC production run)

| Parameter | Value | Source |
|-----------|-------|--------|
| `neighbor_distance` | 1.5 spot diameters | hardcoded default |
| `MAX_NEIGHBORS` | 12 | hardcoded constant |
| `spot_diameter` | `spot_diameter_fullres × 6` | `build_spatial_graph` |
| `min_neighbours` | **1** | `7_spatial_filter_n_matrix.sh` (MIN_NEIGHBOURS=1) |
| `quality_filter` | `baseQ0mapQ0` | SLURM script |

**No `--kept_variants`, `--exclude_vcf`, or `--include_vcf` flags** are used for DLPFC — pure spatial neighbor filtering only.

### Optional filter modes (used for tumor datasets, not DLPFC)
- `--exclude_vcf`: blacklist SNVs from a VCF (e.g., Beagle-filtered-out variants)
- `--include_vcf`: whitelist — keep only SNVs in a reference VCF (e.g., somatic Mutect2 calls)
- `--kept_variants`: bypass spatial filtering entirely for a specified set of variants

### Pipeline order within `run_analysis()`
1. Load spot positions
2. Build spatial graph
3. Load per-barcode SNV position `.txt` files
4. (Optional) Apply exclusion/inclusion/kept-variants filters
5. Apply spatial neighbor filter
6. Save per-barcode filtered `.txt` files + PNG visualizations + summary

---

## SNV Matrix Generation

Script: `scripts/6_spatial_filter/run_generate_matrix.py`

### What it does
- Input: filtered SNV `.txt` files from `spatial_analysis/{quality_filter}/filtered_snvs/`
- Builds a **binary matrix**: rows = barcodes/spots, columns = SNVs (`CHROM_POS` format), values = 0/1
- Output: pickle file saved to `data/dlpfc/{section_id}/matrix/`

### Output filename convention
`{DATASET}_{section_id}_{caller}_{output_name}_{grouping}_matrix.pkl`

Example variants for 151507:
- `DLPFC_151507_bcftools_normal_6_matrix.pkl` — standard spatial filter
- `DLPFC_151507_bcftools_beagle_only_6_matrix.pkl` — Beagle-only variants
- `DLPFC_151507_bcftools_denovo_only_6_matrix.pkl` — de novo variants only
- `DLPFC_151507_bcftools_normal_min1-12_6_matrix.pkl` — various neighbor minimum thresholds

---

## SNV Matrix Visualization / Comparison with Gene Expression

Script: `scripts/6_spatial_filter/plot_snv_matrix.py`

There is **no dedicated standalone comparison script** between SNV and gene expression matrices. This script handles the comparison:
- Loads SNV pickle matrix
- Loads GEX ground-truth cluster labels from `DLPFC12/{section}/gt/tissue_positions_list_GTs.txt`
- Loads spatial image/coordinates via `sc.read_visium` from the `.h5` count file
- Creates an AnnData object merging SNV data with GEX spatial/cluster annotations
- Produces: SNV heatmap sorted by GEX cortical layer clusters, spatial maps of SNV counts per spot
- Fisher's exact test to find cluster-enriched SNVs (`--find-cluster-snvs`)
- Optionally saves as `.h5ad` (`--save-anndata`)

---

## GATK Results (hanliu pipeline)

Script: `/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/main.py` (config-driven)
Unfiltered VCFs: `/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{section}/gatk/output_VCFs/unfiltered/0/`
Config files: `/data/maiziezhou_lab/hanliu/projects/snv_call/configs/DLPFC_spatialLIBD/gatk/{section}.ini`

> Note: These scripts were written by hanliu, not yuqi. They are independent of the pipeline above.

### Tool & settings
- **GATK HaplotypeCaller v4.0.3.0**
- Reference: GRCh38-3.0.0
- `--output-mode EMIT_VARIANTS_ONLY`
- `--minimum-mapping-quality 20`, `--min-base-quality-score 10`
- `--disable-tool-default-read-filters true`
- FORMAT fields: `GT:AD:DP:GQ:PL`

### Sanity check results (as of 2026-05-22)

| Section | VCF count | Notes |
|---------|-----------|-------|
| 151507, 151508, 151510, 151669–151676 | 4992 | Complete |
| **151509** | **4975** | 17 fewer barcodes |
| 151675, 151676 | 4992 VCFs | **No config `.ini` found** |

**151507 variant stats:**
- Barcodes with 0 variants: 5
- Total variants across all barcodes: 1,642,603
- Mean per barcode: 329 | Median: 288 | Max: 1707

### Config key settings (151507.ini)
- Active filter: `FILTER = filtered_by_1000Genome_by_neighbor_1`
- `NEIGHBOR_LIM = 0` → the `unfiltered/0` path
- `RAD_LIM = 150` pixels (spatial neighbor radius in main.py)
- Results path: `data/DLPFC/{section}/gatk/results/filtered_by_1000Genome_by_neighbor_1/0/`

---

## Key File Locations

| Item | Path |
|------|------|
| Pipeline scripts | `snv_calling/scripts/` |
| SLURM jobs (DLPFC) | `snv_calling/run_slurm/dlpfc/` |
| DLPFC SNV data | `snv_calling/data/dlpfc/{section}/` |
| SNV matrices (output) | `snv_calling/data/dlpfc/{section}/matrix/` |
| Spatial filter output | `snv_calling/data/dlpfc/{section}/spatial_analysis/baseQ0mapQ0/` |
| GATK VCFs (hanliu) | `/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{section}/gatk/output_VCFs/unfiltered/0/` |
| GATK configs (hanliu) | `/data/maiziezhou_lab/hanliu/projects/snv_call/configs/DLPFC_spatialLIBD/gatk/{section}.ini` |
| DLPFC12 gene expression | `/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12/{section}/` |
| DLPFC spatialLIBD BAMs | `/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section}/bam_bycell/` |
