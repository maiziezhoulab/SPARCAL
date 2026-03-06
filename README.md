# SPARCAL

**Sp**atial v**ar**iant **cal**ling — a bioinformatics pipeline for distinguishing somatic from germline single nucleotide variants (SNVs) in spatial transcriptomics data.

SPARCAL leverages spatial distribution patterns, tumor purity profiles from [CalicoST](https://github.com/raphael-group/CalicoST), and copy number alteration (CNA) data to improve variant classification accuracy beyond traditional approaches.

---

## Repository Structure

```
SPARCAL/
├── config/
│   ├── config_loader.py            # Centralized config loader (imported by all scripts)
│   ├── configs/                    # One YAML per dataset
│   │   ├── DLPFC.yaml
│   │   ├── P4_TUMOR.yaml
│   │   ├── P6_TUMOR.yaml
│   │   └── DCIS.yaml
│   └── test_config.py
│
├── scripts/
│   ├── 1_calling/
│   │   └── mpileup_pipeline.py     # Step 1: samtools mpileup → bcftools variant calling
│   │
│   ├── 2_genotyping/
│   │   ├── run_beagle.py           # Step 2a: Beagle5.4 phasing against 1000 Genomes
│   │   └── run_beagle_genotype_shifting.py  # Step 2b: Genotype transition analysis
│   │
│   ├── 3_germline_filter/          # Germline-only steps
│   │   ├── 1_sequencing_error_model.py      # BAF/depth threshold-based error filter
│   │   ├── 2_sparcal_net.py                 # 3-class NN classifier (het/hom/no_var)
│   │   ├── 3_refilter_bam_by_snv_pool.py    # Filter per-spot BAMs to SNV positions
│   │   └── 4_germline_spatial_filter.py     # Neighbor-based spatial filtering
│   │
│   ├── 4_somatic_filter/           # Somatic-only steps
│   │   └── 1_somatic_spatial_filter.py      # CalicoST-aware dual scoring (TODO)
│   │
│   └── 5_generate_matrix/
│       └── generate_matrix.py      # Binary spot × SNV matrix (.pkl)
│
├── slurm/                          # SLURM job scripts (one per dataset)
│   ├── submit_all.sh
│   ├── run_DLPFC.sh
│   ├── run_P4_TUMOR.sh
│   ├── run_P6_TUMOR.sh
│   └── run_DCIS.sh
│
├── data/                           # Pipeline output (gitignored)
├── apps/                           # Bundled binaries (gitignored)
├── slurm_output/                   # SLURM logs (gitignored)
└── README.md
```

## Pipeline Overview

SPARCAL implements two pipelines depending on tissue type:

### Germline Pipeline (e.g., DLPFC)

```
mpileup → beagle → genotype_shifting → seq_error_model → sparcal_net
       → bam_filter → germline_spatial_filter → generate_matrix
```

All variants are first called from split BAM files, then phased against the 1000 Genomes reference. Variants present in 1kG ("defined") are kept directly as germline. De novo variants pass through the sequence error model and neural network classifier before spatial filtering.

### Somatic Pipeline (e.g., P4_TUMOR, P6_TUMOR, DCIS)

```
mpileup → beagle → bam_filter → somatic_spatial_filter → generate_matrix
```

Somatic datasets skip the germline-specific classifier steps. The somatic spatial filter integrates CalicoST tumor purity estimates and clone labels to score variants using a dual germline/somatic likelihood framework.

## Prerequisites

### Software

- **Conda environment**: `snv_caller_new`
- **Bundled tools** (in `apps/`): `samtools`, `bcftools`, `bgzip`, `tabix`, `htslib`
- **Beagle 5.4**: JAR file in `apps/`
- **Java**: JDK in `apps/`
- **Python packages**: numpy, pandas, scipy, scikit-learn, matplotlib, pysam, tqdm, scanpy (optional)

### External Data

- **Reference genomes**:
  - GRCh38-3.0.0 (`genome.fa`) for DLPFC, DCIS
  - hg19-2.1.0 (`genome.fa`) for P4_TUMOR, P6_TUMOR
- **1000 Genomes reference panels**:
  - GRCh38: `CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz`
  - hg19: `hg19_chr{chrom}.vcf.gz`
- **CalicoST output** (somatic only): tumor purity, clone labels, CNV segments

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/maiziezhoulab/SPARCAL.git
cd SPARCAL

# 2. Activate the conda environment
module load Anaconda3
source activate snv_caller_new

# 3. Run all datasets via SLURM
bash slurm/submit_all.sh

# Or run a single dataset
sbatch slurm/run_DLPFC.sh

# Or run a single section manually
python scripts/1_calling/mpileup_pipeline.py --dataset DLPFC --section_id 151507
```

## Adding a New Dataset

### Step 1: Create a YAML configuration file

Create `config/configs/YOUR_DATASET.yaml`. Use an existing config as a template.

**Germline dataset template:**

```yaml
# config/configs/MY_GERMLINE.yaml

tissue_type: germline

sections:
  ids: ["section1", "section2"]

reference:
  fasta: /path/to/reference/genome.fa
  chr_prefix: ""          # "" if chromosomes are 1,2,3...; "chr" if chr1,chr2,chr3...
  build: GRCh38           # or hg19

paths:
  project_dir: /path/to/SPARCAL/data
  apps_dir: /path/to/SPARCAL/apps

tools:
  samtools: samtools
  bcftools: bcftools
  bgzip: bgzip
  tabix: tabix

input:
  bam_base_path: /path/to/dataset/base
  bam_pattern: "{section_id}/bam_bycell/*.bam"
  multiple_bams: true

output:
  dir_pattern: "my_dataset/{section_id}"

spatial:
  dir: /path/to/spatial/dir/{section_id}/spatial
  position_file: tissue_positions_list.csv
  scale_factor_file: scalefactors_json.json
  image_file: tissue_hires_image.png
  has_header: false       # true if position CSV has a header row
  in_tissue_column: 1     # column index (0-based) for in-tissue flag; null if no flag

thousand_genomes:
  base_path: /path/to/1000Genomes/
  pattern: "reference_panel_{chrom}.vcf.gz"

beagle:
  jar: beagle.22Jul22.46e.jar
  java: java
  threads: 24
  memory: 10g
  model_scale: 2
  iterations: 0
  impute: false
  gprobs: true

quality_filter:
  base_quality: 0
  mapping_quality: 0

classifier:
  model_type: neural_network
  max_training_samples: 90000

slurm:
  account: your_account
  partition: your_partition
  qos: your_qos
  conda_env: snv_caller_new
  cpus_per_task: 30
  mem: 200GB
  time: "72:00:00"
  mail_user: your.email@example.com
```

**Somatic dataset template** — add `calicost` block, remove `classifier`:

```yaml
# config/configs/MY_TUMOR.yaml

tissue_type: somatic

sections:
  ids: ["1", "2"]

# ... same reference, paths, tools, input, output, spatial, thousand_genomes, beagle ...

calicost:
  base_dir: /path/to/CalicoST/output
  num_clones: 3
  section_folder_map:
    "1": "MY_TUMOR_sec1"
    "2": "MY_TUMOR_sec2"

# No classifier block — somatic pipeline skips SPARCAL-Net
```

### Step 2: Prepare input data

Ensure the following are ready:

1. **Split BAM files**: Per-barcode BAM files from Space Ranger, matching the `bam_pattern` in your YAML
2. **Spatial data**: `tissue_positions_list.csv`, `scalefactors_json.json`, `tissue_hires_image.png`
3. **Reference genome**: FASTA file with matching chromosome naming
4. **1000 Genomes panels**: Per-chromosome VCFs for the matching genome build
5. **(Somatic only)** CalicoST output: `loh_estimator_tumor_prop.tsv`, `clone_labels.tsv`, `cnv_seglevel.tsv`

### Step 3: Validate the configuration

```bash
python config/test_config.py
```

This verifies that all paths resolve, section IDs are valid, and templates expand correctly.

### Step 4: Create a SLURM script

Copy an existing SLURM script and modify the `DATASET`, `SECTIONS`, and step sequence:

```bash
cp slurm/run_DLPFC.sh slurm/run_MY_DATASET.sh
# Edit DATASET="MY_DATASET", SECTIONS=(...), and comment/uncomment steps
```

For germline datasets, keep all steps. For somatic datasets, use the somatic template (skip steps 2b, 3-1, 3-2).

### Step 5: Run

```bash
sbatch slurm/run_MY_DATASET.sh
```

## Configuration System

All pipeline scripts import `config/config_loader.py` which provides:

```python
from config_loader import load_config, load_all_sections, PipelineConfig

# Load config for a single section
cfg = load_config("DLPFC", "151507")

# Access resolved paths
cfg.reference_fasta    # /path/to/genome.fa
cfg.output_dir         # /path/to/SPARCAL/data/dlpfc/151507
cfg.regions            # ["1", "2", ..., "22"] or ["chr1", ...]
cfg.tool("samtools")   # /path/to/apps/samtools
cfg.is_somatic         # True/False
cfg.is_germline        # True/False

# Load all sections for a dataset
all_cfgs = load_all_sections("DLPFC")  # list of 12 PipelineConfig objects
```

The `{section_id}` placeholder in YAML paths is automatically resolved at load time.

## Output Structure

For each dataset section, the pipeline creates:

```
data/{dataset}/{section_id}/
├── output_VCFs/
│   ├── mpileup_multi_bam/          # Step 1 output
│   │   └── merged_sorted_gt.vcf.gz
│   ├── beagle/                     # Step 2 output
│   │   ├── chr1.vcf.gz ... chr22.vcf.gz
│   │   ├── all_filtered_in.vcf.gz  # 1kG variants ("defined")
│   │   └── all_filtered_out.vcf.gz # De novo variants
│   ├── SeqErrModel/                # Step 3-1 output (germline only)
│   ├── Classifier/                 # Step 3-2 output (germline only)
│   └── BAM_filtered/              # Step 3-3 / Step 5 output
├── spatial_analysis/               # Step 6 output
│   └── filtered_snvs/
├── matrix/                         # Step 7 output
│   └── {dataset}_{section}_*.pkl
├── metrics/
└── logs/
```

## Citation

If you use SPARCAL in your research, please cite:

> *Manuscript in preparation*

## License

This project is developed in the [Maizie Zhou Lab](https://maiziezhoulab.github.io/) at Vanderbilt University.