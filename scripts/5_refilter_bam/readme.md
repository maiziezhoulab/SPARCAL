# Summary of Changes to `run_filter_bams_by_snv_pools.py`

## Overview
Unified the output structure to organize per-spot BAM/VCF files into subdirectories while keeping merged summary files at the root level. Changed output directory from `BAM_filtered` to `spotprofiles` to preserve previous outputs.

---

## Key Changes

### 1. **Output Directory Structure (Lines 821-836)**

**BEFORE:**
```
data/{dataset}/{section_id}/output_VCFs/BAM_filtered/{quality_filter}/
├── {barcode}.bam
├── {barcode}.bam.bai
├── ...
└── (BAMs mixed with other files)

(Separate directory)
data/{dataset}/{section_id}/output_VCFs/snv_vcf/
├── {barcode}.vcf.gz
└── ...
```

**AFTER:**
```
data/{dataset}/{section_id}/output_VCFs/spotprofiles/{quality_filter}/
├── bam_by_spot/                                # Subdirectory for per-spot BAMs
│   ├── AAACAACGAATAGTTC-1.bam
│   ├── AAACAACGAATAGTTC-1.bam.bai
│   └── ...
├── vcf_by_spot/                                # Subdirectory for per-spot VCFs
│   ├── AAACAACGAATAGTTC-1.vcf.gz
│   ├── AAACAACGAATAGTTC-1.vcf.gz.tbi
│   └── ...
├── all_variants.vcf.gz                         # Merged: all input SNVs
├── all_variants.vcf.gz.tbi
├── all_detected_variants_summary.vcf.gz        # Merged: detected SNVs with stats
├── all_detected_variants_summary.vcf.gz.tbi
└── filtering_summary.txt                       # Summary report
```

---

### 2. **Modified Functions**

#### **`SNVMatrixGenerator.setup_paths()` (Lines 771-836)**
- Changed `filtered_bam_dir` from `output_VCFs/BAM_filtered` to `output_VCFs/spotprofiles`
- Added `self.bam_by_spot_dir` and `self.vcf_by_spot_dir` subdirectories
- Updated log directory from `logs/BAM_filtered` to `logs/spotprofiles`

#### **`filter_bams_parallel()` (Lines 550-693)**
- **New parameters:**
  - `vcf_output_dir`: Directory for per-spot VCF files (default: subdirectory of `output_dir`)
  - `merged_output_dir`: Directory for merged summary files (default: parent of `output_dir`)
- Updated all `snv_vcf_dir` references to use `vcf_output_dir`
- Changed `all_variants.vcf` creation to save in `merged_output_dir` instead of `output_dir`
- Updated `create_all_variants_summary()` call to pass both directories

#### **`create_all_variants_summary()` (Lines 269-350)**
- **New parameters:**
  - `merged_output_dir`: Where to save the summary file (root level)
  - `vcf_output_dir`: Where to read per-spot VCF files from (vcf_by_spot)
- Removed hardcoded `snv_vcf_dir` construction
- Now saves summary to `merged_output_dir` and reads from `vcf_output_dir`

#### **`SNVMatrixGenerator.filter_bams()` (Lines 965-1047)**
- Removed incorrect `snv_vcf_dir` path construction (was causing Bug #1)
- Updated to pass `bam_by_spot_dir`, `vcf_by_spot_dir`, and `filtered_bam_dir` to `filter_bams_parallel()`
- Enhanced output messages to clarify directory structure
- Updated summary report to document output structure

#### **`SNVMatrixGenerator.index_existing_bams()` (Lines 1049-1069)**
- Changed to index BAMs in `self.bam_by_spot_dir` instead of `self.filtered_bam_dir`

---

### 3. **Bug Fixes**

#### **Bug #1: Inconsistent `snv_vcf_dir` path**
**BEFORE (Line 990):**
```python
snv_vcf_dir = os.path.join(os.path.dirname(self.filtered_bam_dir), "snv_vcf")
# This created snv_vcf at parent of filtered_bam_dir
```

**AFTER:**
```python
# Now properly uses vcf_by_spot_dir set in setup_paths()
# and passed correctly to filter_bams_parallel()
```

#### **Bug #2: Ambiguous output documentation**
- Added comprehensive output structure documentation in usage examples (Lines 1137-1174)
- Enhanced print statements to clearly show where files are saved

#### **Bug #3: Duplicate directory creation**
- Removed duplicate `snv_vcf_dir` creation in `filter_bams()` method
- Unified directory creation in `setup_paths()` method

---

### 4. **Backward Compatibility**

**Previous outputs preserved:**
- Old outputs remain in `data/{dataset}/{section_id}/output_VCFs/BAM_filtered/`
- New outputs go to `data/{dataset}/{section_id}/output_VCFs/spotprofiles/`
- No risk of overwriting existing data

---

## Testing Recommendations

1. **Test on a small section first:**
   ```bash
   python run_filter_bams_by_snv_pools.py \
       --dataset P4_TUMOR \
       --section-id 1 \
       --quality-filter baseQ0mapQ0 \
       --classifier neural_network \
       --max-workers 5
   ```

2. **Verify directory structure:**
   ```bash
   tree data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/
   ```

3. **Check file counts:**
   ```bash
   # Count BAM files
   ls data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/bam_by_spot/*.bam | wc -l
   
   # Count VCF files
   ls data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot/*.vcf.gz | wc -l
   
   # Verify merged files exist
   ls data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/all_*.vcf.gz
   ```

4. **Review summary report:**
   ```bash
   cat data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/filtering_summary.txt
   ```

---

## Impact on Downstream Analysis

**Files to update that reference the old paths:**
- `7_spatial_filter_n_matrix.sh` or related scripts
- Any scripts that read from `BAM_filtered` directory
- Scripts that reference `snv_vcf` directory

**New paths to use:**
```python
# Per-spot BAMs
bam_dir = "data/{dataset}/{section}/output_VCFs/spotprofiles/{qf}/bam_by_spot/"

# Per-spot VCFs
vcf_dir = "data/{dataset}/{section}/output_VCFs/spotprofiles/{qf}/vcf_by_spot/"

# Merged summaries
summary_dir = "data/{dataset}/{section}/output_VCFs/spotprofiles/{qf}/"
```