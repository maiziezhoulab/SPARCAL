# On-going Tasks

Dynamically updated task list. When asked to "check ongoing tasks", Claude should:
1. Run `squeue -j <job_ids>` to check SLURM status
2. Check output files/logs for results
3. Update this file accordingly

---

## 2026-07-02 (later) — DLPFC: SPARCAL_normal rename + UMI dedup wired (dedup NOT yet submitted)

**DLPFC matrix renamed** to `DLPFC_{section}_SPARCAL_normal_matrix.pkl` (step 8
`generate_sparcal_matrices.py --classes normal`; `normal` = all germline). All 12
sections regenerated (151507 = 4226 × 49,602, cols match old, rows now align with
strelka2/gatk 4226). Old `bcftools_normal_6` deleted (all 12). Clustering benchmark
updated: `clustering_benchmark/clustering_config.json` sparcal modality →
`{caller: SPARCAL, filter: normal, grouping: ""}`; `SPARCAL_clustering.py` +
`st_loading_utils.py` drop the grouping token when empty (strelka2/gatk unchanged).

**DLPFC UMI dedup — SUBMITTED 2026-07-02 19:36.** DLPFC has no possorted BAM —
only read-only per-cell `bam_bycell/*.bam` (CB+UB tags). `run_slurm/dlpfc/0_umidedup_split_DLPFC.sh`
(array 0-11): merge → `umi_tools dedup --per-cell` → split by CB →
`data/dlpfc/{s}/bam_bycell_dedup/`. Pipeline reads deduped BAMs via a new optional
`bam_base_path` config field in `mpileup_pipeline.py` + `run_filter_bams_by_snv_pools.py`
(repoints only the BAM glob; `base_path` still serves read-only spatial).

- **`12303350_[0-11]` umidedup_DLPFC — PENDING** (QOSGrpCpuLimit; 16 cpu × 12 tasks
  exceeds the interactive QOS budget → runs a few at a time). This is the good job.
- **`12303351` dlpfc_pipeline — mis-submitted then CANCELLED.** The user ran the
  pipeline immediately after the dedup submit (not chained); tasks 0-3 FAILED in
  ~6 s ("No BAM files found" — deduped BAMs don't exist yet), 4-11 were still
  pending. `scancel 12303351` (also freed QOS CPU for the dedup job).

**CORRECT sequence (pipeline MUST wait for dedup):** let `12303350` finish, then
`bash run_slurm/dlpfc/run_pipeline_DLPFC.sh` — OR chain it now:
`DEP=12303350 bash run_slurm/dlpfc/run_pipeline_DLPFC.sh` (wrapper now supports a
`DEP` env var → `--dependency=afterok`). NOTE: the 12 DLPFC matrices above are
still the PRE-dedup call set — they'll be overwritten once the deduped pipeline runs.

---

## 2026-07-02 — UMI-dedup pipeline FINISHED (all 4 samples) + matrix/pipeline refactor ✅

**All four dedup samples (P4, P6, dcis1, dcis2) are complete through matrices.**
No pipeline jobs running (`squeue` clean). Steps 1–7 were done for all four on
2026-07-01 (P4 needed the step-5 refix rerun `12276786..788`; DCIS `12270329..335`).
Today the matrix step was rebuilt for all four with the new generator.

### New canonical matrix generator — `scripts/6_spatial_filter/generate_sparcal_matrices.py`
Replaces the flaky `run_generate_matrix.py`/`final_snv_mat.py` path juggling (stale
`--filter-subdir`, case-sensitive `data/P6_tumor`, UPV-matrix corruption). Reads step-7
per-barcode `spatial_filter_purity/{qf}/{germline,somatic}/{bc}.txt` (has `race` col
defined/denovo) → **4 binary int8 matrices sharing one row index**, named
**`{STUDY}_{section}_SPARCAL_{class}_matrix.pkl`** (model always `SPARCAL`, no `_6`):

| sample | rows | 1000G | germline (=1000G+UPV) | somatic | merged |
|--------|------|-------|-----------------------|---------|--------|
| P6_TUMOR 1 | 3650 | 125,269 | 127,013 | 65,655 | 192,668 |
| P4_TUMOR 1 | 744  | 42,375  | 51,866  | 19,523 | 71,389  |
| DCIS dcis1 | 1454 | 140,773 | 193,893 | 18,536 | 212,429 |
| DCIS dcis2 | 1807 | 129,764 | 166,087 | 25,154 | 191,241 |

Cross-checks hold (germline=1000G+UPV, merged=germline+somatic). Old-named matrices
(`*_bcftools_normal_6`, `*_bcftools_somatic_6`) **deleted** — superseded & reproducible.
Each sample's `matrix/` now holds exactly the 4 SPARCAL pkls.

### Unified run_pipeline scripts (steps 1–8, resumable `START_STEP` arg)
`run_slurm/{P4_tumor,P6_tumor}/run_pipeline_{P4,P6}.sh`, `run_slurm/DCIS/run_pipeline_DCIS.sh`
(`--array=1-2`), updated `run_slurm/dlpfc/run_pipeline_DLPFC.sh`. One job runs all steps +
matrix, aborts on first failure. `sbatch …/run_pipeline_DCIS.sh 8` = matrix-only rerun.
Step 5 = `run_supplimentary_models.py` (not the buggy `run_sparcal_net.py`).

### DCIS output cleansed to `data/dcis1` / `data/dcis2` only
Removed stale April artifacts `data/dcis{section_id}` (597M), `data/dcis/dcis{1,2}` (29M),
`data/DCIS` (80M). DCIS dual section-id documented: numeric (1/2) for steps 1–6, prefixed
(dcis1/dcis2) for steps 7–8, both → `data/dcis{1,2}`. See CLAUDE.md "Standard run pipeline".

**Remaining (optional):** DLPFC still uses the old `bcftools_normal_6` naming (SPATIAL_SNV
loader keyed on it) — its run_pipeline keeps both that and the new step-8 SPARCAL matrices.

---

## 2026-07-01 — Results check + P6 matrix fix + P4/DCIS dedup submission

### DLPFC benchmarking (clustering, all 12 sections) — DONE ✅
Array `12156700_0..11` all COMPLETED (2026-06-25). Aggregate outputs in
`data/dlpfc/clustering_benchmark/` (`ari_boxplot{,_by_section}.{png,pdf}`,
`ari_table.csv`, `ari_matrix_{mean,best}.csv`). **Mean ARI across 12 sections:**
gene_expr **0.404** > sparcal **0.281** > gatk **0.205** > strelka2 **0.180**.
SPARCAL is the best SNV caller in **8/12** sections; gene_expr best overall 11/12.
Confirms the 151507 pilot ordering (sparcal > gatk > strelka2). Caveat still:
only GATK has the 1000G filter.

### UMI dedup — step 0 done for 4 samples ✅
`umi_tools dedup --per-cell` + `samtools split -M 6000`, all COMPLETED:
| Sample | pre → post reads | dup removed | split BAMs |
|--------|------------------|-------------|------------|
| P6 rep1 (12133315) | 84.3M → 51.5M | ~39% | 4,992 |
| P4 rep1 (12149523) | 104.5M → 84.7M | ~19% | 4,992 |
| DCIS1 (12149524_1) | 508.4M → 329.7M | ~35% | 4,992 |
| DCIS2 (12149524_2) | 488.9M → 311.6M | ~36% | 4,992 |
Deduped `split_BAM/` in place for all; old non-deduped split_BAM backed up aside.

### P6 dedup pipeline (steps 1–7) — filtering DONE; matrix FIXED 2026-07-01 ✅
Steps 1–7 completed (job chain up to 12156808). Categorized VCF counts (deduped):
germline_defined 125,269 · UPV 1,744 · somatic_denovo 65,655 · ambiguous_denovo 590,897.
**Matrix step in job 12156808 FAILED** (`run_generate_matrix.py --filter-subdir
filtered_snvs` → `spatial_analysis/.../filtered_snvs` not found — the known stale-path bug).
**Regenerated manually 2026-07-01** from the real per-barcode dirs →
`data/P6_tumor/1/matrix/`:
- `P6_TUMOR_1_bcftools_normal_6_matrix.pkl` — **3651 × 127,013** (= germline_defined
  125,269 + UPV 1,744; **NOT** the ~1.25M raw-merged set → the old UPV-matrix corruption
  bug is GONE ✅).
- `P6_TUMOR_1_bcftools_somatic_6_matrix.pkl` — 3649 × 65,655.

**Fixes applied 2026-07-01 (so future runs don't repeat the failure):**
- `scripts/6_spatial_filter/run_generate_matrix.py:save_snv_matrix` now writes to the
  dataset's real `output_subdir` (case-sensitive FS: was writing `data/p6_tumor` ≠
  `data/P6_tumor`).
- `run_slurm/{P4_tumor,P6_tumor}/7_spatial_filter_n_matrix.sh` matrix call switched from
  `--filter-subdir filtered_snvs` to `--input-dir .../spatial_filter_purity/${QF}/germline`.

### P4 / DCIS dedup pipeline (steps 1–7) — SUBMITTED 2026-07-01

**DCIS chain — RUNNING ✅** (dependency-chained, dedup ablation, dcis1+dcis2, baseQ0mapQ0):
`12270329` mpileup (array 1-2) → `330` beagle → `331` genotype_shift → `332` seq_err →
`333` nn → `334` filter_bams → `335` spatial_filter. Step 1 confirmed doing real work
(>13 min). DCIS scripts were already correct (env `snv_caller`, `scripts/1_calling/...`).
DCIS matrices come from `8_final_snv_mat_dcis{1,2}.sh` (its step 7 doesn't call
run_generate_matrix) — run those after step 7 finishes.

**P4 chain — FIRST ATTEMPT `12270322..328` FAILED SILENTLY (exit 0, ran in seconds).**
Root cause: stale P4 scripts. Step 1 used a non-existent env `snv_caller_new` and the old
path `scripts/calling/mpileup_pipeline.py` (real: `scripts/1_calling/…`); no error-check →
exit 0 → the `afterok` chain cascaded through, steps 6/7 likely touched stale/old P4 data.
**Fixed 2026-07-01** (env `snv_caller`, correct paths):
- step 1: `snv_caller_new`→`snv_caller`, `scripts/calling/`→`scripts/1_calling/`, comment `module load`
- step 2: dropped stray `--caller Monopogen` (run_beagle.py has no `--caller`), added `--quality-filter baseQ0mapQ0`
- step 3, 4: `snv_caller_new`→`snv_caller`
- (steps 5/6/7 were already OK; ablation edits from earlier still in place)
**P4 chain — 2nd attempt `12270741..747` COMPLETED, but step-5 classifier CRASHED → 0 denovo.**
Steps 1–4 are GOOD (mpileup 2h31m fresh on deduped BAMs; beagle `all_filtered_in` 42,523 /
`all_filtered_out` 529,582; seq-err ran). **Step 5 used `run_sparcal_net.py`, which crashed:
`ValueError: y contains previously unseen labels: 'no_variance'`** (trains on het/hom only,
then hard-looks-up `no_variance` in `apply_to_vcf` → `neural_network_*.vcf.gz` all EMPTY).
Consequence: step 6 saw only the 42,523 defined (1KGP) variants; step 7 classified **100%
germline_defined, 0 UPV, 0 somatic** (pre-dedup P4 had UPV 7,215 / somatic 4,924). The P4
`normal` matrix (745 × 42,523) is therefore **germline-only / incomplete — do not use yet.**
- **Fix 2026-07-01:** P4 `5_neural_network.sh` switched from `run_sparcal_net.py` to
  `run_supplimentary_models.py --model-type neural_network --max-training-samples 90000`
  (exactly what P6/DCIS use and which works). `run_sparcal_net.py` has a latent
  label-encoder bug (no `no_variance` class at train time) — avoid until fixed.
- **ACTION: re-run P4 steps 5 → 6 → 7 only** (steps 1–4 outputs are valid, no need to redo).
  Then verify `germline_denovo`/`somatic_denovo` VCFs are non-empty and the `normal` matrix
  col count ≈ germline_defined + UPV.

**DCIS chain `12270329..335` — HEALTHY ✅.** Uses `run_supplimentary_models.py` (not
sparcal_net); step 5 produced full output (dcis1 hom 143,770 / het 96,767 / no_var 449,510;
dcis2 similar), both "completed successfully". As of this check: steps 1–5 done, step 6
`filter_bams` (12270334) RUNNING, step 7 pending. Then run `8_final_snv_mat_dcis{1,2}.sh`.

**Lesson:** these per-step SLURM scripts swallow errors and exit 0. When chaining with
`afterok`, watch for jobs that "COMPLETE" in seconds (`sacct -j <id> --format=JobID,State,Elapsed`)
— that means the step no-op'd, not that it succeeded.

---

## 2026-06-24 — UMI dedup ablation (P6 rep1) + SpatialSNV/SPARCAL comparison

### Why
SPARCAL's mpileup calling does NOT do **UMI deduplication**, nor GATK
**SplitNCigarReads / BQSR**. UMI dedup is the field standard for 10x variant calling
(cellsnp-lite, vartrix, SpatialSNV all do it) → highest reviewer risk. Plan: a dedup
**ablation** on one pilot section (P6 rep1) to show whether the final variant set /
spatial categories move (turns "you skipped dedup" into a figure). SplitNCigar/BQSR
are lower-risk — defend with a splice-junction-distance figure + the "SparcalNet IS a
learned error model (subsumes BQSR)" argument.

### Terminology (post-dedup)
Each retained alignment = one **molecule**, not a read. In deduped outputs rename:
depth → **UMI count / molecular depth**; alt/ref → **alt/ref UMI count**; VAF →
**molecular VAF** (more accurate — not amplification-inflated).

### Done (P6 rep1)
- **BACKUP:** `data/P6_tumor/1` → `data/P6_tumor/1_pre_umidedup_2026-06-24` (rename).
- **NEW pre-calling step:** `run_slurm/P6_tumor/0_umidedup_split_P6.sh`
  - `umi_tools dedup --per-cell --cell-tag=CB --umi-tag=UB --method=directional`
    (umi_tools 1.1.6, SpaceTracer env) on possorted → `possorted.dedup.bam`
  - `samtools split -d CB` (spatialsnv env v1.23.1) → regenerates `split_BAM/{bc}.bam`.
    `split_BAM/` had been cleaned up; rebuilding it **deduped** means every downstream
    step that reads `split_BAM/` inherits dedup — no other code change. Split naming
    verified to match `get_bam_list_for_tumor` (`{barcode}-1.bam`).
- **Step 0 (dedup+split) DONE** (job 12133315, 2026-06-24/25):
  - **Dedup result: 84,253,988 → 51,485,809 reads = ~39% duplicates removed** (real PCR
    inflation; citable ablation number). `possorted_genome_bam.dedup.bam` = 4.1 G.
  - **🐞 split bug found+fixed:** `samtools split -d` defaults to **`-M 100`** → only 100 of
    ~4,986 cells got a BAM; the rest (3.6 G) went to `_nobarcode.bam`. Fixed with **`-M 6000`**
    (ulimit -n is 524288, no FD concern). Re-split the *existing* dedup.bam (no re-dedup) →
    **4,992 per-spot BAMs** in `…/outs/split_BAM/` (+ indexes). Script patched.
- **Submission order** (from repo root):
  1. ~~`sbatch run_slurm/P6_tumor/0_umidedup_split_P6.sh`~~ **DONE** (split_BAM ready, deduped)
  2. ~~`sbatch run_slurm/P6_tumor/1_mpileup_pipeline.sh`~~ **DONE 2026-06-25 (job 12145890)** —
     COMPLETED, exit 0:0, **5h10m**, MaxRSS ~39 GB. Loaded 3,650 in-tissue / 1,343 out-of-tissue
     BAMs OK (the `.gz` fix worked); 22/22 regions; `merged_sorted_gt.vcf.gz` = 28 MB, 1 sample.
     **Ablation data point (calling stage):** deduped **1,540,060** merged sites vs pre-dedup
     **1,512,275** (`1_pre_umidedup_2026-06-24/`) = **+27,785 (+1.8%)** — i.e. the merged call set
     is essentially unchanged despite ~39% read-duplication removal; the real dedup effect to watch
     is downstream (categories/matrices, steps 7–8). Live log: `slurm_output/P6_tumor/b13m20/…out1`
     (script's duplicate `--output` → SLURM writes `b13m20/…out1`/`…err1`, not `b0m0/`).
  3. ~~`sbatch run_slurm/P6_tumor/2_beagle_pipeline.sh`~~ **DONE 2026-06-25 (job 12152678)** —
     COMPLETED, exit 0:0, **7.5 min**, MaxRSS ~56 GB; 22/22 chromosomes, 0 failures.
     Output: `output_VCFs/beagle/baseQ0mapQ0/{chr*.vcf.gz ×22, all_filtered_in.vcf.gz,
     all_filtered_out.vcf.gz}`. **Ablation (beagle stage):** `filtered_in` (kept) **126,034**
     dedup vs **126,870** pre-dedup = **−836 (−0.7%)**; `filtered_out` 1,414,026 vs 1,385,405.
     → consistent with step 1: dedup barely moves the call set; watch for the effect downstream.
     Live log (script's last `--output`): `slurm_output/P6_tumor/baseQ13mapQ20_section2_beagle_pipeline_P6_tumor.out`.
  4. ~~`sbatch run_slurm/P6_tumor/3_beagle_genotype_shifting.sh`~~ **DONE 2026-06-25 (job 12152771)**
     — COMPLETED, exit 0:0, **2m40s**. (Note: this step does NOT rewrite `all_filtered_{in,out}.vcf.gz`
     — those are step-2/run_beagle outputs; it produces the **genotype-shift analysis** under
     `data/P6_tumor/1/metrics/beagle/baseQ0mapQ0/`: `P6_TUMOR_1_shifted_results.pkl`,
     `_stable_results.pkl`, `_shifted_detailed_counts.csv`, + plots — feeds step 4.)
  5. ~~`sbatch run_slurm/P6_tumor/4_seq_err_model.sh`~~ **DONE 2026-06-25 (job 12152848)** —
     COMPLETED, exit 0:0, **20 s**. (Edited the script: commented out the active `--section_id 2`
     line — sec2 N/A for the ablation.) **Ablation (seq-err stage):** of 1,414,026 variants,
     **188,760 (13.35%) flagged sequence errors**, 1,225,266 (86.65%) kept. Output:
     `output_VCFs/SeqErrModel/baseQ0mapQ0/{sequence_error,sequence_no_error}.vcf.gz` + summary.
  6. ~~`sbatch --array=1 run_slurm/P6_tumor/5_neural_network.sh`~~ **DONE 2026-06-25 (job 12152936_1)**
     — COMPLETED, exit 0:0, **4 min**, MaxRSS ~3.4 GB. Validation variant-only **F1 0.687**; applied
     to 1,225,266 variants. **Filtered (thr 0.5):** hom 1/1 **551,373**, het 0/1 **112,518**,
     no-variance 0/0 **478,398** → **663,891 true variants**. Output:
     `output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_{predictions,homozygous,heterozygous,no_variance}.vcf.gz` + `neural_network_model.pkl`.
  7. ~~`sbatch run_slurm/P6_tumor/6_single_bam_snp_filter.sh`~~ **DONE 2026-06-25 (job 12153130)** —
     COMPLETED, exit 0:0, **2h00m**, MaxRSS ~23 GB. **The previously-FAILED step now WORKS on the
     deduped `split_BAM/`.** SNV pool combined = **789,925** (defined 126,034 + denovo 663,891).
     Processed 4993 BAMs, 4993 filtered / 0 failed; **4991/4993 (99.96%) have detected SNVs**;
     783,565 unique detected variants. **NOTE the real output path is
     `output_VCFs/spotprofiles/baseQ0mapQ0/{bam_by_spot/ (4993), vcf_by_spot/ (4991)}`** +
     `all_detected_variants_summary.vcf.gz` / `all_variants.vcf.gz` — NOT the `BAM_filtered/…/snv_positions/`
     path in CLAUDE.md (stale for this script version). Harmless: per-chrom "fetch called on bamfile
     without index" errors are only for `_nobarcode.bam` (the catch-all bin, correctly skipped).
  8. ~~`sbatch run_slurm/P6_tumor/7_spatial_filter_n_matrix.sh`~~ **SUBMITTED 2026-06-25 (job 12156808)**
     — `run_spatial_snv_filter_enhanced.py` (clone+CNV integration via CalicoST P6_sec1: tumor-purity
     / clone_labels / cnv_seglevel — all verified present) → viz → `run_generate_matrix.py`
     (`--filter-subdir filtered_snvs --output-name normal`) → score-scatter plots. **Edited the script:
     loop `for SECTION_ID in 1` (was `1 2`)** — sec2 N/A. Outputs:
     `data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/{germline,denovo,somatic,...}` + matrices.
     Log `slurm_output/P6_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P6.out`. Then `8_final_snv_mat.sh`.
     ⚠️ **VERIFY the P6 UPV matrix bug is gone** (new `germline_denovo`/UPV matrix cols ≈ its VCF
     record count, NOT ~1.25M; a `somatic` matrix should also be produced).

### ✅ Pre-existing step-1 blocker (P6 rep1) — RESOLVED 2026-06-25
Config `barcode_file = …GSM4565825_barcodes.tsv.gz` but disk had only the uncompressed
`GSM4565825_barcodes.tsv` → step 1 built an empty BAM list. **Fixed:** created
`…/Meta_Data/GSM4565825_barcodes.tsv.gz` via `gzip -c … > ….gz` (env `gzip` has no `-k`;
original kept). Verified 3,650 barcodes in both. Step 1 then launched (job 12145890).

### Runtime
63.2M mapped reads. dedup (single-threaded) is the bottleneck ~4–10 h; split+index ~1 h
→ ~5–12 h in the 48 h window. Watch rate in `…/outs/umidedup.log`; the post-dedup read
count printed in the `.out` = duplication rate (an ablation number itself). Fallback if
too slow: chr-sharded dedup (22 parallel, merge).

### ⚠️ P6 `germline_denovo` (UPV) MATRIX is BROKEN — verify on re-run (found 2026-06-25)
While calibrating SpatialSNV-vs-SPARCAL magnitudes, found the P6 UPV **matrix** is corrupt:
- `P6_tumor/1_pre_umidedup_2026-06-24/final_matrices/baseQ0mapQ0/germline_denovo_spot_snv_matrix.pkl`
  has **1,253,838 columns**, but the categorized UPV **VCF**
  (`spatial_filter_purity/baseQ0mapQ0/germline/denovo/germline_denovo.vcf.gz`) has only **291**
  variants (matches step-7c "P6 → 0/291"). The matrix ≈ the raw merged set (~1.3M) → built from
  the wrong input at step 7b/8.
- P6 `germline_defined` matrix (106,884) is correct; P6 `somatic` matrix is **missing**.
- P4 matrices all match their VCFs (gd 37,985 / UPV 7,215 / som 4,924) — **P4 is fine, only P6 is bad**.
- True P6 SPARCAL total ≈ **120,702** (106,884 + 291 + 13,527), comparable to SpatialSNV ~92k —
  the earlier "P6 SPARCAL = 1.36M, 15× larger" was an artifact of this broken matrix.
- **ACTION:** when the P6 dedup pipeline re-runs steps **7b/8** (matrix generation), **verify the
  new P6 `germline_denovo` matrix has ~291 columns, NOT ~1.25M**, and that a `somatic` matrix is
  produced. Do NOT use the stale matrix for any benchmark in the meantime.

### TODO — Monopogen spot×SNV matrix (missing piece of the 3-way benchmark)
**Status: CONVERTER BUILT 2026-06-25, not yet run at scale.**
- **Script:** `SPARCAL_Benchmarking/monopogen_to_spot_matrix.py` (+ runner
  `SPARCAL_Benchmarking/4_monopogen_to_pkl.slurm`, array 0-3 for the 4 samples).
- **Design:** Monopogen `out/germline/merged.vcf.gz` is **pseudobulk single-sample**
  (one col `possorted_genome_bam`, all PASS, hg19 chr-prefixed) → no per-spot info, so
  we use the **strelka2 precedent**: pysam pileup of the CB-tagged possorted BAM at
  Monopogen PASS-SNV positions, **allele-aware** (≥`--min-alt-reads` ALT-base reads ⇒
  present, default 1). Writes the `.pkl` directly in the SpatialSNV/SPARCAL contract
  (uint8, rows=in-tissue `AAAC…-1`, cols 4-part no-`chr` `{chrom}_{pos}_{ref}_{alt}`).
- **Inputs verified:** all 4 Monopogen VCFs + the 4 hg19 possorted BAMs (CB tags carry
  `-1`) + `tissue_positions.csv` (NEW spaceranger format — **has a header**, unlike
  DLPFC's headerless `tissue_positions_list.csv`; loader skips it). P4_rep1: 51,567 SNV
  alleles (1,143 indels skipped), 750 in-tissue spots. **chr22 smoke test passed**
  (617 SNVs × 750 spots, cols `22_17074104_G_A`).
- **Row-set note:** rows = ALL 750 in-tissue spots; SPARCAL P4 matrix has 744 → intersect
  to the shared spot set before the 3-way compare (same as strelka2/GATK row caveat).
- **NEXT:** `cd SPARCAL_Benchmarking && sbatch 4_monopogen_to_pkl.slurm` (~13 min/sample
  like the original SNP-profile scan), then it joins SpatialSNV + SPARCAL in P4/P6.
- We have Monopogen **VCFs** but the matrix below is the contract to fill.
- **Monopogen VCFs** (collaborator CanLuo): `/lfs/…/CanLuo/ST_SNV/Monopogen/{P4,P6}_rep{1,2}/out/
  germline/merged.vcf.gz`. Counts: P4_rep1 52,710 / P4_rep2 49,957 / P6_rep1 103,013 /
  P6_rep2 111,121. **No DCIS** (Monopogen never run on DCIS → 3-way compare is P4/P6 only).
- Only VCF-level overlaps exist (`run_slurm/overlap/overlap_Monopogen_SPARCAL/`, venn PNGs) — not matrices.

**How we converted before / the standard:**
- For **SpatialSNV** we used `SPARCAL_Benchmarking/callback_to_pkl.py` — but that consumed
  SpatialSNV's **CallBack MTX**. Monopogen has no CallBack; it's a plain VCF.
- The **standard for turning a VCF-based caller into the benchmark matrix** is the **strelka2
  precedent**: scan the merged BAM by `CB` tag at the caller's PASS-SNV positions (allele-aware:
  ≥1 read carrying the ALT base ⇒ "present" in that spot) → per-spot `<barcode>.txt` →
  `run_generate_matrix.py` → spots×SNV `.pkl`.
- **Reference scripts:** `scripts/tools/strelka2_to_spot_snvs.py` (the BAM-scan-by-CB step) +
  `scripts/6_spatial_filter/run_generate_matrix.py` (the `.pkl` builder). See the strelka2
  spot-matrix array in this file ("Strelka2 spot×SNV matrix" section) for the exact pattern.
- **Contract to match:** uint8 binary, rows=barcodes (`AAAC…-1`), cols=`{chrom}_{pos}_{ref}_{alt}`
  (no `chr`), spots×SNVs — same as `…/final_matrices/…/*_spot_snv_matrix.pkl` and the SpatialSNV
  pkls. (Reuse `callback_to_pkl.py`'s `chr`-strip + 4-part-key conventions for consistency.)
- Build per section from the Monopogen VCF + the (deduped, once available) merged/per-spot BAM,
  then it joins SpatialSNV + SPARCAL in the P4/P6 comparison. **(Conversion to be done in a
  separate session.)**

### SpatialSNV vs SPARCAL (the comparison)
| | **SpatialSNV** | **SPARCAL (ours)** |
|---|---|---|
| Type | somatic **caller** | spatial variant **interpretation** framework |
| Calling | GATK **Mutect2** (reassembly, PON, somatic LL, FilterMutectCalls) | mpileup/bcftools + **SparcalNet** (learned 3-class genotype/error NN) |
| Preprocessing | UMI dedup + SplitNCigar + BQSR | none (this ablation tests adding dedup) |
| Germline/somatic | gnomAD/PON priors only; one pseudobulk set | **spatial partition** germline / UPV / somatic |
| Clone/CNV | none | **CalicoST** CNV/LOH + clone labels |
| Phasing | none | Beagle/Eagle vs 1000G |
| Output | pseudobulk VCF + spot alt/ref/depth MTX | 3 spatial per-spot variant sets + matrices |

**Missing vs them (front-end):** somatic-grade caller, UMI dedup, SplitNCigar/BQSR.
**Our novelty:** SparcalNet (learned error model), spatial germline/UPV/somatic
categorization, CalicoST clone/CNV linkage, phasing/imputation, spatial-domain ARI
validation. Framing: *their "model" = off-the-shelf Mutect2 stats; ours = SparcalNet +
spatial model.* SpatialSNV benchmark tracker: `SpatialSNV/On_going.md`.

---

## Planned method additions (TODO — not yet implemented)

Came out of the variant-category naming discussion (2026-06-02). Both refine the
spatial-filter outputs. Design notes + paper wording: [pipeline_intro.md](pipeline_intro.md) §7–§8.
Category naming locked: **germline / UPV (Ubiquitous Private Variants) / somatic**
(was `germline_defined` / `germline_denovo` / `somatic_denovo`; code rename deferred).

- [~] **BAF-GMM sub-filter inside the UPV set — step 7c DRAFTED (2026-06-02).**
  Script: `scripts/6_spatial_filter/upv_baf_gmm_subfilter.py`. Runners:
  `run_slurm/{DCIS/7c_upv_baf_gmm.sh (dcis1+dcis2), P4_tumor/7c_upv_baf_gmm.sh,
  P6_tumor/7c_upv_baf_gmm.sh}`. Also wired into the enhanced filter via
  `run_spatial_snv_filter_enhanced.py --run_baf_gmm_subfilter` (subprocess hook,
  non-fatal on failure). Non-destructive: reads `germline_denovo.vcf.gz`
  (PURITY_CORR) + `merged_sorted_gt.vcf.gz`, fits a 2-D GMM on `[BAF, PURITY_CORR]`,
  writes `germline/denovo/gmm_subfilter/{upv_germline_like,upv_somatic_candidate}.vcf.gz`
  + TSV + `upv_baf_gmm.png` + summary. Ran on all 4 (corrected BAF):
  dcis1 → 1467 somatic-candidate / 2618; dcis2 → 0 / 3584; P4 → 0 / 7215; P6 → 0 / 291.
  - **BUG FOUND + FIXED — BAF field was 0 for all high-depth variants.**
    `merged_sorted_gt.vcf.gz` FORMAT `BAF` is 0 for every site whose I16 contains
    scientific notation (high depth) — root cause: `parse_i16` in
    `scripts/1_calling/mpileup_pipeline.py` used `int(x)` over all 16 I16 values;
    `int("1.38392e+07")` throws → silent `[0]*16` fallback → BAF=0 → falsely flagged
    `DiscordantBAF` (but kept). dcis1: exactly 1344/1,058,191 merged variants, 100%
    coincident with sci-notation I16; in the UPV set 340/4085 (mostly hom-ALT germline,
    true alt-frac ≈1.0). **Fix:** `parse_i16` now `int(float(x))`
    (fixes future step-1 runs). GT unaffected (from PL); variant set unaffected.
    Same latent bug in `mpileup_pipeline_old.py` + `all_caller_pipeline.py` (not fixed).
    **7c works around it now** by recomputing BAF = alt/(ref+alt) from I16 in
    `lookup_baf` — so 7c is correct without re-running step 1.
    **OPEN:** existing merged VCFs (all datasets) still carry the wrong BAF; the
    seq-error model consumed it. Decide whether a step-1 re-run is worth it (only
    0.1% of variants, the high-depth ones; they're kept regardless).
  - **PURITY_CORR is uninformative within UPV (unchanged finding).** Clipped to ≥0
    and flat (UPV are ubiquitous by definition → presence-vs-purity saturated). 2-D
    GMM degenerates to BAF-only. P4/P6/dcis2 have no low-BAF mode → 0 (conservative).
  - **HARD BAF ceiling added (`--somatic-baf-max`, default 0.35).** The soft GMM
    component (mean 0.319) has wide variance and spilled past 0.5, putting germline-
    het (BAF≈0.5) variants in the somatic set. The hard per-variant gate
    `somatic ⇔ (GMM somatic posterior > 0.5) AND (BAF < 0.35)` excludes the het/hom
    modes. dcis1: 1467 → **739** somatic-candidate, BAF range now 0.108–0.349
    (median 0.25); dcis2/P4/P6 still 0. Plot draws the cap line.
  - **ΔBAF experiment DONE (2026-06-02) — NEGATIVE result.**
    `scripts/6_spatial_filter/upv_delta_baf_experiment.py`. Per-spot read counts are
    NOT in any VCF: `vcf_by_spot` records **coverage-presence only** (step 6
    `filter_bam_one_chrom` marks a position "detected" on read overlap, never checks
    the base; sample=`./.`, INFO copied from merged pseudobulk → identical I16 across
    spots). So ΔBAF requires the reads: scanned the CB-tagged `possorted_genome_bam.bam`,
    pooled ref/alt per UPV position into tumor vs normal spot groups (tumor_proportion
    tertiles). dcis1, 1753 variants w/ ≥5 reads each side: ΔBAF symmetric around 0
    (q5/q95 = −0.33/+0.34, median 0); current BAF<0.35 somatic-candidates average
    ΔBAF = **−0.032** (NOT tumor-enriched). ⇒ **No tumor-associated somatic population
    inside UPV**; the low-BAF UPV variants are tumor-independent = ASE-skewed germline.
    Outputs: `gmm_subfilter/upv_delta_baf.{tsv,png}`.
    Caveats on the negative: tumor/normal spots are mixtures (tumor≥0.66, normal≤0.39
    purity → diluted contrast), per-spot reads sparse, CalicoST purity inferred — but
    the signal is flat enough that a strong somatic population would still show.
  - **CALL:** treat UPV as a germline-dominated mixed bucket; do NOT force a somatic
    sub-call. 7c (BAF-GMM) stays exploratory/supplementary, not a reportable somatic set.
  - **Step 6 per-spot `AD` — DRAFTED + unit-validated (2026-06-02).**
    `scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py` now tallies real per-spot
    ref/alt read counts while scanning each barcode BAM and writes
    `FORMAT=GT:AD:DP` (was `./.`) into `vcf_by_spot/{barcode}.vcf.gz`.
    `filter_bam_one_chrom(... pos_to_info)` returns `allele_counts`; aggregated in
    `filter_bam_by_positions` → written in `save_detected_snvs`. AD verified equal to an
    independent pysam pileup at every tested position. **Two behavior changes to know
    before re-running step 6:** (1) detection no longer `break`s at the leftmost SNV a
    read covers — every covered position is now recorded (fixes under-detection in dense
    regions, so per-spot variant counts may rise); (2) sample is real `GT:AD:DP`, with
    `./.` only where a read spans the locus but has no aligned base (deletion/clip).
    Perf: adds one `get_reference_positions` per kept read (slower; still parallel/30-worker).
    **Not yet re-run** — re-running regenerates all `vcf_by_spot` VCFs (and any matrix
    built from them). Payoff: ΔBAF + an allele-aware spot×SNV matrix (use ALT-count≥1 as
    presence instead of coverage) become computable straight from the VCFs, no BAM re-scan.
    `final_snv_mat.py`/`run_generate_matrix.py` can be updated to consume `AD` when desired.
  - **Candidate effective features (the principle: anything that does NOT depend on
    spatial presence-distribution, which UPV saturates).** Ranked:
    (1) **gnomAD/dbSNP annotation** — cheapest, highest value; peels off the
    rare-germline-not-in-1000G contaminant directly (UPV = "not in 1KGP", but gnomAD
    is far larger). (2) **per-clone ΔBAF** (read-level VAF, tumor vs normal spots).
    (3) **CNV/LOH allelic-imbalance consistency** vs CalicoST per-clone allele-specific
    CN. (4) **phasing consistency** (germline co-segregates with a Beagle/Eagle
    haplotype; somatic breaks phase). (5) mutational trinucleotide context / SBS
    signature (weak per-variant). BAF alone is necessary-not-sufficient (ASE confound).
  - Eventually fold the chosen version into Stage 1 of `run_spatial_snv_filter_enhanced.py`.
- [ ] **CHIP rule-out on the final somatic set.** Drop variants in common CHIP
  genes (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, …) from category (c) — likely
  clonal-hematopoiesis blood contamination. Implement as a **post-processing**
  filter on the somatic VCF (gene-list intersection), not in the core cascade.
- [ ] **Code rename** (deferred): `germline_defined→germline`,
  `germline_denovo→upv`, `somatic_denovo→somatic` across
  `run_spatial_snv_filter_enhanced.py` + `final_snv_mat.py` (keep 3-way output).

---

## SPATIAL_SNV benchmark clustering (post-processing — added 2026-06-09)

**Post-processing step: spatial-clustering benchmark of all callers.** Cluster each
caller's spot×SNV matrix (+ a gene-expression baseline) with STAGATE→mclust and score
spatial domains against the DLPFC layer annotations (ARI). This is the downstream
comparison that consumes the per-caller matrices.

Files in `SPATIAL_SNV/`:
- `clustering_config.json` — config (sections, modality→caller/filter/grouping, paths, model params)
- `SPARCAL_clustering.py` — main script: load matrix → STAGATE → mclust → ARI, saves outputs (PNG **+ editable PDF**)
- `run_clustering.slurm` — array job; `--array=0-0` (151507) / `--array=0-11` (all 12)
- `make_ari_boxplot.py` — aggregate every section's `summary.csv` → 4-modality ARI box plot (PNG+PDF) + `ari_table.csv`/`ari_matrix.csv`
- `make_combined_figure.py` — per-section concatenated figure: row1 = spatial domains (GT + each modality, colors Hungarian-matched to GT layers), row2 = per-modality UMAP colored by true layer (PNG+PDF)

**Four modalities** (config `modalities`): `sparcal` (bcftools/normal), `strelka2` (germline),
`gatk` (germline), `gene_expr` (Visium h5 baseline). gene_expr loads the SpaceRanger
`{section}_filtered_feature_bc_matrix.h5` directly (no pkl) via `load_gene_expr_section` —
same `normalize_total+log1p`, no HVG, so all 4 share identical STAGATE preprocessing.

**Outputs:**
- `data/dlpfc/{section}/clustering/{modality}/` — embedding.npy, cluster_labels.csv, ari.txt, umap.{png,pdf}, spatial.{png,pdf}
- `data/dlpfc/{section}/clustering/summary.csv` — per-section ARI table
- `data/dlpfc/{section}/clustering/combined_{section}.{png,pdf}` — concatenated figure
- `data/dlpfc/clustering_benchmark/ari_boxplot.{png,pdf}` + `ari_table.csv` + `ari_matrix.csv`

**Run (from repo root; then activate snv_clustering for the figure scripts):**
```bash
sbatch --array=0-11 SPATIAL_SNV/run_clustering.slurm     # 12 sections × 4 modalities
source activate snv_clustering
python SPATIAL_SNV/make_ari_boxplot.py
python SPATIAL_SNV/make_combined_figure.py --section_id 151507
```

**GPU (2026-06-09):** `run_clustering.slurm` runs on `batch_gpu` / `maiziezhou_lab_acc`
with `--gres=gpu:nvidia_rtx_a6000:1`. `train_STAGATE` auto-uses cuda:0 (no code change).
A6000 ≈ 14 epochs/s vs 8 s/epoch on CPU (~110×); ~5 min/section for all 4 modalities.
**Constraint:** our account is authorized for 2080_ti/titan_x/quadro/a6000, but only the
**a6000 still has live nodes** (others decommissioned) and the a6000 cap is **2 concurrent
GPUs** — so a 12-section array runs ~2 at a time. (Stale Slurm quota: 51 on the dead
2080_ti, 2 on the live a6000.) Possible ask to ACCRE/PI: rebalance the quota toward a6000,
or grant access to the idle l40s/a100 nodes.

**Bugs fixed during setup (2026-06-09):**
- `STAGATE_pyG/utils.py` `Transfer_pytorch_Data`: `torch.Tensor(lbl)` crashed on string labels — LabelEncode first.
- `STAGATE_pyG/gat_conv.py`: `NoneType` removed from `torch_geometric.typing` (PyG 2.x) — added `NoneType = type(None)` locally.
- `SPARCAL_clustering.py` numpy shim: pkl serialized with NumPy 2.x (snv_caller); env has 1.24.4 — register `numpy._core` alias for `numpy.core` before unpickling.
- **`mclust_R` rpy2 3.6.x regression (fixed 2026-06-09).** Job "completed" but ARI=NaN for
  all modalities. Two stacked breaks vs the older cluster: (1) `numpy2ri.activate()` is
  removed in rpy2 3.6 (now raises "activate and deactivate are deprecated"); (2) passing the
  embedding positionally into `Mclust` throws `length of 'dimnames' [2] not equal to array
  extent` under rpy2 3.6.4 / mclust 6.1.2. **Fix:** push the embedding into R's global env
  under a `localconverter` context, then call `Mclust(emb_mat, G=k, modelNames="EEE")` from an
  R string. Verified. Not a logic bug — an env/version regression (will bite other scripts
  that still call mclust the old way: the notebooks, `classification_codepart.py`, `st_loading_utils.py`).

**Environment:** `snv_clustering` (Python 3.10, torch 2.2.2+cu121, PyG 2.5.2, scanpy 1.11.4,
rpy2 3.6.4, R 4.5.1 mclust 6.1.2). `STAGATE_pyG` via PYTHONPATH (setup.py refs missing README.rst).

**Status (2026-06-09):** pipeline working end-to-end on **151507** (job 11731374, ~5 min on
a6000) — ARI **sparcal 0.304 > gatk 0.184 > strelka2 0.128**. gene_expr not yet clustered.
**NEXT:** resubmit `--array=0-11` (now 4 modalities) → run the two figure scripts for the
full 12-section boxplot + combined figures. Caveat: filtering differs per caller (only GATK
has the 1000G filter) — note in methods.

---

## Active SLURM Jobs

### Step 6 re-run with per-spot AD fix — submitted 2026-06-02

Re-running `6_single_bam_snp_filter.sh` to regenerate `vcf_by_spot/` with real
`GT:AD:DP` (the AD correction + leftmost-`break` under-detection fix in
`run_filter_bams_by_snv_pools.py`).

| Dataset | Section | Job | Status |
|---------|---------|-----|--------|
| DCIS | 1 + 2 | 11548544 (`filter_bams_DCIS`) | RUNNING (cn1817) |
| P4_TUMOR | 1 | 11548546 (`filter_bams_P4_TUMOR`) | **FAILED** |
| P6_TUMOR | 1 | (`filter_bams_P6_TUMOR`) | **FAILED** |

**P4/P6 failure = missing input, NOT the AD code.** `run_filter_bams_by_snv_pools.py`
→ `FileNotFoundError: No BAM files found at .../P{4,6}_Tumor_output/outs/split_BAM/*.bam`
(the bash loop still prints "All replicates processed", so it looks done in seconds).
Diagnosis: the `outs/` dir EXISTS and still has `possorted_genome_bam.bam` (+.bai), but
the **`split_BAM/` subdir is gone** — the per-barcode BAMs were cleaned up after the
original Feb-2026 run. **To unblock P4/P6:** re-run **step 0** (`0_split_bam.sh`,
splits possorted BAM by CB tag + barcode list) to regenerate `split_BAM/`, THEN re-run
step 6. (Alternative, bigger: rework step 6 to scan the possorted BAM by CB tag directly
— like the ΔBAF experiment — and skip storing thousands of per-barcode BAMs.) DCIS
unaffected (its `split_BAM/` exists).

> 🔔 **REMIND THE USER next session:** P4 sec1 and P6 sec1 have **no `split_BAM/`**
> (only the possorted BAM survives). If this is still not resolved when you next read
> this, tell the user to **ask their colleagues** (who own/produced the P4/P6 Visium
> data) to restore or regenerate the per-barcode split BAMs at
> `/lfs/.../STmut_Data/P{4,6}_Visium/spaceranger_align_rep1_hg19/P{4,6}_Tumor_output/outs/split_BAM/`.
> Until then P4/P6 step 6 (and the whole AD/UPV re-run chain) stays blocked.

> ⚠️ **DOWNSTREAM RE-RUN REQUIRED after step 6 finishes (read this before using any
> result for the affected sections).** The step-6 re-run changes `vcf_by_spot/` (real
> AD + more detected positions), so everything built from it is now **stale** and must
> be re-run in order:
> 1. **Step 7** spatial filter (`7_spatial_filter_n_matrix.sh` → `run_spatial_snv_filter_enhanced.py`)
>    — regenerates the UPV (`germline_denovo`) / germline / somatic per-spot sets.
> 2. **Step 8** final matrices (`8_final_snv_mat_*.sh` → `final_snv_mat.py`).
> 3. **7c** UPV BAF-GMM (`7c_upv_baf_gmm.sh`) — depends on the new step-7 UPV set.
> Also: any benchmark/comparison matrix and the ΔBAF experiment derive from these.
> Currently only **DCIS** will have fresh step-6 output; P4/P6 are blocked (see above).



### Strelka2 germline array — RESOLVED 2026-06-01 (bioconda 2.9.10 rebuild)

Script: `strelka2/run_slurm/strelka2_germline_dlpfc.sh`
**Root cause (finally):** the prebuilt `strelka-2.9.2.centos6_x86_64` binary is broken against
the current OS (fails even on the login node with a 5 KB demo file — NOT compute-specific, NOT
a path issue). All the realpath/panfs/`/tmp`/`/dev/shm` work was a phantom. Fixed by rebuilding
via bioconda: `strelka=2.9.10=hdfd78af_2` (see `strelka2/scripts/install_strelka_conda.sh`,
which also re-applies the Bug 1 SMTP-timeout patch). Runner now resolves the configure script
via `$STRELKA_CONFIG`/PATH. Full details: `strelka2/DEBUGGING.md` (2026-06-01 entry).

**ALL 12 SECTIONS — DONE** (job 11528358_0 + array 11528559_1..11, all COMPLETED exit 0,
~11 min each). Validated 2026-06-01 via `strelka2/scripts/validate_strelka2_outputs.sh`:
all 12 `variants.vcf.gz` integrity-OK, PASS-SNV 58k–88k/section, all 24 GRCh38 contigs.
Output: `data/dlpfc/{section}/strelka2/results/variants/{genome,variants}.vcf.gz`.

---

### Strelka2 spot×SNV matrix (cross-tool comparison) — job array 11534438 (2026-06-01) — IN PROGRESS

Script: `scripts/tools/strelka2_spot_matrix_dlpfc.sh` (array 0-11). Two steps/section:
1. `scripts/tools/strelka2_to_spot_snvs.py` — scan merged BAM by CB tag at strelka2 PASS-SNV
   positions (allele-aware: ≥1 read carrying the ALT base = present), PASS SNVs only →
   one `<barcode>.txt` per in-tissue spot in `data/dlpfc/{section}/strelka2/spot_snvs/`.
2. `scripts/6_spatial_filter/run_generate_matrix.py --caller strelka2 --output-name germline` →
   `data/dlpfc/{section}/matrix/DLPFC_{section}_strelka2_germline_6_matrix.pkl`.
Resources: 22 CPU / 64 GB / 4 h. Actual: ~34–49 min, ~6–7 GB RSS per section.
Rationale/design: see [[project-strelka2-matrix-comparison]] + this session's notes. The merged
BAM carries CB:Z: tags so the single-BAM scan works (no per-cell BAM iteration).

| Task | Section | Status | Matrix (spots × SNVs) |
|------|---------|--------|------------------------|
| 0 | 151507 | DONE | 4226 × 58,979 |
| 1 | 151508 | DONE | 4384 × 53,032 |
| 2 | 151509 | DONE | 4789 × 60,951 |
| 3 | 151510 | DONE | 4634 × 57,929 |
| 4 | 151669 | DONE | 3661 × 58,319 |
| 5 | 151670 | RUNNING (cn1815) | — |
| 6 | 151671 | RUNNING (cn1816) | — |
| 7 | 151672 | RUNNING (cn1817) | — |
| 8 | 151673 | RUNNING (cn1815) | — |
| 9 | 151674 | RUNNING (cn1816) | — |
| 10 | 151675 | PENDING (QOSGrpCpuLimit) | — |
| 11 | 151676 | PENDING (QOSGrpCpuLimit) | — |

Matrices validated (tasks 0-4): binary int8 {0,1}, rows=barcodes, cols=`chrom_pos` (same key
format as pipeline matrices → directly comparable), spots = in-tissue counts, ~82% of PASS SNVs
observed in ≥1 spot, median 84–149 SNVs/spot. Logs: `strelka2/slurm_output/spot_matrix_{0..11}-*.{out,err}`.

**Next (optional):** head-to-head vs pipeline's `DLPFC_{section}_bcftools_normal_6_matrix.pkl`
(shared-spot count, SNV-column Jaccard).

---

### DLPFC Stage 7+7b rerun — job array 11333398 (submitted 2026-05-26) — COMPLETED WITH PARTIAL FAILURE

Script: `run_slurm/dlpfc/rerun_stage7_DLPFC.sh`
Reason: Stage 7 in job 11288684 used wrong script (`run_spatial_snv_filter.py` reading from empty `BAM_filtered/snv_vcf/`). Now uses `run_spatial_snv_filter_enhanced.py` which reads from `spotprofiles/baseQ0mapQ0/vcf_by_spot/` (same as P4/DCIS).
Dataset: DLPFC · all 12 sections · `baseQ0mapQ0` · 4 h / 4 CPU / 64 GB per task
`--tumor_purity_file` made optional in enhanced script (normal tissue → all spots default to purity 0.0).

| Task | Section | Stage 7 | Stage 7b | Notes |
|------|---------|---------|----------|-------|
| 0 | 151507 | DONE | FAILED | `run_generate_matrix.py` → `FileNotFoundError: spatial_analysis/baseQ0mapQ0/filtered_snvs` missing |
| 1 | 151508 | DONE | FAILED | same as 151507 |
| 2 | 151509 | DONE | FAILED | same as 151507 |
| 3 | 151510 | DONE | DONE | |
| 4 | 151669 | DONE | FAILED | same as 151507 |
| 5 | 151670 | DONE | FAILED | same as 151507 |
| 6 | 151671 | DONE | DONE | |
| 7 | 151672 | DONE | DONE | |
| 8 | 151673 | DONE | DONE | |
| 9 | 151674 | DONE | DONE | |
| 10 | 151675 | DONE | DONE | |
| 11 | 151676 | DONE | DONE | |

**Stage 7b failure root cause:** `run_generate_matrix.py` reads from `spatial_analysis/baseQ0mapQ0/filtered_snvs` (old path), empty for these 5 sections. Stage 7 (enhanced) already wrote per-barcode `.txt` to `spatial_filter_purity/baseQ0mapQ0/germline/` ✓.

**Fix applied (2026-05-27):**
- `run_generate_matrix.py`: added `--input-dir` flag (bypasses old path construction); added header-line skip (`pos.isdigit()` guard).
- New rerun script: `run_slurm/dlpfc/rerun_stage7b_DLPFC.sh` — **array 0-11 for all 12 sections**, reads from `spatial_filter_purity/{QUALITY_FILTER}/germline/`. Covers all 12 for consistency (7 already had old-pipeline matrices; those will be overwritten with enhanced-pipeline output).
- **Ready to submit:** `cd /panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling && bash run_slurm/dlpfc/rerun_stage7b_DLPFC.sh`

**Logs:** `slurm_output/DLPFC/baseQ0mapQ0/stage7_{0..11}.{out,err}`

---

### Strelka2 germline array — all prior attempts FAILED (see strelka2/DEBUGGING.md)

Jobs 11272103, 11287046, 11287211, 11287522, etc. — all 12 sections, all failed.
Three bugs identified and fixed (SMTP hang, panfs path resolution, shutil.copy2 truncation).
Superseded by jobs 11390244 + 11390247 above.

## Completed Tasks

| Job ID | Script | Config | Completed | Notes |
|--------|--------|--------|-----------|-------|
| 11360329 | `rerun_stage7b_DLPFC.sh` | DLPFC all 12 sections · `baseQ0mapQ0` | 2026-05-27 | Stage 7b ✓ all 12 sections. Matrices in `data/dlpfc/{section}/matrix/DLPFC_{section}_bcftools_normal_6_matrix.pkl`. Input: `spatial_filter_purity/baseQ0mapQ0/germline/` (enhanced pipeline output). |
|--------|--------|--------|-----------|-------|
| 11288684 | `run_pipeline_DLPFC.sh` | DLPFC all 12 sections · `baseQ0mapQ0` | 2026-05-25/26 | Stages 1–6 ✓ all sections. Stage 7 used wrong script (bug). Stage 7b: 5 sections failed (151507/08/09/151669/70 — no pre-existing filtered_snvs), 7 succeeded via old Jul-2025 data. Rerun as job 11333398. |
| 11065771 | `vcf_visualizer.sh` | DCIS sec1 · germline_denovo · exome BED | 2026-05-18 | exit 0 · 1h09m |
| 11065872 | `vcf_visualizer.sh` | DCIS sec1 · germline_denovo · exome BED | 2026-05-18 | exit 0 · 1h05m |
| 11066319_1 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · germline denovo | 2026-05-18 | exit 0 · 25s · 176 unique SNVs (exome) |
| 11066319_2 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · somatic denovo | 2026-05-18 | exit 0 · 33s · 837 unique SNVs (exome) |
| 11066319_3 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · germline defined | 2026-05-18 | exit 0 · 36s · 4,716 unique SNVs (exome) |
| 11066713_1 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · germline denovo | 2026-05-18 | exit 0 · 10s · 242 unique SNVs |
| 11066713_2 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · somatic denovo | 2026-05-18 | exit 0 · 3s · 639 unique SNVs |
| 11066713_3 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · germline defined | 2026-05-18 | exit 0 · 12s · 5,182 unique SNVs |
| 11066715_1 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · germline denovo | 2026-05-18 | exit 0 · 7s · 479 unique SNVs |
| 11066715_2 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · somatic denovo | 2026-05-18 | exit 0 · 7s · 545 unique SNVs |
| 11066715_3 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · germline defined | 2026-05-18 | exit 0 · 9s |
| 11066716_1 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · germline denovo | 2026-05-18 | exit 0 · 10s · 106 unique SNVs |
| 11066716_2 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · somatic denovo | 2026-05-18 | exit 0 · 10s · 536 unique SNVs |
| 11066716_3 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · germline defined | 2026-05-18 | exit 0 · 7s |

### Output files — per_barcode_visualizer (all datasets, sec1)

**DCIS sec1** (`data/dcis1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 242 SNVs, median 12/spot, 1434/1454 spots covered
- `germline/per_barcode_visualizer/germline_defined_*` — 5,182 SNVs, median 82/spot
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 639 SNVs, median 2/spot, 1040/1454 spots covered

**P4 tumor sec1** (`data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 479 SNVs, median 228/spot (743/744 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 3,377 SNVs, median 132/spot (741/744 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 545 SNVs, median 3/spot (574/744 spots)

**P6 tumor sec1** (`data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 106 SNVs, median 16/spot (3605/3650 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 3,251 SNVs, median 15/spot (3637/3650 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 536 SNVs, median 0/spot (1609/3650 spots)

**DCIS sec2** (`data/dcis2/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 176 SNVs, median 18/spot (1803/1807 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 4,716 SNVs, median 77/spot (1803/1807 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 837 SNVs, median 2/spot (1371/1807 spots)
