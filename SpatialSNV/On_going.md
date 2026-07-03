# SpatialSNV — On_going tasks

Live tracker for the SpatialSNV benchmark runs. When asked to **"check ongoing tasks"**:
run `squeue -u $USER`, inspect the per-job `.out`/`.err` under `slurm/slurm_output/` and
the outputs under `results/<sample>/`, and update this file in place.

Tool = github.com/YoungLi88/SpatialSNV (Mutect2-based spatial SNV caller), benchmarked
against SPARCAL. Env: `spatialsnv` (conda; py3.10.14 + gatk4 4.6.2 + samtools + picard +
bcftools + openjdk17). Driver: `scripts/run_spatialsnv.sh` (prep → Mutect2/Filter →
CallBack). Generic SLURM wrapper: `slurm/run_dataset.sh <config.env> [stages] [subset]`.

---

## ⚠️ SpatialSNV tool gotchas (tell colleagues before they run it)
Tool/behavior issues (not env/path mistakes), each with time it can burn if unknown:

- **Call 10x with `-c CR -u UR`, NEVER `-c CB -u UB`.** `PerpareBAMforCalling` builds
  `LY=<barcode>-<umi>` and Picard MarkDuplicates validates it against `^[ATCGNatcgn-]*$`; the 10x
  `CB` `-1` GEM suffix (a digit) is illegal → **every section dies at MarkDuplicates** ("UMI found
  with illegal characters"). CR/UR (the documented defaults) have no `-1`. CallBack may still use
  CB/UB. ⏱ **~½–1 day**: prep runs ~1 h before it fails, the Picard error is cryptic, and you
  re-submit every section (we lost a full 4-section cycle).
- **Failures are SILENT.** The tool's `RunCMD` prints a failed sub-command's stderr but **returns
  success** — a failed GATK/Picard step does not stop the run. Verify the expected output file
  exists after each step. ⏱ **hours per failure** chasing an empty VCF / missing input when an
  "OK" step actually failed.
- **Undocumented required args.** `PerpareBAMforCalling` needs `--dbsnp`; `SNVCalling` needs
  **both** `--pon` and `--germline` (README implies optional). The arg error is instant, ⏱ **but
  hours to source build-matched hg19/hg38 resources** (PON, gnomAD, dbSNP — incl. recompressing
  the Broad dbSNP to BGZF).
- **Naming traps.** Import is `spatialsnvtools` (not `spatialsnv`); the prepare command is
  misspelled `PerpareBAMforCalling`; top-level `SplitChromBAM`/`SNVCalling`/… are **modules, not
  functions** — CLI only, no Python API. ⏱ **1–2 h** debugging against a non-existent Python API.
- **hg19 chrM length clash** (hg38 fine). Mutect2 rejects hg19 resources: hg19 `chrM`=16,571 bp
  vs b37 PON/gnomAD=16,569 → "incompatible contigs chrM". GATK reads lengths from the VCF
  **header**, so drop `chrM` from records **and** `##contig` header. ⏱ **hours–1 day**: fails only
  after prep completes, then rebuild resources + re-run.
- **CallBack matrices are inconsistent.** `alt`/`depth`/`ref` are **not co-indexed** (different
  barcode axes) and **`depth.total==alt.total`** (depth ≠ real coverage → no per-spot VAF); only
  **ALT** is reliable. `barcodes.tsv` has no tab → trailing `\n` on naive parse. ⏱ **hours–½ day**
  computing VAF / combining matrices and getting nonsense before realizing they're mis-indexed.

---

## ACTIVE / status (full whole-genome runs; submitted 2026-06-13 09:10)

**ALL 4 COMPLETE** (verified 2026-06-24). VCF + CallBack matrices present for every sample:

| Job ID | Sample | Build | Status (verified 2026-06-24) |
|--------|--------|-------|--------|
| 11869087 | dcis1 | hg38 | **DONE** — `results/dcis1/dcis1.vcf.gz` 529,673 var / 151,566 PASS; `matrix/dcis1_{alt,depth,ref}`. |
| 11869088 | dcis2 | hg38 | **DONE** — `results/dcis2/dcis2.vcf.gz` 512,397 var / 148,536 PASS; `matrix/dcis2_{alt,depth,ref}`. |
| 11869090 | p4    | hg19 | superseded (chrM bug) → **11893162 DONE** — `results/p4/p4.vcf.gz` + `matrix/p4_{alt,depth,ref}`. |
| 11869091 | p6    | hg19 | superseded (chrM bug) → **11893164 DONE** — `results/p6/p6.vcf.gz` + `matrix/p6_{alt,depth,ref}`. |

### 2026-06-24 — mission update: section-level SPARCAL benchmark + DCIS reuse
- **New parallel track:** `/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/` runs SpatialSNV at
  **section level** — P4_sec1, P4_sec2, P6_sec1, P6_sec2 (this effort's `p4`/`p6` are single
  sections only). Array job **`12123732`** (`2_run_spatialsnv_array.sh`, spatialsnv env, hg19,
  `-c CR -u UR`) RUNNING as of 2026-06-24 — all 4 in GATK prep (`nc.bam`), no VCF yet. Outputs →
  `…/SPARCAL/data/benchmarks/spatialsnv/{sample}/`. Tracker: `SPARCAL_Benchmarking/CLAUDE.md`.
  - First attempt `12123230` died at MarkDuplicates: 10x `CB` tag's `-1` GEM suffix is an
    illegal Picard UMI char → use `CR`/`UR` (same lesson, different surface, as the chrM fix).
- **DCIS = reuse, do NOT re-run** (user decision). The section-level benchmark consumes this
  effort's existing `results/dcis{1,2}/` VCFs + CallBack matrices directly. P4/P6 sec2 are the
  only genuinely new SpatialSNV runs.

### 2026-06-24 — DCIS CallBack → SPARCAL-contract `.pkl` — DONE (filter relaxed; rerun pending)
Goal: convert DCIS SpatialSNV output to the SPARCAL **DCIS** matrix contract so SpatialSNV /
SPARCAL / Monopogen compare on a common spots×SNV matrix.
- **Target contract = SPARCAL DCIS matrices** (`data/dcis{1,2}/final_matrices/baseQ0mapQ0/
  *_spot_snv_matrix.pkl`): pandas **uint8** binary; rows=barcodes (`AAAC…-1`); cols=
  **`{chrom}_{pos}_{ref}_{alt}`** (4-part, allele-aware, e.g. `1_629218_A_G`); **NO** header row.
  (NOT the DLPFC `chrom_pos` 2-part key — that was the wrong target initially.)
- **SpatialSNV CallBack output** = 10x MTX triplets `{alt,depth,ref}`, features×barcodes,
  feature key `chrom_pos:ref>alt`.
- **⚠️ FORMAT FINDINGS:** (1) `alt`/`depth`/`ref` are **NOT co-indexed** (diff barcode axes;
  dcis1 alt/depth=4834, ref=4990) and **`depth.total == alt.total`** → depth is **not real
  coverage**, per-spot **VAF NOT recoverable** from CallBack; use **ALT only**, presence=`alt>0`.
  (2) barcodes.tsv has no tab so a trailing `\n` survived a naive split — stripped now.
- **Converter:** `SPARCAL_Benchmarking/callback_to_pkl.py` + `3_dcis_callback_to_pkl.slurm`
  (array 0-1, snv_caller env, 96 GB). Per sample, next to the MTX:
  `{s}_spatialsnv_presence_matrix.pkl` (ORIGINAL) and `{s}_spatialsnv_presence_filtered_matrix.pkl`.
- **Compatibility verified** (job 12134294): col keys `1_14678_A_G`, clean barcodes,
  **1445/1446 barcode overlap** + **1488 SNV-key overlap** with SPARCAL germline_denovo.
- **FILTER RELAXED:** SPARCAL DCIS total is ~139k SNVs (germline_defined 121,903 +
  germline_denovo 4,085 + somatic_denovo 13,112), NOT 50k. Old 30-spot/3-UMI floor → only
  1,917 (way too few). FILTERED now defaults to **PASS ∩ SNV** (dcis1: **115,803**; ≥ matches
  SPARCAL scale). Spatial floors are CLI-tunable: `--min-spot-cov 2` → ~24k, `3` → ~13k.
- **STATUS: DONE (job 12141582, 2026-06-25).** All 4 pkls written to
  `results/dcis{1,2}/matrix/`. FILTERED (PASS∩SNV) = **dcis1 115,803 / dcis2 115,703** SNVs
  (vs old 30/3 floor 1,917); ORIGINAL = all 281,077 / 267,586. (History: 12131988 crashed→caught
  the cross-axis bug; 12134294 format-correct but over-filtered; 12141582 final relaxed.)
  Tune floors via `--min-spot-cov` (2→~24k, 3→~13k) if a smaller set is wanted.

### Section-level P4/P6 array `12123732` — DONE (2026-06-25)
All 4 calls complete (Mutect2+FilterMutectCalls). Outputs `…/SPARCAL/data/benchmarks/
spatialsnv/{sample}/calls/{sample}.vcf.gz`:

| Sample | variants | PASS |
|---|---|---|
| P4_sec1 | 214,663 | 61,291 |
| P4_sec2 | 159,139 | 53,496 |
| P6_sec1 | 295,437 | 93,575 |
| P6_sec2 | 346,850 | 96,942 |

(First attempt `12123230` died at MarkDuplicates — 10x `CB` `-1` suffix is an illegal Picard
UMI char; fixed to `-c CR -u UR`.) **NEXT for these:** run **CallBack** to get spot matrices,
then convert to the SPARCAL-contract `.pkl` (same `callback_to_pkl.py` as DCIS) for the
section-level comparison.

### 2026-06-14 — hg19 chrM bug + fix (P4/P6)
Mutect2 rejected the hg19 germline/PON: `incompatible contigs chrM ref=16571 vs features=16569`.
hg19's `chrM` is the old 16,571 bp sequence; b37's `MT` (renamed to chrM) is the 16,569 bp
rCRS. GATK reads contig lengths from the VCF **header**, so filtering records isn't enough.
**Fix applied:** rebuilt `resources/hg19/{af-only-gnomad,1000g_pon}.hg19.chr.vcf.gz` dropping
chrM (records + header `##contig`); `.withMT.*.bak` kept. Validated: Mutect2 now initializes
+ traverses cleanly on `p4.rdfcall.bam` chr22. DCIS/hg38 unaffected (chrM lengths match).

**Re-run P4/P6 (prep already done — only call+callback):**
```bash
cd /panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV
sbatch --job-name=ssnv_p4 slurm/run_dataset.sh configs/p4.env call,callback
sbatch --job-name=ssnv_p6 slurm/run_dataset.sh configs/p6.env call,callback
```

Resources/job: 16 CPU / 160 GB / 48 h, partition `interactive`, qos/acct `maiziezhou_lab_phd_int`.
Logs: `slurm/slurm_output/ssnv_<sample>-<jobid>.{out,err}`.

**Stages per job** (watch progress in the `.err`): BarcodeUMIBinding → MarkDuplicates →
AddOrReplaceReadGroups → SplitNCigarReads → BaseRecalibrator → ApplyBQSR (= prep) →
Mutect2 → FilterMutectCalls (= call) → CallBack (= matrix).

**Expected outputs per sample** (`results/<sample>/`):
- `<sample>.rdfcall.bam` (+`.bam.bai`) — BQSR'd BAM
- `<sample>.raw.vcf.gz` (Mutect2) and `<sample>.vcf.gz` (FilterMutectCalls, final)
- `matrix/<sample>_{alt,depth,ref}/` — 10x MTX triplets (SNV×spot counts).
  Features = `chrom_pos:ref>alt`; barcodes = Visium spots.

**Runtime expectation:** chr22-only prep took ~50 min, so whole-genome is long — DCIS
(27 GB BAM) likely most of a day; P4/P6 (7–9 GB) less. Peak RSS in smoke = 67 GB.

---

## DONE
- **2026-06-13 — chr22 smoke test on DCIS1 (job 11866762): PASSED.** Full chain validated
  (prep → Mutect2 [2,971 PASS SNVs on chr22] → CallBack → matrix 7,284 SNV × 2,276 spots).
  Confirmed Java17 fix, contig-matched hg38 resources accepted, `.bam.bai` indexing fix.
- Env rebuilt; Broad Mutect2 resources downloaded + contig-harmonized
  (`resources/hg38/*.nochr.vcf.gz`, `resources/hg19/*.chr.vcf.gz`); BQSR known-sites reuse
  af-only-gnomad (no separate dbSNP download).

## TODO (after runs finish)
- Build adapter: SpatialSNV `<sample>_alt` MTX → **binary spots×SNVs** matrix (binarize
  `alt>0`, transpose) to match the SPARCAL / Strelka2 / GATK benchmark matrix contract
  (binary int8, rows=barcodes, cols=`chrom_pos`). Cf. SPARCAL `claude.md` "Benchmark matrices".
- Compare SpatialSNV vs SPARCAL (+ truth sets: P4/P6 bulk-WES Mutect2/GATK SNP VCFs).
- **SpaceTracer** bring-up is separate and NOT started — see
  `/data/maiziezhou_lab/leiy4/SpaceTracer/CLAUDE.md`.
