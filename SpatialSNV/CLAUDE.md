# CLAUDE.md — SpatialSNV benchmark

Guidance for working with SpatialSNV in this repo. **SpatialSNV is FUNCTIONAL** (validated
end-to-end 2026-06-13) — one of two external SNV callers benchmarked against **SPARCAL**
(the other is **SpaceTracer**, not yet started — see
`/data/maiziezhou_lab/leiy4/SpaceTracer/CLAUDE.md`).

Tool: github.com/YoungLi88/SpatialSNV — a **Mutect2-based** spatial-transcriptomics SNV
caller. Paper: GigaScience giaf065 / PMC12166308.

**Goal:** run on **P4, P6, DCIS1, DCIS2** and produce spot×SNV matrices comparable to the
SPARCAL / Strelka2 / GATK matrices (binary `int8`, rows = in-tissue barcodes, cols = `chrom_pos`).

**Live status / job tracking: see [On_going.md](On_going.md).** As of 2026-06-13 the chr22
smoke test passed and 4 whole-genome runs are running.

---

## Layout

```
SpatialSNV/
  repo/                      cloned tool (pip-installed editable into the env)
  scripts/run_spatialsnv.sh  pipeline driver (prep -> Mutect2/Filter -> CallBack)
  configs/{dcis1,dcis2,p4,p6}.env   per-dataset variables (sourced by the driver)
  slurm/run_dataset.sh       generic sbatch wrapper: run_dataset.sh <config.env> [stages] [subset_region]
  slurm/slurm_output/        job logs ssnv_<sample>-<jobid>.{out,err}
  resources/{hg38,hg19}/     contig-harmonized Mutect2 resources (see below)
  resources/_dl/             raw Broad downloads (kept; can be deleted to reclaim ~25 GB)
  reheader_resources.sh      one-shot script that built resources/{hg38,hg19}
  results/<sample>/          outputs
```

## Environment — `spatialsnv` (conda)
`/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv`. Built fresh (the
pre-existing env was an empty stub). Contents: **python 3.10.14, spatialsnv 1.1.0
(`spatialsnvtools` CLI), GATK 4.6.2.0, samtools 1.23.1, picard 3.4.0, bcftools 1.23.1,
openjdk 17**. Activate with `source activate spatialsnv`.

**Build recipe** (if it ever needs rebuilding): `conda create -n spatialsnv -c conda-forge
-c bioconda python=3.10.14 gatk4 samtools picard pip pycairo cairo` → `pip install -e repo`
→ `conda install -c bioconda -c conda-forge bcftools "openjdk=17"`. Do **not** use the repo
`requirements.yaml` directly — `igraph=0.11.5`/`matplotlib=3.7.5` are PyPI versions that
fail the conda solver; let `pip install -e repo` pull the Python deps.

### Environment gotchas (important)
- **GATK/picard need OpenJDK 17 AND the env on PATH.** The env originally shipped a broken
  OpenJDK 25 (`libjli.so` missing). Even after installing openjdk 17, the `gatk`/`picard`
  wrappers call `java` **unqualified**, so the env's `bin` must be on PATH. The driver does
  `export PATH="$ENV/bin:$PATH"`; SLURM jobs `source activate spatialsnv`. Without this GATK
  dies with `libjli.so: cannot open shared object file`.
- The tool's `--picard` flag wants a **JAR** (`$ENV/share/picard-3.4.0-0/picard.jar`), not
  the conda `picard` wrapper. `--gatk` wants the wrapper (`$ENV/bin/gatk`).

## Reference resources
References are local & GATK-ready (`.fa`+`.fai`+`.dict`):
- **DCIS = hg38**, contigs **no `chr`** → `/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa`
- **P4/P6 = hg19**, contigs **with `chr`** → `/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa`
(Verified from BAM headers. The genome build / contig prefix is the #1 failure mode.)

**Mutect2 resources** (`--pon` and `--germline` are **both REQUIRED** by the code, despite
the README calling them optional). Downloaded from the Broad somatic bundle and
**contig-harmonized to each ref** by `reheader_resources.sh` (restrict to main chroms +
rename), giving:
- `resources/hg38/af-only-gnomad.hg38.nochr.vcf.gz` (germline), `resources/hg38/1000g_pon.hg38.nochr.vcf.gz` (chr→1,2,…,MT)
- `resources/hg19/af-only-gnomad.hg19.chr.vcf.gz` (from b37), `resources/hg19/1000g_pon.hg19.chr.vcf.gz` (1,2,…,MT→chr…; b37≡hg19 coords for chr1–22)
- **BQSR known-sites (`--dbsnp`, required) reuse the af-only-gnomad file** — avoids a ~10 GB
  dbSNP download; a valid known-sites VCF.

## Pipeline (what the driver runs)
`scripts/run_spatialsnv.sh <config.env> [stages] ` — stages default `prep,call,callback`:
1. **prep** = `spatialsnvtools PerpareBAMforCalling` — BarcodeUMIBinding (tag `LY=barcode-umi`
   from **CR/UR**) → picard MarkDuplicates (dedup on `LY`) → AddReadGroups (SM=tumor) →
   GATK SplitNCigarReads → BaseRecalibrator → ApplyBQSR → `<sample>.rdfcall.bam`.
   Then `samtools index` it (see gotcha).
2. **call** = `spatialsnvtools SNVCalling` — Mutect2 (`-tumor tumor`, `--germline`, `--pon`,
   minMapQ 10) → FilterMutectCalls → `<sample>.vcf.gz` (raw kept as `<sample>.raw.vcf.gz`).
3. **callback** = `spatialsnvtools CallBack` — pileup at PASS biallelic SNVs (excludes
   weak_evidence/germline/strand_bias/slippage/contamination), assign reads to spots by
   **CB/UB** tags (`--only_autosome`) → 10x MTX matrices.

Smoke testing: pass a contig as 3rd arg, e.g. `run_dataset.sh configs/dcis1.env prep,call,callback 22`
→ subsets the BAM to that contig first (fast). The driver subsets + sets Mutect2 `-L`.

### Pipeline gotchas
- **`RunCMD` swallows non-zero exits** (prints stderr, does not raise). A failed GATK step
  won't stop the pipeline — the driver therefore checks that each expected output exists.
- **GATK ApplyBQSR writes `<x>.bai`, but CallBack/Mutect2 want `<x>.bam.bai`** → the driver
  runs `samtools index` after prep.
- **`-Xmx100g` is hardcoded** in every GATK/picard call (`repo/spatialsnvtools/utils.py`),
  so jobs need ≥~110 GB mem (we use 160 GB; smoke peak RSS = 67 GB) and gatk steps are
  effectively serial — **don't run multiple gatk invocations concurrently** (each grabs 100 g).
- **Mutect2 is single-threaded per call** (the tool hardcodes `threads=1` for SNVCalling;
  only `--native-pair-hmm-threads 10`). Whole-genome is slow — chr22-only prep took ~50 min.
- **`sbatch --export=ALL,STAGES=a,b,c` is broken** — commas split into separate vars. Use
  the **positional** form (`run_dataset.sh <config> <stages> <region>`) instead.

## Outputs (per sample, `results/<sample>/`)
- `<sample>.rdfcall.bam` (+`.bam.bai`), `<sample>.raw.vcf.gz`, `<sample>.vcf.gz` (final, filtered)
- `matrix/<sample>_{alt,depth,ref}/` — **10x MatrixMarket triplets** (`barcodes.tsv.gz`,
  `features.tsv.gz`, `matrix.mtx.gz`), orientation **SNV×spot** (integer counts).
  Features = `chrom_pos:ref>alt` (e.g. `22_19724571:C>T`); barcodes = Visium spots (`…-1`).
  alt = ALT-supporting read count, ref = REF count, depth = total.

## Benchmark adapter (TODO, after runs finish)
SpatialSNV emits **count** matrices, SNV×spot. To compare with SPARCAL/Strelka2/GATK
(binary `int8`, **spots×SNVs**, cols=`chrom_pos`): take `<sample>_alt`, **binarize
`alt>0`**, **transpose**, trim feature id to `chrom_pos`, intersect to the shared in-tissue
spot set. Analogous to the Strelka2/GATK spot-projection in the SPARCAL `claude.md`
("Benchmark matrices"). Truth sets for P4/P6: bulk-WES Mutect2/GATK SNP VCFs under
`…/Datasets/ST_datasets/STmut_Data/P{4,6}_Somatic_{Mutect2,GATK}/` (+ exome BEDs).

## Datasets (rep1 used for P4/P6, matching the SPARCAL benchmark sections)
`…` = `/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets`
- DCIS1: `…/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.bam`
- DCIS2: `…/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.bam`
- P4 rep1: `…/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam`
- P6 rep1: `…/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam`
All carry CB/CR/UB/UR tags (standard 10x Visium).
