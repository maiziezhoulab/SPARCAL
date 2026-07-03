# CLAUDE.md — SpaceTracer benchmark setup

This documents the **current (broken) state** of the SpaceTracer bring-up and what
remains undone. SpaceTracer is one of two external SNV-calling tools being
benchmarked against **SPARCAL** (the other is **SpatialSNV**, set up under
`../SpatialSNV/` — that one is functional; use it as the template for finishing this).

**Status as of 2026-06-13: NOT FUNCTIONAL. Never successfully run.** The repo, configs,
and SLURM scripts are written, but the conda env is incomplete, the reference resources
failed to download, and no calling job has ever executed.

Goal: run SpaceTracer on **P4, P6, DCIS1, DCIS2** and produce spot×SNV matrices
comparable to the SPARCAL / SpatialSNV / Strelka2 / GATK matrices.

---

## What exists (done)

| Item | Path | State |
|------|------|-------|
| Cloned repo | `repo/` | github.com/douymLab/SpaceTracer @ `af00d16` (v2.0.0, PR #29 merged). Full `docs/` present. |
| Per-dataset configs | `configs/dcis1_config.yaml`, `configs/dcis2_config.yaml` | Written. **Only DCIS1/DCIS2 — P4 and P6 configs do NOT exist yet.** |
| SLURM run scripts | `slurm/run_dcis1.sh`, `slurm/run_dcis2.sh` | Written; both `source activate SpaceTracer_dcis` then `spacetracer run --config ...`. **No P4/P6 scripts.** |
| Install/download scripts | `install.sh`, `download_resources.sh`, `slurm/download_resources.sh` | Written; the resource download is **broken** (see below). |
| Env spec | `environment.yml` | python 3.9.23, pytorch 1.9.1 (cpuonly), scanpy 1.10.3, etc. CLI entry point `spacetracer = SpaceTracer.cli.main:main`. |

---

## What is broken / undone

### 1. Conda env is incomplete — CLI missing
- The env that exists is named **`spacetracer`** (python 3.9.25, only ~27 packages:
  anyio, biopython, biothings_client, httpcore… **no pytorch / scanpy / SpaGCN, no
  `spacetracer` binary**). `install.sh` was never run to completion.
- **Name mismatch:** `install.sh` and both SLURM scripts use env name
  **`SpaceTracer_dcis`**, which **does not exist**. So even the written SLURM jobs would
  die immediately at `source activate SpaceTracer_dcis`.
- **TODO:** run `bash install.sh` (creates env from `environment.yml` then
  `pip install -e repo`). Decide on ONE env name and make `install.sh`, both SLURM
  scripts, and any P4/P6 scripts agree. Verify `spacetracer --help` works afterward.
  Watch for the same Java/PATH and stub-env pitfalls hit during SpatialSNV setup
  (see `../SpatialSNV/` notes / the SPARCAL `claude.md`).

### 2. Reference resources never downloaded (the only job ever run — it failed)
- `resources/resources.tar` is a **662-byte JSON blob, not the ~7 GB tarball**.
  `download_resources.sh` requests
  `https://zenodo.org/api/records/19896967/files/resources.tar?download=1`, but that
  `/api/records/.../files/<name>` endpoint returns **file metadata JSON**, not content.
  `tar` then failed ("does not look like a tar archive"); `resources/hg38/` is empty.
  (The `…/files/resources.tar/content` variant returns **404** — the real download URL
  still needs to be resolved.)
- The archive (Zenodo record **19896967**, DOI 10.64898/2026.02.04.703493) should contain
  `hg38_resources.tar.zst` (+ `mm10_resources.tar.zst`); needs `zstd` + `tar` to extract
  into `resources/hg38/`.
- **TODO:** find the working Zenodo download URL (try the web record page
  `https://zenodo.org/records/19896967`, or the `files-archive` endpoint, or the
  per-file `links.self` in the JSON blob already saved in `resources/resources.tar`),
  re-download, extract, and confirm `resources/hg38/` is populated. The SLURM run
  scripts already guard on `resources/hg38` being non-empty.

### 3. P4 / P6 not configured
- `configs/` and `slurm/` only cover DCIS1/DCIS2. Need configs + run scripts for
  **P4 and P6** (hg19, `chr`-prefixed; Visium breast tumor). DCIS is hg38, **no `chr`
  prefix** (confirmed from the DCIS BAM header during SpatialSNV setup — note the
  dcis configs already say `genome: hg38` / "BAM contigs have NO chr prefix").
- SpaceTracer's hg38 resources are pre-packaged (Zenodo); **whether it ships/needs an
  hg19 resource bundle for P4/P6 is unknown — investigate.** The Zenodo tar only
  mentions hg38 + mm10. This may block P4/P6 entirely or require building hg19 resources.

### 4. Never run end-to-end
- `results/dcis1/` and `results/dcis2/` are **empty**. No calling has happened.
- **TODO:** after the env + resources are fixed, do a **small smoke test first**
  (one section, ideally subset) before launching full jobs — same approach used for
  SpatialSNV. SpaceTracer is a multi-step pipeline (clustering → BAM processing →
  mpileup → genotyping → spatial/read/RNA features → phasing → filtration → mutation
  prediction; see `repo/docs/steps/` and the `steps:` block in the config).

---

## Dataset paths (verified, reusable for P4/P6 configs)

- **DCIS1 BAM** (hg38, no-chr): `…/ST_datasets/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/` (spaceranger_dir; resolves BAM + tissue_positions + h5)
- **DCIS2 BAM**: `…/ST_datasets/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/`
- **P4 rep1 BAM** (hg19, chr): `…/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/`
- **P6 rep1 BAM** (hg19, chr): `…/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/`
  (`…` = `/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets`)
- hg38 ref: `/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa` (no chr, has `.fai`+`.dict`)
- hg19 ref: `/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa` (chr, has `.fai`+`.dict`)

## Other things to investigate

- **Alternate clone:** `install.sh` prefers an existing repo at
  `/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/SpaceTracer` (outside `snv_calling/`,
  exists, dated Apr 7). Check whether it's more complete / has a working env before
  re-cloning. There may be two diverging copies.
- **Comparison target:** SpaceTracer's matrix must end up as a spot×SNV binary `int8`
  matrix (rows = in-tissue barcodes, cols = `chrom_pos`) to be comparable — same shape
  contract as the SPARCAL/Strelka2/GATK matrices (see SPARCAL `claude.md`
  "Benchmark matrices"). Confirm SpaceTracer's `outputs` format (`repo/docs/outputs.md`)
  and write a projection/adapter if needed.

## Reference: how SpatialSNV (the sibling tool) was brought up
See `../SpatialSNV/` — env rebuilt from scratch (the pre-existing stub env was empty),
Broad Mutect2 resources downloaded + contig-harmonized per build, a parametrized driver
`scripts/run_spatialsnv.sh` + `configs/*.env` + `slurm/run_dataset.sh`, validated with a
single-chromosome smoke test before full runs. Mirror that workflow here.
