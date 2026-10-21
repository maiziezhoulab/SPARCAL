#!/usr/bin/env python3
from __future__ import annotations
import gzip
import json
from pathlib import Path
import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
ROOT = PROJECT / "data/p4p6_caller_panel_2026-10-06"
SCRIPTS = PROJECT / "scripts/benchmark_p4p6"
LOGS = PROJECT / "slurm_output/benchmark_p4p6"
SECTIONS = ("P4_rep1", "P4_rep2", "P6_rep1", "P6_rep2")


def n_vcf(path: Path) -> int:
    total = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            total += bool(line and line[0] != "#")
    return total


def matrix_count(path: Path) -> tuple[int, int, int]:
    frame = pd.read_pickle(path)
    return frame.shape[0], frame.shape[1], int(frame.to_numpy(copy=False).sum())


def read_optional(path: Path) -> str:
    return path.read_text().strip() if path.exists() else "not recorded"


def main():
    rows = []
    for section in SECTIONS:
        row = {"section": section}
        for caller in ("strelka2", "gatk"):
            directory = ROOT / section / caller
            row[f"{caller}_pass"] = n_vcf(directory / f"{section}_{caller}_pass_snvs.vcf.gz")
            row[f"{caller}_matrix"] = matrix_count(directory / f"{section}_{caller}_presence_matrix.pkl")
            row[f"{caller}_no1kgp"] = matrix_count(directory / f"{section}_{caller}_no1kgp_presence_matrix.pkl")
        spatial = ROOT / section / "spatialsnv"
        if spatial.exists():
            row["spatialsnv"] = (
                matrix_count(spatial / f"{section}_spatialsnv_presence_matrix.pkl"),
                matrix_count(spatial / f"{section}_spatialsnv_presence_filtered_matrix.pkl"),
            )
        rows.append(row)

    smoke = json.loads((ROOT / "_smoke_chr21/smoke_validation.json").read_text())
    job_smoke = read_optional(ROOT / "job_ids_smoke.tsv")
    job_full = read_optional(ROOT / "job_ids_full.tsv")
    count_lines = []
    for row in rows:
        s = row["section"]
        st = row["strelka2_matrix"]
        stn = row["strelka2_no1kgp"]
        ga = row["gatk_matrix"]
        gan = row["gatk_no1kgp"]
        if "spatialsnv" in row:
            spo, spf = row["spatialsnv"]
            spatial_text = f"{spo[1]:,} / {spf[1]:,}"
        else:
            spatial_text = "not run (rep1)"
        count_lines.append(
            f"| {s} | {st[0]:,} | {row['strelka2_pass']:,} | {st[1]:,} | {stn[1]:,} | "
            f"{row['gatk_pass']:,} | {ga[1]:,} | {gan[1]:,} | {spatial_text} |"
        )

    smoke_lines = []
    for caller in ("strelka2", "gatk"):
        item = smoke["callers"][caller]
        smoke_lines.append(
            f"| {caller} | {item['pass_vcf']['pass_biallelic_snvs']:,} | "
            f"{item['matrix']['shape'][1]:,} | {item['matrix']['presences']:,} | "
            f"{item['no1kgp_matrix']['shape'][1]:,} |"
        )
    spatial = smoke["callers"]["spatialsnv"]
    smoke_lines.append(
        f"| spatialsnv | {spatial['chr21_vcf']['pass_biallelic_snvs']:,} | "
        f"{spatial['presence_shape'][1]:,} | n/a | {spatial['filtered_shape'][1]:,} filtered |"
    )

    readme = f"""# P4/P6 caller panel (2026-10-06)

All inputs were treated as read-only. All calls use the chr-prefixed hg19 reference
`/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa`.

## What was run

- SpatialSNV 1.1.0: CallBack only for P4_rep2 and P6_rep2, using the existing section VCF and prepared RDFCall BAM. Only the ALT CallBack matrix was converted. The converter imports the established `SPARCAL_Benchmarking/callback_to_pkl.py`; filtered means PASS intersect biallelic SNV with its default 1-spot/1-ALT-UMI floors.
- Strelka 2.9.10 (bioconda env `strelka`): plain germline workflow on each whole possorted BAM; no `--rna`; retained PASS biallelic SNVs.
- GATK 4.6.2.0 (env `spatialsnv`): SplitNCigarReads, chromosome-scattered HaplotypeCaller (`--dont-use-soft-clipped-bases true -stand-call-conf 20`), GatherVcfs, RNA hard filters FS>30, QD<2, cluster window 35/size 3, then PASS biallelic SNVs. No BQSR was performed because no cross-caller-matched known-sites set was specified.
- Strelka2 and GATK projection: identical CB-tag BAM pileup, exact ALT, >=1 ALT read, in-tissue spots only, preserving the existing projector's read/pileup filters. Matrices are pandas uint8, rows=barcodes, columns=`chrom_pos_ref_alt` without `chr`.
- 1KGP removal: exact CHROM/POS/REF/ALT membership copied from `somatic_callset_validation_2026-09-07/run_analysis.py`. Panel header verified as 1000 Genomes Phase 3 b37/hs37d5, coordinate-compatible with hg19 autosomes.

## Exact command forms

```bash
# SpatialSNV CallBack
spatialsnvtools CallBack -b PREPARED.rdfcall.bam -v EXISTING.vcf.gz -s SECTION \\
  -o CALLBACK_DIR --tmpdir TMP --only_autosome -c CB -u UB -@ 16
python callback_to_pkl_adapter.py --matrix-dir CALLBACK_DIR --sample SECTION \\
  --vcf EXISTING.vcf.gz --outdir SECTION/spatialsnv

# Strelka2
python $STRELKA_CONFIG --bam POSSORTED.bam --referenceFasta genome.fa --runDir WORKFLOW \\
  [--callRegions chr21.callRegions.bed.gz]
python WORKFLOW/runWorkflow.py -m local -j 8
bcftools view -f PASS -m2 -M2 -v snps -Oz -o SECTION_strelka2_pass_snvs.vcf.gz SOURCE.vcf.gz

# GATK
gatk --java-options -Xmx40g SplitNCigarReads -R genome.fa -I POSSORTED.bam -O SPLIT.bam [ -L chr21 ]
gatk --java-options -Xmx26g HaplotypeCaller -R genome.fa -I SPLIT.bam -L CHROM \\
  --dont-use-soft-clipped-bases true -stand-call-conf 20 --native-pair-hmm-threads 4 -O SHARD.vcf.gz
gatk GatherVcfs -I SHARD1.vcf.gz ... -O GATHERED.vcf.gz
gatk VariantFiltration -R genome.fa -V GATHERED.vcf.gz -O FILTERED.vcf.gz \\
  --filter-name RNA_FS --filter-expression 'FS > 30.0' \\
  --filter-name RNA_QD --filter-expression 'QD < 2.0' --cluster-window-size 35 --cluster-size 3
bcftools view -f PASS -m2 -M2 -v snps -Oz -o SECTION_gatk_pass_snvs.vcf.gz FILTERED.vcf.gz

# Projection and exact-allele 1KGP removal
python project_vcf_to_spots.py --vcf PASS.vcf.gz --bam POSSORTED.bam \\
  --tissue-positions tissue_positions.csv --outdir CALLER_DIR --section SECTION \\
  --caller CALLER --min-alt-reads 1 --max-workers 8
python drop_1kgp.py --matrix STRELKA_MATRIX.pkl --matrix GATK_MATRIX.pkl
```

The literal executable scripts, arguments, resource requests, and path maps are under `scripts/benchmark_p4p6/`.

## Counts per section

Matrix counts are columns; spot count is identical across Strelka2/GATK for each section. SpatialSNV is shown as original / filtered columns.

| section | in-tissue spots | Strelka PASS SNVs | Strelka observed | Strelka no1KGP | GATK PASS SNVs | GATK observed | GATK no1KGP | SpatialSNV original / filtered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(count_lines)}

## chr21 smoke test (P4_rep2)

Status: **{smoke['status']}**. SpatialSNV rep1 builder reproduction was exact, including the required 4,855 x 45,038 filtered shape, barcode/column order, values, and uint8 dtype.

| caller | PASS biallelic SNVs | observed matrix SNVs | spot-SNV presences | after 1KGP / filtered |
|---|---:|---:|---:|---:|
{chr(10).join(smoke_lines)}

## Submitted job IDs

Smoke:

```text
{job_smoke}
```

Full:

```text
{job_full}
```

## Deviations

No method-level deviations. `GatherVcfs` was selected from the allowed GatherVcfs/MergeVcfs alternatives. Two smoke-test plumbing issues were repaired before full submission: the archived SpatialSNV RDFCall index uses `.bai`, so read-only BAM/index symlinks with the `.bam.bai` name expected by CallBack are created inside this output tree; and block-copy `GatherVcfs` cannot create an index even with `--CREATE_INDEX true`, so `gatk IndexFeatureFile` is run explicitly before VariantFiltration. First failed smoke jobs were 14433061 (CallBack index name), 14433065 and 14433136 (gathered-VCF indexing). During full GATK `SplitNCigarReads`, P4 tasks 14433318_0 and 14433318_1 wrote complete BAM/index outputs and logged normal GATK completion, then conda OpenJDK 17 raised SIGBUS in `_dl_fini` during JVM shutdown (exit 250). `samtools` 1.23 `quickcheck` validated all four split BAMs in repair array 14439867, so downstream calling reused those complete outputs without repeating the split. Full projection job 14439893 exposed Strelka hg19 decoy-contig names containing underscores; the matrix-key sorter was corrected to parse `chrom_pos_ref_alt` from the right, dry-checked, and projection was resubmitted as 14444011 with summary 14444012. SpatialSNV raw CallBack MTX files, Strelka workflow files, GATK split BAMs, and chromosome shards are retained as provenance/intermediates beneath each caller directory.
"""
    readme = readme.replace("\n+", "\n")
    (ROOT / "README.md").write_text(readme)

    manifest_paths = []
    for base in (ROOT, SCRIPTS, LOGS):
        manifest_paths.extend(path for path in base.rglob("*") if path.is_file())
    manifest = ROOT / "files_created.tsv"
    with manifest.open("w") as handle:
        handle.write("path\tbytes\n")
        for path in sorted(set(manifest_paths) | {manifest}):
            size = path.stat().st_size if path.exists() else 0
            handle.write(f"{path}\t{size}\n")
    print(f"Wrote {ROOT / 'README.md'}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
