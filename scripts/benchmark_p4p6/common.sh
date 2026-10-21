#!/usr/bin/env bash
# Shared, read-only input map for the P4/P6 caller panel.

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
OUTROOT="$PROJECT/data/p4p6_caller_panel_2026-10-06"
LOGROOT="$PROJECT/slurm_output/benchmark_p4p6"
SCRIPTROOT="$PROJECT/scripts/benchmark_p4p6"
DATA=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data
REF=/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa
STRELKA_ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/strelka
SPATIALSNV_ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv
SNV_CALLER_ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller
BCFTOOLS="$SPATIALSNV_ENV/bin/bcftools"
SAMTOOLS="$SPATIALSNV_ENV/bin/samtools"
GATK="$SPATIALSNV_ENV/bin/gatk"
BGZIP="$PROJECT/apps/bgzip"
TABIX="$PROJECT/apps/tabix"

SECTIONS=(P4_rep1 P4_rep2 P6_rep1 P6_rep2)
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY)

section_bam() {
    case "$1" in
        P4_rep1) echo "$DATA/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam" ;;
        P4_rep2) echo "$DATA/P4_Visium/spaceranger_align_rep2_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam" ;;
        P6_rep1) echo "$DATA/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam" ;;
        P6_rep2) echo "$DATA/P6_Visium/spaceranger_align_rep2_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam" ;;
        *) echo "Unknown section: $1" >&2; return 2 ;;
    esac
}

section_outs() {
    dirname "$(section_bam "$1")"
}

section_positions() {
    local outs
    outs=$(section_outs "$1")
    if [[ -f "$outs/spatial/tissue_positions.csv" ]]; then
        echo "$outs/spatial/tissue_positions.csv"
    elif [[ -f "$outs/spatial/tissue_positions_list.csv" ]]; then
        echo "$outs/spatial/tissue_positions_list.csv"
    else
        echo "No tissue positions file for $1 under $outs/spatial" >&2
        return 2
    fi
}

mode_root() {
    case "$1" in
        smoke) echo "$OUTROOT/_smoke_chr21" ;;
        full) echo "$OUTROOT" ;;
        *) echo "Unknown run mode: $1" >&2; return 2 ;;
    esac
}

require_file() {
    [[ -s "$1" ]] || { echo "Required nonempty file missing: $1" >&2; exit 1; }
}

caller_dir() {
    local mode=$1 section=$2 caller=$3
    echo "$(mode_root "$mode")/$section/$caller"
}
