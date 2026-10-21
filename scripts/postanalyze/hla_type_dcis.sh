#!/bin/bash
# HLA-type the DCIS Visium sections with arcasHLA (env hla_typing).
#
# WHY NOT `arcasHLA extract`: it pulls ALL of chromosome 6 (~16-18 GB of SAM per
# section here) and then name-sorts it, which failed on this data.  HLA typing
# only needs reads over the HLA genes, so we cut the region directly.  GRCh38
# HLA class I+II span ~chr6:29.6-33.1 Mb; 29.0-33.6 Mb is used for margin.
# The DCIS BAMs use BARE contig names (no "chr" prefix).
#
# Trade-off vs stock arcasHLA extract: unmapped reads and chr6 alt/decoy contigs
# are NOT included, so a read whose true HLA allele diverges enough to fail to
# map to the region is missed.  That biases toward the reference allele, which
# is the conservative direction for our question (it can only UNDER-state
# divergence from GRCh38).
set -euo pipefail

# NOTE: the `hla_typing` env ships samtools 1.2 (bioconda pins it for arcasHLA),
# which has no `samtools fastq` and no `sort -m`.  Use the modern samtools from
# the `spacetracer` env for the BAM handling; arcasHLA itself still runs from
# `hla_typing`.
SAMTOOLS=${SAMTOOLS:-/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spacetracer/bin/samtools}

OUT=data/hla_typing_2026-09-07
REGION="6:29000000-33600000"
D=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium
THREADS=${THREADS:-8}

for S in DCIS1 DCIS2; do
  BAM=$D/$S/spaceranger_align_${S}_hg38/${S}_output/outs/possorted_genome_bam.bam
  mkdir -p $OUT/$S
  echo "=== $S: cutting $REGION ==="
  $SAMTOOLS view -b -F 4 -@ $THREADS "$BAM" "$REGION" -o $OUT/$S/hla_region.bam
  $SAMTOOLS sort -n -@ $THREADS -m 2G -T $OUT/$S/sorttmp \
      $OUT/$S/hla_region.bam -o $OUT/$S/hla_region.qsort.bam
  $SAMTOOLS fastq -@ $THREADS -0 $OUT/$S/hla.fq.gz -s /dev/null -n \
      $OUT/$S/hla_region.qsort.bam
  echo "  reads: $($SAMTOOLS view -c -@ $THREADS $OUT/$S/hla_region.bam)"
  echo "=== $S: genotyping ==="
  arcasHLA genotype --single -t $THREADS -o $OUT/$S -v \
      $OUT/$S/hla.fq.gz 2>&1 | grep -viE "SyntaxWarning|^  " | tail -20
  rm -f $OUT/$S/hla_region.qsort.bam
done
echo "=== results ==="
for S in DCIS1 DCIS2; do echo "--- $S ---"; cat $OUT/$S/*.genotype.json 2>/dev/null; echo; done
