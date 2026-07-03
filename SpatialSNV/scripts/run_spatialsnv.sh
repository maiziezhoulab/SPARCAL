#!/usr/bin/env bash
# =============================================================================
# SpatialSNV pipeline driver (10x Visium)
#   Stage 1  PerpareBAMforCalling : BarcodeUMIBinding -> MarkDup -> AddRG ->
#                                   SplitNCigarReads -> BQSR -> {sample}.rdfcall.bam
#   Stage 1b samtools index .rdfcall.bam  (tool writes .bai, CallBack wants .bam.bai)
#   Stage 2  SNVCalling           : Mutect2 (+germline +PON) -> FilterMutectCalls
#   Stage 3  CallBack             : project PASS-SNVs onto spots -> spot x SNV matrix
#
# Config is sourced from $1 (a *.env file). Optional SUBSET_REGION runs a fast
# single-contig smoke test (subsets the BAM first). STAGES selects which stages.
# =============================================================================
set -euo pipefail

CONFIG="${1:?usage: run_spatialsnv.sh <config.env> [stages]}"
STAGES="${2:-prep,call,callback}"
source "$CONFIG"

ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv
export PATH="$ENV/bin:$PATH"   # gatk/picard wrappers call `java` unqualified -> need env java on PATH
PICARD_JAR="$ENV/share/picard-3.4.0-0/picard.jar"
GATK="$ENV/bin/gatk"
SAMTOOLS="$ENV/bin/samtools"
SST="$ENV/bin/spatialsnvtools"

: "${THREADS:=16}"
: "${BARCODE_PREP:=CR}"; : "${UMI_PREP:=UR}"
: "${BARCODE_CB:=CB}";  : "${UMI_CB:=UB}"
: "${DBSNP:=$GERMLINE}"          # reuse af-only-gnomad as BQSR known-sites
mkdir -p "$OUTDIR" "$OUTDIR/tmp" "$OUTDIR/matrix"

INPUT_BAM="$BAM"
CHROM_ARG=()
# ---- optional smoke-test subset ----
if [ -n "${SUBSET_REGION:-}" ]; then
  echo "### SMOKE: subsetting BAM to region '$SUBSET_REGION'"
  SUB="$OUTDIR/${SAMPLE}.subset.bam"
  "$SAMTOOLS" view -b -@ "$THREADS" "$BAM" "$SUBSET_REGION" -o "$SUB"
  "$SAMTOOLS" index -@ "$THREADS" "$SUB"
  INPUT_BAM="$SUB"
  CHROM_ARG=(-L "$SUBSET_REGION")
elif [ -n "${CHROM:-}" ]; then
  CHROM_ARG=(-L "$CHROM")
fi

RDFCALL="$OUTDIR/${SAMPLE}.rdfcall.bam"
OUTVCF="$OUTDIR/${SAMPLE}.vcf.gz"

run_stage () { [[ ",$STAGES," == *",$1,"* ]]; }

echo "================ SpatialSNV : $SAMPLE ================"
echo "config=$CONFIG  stages=$STAGES  threads=$THREADS"
echo "bam=$INPUT_BAM"; echo "fasta=$FASTA"; echo "germline=$GERMLINE"; echo "pon=$PON"
echo "out=$OUTDIR"; date

# ---------------- Stage 1: prep ----------------
if run_stage prep; then
  echo "### [1] PerpareBAMforCalling"
  "$SST" PerpareBAMforCalling \
      -b "$INPUT_BAM" -o "$OUTDIR" -s "$SAMPLE" \
      -f "$FASTA" -d "$DBSNP" \
      -c "$BARCODE_PREP" -u "$UMI_PREP" \
      -@ "$THREADS" \
      --samtools "$SAMTOOLS" --picard "$PICARD_JAR" --gatk "$GATK" \
      --tmpdir "$OUTDIR/tmp"
  [ -s "$RDFCALL" ] || { echo "FATAL: $RDFCALL not produced (check log above; RunCMD swallows GATK errors)"; exit 1; }
  echo "### [1b] index $RDFCALL"
  "$SAMTOOLS" index -@ "$THREADS" "$RDFCALL"
  echo "prep done: $(date)"
fi

# ---------------- Stage 2: call ----------------
if run_stage call; then
  echo "### [2] SNVCalling (Mutect2 + FilterMutectCalls)"
  [ -f "${RDFCALL}.bai" ] || "$SAMTOOLS" index -@ "$THREADS" "$RDFCALL"
  "$SST" SNVCalling \
      -s tumor -b "$RDFCALL" \
      -o "$OUTVCF" -f "$FASTA" \
      --pon "$PON" --germline "$GERMLINE" \
      --gatk "$GATK" "${CHROM_ARG[@]}"
  [ -s "$OUTVCF" ] || { echo "FATAL: $OUTVCF not produced"; exit 1; }
  echo "PASS SNV count: $($ENV/bin/bcftools view -H -f PASS "$OUTVCF" 2>/dev/null | awk 'length($4)==1&&length($5)==1' | wc -l)"
  echo "call done: $(date)"
fi

# ---------------- Stage 3: callback ----------------
if run_stage callback; then
  echo "### [3] CallBack -> spot x SNV matrix"
  "$SST" CallBack \
      -b "$RDFCALL" -v "$OUTVCF" -s "$SAMPLE" \
      -o "$OUTDIR/matrix" --tmpdir "$OUTDIR/tmp_callback" \
      --only_autosome \
      -c "$BARCODE_CB" -u "$UMI_CB" -@ "$THREADS"
  echo "callback done: $(date)"
  ls -la "$OUTDIR/matrix"
fi
echo "================ DONE: $SAMPLE $(date) ================"
