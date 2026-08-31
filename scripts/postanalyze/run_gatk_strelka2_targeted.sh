#!/usr/bin/env bash
# Targeted GATK4 HaplotypeCaller + Strelka2 germline calling on P4/P6 tumor RNA
# BAMs, restricted to the exact P1-1 callable-site target set (truth-union-SPARCAL
# positions from data/germline_concordance_2026-08-23/{P4,P6}_targets.tsv), so the
# P1-1 comparison is three-way rather than self-referential.
#
# Non-destructive: reads existing shipped BAMs/VCFs read-only, writes ONLY under
# data/germline_and_contrasts_2026-08-DD/three_way_calls/. Does not touch
# scripts/1_calling..6_spatial_filter, data/<sample>/output_VCFs, or any frozen dir.
#
# Usage: run_gatk_strelka2_targeted.sh <P4|P6> <gatk|strelka2>
set -euo pipefail

SAMPLE=$1     # P4 or P6
CALLER=$2     # gatk or strelka2

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
OUTROOT=$PROJECT/data/germline_and_contrasts_2026-08-27
TARGETDIR=$PROJECT/data/germline_concordance_2026-08-23
REF=/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa

case $SAMPLE in
  P4) BAM=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam ;;
  P6) BAM=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam ;;
  *) echo "unknown sample $SAMPLE"; exit 1 ;;
esac

TARGETS=$TARGETDIR/${SAMPLE}_targets.tsv
OUTDIR=$OUTROOT/three_way_calls/$SAMPLE/$CALLER
mkdir -p $OUTDIR

if [ "$CALLER" = "gatk" ]; then
  JAVA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/java/17.0.6/bin/java
  GATKJAR=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/gatk/4.4.0.0/gatk-package-4.4.0.0-local.jar
  INTERVALS=$OUTDIR/${SAMPLE}.intervals
  awk '{print $1":"$2"-"$2}' $TARGETS > $INTERVALS
  echo "[$(date)] GATK HaplotypeCaller: $SAMPLE, $(wc -l < $INTERVALS) intervals"
  $JAVA -Xmx12g -jar $GATKJAR HaplotypeCaller \
    -R $REF -I $BAM \
    -L $INTERVALS --interval-padding 0 \
    --dont-use-soft-clipped-bases -stand-call-conf 10 \
    --disable-read-filter MappingQualityAvailableReadFilter \
    --mapping-quality-threshold-for-genotyping 0 \
    -O $OUTDIR/${SAMPLE}_gatk_targeted.vcf.gz \
    > $OUTDIR/gatk.log 2>&1
  echo "[$(date)] GATK done for $SAMPLE. Records:"
  zcat $OUTDIR/${SAMPLE}_gatk_targeted.vcf.gz | grep -vc "^#"

elif [ "$CALLER" = "strelka2" ]; then
  STRELKA_ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/strelka
  export PATH=$STRELKA_ENV/bin:$PATH
  STRELKA_CONFIG=$STRELKA_ENV/bin/configureStrelkaGermlineWorkflow.py
  BEDGZ=$OUTDIR/${SAMPLE}.callregions.bed.gz
  awk 'BEGIN{OFS="\t"}{print $1,$2-1,$2}' $TARGETS | sort -k1,1 -k2,2n > $OUTDIR/${SAMPLE}.callregions.bed
  $PROJECT/apps/bgzip -f $OUTDIR/${SAMPLE}.callregions.bed
  $PROJECT/apps/tabix -p bed $BEDGZ
  RUNDIR=$OUTDIR/run
  rm -rf $RUNDIR
  echo "[$(date)] Strelka2 configure: $SAMPLE"
  python $STRELKA_CONFIG --bam $BAM --referenceFasta $REF --runDir $RUNDIR \
    --callRegions $BEDGZ > $OUTDIR/strelka2_configure.log 2>&1
  echo "[$(date)] Strelka2 runWorkflow: $SAMPLE"
  python $RUNDIR/runWorkflow.py -m local -j 6 > $OUTDIR/strelka2_run.log 2>&1
  cp $RUNDIR/results/variants/genome.S1.vcf.gz $OUTDIR/${SAMPLE}_strelka2_targeted.vcf.gz 2>/dev/null || \
    cp $RUNDIR/results/variants/genome.vcf.gz $OUTDIR/${SAMPLE}_strelka2_targeted.vcf.gz
  echo "[$(date)] Strelka2 done for $SAMPLE. Records:"
  zcat $OUTDIR/${SAMPLE}_strelka2_targeted.vcf.gz | grep -vc "^#"
else
  echo "unknown caller $CALLER"; exit 1
fi

echo "[$(date)] DONE $SAMPLE $CALLER"
