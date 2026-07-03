#!/usr/bin/env bash
# Harmonize Broad Mutect2 resource contigs to each reference build.
#   hg38 (DCIS, GRCh38-3.0.0, no-chr): chr1->1 ... chrX->X chrY->Y chrM->MT, main chroms only
#   hg19 (P4/P6, refdata-hg19-2.1.0, chr): 1->chr1 ... X->chrX Y->chrY MT->chrM, main chroms only
set -euo pipefail
BC=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/bcftools
TBX=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/tabix
BGZIP=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/bgzip
ROOT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/resources
DL=$ROOT/_dl
THREADS=8

# ---- rename maps ----
HG38_MAP=$ROOT/_dl/map_hg38_strip.txt   # chrN -> N
HG19_MAP=$ROOT/_dl/map_hg19_add.txt     # N -> chrN
: > "$HG38_MAP"; : > "$HG19_MAP"
for c in $(seq 1 22) X Y; do echo -e "chr${c}\t${c}" >> "$HG38_MAP"; echo -e "${c}\tchr${c}" >> "$HG19_MAP"; done
echo -e "chrM\tMT" >> "$HG38_MAP"
echo -e "MT\tchrM"  >> "$HG19_MAP"

HG38_REGIONS=$(printf "chr%s," $(seq 1 22)); HG38_REGIONS="${HG38_REGIONS}chrX,chrY,chrM"
B37_REGIONS=$(printf "%s," $(seq 1 22)); B37_REGIONS="${B37_REGIONS}X,Y,MT"
HG19_REGIONS=$(printf "chr%s," $(seq 1 22)); HG19_REGIONS="${HG19_REGIONS}chrX,chrY,chrM"

# ================= hg38 (DCIS) =================
echo "[hg38] af-only-gnomad: restrict main chroms (chr*) then strip chr ..."
$BC view --threads $THREADS -r "$HG38_REGIONS" "$DL/hg38/af-only-gnomad.hg38.vcf.gz" -Ou \
  | $BC annotate --threads $THREADS --rename-chrs "$HG38_MAP" -Oz -o "$ROOT/hg38/af-only-gnomad.hg38.nochr.vcf.gz"
$TBX -p vcf "$ROOT/hg38/af-only-gnomad.hg38.nochr.vcf.gz"
echo "[hg38] 1000g_pon ..."
$BC view --threads $THREADS -r "$HG38_REGIONS" "$DL/hg38/1000g_pon.hg38.vcf.gz" -Ou \
  | $BC annotate --threads $THREADS --rename-chrs "$HG38_MAP" -Oz -o "$ROOT/hg38/1000g_pon.hg38.nochr.vcf.gz"
$TBX -p vcf "$ROOT/hg38/1000g_pon.hg38.nochr.vcf.gz"

# ================= hg19 (P4/P6) from b37 =================
echo "[hg19] bgzip + index b37 af-only (14GB, slow) ..."
if [ ! -f "$DL/b37/af-only-gnomad.raw.sites.b37.vcf.gz" ]; then
  $BGZIP -@ $THREADS -c "$DL/b37/af-only-gnomad.raw.sites.b37.vcf" > "$DL/b37/af-only-gnomad.raw.sites.b37.vcf.gz"
fi
$TBX -f -p vcf "$DL/b37/af-only-gnomad.raw.sites.b37.vcf.gz"
echo "[hg19] af-only: restrict main chroms then add chr ..."
$BC view --threads $THREADS -r "$B37_REGIONS" "$DL/b37/af-only-gnomad.raw.sites.b37.vcf.gz" -Ou \
  | $BC annotate --threads $THREADS --rename-chrs "$HG19_MAP" -Oz -o "$ROOT/hg19/af-only-gnomad.hg19.chr.vcf.gz"
$TBX -p vcf "$ROOT/hg19/af-only-gnomad.hg19.chr.vcf.gz"

echo "[hg19] PON: detect compression, index, restrict, add chr ..."
PON_SRC="$DL/b37/Mutect2-WGS-panel-b37.vcf.gz"
if $BGZIP -t "$PON_SRC" 2>/dev/null; then echo "  PON is bgzipped"; else
  echo "  PON not bgzipped -> recompressing"; $BC view "$PON_SRC" -Oz -o "$DL/b37/pon_b37.bgz.vcf.gz"; PON_SRC="$DL/b37/pon_b37.bgz.vcf.gz"
fi
$TBX -f -p vcf "$PON_SRC"
$BC view --threads $THREADS -r "$B37_REGIONS" "$PON_SRC" -Ou \
  | $BC annotate --threads $THREADS --rename-chrs "$HG19_MAP" -Oz -o "$ROOT/hg19/1000g_pon.hg19.chr.vcf.gz"
$TBX -p vcf "$ROOT/hg19/1000g_pon.hg19.chr.vcf.gz"

echo "=== DONE. Outputs: ==="
ls -la "$ROOT/hg38/" "$ROOT/hg19/"
echo "=== contig sanity (first record each) ==="
for f in "$ROOT/hg38/af-only-gnomad.hg38.nochr.vcf.gz" "$ROOT/hg19/af-only-gnomad.hg19.chr.vcf.gz"; do
  echo "-- $f"; $BC view -H "$f" 2>/dev/null | head -1 | cut -f1-2
done
