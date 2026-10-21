#!/bin/bash
# Freeze full content identities for every Beagle reference panel used by SPARCAL.

#SBATCH --job-name=beagle_ref_hash
#SBATCH --output=slurm_output/beagle_finalization/reference_hash_%j.out
#SBATCH --error=slurm_output/beagle_finalization/reference_hash_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
ROOT="$PROJECT/data/experiments/beagle_reference_panel_checksums_20260913_v2"
HG19=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19
GRCH38=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38

cd "$PROJECT"
if [ -e "$ROOT" ]; then
    echo "ERROR: refusing to overwrite existing reference checksum root: $ROOT"
    exit 2
fi
mkdir -p "$ROOT/code_snapshot"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$ROOT/RUN_FAILED.txt"' ERR
cp run_slurm/hash_beagle_reference_panels_v1.sh "$ROOT/code_snapshot/"
sha256sum "$ROOT/code_snapshot/hash_beagle_reference_panels_v1.sh" > "$ROOT/CODE_SHA256SUMS.txt"

PATHS="$ROOT/REFERENCE_PATHS.txt"
for chrom in $(seq 1 22); do
    panel="$HG19/hg19_chr$chrom.vcf.gz"
    printf "%s\n%s\n" "$panel" "$panel.tbi" >> "$PATHS"
done
for chrom in $(seq 1 22); do
    panel="$GRCH38/CCDG_14151_B01_GRM_WGS_2020-08-05_chr$chrom.filtered.shapeit2-duohmm-phased.vcf.gz"
    printf "%s\n%s\n" "$panel" "$panel.tbi" >> "$PATHS"
done

while IFS= read -r path; do
    [ -s "$path" ] || {
        echo "ERROR: missing or empty reference artifact: $path"
        exit 1
    }
done < "$PATHS"

printf "path\tsize_bytes\tmtime_epoch\n" > "$ROOT/FILE_METADATA.tsv"
while IFS= read -r path; do
    printf "%s\t%s\t%s\n" "$path" "$(stat -c %s "$path")" "$(stat -c %Y "$path")" >> "$ROOT/FILE_METADATA.tsv"
done < "$PATHS"

while IFS= read -r path; do
    sha256sum "$path"
done < "$PATHS" > "$ROOT/REFERENCE_SHA256SUMS.txt"

{
    printf "schema=sparcal_beagle_reference_ledger_v1\n"
    printf "production_inputs_include_wes=false\n"
    printf "hg19_panel_count=22\n"
    printf "grch38_panel_count=22\n"
    printf "artifact_count=88\n"
    printf "created_utc=%s\n" "$(date --iso-8601=seconds)"
    printf "git_head=%s\n" "$(git rev-parse HEAD)"
} > "$ROOT/MANIFEST.txt"

sha256sum "$ROOT/CODE_SHA256SUMS.txt" "$ROOT/FILE_METADATA.tsv" "$ROOT/MANIFEST.txt" "$ROOT/REFERENCE_PATHS.txt" "$ROOT/REFERENCE_SHA256SUMS.txt" > "$ROOT/OUTPUT_SHA256SUMS.txt"
printf "complete\n" > "$ROOT/RUN_COMPLETE"
trap - ERR
