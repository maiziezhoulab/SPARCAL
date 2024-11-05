#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <input_bam> <num_reads_per_batch> <num_threads> <output_dir> <shell_path>"
  exit 1
fi

inbam=$(realpath $1)
num_reads_per_batch=$2
t=$3
out=$4
shell_path=$5

# Ensure the output directory exists
mkdir -p $out

# Create necessary directories
mkdir -p $out/bam_by_batch
mkdir -p $out/bam_bycell

# Split bam into batches
samtools view $inbam | split -l ${num_reads_per_batch} - $out/cells
samtools view -H $inbam > $out/header

ls $out/cells* | xargs -I {} -P $t sh -c "echo {}; cat $out/header {} | samtools view -Sb > $out/bam_by_batch/\$(basename {}).bam"

rm $out/cells*

### Split batched bam by cell
ls $out/bam_by_batch/*bam | xargs -I {} -P $t sh -c "bash $shell_path/process_one_bam.sh {} $out"

# Extract all unique cell names
ls $out/process_by_batch/cells*/ | sort | uniq | grep -v "cel" | tail -n +2 | sed "s/.sam//g" > $out/uniq_cell.txt

# Collect all cell bams from batch processing result
cat $out/uniq_cell.txt | xargs -I {} -P $t sh -c "echo {}; cat $out/header $out/process_by_batch/cells*/{}.sam | samtools view -Sb | samtools sort -o $out/bam_bycell/{}.bam; samtools index $out/bam_bycell/{}.bam"

# Remove intermediate results
rm -r $out/bam_by_batch $out/process_by_batch $out/header
