#!/bin/bash
#################
#SBATCH --job-name=snv_perf_test
#################
#SBATCH --output=perf_test.out
#################
#SBATCH --error=perf_test.err
#################
#SBATCH --time=24:00:00
#SBATCH --account=cgw_maizie
#SBATCH --partition=cgw-maizie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#################
#SBATCH --mem=200GB
#################
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.cheng@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load GCC/11.3.0
# module load BCFtools/1.18
# module load SAMtools/1.18
module load Anaconda3
source activate snv_caller

# Define parameters
spatialid="151507"
project_dir="/data/maiziezhou_lab/yuqi/snv_calling"

# Define paths
BAM_DIR="/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/${spatialid}/bam_bycell"
REFERENCE_SEQ="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BEDFILE="/data/maiziezhou_lab/yuqi/snv_calling/data/reference/GRCh38.bed"
HEADER="/data/maiziezhou_lab/yuqi/snv_calling/data/reference/header.txt"
OLD_CALLER="${project_dir}/scripts/calling/self_caller.py"
NEW_CALLER="${project_dir}/scripts/calling/new_caller.py"
PERF_DIR="${project_dir}/data/dlpfc/${spatialid}/performance_test"
export PATH="/data/maiziezhou_lab/yuqi/snv_calling/apps:$PATH"
export LD_LIBRARY_PATH="/data/maiziezhou_lab/yuqi/snv_calling/apps:$LD_LIBRARY_PATH"
# Remember to add LD library path to the environment

# Monopogen Samtools paths
APP_DIR="/data/maiziezhou_lab/yuqi/snv_calling/apps"
SAMTOOLS="${APP_DIR}/samtools"
BCFTOOLS="${APP_DIR}/bcftools"
BGZIP="${APP_DIR}/bgzip"

# Create directories
mkdir -p ${PERF_DIR}/{old_caller,new_caller,mpileup,logs}
# mkdir -p ${PERF_DIR}/mpileup/

# Create performance log file
PERF_LOG="${PERF_DIR}/performance_comparison.log"
echo "Performance Comparison Log - $(date)" > ${PERF_LOG}
echo "=================================" >> ${PERF_LOG}

# Randomly select 5 BAM files
cd ${BAM_DIR}
selected_bams=($(ls *.bam | shuf -n 1))
echo "Selected BAM files:" >> ${PERF_LOG}
printf '%s\n' "${selected_bams[@]}" >> ${PERF_LOG}
echo "=================================" >> ${PERF_LOG}

# Create BAM_LIST file
BAM_LIST="${PERF_DIR}/bam_list.txt"
> ${BAM_LIST}
echo "Creating BAM list file: ${BAM_LIST}"
for bam in "${selected_bams[@]}"; do
    echo "${BAM_DIR}/${bam}" >> ${BAM_LIST}
done
# Show the contents of the BAM_LIST file
echo "Contents of BAM list file:"
cat ${BAM_LIST}

# # Function to run mpileup for a single BAM file
#!/bin/bash

# Function to run mpileup for a single BAM file
run_mpileup() {
    bam_file=$1
    basename=$(basename ${bam_file} .bam)
    # bamlist="${PERF_DIR}/logs/${basename}_bamlist.txt"
    output_vcf="${PERF_DIR}/mpileup/${basename}.vcf"
    log_file="${PERF_DIR}/logs/${basename}_mpileup.log"
    
    echo "[$(date)] Running mpileup for ${basename}"
    
    # Time the mpileup process
    start_time=$(date +%s.%N)
    
    # Construct and execute the command
    CMD="${SAMTOOLS} mpileup -b ${BAM_LIST} -f ${REFERENCE_SEQ} -q 0 -Q 0 --incl-flags 0 --excl-flags 0 -t DP -d 10000000 -v"
    CMD+=" | ${BCFTOOLS} view"
    CMD+=" | ${BCFTOOLS} filter -e 'REF !~ \"^[ATGC]$\"'"
    CMD+=" | ${BCFTOOLS} norm -m-both -f ${REFERENCE_SEQ}"
    CMD+=" | grep -v '<X>\|INDEL'"
    CMD+=" > ${output_vcf}"
    
    echo "Running command: ${CMD}" >> ${log_file}
    eval "${CMD}" 2>> ${log_file}
    
    end_time=$(date +%s.%N)
    execution_time=$(echo "${end_time} - ${start_time}" | bc)
    
    echo "${bam_file},mpileup,ALL,${execution_time}" >> ${PERF_DIR}/timing.csv
}


# Old
# run_mpileup() {
#     local bam_file=$1
#     local basename=$(basename ${bam_file} .bam)
#     local output_vcf="${PERF_DIR}/mpileup/${basename}.vcf"  # Changed from .vcf.gz to .vcf
#     local log_file="${PERF_DIR}/logs/${basename}_mpileup.log"
    
#     echo "[$(date)] Running mpileup for ${basename}"
    
#     # Time the mpileup process
#     start_time=$(date +%s.%N)
    
#     bcftools mpileup -f ${REFERENCE_SEQ} ${BAM_DIR}/${bam_file} \
#     -q 13 -Q 20 -d 10000000 | \
#     bcftools view | \
#     bcftools filter -e 'REF !~ "^[ATGC]$"' | \
#     bcftools norm -m-both -f ${REFERENCE_SEQ} | \
#     grep -v '<*>' | grep -v INDEL > ${output_vcf} 2> ${log_file}
#     # Removed bgzip -c and changed output redirection
    
#     end_time=$(date +%s.%N)
#     execution_time=$(echo "${end_time} - ${start_time}" | bc)
    
#     echo "${bam_file},mpileup,ALL,${execution_time}" >> ${PERF_DIR}/timing.csv
# }


# Function to process a single chromosome with a caller
process_chromosome() {
    local caller=$1
    local caller_path=$2
    local bam_file=$3
    local chromosome=$4
    local output_dir=$5
    
    local basename=$(basename ${bam_file} .bam)
    local output_file="${output_dir}/${basename}.vcf"
    
    # Ensure header exists in output file
    if [ ! -f ${output_file} ]; then
        cat ${HEADER} > ${output_file}
    fi
    
    # Time the processing
    start_time=$(date +%s.%N)
    
    python ${caller_path} \
        --reference_seq ${REFERENCE_SEQ} \
        --chromosome ${chromosome} \
        --bamfile ${BAM_DIR}/${bam_file} \
        --bedfile ${BEDFILE} \
        --header ${HEADER} \
        --out ${output_file} \
        2> ${PERF_DIR}/logs/${basename}_${caller}_chr${chromosome}.log
        
    end_time=$(date +%s.%N)
    execution_time=$(echo "${end_time} - ${start_time}" | bc)
    echo "${bam_file},${caller},${chromosome},${execution_time}" >> ${PERF_DIR}/timing.csv
}

# Create timing CSV file
echo "bam_file,caller,chromosome,time" > ${PERF_DIR}/timing.csv

# Process each BAM file with all callers
for bam in "${selected_bams[@]}"; do
    echo "[$(date)] Processing ${bam}"
    
    # # Process each chromosome with both callers
    for chromosome in {1..22} X Y; do
        echo "  Processing chromosome ${chromosome}"
        
    #     # Run old caller
        process_chromosome "old_caller" ${OLD_CALLER} ${bam} ${chromosome} ${PERF_DIR}/old_caller
        
        # Run new caller
        process_chromosome "new_caller" ${NEW_CALLER} ${bam} ${chromosome} ${PERF_DIR}/new_caller
    done
        
    # Run mpileup
    run_mpileup ${bam}
done

# Generate comparison results
python ${project_dir}/compare_caller_output.py \
    --vcffolder /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/performance_test \
    --labels "New Caller" "Old Caller" "MPileup" \
    --output_dir ./comparison_results
echo "[$(date)] Comparison results generated"

# Generate summary statistics using Python
cat > ${PERF_DIR}/analyze_performance.py << 'EOL'
import pandas as pd
import numpy as np

# Read timing data
df = pd.read_csv('timing.csv')

# Calculate statistics per caller
stats = df.groupby('caller').agg({
    'time': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

# Calculate per-chromosome statistics (excluding mpileup which processes all chromosomes at once)
chrom_stats = df[df['caller'] != 'mpileup'].groupby(['caller', 'chromosome'])['time'].mean().unstack()

# Save statistics
with open('performance_summary.txt', 'w') as f:
    f.write('Overall Statistics:\n')
    f.write('=================\n')
    f.write(f'{stats.to_string()}\n\n')
    
    f.write('Per-Chromosome Average Times:\n')
    f.write('===========================\n')
    f.write(f'{chrom_stats.to_string()}\n\n')

    # Calculate total time per caller
    total_times = df.groupby('caller')['time'].sum()
    f.write('\nTotal Processing Times:\n')
    f.write('=====================\n')
    f.write(f'{total_times.to_string()}\n')
EOL

# Run analysis
cd ${PERF_DIR}
python analyze_performance.py

# Add summary to main log
cat ${PERF_DIR}/performance_summary.txt >> ${PERF_LOG}

# Run the SNV comparison for each BAM file


echo "[$(date)] Performance testing complete"
echo "Results are in: ${PERF_DIR}"
echo "See ${PERF_LOG} for detailed performance comparison"

# Print quick summary to console
# echo "=================================" bvn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
echo "Quick Summary:"
echo "================================="
tail -n 20 ${PERF_LOG}