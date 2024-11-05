#!/bin/bash
#################
#SBATCH --job-name=caller_testing
#################
#SBATCH --output=testing.out
#################
#SBATCH --error=testing.err
#################
#SBATCH --time=7-00:00:00
#SBATCH --account=cgw_maizie
#SBATCH --partition=cgw-maizie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#################
#SBATCH --mem=200GB
#################
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=han.liu.1@vanderbilt.edu

# Load necessary modules
#module load Anaconda3/2022.05
#module load GCC/11.3.0
#module load SAMtools/1.18

echo "SLURM_JOBID: $SLURM_JOBID"

## DEFINE PARAMETERS
# DLPFC regular parameters
spatialid="151669"
rad_lim="150"
neighbor_lim="0"
num_cluster="5"
snv_tool="self"

# Filtering parameters
filter="filter_by_neighbor"
threshold='1'

project_dir="/data/maiziezhou_lab/hanliu/projects/snv_calling"

##--------------------Generate config files---------------------##
# original_ini_file="${project_dir}/configs/dlpfc/sample_config.ini"
# config_output_dir="${project_dir}/configs/dlpfc/${spatialid}/"

# # Run the modified script with the defined parameters
# "${project_dir}/scripts/update_ini.sh" \
#   -i "$original_ini_file" \
#   -o "$config_output_dir" \
#   -s "$spatialid" \
#   -r "$rad_lim" \
#   -n "$neighbor_lim" \
#   -c "$num_cluster" \
#   -t "$snv_tool" \
#   -f "$filter" \
#   -d "$threshold"

##--------------------Calling SNVs------------------------------##
config_path="${project_dir}/configs/dlpfc/${spatialid}/${spatialid}_${snv_tool}_${neighbor_lim}_${filter}.ini"

# # Check which SNV tool to use
# if [ "$snv_tool" = "gatk" ]; then
#     echo "Running GATK SNV Caller..."
#     python "${project_dir}/scripts/calling/gatk_caller.py" -c "${config_path}"

# elif [ "$snv_tool" = "self" ]; then
#     echo "Running Self SNV Caller..."
#     BAM_DIR="/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/${spatialid}/bam_bycell"
#     REFERENCE_SEQ="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
#     BEDFILE="/data/maiziezhou_lab/hanliu/projects/snv_calling/data/reference/GRCh38.bed"
#     HEADER="/data/maiziezhou_lab/hanliu/projects/snv_calling/data/reference/header.txt"
#     VCF_DIR="/data/maiziezhou_lab/hanliu/projects/snv_calling/data/dlpfc/${spatialid}/output_VCFs/${spatialid}_${snv_tool}_${neighbor_lim}_unfiltered"
#     SCRIPT_PATH="/data/maiziezhou_lab/hanliu/projects/snv_calling/scripts/calling/self_caller.py"

#     mkdir -p ${VCF_DIR}

#     process_bam() {
#         local BAMFILE="$1"
#         local BASENAME=$(basename "$BAMFILE" .bam)
#         local OUTPUT_PATH="$VCF_DIR/${BASENAME}.vcf"

#         # Write the header to the output VCF
#         cat "$HEADER" > "$OUTPUT_PATH"

#         # Loop through chromosomes 1-22, X, Y
#         for CHROMOSOME in {1..22} X Y; do
#             echo "Processing $BAMFILE chromosome: chr$CHROMOSOME"
#             python "${SCRIPT_PATH}" --reference_seq "$REFERENCE_SEQ" \
#                                                 --chromosome "$CHROMOSOME" \
#                                                 --bamfile "$BAMFILE" \
#                                                 --bedfile "$BEDFILE" \
#                                                 --header "$HEADER" \
#                                                 --out "$OUTPUT_PATH"
#             echo "Completed chromosome: chr$CHROMOSOME for $BAMFILE"
#         done
#     }

#     export -f process_bam

#     # Get a list of all BAM files
#     BAMFILES=("$BAM_DIR"/*.bam)
    
#     # Define the batch size
#     BATCH_SIZE=30

#     # Loop through the BAM files in batches
#     # for ((i=0; i<${#BAMFILES[@]}; i+=BATCH_SIZE)); do
#     for ((i=0; i<30; i+=BATCH_SIZE)); do
#         for ((j=i; j<i+BATCH_SIZE && j<${#BAMFILES[@]}; j++)); do
#             process_bam "${BAMFILES[j]}" &
#         done
#         wait
#     done

# else
#     echo "Error: Unknown SNV_TOOL specified. Please use 'gatk' or 'self'."
#     exit 1
# fi

##-------------------Filtering-----------------------------##
# script_dir="${project_dir}/scripts/filtering/"

# # Step 1: FILTER_by_1000Genome

# # Step 2: Filter by other metrics
# if [ "${filter}" = "filter_by_neighbor" ]; then
#     echo "Filtering by neighbor"
    
#     neighbor_data_dir="${project_dir}/data/dlpfc/${spatialid}/"
#     input_dir="/data/maiziezhou_lab/hanliu/projects/snv_calling/data/dlpfc/${spatialid}/output_VCFs/${spatialid}_${snv_tool}_${neighbor_lim}_unfiltered"
#     output_dir="/data/maiziezhou_lab/hanliu/projects/snv_calling/data/dlpfc/${spatialid}/output_VCFs/${spatialid}_${snv_tool}_${neighbor_lim}_${filter}"
#     mkdir -p "${output_dir}"
#     mkdir -p "${output_dir}/tmp/sum"
#     mkdir -p "${output_dir}/tmp/matrix"
#     python "${script_dir}/filter_by_neighbor.py" "${input_dir}" "${output_dir}" "${neighbor_data_dir}" "${threshold}"

# elif [ "${filter}" = "filter_by_BAF" ]; then
#     echo "Filtering by BAF"
#     # Additional BAF filtering logic...
# else
#     echo "Error: Unknown filter specified. Please use 'filter_by_neighbor' or 'filter_by_BAF'"
#     exit 1
# fi
# echo "Filtering is Done!"


##-------------------Matrix Generation-----------------------------##
mkdir -p "${project_dir}/data/dlpfc/${spatialid}/processed_data/processed_vcf_tables/${spatialid}_${snv_tool}_${neighbor_lim}_${filter}"
# mkdir -p "${project_dir}/data/dlpfc/${spatialid}/processed_data/processed_vcf_tables/${spatialid}_${snv_tool}_${neighbor_lim}_${filter}"
python "${project_dir}/scripts/postprocess/generate_matrix.py" "${config_path}"
# echo "Matrix Generation is Done"
