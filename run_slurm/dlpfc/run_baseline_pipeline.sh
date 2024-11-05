spatialid="151507"
project_dir="/data/maiziezhou_lab/yuqi/snv_calling"

# Define paths
BAM_DIR="/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/${spatialid}/bam_bycell"
REFERENCE_SEQ="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BEDFILE="/data/maiziezhou_lab/yuqi/snv_calling/data/reference/GRCh38.bed"
HEADER="/data/maiziezhou_lab/yuqi/snv_calling/data/reference/header.txt"
SCRIPT_PATH="/data/maiziezhou_lab/yuqi/snv_calling/scripts/calling/self_caller.py"
BENCHMARK_SCRIPT="${project_dir}/scripts/calling/test_caller_pipeline.py"
OUTPUT_DIR="${project_dir}/data/dlpfc/${spatialid}/benchmark_results"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
python ${BENCHMARK_SCRIPT} \
    --bam_dir ${BAM_DIR} \
    --sample_size 5 \
    --script_path ${SCRIPT_PATH} \
    --reference_seq ${REFERENCE_SEQ} \
    --bedfile ${BEDFILE} \
    --header ${HEADER} \
    --output_dir ${OUTPUT_DIR}