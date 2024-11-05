#!/bin/bash

# Common path for scripts
TOOL="self"
SCRIPT_PATH="/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/scripts"
CONFIG_PATH="/data/maiziezhou_lab/hanliu/projects/snv_call/configs/DLPFC_spatialLIBD/${TOOL}"
# Define the list of IDs
ids=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
# ids=(151507)
# Function to run all steps for a given ID
run_steps_for_id() {
    id=$1
    echo "Processing ID: $id"

    # # Step 1
    # echo "Step 1: Running vcf2tables.py for ID $id"
    # python "$SCRIPT_PATH/vcf2tables.py" $CONFIG_PATH/$id

    # # Step 2
    # echo "Step 2: Running process_vcf_tables.py for ID $id"
    # python "$SCRIPT_PATH/process_vcf_tables.py" $CONFIG_PATH/$id

    # # Step 3
    # echo "Step 3: Running get_union_set.py for ID $id"
    # python "$SCRIPT_PATH/get_union_set.py" $CONFIG_PATH/$id

    # # Step 4
    # echo "Step 4: Running generate_matrix.py for ID $id"
    # python "$SCRIPT_PATH/generate_matrix.py" $CONFIG_PATH/$id

    # # Step 5
    # echo "Step 5: Running fill_zeros_matrix.py for ID $id"
    # python "$SCRIPT_PATH/fill_zeros_matrix.py" $CONFIG_PATH/$id

    # step 6
    echo "Step 6: Running clustering_pkl_matrix.py for ID $id"
    python "$SCRIPT_PATH/clustering_pkl_matrix.py" $CONFIG_PATH/$id

    # # # step 7
    # echo "Step 7: Running visual_clusters.py for ID $id"
    # python "$SCRIPT_PATH/visual_clusters.py" $CONFIG_PATH/$id

    # # Step 8
    # echo "Step 8: Running sum_snps.py for ID $id"
    # python "$SCRIPT_PATH/sum_snps.py" $CONFIG_PATH/$id

    # # Step 9
    # echo "Step 9: Running analysis.py for ID $id"
    # python "$SCRIPT_PATH/analysis.py" $CONFIG_PATH/$id

    # # Step 10
    # echo "Step 10: Running cal_ari.py for ID $id"
    # python "$SCRIPT_PATH/cal_ari.py" $CONFIG_PATH/$id

    echo "Completed processing for ID: $id"
}

# Run steps for each ID in parallel
for id in "${ids[@]}"
do
    run_steps_for_id $id &
done

# Wait for all background processes to finish
wait
echo "All processes completed"
