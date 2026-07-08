#!/bin/bash
# Submit OVAR_P5 pipeline steps 2->7 as an afterok dependency chain.
# Each step starts only if the previous one COMPLETED successfully.
# Step 1 (mpileup, job 12368216) is already done.
#
# Usage:  bash run_slurm/ovar_p5/submit_chain_2_7.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

mkdir -p slurm_output/OVAR_P5/baseQ0mapQ0

D=run_slurm/ovar_p5

j2=$(sbatch --parsable                       $D/2_beagle.sh)
echo "step2 beagle              -> $j2"
j3=$(sbatch --parsable --dependency=afterok:$j2 $D/3_genotype_shifting.sh)
echo "step3 genotype_shifting   -> $j3  (afterok:$j2)"
j4=$(sbatch --parsable --dependency=afterok:$j3 $D/4_sequence_error_model.sh)
echo "step4 seq_error_model     -> $j4  (afterok:$j3)"
j5=$(sbatch --parsable --dependency=afterok:$j4 $D/5_neural_network.sh)
echo "step5 nn_classifier       -> $j5  (afterok:$j4)"
j6=$(sbatch --parsable --dependency=afterok:$j5 $D/6_single_bam_snp_filter.sh)
echo "step6 single_bam_filter   -> $j6  (afterok:$j5)"
j7=$(sbatch --parsable --dependency=afterok:$j6 $D/7_spatial_filter_n_matrix.sh)
echo "step7 spatial_filter      -> $j7  (afterok:$j6)"

echo
echo "Chain submitted. Watch with:"
echo "  squeue -u \$USER -j $j2,$j3,$j4,$j5,$j6,$j7"
echo "If any step fails, its dependents stay PENDING with reason (DependencyNeverSatisfied) and can be cancelled with:"
echo "  scancel $j3 $j4 $j5 $j6 $j7"
