# To run 
```
module load Anaconda3
module load GCC
module load SAMtools
```
# Scripts:
```
# Caller file:
/data/maiziezhou_lab/yuqi/snv_calling/scripts/calling/new_caller.py

# To call the caller
/data/maiziezhou_lab/yuqi/snv_calling/run_slurm/dlpfc/run_new_old_caller_n_mpileup.sh

# To show the performance and compare
python compare_caller_output.py \
    --vcffolder /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/performance_test \
    --labels "New Caller" "Old Caller" "MPileup" \
    --output_dir ./comparison_results

# The result data is in:
/data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/performance_test
```
