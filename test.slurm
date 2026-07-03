#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --time=1:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

##SBATCH --account=maiziezhou_lab_phd
##SBATCH --partition=batch

#SBATCH --account=maiziezhou_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_titan_x:1

# Load any required modules (if needed)
# module load python/3.x
module load miniconda3
source activate calicost_env
alias python='/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python'
which python

# Run your Python script
python test.py