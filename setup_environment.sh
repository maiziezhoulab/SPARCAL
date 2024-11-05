#!/bin/bash

# Define paths
ENV_NAME="snv_caller"
ENV_FILE="environment.yml"

# Create environment file
cat > ${ENV_FILE} << 'EOL'
name: snv_caller
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - pysam
  - samtools>=1.18
  - pip
  - ipython
  - jupyter
  - tqdm
  - matplotlib
  - seaborn
  - biopython
  - pyvcf
  - bedtools
  pip:
    - memory_profiler
EOL

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    read -p "Environment ${ENV_NAME} already exists. Do you want to remove and recreate it? (y/n) " choice
    if [ "$choice" = "y" ]; then
        conda env remove -n ${ENV_NAME}
    else
        echo "Exiting without making changes."
        exit 0
    fi
fi

# Create new environment
echo "Creating new conda environment: ${ENV_NAME}"
conda env create -f ${ENV_FILE}

# Activate environment and verify installation
echo "Activating environment and verifying installation..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify key packages
echo "Verifying installations..."
python -c "import numpy; print('numpy version:', numpy.__version__)"
python -c "import pandas; print('pandas version:', pandas.__version__)"
python -c "import pysam; print('pysam version:', pysam.__version__)"

echo "Environment setup complete. To activate, use: conda activate ${ENV_NAME}"