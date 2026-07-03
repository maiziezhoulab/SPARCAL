import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict
import gzip
from pathlib import Path
from tqdm import tqdm

# Path configurations as in mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
}

def setup_environment():
    """Setup environment variables for tools."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

class CoverageAnalyzer:
    def __init__(self, matrix_file: str, barcode_file: str):
        """
        Initialize the coverage analyzer.
        
        Args:
            matrix_file: Path to the matrix file containing positions
            barcode_file: Path to the tissue barcodes file
        """
        # Setup environment
        self.env = setup_environment()
        self.matrix_file = matrix_file
        self.barcode_file = barcode_file
        self.tissue_barcodes = self._load_tissue_barcodes()
        self.positions = self._parse_matrix_file()

    def _load_tissue_barcodes(self) -> Set[str]:
        """Load tissue barcodes from file."""
        barcodes = set()
        opener = gzip.open if self.barcode_file.endswith('.gz') else open
        mode = 'rt' if self.barcode_file.endswith('.gz') else 'r'
        
        with opener(self.barcode_file, mode) as f:
            for line in f:
                barcode = line.strip().split('\t')[0]
                barcodes.add(barcode)
                
        print(f"Loaded {len(barcodes)} tissue barcodes")
        return barcodes

    def _parse_matrix_file(self) -> List[tuple]:
        """Parse positions from matrix file."""
        positions = []
        with open(self.matrix_file, 'r') as f:
            # Skip header lines
            for line in f:
                if line.startswith('%'):
                    continue
                fields = line.strip().split()
                if len(fields) >= 3:
                    # Extract spot coordinate and position
                    row = int(fields[0])
                    col = int(fields[1])
                    pos = int(fields[2])
                    positions.append((row, col, pos))
        
        print(f"Loaded {len(positions)} positions from matrix file")
        return positions

    def calculate_coverage(self, bam_dir: str) -> Dict:
        """
        Calculate average non-zero coverage for each position.
        
        Args:
            bam_dir: Directory containing cell BAM files
            
        Returns:
            Dictionary with coverage statistics per position
        """
        coverage_stats = defaultdict(list)
        
        # Process each tissue barcode (limit to 10 for testing)
        for barcode in tqdm(list(self.tissue_barcodes)[:10], desc="Calculating coverage"):
            bam_file = os.path.join(bam_dir, f"{barcode}.bam")
            if not os.path.exists(bam_file):
                continue
                
            # Use samtools depth to get coverage
            cmd = [PATH_CONFIG['SAMTOOLS'], 'depth', bam_file]
            try:
                result = subprocess.run(cmd, 
                                     capture_output=True, 
                                     text=True, 
                                     check=True,
                                     env=self.env)
                
                # Process coverage output
                for line in result.stdout.splitlines():
                    chrom, pos, depth = line.split('\t')
                    pos = int(pos)
                    depth = int(depth)
                    
                    # Only store non-zero depths
                    if depth > 0:
                        coverage_stats[pos].append(depth)
                        
            except subprocess.CalledProcessError as e:
                print(f"Error processing {bam_file}: {str(e)}")
                continue

        # Calculate statistics
        results = {}
        for pos, depths in coverage_stats.items():
            if depths:  # Only calculate for positions with non-zero depths
                results[pos] = {
                    'mean_coverage': np.mean(depths),
                    'median_coverage': np.median(depths),
                    'std_coverage': np.std(depths),
                    'num_cells': len(depths),
                    'total_cells': len(self.tissue_barcodes)
                }
        
        return results

    def save_results(self, results: Dict, output_file: str):
        """Save coverage analysis results to file."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'position'
        df.to_csv(output_file)
        print(f"Results saved to {output_file}")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Calculate coverage statistics for tissue spots')
    parser.add_argument('--matrix', required=True, help='Path to matrix file')
    parser.add_argument('--barcodes', required=True, help='Path to tissue barcodes file')
    parser.add_argument('--bam-dir', required=True, help='Directory containing cell BAM files')
    parser.add_argument('--output', required=True, help='Output file path')
    args = parser.parse_args()

    # Run analysis
    analyzer = CoverageAnalyzer(args.matrix, args.barcodes)
    results = analyzer.calculate_coverage(args.bam_dir)
    analyzer.save_results(results, args.output)

if __name__ == '__main__':
    main()

# Usage for P4_tumor gene expression:
# --matrix: /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/Meta_Data/GSM4565823_matrix.mtx
# --barcodes: /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/Meta_Data/GSM4565823_barcodes.tsv
# --bam-dir: /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/P4_Tumor_output/outs/split_BAM/",
# --output: slurm_output/P4_tumor_coverage.csv
# python scripts/tools/get_barcode_coverage.py --matrix /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/Meta_Data/GSM4565823_matrix.mtx --barcodes /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/Meta_Data/GSM4565823_barcodes.tsv --bam-dir /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1/P4_Tumor_output/outs/split_BAM/ --output slurm_output/P4_tumor_coverage.csv