import os
import subprocess
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import tempfile

# Configuration for datasets
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "bam_pattern": "{section_id}/bam_bycell/*.bam",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True
    },
    "DLPFC_SVM_FILTERED": {
        "base_path": "/data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc",
        "bam_pattern": "{section_id}/output_VCFs/BAM_filtered/baseQ13mapQ20/*.bam",
        "output_dir": "data/dlpfc_svm_filtered/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True
    }
}

# File Paths from mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
}

def setup_environment():
    """Setup environment variables for library paths."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return os.environ.copy()

# Utility function to count reads in a BAM file using os.system
def count_reads(bam_file):
    try:
        env = setup_environment()
        
        # Create a temporary file to store the count
        with tempfile.NamedTemporaryFile(mode='w+') as tmp_file:
            # Use os.system to execute samtools and redirect output to temp file
            cmd = f"{PATH_CONFIG['SAMTOOLS']} view -c {bam_file} > {tmp_file.name}"
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                raise Exception(f"Command failed with exit code {exit_code}")
            
            # Read the count from temp file
            tmp_file.seek(0)
            read_count = int(tmp_file.read().strip())
            
            return {
                'bam_file': os.path.basename(bam_file),
                'read_count': read_count,
                'status': 'success'
            }
    except Exception as e:
        return {
            'bam_file': os.path.basename(bam_file),
            'read_count': 0,
            'status': 'failed',
            'error': str(e)
        }

# Process BAM files in parallel
def process_bam_files(bam_files, max_workers=30):
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_bam = {
            executor.submit(count_reads, bam): bam 
            for bam in bam_files
        }
        
        with tqdm(total=len(bam_files), desc="Processing BAM files") as pbar:
            for future in as_completed(future_to_bam):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    return pd.DataFrame(results)

# Generate statistics and plots
def generate_statistics(df, dataset_name, section_id, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out failed BAMs
    success_df = df[df['status'] == 'success']
    
    # Basic statistics
    stats = {
        'total_bams': len(df),
        'successful_bams': len(success_df),
        'failed_bams': len(df) - len(success_df),
        'total_reads': success_df['read_count'].sum(),
        'avg_reads_per_bam': success_df['read_count'].mean(),
        'median_reads_per_bam': success_df['read_count'].median(),
        'min_reads': success_df['read_count'].min(),
        'max_reads': success_df['read_count'].max(),
        'std_dev_reads': success_df['read_count'].std()
    }
    
    # Save statistics to file
    stats_df = pd.DataFrame([stats])
    stats_file = os.path.join(output_dir, f"{dataset_name}_{section_id}_stats.csv")
    stats_df.to_csv(stats_file, index=False)
    
    # Save read counts to file
    counts_file = os.path.join(output_dir, f"{dataset_name}_{section_id}_read_counts.csv")
    success_df.to_csv(counts_file, index=False)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Histogram with log scale
    plt.subplot(2, 2, 1)
    bins = np.logspace(0, np.log10(success_df['read_count'].max() + 1), 50)
    plt.hist(success_df['read_count'], bins=bins, alpha=0.7)
    plt.xscale('log')
    plt.title(f'{dataset_name} - {section_id}: Read Count Distribution (Log Scale)')
    plt.xlabel('Read Count (log scale)')
    plt.ylabel('Number of BAM Files')
    plt.grid(True, alpha=0.3)
    
    # Linear scale histogram with more detailed binning
    plt.subplot(2, 2, 2)
    if success_df['read_count'].max() > 1000:
        bins = np.linspace(0, min(10000, success_df['read_count'].max()), 50)
    else:
        bins = np.linspace(0, success_df['read_count'].max(), 50)
    plt.hist(success_df['read_count'], bins=bins, alpha=0.7, color='green')
    plt.title(f'{dataset_name} - {section_id}: Read Count Distribution (Linear Scale)')
    plt.xlabel('Read Count')
    plt.ylabel('Number of BAM Files')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 3)
    sns.boxplot(x=success_df['read_count'])
    plt.title(f'{dataset_name} - {section_id}: Read Count Box Plot')
    plt.xlabel('Read Count')
    plt.grid(True, alpha=0.3)
    
    # CDF plot
    plt.subplot(2, 2, 4)
    sorted_data = np.sort(success_df['read_count'])
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, yvals, color='purple')
    plt.title(f'{dataset_name} - {section_id}: Cumulative Distribution')
    plt.xlabel('Read Count')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{dataset_name}_{section_id}_read_distribution.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    return stats, success_df

def compare_datasets(results, section_id, output_dir):
    """Generate comparison plots between datasets"""
    plt.figure(figsize=(18, 14))
    
    # Prepare data
    datasets = list(results.keys())
    dfs = [results[dataset]['df'] for dataset in datasets]
    stats = [results[dataset]['stats'] for dataset in datasets]
    
    # Box plot comparison
    plt.subplot(2, 2, 1)
    data_to_plot = [df['read_count'] for df in dfs]
    sns.boxplot(data=data_to_plot)
    plt.xticks(range(len(datasets)), datasets)
    plt.title(f'Read Count Comparison (Box Plot) - Section {section_id}')
    plt.ylabel('Read Count')
    plt.grid(True, alpha=0.3)
    
    # Violin plot comparison
    plt.subplot(2, 2, 2)
    sns.violinplot(data=data_to_plot)
    plt.xticks(range(len(datasets)), datasets)
    plt.title(f'Read Count Comparison (Violin Plot) - Section {section_id}')
    plt.ylabel('Read Count')
    plt.grid(True, alpha=0.3)
    
    # Histogram comparison (log scale)
    plt.subplot(2, 2, 3)
    max_count = max([df['read_count'].max() for df in dfs])
    bins = np.logspace(0, np.log10(max_count + 1), 50)
    
    for i, dataset in enumerate(datasets):
        plt.hist(
            dfs[i]['read_count'], 
            bins=bins, 
            alpha=0.5, 
            label=dataset
        )
    
    plt.xscale('log')
    plt.title(f'Read Count Distribution Comparison - Section {section_id}')
    plt.xlabel('Read Count (log scale)')
    plt.ylabel('Number of BAM Files')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CDF comparison
    plt.subplot(2, 2, 4)
    for i, dataset in enumerate(datasets):
        sorted_data = np.sort(dfs[i]['read_count'])
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, yvals, label=dataset)
    
    plt.title(f'Cumulative Distribution Comparison - Section {section_id}')
    plt.xlabel('Read Count')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"comparison_{section_id}_read_distribution.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    # Create additional comparison plot for bin-based histogram
    plt.figure(figsize=(12, 8))
    
    # Define common bins for both datasets
    min_count = min([df['read_count'].min() for df in dfs])
    max_count = max([df['read_count'].max() for df in dfs])
    
    # Create bins for the comparison
    if max_count > 5000:
        bins = [0, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, max_count + 1]
    else:
        bins = np.linspace(0, max_count + 1, 15)
        
    # Count BAMs in each bin for each dataset
    bin_data = {}
    for i, dataset in enumerate(datasets):
        hist_data, _ = np.histogram(dfs[i]['read_count'], bins=bins)
        bin_data[dataset] = hist_data
    
    # Convert to DataFrame for easier plotting
    bin_df = pd.DataFrame(bin_data)
    bin_df['bin_labels'] = [f"{bins[i]}-{bins[i+1]}" if i < len(bins)-2 else f"{bins[i]}+" 
                           for i in range(len(bins)-1)]
    
    # Create bar plot
    bin_df.plot(
        x='bin_labels', 
        y=datasets, 
        kind='bar', 
        figsize=(14, 8),
        width=0.8
    )
    
    plt.title(f'BAM File Counts by Read Count Range - Section {section_id}')
    plt.xlabel('Read Count Range')
    plt.ylabel('Number of BAM Files')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    bin_plot_file = os.path.join(output_dir, f"comparison_{section_id}_binned_counts.png")
    plt.savefig(bin_plot_file, dpi=300)
    plt.close()
    
    # Summary comparison table
    comparison_data = {}
    for metric in stats[0].keys():
        comparison_data[metric] = [s[metric] for s in stats]
    
    comparison_df = pd.DataFrame(comparison_data, index=datasets)
    comparison_file = os.path.join(output_dir, f"comparison_{section_id}_stats.csv")
    comparison_df.to_csv(comparison_file)
    
    # Print summary comparison
    print("\nDataset Comparison:")
    for metric in ['total_bams', 'successful_bams', 'total_reads', 'avg_reads_per_bam', 
                  'median_reads_per_bam', 'std_dev_reads']:
        values = [s[metric] for s in stats]
        if metric in ['total_reads']:
            values = [f"{v:,}" for v in values]
        elif metric in ['avg_reads_per_bam', 'median_reads_per_bam', 'std_dev_reads']:
            values = [f"{v:.2f}" for v in values]
        
        print(f"{metric}: {' vs '.join(str(v) for v in values)}")
    
    # Reduction statistics if we have DLPFC_SVM_FILTERED
    if 'DLPFC' in results and 'DLPFC_SVM_FILTERED' in results:
        orig_bams = stats[0]['successful_bams']
        filtered_bams = stats[1]['successful_bams']
        reduction_pct = ((orig_bams - filtered_bams) / orig_bams) * 100
        
        orig_reads = stats[0]['total_reads']
        filtered_reads = stats[1]['total_reads']
        read_reduction_pct = ((orig_reads - filtered_reads) / orig_reads) * 100
        
        print(f"\nReduction Statistics:")
        print(f"BAM file reduction: {orig_bams - filtered_bams:,} files ({reduction_pct:.2f}%)")
        print(f"Read count reduction: {orig_reads - filtered_reads:,} reads ({read_reduction_pct:.2f}%)")
        
        # Calculate average reads per remaining BAM after filtering
        if filtered_bams > 0:
            avg_reads_before = orig_reads / orig_bams
            avg_reads_after = filtered_reads / filtered_bams
            avg_change_pct = ((avg_reads_after - avg_reads_before) / avg_reads_before) * 100
            print(f"Average reads per BAM changed: {avg_reads_before:.2f} → {avg_reads_after:.2f} ({avg_change_pct:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze BAM file read counts")
    parser.add_argument("--section_id", default="151507", help="Section ID to analyze")
    parser.add_argument("--quality_filter", default="baseQ13mapQ20", help="Quality filter to use")
    parser.add_argument("--output_dir", default="./bam_analysis", help="Directory for output files")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to use")
    args = parser.parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup environment
    env = setup_environment()
    print("Environment setup complete.")
    
    # Process both datasets
    all_results = {}
    
    for dataset_name in ["DLPFC", "DLPFC_SVM_FILTERED"]:
        print(f"\nProcessing {dataset_name} dataset...")
        config = DATASET_CONFIGS[dataset_name]
        
        # Get BAM files
        if config["has_sections"]:
            bam_pattern = os.path.join(
                config["base_path"],
                config["bam_pattern"].format(section_id=args.section_id)
            )
        else:
            bam_pattern = os.path.join(config["base_path"], config["bam_pattern"])
            
        bam_files = glob.glob(bam_pattern)
        if not bam_files:
            print(f"No BAM files found at: {bam_pattern}")
            continue
            
        print(f"Found {len(bam_files)} BAM files")
        
        # Process BAM files
        results_df = process_bam_files(bam_files, max_workers=args.threads)
        
        # Generate statistics and plots
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        stats, filtered_df = generate_statistics(
            results_df, dataset_name, args.section_id, dataset_output_dir
        )
        
        all_results[dataset_name] = {
            'stats': stats,
            'df': filtered_df
        }
        
        # Print summary
        print(f"\n{dataset_name} Summary:")
        print(f"Total BAM files: {stats['total_bams']}")
        print(f"Successfully processed: {stats['successful_bams']}")
        print(f"Failed: {stats['failed_bams']}")
        print(f"Total reads: {stats['total_reads']:,}")
        print(f"Average reads per BAM: {stats['avg_reads_per_bam']:.2f}")
        print(f"Median reads per BAM: {stats['median_reads_per_bam']:.2f}")
        print(f"Standard deviation: {stats['std_dev_reads']:.2f}")
        print(f"Min reads: {stats['min_reads']}")
        print(f"Max reads: {stats['max_reads']}")
        
    # Compare datasets if both were processed
    if len(all_results) == 2:
        print("\nGenerating comparison plots...")
        compare_datasets(all_results, args.section_id, args.output_dir)

if __name__ == "__main__":
    main()
    
# Usage
# python scripts/tools/bam_file_analysis.py --section_id 151507 --quality_filter baseQ13mapQ20 --output_dir ./bam_analysis