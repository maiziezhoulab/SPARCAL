#!/usr/bin/env python3
"""
VCF Feature Distribution Comparison
Explores and visualizes feature distributions across up to 3 VCF files
Designed to compare somatic vs germline variants from SPARCAL pipeline outputs
"""

import sys
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette for up to 3 VCFs
COLORS = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green


def parse_vcf(vcf_file):
    """
    Parse VCF file and extract relevant features
    
    Returns:
        dict: Dictionary containing lists of feature values
    """
    features = {
        'DP': [],           # Depth
        'QUAL': [],         # Quality score
        'GQ': [],           # Genotype quality
        'BAF': [],          # B-allele frequency
        'QS': [],           # Quality sum (ratio for ALT)
        'SGB': [],          # Segregation based metric
        'MQ0F': [],         # Fraction of MQ0 reads
        'VDB': [],          # Variant distance bias
        'RPB': [],          # Read position bias
        'MQB': [],          # Mapping quality bias
        'BQB': [],          # Base quality bias
        'MQSB': [],         # Mapping quality squared bias
        'GT': [],           # Genotype
        'FILTER': [],       # Filter status
        'I16_stats': defaultdict(list),  # I16 individual statistics
    }
    
    open_func = gzip.open if vcf_file.endswith('.gz') else open
    
    with open_func(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue
            
            chrom, pos, id_, ref, alt, qual, filt, info, format_, sample = fields[:10]
            
            # Parse INFO field
            info_dict = {}
            for item in info.split(';'):
                if '=' in item:
                    key, value = item.split('=', 1)
                    info_dict[key] = value
                else:
                    info_dict[item] = True
            
            # Parse FORMAT field
            format_fields = format_.split(':')
            sample_values = sample.split(':')
            sample_dict = dict(zip(format_fields, sample_values))
            
            # Extract features
            # Depth
            if 'DP' in info_dict:
                try:
                    features['DP'].append(int(info_dict['DP']))
                except ValueError:
                    pass
            
            # Quality
            try:
                features['QUAL'].append(float(qual))
            except ValueError:
                pass
            
            # Genotype Quality
            if 'GQ' in sample_dict:
                try:
                    features['GQ'].append(int(sample_dict['GQ']))
                except ValueError:
                    pass
            
            # B-allele frequency
            if 'BAF' in sample_dict:
                try:
                    features['BAF'].append(float(sample_dict['BAF']))
                except ValueError:
                    pass
            
            # Quality sum ratio (ALT allele quality)
            if 'QS' in info_dict:
                try:
                    qs_values = [float(x) for x in info_dict['QS'].split(',')]
                    if len(qs_values) >= 2:
                        features['QS'].append(qs_values[1])  # ALT quality ratio
                except (ValueError, IndexError):
                    pass
            
            # SGB
            if 'SGB' in info_dict:
                try:
                    features['SGB'].append(float(info_dict['SGB']))
                except ValueError:
                    pass
            
            # MQ0F
            if 'MQ0F' in info_dict:
                try:
                    features['MQ0F'].append(float(info_dict['MQ0F']))
                except ValueError:
                    pass
            
            # VDB
            if 'VDB' in info_dict:
                try:
                    features['VDB'].append(float(info_dict['VDB']))
                except ValueError:
                    pass
            
            # Bias metrics
            for bias_key in ['RPB', 'MQB', 'BQB', 'MQSB']:
                if bias_key in info_dict:
                    try:
                        features[bias_key].append(float(info_dict[bias_key]))
                    except ValueError:
                        pass
            
            # Genotype
            if 'GT' in sample_dict:
                features['GT'].append(sample_dict['GT'])
            
            # Filter
            features['FILTER'].append(filt)
            
            # Parse I16 statistics
            if 'I16' in info_dict:
                try:
                    i16_values = [int(x) for x in info_dict['I16'].split(',')]
                    if len(i16_values) == 16:
                        # I16 format: ref_fwd, ref_rev, alt_fwd, alt_rev, ref_qual, alt_qual, ...
                        features['I16_stats']['ref_fwd'].append(i16_values[0])
                        features['I16_stats']['ref_rev'].append(i16_values[1])
                        features['I16_stats']['alt_fwd'].append(i16_values[2])
                        features['I16_stats']['alt_rev'].append(i16_values[3])
                        features['I16_stats']['ref_baseQ'].append(i16_values[4])
                        features['I16_stats']['alt_baseQ'].append(i16_values[5])
                        
                        # Calculate strand bias (ratio of forward reads)
                        total_ref = i16_values[0] + i16_values[1]
                        total_alt = i16_values[2] + i16_values[3]
                        if total_ref > 0:
                            features['I16_stats']['ref_strand_ratio'].append(i16_values[0] / total_ref)
                        if total_alt > 0:
                            features['I16_stats']['alt_strand_ratio'].append(i16_values[2] / total_alt)
                except (ValueError, IndexError):
                    pass
    
    return features


def plot_distribution_comparison(data_dict, feature_name, output_prefix, 
                                  bins=None, log_scale=True, xlim=None):
    """
    Plot distribution comparison for a single feature across VCF files
    
    Args:
        data_dict: Dictionary {vcf_name: [values]}
        feature_name: Name of the feature
        output_prefix: Output file prefix
        bins: Number of bins or bin edges
        log_scale: Whether to use log scale for y-axis (default True for large count differences)
        xlim: x-axis limits
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Main distribution plot
    ax1 = axes[0]
    all_data = []
    labels = []
    
    for idx, (vcf_name, values) in enumerate(data_dict.items()):
        if len(values) == 0:
            continue
        
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            continue
        
        all_data.append(values)
        labels.append(vcf_name)
        
        # Histogram
        if bins is None:
            # Auto-determine bins based on data range
            if feature_name == 'DP':
                bins = np.arange(0, min(np.max(values), 200) + 10, 10)
            else:
                bins = 50
        
        ax1.hist(values, bins=bins, alpha=0.6, color=COLORS[idx], 
                label=f'{vcf_name} (n={len(values)})', density=False)
    
    ax1.set_ylabel('Count')
    ax1.set_title(f'{feature_name} Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if log_scale:
        ax1.set_yscale('log')
    
    if xlim:
        ax1.set_xlim(xlim)
    
    # Box plot
    ax2 = axes[1]
    positions = range(1, len(all_data) + 1)
    bp = ax2.boxplot(all_data, positions=positions, labels=labels,
                      patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], COLORS[:len(all_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{feature_name}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical comparison
    if len(all_data) >= 2:
        print(f"\n{feature_name} Statistics:")
        for idx, (label, data) in enumerate(zip(labels, all_data)):
            print(f"  {label}:")
            print(f"    Mean: {np.mean(data):.4f}")
            print(f"    Median: {np.median(data):.4f}")
            print(f"    Std: {np.std(data):.4f}")
            print(f"    Min: {np.min(data):.4f}")
            print(f"    Max: {np.max(data):.4f}")
        
        # Statistical tests
        if len(all_data) == 2:
            # Mann-Whitney U test (non-parametric)
            statistic, pvalue = stats.mannwhitneyu(all_data[0], all_data[1], alternative='two-sided')
            print(f"  Mann-Whitney U test: U={statistic:.2f}, p={pvalue:.4e}")
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(all_data[0], all_data[1])
            print(f"  Kolmogorov-Smirnov test: D={ks_stat:.4f}, p={ks_pvalue:.4e}")


def plot_categorical_comparison(data_dict, feature_name, output_prefix):
    """
    Plot comparison for categorical features (GT, FILTER)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    ax1 = axes[0]
    all_categories = set()
    for values in data_dict.values():
        all_categories.update(values)
    
    all_categories = sorted(list(all_categories))
    
    x_pos = np.arange(len(all_categories))
    width = 0.8 / len(data_dict)
    
    for idx, (vcf_name, values) in enumerate(data_dict.items()):
        counts = [values.count(cat) for cat in all_categories]
        offset = (idx - len(data_dict)/2 + 0.5) * width
        ax1.bar(x_pos + offset, counts, width, alpha=0.7, 
               color=COLORS[idx], label=vcf_name)
    
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{feature_name} Distribution')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Proportion plot
    ax2 = axes[1]
    for idx, (vcf_name, values) in enumerate(data_dict.items()):
        total = len(values)
        proportions = [values.count(cat) / total * 100 for cat in all_categories]
        offset = (idx - len(data_dict)/2 + 0.5) * width
        ax2.bar(x_pos + offset, proportions, width, alpha=0.7,
               color=COLORS[idx], label=vcf_name)
    
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'{feature_name} Proportions')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(all_categories, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{feature_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n{feature_name} Statistics:")
    for vcf_name, values in data_dict.items():
        print(f"  {vcf_name}:")
        total = len(values)
        for cat in all_categories:
            count = values.count(cat)
            print(f"    {cat}: {count} ({count/total*100:.2f}%)")


def plot_2d_comparison(data_dict, feature1, feature2, output_prefix):
    """
    Plot 2D scatter comparison between two features
    """
    fig, axes = plt.subplots(1, len(data_dict), figsize=(6*len(data_dict), 5))
    
    if len(data_dict) == 1:
        axes = [axes]
    
    for idx, (vcf_name, features) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        x = np.array(features[feature1])
        y = np.array(features[feature2])
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) == 0:
            continue
        
        # 2D histogram
        h = ax.hist2d(x, y, bins=50, cmap='viridis', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count')
        
        # Calculate correlation
        if len(x) > 1:
            corr, pval = stats.spearmanr(x, y)
            ax.text(0.05, 0.95, f'ρ={corr:.3f}, p={pval:.2e}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f'{vcf_name}: {feature1} vs {feature2}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_{feature1}_vs_{feature2}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(all_features, output_prefix):
    """
    Create summary statistics table
    """
    summary_data = []
    
    numeric_features = ['DP', 'QUAL', 'GQ', 'BAF', 'QS', 'SGB', 'MQ0F', 
                       'VDB', 'RPB', 'MQB', 'BQB', 'MQSB']
    
    for vcf_name, features in all_features.items():
        row = {'VCF': vcf_name}
        row['Total_Variants'] = len(features['FILTER'])
        
        for feat in numeric_features:
            if len(features[feat]) > 0:
                values = np.array(features[feat])
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    row[f'{feat}_mean'] = np.mean(values)
                    row[f'{feat}_median'] = np.median(values)
                    row[f'{feat}_std'] = np.std(values)
                else:
                    row[f'{feat}_mean'] = np.nan
                    row[f'{feat}_median'] = np.nan
                    row[f'{feat}_std'] = np.nan
            else:
                row[f'{feat}_mean'] = np.nan
                row[f'{feat}_median'] = np.nan
                row[f'{feat}_std'] = np.nan
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{output_prefix}_summary_statistics.csv', index=False)
    print(f"\nSummary statistics saved to {output_prefix}_summary_statistics.csv")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compare VCF feature distributions across multiple files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare somatic vs germline
  python draw_vcf_statistics.py -i somatic.vcf.gz germline.vcf.gz -n Somatic Germline -o comparison
  
  # Single VCF analysis
  python draw_vcf_statistics.py -i variants.vcf.gz -n MyVariants -o output
  
  # Three-way comparison
  python draw_vcf_statistics.py -i file1.vcf.gz file2.vcf.gz file3.vcf.gz \\
                                 -n Sample1 Sample2 Sample3 -o comparison
        """
    )
    
    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input VCF files (up to 3, can be gzipped)')
    parser.add_argument('-n', '--names', nargs='+', required=True,
                       help='Names for each VCF file (must match number of inputs)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output prefix for plots and statistics')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.input) > 3:
        print("Error: Maximum 3 VCF files supported")
        sys.exit(1)
    
    if len(args.input) != len(args.names):
        print("Error: Number of names must match number of input files")
        sys.exit(1)
    
    print("="*80)
    print("VCF Feature Distribution Analysis")
    print("="*80)
    
    # Parse all VCF files
    all_features = {}
    for vcf_file, vcf_name in zip(args.input, args.names):
        print(f"\nParsing {vcf_name}: {vcf_file}")
        features = parse_vcf(vcf_file)
        all_features[vcf_name] = features
        print(f"  Found {len(features['FILTER'])} variants")
    
    # Create summary table
    print("\n" + "="*80)
    print("Creating summary statistics...")
    summary_df = create_summary_table(all_features, args.output)
    
    # Plot numeric features
    print("\n" + "="*80)
    print("Generating distribution plots...")
    
    numeric_features = {
        'DP': {'bins': np.arange(0, 201, 10), 'log_scale': True, 'xlim': (0, 200)},
        'QUAL': {'bins': 50, 'log_scale': True, 'xlim': None},
        'GQ': {'bins': np.arange(0, 101, 5), 'log_scale': True, 'xlim': (0, 100)},
        'BAF': {'bins': np.arange(0, 1.05, 0.05), 'log_scale': False, 'xlim': (0, 1)},
        'QS': {'bins': 50, 'log_scale': True, 'xlim': (0, 1)},
        'SGB': {'bins': 50, 'log_scale': True, 'xlim': None},
        'MQ0F': {'bins': np.arange(0, 1.05, 0.05), 'log_scale': False, 'xlim': (0, 1)},
        'VDB': {'bins': np.arange(0, 1.05, 0.05), 'log_scale': False, 'xlim': (0, 1)},
        'RPB': {'bins': 50, 'log_scale': False, 'xlim': (0, 1)},
        'MQB': {'bins': 50, 'log_scale': False, 'xlim': (0, 1)},
        'BQB': {'bins': 50, 'log_scale': False, 'xlim': (0, 1)},
        'MQSB': {'bins': 50, 'log_scale': False, 'xlim': (0, 1)},
    }
    
    for feature_name, plot_params in numeric_features.items():
        data_dict = {}
        for vcf_name, features in all_features.items():
            if len(features[feature_name]) > 0:
                data_dict[vcf_name] = features[feature_name]
        
        if len(data_dict) > 0:
            print(f"  Plotting {feature_name}...")
            plot_distribution_comparison(data_dict, feature_name, args.output, **plot_params)
    
    # Plot categorical features
    print("\nGenerating categorical plots...")
    for feature_name in ['GT', 'FILTER']:
        data_dict = {}
        for vcf_name, features in all_features.items():
            if len(features[feature_name]) > 0:
                data_dict[vcf_name] = features[feature_name]
        
        if len(data_dict) > 0:
            print(f"  Plotting {feature_name}...")
            plot_categorical_comparison(data_dict, feature_name, args.output)
    
    # Plot 2D comparisons
    print("\nGenerating 2D comparison plots...")
    comparison_pairs = [
        ('DP', 'GQ'),
        ('DP', 'BAF'),
        ('BAF', 'GQ'),
        ('QS', 'BAF'),
    ]
    
    for feat1, feat2 in comparison_pairs:
        data_dict = {}
        for vcf_name, features in all_features.items():
            if len(features[feat1]) > 0 and len(features[feat2]) > 0:
                data_dict[vcf_name] = features
        
        if len(data_dict) > 0:
            print(f"  Plotting {feat1} vs {feat2}...")
            plot_2d_comparison(data_dict, feat1, feat2, args.output)
    
    # I16 statistics if available
    print("\nGenerating I16 statistics plots...")
    i16_features = ['alt_strand_ratio', 'ref_strand_ratio']
    for i16_feat in i16_features:
        data_dict = {}
        for vcf_name, features in all_features.items():
            if i16_feat in features['I16_stats'] and len(features['I16_stats'][i16_feat]) > 0:
                data_dict[vcf_name] = features['I16_stats'][i16_feat]
        
        if len(data_dict) > 0:
            print(f"  Plotting I16 {i16_feat}...")
            plot_distribution_comparison(data_dict, f'I16_{i16_feat}', args.output,
                                        bins=np.arange(0, 1.05, 0.05), xlim=(0, 1))
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"All plots and statistics saved with prefix: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()

###
# python draw_vcf_statistics.py \
#     -i /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_MPILEUP/overlap_MPILEUP_P4_somatic_Mutect2_all/0000.vcf.gz \
#        /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_MPILEUP/overlap_MPILEUP_P4_normal_wes_all/0000.vcf.gz \
#     -n Somatic_Mutect2 Germline_WES \
#     -o P4_somatic_vs_germline

###