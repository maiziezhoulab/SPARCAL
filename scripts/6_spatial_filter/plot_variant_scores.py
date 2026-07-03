#!/usr/bin/env python3
"""
Visualize Germline vs Somatic Scores
=====================================
Creates scatter plots to visualize the distribution of variants based on their
germline and somatic scores from the spatial SNV filter enhanced pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Default thresholds — must match the SLURM script and run_spatial_filter_enhanced.py
DEFAULT_GERMLINE_THRESHOLD = 0.3
DEFAULT_SOMATIC_THRESHOLD  = 0.2

def load_variant_scores(txt_file):
    """Load variant scores from text file."""
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"File not found: {txt_file}")
    
    df = pd.read_csv(txt_file, sep='\t')
    print(f"Loaded {len(df)} variants from {txt_file}")
    print(f"Columns: {list(df.columns)}")
    return df

def plot_score_scatter(df, output_file, title=None, 
                      germline_threshold=DEFAULT_GERMLINE_THRESHOLD,
                      somatic_threshold=DEFAULT_SOMATIC_THRESHOLD):
    """
    Create a scatter plot of germline_score vs somatic_score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with germline_score and somatic_score columns
    output_file : str
        Path to save the plot
    title : str
        Custom title for the plot
    germline_threshold : float
        Threshold for germline classification (default: 0.5)
    somatic_threshold : float
        Threshold for somatic classification (default: 0.4)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine classification for each variant based on scores
    def classify_variant(row):
        g_score = row['germline_score']
        s_score = row['somatic_score']
        
        if g_score > germline_threshold and s_score < somatic_threshold:
            return 'Germline'
        elif s_score > somatic_threshold and g_score < germline_threshold:
            return 'Somatic'
        else:
            return 'Ambiguous'
    
    df['classification'] = df.apply(classify_variant, axis=1)
    
    # Color mapping
    color_map = {
        'Germline': '#3498db',    # Blue
        'Somatic': '#e74c3c',     # Red
        'Ambiguous': '#95a5a6'    # Gray
    }
    
    # Plot each classification separately
    for classification in ['Ambiguous', 'Germline', 'Somatic']:  # Plot ambiguous first (background)
        subset = df[df['classification'] == classification]
        if len(subset) > 0:
            ax.scatter(subset['somatic_score'], subset['germline_score'],
                      c=color_map[classification], label=classification,
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add classification region boundaries
    # Germline region (top-left)
    germline_rect = Rectangle((0, germline_threshold), somatic_threshold, 
                              1-germline_threshold, 
                              facecolor='blue', alpha=0.05, edgecolor='blue', 
                              linewidth=2, linestyle='--')
    ax.add_patch(germline_rect)
    
    # Somatic region (bottom-right)
    somatic_rect = Rectangle((somatic_threshold, 0), 1-somatic_threshold, 
                             1,
                             facecolor='red', alpha=0.05, edgecolor='red',
                             linewidth=2, linestyle='--')
    ax.add_patch(somatic_rect)
    
    # Add threshold lines
    ax.axhline(y=germline_threshold, color='blue', linestyle='--', 
              linewidth=1.5, alpha=0.5, label=f'Germline threshold ({germline_threshold})')
    ax.axvline(x=somatic_threshold, color='red', linestyle='--', 
              linewidth=1.5, alpha=0.5, label=f'Somatic threshold ({somatic_threshold})')
    
    # Add diagonal reference line (where germline_score = somatic_score)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Equal scores')
    
    # Labels and title
    ax.set_xlabel('Somatic Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Germline Score', fontsize=14, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title('Germline vs Somatic Score Distribution', 
                    fontsize=16, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Legend
    ax.legend(loc='center right', fontsize=10, framealpha=0.9)
    
    # Add statistics text box
    stats_text = f"""Classification Summary:
Germline: {len(df[df['classification']=='Germline'])} ({len(df[df['classification']=='Germline'])/len(df)*100:.1f}%)
Somatic: {len(df[df['classification']=='Somatic'])} ({len(df[df['classification']=='Somatic'])/len(df)*100:.1f}%)
Ambiguous: {len(df[df['classification']=='Ambiguous'])} ({len(df[df['classification']=='Ambiguous'])/len(df)*100:.1f}%)
Total: {len(df)}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_file}")
    
    plt.close()
    
    return df

def plot_race_comparison(df, output_file, title=None):
    """
    Create a scatter plot colored by race (defined vs denovo).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color mapping for race
    race_colors = {
        'defined': '#2ecc71',  # Green
        'denovo': '#9b59b6',   # Purple
        'unknown': '#95a5a6'   # Gray
    }
    
    # Plot each race separately
    for race in ['unknown', 'denovo', 'defined']:  # Plot unknown first (background)
        subset = df[df['race'] == race]
        if len(subset) > 0:
            ax.scatter(subset['somatic_score'], subset['germline_score'],
                      c=race_colors[race], label=race.capitalize(),
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Equal scores')
    
    # Labels and title
    ax.set_xlabel('Somatic Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Germline Score', fontsize=14, fontweight='bold')
    
    if title:
        ax.set_title(f'{title} - By Variant Race', fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title('Variant Scores by Race (Defined vs Denovo)', 
                    fontsize=16, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistics
    stats_text = f"""Race Distribution:
Defined: {len(df[df['race']=='defined'])} ({len(df[df['race']=='defined'])/len(df)*100:.1f}%)
Denovo: {len(df[df['race']=='denovo'])} ({len(df[df['race']=='denovo'])/len(df)*100:.1f}%)
Unknown: {len(df[df['race']=='unknown'])} ({len(df[df['race']=='unknown'])/len(df)*100:.1f}%)
Total: {len(df)}"""
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Race comparison plot saved to: {output_file}")
    plt.close()

def plot_combined_view(df, output_file, title=None,
                      germline_threshold=DEFAULT_GERMLINE_THRESHOLD,
                      somatic_threshold=DEFAULT_SOMATIC_THRESHOLD):
    """
    Create a combined view with multiple subplots.
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main scatter plot (classification)
    ax1 = fig.add_subplot(gs[0, :])
    
    def classify_variant(row):
        g_score = row['germline_score']
        s_score = row['somatic_score']
        if g_score > germline_threshold and s_score < somatic_threshold:
            return 'Germline'
        elif s_score > somatic_threshold and g_score < germline_threshold:
            return 'Somatic'
        else:
            return 'Ambiguous'
    
    df['classification'] = df.apply(classify_variant, axis=1)
    
    color_map = {'Germline': '#3498db', 'Somatic': '#e74c3c', 'Ambiguous': '#95a5a6'}
    
    for classification in ['Ambiguous', 'Germline', 'Somatic']:
        subset = df[df['classification'] == classification]
        if len(subset) > 0:
            ax1.scatter(subset['somatic_score'], subset['germline_score'],
                       c=color_map[classification], label=classification,
                       alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    
    ax1.axhline(y=germline_threshold, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axvline(x=somatic_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Somatic Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Germline Score', fontsize=12, fontweight='bold')
    ax1.set_title('Classification by Score Thresholds', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Race distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    race_colors = {'defined': '#2ecc71', 'denovo': '#9b59b6', 'unknown': '#95a5a6'}
    
    for race in ['unknown', 'denovo', 'defined']:
        subset = df[df['race'] == race]
        if len(subset) > 0:
            ax2.scatter(subset['somatic_score'], subset['germline_score'],
                       c=race_colors[race], label=race.capitalize(),
                       alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Somatic Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Germline Score', fontsize=12, fontweight='bold')
    ax2.set_title('Variant Race Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    
    # 3. Score distributions (histograms)
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.hist(df['germline_score'], bins=30, alpha=0.5, color='blue', 
            label='Germline Score', edgecolor='black', linewidth=0.5)
    ax3.hist(df['somatic_score'], bins=30, alpha=0.5, color='red', 
            label='Somatic Score', edgecolor='black', linewidth=0.5)
    
    ax3.axvline(x=germline_threshold, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(x=somatic_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Score Distributions', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Variant Score Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined view saved to: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize germline vs somatic scores from spatial SNV filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from all_variant_scores.txt
  python %(prog)s \\
    --input /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt \\
    --output /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/score_scatter.png

  # Plot from germline_defined.txt
  python %(prog)s \\
    --input /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/defined/germline_defined.txt \\
    --output germline_defined_scatter.png \\
    --title "P4 Tumor Section 1 - Germline Defined Variants"

  # Generate all three plot types
./plot_variant_scores.py     --input /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt --output_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/plots/ --all
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input txt file with variant scores')
    parser.add_argument('--output', default=None,
                       help='Output file for single plot (PNG format)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for all plots (when using --all)')
    parser.add_argument('--title', default=None,
                       help='Custom title for the plot')
    parser.add_argument('--germline_threshold', type=float, default=DEFAULT_GERMLINE_THRESHOLD,
                       help=f'Germline score threshold (default: {DEFAULT_GERMLINE_THRESHOLD})')
    parser.add_argument('--somatic_threshold', type=float, default=DEFAULT_SOMATIC_THRESHOLD,
                       help=f'Somatic score threshold (default: {DEFAULT_SOMATIC_THRESHOLD})')
    parser.add_argument('--all', action='store_true',
                       help='Generate all plot types (requires --output_dir)')
    parser.add_argument('--plot_type', default='classification',
                       choices=['classification', 'race', 'combined'],
                       help='Type of plot to generate (default: classification)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading variant scores from: {args.input}")
    df = load_variant_scores(args.input)
    
    # Validate required columns
    required_cols = ['germline_score', 'somatic_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Check for race column
    if 'race' not in df.columns:
        print("Warning: 'race' column not found. Setting all variants to 'unknown'")
        df['race'] = 'unknown'
    
    # Generate plots
    if args.all:
        if not args.output_dir:
            print("Error: --output_dir is required when using --all")
            sys.exit(1)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate all three types
        plot_score_scatter(
            df, 
            os.path.join(args.output_dir, 'classification_scatter.png'),
            title=args.title,
            germline_threshold=args.germline_threshold,
            somatic_threshold=args.somatic_threshold
        )
        
        plot_race_comparison(
            df,
            os.path.join(args.output_dir, 'race_scatter.png'),
            title=args.title
        )
        
        plot_combined_view(
            df,
            os.path.join(args.output_dir, 'combined_view.png'),
            title=args.title,
            germline_threshold=args.germline_threshold,
            somatic_threshold=args.somatic_threshold
        )
        
        print(f"\nAll plots saved to: {args.output_dir}")
        
    else:
        if not args.output:
            print("Error: --output is required when not using --all")
            sys.exit(1)
        
        # Generate single plot based on type
        if args.plot_type == 'classification':
            df = plot_score_scatter(
                df, 
                args.output,
                title=args.title,
                germline_threshold=args.germline_threshold,
                somatic_threshold=args.somatic_threshold
            )
        elif args.plot_type == 'race':
            plot_race_comparison(df, args.output, title=args.title)
        elif args.plot_type == 'combined':
            plot_combined_view(
                df, 
                args.output,
                title=args.title,
                germline_threshold=args.germline_threshold,
                somatic_threshold=args.somatic_threshold
            )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total variants: {len(df)}")
    print(f"\nGermline Score - Mean: {df['germline_score'].mean():.4f}, "
          f"Median: {df['germline_score'].median():.4f}, "
          f"Std: {df['germline_score'].std():.4f}")
    print(f"Somatic Score  - Mean: {df['somatic_score'].mean():.4f}, "
          f"Median: {df['somatic_score'].median():.4f}, "
          f"Std: {df['somatic_score'].std():.4f}")
    
    if 'classification' in df.columns:
        print(f"\nClassification breakdown:")
        for cls in ['Germline', 'Somatic', 'Ambiguous']:
            count = len(df[df['classification'] == cls])
            pct = count / len(df) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")
    
    if 'race' in df.columns and df['race'].notna().any():
        print(f"\nRace breakdown:")
        for race in df['race'].unique():
            count = len(df[df['race'] == race])
            pct = count / len(df) * 100
            print(f"  {race}: {count} ({pct:.1f}%)")
    
    print("="*60)

if __name__ == "__main__":
    main()