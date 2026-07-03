import os
import gzip
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_strelka_output(strelka_dir, quality_filter="strelkaQ0", barcodes=None):
    """
    Analyze Strelka2 output files to verify they contain SNPs.
    
    Args:
        strelka_dir: Base directory containing Strelka2 output
        quality_filter: Quality filter subdirectory
        barcodes: List of specific barcodes to analyze (None for all)
    
    Returns:
        Dictionary containing variant type statistics
    """
    stats = {
        'total_variants': 0,
        'snps': 0,
        'indels': 0,
        'other': 0,
        'variant_types': defaultdict(int)
    }
    
    base_path = os.path.join(strelka_dir, quality_filter)
    print(f"Analyzing Strelka2 output in {base_path}")
    
    # Find all VCF files using more specific path structure
    vcf_files = []
    
    # If barcodes are specified, use them directly to construct paths
    if barcodes:
        for barcode in barcodes:
            # Construct the expected path directly
            vcf_path = os.path.join(base_path, f"{barcode}-1", "results", "variants", "variants.vcf.gz")
            if os.path.exists(vcf_path):
                vcf_files.append(vcf_path)
    else:
        # If no barcodes specified, list all barcode directories directly
        try:
            barcode_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for barcode_dir in barcode_dirs:
                vcf_path = os.path.join(base_path, barcode_dir, "results", "variants", "variants.vcf.gz")
                if os.path.exists(vcf_path):
                    vcf_files.append(vcf_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error accessing directory {base_path}: {e}")
    
    print(f"Found {len(vcf_files)} VCF files to analyze")
    
    # Process each file with progress bar
    for vcf_file in tqdm(vcf_files, desc="Processing VCF files"):
        with gzip.open(vcf_file, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                ref = fields[3]
                alt = fields[4]
                
                stats['total_variants'] += 1
                
                # Classify variant type
                if len(ref) == 1 and len(alt) == 1:
                    stats['snps'] += 1
                    variant_type = f"{ref}>{alt}"
                    stats['variant_types'][variant_type] += 1
                elif len(ref) > len(alt):
                    stats['indels'] += 1
                    stats['variant_types']['deletion'] += 1
                elif len(ref) < len(alt):
                    stats['indels'] += 1
                    stats['variant_types']['insertion'] += 1
                else:
                    stats['other'] += 1
                    stats['variant_types']['other'] += 1
    
    return stats

def plot_variant_stats(stats, output_dir):
    """Create plots to visualize variant statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Pie chart of variant types
    labels = ['SNPs', 'Indels', 'Other']
    sizes = [stats['snps'], stats['indels'], stats['other']]
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Strelka2 Variant Type Distribution')
    plt.savefig(os.path.join(output_dir, 'variant_types_pie.png'), dpi=300)
    plt.close()
    
    # Bar chart of SNP transitions/transversions
    snp_types = {k: v for k, v in stats['variant_types'].items() if '>' in k}
    
    if snp_types:
        plt.figure(figsize=(12, 8))
        plt.bar(snp_types.keys(), snp_types.values())
        plt.xticks(rotation=45)
        plt.title('SNP Types Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'snp_types_bar.png'), dpi=300)
        plt.close()
    
    # Save statistics to text file
    with open(os.path.join(output_dir, 'variant_stats.txt'), 'w') as f:
        f.write(f"Total variants: {stats['total_variants']}\n")
        f.write(f"SNPs: {stats['snps']} ({stats['snps']/stats['total_variants']*100:.2f}%)\n")
        f.write(f"Indels: {stats['indels']} ({stats['indels']/stats['total_variants']*100:.2f}%)\n")
        f.write(f"Other: {stats['other']} ({stats['other']/stats['total_variants']*100:.2f}%)\n\n")
        
        f.write("Variant type counts:\n")
        for vtype, count in sorted(stats['variant_types'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"{vtype}: {count}\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze Strelka2 output to verify SNPs')
    parser.add_argument('--strelka-dir', required=True, help='Directory containing Strelka2 output')
    parser.add_argument('--quality-filter', default='strelkaQ0', help='Quality filter subdirectory')
    parser.add_argument('--output-dir', default='strelka_validation', help='Output directory for statistics')
    parser.add_argument('--barcodes', nargs='+', help='Specific barcodes to analyze (faster results)')
    parser.add_argument('--sample', type=int, help='Randomly sample N barcodes for analysis')
    
    args = parser.parse_args()
    
    barcodes = None
    if args.barcodes:
        barcodes = args.barcodes
        print(f"Analyzing specific barcodes: {', '.join(barcodes)}")
    elif args.sample:
        # Find available barcodes directly from the directory structure
        base_path = os.path.join(args.strelka_dir, args.quality_filter)
        try:
            all_barcodes = [d.split('-')[0] for d in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, d)) and '-' in d]
            
            if all_barcodes:
                import random
                sample_size = min(args.sample, len(all_barcodes))
                barcodes = random.sample(all_barcodes, sample_size)
                print(f"Randomly sampling {sample_size} barcodes for analysis")
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error accessing directory {base_path}: {e}")
            return
    
    stats = analyze_strelka_output(args.strelka_dir, args.quality_filter, barcodes)
    
    if stats['total_variants'] == 0:
        print("No variants found! Check your input directory and barcode specifications.")
        return
    
    plot_variant_stats(stats, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    print(f"Total variants: {stats['total_variants']}")
    print(f"SNPs: {stats['snps']} ({stats['snps']/stats['total_variants']*100:.2f}%)")
    print(f"Indels: {stats['indels']} ({stats['indels']/stats['total_variants']*100:.2f}%)")
    print(f"Other: {stats['other']} ({stats['other']/stats['total_variants']*100:.2f}%)")

if __name__ == "__main__":
    main()

# Run the script for DLPFC, path: /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka/strelkaQ0
# python scripts/strelka/check_strelka_spatial.py --strelka-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka --quality-filter strelkaQ0 --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka/strelkaQ0_validation

# Run with specific barcodes for faster results:
# python scripts/strelka/check_strelka_spatial.py --strelka-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka --quality-filter strelkaQ0 --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka/strelkaQ0_validation --barcodes AAACGAAAGGGCCGTT AAACGAAAGTTGCCTA

# Run with random sampling:
# python scripts/strelka/check_strelka_spatial.py --strelka-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka --quality-filter strelkaQ0 --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/strelka/strelkaQ0_validation --sample 10

# Purpose of this scripts:
# 1. Check the quality of the strelka output
# 2. Generate the statistics of the strelka output
# 3. Generate the plots of the statistics
