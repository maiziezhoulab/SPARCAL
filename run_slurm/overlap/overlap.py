#!/usr/bin/env python3
"""
Script to create a Venn diagram showing overlap between two VCF files
Usage: python vcf_venn_diagram.py --file1 <vcf1.gz> --file2 <vcf2.gz> --overlap <overlap_dir> --output <output.png>
"""

import argparse
import subprocess
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os


def count_variants(vcf_file):
    """Count variants in a VCF file (excluding header lines)"""
    cmd = f"zcat {vcf_file} | grep -v '##' | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # Subtract 1 for the #CHROM header line
    count = int(result.stdout.strip()) - 1
    return max(0, count)


def get_file_label(vcf_file):
    """Extract a clean label from the VCF filename"""
    basename = os.path.basename(vcf_file)
    # Remove .vcf.gz extension
    label = basename.replace('.vcf.gz', '').replace('.vcf', '')
    return label


def create_venn_diagram(file1, file2, overlap_dir, output_name, label1=None, label2=None):
    """
    Create a Venn diagram from two VCF files and their intersection
    
    Parameters:
    -----------
    file1 : str
        Path to first VCF file
    file2 : str
        Path to second VCF file
    overlap_dir : str
        Directory containing bcftools isec output (should have 0000.vcf.gz)
    output_name : str
        Output filename for the Venn diagram
    label1 : str, optional
        Custom label for file1
    label2 : str, optional
        Custom label for file2
    """
    
    # Count variants in each file
    print("Counting variants in input files...")
    total_file1 = count_variants(file1)
    total_file2 = count_variants(file2)
    
    # Count overlapping variants
    overlap_file = os.path.join(overlap_dir, "0000.vcf.gz")
    if not os.path.exists(overlap_file):
        raise FileNotFoundError(f"Overlap file not found: {overlap_file}")
    
    print("Counting overlapping variants...")
    overlap_count = count_variants(overlap_file)
    
    # Calculate unique variants
    unique_file1 = total_file1 - overlap_count
    unique_file2 = total_file2 - overlap_count
    
    # Print summary
    print("\n" + "="*60)
    print("Variant Count Summary:")
    print("="*60)
    print(f"File 1 total: {total_file1:,}")
    print(f"File 2 total: {total_file2:,}")
    print(f"Overlap: {overlap_count:,}")
    print(f"File 1 unique: {unique_file1:,}")
    print(f"File 2 unique: {unique_file2:,}")
    print("="*60 + "\n")
    
    # Create labels
    if label1 is None:
        label1 = get_file_label(file1)
    if label2 is None:
        label2 = get_file_label(file2)
    
    # Create Venn diagram
    plt.figure(figsize=(10, 8))
    
    # venn2 expects (Ab, aB, AB) where:
    # Ab = unique to set A
    # aB = unique to set B
    # AB = intersection
    venn = venn2(subsets=(unique_file1, unique_file2, overlap_count),
                 set_labels=(label1, label2))
    
    # Customize appearance
    if venn.get_label_by_id('10'):
        venn.get_label_by_id('10').set_text(f'{unique_file1:,}')
    if venn.get_label_by_id('01'):
        venn.get_label_by_id('01').set_text(f'{unique_file2:,}')
    if venn.get_label_by_id('11'):
        venn.get_label_by_id('11').set_text(f'{overlap_count:,}')
    
    # Add title
    plt.title('VCF Variant Overlap Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Venn diagram saved to: {output_name}")
    
    return {
        'file1_total': total_file1,
        'file2_total': total_file2,
        'overlap': overlap_count,
        'file1_unique': unique_file1,
        'file2_unique': unique_file2
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create Venn diagram from VCF intersection results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python vcf_venn_diagram.py \\
    --file1 /path/to/file1.vcf.gz \\
    --file2 /path/to/file2.vcf.gz \\
    --overlap /path/to/overlap_output_dir \\
    --label1 "Monopogen" \\
    --label2 "GATK"
  
  # Output will be: venn_overlap_output_dir.png
  
  # Or specify custom output name:
  python vcf_venn_diagram.py \\
    --file1 /path/to/file1.vcf.gz \\
    --file2 /path/to/file2.vcf.gz \\
    --overlap /path/to/overlap_output_dir \\
    --output custom_name.png
        """
    )
    
    parser.add_argument('--file1', required=True, help='Path to first VCF file')
    parser.add_argument('--file2', required=True, help='Path to second VCF file')
    parser.add_argument('--overlap', required=True, help='Directory with bcftools isec output')
    parser.add_argument('--output', help='Output filename (default: venn_<overlap_dirname>.png)')
    parser.add_argument('--label1', help='Custom label for file1')
    parser.add_argument('--label2', help='Custom label for file2')
    
    args = parser.parse_args()
    
    # Generate default output name if not provided
    if args.output is None:
        overlap_dirname = os.path.basename(args.overlap.rstrip('/'))
        args.output = f'venn_{overlap_dirname}.png'
        print(f"Using default output name: {args.output}\n")
    
    create_venn_diagram(
        args.file1,
        args.file2,
        args.overlap,
        args.output,
        args.label1,
        args.label2
    )


if __name__ == '__main__':
    main()



# Usage
# Output will be: venn_overlap_p4tumorgatk_monopogen.png
