#!/usr/bin/env python3
"""
Script to investigate and validate VCF position offsets between different callers.
Checks if there's a systematic +1 or -1 offset in positions.
"""

import gzip
import argparse
from collections import defaultdict
import sys

def open_vcf(filename):
    """Open VCF file (handles both gzipped and uncompressed)"""
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    return open(filename, 'r')

def parse_vcf_positions(vcf_file, max_variants=100000):
    """
    Parse VCF file and extract position information
    Returns: dict with chr as key and list of (pos, ref, alt) tuples
    """
    positions = defaultdict(list)
    count = 0
    
    with open_vcf(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]
            
            positions[chrom].append((pos, ref, alt))
            count += 1
            
            if count >= max_variants:
                break
    
    return positions, count

def check_offset(pos_dict1, pos_dict2, label1, label2):
    """
    Check if there's a systematic position offset between two VCF files
    """
    print(f"\n{'='*60}")
    print(f"Checking position offset between {label1} and {label2}")
    print(f"{'='*60}\n")
    
    # Track matches at different offsets
    matches_same = 0  # Same position
    matches_plus1 = 0  # dict2 position is dict1 + 1
    matches_minus1 = 0  # dict2 position is dict1 - 1
    
    total_checked = 0
    examples_same = []
    examples_plus1 = []
    examples_minus1 = []
    
    # Check each chromosome
    for chrom in pos_dict1.keys():
        if chrom not in pos_dict2:
            continue
        
        # Create position sets for quick lookup
        pos_set1 = {(pos, ref, alt) for pos, ref, alt in pos_dict1[chrom]}
        pos_set2 = {(pos, ref, alt) for pos, ref, alt in pos_dict2[chrom]}
        
        # Check for matches at different offsets
        for pos1, ref1, alt1 in pos_dict1[chrom][:10000]:  # Check first 1000 per chr
            total_checked += 1
            
            # Check same position
            if (pos1, ref1, alt1) in pos_set2:
                matches_same += 1
                if len(examples_same) < 5:
                    examples_same.append((chrom, pos1, ref1, alt1))
            
            # Check +1 offset (dict2 position is 1 more)
            if (pos1 + 1, ref1, alt1) in pos_set2:
                matches_plus1 += 1
                if len(examples_plus1) < 5:
                    examples_plus1.append((chrom, pos1, pos1+1, ref1, alt1))
            
            # Check -1 offset (dict2 position is 1 less)
            if (pos1 - 1, ref1, alt1) in pos_set2:
                matches_minus1 += 1
                if len(examples_minus1) < 5:
                    examples_minus1.append((chrom, pos1, pos1-1, ref1, alt1))
    
    # Report results
    print(f"Total variants checked: {total_checked}")
    print(f"\nMatching statistics:")
    print(f"  Same position (0 offset):  {matches_same:6d} ({100*matches_same/total_checked:.2f}%)")
    print(f"  +1 offset ({label2} = {label1}+1): {matches_plus1:6d} ({100*matches_plus1/total_checked:.2f}%)")
    print(f"  -1 offset ({label2} = {label1}-1): {matches_minus1:6d} ({100*matches_minus1/total_checked:.2f}%)")
    
    # Show examples
    if examples_same:
        print(f"\nExamples of SAME position matches:")
        print("  Chr\tPosition\tRef\tAlt")
        for chrom, pos, ref, alt in examples_same[:3]:
            print(f"  {chrom}\t{pos}\t{ref}\t{alt}")
    
    if examples_plus1:
        print(f"\nExamples of +1 offset matches ({label1} -> {label2}):")
        print(f"  Chr\t{label1}_pos\t{label2}_pos\tRef\tAlt")
        for chrom, pos1, pos2, ref, alt in examples_plus1[:3]:
            print(f"  {chrom}\t{pos1}\t{pos2}\t{ref}\t{alt}")
    
    if examples_minus1:
        print(f"\nExamples of -1 offset matches ({label1} -> {label2}):")
        print(f"  Chr\t{label1}_pos\t{label2}_pos\tRef\tAlt")
        for chrom, pos1, pos2, ref, alt in examples_minus1[:3]:
            print(f"  {chrom}\t{pos1}\t{pos2}\t{ref}\t{alt}")
    
    # Determine recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print(f"{'='*60}")
    
    max_matches = max(matches_same, matches_plus1, matches_minus1)
    if max_matches == matches_same and matches_same > total_checked * 0.1:
        print(f"✓ Positions appear to be CORRECTLY ALIGNED (no offset needed)")
    elif max_matches == matches_plus1 and matches_plus1 > total_checked * 0.1:
        print(f"⚠ {label1} positions need +1 adjustment to match {label2}")
        print(f"  OR {label2} positions need -1 adjustment to match {label1}")
    elif max_matches == matches_minus1 and matches_minus1 > total_checked * 0.1:
        print(f"⚠ {label1} positions need -1 adjustment to match {label2}")
        print(f"  OR {label2} positions need +1 adjustment to match {label1}")
    else:
        print(f"⚠ WARNING: Low match rate detected. Issues may be more complex than simple offset.")
    
    return matches_same, matches_plus1, matches_minus1

def main():
    parser = argparse.ArgumentParser(
        description='Check for position offsets between VCF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if mpileup needs position adjustment
  python check_vcf_positions.py \\
    --vcf1 mpileup.vcf.gz --label1 MPILEUP \\
    --vcf2 mutect2.vcf.gz --label2 MUTECT2

  # Check multiple comparisons
  python check_vcf_positions.py \\
    --vcf1 mpileup.vcf.gz --label1 MPILEUP \\
    --vcf2 sparcal.vcf.gz --label2 SPARCAL \\
    --vcf3 beagle.vcf.gz --label3 BEAGLE
        """
    )
    
    parser.add_argument('--vcf1', required=True, help='First VCF file')
    parser.add_argument('--label1', required=True, help='Label for first VCF')
    parser.add_argument('--vcf2', required=True, help='Second VCF file')
    parser.add_argument('--label2', required=True, help='Label for second VCF')
    parser.add_argument('--vcf3', help='Optional third VCF file')
    parser.add_argument('--label3', help='Label for third VCF')
    parser.add_argument('--max-variants', type=int, default=10000,
                       help='Maximum variants to check per file (default: 10000)')
    
    args = parser.parse_args()
    
    # Parse VCF files
    print("Parsing VCF files...")
    print(f"Reading {args.label1}: {args.vcf1}")
    pos1, count1 = parse_vcf_positions(args.vcf1, args.max_variants)
    print(f"  Found {count1} variants")
    
    print(f"Reading {args.label2}: {args.vcf2}")
    pos2, count2 = parse_vcf_positions(args.vcf2, args.max_variants)
    print(f"  Found {count2} variants")
    
    # Compare vcf1 vs vcf2
    check_offset(pos1, pos2, args.label1, args.label2)
    
    # If third VCF provided, do additional comparisons
    if args.vcf3 and args.label3:
        print(f"\nReading {args.label3}: {args.vcf3}")
        pos3, count3 = parse_vcf_positions(args.vcf3, args.max_variants)
        print(f"  Found {count3} variants")
        
        check_offset(pos1, pos3, args.label1, args.label3)
        check_offset(pos2, pos3, args.label2, args.label3)

if __name__ == '__main__':
    main()