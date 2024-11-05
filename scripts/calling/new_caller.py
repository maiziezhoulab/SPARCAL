from typing import List, Tuple, Optional
import pysam
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple
import pysam
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class VariantCall:
    """Class to store variant information"""
    position: int
    reference: str
    alternate: str
    read_name: str
    base_quality: int


@dataclass
class VCFRecord:
    chrom: str
    pos: int
    ref: str
    alt: str
    info: str  # Will contain A:C:G:T ratio
    format: str  # Will contain genotype (0/1 or 1/1)

    def __str__(self) -> str:
        """Convert record to VCF format string"""
        return f"{self.chrom}\t{self.pos}\t.\t{self.ref}\t{self.alt}\t.\tPASS\t{self.info}\tGT\t{self.format}"

def determine_genotype(base_counts: dict, ref_base: str) -> str:
    """
    Simple genotype determination
    
    Args:
        base_counts: Dictionary of base counts (A, C, G, T)
        ref_base: Reference base
        
    Returns:
        Genotype string (0/1 or 1/1)
    """
    # Get alternate bases that have non-zero counts
    alt_bases = [base for base, count in base_counts.items() 
                if count > 0 and base != ref_base]
    
    # If we have multiple alternate alleles, return 1/1
    return "1/1" if len(alt_bases) > 1 else "0/1"

def get_valid_chromosome_name(bam_file: str, chromosome: str) -> Optional[str]:
    """
    Get the valid chromosome name as it appears in the BAM file.
    
    Args:
        bam_file: Path to the BAM file
        chromosome: Input chromosome name (e.g., 'chr1', '1', 'X')
    
    Returns:
        Valid chromosome name or None if not found
    """
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Get list of reference names from BAM file
        valid_refs = bam.references
        
        # Different possible formats of chromosome names
        possible_names = [
            chromosome,
            chromosome.replace('chr', ''),
            f'chr{chromosome}',
            chromosome.upper(),
            chromosome.lower()
        ]
        
        # Try to find a match
        for name in possible_names:
            if name in valid_refs:
                return name
    
    return None

class RegionFilter:
    """Class to handle BED region filtering"""
    def __init__(self, bed_file: str, chromosome: str):
        self.regions = self._load_regions(bed_file, chromosome)
    
    def _load_regions(self, bed_file: str, chromosome: str) -> List[Tuple[int, int]]:
        """Load regions from BED file for given chromosome"""
        regions = []
        with open(bed_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == str(chromosome):
                    regions.append((int(parts[1]), int(parts[2])))
        return sorted(regions)
    
    def in_regions(self, position: int) -> bool:
        """Check if position falls within any region"""
        for start, end in self.regions:
            if start <= position <= end:
                return True
            if position < start:
                break
        return False
    
# class BaseCounter:
#     """Class to handle base counting at each position"""
#     def __init__(self):
#         self.counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    
#     def add_base(self, base: str):
#         """Add a base observation"""
#         if base in self.counts:
#             self.counts[base] += 1
            
#     def get_count_string(self) -> str:
#         """Get counts in A:C:G:T format"""
#         return f"{self.counts['A']}:{self.counts['C']}:{self.counts['G']}:{self.counts['T']}"
    
#     def get_total(self) -> int:
#         """Get total number of reads"""
#         return sum(self.counts.values())
    
#     def has_variants(self, ref_base: str) -> bool:
#         """Check if there are any alternate alleles"""
#         return any(base != ref_base and count > 0 
#                   for base, count in self.counts.items())


class EnhancedVariantCaller:
    def __init__(self, reference: str, chromosome: str, bamfile: str, 
                 bedfile: str, min_base_quality: int = 0, min_mapping_quality: int = 0):
        self.reference = reference
        self.chromosome = chromosome
        self.bamfile = bamfile
        self.region_filter = RegionFilter(bedfile, chromosome)
        self.min_base_quality = min_base_quality
        self.min_mapping_quality = min_base_quality
        
    def process_pileup_column(self, pileup_column, ref_base: str) -> Dict:
        """Process a pileup column and count bases"""
        base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        
        for pileup_read in pileup_column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue
                
            read = pileup_read.alignment
            base = read.query_sequence[pileup_read.query_position].upper()
            base_counts[base] += 1
                
        return base_counts
    
    def call_variants(self) -> List[VCFRecord]:
        """Main variant calling method"""
        vcf_records = []
        
        try:
            with pysam.AlignmentFile(self.bamfile, "rb") as bam, \
                pysam.FastaFile(self.reference) as ref:
                
                valid_chrom = get_valid_chromosome_name(self.bamfile, self.chromosome)
                if valid_chrom is None:
                    raise ValueError(f"Chromosome {self.chromosome} not found in BAM file")
                
                for pileup_column in bam.pileup(
                    valid_chrom,
                    min_mapping_quality=self.min_mapping_quality,
                    min_base_quality=self.min_base_quality,
                    truncate=True,  # changed
                    max_depth=1000000,
                    stepper='nofilter'  # changed
                ):
                    position = pileup_column.pos
                    
                    if not self.region_filter.in_regions(position):
                        continue
                    
                    try:
                        ref_base = ref.fetch(valid_chrom, position, position + 1).upper()
                    except Exception:
                        continue
                    
                    if not ref_base:
                        continue
                    
                    # Count bases
                    base_counts = self.process_pileup_column(pileup_column, ref_base)

                    # Get alternate bases
                    alt_bases = {base: count for base, count in base_counts.items() 
                               if base != ref_base and count > 0}
                    if not alt_bases:
                        continue
                        
                    # Sort alternate bases by count
                    alt_bases_sorted = sorted(alt_bases.items(), key=lambda x: x[1], reverse=True)
                    alt = ','.join(base for base, _ in alt_bases_sorted)
                    
                    # Create base count string
                    base_ratio = f"BaseCounts={base_counts['A']},{base_counts['C']}," \
                               f"{base_counts['G']},{base_counts['T']}"
                    
                    # Determine genotype with ref_base
                    genotype = determine_genotype(base_counts, ref_base)
                    
                    # Create VCF record
                    record = VCFRecord(
                        chrom=self.chromosome,
                        pos=position + 1,
                        ref=ref_base,
                        alt=alt,
                        info=base_ratio,
                        format=genotype
                    )
                    
                    vcf_records.append(record)
                        
        except Exception as e:
            print(f"Error processing BAM file: {str(e)}")
            raise
                
        return vcf_records
    
# class VariantCaller:
#     def __init__(self, reference: str, chromosome: str, bamfile: str, 
#                  bedfile: str, min_base_quality: int = 0, min_mapping_quality: int = 0):
#         self.reference = reference
#         self.chromosome = chromosome
#         self.bamfile = bamfile
#         self.region_filter = RegionFilter(bedfile, chromosome)
#         self.min_base_quality = min_base_quality
#         self.min_mapping_quality = min_base_quality
        
#     def process_pileup_column(self, pileup_column, ref_base: str) -> BaseCounter:
#         """Process a pileup column and count bases"""
#         counter = BaseCounter()
        
#         for pileup_read in pileup_column.pileups:
#             # Skip deletions and refskips
#             if pileup_read.is_del or pileup_read.is_refskip:
#                 continue
                
#             read = pileup_read.alignment
#             base_quality = read.query_qualities[pileup_read.query_position]
            
#             # Skip low quality bases
#             if base_quality < self.min_base_quality:
#                 continue
                
#             read_base = read.query_sequence[pileup_read.query_position].upper()
#             counter.add_base(read_base)
            
#         return counter
    
#     def call_variants(self) -> List[VCFRecord]:
#         """Main variant calling method"""
#         vcf_records = []
        
#         try:
#             with pysam.AlignmentFile(self.bamfile, "rb") as bam, \
#                 pysam.FastaFile(self.reference) as ref:
                
#                 valid_chrom = get_valid_chromosome_name(self.bamfile, self.chromosome)
#                 if valid_chrom is None:
#                     raise ValueError(f"Chromosome {self.chromosome} not found in BAM file")
                
#                 for pileup_column in bam.pileup(
#                     valid_chrom,
#                     min_mapping_quality=self.min_mapping_quality,
#                     min_base_quality=self.min_base_quality, # default is not 0...
#                     truncate=True,
#                     max_depth=1000000,
#                     stepper='nofilter'
#                 ):
#                     position = pileup_column.pos
                    
#                     # BED file filtering here
#                     # if not self.region_filter.in_regions(position):
#                     #     continue
                    
#                     # Get reference base
#                     try:
#                         ref_base = ref.fetch(valid_chrom, position, position + 1).upper()
#                     except Exception:
#                         continue
                    
#                     if not ref_base:
#                         continue
                    
#                     # Count bases at this position
#                     base_counter = self.process_pileup_column(pileup_column, ref_base)
                    
#                     # Skip positions with no coverage
#                     if base_counter.get_total() == 0:
#                         continue
                    
#                     # Create VCF record if variants exist
#                     record = self._create_vcf_record(position, ref_base, base_counter)
#                     if record is not None:  # Only add record if variants were found
#                         vcf_records.append(record)
                        
#         except Exception as e:
#             print(f"Error processing BAM file: {str(e)}")
#             raise
                
#         return vcf_records

#     def _create_vcf_record(self, position: int, ref_base: str, 
#                         base_counter: BaseCounter) -> Optional[VCFRecord]:
#         """
#         Create a VCF record with base counts, only if alternates exist.
#         Returns None if no alternate alleles are found.
#         """
#         # Get counts for all bases
#         base_counts = base_counter.counts
        
#         # Find alternate alleles (any base different from reference with > 0 counts)
#         alt_bases = []
#         for base, count in base_counts.items():
#             if base != ref_base and count > 0:
#                 alt_bases.append(base)
        
#         # If no alternate alleles found, return None
#         if not alt_bases:
#             return None
            
#         # Sort alternate bases by count (highest to lowest)
#         alt_bases.sort(key=lambda b: base_counts[b], reverse=True)
        
#         # Create alt field
#         alt = ','.join(alt_bases)

#         return VCFRecord(
#             chrom=self.chromosome,
#             pos=position + 1,  # Convert to 1-based position
#             id=".",
#             ref=ref_base,
#             alt=alt,
#             qual="20",
#             filter="PASS",
#             info=f"SVTYPE=SNV",
#             base_counts=f"{base_counts['A']}:{base_counts['C']}:{base_counts['G']}:{base_counts['T']}"
#         )

def write_vcf(records: List[VCFRecord], header_file: str, output_file: str):
    """Write VCF records to file"""
    with open(output_file, 'a') as out:
        # Copy header if output file is empty
        if Path(output_file).stat().st_size == 0:
            with open(header_file, 'r') as header:
                for line in header:
                    # Add new format line for ACGT counts before the #CHROM line
                    if line.startswith('#CHROM'):
                        out.write('##INFO=<ID=BaseCount,Number=4,Type=Integer,Description="Base counts in order A,C,G,T">\n')
                        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                    out.write(line)
        
        # Write records
        for record in records:
            out.write(str(record) + '\n')

# def write_vcf(records: List[VCFRecord], header_file: str, output_file: str):
#     """Write VCF records to file with modified format"""
#     with open(output_file, 'w') as out:
#         # Write header
#         with open(header_file, 'r') as header:
#             for line in header:
#                 if line.startswith('#CHROM'):
#                     # Add format field description before #CHROM line
#                     out.write('##INFO=<ID=BaseCount,Number=4,Type=Integer,Description="Base counts in order A,C,G,T">\n')
#                     out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
#                 out.write(line)
        
#         # Write records
#         for record in records:
#             out.write(str(record) + '\n')

def gene(standard: str, chromo: str, bamfile: str, bedfile: str, 
         header: str, out: str):
    """
    Main function to process BAM file and call variants.
    
    Args:
        standard: Path to reference sequence
        chromo: Chromosome name
        bamfile: Path to BAM file
        bedfile: Path to BED file
        header: Path to VCF header file
        out: Path to output VCF file
    """
    try:
        # Initialize variant caller
        caller = EnhancedVariantCaller(
            reference=standard,
            chromosome=chromo,
            bamfile=bamfile,
            bedfile=bedfile,
            min_base_quality=0,
            min_mapping_quality=0
        )
        
        # Call variants
        vcf_records = caller.call_variants()
        
        # Write to VCF
        write_vcf(vcf_records, header, out)
        
    except Exception as e:
        print(f"Error processing {bamfile} for chromosome {chromo}: {str(e)}")
        raise

def main():
    """Command line interface"""
    import argparse
    parser = argparse.ArgumentParser(description="SNV calling script")
    parser.add_argument('--reference_seq', required=True)
    parser.add_argument('--chromosome', type=str, required=True)
    parser.add_argument('--bamfile', required=True)
    parser.add_argument('--bedfile', required=True)
    parser.add_argument('--header', required=True)
    parser.add_argument('--out', required=True)
    
    args = parser.parse_args()
    
    gene(
        args.reference_seq,
        args.chromosome,
        args.bamfile,
        args.bedfile,
        args.header,
        args.out
    )

if __name__ == "__main__":
    main()
# demonstrate_usage()

