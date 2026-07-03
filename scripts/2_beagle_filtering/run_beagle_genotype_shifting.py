import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import pickle
import subprocess
import csv
from pathlib import Path

REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # 1, 2, 3, ..., 22
    },
    "CHR_PREFIX": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    }
}

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC"
    },
    "OVAR_P5": {
        # GRCh38, chr prefix — merged VCF is chr-prefixed, so use CHR_PREFIX.
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "CHR_PREFIX"
    }
}


@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str

class BaseGenotypeAnalyzer:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.base_dir = "/data/maiziezhou_lab/leiy4/snv_calling"
        self.validate_dataset_config()
        self.setup_environment()
        
        # Analysis bins
        self.baf_bins = np.arange(0, 1.05, 0.05)  # BAF bins from 0 to 1 in steps of 0.05
        self.depth_bins = np.arange(0, 210, 10)   # Depth bins from 0 to 200 in steps of 10
        self.metrics_by_transition = defaultdict(list)
        self.total_variants = 0  # This will be incremented in _process_original_variants
        
        # Track all transitions for comprehensive statistics
        self.all_transitions = Counter() 
        self.valid_genotypes = ['0/0', '0/1', '1/1']

    def validate_dataset_config(self):
        """Validate dataset configuration and section ID if required."""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        if dataset_config["has_sections"]:
            if not self.section_id:
                raise ValueError(f"Dataset {self.dataset_name} requires a section_id")
            if "section_ids" in dataset_config:
                if self.section_id not in dataset_config["section_ids"]:
                    raise ValueError(f"Invalid section_id for {self.dataset_name}. "
                                  f"Valid section IDs are: {dataset_config['section_ids']}")

    def setup_environment(self):
        """Setup bcftools/tabix environment"""
        apps_dir = "/data/maiziezhou_lab/leiy4/snv_calling/apps"
        os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{apps_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # Setup paths based on dataset configuration
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        if dataset_config["has_sections"]:
            self.output_base = os.path.join(
                self.base_dir,
                dataset_config["output_dir"].format(section_id=self.section_id)
            )
        else:
            self.output_base = os.path.join(
                self.base_dir,
                dataset_config["output_dir"]
            )
        
        # Setup directory structure
        self.orig_vcf_dir = os.path.join(self.output_base, "output_VCFs/mpileup_multi_bam", 
                                       self.quality_filter)
        self.beagle_vcf_dir = os.path.join(self.output_base, "output_VCFs/beagle", 
                                         self.quality_filter)
        self.output_dir = os.path.join(self.output_base, "metrics/beagle", 
                                    self.quality_filter)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_transition_key(self, orig_gt: str, new_gt: str, ref: str, alt: str) -> str:
        return f"{orig_gt}->{new_gt}_{ref}_{alt}"
    
    def get_simple_transition_key(self, orig_gt: str, new_gt: str) -> str:
        """Return a simple transition key without ref/alt for counting purposes"""
        return f"{orig_gt}->{new_gt}"
    
    def get_display_name(self, key: str) -> str:
        gt_part, ref, alt = key.split('_')
        orig_gt, new_gt = gt_part.split('->')
        return f"{orig_gt}({ref}{alt})->{new_gt}({ref}{alt})"
    
    def analyze(self, chromosome: str):
        """Analyze genotype transitions for a chromosome"""
        orig_vcf = os.path.join(self.orig_vcf_dir, "merged_sorted_gt.vcf.gz")
        beagle_vcf = os.path.join(self.beagle_vcf_dir, f"{chromosome}.vcf.gz")

        print(f"Analyzing {self.__class__.__name__} for {chromosome}")
        try:
            with gzip.open(beagle_vcf, 'rt') as beagle:
                beagle_variants = self._load_beagle_variants(beagle, chromosome)
                # print total counts of beagle variants
                print(f"Total beagle variants: {len(beagle_variants)}")
            
            with gzip.open(orig_vcf, 'rt') as orig:
                self._process_original_variants(orig, beagle_variants, chromosome)
                
        except Exception as e:
            print(f"Error processing {chromosome}: {str(e)}")
            raise

    def extract_format_field(self, format_str: str, value_str: str, field_name: str) -> Optional[str]:
        """Extract a specific field from VCF FORMAT column with dataset-specific handling."""
        try:
            format_fields = format_str.split(':')
            if field_name not in format_fields:
                return None
            
            field_idx = format_fields.index(field_name)
            value_fields = value_str.split(':')
            
            if field_idx >= len(value_fields):
                return None
                
            return value_fields[field_idx]
        except (ValueError, IndexError):
            return None

    def extract_info_field(self, info_str: str, field_name: str) -> Optional[str]:
        """Extract a specific field from VCF INFO column with dataset-specific handling."""
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None
    
    def extract_metrics(self, info_str: str, format_str: str, value_str: str) -> Tuple[Optional[float], Optional[int]]:
        """Extract BAF and depth with dataset-specific handling"""
        # Handle different VCF formats based on dataset
        if self.dataset_name in ["DLPFC", "10X_BC_6.5MM", "10X_BC_FFPE", "DCIS"]:
            # Extract BAF from FORMAT column
            baf_str = self.extract_format_field(format_str, value_str, "BAF")
            baf = float(baf_str) if baf_str is not None else None
            
            # Extract depth from INFO column
            depth_str = self.extract_info_field(info_str, "DP")
            depth = int(depth_str) if depth_str is not None else None
            
        elif self.dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
            # These datasets might have different field names or locations
            baf_str = self.extract_format_field(format_str, value_str, "BAF")  # Using AF instead of BAF
            baf = float(baf_str) if baf_str is not None else None
            
            depth_str = self.extract_info_field(info_str, "DP")
            depth = int(depth_str) if depth_str is not None else None
            
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_name}")
            
        return baf, depth
    
    def extract_genotype(self, format_str: str, value_str: str) -> Optional[str]:
        """Extract GT field from VCF FORMAT/VALUE strings"""
        try:
            gt_idx = format_str.split(':').index('GT')
            return value_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None

    def print_summary(self):
        """Print summary statistics of transitions"""
        print(f"\n{self.__class__.__name__} Summary:")
        print("-" * 50)
        print(f"Dataset: {self.dataset_name}")
        if self.section_id:
            print(f"Section ID: {self.section_id}")
        print(f"Total variants processed: {self.total_variants:,}")
        
        # Print comprehensive transition statistics
        print("\nAll Transitions:")
        print("-" * 20)
        
        # Sort transitions for better readability
        sorted_transitions = sorted(self.all_transitions.items(), 
                                   key=lambda x: (x[0].split('->')[0], x[0].split('->')[1]))
        
        total_count = sum(self.all_transitions.values())
        for transition, count in sorted_transitions:
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f"{transition}: {count:,} ({percentage:.2f}%)")
        
        print("\nTransition counts:")
        print("-" * 20)
        
        total_changed = 0
        for key, metrics in sorted(self.metrics_by_transition.items()):
            count = len(metrics)
            total_changed += count
            display_name = self.get_display_name(key)
            print(f"{display_name}: {count:,}")
        
        if isinstance(self, ShiftedGenotypeAnalyzer):
            print(f"\nTotal changed genotypes: {total_changed:,}")
            print(f"Percentage of total variants: {(total_changed/self.total_variants)*100:.2f}%")
        elif isinstance(self, StableGenotypeAnalyzer):
            print(f"\nTotal stable genotypes: {total_changed:,}")
            print(f"Percentage of total variants: {(total_changed/self.total_variants)*100:.2f}%")
    
    def save_transition_counts(self, prefix: str):
        """Save all transition counts to CSV files"""
        # Save the comprehensive transition counts
        transition_counts_path = os.path.join(self.output_dir, f"{prefix}_transition_counts.csv")
        
        with open(transition_counts_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['transition', 'count', 'percentage'])
            
            total_count = sum(self.all_transitions.values())
            for transition, count in sorted(self.all_transitions.items()):
                percentage = (count / total_count * 100) if total_count > 0 else 0
                writer.writerow([transition, count, f"{percentage:.2f}"])
        
        # Also save a format more suitable for detailed analysis
        detailed_counts_path = os.path.join(self.output_dir, f"{prefix}_detailed_counts.csv")
        
        with open(detailed_counts_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['transition_format', 'count', 'orig_gt', 'new_gt'])
            writer.writeheader()
            
            for transition, count in sorted(self.all_transitions.items()):
                orig_gt, new_gt = transition.split('->')
                writer.writerow({
                    'transition_format': transition,
                    'count': count,
                    'orig_gt': orig_gt,
                    'new_gt': new_gt
                })
        
        # Save simplified text summary for backward compatibility
        summary_path = os.path.join(self.output_dir, f"{prefix}_counts.txt")
        
        with open(summary_path, 'w') as f:
            if isinstance(self, ShiftedGenotypeAnalyzer):
                f.write(f"Total changed genotypes: {sum(self.all_transitions.values()):,}\n")
            
            for transition, count in sorted(self.all_transitions.items()):
                f.write(f"{transition}: {count:,}\n")
                
        print(f"Saved transition counts to {transition_counts_path}")
        print(f"Saved detailed counts to {detailed_counts_path}")
        print(f"Saved text summary to {summary_path}")

    def _load_beagle_variants(self, beagle_file, chromosome: str) -> Dict:
        """Load variants from Beagle VCF file with correct field extraction."""
        beagle_variants = {}
        
        # Skip header lines
        for line in beagle_file:
            if not line.startswith('#'):
                break
                
        for line in beagle_file:
            fields = line.strip().split('\t')
            # Handle chromosome format differences
            if fields[0] != chromosome:
                print(f"Warning: Chromosome mismatch in Beagle file: {fields[0]} vs {chromosome}")
                continue
                
            try:
                pos = int(fields[1])
                ref, alt = fields[3], fields[4]
                gt = self.extract_genotype(fields[8], fields[9])
                baf, depth = self.extract_metrics(fields[7], fields[8], fields[9])
                
                # print(f"Processing Beagle variant at {chromosome}:{pos} with GT: {gt}, BAF: {baf}, Depth: {depth}")
                if all(x is not None for x in [gt, baf, depth]):
                    beagle_variants[pos] = (ref, alt, gt, baf, depth)
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Error processing line: {line.strip()}")
                print(f"Error details: {str(e)}")
                continue
        
        return beagle_variants
            
    def _process_original_variants(self, orig_file, beagle_variants: Dict, chromosome: str):
        """Process original variants with proper format handling"""
        header_lines = []
        total_variants_processed = 0
        
        # Read header lines first
        for line in orig_file:
            if line.startswith('#'):
                header_lines.append(line)
                if line.startswith('#CHROM'):
                    break
        
        # Process variant lines
        for line in orig_file:
            total_variants_processed += 1
            fields = line.strip().split('\t')
            
            # Handle chromosome format differences
            if fields[0] != chromosome:
                continue

            pos = int(fields[1])
            if pos not in beagle_variants:
                continue

            ref, alt = fields[3], fields[4]
            beagle_ref, beagle_alt, beagle_gt, baf, depth = beagle_variants[pos]

            if ref != beagle_ref or alt != beagle_alt:
                continue

            orig_gt = self.extract_genotype(fields[8], fields[9])
            if orig_gt is None:
                continue

            variant_data = {
                'metrics': GenotypeMetrics(baf, depth, ref, alt),
                'header_lines': header_lines,
                'line': line,
                'orig_gt': orig_gt,
                'beagle_gt': beagle_gt
            }

            # Track all transitions
            if orig_gt in self.valid_genotypes and beagle_gt in self.valid_genotypes:
                self.all_transitions[self.get_simple_transition_key(orig_gt, beagle_gt)] += 1

            if isinstance(self, ShiftedGenotypeAnalyzer):
                if (orig_gt, beagle_gt) in self.target_transitions:
                    key = self.get_transition_key(orig_gt, beagle_gt, ref, alt)
                    self.metrics_by_transition[key].append(variant_data)
            elif isinstance(self, StableGenotypeAnalyzer):
                if orig_gt in self.target_genotypes and orig_gt == beagle_gt:
                    key = self.get_transition_key(orig_gt, orig_gt, ref, alt)
                    self.metrics_by_transition[key].append(variant_data)
        
        # Update total variants counter
        self.total_variants += total_variants_processed
        print(f"Processed {total_variants_processed} variants from {chromosome}")

    def plot_metrics(self, output_dir: str, title_prefix: str):
        """Generate all plots for the transitions"""
        # Group transitions by genotype change
        transitions_by_change = defaultdict(list)
        for key in self.metrics_by_transition.keys():
            gt_change = key.split('_')[0]  # Get "orig->new" part
            transitions_by_change[gt_change].append(key)
            
        for gt_change, transition_keys in transitions_by_change.items():
            if not transition_keys:
                continue

            # Calculate number of rows needed (1 transition per row, 3 plots per transition)
            n_transitions = len(transition_keys)
            
            fig = plt.figure(figsize=(20, 6 * n_transitions))
            fig.suptitle(f'{title_prefix}: {gt_change}', fontsize=16)
            
            for idx, transition_key in enumerate(transition_keys):
                metrics_dicts = self.metrics_by_transition[transition_key]
                # print(f"metrics is like: {metrics}")
                if not metrics_dicts:
                    continue
                    
                if idx == 0:
                    print(f"metrics_dicts is like: {metrics_dicts[idx]['metrics']}")
                display_name = self.get_display_name(transition_key)
                bafs = [m['metrics'].baf for m in metrics_dicts]
                depths = [min(m['metrics'].depth, 200) for m in metrics_dicts]
                
                # Create three subplots for this transition
                base_idx = idx * 3
                
                # 1. BAF Distribution
                ax1 = plt.subplot(n_transitions, 3, base_idx + 1)
                ax1.hist(bafs, bins=self.baf_bins, density=True, alpha=0.7)
                ax1.set_title(f'{display_name}\nBAF Distribution (N={len(metrics_dicts)})')
                ax1.set_xlabel('BAF Value')
                ax1.set_ylabel('Density')
                ax1.axvline(np.median(bafs), color='red', linestyle='--')
                
                # 2. Depth Distribution (linear scale with capped values)
                ax2 = plt.subplot(n_transitions, 3, base_idx + 2)
                ax2.hist(depths, bins=self.depth_bins, density=True, alpha=0.7)
                ax2.set_title(f'{display_name}\nRead Depth Distribution (capped at 200)')
                ax2.set_xlabel('Read Depth')
                ax2.set_ylabel('Density')
                ax2.axvline(np.median(depths), color='red', linestyle='--')
                
                # 3. BAF vs Depth Scatter (linear scale with capped values)
                ax3 = plt.subplot(n_transitions, 3, base_idx + 3)
                ax3.scatter(bafs, depths, alpha=0.5, s=30)
                ax3.set_title(f'{display_name}\nBAF vs Read Depth')
                ax3.set_xlabel('BAF Value')
                ax3.set_ylabel('Read Depth (capped at 200)')
                
                # Add summary statistics to the scatter plot
                stats_text = f'Median BAF: {np.median(bafs):.3f}\n'
                stats_text += f'Median Depth: {np.median(depths):.1f}'
                ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
                        fontsize=8, verticalalignment='top')
            
            plt.tight_layout()
            safe_change = gt_change.replace('/', '_').replace('->', '_to_')
            plt.savefig(os.path.join(output_dir, f'{safe_change}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

class ShiftedGenotypeAnalyzer(BaseGenotypeAnalyzer):
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        super().__init__(dataset_name, quality_filter, section_id)
        # Keep the same target transitions for backwards compatibility
        self.target_transitions = [
            ('0/0', '0/1'),
            ('0/1', '0/0'),
            ('1/1', '0/0'),
            ('1/1', '0/1'),
            ('0/1', '1/1'),
        ]

class StableGenotypeAnalyzer(BaseGenotypeAnalyzer):
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        super().__init__(dataset_name, quality_filter, section_id)
        self.target_genotypes = ['0/0', '0/1', '1/1']

def main():
    parser = argparse.ArgumentParser(description="Analyze genotype shifts in Beagle processing")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    parser.add_argument("--chromosomes", nargs="+", default=None,
                      help="List of chromosomes to process (e.g., chr1 chr2 chr3)")
    args = parser.parse_args()
    
    # Get chromosome format from dataset config
    dataset_config = DATASET_CONFIGS[args.dataset]
    reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
    chromosomes = args.chromosomes if args.chromosomes else reference_config['regions']
    
    # Initialize analyzers
    shifted_analyzer = ShiftedGenotypeAnalyzer(args.dataset, args.quality_filter, args.section_id)
    stable_analyzer = StableGenotypeAnalyzer(args.dataset, args.quality_filter, args.section_id)

    for chrom in chromosomes:
        print(f"\nProcessing {chrom}...")
        
        # Process both shifted and stable genotypes
        shifted_analyzer.analyze(chrom)
        stable_analyzer.analyze(chrom)

        # Save intermediate results after each chromosome
        shifted_file = os.path.join(shifted_analyzer.output_dir, f'shifted_{chrom}_results.pkl')
        stable_file = os.path.join(stable_analyzer.output_dir, f'stable_{chrom}_results.pkl')
        
        # Save shifted results
        with open(shifted_file, 'wb') as f:
            pickle.dump({
                'metrics_by_transition': dict(shifted_analyzer.metrics_by_transition),
                'total_variants': shifted_analyzer.total_variants,
                'all_transitions': dict(shifted_analyzer.all_transitions)
            }, f)
        
        # Save stable results
        with open(stable_file, 'wb') as f:
            pickle.dump({
                'metrics_by_transition': dict(stable_analyzer.metrics_by_transition),
                'total_variants': stable_analyzer.total_variants,
                'all_transitions': dict(stable_analyzer.all_transitions)
            }, f)

    # Print summaries
    print("\n=== Final Statistics ===")
    shifted_analyzer.print_summary()
    stable_analyzer.print_summary()

    # Generate transition plots
    print("\nGenerating plots...")
    shifted_plot_dir = os.path.join(shifted_analyzer.output_dir, 'plots')
    stable_plot_dir = os.path.join(stable_analyzer.output_dir, 'plots')
    os.makedirs(shifted_plot_dir, exist_ok=True)
    os.makedirs(stable_plot_dir, exist_ok=True)

    # Save the transition counts to CSV files
    result_prefix = args.dataset
    if args.section_id:
        result_prefix = f"{args.dataset}_{args.section_id}"
    
    shifted_analyzer.save_transition_counts(f"{result_prefix}_shifted")
    stable_analyzer.save_transition_counts(f"{result_prefix}_stable")

    # Generate plots
    shifted_analyzer.plot_metrics(shifted_plot_dir, "Shifted Genotypes")
    stable_analyzer.plot_metrics(stable_plot_dir, "Stable Genotypes")

    # Save final results
    final_results = {
        'shifted': {
            'metrics_by_transition': dict(shifted_analyzer.metrics_by_transition),
            'total_variants': shifted_analyzer.total_variants,
            'all_transitions': dict(shifted_analyzer.all_transitions)
        },
        'stable': {
            'metrics_by_transition': dict(stable_analyzer.metrics_by_transition),
            'total_variants': stable_analyzer.total_variants,
            'all_transitions': dict(stable_analyzer.all_transitions)
        }
    }

    # Save results with dataset-specific naming
    shifted_file = os.path.join(shifted_analyzer.output_dir, f'{result_prefix}_shifted_results.pkl')
    stable_file = os.path.join(stable_analyzer.output_dir, f'{result_prefix}_stable_results.pkl')

    with open(shifted_file, 'wb') as f:
        pickle.dump(final_results['shifted'], f)
    
    with open(stable_file, 'wb') as f:
        pickle.dump(final_results['stable'], f)

    print(f"\nAnalysis complete!")
    print(f"Results saved to:")
    print(f"  Shifted results: {shifted_file}")
    print(f"  Stable results: {stable_file}")
    print(f"  Plots directory: {shifted_plot_dir}")
    print(f"  Transition counts: {shifted_analyzer.output_dir}/{result_prefix}_shifted_detailed_counts.csv")

if __name__ == "__main__":
    main()

# Run P4_tumor section 1 baseQ0mapQ0
# python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ0mapQ0

# Run P6_tumor section 1 baseQ0mapQ0
# python scripts/postprocess/run_beagle_genotype_shifting.py --dataset P6_TUMOR --section_id 2 --quality_filter baseQ0mapQ0