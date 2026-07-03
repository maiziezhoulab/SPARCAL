import os
import gzip
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class FileConfig:
    name: str
    path: str
    is_beagle_chr: bool = False

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "bam_pattern": "{section_id}/bam_bycell/*.bam",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True
    },
    "P4_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P4_Tumor_output/outs/split_BAM/",
        "barcode_file": "spaceranger_align_rep{section_id}/Meta_Data/GSM4565823_barcodes.tsv.gz",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P6_Tumor_output/outs/split_BAM/",
        "barcode_file": "spaceranger_align_rep{section_id}/Meta_Data/GSM4565825_barcodes.tsv.gz",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    }
}

class GenotypeCounter:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()

    def setup_paths(self):
        """Setup paths for all VCF files to analyze"""
        dataset_config = DATASET_CONFIGS.get(self.section_id.split('_')[0])
        if dataset_config:
            self.setup_paths_for_dataset(dataset_config)
        else:
            raise ValueError(f"Unknown dataset for section_id: {self.section_id}")

    def setup_paths_for_dataset(self, config):
        """Setup paths based on dataset configuration"""
        section_path = os.path.join(self.base_dir, config["output_dir"].format(section_id=self.section_id))
        
        # Base paths for different outputs
        beagle_base = os.path.join(section_path, "output_VCFs/beagle", self.quality_filter)
        seq_err_base = os.path.join(section_path, "output_VCFs/SeqErrModel", self.quality_filter)
        svm1_base = os.path.join(section_path, "output_VCFs/SVMModel", self.quality_filter, "results")
        svm2_base = os.path.join(section_path, "output_VCFs/SVM2Model", self.quality_filter)
        mpileup_base = os.path.join(section_path, "output_VCFs/mpileup_multi_bam", self.quality_filter)

        # Create output directories for SVM2 high/low confidence
        self.svm2_output_dir = os.path.join(svm2_base, "split_confidence")
        os.makedirs(self.svm2_output_dir, exist_ok=True)
        
        self.files_to_analyze = [
            # Mpileup output
            FileConfig("mpileup_output", os.path.join(mpileup_base, "merged_sorted_gt.vcf.gz")),
            
            # Beagle filtered output
            FileConfig("beagle_filtered_in", os.path.join(beagle_base, "all_filtered_in.vcf.gz")),
            FileConfig("beagle_filtered_out", os.path.join(beagle_base, "all_filtered_out.vcf.gz")),
            
            # Sequence error model outputs
            FileConfig("seq_error", os.path.join(seq_err_base, "sequence_error.vcf.gz")),
            FileConfig("seq_no_error", os.path.join(seq_err_base, "sequence_no_error.vcf.gz")),
            
            # SVM1 outputs
            FileConfig("svm1_high_conf", os.path.join(svm1_base, "high_confidence.vcf.gz")),
            FileConfig("svm1_low_conf", os.path.join(svm1_base, "low_confidence.vcf.gz")),
            
            # SVM2 outputs (will be created during processing)
            FileConfig("svm2_high_conf", os.path.join(self.svm2_output_dir, "high_confidence.vcf.gz")),
            FileConfig("svm2_low_conf", os.path.join(self.svm2_output_dir, "low_confidence.vcf.gz"))
        ]

    def split_svm2_predictions(self, input_vcf: str):
        """Split SVM2 predictions into high and low confidence files"""
        high_conf_path = os.path.join(self.svm2_output_dir, "high_confidence.vcf.gz")
        low_conf_path = os.path.join(self.svm2_output_dir, "low_confidence.vcf.gz")
        
        with gzip.open(input_vcf, 'rt') as f_in, \
             gzip.open(high_conf_path, 'wt') as f_high, \
             gzip.open(low_conf_path, 'wt')as f_low:
            
            # Process header
            for line in f_in:
                if line.startswith('#'):
                    f_high.write(line)
                    f_low.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            # Process variants
            for line in f_in:
                fields = line.strip().split('\t')
                info = fields[7]
                
                # Check SVM2_PRED value
                svm2_pred = None
                for field in info.split(';'):
                    if field.startswith('SVM2_PRED='):
                        svm2_pred = field.split('=')[1]
                        break
                
                if svm2_pred == '1':
                    f_high.write(line)
                elif svm2_pred == '0':
                    f_low.write(line)

    def count_genotypes(self, vcf_path: str) -> Dict[str, int]:
        """Count specific genotypes (0/1 and 1/1) in a VCF file"""
        counts = defaultdict(int)
        try:
            with gzip.open(vcf_path, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    fields = line.strip().split('\t')
                    format_str = fields[8]
                    sample_str = fields[9]
                    
                    # Extract GT field
                    gt_idx = format_str.split(':').index('GT')
                    gt = sample_str.split(':')[gt_idx]
                    if gt in ['0/1', '1/1']:
                        counts[gt] += 1
                        
        except Exception as e:
            print(f"Error processing {vcf_path}: {str(e)}")
            
        return counts

    def read_shifted_counts(self, shifted_counts_path: str) -> Dict[str, int]:
        """Read shifted counts from the specified file"""
        counts = {}
        try:
            with open(shifted_counts_path, 'r') as f:
                for line in f:
                    if line.startswith("Total changed genotypes:"):
                        counts['total_changed_genotypes'] = int(line.split(':')[-1].strip().replace(',', ''))
                    elif "->" in line:
                        transition, count = line.split(':')
                        counts[transition.strip()] = int(count.strip().replace(',', ''))
        except Exception as e:
            print(f"Error reading shifted counts from {shifted_counts_path}: {str(e)}")
        return counts

    def read_stable_counts(self, stable_counts_path: str) -> Dict[str, int]:
        """Read stable counts from the specified file"""
        counts = {}
        try:
            with open(stable_counts_path, 'r') as f:
                for line in f:
                    if "->" in line:
                        transition, count = line.split(':')
                        counts[transition.strip()] = int(count.strip().replace(',', ''))
        except Exception as e:
            print(f"Error reading stable counts from {stable_counts_path}: {str(e)}")
        return counts

    def read_transition_counts(self, csv_path: str) -> Dict[str, int]:
        """Read transition counts from a CSV file"""
        counts = defaultdict(int)
        try:
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    transition_format = row['transition_format']
                    count = int(row['count'])
                    counts[transition_format] += count
        except Exception as e:
            print(f"Error reading transition counts from {csv_path}: {str(e)}")
        return counts

    def process_all_files(self) -> Dict[str, Dict[str, int]]:
        """Process all VCF files and collect genotype counts"""
        results = {}
        
        # First, split SVM2 predictions if they exist
        svm2_input = os.path.join(os.path.dirname(self.svm2_output_dir), "svm2_predictions.vcf.gz")
        if os.path.exists(svm2_input):
            print("Splitting SVM2 predictions into high/low confidence files...")
            self.split_svm2_predictions(svm2_input)
        
        for file_config in self.files_to_analyze:
            if os.path.exists(file_config.path):
                counts = self.count_genotypes(file_config.path)
                results[file_config.name] = dict(counts)
        
        # Read shifted counts
        shifted_counts_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id, "metrics/beagle", self.quality_filter, "DLPFC_151507_shifted_counts.txt")
        shifted_counts = self.read_shifted_counts(shifted_counts_path)
        results['shifted_counts'] = shifted_counts
        
        # Read stable counts
        stable_counts_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id, "metrics/beagle", self.quality_filter, "DLPFC_151507_stable_counts.txt")
        stable_counts = self.read_stable_counts(stable_counts_path)
        results['stable_counts'] = stable_counts

        # Read transition counts from CSV files
        shifted_transition_csv = os.path.join(self.base_dir, "data/dlpfc", self.section_id, "metrics/beagle", self.quality_filter, "shifted_transition_counts.csv")
        stable_transition_csv = os.path.join(self.base_dir, "data/dlpfc", self.section_id, "metrics/beagle", self.quality_filter, "stable_transition_counts.csv")
        
        shifted_transition_counts = self.read_transition_counts(shifted_transition_csv)
        stable_transition_counts = self.read_transition_counts(stable_transition_csv)
        
        results['shifted_transition_counts'] = shifted_transition_counts
        results['stable_transition_counts'] = stable_transition_counts
        
        return results

    def save_to_csv(self, results: Dict[str, Dict[str, int]], csv_path: str):
        """Save results to CSV file, appending if file exists"""
        # Prepare header row with all possible genotypes
        all_genotypes = set()
        for counts in results.values():
            all_genotypes.update(counts.keys())
        genotypes = sorted(all_genotypes)
        
        # Prepare rows for writing
        rows = []
        svm1_positive_set_counts = {'0/1': 0, '1/1': 0}
        for file_name, counts in results.items():
            row = {
                'section_id': self.section_id,
                'quality_filter': self.quality_filter,
                'file': file_name
            }
            if file_name in ['shifted_counts', 'stable_counts']:
                row.update(counts)
            elif file_name in ['shifted_transition_counts', 'stable_transition_counts']:
                for transition_format in ['1/1->1/1', '0/1->1/1', '1/1->0/1', '0/1->0/1', '0/1->0/0', '1/1->0/0']:
                    row[transition_format] = counts.get(transition_format, 0)
                row['beagle_predicted_variance'] = (
                    counts.get('1/1->1/1', 0) + counts.get('0/1->1/1', 0) +
                    counts.get('1/1->0/1', 0) + counts.get('0/1->0/1', 0)
                )
                row['beagle_predicted_no_variance'] = (
                    counts.get('0/1->0/0', 0) + counts.get('1/1->0/0', 0)
                )
                svm1_positive_set_counts['0/1'] += counts.get('0/1->1/1', 0) + counts.get('0/1->0/1', 0)
                svm1_positive_set_counts['1/1'] += counts.get('1/1->1/1', 0) + counts.get('1/1->0/1', 0)
            else:
                row['0/1_count'] = counts.get('0/1', 0)
                row['1/1_count'] = counts.get('1/1', 0)
            rows.append(row)
        
        # Calculate final_kept counts
        final_kept_counts = {
            '0/1': svm1_positive_set_counts['0/1'] + results['svm2_high_conf'].get('0/1', 0) + results['svm2_high_conf'].get('1/1', 0),
            '1/1': svm1_positive_set_counts['1/1'] + results['svm2_low_conf'].get('0/1', 0) + results['svm2_low_conf'].get('1/1', 0)
        }
        final_kept_counts['total'] = final_kept_counts['0/1'] + final_kept_counts['1/1']
        
        # Add final_kept row
        final_kept_row = {
            'section_id': self.section_id,
            'quality_filter': self.quality_filter,
            'file': 'final_kept',
            '0/1_count': final_kept_counts['0/1'],
            '1/1_count': final_kept_counts['1/1'],
            'total': final_kept_counts['total']
        }
        rows.append(final_kept_row)
        
        # Define fieldnames
        fieldnames = ['section_id', 'quality_filter', 'file', '0/1_count', '1/1_count', 'total_changed_genotypes',
                      '1/1->1/1', '0/1->1/1', '1/1->0/1', '0/1->0/1', '0/1->0/0', '1/1->0/0',
                      'beagle_predicted_variance', 'beagle_predicted_no_variance', 'total']
        
        # Check if file exists and has same headers
        file_exists = os.path.exists(csv_path)
        if file_exists:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames
                if existing_fields != fieldnames:
                    print(f"Warning: CSV format mismatch. Creating new file.")
                    file_exists = False
        
        # Write to CSV
        mode = 'a' if file_exists else 'w'
        with open(csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def print_results(self, results: Dict[str, Dict[str, int]]):
        """Print results in a formatted table"""
        print("\nGenotype Counts Summary")
        print("=" * 80)
        
        beagle_predicted_variance = 0
        beagle_predicted_no_variance = 0
        seq_error_count = 0
        svm1_positive_set_counts = {'0/1': 0, '1/1': 0}
        
        for file_name, counts in results.items():
            print(f"\n{file_name}:")
            print("-" * 40)
            total = sum(counts.values())
            
            for gt, count in sorted(counts.items()):
                percentage = (count / total * 100) if total > 0 else 0
                print(f"{gt:<10} {count:>10,} ({percentage:>6.2f}%)")
            print(f"{'Total':<10} {total:>10,}")
            
            if file_name == 'shifted_transition_counts' or file_name == 'stable_transition_counts':
                beagle_predicted_variance += (
                    counts.get('1/1->1/1', 0) + counts.get('0/1->1/1', 0) +
                    counts.get('1/1->0/1', 0) + counts.get('0/1->0/1', 0)
                )
                beagle_predicted_no_variance += (
                    counts.get('0/1->0/0', 0) + counts.get('1/1->0/0', 0)
                )
                svm1_positive_set_counts['0/1'] += counts.get('0/1->1/1', 0) + counts.get('0/1->0/1', 0)
                svm1_positive_set_counts['1/1'] += counts.get('1/1->1/1', 0) + counts.get('1/1->0/1', 0)
            if file_name == 'seq_error':
                seq_error_count = counts.get('0/1', 0) + counts.get('1/1', 0)
        
        print(f"\nbeagle_predicted_variance: {beagle_predicted_variance}")
        print(f"beagle_predicted_no_variance: {beagle_predicted_no_variance}")
        
        svm1_positive_set = beagle_predicted_variance
        svm1_negative_set = beagle_predicted_no_variance + seq_error_count
        
        print(f"\nSVM1 positive set: {svm1_positive_set}")
        print(f"SVM1 negative set: {svm1_negative_set}")
        
        final_kept_counts = {
            '0/1': svm1_positive_set_counts['0/1'] + results['svm2_high_conf'].get('0/1', 0) + results['svm2_high_conf'].get('1/1', 0),
            '1/1': svm1_positive_set_counts['1/1'] + results['svm2_low_conf'].get('0/1', 0) + results['svm2_low_conf'].get('1/1', 0)
        }
        final_kept_counts['total'] = final_kept_counts['0/1'] + final_kept_counts['1/1']
        
        print(f"\nFinal kept:")
        print(f"0/1: {final_kept_counts['0/1']}")
        print(f"1/1: {final_kept_counts['1/1']}")
        print(f"Total: {final_kept_counts['total']}")

def main():
    parser = argparse.ArgumentParser(description="Count genotypes in VCF files")
    parser.add_argument("--section_id", required=True, help="Section ID")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0", 
                      help="Quality filter (default: baseQ0mapQ0)")
    parser.add_argument("--output-csv", default="genotype_counts.csv",
                      help="Output CSV file (default: genotype_counts.csv)")
    args = parser.parse_args()
    
    counter = GenotypeCounter(args.section_id, args.quality_filter)
    results = counter.process_all_files()
    
    # Print results to console
    counter.print_results(results)
    
    # Save results to CSV
    counter.save_to_csv(results, args.output_csv)
    print(f"\nResults appended to: {args.output_csv}")

if __name__ == "__main__":
    main()

# example for getting dlpfc 151507:
# python scripts/tools/get_vcf_count.py --section_id 151507 --quality-filter baseQ13mapQ20

# example for getting p4_tumor 1:
# python scripts/tools/get_vcf_count.py --section_id P4_TUMOR_1 --quality-filter baseQ13mapQ20