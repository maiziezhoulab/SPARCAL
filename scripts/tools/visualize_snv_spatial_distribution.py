#!/usr/bin/env python3
"""
SNV Spatial Distribution Visualizer
Visualizes presence/absence of SNVs across spatial spots
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Dataset configurations
DATASET_CONFIGS = {
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_positions_list.csv",
            "2": "GSM4565824_P4_rep2_tissue_positions_list.csv"
        },
        "scale_file_patterns": {
            "1": "GSM4565823_P4_rep1_scalefactors_json.json",
            "2": "GSM4565824_P4_rep2_scalefactors_json.json"
        },
        "image_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_hires_image.png",
            "2": "GSM4565824_P4_rep2_tissue_hires_image.png"
        }
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_positions_list.csv",
            "2": "GSM4565826_P6_rep2_tissue_positions_list.csv"
        },
        "scale_file_patterns": {
            "1": "GSM4565825_P6_rep1_scalefactors_json.json",
            "2": "GSM4565826_P6_rep2_scalefactors_json.json"
        },
        "image_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_hires_image.png",
            "2": "GSM4565826_P6_rep2_tissue_hires_image.png"
        }
    }
}


class SNVSpatialVisualizer:
    def __init__(self, dataset, section_id, output_dir):
        self.dataset = dataset.upper()
        self.section_id = str(section_id)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {self.dataset} not found")
        
        self.config = DATASET_CONFIGS[self.dataset]
        self._load_spatial_data()
        
        print(f"Loaded {len(self.positions)} spatial spots")
    
    def _load_spatial_data(self):
        base_path = self.config['base_path']
        spaceranger_dir = self.config['spaceranger_dir_template'].format(section_id=self.section_id)
        spatial_dir = os.path.join(base_path, spaceranger_dir, self.config['spatial_subdir'])
        
        # Load positions
        position_file = self.config['position_file_patterns'][self.section_id]
        position_path = os.path.join(spatial_dir, position_file)
        
        self.positions = pd.read_csv(position_path, header=None,
                                     names=['barcode', 'in_tissue', 'array_row', 'array_col', 
                                           'pxl_row_in_fullres', 'pxl_col_in_fullres'])
        self.positions = self.positions[self.positions['in_tissue'] == 1].copy()
        
        # Load scale factors
        scale_file = self.config['scale_file_patterns'][self.section_id]
        scale_path = os.path.join(spatial_dir, scale_file)
        with open(scale_path, 'r') as f:
            self.scale_factors = json.load(f)
        
        # Load image
        image_file = self.config['image_file_patterns'][self.section_id]
        image_path = os.path.join(spatial_dir, image_file)
        self.tissue_image = Image.open(image_path)
        
        # Calculate scaled coordinates
        hires_scale = self.scale_factors['tissue_hires_scalef']
        spot_diameter_fullres = self.scale_factors['spot_diameter_fullres']
        
        self.positions['pxl_row_hires'] = self.positions['pxl_row_in_fullres'] * hires_scale
        self.positions['pxl_col_hires'] = self.positions['pxl_col_in_fullres'] * hires_scale
        self.spot_radius_hires = (spot_diameter_fullres * hires_scale) / 2
    
    def parse_snv_list_from_vcf(self, vcf_path):
        """Parse SNV list from overlap VCF using zcat"""
        snvs = {}
        
        cmd = f"zcat {vcf_path} | grep -v '^#'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 5:
                continue
            
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]
            
            snvs[(chrom, pos)] = {'ref': ref, 'alt': alt}
        
        return snvs
    
    def check_snv_in_barcode_vcf(self, barcode, snv_vcf_dir, chrom, pos):
        """Check if SNV exists in barcode's VCF using zcat"""
        vcf_path = os.path.join(snv_vcf_dir, f"{barcode}.vcf.gz")
        
        if not os.path.exists(vcf_path):
            return False
        
        try:
            cmd = f"zcat {vcf_path} | grep -v '^#' | awk '$1==\"{chrom}\" && $2=={pos}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except Exception:
            return False
    
    def get_snv_presence_across_barcodes(self, snv_vcf_dir, chrom, pos):
        """Get boolean presence of SNV across all barcodes"""
        presence = {}
        
        for _, spot in self.positions.iterrows():
            barcode = spot['barcode']
            is_present = self.check_snv_in_barcode_vcf(barcode, snv_vcf_dir, chrom, pos)
            presence[barcode] = is_present
        
        return presence
    
    def visualize_snv(self, chrom, pos, ref, alt, presence, output_dir):
        """Visualize SNV presence/absence"""
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(self.tissue_image)
        
        # Plot spots
        for _, spot in self.positions.iterrows():
            barcode = spot['barcode']
            x = spot['pxl_col_hires']
            y = spot['pxl_row_hires']
            
            is_present = presence.get(barcode, False)
            color = 'red' if is_present else 'lightgray'
            alpha = 0.8 if is_present else 0.3
            
            circle = Circle((x, y), self.spot_radius_hires, 
                          color=color, alpha=alpha, linewidth=0.5, edgecolor='gray')
            ax.add_patch(circle)
        
        # Title
        n_present = sum(presence.values())
        n_total = len(presence)
        title = f'{chrom}:{pos} ({ref}>{alt})\nDetected in {n_present}/{n_total} spots'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Save
        output_file = os.path.join(output_dir, f"SNV_{chrom}_{pos}_{ref}_{alt}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def visualize_all_snvs_combined(self, all_snv_presence, output_dir):
        """Visualize all SNVs combined on one plot"""
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(self.tissue_image)
        
        # Calculate total SNVs detected per spot
        spot_snv_counts = defaultdict(int)
        for presence_dict in all_snv_presence.values():
            for barcode, is_present in presence_dict.items():
                if is_present:
                    spot_snv_counts[barcode] += 1
        
        # Get max count for color normalization
        max_count = max(spot_snv_counts.values()) if spot_snv_counts else 1
        
        # Plot spots
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['lightgray', 'yellow', 'orange', 'red', 'darkred']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('snv_count', colors, N=n_bins)
        
        for _, spot in self.positions.iterrows():
            barcode = spot['barcode']
            x = spot['pxl_col_hires']
            y = spot['pxl_row_hires']
            
            count = spot_snv_counts.get(barcode, 0)
            
            if count == 0:
                color = 'lightgray'
                alpha = 0.3
            else:
                color = cmap(count / max_count)
                alpha = 0.8
            
            circle = Circle((x, y), self.spot_radius_hires, 
                          color=color, alpha=alpha, linewidth=0.5, edgecolor='gray')
            ax.add_patch(circle)
        
        # Add colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_count))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of SNVs', rotation=270, labelpad=20)
        
        # Title
        total_snvs = len(all_snv_presence)
        spots_with_snvs = len([c for c in spot_snv_counts.values() if c > 0])
        title = f'All SNVs Combined (n={total_snvs})\n{spots_with_snvs}/{len(self.positions)} spots with variants'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Save
        output_file = os.path.join(output_dir, 'all_snvs_combined.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined plot saved: {output_file}")
        return output_file
    
    def visualize_all_snvs(self, snv_list_vcf, snv_vcf_dir, output_subdir='SNV', max_snvs=None):
        """Visualize all SNVs from list"""
        # Parse SNV list
        print(f"Parsing SNV list from: {snv_list_vcf}")
        snvs = self.parse_snv_list_from_vcf(snv_list_vcf)
        print(f"Found {len(snvs)} SNVs")
        
        if len(snvs) == 0:
            print("No SNVs to visualize")
            return []
        
        # Limit if specified
        snv_items = list(snvs.items())
        if max_snvs is not None and len(snv_items) > max_snvs:
            print(f"Limiting to first {max_snvs} SNVs")
            snv_items = snv_items[:max_snvs]
        
        # Create output directory
        output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all SNV presence data for combined plot
        all_snv_presence = {}
        
        # Open count file
        count_file = os.path.join(output_dir, 'snv_counts.txt')
        with open(count_file, 'w') as f:
            f.write("chrom\tpos\tref\talt\tn_detected\tn_total\n")
            
            # Process each SNV
            output_files = []
            for i, ((chrom, pos), snv_data) in enumerate(snv_items, 1):
                print(f"Processing SNV {i}/{len(snv_items)}: {chrom}:{pos}")
                
                # Check presence across barcodes
                presence = self.get_snv_presence_across_barcodes(snv_vcf_dir, chrom, pos)
                
                # Count
                n_present = sum(presence.values())
                n_total = len(presence)
                
                # Write to count file
                f.write(f"{chrom}\t{pos}\t{snv_data['ref']}\t{snv_data['alt']}\t{n_present}\t{n_total}\n")
                
                # Store for combined plot
                all_snv_presence[(chrom, pos)] = presence
                
                # Skip if not present in any spot
                if n_present == 0:
                    print(f"  Skipped (not detected in any spot)")
                    continue
                
                print(f"  Detected in {n_present}/{n_total} spots")
                
                # Visualize
                output_file = self.visualize_snv(
                    chrom, pos, snv_data['ref'], snv_data['alt'], 
                    presence, output_dir
                )
                output_files.append(output_file)
        
        # Create combined visualization
        print("\nCreating combined visualization...")
        self.visualize_all_snvs_combined(all_snv_presence, output_dir)
        
        print(f"\nCompleted: {len(output_files)} individual plots")
        print(f"Counts saved: {count_file}")
        print(f"Output: {output_dir}")
        
        return output_files


def main():
    parser = argparse.ArgumentParser(description='Visualize SNV spatial distribution')
    
    parser.add_argument('--dataset', required=True, choices=['P4_TUMOR', 'P6_TUMOR'])
    parser.add_argument('--section_id', required=True, choices=['1', '2'])
    parser.add_argument('--snv_list_vcf', required=True, help='VCF with SNV list (overlap VCF)')
    parser.add_argument('--snv_vcf_dir', required=True, help='Directory with per-barcode VCFs')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_snvs', type=int, default=None)
    
    args = parser.parse_args()
    
    print("="*80)
    print("SNV Spatial Distribution Visualizer")
    print("="*80)
    
    visualizer = SNVSpatialVisualizer(
        dataset=args.dataset,
        section_id=args.section_id,
        output_dir=args.output_dir
    )
    
    visualizer.visualize_all_snvs(
        snv_list_vcf=args.snv_list_vcf,
        snv_vcf_dir=args.snv_vcf_dir,
        max_snvs=args.max_snvs
    )
    
    print("="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    main()