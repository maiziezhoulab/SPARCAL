import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse

class DiscrepancyVisualizer:
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
    
    def load_comparison_results(self, results_file):
        """Load comparison results between mpileup and Strelka2"""
        try:
            df = pd.read_csv(results_file)
            print(f"Loaded comparison results for {len(df)} barcodes")
            return df
        except Exception as e:
            print(f"Error loading comparison results: {e}")
            return pd.DataFrame()
    
    def load_positions_csv(self, csv_file):
        """Load barcodes and positions from CSV file"""
        try:
            df = pd.read_csv(csv_file, header=None)
            # Create dictionary with barcode as key and coordinates as value
            barcodes_with_pos = {}
            for _, row in df.iterrows():
                barcode = row[0]
                array_x = float(row[4])
                array_y = float(row[5])
                barcodes_with_pos[barcode] = (array_x, array_y)
            
            print(f"Loaded {len(barcodes_with_pos)} barcodes with positions")
            return barcodes_with_pos
        except Exception as e:
            print(f"Error loading positions CSV: {e}")
            return {}
    
    def load_scale_factors(self, json_file):
        """Load scale factors from JSON file"""
        try:
            with open(json_file, 'r') as f:
                scale_factors = json.load(f)
            return scale_factors
        except Exception as e:
            print(f"Error loading scale factors: {e}")
            return None
    
    def calculate_f1_scores(self, comparison_df):
        """Calculate F1 scores for each barcode"""
        # Calculate precision and recall
        comparison_df['Precision'] = comparison_df['Common_Variants'] / comparison_df['Strelka_Variants'].replace(0, np.nan)
        comparison_df['Recall'] = comparison_df['Common_Variants'] / comparison_df['Mpileup_Variants'].replace(0, np.nan)
        
        # Calculate F1 score
        comparison_df['F1_Score'] = 2 * (comparison_df['Precision'] * comparison_df['Recall']) / (comparison_df['Precision'] + comparison_df['Recall']).replace(0, np.nan)
        
        return comparison_df
    
    def create_visualization(self, img_file, comparison_df, positions_dict, scale_factors, 
                           output_file, threshold=0.5, metric='F1_Score', colormap='coolwarm'):
        """
        Create visualization highlighting barcodes with low F1 scores on the tissue image
        
        Args:
            img_file: Path to high-resolution tissue image
            comparison_df: DataFrame with comparison metrics
            positions_dict: Dictionary mapping barcodes to positions
            scale_factors: Scale factors from JSON file
            output_file: Output file path
            threshold: Threshold for highlighting low scores (spots below this are highlighted)
            metric: Metric to use for coloring (F1_Score, Precision, Recall)
            colormap: Matplotlib colormap to use
        """
        try:
            # Load the image
            img = plt.imread(img_file)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(16, 16))
            
            # Display the image
            ax.imshow(img, origin='upper')
            
            # Get scale factor
            scale = scale_factors['tissue_hires_scalef']
            
            # Filter barcodes that exist in both the comparison results and positions dict
            valid_barcodes = set(comparison_df['Barcode']).intersection(set(positions_dict.keys()))
            
            # Extract coordinates and scores for valid barcodes
            x_coords = []
            y_coords = []
            scores = []
            
            for barcode in valid_barcodes:
                if barcode in positions_dict:
                    # Get score from comparison DataFrame
                    score_row = comparison_df[comparison_df['Barcode'] == barcode][metric]
                    if len(score_row) > 0 and not pd.isna(score_row.iloc[0]):
                        score = score_row.iloc[0]
                        
                        # Get coordinates
                        x, y = positions_dict[barcode]
                        # Scale coordinates
                        x_scaled = y * scale
                        y_scaled = x * scale
                        
                        x_coords.append(x_scaled)
                        y_coords.append(y_scaled)
                        scores.append(score)
            
            # Create a scatter plot with color based on score
            scatter = ax.scatter(x_coords, y_coords, 
                               c=scores, 
                               cmap=colormap,
                               s=50,  # Size of markers
                               alpha=0.8,  # Transparency
                               vmin=0,
                               vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(metric)
            
            # Add title
            plt.title(f'Strelka2 vs Mpileup {metric} Scores on Tissue Image')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nVisualization saved to: {output_file}")
            
            # Also create a filtered version highlighting only spots below threshold
            fig, ax = plt.subplots(figsize=(16, 16))
            ax.imshow(img, origin='upper')
            
            # Filter spots with low scores
            low_score_indices = [i for i, score in enumerate(scores) if score < threshold]
            low_x = [x_coords[i] for i in low_score_indices]
            low_y = [y_coords[i] for i in low_score_indices]
            low_scores = [scores[i] for i in low_score_indices]
            
            # Plot low-scoring spots
            scatter = ax.scatter(low_x, low_y, 
                               c=low_scores, 
                               cmap=colormap,
                               s=50,
                               alpha=0.8,
                               vmin=0,
                               vmax=1,
                               edgecolor='black')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(metric)
            
            # Add title
            plt.title(f'Spots with {metric} < {threshold} (n={len(low_x)})')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Save filtered figure
            filtered_output = output_file.replace('.png', f'_below_{threshold}.png')
            plt.savefig(filtered_output, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Filtered visualization saved to: {filtered_output}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            raise e

    def run_analysis(self, comparison_file, positions_file, json_file, img_file, output_prefix, threshold=0.5):
        """Run the complete analysis pipeline"""
        # Load data
        comparison_df = self.load_comparison_results(comparison_file)
        if comparison_df.empty:
            print("Error: Empty comparison results")
            return
            
        positions_dict = self.load_positions_csv(positions_file)
        if not positions_dict:
            print("Error: Failed to load positions data")
            return
            
        scale_factors = self.load_scale_factors(json_file)
        if not scale_factors:
            print("Error: Failed to load scale factors")
            return
        
        # Calculate F1 scores
        comparison_df = self.calculate_f1_scores(comparison_df)
        
        # Create visualizations for different metrics
        for metric in ['F1_Score', 'Precision', 'Recall']:
            output_file = f"{output_prefix}_{metric}.png"
            self.create_visualization(
                img_file,
                comparison_df,
                positions_dict,
                scale_factors,
                output_file,
                threshold=threshold,
                metric=metric
            )
        
        # Save barcode information with scores to CSV
        output_csv = f"{output_prefix}_scores.csv"
        comparison_df.to_csv(output_csv, index=False)
        print(f"Scores saved to: {output_csv}")
        
        # Summarize results
        print("\nSummary of Comparison Metrics:")
        print(f"Mean F1 Score: {comparison_df['F1_Score'].mean():.3f}")
        print(f"Mean Precision: {comparison_df['Precision'].mean():.3f}")
        print(f"Mean Recall: {comparison_df['Recall'].mean():.3f}")
        print(f"Spots with F1 Score < {threshold}: {(comparison_df['F1_Score'] < threshold).sum()}")
        
        return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Visualize discrepancies between mpileup and Strelka2')
    parser.add_argument('--comparison-file', required=True, help='Path to comparison summary CSV')
    parser.add_argument('--positions-file', required=True, help='Path to tissue positions CSV')
    parser.add_argument('--json-file', required=True, help='Path to scale factors JSON')
    parser.add_argument('--img-file', required=True, help='Path to tissue high-res image')
    parser.add_argument('--output-prefix', default='discrepancy_viz', help='Prefix for output files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for highlighting low F1 scores')
    
    args = parser.parse_args()
    
    visualizer = DiscrepancyVisualizer()
    visualizer.run_analysis(
        args.comparison_file,
        args.positions_file,
        args.json_file,
        args.img_file,
        args.output_prefix,
        args.threshold
    )

if __name__ == "__main__":
    main()