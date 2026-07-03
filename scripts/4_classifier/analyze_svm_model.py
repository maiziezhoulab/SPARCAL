#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA

# Import from run_svm_hetero_finding.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_svm_hetero_finding import SVMWithPCA, DATASET_CONFIGS

def load_model(model_path):
    """Load a trained model from disk"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize an existing SVM-PCA model")
    parser.add_argument("--model-path", required=True, 
                      help="Path to the saved SVM-PCA model pickle file")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset name")
    parser.add_argument("--section-id",
                      help="Section ID (required for some datasets)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter used")
    parser.add_argument("--sample-size", type=int, default=500,
                      help="Number of samples to use for PCA projection (default: 500)")
    parser.add_argument("--output-dir",
                      help="Output directory for plots (defaults to model directory)")
    parser.add_argument("--skip-pca-projection", action="store_true",
                      help="Skip PCA projection visualization")
    parser.add_argument("--skip-confidence", action="store_true",
                      help="Skip confidence distribution analysis")
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
    
    print(f"Loading model from {args.model_path}...")
    try:
        model_data = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    print("Model loaded successfully.")
    
    # Initialize SVMWithPCA to use its methods
    svm_pca = SVMWithPCA(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id
    )
    
    # Load saved components from the model data
    svm_pca.svm = model_data['svm_model']
    svm_pca.pca = model_data['pca_model']
    svm_pca.scaler = model_data['scaler']
    svm_pca.feature_columns = model_data['feature_columns']
    svm_pca.n_components = model_data['n_components']
    svm_pca.explained_variance_ratios = model_data['explained_variance_ratios']
    svm_pca.model_loaded = True
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"PCA Components: {svm_pca.n_components}")
    print(f"Training accuracy: {model_data.get('training_accuracy', 'N/A')}")
    print(f"Validation accuracy: {model_data.get('validation_accuracy', 'N/A')}")
    print(f"ROC AUC: {model_data.get('roc_auc', 'N/A')}")
    print(f"PR AUC: {model_data.get('pr_auc', 'N/A')}")
    
    # Skip to avoid loading the data if both analyses are skipped
    if args.skip_pca_projection and args.skip_confidence:
        print("\nSkipping all analyses as requested.")
        return 0
    
    # Extract features from training and validation data
    print("\nExtracting features to analyze confidence distributions and PCA projections...")
    
    try:
        # Extract and preprocess features
        X_train, X_val, y_train, y_val, X_full, y_full = svm_pca.extract_and_preprocess_features()
        
        if not args.skip_confidence:
            # Generate confidence distribution plots
            print("\nAnalyzing confidence distributions...")
            try:
                svm_pca.plot_confidence_distribution(
                    X_train, X_val, y_train, y_val,
                    save_path=os.path.join(output_dir, "confidence_distribution_analysis.png")
                )
            except Exception as e:
                print(f"Error analyzing confidence distribution: {e}")
                import traceback
                traceback.print_exc()
        
        if not args.skip_pca_projection:
            # Generate PCA projection plots
            print("\nCreating PCA projections...")
            try:
                svm_pca.plot_pca_projections(
                    X_train, X_val, y_train, y_val,
                    sample_size=args.sample_size,
                    save_path=os.path.join(output_dir, "pca_projection_analysis.png")
                )
            except Exception as e:
                print(f"Error creating PCA projections: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nAnalysis complete. Visualizations saved to: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# DLPFC 151507 usage example

# python scripts/postprocess/analyze_svm_model.py --model-path data/dlpfc/151507/output_VCFs/SVMModel/baseQ0mapQ0/results/svm_pca_model.pkl --dataset DLPFC --section-id 151507