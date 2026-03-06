"""
SPARCAL Step 4: SPARCAL-Net Classifier (Germline pipeline only)
================================================================
Trains a 3-class ML model (homozygous / heterozygous / no_variance)
using genotype-shift data from Steps 2b-3, then applies it to de novo
variants to produce classified VCFs.

This step is ONLY used in the germline pipeline. Somatic pipelines
skip directly from Beagle to single_bam_filter.

Usage:
    python run_sparcal_net.py --dataset DLPFC --section_id 151507 --model-type neural_network
    python run_sparcal_net.py --dataset DLPFC --section_id 151507 --skip-training --model-type svm
"""

import os
import sys
import gzip
import pickle
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_curve, roc_auc_score, make_scorer)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str


# ===========================================================================
# Feature Extraction
# ===========================================================================
class FeatureExtractor:
    """Extract VCF INFO/FORMAT fields into a feature DataFrame."""

    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def extract_info_field(info_str: str, field: str) -> Optional[float]:
        for item in info_str.split(';'):
            if item.startswith(f"{field}="):
                try:
                    return float(item.split('=')[1].split(',')[0])
                except (ValueError, IndexError):
                    return None
        return None

    @staticmethod
    def extract_format_field(format_str: str, sample_str: str,
                             field: str) -> Optional:
        try:
            idx = format_str.split(':').index(field)
            value = sample_str.split(':')[idx]
            if field == 'PL':
                return float(value.split(',')[0])
            return value if field == 'GT' else float(value)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def extract_i16_values(info_str: str) -> List[float]:
        for item in info_str.split(';'):
            if item.startswith('I16='):
                try:
                    values = [float(x) for x in item.split('=')[1].split(',')]
                    if len(values) == 16:
                        return values
                except (ValueError, IndexError):
                    pass
        return [np.nan] * 16

    def extract_features(self, vcf_path: str) -> pd.DataFrame:
        """Extract features from a VCF file."""
        features = []
        numeric_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB',
                          'MQSB', 'SGB', 'MQ0F', 'BAF', 'GQ']
        info_fields = ['DP', 'VDB', 'RPB', 'MQB', 'BQB', 'SGB', 'BAF']
        custom_fields = ['BAF', 'GQ']

        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                try:
                    fd = {fld: 0.0 for fld in numeric_fields}
                    fd['POS'] = int(fields[1])

                    for fld in info_fields:
                        v = self.extract_info_field(fields[7], fld)
                        if v is not None:
                            fd[fld] = v

                    for fld in custom_fields:
                        v = self.extract_format_field(fields[8], fields[9], fld)
                        if v is not None:
                            fd[fld] = v

                    i16 = self.extract_i16_values(fields[7])
                    for i, v in enumerate(i16):
                        if not np.isnan(v):
                            fd[f'I16_{i}'] = v

                    features.append(fd)
                except Exception:
                    continue

        if not features:
            raise ValueError(f"No valid features extracted from {vcf_path}")
        return pd.DataFrame(features)


# ===========================================================================
# Training Set Builder
# ===========================================================================
class TrainingSetBuilder:
    """Build 3-class training sets from genotype-shift pickles + seq error VCF."""

    def __init__(self, cfg: PipelineConfig, max_training_samples: int = 90000):
        self.cfg = cfg
        self.max_training_samples = max_training_samples
        self.setup_paths()
        self.setup_environment()

    def setup_environment(self):
        apps = self.cfg.apps_dir
        os.environ['PATH'] = f"{apps}:{os.environ.get('PATH', '')}"
        ld = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{apps}:{ld}" if ld else apps

    def setup_paths(self):
        base = self.cfg.output_dir
        sid = f"_{self.cfg.section_id}" if self.cfg.section_id else ""
        prefix = f"{self.cfg.dataset_name}{sid}"

        self.shifted_results = os.path.join(
            base, "metrics", "beagle", f"{prefix}_shifted_results.pkl")
        self.stable_results = os.path.join(
            base, "metrics", "beagle", f"{prefix}_stable_results.pkl")
        self.input_vcf = os.path.join(
            base, "output_VCFs", "mpileup_multi_bam", "merged_sorted_gt.vcf.gz")
        self.seq_error_vcf = os.path.join(
            base, "output_VCFs", "SeqErrModel", "sequence_error.vcf.gz")
        self.seq_no_error_vcf = os.path.join(
            base, "output_VCFs", "SeqErrModel", "sequence_no_error.vcf.gz")

        self.output_dir = os.path.join(base, "output_VCFs", "Classifier")
        self.model_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.model_dir, exist_ok=True)

    # --------------------------------------------------------- pickle helpers
    @staticmethod
    def load_pickle_results(file_path: str) -> Dict:
        with open(file_path, 'rb') as f:
            return pickle.load(f).get('metrics_by_transition', {})

    def collect_variants_from_metrics(self, metrics_dict: Dict,
                                      target_transition: Tuple[str, str]) -> List[Dict]:
        orig_gt, new_gt = target_transition
        key = f"{orig_gt}->{new_gt}"
        variants = []
        for tkey, metrics in metrics_dict.items():
            if tkey.startswith(key):
                for m in metrics:
                    if 'line' in m:
                        variants.append({'line': m['line'],
                                         'original_gt': orig_gt, 'new_gt': new_gt})
        print(f"Collected: {key}: {len(variants)}")
        return variants

    def collect_seq_error_variants(self) -> List[Dict]:
        variants = []
        with gzip.open(self.seq_error_vcf, 'rt') as f:
            header_lines = []
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line)
                    continue
                variants.append({'line': line, 'header_lines': header_lines})
        return variants

    # --------------------------------------------------------- build
    def build_training_sets(self):
        """Build 3-class training VCFs (homozygous / heterozygous / no_variance)."""
        print("\nLoading transition metrics...")
        shifted = self.load_pickle_results(self.shifted_results)
        stable = self.load_pickle_results(self.stable_results)

        hom = (self.collect_variants_from_metrics(stable, ("1/1", "1/1")) +
               self.collect_variants_from_metrics(shifted, ("0/1", "1/1")))
        het = (self.collect_variants_from_metrics(stable, ("0/1", "0/1")) +
               self.collect_variants_from_metrics(shifted, ("1/1", "0/1")))
        novar = (self.collect_variants_from_metrics(shifted, ("0/1", "0/0")) +
                 self.collect_variants_from_metrics(shifted, ("1/1", "0/0")) +
                 self.collect_seq_error_variants())

        for v in hom:   v['class'] = 'homozygous'
        for v in het:   v['class'] = 'heterozygous'
        for v in novar: v['class'] = 'no_variance'

        total = len(hom) + len(het) + len(novar)
        if total > self.max_training_samples:
            ratio = total / self.max_training_samples
            hom = _random_sample(hom, int(len(hom) / ratio))
            het = _random_sample(het, int(len(het) / ratio))
            novar = _random_sample(novar, self.max_training_samples - len(hom) - len(het))
            print(f"Sampled: {len(hom)} hom, {len(het)} het, {len(novar)} novar")

        self.save_variants(hom, "homozygous_training.vcf.gz")
        self.save_variants(het, "heterozygous_training.vcf.gz")
        self.save_variants(novar, "no_variance_training.vcf.gz")
        self.save_variants(hom + het + novar, "all_classes_training.vcf.gz")

        self.homozygous_variants = hom
        self.heterozygous_variants = het
        self.novar_variants = novar

        print(f"\nTraining Set: {len(hom)} hom, {len(het)} het, {len(novar)} novar")

    def save_variants(self, variants: List[Dict], filename: str):
        """Save variants to a bgzipped+indexed VCF."""
        output_path = os.path.join(self.output_dir, filename)
        temp_vcf = output_path.replace('.vcf.gz', '.temp.vcf')
        bgzip = self.cfg.tool("bgzip")
        tabix = self.cfg.tool("tabix")

        if not variants:
            with open(temp_vcf, 'w') as f:
                f.write("##fileformat=VCFv4.2\n")
                f.write(f"##reference={self.cfg.reference_fasta}\n")
                f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        else:
            with open(temp_vcf, 'w') as f:
                header_written = False
                for v in variants:
                    if not header_written and 'header_lines' in v and v['header_lines']:
                        for h in v['header_lines']:
                            f.write(h)
                        header_written = True
                        break
                if not header_written:
                    f.write("##fileformat=VCFv4.2\n")
                    f.write(f"##reference={self.cfg.reference_fasta}\n")
                    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                for v in variants:
                    if 'line' in v:
                        f.write(v['line'])

        subprocess.run(f"{bgzip} -c {temp_vcf} > {output_path}", shell=True, check=True)
        subprocess.run([tabix, '-p', 'vcf', output_path], check=True)
        if os.path.exists(temp_vcf):
            os.remove(temp_vcf)


def _random_sample(variants: List[Dict], n: int) -> List[Dict]:
    if len(variants) <= n:
        return variants
    idx = np.random.choice(len(variants), size=n, replace=False)
    return [variants[i] for i in idx]


# ===========================================================================
# Custom scorer — F1 on variant classes only (het + hom)
# ===========================================================================
def variant_only_f1(y_true, y_pred, label_encoder=None):
    if label_encoder is not None:
        het_idx = label_encoder.transform(['heterozygous'])[0]
        hom_idx = label_encoder.transform(['homozygous'])[0]
    else:
        het_idx, hom_idx = 1, 2
    mask = (y_true == het_idx) | (y_true == hom_idx)
    if not np.any(mask):
        return 0.0
    return f1_score(y_true[mask], y_pred[mask],
                    labels=[het_idx, hom_idx], average='macro')

variant_f1_scorer = make_scorer(variant_only_f1)


# ===========================================================================
# Model Trainer
# ===========================================================================
class ModelTrainer:
    """Train and apply a 3-class classifier for de novo variant genotyping."""

    def __init__(self, cfg: PipelineConfig, max_training_samples: int = 90000):
        self.cfg = cfg
        self.max_training_samples = max_training_samples
        self.feature_extractor = FeatureExtractor()
        self.builder = TrainingSetBuilder(cfg, max_training_samples)
        self.use_pca = False
        self.pca = None
        self.model = None
        self.scaler = StandardScaler()
        self.explained_variance_ratios = []
        self.n_components = None
        self.model_loaded = False
        self.model_type = "svm"
        self.class_labels = ['no_variance', 'heterozygous', 'homozygous']
        self.label_encoder = LabelEncoder()
        self.multiclass = True

    def build_training_sets(self):
        self.builder.build_training_sets()

    # ------------------------------------------------------- feature prep
    def extract_and_preprocess_features(self):
        """Extract features from 3-class training VCFs, scale, split."""
        print("\nExtracting features...")
        dfs = []
        for label, fname in [('homozygous', 'homozygous_training.vcf.gz'),
                              ('heterozygous', 'heterozygous_training.vcf.gz'),
                              ('no_variance', 'no_variance_training.vcf.gz')]:
            df = self.feature_extractor.extract_features(
                os.path.join(self.builder.output_dir, fname))
            df['class'] = label
            dfs.append(df)

        features = pd.concat(dfs).fillna(0)
        X = features.drop('class', axis=1)
        y_raw = features['class']

        self.label_encoder.fit(self.class_labels)
        y = self.label_encoder.transform(y_raw)
        self.feature_columns = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_val, y_train, y_val, X, y

    # ------------------------------------------------------- PCA
    def find_optimal_pca_components(self, X_train, threshold=0.95):
        max_c = min(X_train.shape)
        pca_full = PCA(n_components=max_c)
        pca_full.fit(X_train)
        self.explained_variance_ratios = pca_full.explained_variance_ratio_
        cumvar = np.cumsum(self.explained_variance_ratios)
        self.n_components = int(np.argmax(cumvar >= threshold) + 1)
        print(f"PCA: {self.n_components} components for {threshold*100}% variance")
        return self.n_components

    def plot_explained_variance(self, save_path=None):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.explained_variance_ratios)+1),
                self.explained_variance_ratios, alpha=0.7)
        plt.xlabel('Component'); plt.ylabel('Explained Variance')
        plt.subplot(1, 2, 2)
        cumvar = np.cumsum(self.explained_variance_ratios)
        plt.plot(range(1, len(cumvar)+1), cumvar, 'o-', color='green')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
        plt.axvline(x=self.n_components, color='gray', linestyle='--')
        plt.xlabel('Components'); plt.ylabel('Cumulative Variance'); plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------- train
    def train_model(self, model_type='svm', variance_threshold=0.95,
                    use_pca=False, grid_search=False):
        self.model_type = model_type
        self.use_pca = use_pca

        X_train, X_val, y_train, y_val, _, _ = self.extract_and_preprocess_features()

        if use_pca:
            n = self.find_optimal_pca_components(X_train, variance_threshold)
            self.pca = PCA(n_components=n)
            X_tr = self.pca.fit_transform(X_train)
            X_va = self.pca.transform(X_val)
        else:
            X_tr, X_va = X_train, X_val

        # Init model
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True,
                             decision_function_shape='ovr', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not available")
            self.model = xgb.XGBClassifier(objective='multi:softprob',
                                           num_class=3, random_state=42)
        elif model_type == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32),
                                       activation='relu', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if grid_search:
            param_grids = {
                'svm': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]},
                'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
                'xgboost': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
                'neural_network': {'hidden_layer_sizes': [(32,16), (64,32), (128,64)], 'alpha': [0.0001, 0.001, 0.01]},
            }
            pg = param_grids.get(model_type, {})
            if pg:
                grid = GridSearchCV(self.model, pg, cv=5,
                                    scoring=variant_f1_scorer, verbose=1, n_jobs=-1)
                grid.fit(X_tr, y_train)
                self.model = grid.best_estimator_
                print(f"Best params: {grid.best_params_}")

        self.model.fit(X_tr, y_train)

        # Evaluate
        y_tr_pred = self.model.predict(X_tr)
        y_va_pred = self.model.predict(X_va)
        tr_acc = self.model.score(X_tr, y_train)
        va_acc = self.model.score(X_va, y_val)
        tr_f1 = variant_only_f1(y_train, y_tr_pred, self.label_encoder)
        va_f1 = variant_only_f1(y_val, y_va_pred, self.label_encoder)

        print(f"\nTrain acc: {tr_acc:.3f}  Val acc: {va_acc:.3f}")
        print(f"Train variant-F1: {tr_f1:.3f}  Val variant-F1: {va_f1:.3f}")

        report = classification_report(y_val, y_va_pred, target_names=self.class_labels)
        print(f"\n{report}")

        cm = confusion_matrix(y_val, y_va_pred)
        self._plot_confusion_matrix(cm,
            os.path.join(self.builder.model_dir, f"{model_type}_confusion_matrix.png"))

        # Save model bundle
        bundle = {
            'model': self.model, 'model_type': model_type,
            'pca_model': self.pca if use_pca else None,
            'scaler': self.scaler, 'feature_columns': self.feature_columns,
            'n_components': self.n_components if use_pca else None,
            'training_accuracy': tr_acc, 'validation_accuracy': va_acc,
            'variant_train_f1': tr_f1, 'variant_val_f1': va_f1,
            'explained_variance_ratios': self.explained_variance_ratios if use_pca else None,
            'label_encoder': self.label_encoder,
            'class_labels': self.class_labels,
            'use_pca': use_pca, 'multiclass': True,
        }
        with open(os.path.join(self.builder.model_dir, f"{model_type}_model.pkl"), 'wb') as f:
            pickle.dump(bundle, f)

        # Save text metrics
        with open(os.path.join(self.builder.model_dir, f"{model_type}_metrics.txt"), 'w') as f:
            f.write(f"{model_type.upper()} Model (3-class)\n")
            f.write(f"Dataset: {self.cfg.dataset_name}  Section: {self.cfg.section_id}\n\n")
            f.write(f"Train acc: {tr_acc:.3f}  Val acc: {va_acc:.3f}\n")
            f.write(f"Train variant-F1: {tr_f1:.3f}  Val variant-F1: {va_f1:.3f}\n\n")
            f.write(report + "\n")
            f.write(f"Confusion matrix:\n{cm}\n")

        print(f"Model saved to: {self.builder.model_dir}")
        return bundle

    def _plot_confusion_matrix(self, cm, save_path):
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix'); plt.colorbar()
        ticks = np.arange(len(self.class_labels))
        plt.xticks(ticks, self.class_labels, rotation=45)
        plt.yticks(ticks, self.class_labels)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------- load
    def load_model(self, model_type='svm'):
        path = os.path.join(self.builder.model_dir, f"{model_type}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.model_type = data.get('model_type', model_type)
        self.pca = data['pca_model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.use_pca = data.get('use_pca', self.pca is not None)
        self.n_components = data['n_components'] if self.use_pca else None
        self.label_encoder = data.get('label_encoder', self.label_encoder)
        self.class_labels = data.get('class_labels', self.class_labels)
        self.model_loaded = True
        print(f"Loaded {self.model_type} model — val acc: {data['validation_accuracy']:.3f}")
        return data

    # ------------------------------------------------------- apply
    def apply_model_to_vcf(self, input_vcf, output_vcf=None,
                           model_type=None, conf_threshold=0.5):
        if model_type:
            self.model_type = model_type
        if not output_vcf:
            output_vcf = os.path.join(self.builder.model_dir,
                                       f"{self.model_type}_predictions.vcf.gz")
        if not self.model_loaded:
            self.load_model(self.model_type)

        print(f"\nApplying {self.model_type.upper()} to {input_vcf}...")

        features = self.feature_extractor.extract_features(input_vcf)
        features = features.fillna(features.mean())
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        X = features[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        X_t = self.pca.transform(X_scaled) if self.use_pca and self.pca else X_scaled

        predictions = self.model.predict(X_t)
        probs = self.model.predict_proba(X_t)
        class_preds = self.label_encoder.inverse_transform(predictions)

        mt = self.model_type.upper()
        with gzip.open(input_vcf, 'rt') as fin, gzip.open(output_vcf, 'wt') as fout:
            for line in fin:
                if line.startswith('#'):
                    if line.startswith('#CHROM'):
                        fout.write(f'##INFO=<ID={mt}_CLASS,Number=1,Type=String,'
                                   f'Description="Predicted class">\n')
                        fout.write(f'##INFO=<ID={mt}_HOMO,Number=1,Type=Float,'
                                   f'Description="P(homozygous)">\n')
                        fout.write(f'##INFO=<ID={mt}_HETERO,Number=1,Type=Float,'
                                   f'Description="P(heterozygous)">\n')
                        fout.write(f'##INFO=<ID={mt}_NOVAR,Number=1,Type=Float,'
                                   f'Description="P(no_variance)">\n')
                    fout.write(line)
                    if line.startswith('#CHROM'):
                        break

            idx = 0
            for line in fin:
                if idx < len(predictions):
                    fields = line.strip().split('\t')
                    cp = class_preds[idx]
                    p = probs[idx]
                    fields[7] += (f";{mt}_CLASS={cp}"
                                  f";{mt}_HOMO={p[2]:.4f}"
                                  f";{mt}_HETERO={p[1]:.4f}"
                                  f";{mt}_NOVAR={p[0]:.4f}")
                    fout.write('\t'.join(fields) + '\n')
                    idx += 1
                else:
                    fout.write(line)

        print(f"Classified {idx} variants → {output_vcf}")
        self._create_filtered_vcfs(output_vcf, class_preds, probs, conf_threshold)
        return output_vcf

    def _create_filtered_vcfs(self, input_vcf, class_preds, probs, threshold):
        """Split predictions into per-class VCFs."""
        bgzip = self.cfg.tool("bgzip")
        tabix = self.cfg.tool("tabix")
        names = {'homozygous': 'homozygous', 'heterozygous': 'heterozygous',
                 'no_variance': 'no_variance'}
        handles = {}
        paths = {}
        for cls in names:
            p = os.path.join(self.builder.model_dir,
                             f"{self.model_type}_{cls}.vcf")
            paths[cls] = p
            handles[cls] = open(p, 'w')

        with gzip.open(input_vcf, 'rt') as fin:
            for line in fin:
                if line.startswith('#'):
                    for h in handles.values():
                        h.write(line)
                    if line.startswith('#CHROM'):
                        break

            idx = 0
            counts = {c: 0 for c in names}
            for line in fin:
                if idx < len(class_preds):
                    cls = class_preds[idx]
                    p = probs[idx][self.label_encoder.transform([cls])[0]]
                    if p >= threshold:
                        handles[cls].write(line)
                        counts[cls] += 1
                    idx += 1

        for h in handles.values():
            h.close()

        for cls, vcf_path in paths.items():
            gz = vcf_path + ".gz"
            subprocess.run([bgzip, '-f', vcf_path], check=True)
            subprocess.run([tabix, '-p', 'vcf', gz], check=True)

        print(f"\nFiltered (threshold={threshold}):")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 4: SPARCAL-Net Classifier")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match a YAML config)")
    parser.add_argument("--section_id",
                        help="Section ID")
    parser.add_argument("--model-type", default="neural_network",
                        choices=["svm", "random_forest", "xgboost", "neural_network"])
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--input-vcf", help="Custom input VCF for prediction")
    parser.add_argument("--max-training-samples", type=int, default=90000)
    parser.add_argument("--use-pca", action="store_true")
    args = parser.parse_args()

    if args.model_type == "xgboost" and not XGB_AVAILABLE:
        parser.error("XGBoost not installed")

    cfg = load_config(args.dataset, args.section_id)
    print(f"Dataset:  {cfg.dataset_name}")
    print(f"Section:  {cfg.section_id}")
    print(f"Model:    {args.model_type}")

    trainer = ModelTrainer(cfg, args.max_training_samples)

    if not args.skip_training:
        trainer.build_training_sets()
        trainer.train_model(model_type=args.model_type,
                            variance_threshold=args.variance_threshold,
                            use_pca=args.use_pca,
                            grid_search=args.grid_search)

        input_vcf = args.input_vcf or trainer.builder.seq_no_error_vcf
        trainer.apply_model_to_vcf(input_vcf, model_type=args.model_type,
                                   conf_threshold=args.confidence_threshold)
    else:
        trainer.load_model(args.model_type)
        input_vcf = args.input_vcf or trainer.builder.seq_no_error_vcf
        trainer.apply_model_to_vcf(input_vcf, model_type=args.model_type,
                                   conf_threshold=args.confidence_threshold)


if __name__ == "__main__":
    main()