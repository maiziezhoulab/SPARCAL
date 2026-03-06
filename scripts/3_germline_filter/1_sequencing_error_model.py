"""
SPARCAL Step 3: Sequence Error Model
======================================
Builds per-ref/alt BAF and depth thresholds from genotype shift data
(Step 2b) and applies them to de novo variants (all_filtered_out.vcf.gz)
to separate likely sequence errors from true variants.

Produces:
  - sequence_error.vcf.gz     (variants classified as sequence errors)
  - sequence_no_error.vcf.gz  (variants passing the error model)
  - sequence_error_summary.txt

Usage:
    python run_sequence_error_model.py --dataset DLPFC --section_id 151507
    python run_sequence_error_model.py --dataset P4_TUMOR --section_id 1
    python run_sequence_error_model.py --dataset DCIS --section_id 1 --hom_baf_threshold 0.95
"""

import os
import sys
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm

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


@dataclass
class TransitionMetrics:
    baf_threshold: float
    depth_threshold: float
    transition_type: str
    genotype: str  # "0/1" or "1/1"


# ===========================================================================
# Sequence Error Model
# ===========================================================================
class SequenceErrorModel:
    """
    Uses genotype-shifting statistics from Step 2b to build per-ref/alt
    thresholds that distinguish sequence errors from true de novo variants.
    """

    def __init__(self, cfg: PipelineConfig,
                 het_baf_threshold: float = None,
                 het_depth_threshold: float = None,
                 hom_baf_threshold: float = 0.9,
                 hom_depth_threshold: float = None):
        self.cfg = cfg

        # Manual threshold overrides (None → auto-calculate from data)
        self.manual_het_baf_threshold = het_baf_threshold
        self.manual_het_depth_threshold = het_depth_threshold
        self.manual_hom_baf_threshold = hom_baf_threshold
        self.manual_hom_depth_threshold = hom_depth_threshold

        # Per-ref/alt transition metrics
        self.het_transition_metrics = {}
        self.hom_transition_metrics = {}

        self.setup_directories()

    # ---------------------------------------------------------------- paths
    def setup_directories(self):
        """Set up input/output directories from config."""
        base = self.cfg.output_dir

        self.filtered_vcf = os.path.join(
            base, "output_VCFs", "beagle", "all_filtered_out.vcf.gz")
        self.output_dir = os.path.join(
            base, "output_VCFs", "SeqErrModel")
        self.metrics_dir = os.path.join(
            base, "metrics", "beagle")

        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------------- VCF helpers
    @staticmethod
    def extract_format_field(format_str: str, value_str: str,
                             field_name: str) -> Optional[str]:
        try:
            fields = format_str.split(':')
            if field_name not in fields:
                return None
            idx = fields.index(field_name)
            values = value_str.split(':')
            return values[idx] if idx < len(values) else None
        except (ValueError, IndexError):
            return None

    @staticmethod
    def extract_info_field(info_str: str, field_name: str) -> Optional[str]:
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None

    def _extract_metrics(self, info_str: str, format_str: str,
                         value_str: str) -> Tuple[Optional[float], Optional[int]]:
        """Extract BAF and depth from VCF fields."""
        baf_str = self.extract_format_field(format_str, value_str, "BAF")
        baf = float(baf_str) if baf_str is not None else None

        depth_str = self.extract_info_field(info_str, "DP")
        depth = int(depth_str) if depth_str is not None else None

        return baf, depth

    @staticmethod
    def _extract_genotype(format_str: str, value_str: str) -> Optional[str]:
        try:
            gt_idx = format_str.split(':').index('GT')
            return value_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------ threshold calculation
    def _load_transition_metrics(self) -> Dict:
        """Load genotype-shift metrics from Step 2b pickle cache."""
        sid = f"_{self.cfg.section_id}" if self.cfg.section_id else ""
        cache_file = os.path.join(
            self.metrics_dir,
            f"{self.cfg.dataset_name}{sid}_shifted_results.pkl")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Shifted results cache not found: {cache_file}\n"
                "Run Step 2b (beagle_genotype_shifting) first.")

        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
        return results['metrics_by_transition']

    def calculate_transition_thresholds(self):
        """
        Calculate BAF and depth thresholds per ref→alt pair for both
        heterozygous (0/1→0/0) and homozygous (1/1→0/0) transitions.
        """
        metrics_by_transition = self._load_transition_metrics()

        # ---- HET thresholds (0/1 → 0/0) ----
        print("\nHET (0/1→0/0) thresholds:")
        print("-" * 80)
        print(f"{'Ref→Alt':<10} {'BAF Thr':<15} {'Depth Thr':<15} {'N Variants':<15}")
        print("-" * 80)

        for transition, metrics in metrics_by_transition.items():
            if not metrics:
                continue
            parts = transition.split('_')[0].split('->')
            orig_gt, new_gt = parts[0], parts[1]

            if orig_gt == '0/1' and new_gt == '0/0':
                ref, alt = metrics[0]['metrics'].ref, metrics[0]['metrics'].alt
                key = f"{ref}->{alt}"
                bafs = [m['metrics'].baf for m in metrics]
                depths = [m['metrics'].depth for m in metrics]

                baf_thr = (self.manual_het_baf_threshold
                           if self.manual_het_baf_threshold is not None
                           else np.median(bafs))
                depth_thr = (self.manual_het_depth_threshold
                             if self.manual_het_depth_threshold is not None
                             else np.median(depths))

                self.het_transition_metrics[key] = TransitionMetrics(
                    baf_threshold=baf_thr, depth_threshold=depth_thr,
                    transition_type=key, genotype='0/1')
                print(f"{key:<10} {baf_thr:>13.3f}  {depth_thr:>13.1f}  {len(metrics):>13,}")

        # ---- HOM thresholds (1/1 → 0/0) ----
        print(f"\nHOM (1/1→0/0) thresholds:")
        print("-" * 80)
        print(f"{'Ref→Alt':<10} {'BAF Thr':<15} {'Depth Thr':<15} {'N Variants':<15}")
        print("-" * 80)

        for transition, metrics in metrics_by_transition.items():
            if not metrics:
                continue
            parts = transition.split('_')[0].split('->')
            orig_gt, new_gt = parts[0], parts[1]

            if orig_gt == '1/1' and new_gt == '0/0':
                ref, alt = metrics[0]['metrics'].ref, metrics[0]['metrics'].alt
                key = f"{ref}->{alt}"
                depths = [m['metrics'].depth for m in metrics]

                baf_thr = self.manual_hom_baf_threshold
                depth_thr = (self.manual_hom_depth_threshold
                             if self.manual_hom_depth_threshold is not None
                             else np.median(depths))

                self.hom_transition_metrics[key] = TransitionMetrics(
                    baf_threshold=baf_thr, depth_threshold=depth_thr,
                    transition_type=key, genotype='1/1')
                print(f"{key:<10} {baf_thr:>13.3f}  {depth_thr:>13.1f}  {len(metrics):>13,}")

    # --------------------------------------------------------- apply model
    def apply_model(self):
        """
        Apply het + hom sequence error thresholds to de novo variants.
        Produces sequence_error.vcf.gz and sequence_no_error.vcf.gz.
        """
        if not self.het_transition_metrics and not self.hom_transition_metrics:
            self.calculate_transition_thresholds()

        seq_error_vcf = os.path.join(self.output_dir, "sequence_error.vcf.gz")
        no_seq_error_vcf = os.path.join(self.output_dir, "sequence_no_error.vcf.gz")

        counters = {
            'total': 0, 'het_seq_error': 0, 'hom_seq_error': 0,
            'total_het': 0, 'total_hom': 0, 'no_error': 0,
            'filtered_0_0': 0, 'missing_info': 0,
        }

        # Count total for progress bar
        print("\nCounting variants...")
        total = sum(1 for l in gzip.open(self.filtered_vcf, 'rt')
                    if not l.startswith('#'))
        print(f"Found {total:,} variants to process")

        with gzip.open(self.filtered_vcf, 'rt') as f_in, \
             gzip.open(seq_error_vcf, 'wt') as f_err, \
             gzip.open(no_seq_error_vcf, 'wt') as f_ok:

            # Copy header
            for line in f_in:
                if line.startswith('#'):
                    f_err.write(line)
                    f_ok.write(line)
                    if line.startswith('#CHROM'):
                        f_err.write('##INFO=<ID=SEQ_ERROR_MODEL,Number=1,Type=String,'
                                    'Description="Sequence error model (HET or HOM)">\n')
                        break

            pbar = tqdm(total=total, desc="Applying model", unit="var")
            for line in f_in:
                if line.startswith('#'):
                    continue

                counters['total'] += 1
                fields = line.strip().split('\t')
                ref, alt = fields[3], fields[4]
                baf, depth = self._extract_metrics(fields[7], fields[8], fields[9])
                gt = self._extract_genotype(fields[8], fields[9])

                if None in (baf, depth, gt):
                    f_ok.write(line)
                    counters['missing_info'] += 1
                    pbar.update(1)
                    continue

                if gt == '0/0':
                    counters['filtered_0_0'] += 1
                    pbar.update(1)
                    continue

                key = f"{ref}->{alt}"
                is_error = False
                model_label = None

                if gt == '0/1':
                    counters['total_het'] += 1
                    if key in self.het_transition_metrics:
                        m = self.het_transition_metrics[key]
                        if baf <= m.baf_threshold and depth <= m.depth_threshold:
                            is_error = True
                            model_label = "HET"
                            counters['het_seq_error'] += 1

                elif gt == '1/1':
                    counters['total_hom'] += 1
                    if key in self.hom_transition_metrics:
                        m = self.hom_transition_metrics[key]
                        if baf <= m.baf_threshold and depth <= m.depth_threshold:
                            is_error = True
                            model_label = "HOM"
                            counters['hom_seq_error'] += 1

                if is_error:
                    fields[7] += f";SEQ_ERROR_MODEL={model_label}"
                    f_err.write('\t'.join(fields) + '\n')
                else:
                    f_ok.write(line)
                    counters['no_error'] += 1

                pbar.update(1)
            pbar.close()

        # ---- Summary ----
        total_nonref = counters['total'] - counters['filtered_0_0']
        total_err = counters['het_seq_error'] + counters['hom_seq_error']
        het_pct = (counters['het_seq_error'] / counters['total_het'] * 100
                   if counters['total_het'] > 0 else 0)
        hom_pct = (counters['hom_seq_error'] / counters['total_hom'] * 100
                   if counters['total_hom'] > 0 else 0)

        summary_lines = [
            "Sequence Error Model Summary",
            "=" * 60,
            f"Dataset: {self.cfg.dataset_name}  Section: {self.cfg.section_id}",
            "",
            "Threshold Settings:",
            f"  HET BAF:   {self.manual_het_baf_threshold or 'auto (median)'}",
            f"  HET depth: {self.manual_het_depth_threshold or 'auto (median)'}",
            f"  HOM BAF:   {self.manual_hom_baf_threshold}",
            f"  HOM depth: {self.manual_hom_depth_threshold or 'auto (median)'}",
            "",
            f"Total variants:       {counters['total']:,}",
            f"  0/0 filtered out:   {counters['filtered_0_0']:,}",
            f"  Non-ref variants:   {total_nonref:,}",
            f"  Het (0/1):          {counters['total_het']:,}",
            f"  Hom (1/1):          {counters['total_hom']:,}",
            "-" * 60,
            f"Het seq errors:       {counters['het_seq_error']:,} ({het_pct:.2f}% of het)",
            f"Hom seq errors:       {counters['hom_seq_error']:,} ({hom_pct:.2f}% of hom)",
            f"Total seq errors:     {total_err:,} ({total_err / total_nonref * 100:.2f}% of non-ref)" if total_nonref else "",
            f"Non-errors:           {counters['no_error']:,}",
            f"Missing info:         {counters['missing_info']:,}",
        ]

        print("\n" + "\n".join(summary_lines))

        summary_file = os.path.join(self.output_dir, "sequence_error_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"\nOutputs:")
        print(f"  Seq errors:     {seq_error_vcf}")
        print(f"  Non-errors:     {no_seq_error_vcf}")
        print(f"  Summary:        {summary_file}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 3: Sequence Error Model")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match a YAML config)")
    parser.add_argument("--section_id",
                        help="Section ID (required for multi-section datasets)")
    parser.add_argument("--het_baf_threshold", type=float, default=None,
                        help="Manual BAF threshold for het variants (default: auto)")
    parser.add_argument("--het_depth_threshold", type=float, default=None,
                        help="Manual depth threshold for het variants (default: auto)")
    parser.add_argument("--hom_baf_threshold", type=float, default=0.99,
                        help="BAF threshold for hom variants (default: 0.99)")
    parser.add_argument("--hom_depth_threshold", type=float, default=None,
                        help="Manual depth threshold for hom variants (default: auto)")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.section_id)
    print(f"Dataset:  {cfg.dataset_name}")
    print(f"Section:  {cfg.section_id}")

    model = SequenceErrorModel(
        cfg,
        het_baf_threshold=args.het_baf_threshold,
        het_depth_threshold=args.het_depth_threshold,
        hom_baf_threshold=args.hom_baf_threshold,
        hom_depth_threshold=args.hom_depth_threshold,
    )
    model.calculate_transition_thresholds()
    model.apply_model()


if __name__ == "__main__":
    main()