"""
SPARCAL Step 2b: Beagle Genotype Shifting Analysis
====================================================
Compares original mpileup genotypes with Beagle-phased genotypes to
characterize genotype transitions (shifted vs stable). Produces:
  - Per-chromosome pickle results
  - Transition count CSVs
  - BAF/depth distribution plots

This step feeds into the sequence error model (Step 3) by providing
the thresholds for genotype transition boundaries.

Usage:
    python run_beagle_genotype_shifting.py --dataset DLPFC --section_id 151507
    python run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 1
    python run_beagle_genotype_shifting.py --dataset DCIS --section_id 1
"""

import os
import sys
import gzip
import csv
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
# Base Analyzer
# ===========================================================================
class BaseGenotypeAnalyzer:
    """Compare original vs Beagle genotypes across chromosomes."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.setup_paths()
        self.setup_environment()

        # Analysis bins
        self.baf_bins = np.arange(0, 1.05, 0.05)
        self.depth_bins = np.arange(0, 210, 10)
        self.metrics_by_transition = defaultdict(list)
        self.total_variants = 0

        # Track all transitions
        self.all_transitions = Counter()
        self.valid_genotypes = ['0/0', '0/1', '1/1']

    def setup_paths(self):
        """Set up input/output directories from config."""
        base = self.cfg.output_dir
        self.orig_vcf_dir   = os.path.join(base, "output_VCFs", "mpileup_multi_bam")
        self.beagle_vcf_dir = os.path.join(base, "output_VCFs", "beagle")
        self.output_dir     = os.path.join(base, "metrics", "beagle")
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_environment(self):
        """Prepend apps_dir to PATH / LD_LIBRARY_PATH."""
        apps = self.cfg.apps_dir
        os.environ['PATH'] = f"{apps}:{os.environ.get('PATH', '')}"
        ld = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{apps}:{ld}" if ld else apps

    # ---------------------------------------------------------------- keys
    def get_transition_key(self, orig_gt: str, new_gt: str,
                           ref: str, alt: str) -> str:
        return f"{orig_gt}->{new_gt}_{ref}_{alt}"

    def get_simple_transition_key(self, orig_gt: str, new_gt: str) -> str:
        return f"{orig_gt}->{new_gt}"

    def get_display_name(self, key: str) -> str:
        gt_part, ref, alt = key.split('_')
        orig_gt, new_gt = gt_part.split('->')
        return f"{orig_gt}({ref}{alt})->{new_gt}({ref}{alt})"

    # ----------------------------------------------------------- VCF helpers
    def extract_format_field(self, format_str: str, value_str: str,
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

    def extract_info_field(self, info_str: str, field_name: str) -> Optional[str]:
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None

    def extract_metrics(self, info_str: str, format_str: str,
                        value_str: str) -> Tuple[Optional[float], Optional[int]]:
        """Extract BAF and depth from VCF fields (dataset-agnostic)."""
        baf_str = self.extract_format_field(format_str, value_str, "BAF")
        baf = float(baf_str) if baf_str is not None else None

        depth_str = self.extract_info_field(info_str, "DP")
        depth = int(depth_str) if depth_str is not None else None

        return baf, depth

    def extract_genotype(self, format_str: str, value_str: str) -> Optional[str]:
        try:
            gt_idx = format_str.split(':').index('GT')
            return value_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None

    # ----------------------------------------------------------- core logic
    def analyze(self, chromosome: str):
        """Analyze genotype transitions for a single chromosome."""
        orig_vcf = os.path.join(self.orig_vcf_dir, "merged_sorted_gt.vcf.gz")
        beagle_vcf = os.path.join(self.beagle_vcf_dir, f"{chromosome}.vcf.gz")

        print(f"Analyzing {self.__class__.__name__} for {chromosome}")
        try:
            with gzip.open(beagle_vcf, 'rt') as f:
                beagle_variants = self._load_beagle_variants(f, chromosome)
                print(f"Total beagle variants: {len(beagle_variants)}")

            with gzip.open(orig_vcf, 'rt') as f:
                self._process_original_variants(f, beagle_variants, chromosome)
        except Exception as e:
            print(f"Error processing {chromosome}: {e}")
            raise

    def _load_beagle_variants(self, beagle_file, chromosome: str) -> Dict:
        """Load variants from Beagle VCF into {pos: (ref, alt, gt, baf, depth)}."""
        variants = {}
        for line in beagle_file:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[0] != chromosome:
                continue
            try:
                pos = int(fields[1])
                ref, alt = fields[3], fields[4]
                gt = self.extract_genotype(fields[8], fields[9])
                baf, depth = self.extract_metrics(fields[7], fields[8], fields[9])
                if all(x is not None for x in [gt, baf, depth]):
                    variants[pos] = (ref, alt, gt, baf, depth)
            except (ValueError, IndexError):
                continue
        return variants

    def _process_original_variants(self, orig_file, beagle_variants: Dict,
                                   chromosome: str):
        """Compare original genotypes against Beagle-phased genotypes."""
        header_lines = []
        count = 0

        for line in orig_file:
            if line.startswith('#'):
                header_lines.append(line)
                if line.startswith('#CHROM'):
                    break

        for line in orig_file:
            count += 1
            fields = line.strip().split('\t')
            if fields[0] != chromosome:
                continue

            pos = int(fields[1])
            if pos not in beagle_variants:
                continue

            ref, alt = fields[3], fields[4]
            b_ref, b_alt, b_gt, baf, depth = beagle_variants[pos]
            if ref != b_ref or alt != b_alt:
                continue

            orig_gt = self.extract_genotype(fields[8], fields[9])
            if orig_gt is None:
                continue

            variant_data = {
                'metrics': GenotypeMetrics(baf, depth, ref, alt),
                'header_lines': header_lines,
                'line': line,
                'orig_gt': orig_gt,
                'beagle_gt': b_gt,
            }

            if orig_gt in self.valid_genotypes and b_gt in self.valid_genotypes:
                self.all_transitions[self.get_simple_transition_key(orig_gt, b_gt)] += 1

            # Subclass-specific filtering
            self._collect_variant(variant_data, orig_gt, b_gt, ref, alt)

        self.total_variants += count
        print(f"Processed {count} variants from {chromosome}")

    def _collect_variant(self, variant_data: Dict, orig_gt: str,
                         beagle_gt: str, ref: str, alt: str):
        """Override in subclasses to filter shifted vs stable."""
        raise NotImplementedError

    # ----------------------------------------------------------- reporting
    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{self.__class__.__name__} Summary:")
        print("-" * 50)
        print(f"Dataset: {self.cfg.dataset_name}  Section: {self.cfg.section_id}")
        print(f"Total variants processed: {self.total_variants:,}")

        print("\nAll Transitions:")
        print("-" * 20)
        total_count = sum(self.all_transitions.values())
        for transition, count in sorted(self.all_transitions.items(),
                                        key=lambda x: (x[0].split('->')[0],
                                                       x[0].split('->')[1])):
            pct = (count / total_count * 100) if total_count > 0 else 0
            print(f"{transition}: {count:,} ({pct:.2f}%)")

        print("\nDetailed transition counts:")
        print("-" * 20)
        total_matched = 0
        for key, metrics in sorted(self.metrics_by_transition.items()):
            n = len(metrics)
            total_matched += n
            print(f"{self.get_display_name(key)}: {n:,}")

        label = "changed" if isinstance(self, ShiftedGenotypeAnalyzer) else "stable"
        print(f"\nTotal {label} genotypes: {total_matched:,}")
        if self.total_variants > 0:
            print(f"Percentage of total: {total_matched / self.total_variants * 100:.2f}%")

    def save_transition_counts(self, prefix: str):
        """Save transition counts to CSV files."""
        total = sum(self.all_transitions.values())

        # Comprehensive CSV
        path_csv = os.path.join(self.output_dir, f"{prefix}_transition_counts.csv")
        with open(path_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['transition', 'count', 'percentage'])
            for t, c in sorted(self.all_transitions.items()):
                w.writerow([t, c, f"{c / total * 100:.2f}" if total else "0"])

        # Detailed CSV
        path_detail = os.path.join(self.output_dir, f"{prefix}_detailed_counts.csv")
        with open(path_detail, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['transition_format', 'count',
                                              'orig_gt', 'new_gt'])
            w.writeheader()
            for t, c in sorted(self.all_transitions.items()):
                orig, new = t.split('->')
                w.writerow({'transition_format': t, 'count': c,
                            'orig_gt': orig, 'new_gt': new})

        # Text summary
        path_txt = os.path.join(self.output_dir, f"{prefix}_counts.txt")
        with open(path_txt, 'w') as f:
            if isinstance(self, ShiftedGenotypeAnalyzer):
                f.write(f"Total changed genotypes: {total:,}\n")
            for t, c in sorted(self.all_transitions.items()):
                f.write(f"{t}: {c:,}\n")

        print(f"Saved: {path_csv}")
        print(f"Saved: {path_detail}")
        print(f"Saved: {path_txt}")

    # ----------------------------------------------------------- plotting
    def plot_metrics(self, output_dir: str, title_prefix: str):
        """Generate BAF / depth / scatter plots per transition type."""
        os.makedirs(output_dir, exist_ok=True)

        transitions_by_change = defaultdict(list)
        for key in self.metrics_by_transition:
            gt_change = key.split('_')[0]
            transitions_by_change[gt_change].append(key)

        for gt_change, keys in transitions_by_change.items():
            if not keys:
                continue

            n = len(keys)
            fig = plt.figure(figsize=(20, 6 * n))
            fig.suptitle(f'{title_prefix}: {gt_change}', fontsize=16)

            for idx, tkey in enumerate(keys):
                data = self.metrics_by_transition[tkey]
                if not data:
                    continue

                display = self.get_display_name(tkey)
                bafs = [m['metrics'].baf for m in data]
                depths = [min(m['metrics'].depth, 200) for m in data]
                base = idx * 3

                ax1 = plt.subplot(n, 3, base + 1)
                ax1.hist(bafs, bins=self.baf_bins, density=True, alpha=0.7)
                ax1.set_title(f'{display}\nBAF Distribution (N={len(data)})')
                ax1.set_xlabel('BAF'); ax1.set_ylabel('Density')
                ax1.axvline(np.median(bafs), color='red', linestyle='--')

                ax2 = plt.subplot(n, 3, base + 2)
                ax2.hist(depths, bins=self.depth_bins, density=True, alpha=0.7)
                ax2.set_title(f'{display}\nDepth Distribution (capped 200)')
                ax2.set_xlabel('Read Depth'); ax2.set_ylabel('Density')
                ax2.axvline(np.median(depths), color='red', linestyle='--')

                ax3 = plt.subplot(n, 3, base + 3)
                ax3.scatter(bafs, depths, alpha=0.5, s=30)
                ax3.set_title(f'{display}\nBAF vs Depth')
                ax3.set_xlabel('BAF'); ax3.set_ylabel('Depth (capped 200)')
                stats = (f'Median BAF: {np.median(bafs):.3f}\n'
                         f'Median Depth: {np.median(depths):.1f}')
                ax3.text(0.05, 0.95, stats, transform=ax3.transAxes,
                         fontsize=8, verticalalignment='top')

            plt.tight_layout()
            safe = gt_change.replace('/', '_').replace('->', '_to_')
            plt.savefig(os.path.join(output_dir, f'{safe}_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


# ===========================================================================
# Shifted Genotype Analyzer
# ===========================================================================
class ShiftedGenotypeAnalyzer(BaseGenotypeAnalyzer):
    """Tracks genotypes that changed after Beagle phasing."""

    def __init__(self, cfg: PipelineConfig):
        self.target_transitions = [
            ('0/0', '0/1'),
            ('0/1', '0/0'),
            ('1/1', '0/0'),
            ('1/1', '0/1'),
            ('0/1', '1/1'),
        ]
        super().__init__(cfg)

    def _collect_variant(self, variant_data, orig_gt, beagle_gt, ref, alt):
        if (orig_gt, beagle_gt) in self.target_transitions:
            key = self.get_transition_key(orig_gt, beagle_gt, ref, alt)
            self.metrics_by_transition[key].append(variant_data)


# ===========================================================================
# Stable Genotype Analyzer
# ===========================================================================
class StableGenotypeAnalyzer(BaseGenotypeAnalyzer):
    """Tracks genotypes that remained unchanged after Beagle phasing."""

    def __init__(self, cfg: PipelineConfig):
        self.target_genotypes = ['0/0', '0/1', '1/1']
        super().__init__(cfg)

    def _collect_variant(self, variant_data, orig_gt, beagle_gt, ref, alt):
        if orig_gt in self.target_genotypes and orig_gt == beagle_gt:
            key = self.get_transition_key(orig_gt, orig_gt, ref, alt)
            self.metrics_by_transition[key].append(variant_data)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 2b: Beagle Genotype Shifting Analysis")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match a YAML config)")
    parser.add_argument("--section_id",
                        help="Section ID (required for multi-section datasets)")
    parser.add_argument("--chromosomes", nargs="+", default=None,
                        help="Specific chromosomes to process (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.section_id)
    chromosomes = args.chromosomes if args.chromosomes else cfg.regions

    print(f"Dataset:     {cfg.dataset_name}")
    print(f"Section:     {cfg.section_id}")
    print(f"Chromosomes: {chromosomes[0]}..{chromosomes[-1]} ({len(chromosomes)} total)")

    shifted = ShiftedGenotypeAnalyzer(cfg)
    stable = StableGenotypeAnalyzer(cfg)

    for chrom in chromosomes:
        print(f"\nProcessing {chrom}...")
        shifted.analyze(chrom)
        stable.analyze(chrom)

        # Save per-chromosome checkpoints
        for analyzer, label in [(shifted, "shifted"), (stable, "stable")]:
            ckpt = os.path.join(analyzer.output_dir, f'{label}_{chrom}_results.pkl')
            with open(ckpt, 'wb') as f:
                pickle.dump({
                    'metrics_by_transition': dict(analyzer.metrics_by_transition),
                    'total_variants': analyzer.total_variants,
                    'all_transitions': dict(analyzer.all_transitions),
                }, f)

    # Final summary
    print("\n=== Final Statistics ===")
    shifted.print_summary()
    stable.print_summary()

    # Save transition counts
    prefix = cfg.dataset_name
    if cfg.section_id:
        prefix = f"{cfg.dataset_name}_{cfg.section_id}"
    shifted.save_transition_counts(f"{prefix}_shifted")
    stable.save_transition_counts(f"{prefix}_stable")

    # Generate plots
    print("\nGenerating plots...")
    for analyzer, label, title in [
        (shifted, "shifted", "Shifted Genotypes"),
        (stable, "stable", "Stable Genotypes"),
    ]:
        plot_dir = os.path.join(analyzer.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        analyzer.plot_metrics(plot_dir, title)

    # Save final combined results
    for analyzer, label in [(shifted, "shifted"), (stable, "stable")]:
        out = os.path.join(analyzer.output_dir, f'{prefix}_{label}_results.pkl')
        with open(out, 'wb') as f:
            pickle.dump({
                'metrics_by_transition': dict(analyzer.metrics_by_transition),
                'total_variants': analyzer.total_variants,
                'all_transitions': dict(analyzer.all_transitions),
            }, f)
        print(f"Saved {label} results: {out}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()