#!/usr/bin/env python3
"""
Feature Analysis for SPARCAL Spatial SNV Filter
================================================
Analyses discriminative power of each spatial/purity feature computed by
run_spatial_filter_enhanced.py.  Reads the all_variant_scores.txt output
(one row per variant) and produces:

  1. Per-feature violin + strip plots: distribution across germline / somatic /
     ambiguous classes (denovo only, since defined are forced germline).
  2. Correlation analysis: Spearman rho between each feature and the
     numeric class label, plus vs. the two composite scores.
  3. Heatmap: feature × class mean/median z-scored overview.
  4. Pairwise scatter matrix (seaborn PairGrid) coloured by class — reveals
     which feature pairs separate classes best.
  5. Feature importance proxy via ablation: for each feature, zero it out,
     recompute the composite germline/somatic scores using the documented
     weights, and count how many denovo variants flip classification.
  6. Redundancy check: feature–feature Spearman correlation matrix — flags
     highly correlated features (|r| > 0.7) that carry duplicate information
     (especially the known ε = δ × ζ proxy issue).
  7. Summary table printed to stdout + saved as feature_summary.tsv.

Usage
-----
    python feature_analysis.py \\
        --input  /path/to/all_variant_scores.txt \\
        --output_dir /path/to/output_plots/ \\
        [--germline_threshold 0.3] \\
        [--somatic_threshold  0.2] \\
        [--title "P4 Tumor Section 1"]

The script operates on denovo variants only for every analysis that touches
classification (defined variants are forced germline regardless of features,
so including them would bias the distributions).  The "race == defined"
rows are kept for the composite-score plots but excluded from all
feature-level analyses.

Author: Yuki & Claude
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────

# Must match run_spatial_filter_enhanced.py
DEFAULT_GERMLINE_THRESHOLD = 0.3
DEFAULT_SOMATIC_THRESHOLD  = 0.2

# Weights from run_spatial_filter_enhanced.py (clone+CNV mode)
WEIGHTS_GERMLINE = {'f_spatial_uniformity': 0.4,
                    'f_global_prevalence':   0.3,
                    'f_purity_independence': 0.3}

WEIGHTS_SOMATIC_CLONE = {'f_purity_correlation':   0.25,
                          'f_clone_specific_proxy': 0.15,
                          'f_spatial_clustering':   0.20,
                          'f_clone_enrichment':     0.25,
                          'f_cnv_consistency':      0.15}

WEIGHTS_SOMATIC_PURITY = {'f_purity_correlation':   0.50,
                           'f_clone_specific_proxy': 0.20,
                           'f_spatial_clustering':   0.30}

GERMLINE_FEATURES = ['f_spatial_uniformity', 'f_global_prevalence',
                     'f_purity_independence']
SOMATIC_FEATURES  = ['f_purity_correlation', 'f_spatial_clustering',
                     'f_clone_specific_proxy', 'f_clone_enrichment',
                     'f_cnv_consistency']
ALL_FEATURES = GERMLINE_FEATURES + SOMATIC_FEATURES

FEATURE_LABELS = {
    'f_spatial_uniformity':   'Spatial\nUniformity (α)',
    'f_global_prevalence':    'Global\nPrevalence (β)',
    'f_purity_independence':  'Purity\nIndependence (γ)',
    'f_purity_correlation':   'Purity\nCorrelation (δ)',
    'f_spatial_clustering':   'Spatial\nClustering (ζ)',
    'f_clone_specific_proxy': 'Clone-Specific\nProxy (ε)',
    'f_clone_enrichment':     'Clone\nEnrichment (η)',
    'f_cnv_consistency':      'CNV\nConsistency (θ)',
}

CLASS_ORDER  = ['germline', 'somatic', 'ambiguous']
CLASS_COLORS = {'germline': '#3498db', 'somatic': '#e74c3c', 'ambiguous': '#95a5a6'}
CLASS_NUMERIC = {'germline': 0, 'somatic': 2, 'ambiguous': 1}  # ordinal for correlation

DPI = 180


# ── Data loading ─────────────────────────────────────────────────────────────

def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', na_values=['NA', 'nan', ''])
    print(f"Loaded {len(df)} variants from {path}")
    print(f"  Columns : {list(df.columns)}")

    # Coerce numeric columns
    numeric_cols = ['germline_score', 'somatic_score'] + ALL_FEATURES
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    race_counts = df['race'].value_counts().to_dict() if 'race' in df.columns else {}
    print(f"  Race     : {race_counts}")
    cls_counts  = df['classification'].value_counts().to_dict() if 'classification' in df.columns else {}
    print(f"  Class    : {cls_counts}")
    return df


def get_denovo(df: pd.DataFrame) -> pd.DataFrame:
    """Return only denovo variants that have real feature scores."""
    if 'race' in df.columns:
        dn = df[df['race'] == 'denovo'].copy()
    else:
        dn = df.copy()

    # Drop rows where ALL features are NaN
    feat_cols = [c for c in ALL_FEATURES if c in dn.columns]
    dn = dn.dropna(subset=feat_cols, how='all')
    print(f"  Denovo variants with feature scores: {len(dn)}")
    return dn


def available_features(df: pd.DataFrame) -> list:
    """Return features present in the file and not entirely NaN."""
    avail = []
    for f in ALL_FEATURES:
        if f in df.columns and df[f].notna().sum() > 5:
            avail.append(f)
    return avail


# ── 1. Violin / strip plots ───────────────────────────────────────────────────

def plot_feature_distributions(dn: pd.DataFrame, feats: list,
                                output_dir: str, title_prefix: str):
    """
    One figure with one subplot per feature.
    Each subplot: violin + overlaid strip plot, x=class, y=feature value.
    """
    n = len(feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.0),
                             squeeze=False)

    for idx, feat in enumerate(feats):
        ax = axes[idx // ncols][idx % ncols]

        plot_data = []
        for cls in CLASS_ORDER:
            sub = dn[dn['classification'] == cls][feat].dropna()
            for v in sub:
                plot_data.append({'class': cls, 'value': v})

        if not plot_data:
            ax.set_visible(False)
            continue

        pdf = pd.DataFrame(plot_data)

        # Violin
        parts = ax.violinplot(
            [pdf[pdf['class'] == cls]['value'].values for cls in CLASS_ORDER],
            positions=range(len(CLASS_ORDER)),
            showmedians=True, showextrema=True, widths=0.7
        )
        for pc, cls in zip(parts['bodies'], CLASS_ORDER):
            pc.set_facecolor(CLASS_COLORS[cls])
            pc.set_alpha(0.55)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

        # Strip
        for i, cls in enumerate(CLASS_ORDER):
            sub = pdf[pdf['class'] == cls]['value'].values
            jitter = np.random.uniform(-0.08, 0.08, size=len(sub))
            ax.scatter(i + jitter, sub,
                       color=CLASS_COLORS[cls], alpha=0.35, s=8, zorder=3)

        # Median labels
        for i, cls in enumerate(CLASS_ORDER):
            sub = pdf[pdf['class'] == cls]['value'].dropna()
            if len(sub):
                ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.05,
                        f'n={len(sub)}', ha='center', va='top', fontsize=7,
                        color='dimgray')

        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER], fontsize=9)
        ax.set_ylabel('Feature value', fontsize=9)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_patches = [Patch(color=CLASS_COLORS[c], label=c.capitalize())
                      for c in CLASS_ORDER]
    fig.legend(handles=legend_patches, loc='lower right',
               fontsize=10, framealpha=0.9)

    suptitle = f"{title_prefix} — Feature Distributions by Class (denovo only)"
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 2. Correlation analysis ───────────────────────────────────────────────────

def correlation_analysis(dn: pd.DataFrame, feats: list,
                         germline_threshold: float, somatic_threshold: float
                         ) -> pd.DataFrame:
    """
    For each feature compute:
      - Spearman rho with numeric class label (germline=0, ambiguous=1, somatic=2)
      - Spearman rho with germline_score composite
      - Spearman rho with somatic_score composite
      - Kruskal-Wallis H statistic and p-value across the three classes
        (nonparametric ANOVA — tests whether distributions differ at all)
      - Cohen's d between germline and somatic groups (effect size)
    """
    dn = dn.copy()
    dn['class_numeric'] = dn['classification'].map(CLASS_NUMERIC)

    records = []
    for feat in feats:
        col = dn[feat].dropna()
        idx = col.index

        row = {'feature': feat}

        # vs class label
        if len(idx) > 5 and dn.loc[idx, 'class_numeric'].notna().sum() > 5:
            rho, pval = stats.spearmanr(dn.loc[idx, feat],
                                        dn.loc[idx, 'class_numeric'])
            row['rho_vs_class']  = rho
            row['p_vs_class']    = pval
        else:
            row['rho_vs_class'] = np.nan
            row['p_vs_class']   = np.nan

        # vs composite scores
        for score_col in ['germline_score', 'somatic_score']:
            if score_col in dn.columns:
                common = dn[[feat, score_col]].dropna()
                if len(common) > 5:
                    rho, pval = stats.spearmanr(common[feat], common[score_col])
                    row[f'rho_vs_{score_col}'] = rho
                    row[f'p_vs_{score_col}']   = pval
                else:
                    row[f'rho_vs_{score_col}'] = np.nan
                    row[f'p_vs_{score_col}']   = np.nan

        # Kruskal-Wallis across three classes
        groups = [dn[dn['classification'] == cls][feat].dropna().values
                  for cls in CLASS_ORDER]
        groups = [g for g in groups if len(g) >= 3]
        # Kruskal requires at least some variance — skip if all values across
        # all groups are identical (happens with sparse real data, e.g. all
        # spatial_clustering == 0.0 in the somatic group with very few variants)
        all_vals = np.concatenate(groups) if groups else np.array([])
        if len(groups) >= 2 and len(np.unique(all_vals)) > 1:
            try:
                h, p = stats.kruskal(*groups)
                row['kruskal_H'] = h
                row['kruskal_p'] = p
            except ValueError:
                row['kruskal_H'] = np.nan
                row['kruskal_p'] = np.nan
        else:
            row['kruskal_H'] = np.nan
            row['kruskal_p'] = np.nan

        # Cohen's d: germline vs somatic
        g = dn[dn['classification'] == 'germline'][feat].dropna().values
        s = dn[dn['classification'] == 'somatic'][feat].dropna().values
        if len(g) >= 2 and len(s) >= 2:
            pooled_std = np.sqrt((np.var(g, ddof=1) + np.var(s, ddof=1)) / 2)
            row['cohens_d_germ_vs_som'] = (np.mean(g) - np.mean(s)) / pooled_std \
                if pooled_std > 0 else np.nan
        else:
            row['cohens_d_germ_vs_som'] = np.nan

        # Per-class medians for the summary table
        for cls in CLASS_ORDER:
            sub = dn[dn['classification'] == cls][feat].dropna()
            row[f'median_{cls}'] = sub.median() if len(sub) else np.nan

        records.append(row)

    return pd.DataFrame(records).set_index('feature')


def plot_correlation_bar(corr_df: pd.DataFrame, output_dir: str,
                          title_prefix: str):
    """
    Horizontal bar chart of |Spearman rho| vs class label,
    coloured by direction (positive = somatic signal, negative = germline signal).
    """
    df = corr_df[['rho_vs_class']].dropna().copy()
    df['abs_rho'] = df['rho_vs_class'].abs()
    df['direction'] = df['rho_vs_class'].apply(
        lambda x: 'Somatic signal' if x > 0 else 'Germline signal')
    df = df.sort_values('abs_rho', ascending=True)

    colors = ['#e74c3c' if d == 'Somatic signal' else '#3498db'
              for d in df['direction']]
    labels = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in df.index]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 2))
    bars = ax.barh(labels, df['abs_rho'], color=colors, edgecolor='white',
                   linewidth=0.5, height=0.6)

    # Annotate rho value
    for bar, rho in zip(bars, df['rho_vs_class']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{rho:+.3f}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('|Spearman ρ| with class label\n(germline=0, ambiguous=1, somatic=2)',
                  fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label='|ρ|=0.3 reference')
    legend_patches = [Patch(color='#e74c3c', label='Somatic signal (+ρ)'),
                      Patch(color='#3498db', label='Germline signal (−ρ)')]
    ax.legend(handles=legend_patches, fontsize=9, loc='lower right')
    ax.set_title(f'{title_prefix}\nSpearman ρ: Feature vs Classification',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    plt.tight_layout()
    out = os.path.join(output_dir, 'feature_correlation_bar.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: str,
                              title_prefix: str):
    """
    Heatmap: features × {rho_vs_class, rho_vs_germline_score, rho_vs_somatic_score}.
    """
    cols = ['rho_vs_class', 'rho_vs_germline_score', 'rho_vs_somatic_score']
    cols = [c for c in cols if c in corr_df.columns]
    plot_df = corr_df[cols].copy()
    plot_df.index = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in plot_df.index]
    col_labels = {'rho_vs_class': 'vs Class label',
                  'rho_vs_germline_score': 'vs Germline score',
                  'rho_vs_somatic_score':  'vs Somatic score'}
    plot_df.columns = [col_labels.get(c, c) for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(plot_df) + 2))
    sns.heatmap(plot_df.astype(float), annot=True, fmt='.3f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Spearman ρ'})
    ax.set_title(f'{title_prefix}\nSpearman ρ — Feature × Score/Class',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()

    out = os.path.join(output_dir, 'feature_correlation_heatmap.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 3. Class mean/median heatmap ──────────────────────────────────────────────

def plot_class_mean_heatmap(dn: pd.DataFrame, feats: list,
                             output_dir: str, title_prefix: str):
    """
    Z-scored per-feature class means as a heatmap.
    Rows = features, columns = classes.
    """
    rows = {}
    for feat in feats:
        row = {}
        for cls in CLASS_ORDER:
            sub = dn[dn['classification'] == cls][feat].dropna()
            row[cls] = sub.mean() if len(sub) else np.nan
        rows[feat] = row

    mean_df = pd.DataFrame(rows).T  # features × classes
    mean_df.index = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in mean_df.index]

    # Z-score across classes per feature
    z_df = mean_df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-9), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(feats) + 2.5),
                              gridspec_kw={'width_ratios': [1, 1]})

    sns.heatmap(mean_df.astype(float), annot=True, fmt='.3f',
                cmap='Blues', ax=axes[0], linewidths=0.5,
                cbar_kws={'label': 'Mean feature value'})
    axes[0].set_title('Mean feature value per class', fontsize=10, fontweight='bold')

    sns.heatmap(z_df.astype(float), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=axes[1], linewidths=0.5,
                cbar_kws={'label': 'Z-score'})
    axes[1].set_title('Z-scored means\n(highlights relative differences)', fontsize=10,
                      fontweight='bold')

    fig.suptitle(f'{title_prefix} — Per-class Feature Averages',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(output_dir, 'feature_class_heatmap.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 4. Pairwise scatter matrix ────────────────────────────────────────────────

def plot_pairwise_scatter(dn: pd.DataFrame, feats: list,
                           output_dir: str, title_prefix: str):
    """
    Seaborn PairGrid on available features, coloured by class.
    Upper = scatter, diagonal = KDE, lower = KDE contours.
    Only uses up to 6 features to keep the figure manageable.
    Constant-valued features (zero variance) are dropped before plotting
    because seaborn KDE fails on degenerate distributions.
    """
    plot_feats = feats[:6]  # cap at 6 for readability
    sub = dn[plot_feats + ['classification']].dropna()

    if len(sub) < 10:
        print("  Skipping pairwise scatter (too few complete rows)")
        return

    # Drop features with zero variance (all-identical values) — KDE crashes on these
    varied_feats = [f for f in plot_feats if sub[f].nunique() > 1]
    if len(varied_feats) < 2:
        print("  Skipping pairwise scatter (fewer than 2 features with variance)")
        return
    if len(varied_feats) < len(plot_feats):
        dropped = set(plot_feats) - set(varied_feats)
        print(f"  Dropped constant features from pairwise scatter: {dropped}")
    plot_feats = varied_feats
    sub = sub[plot_feats + ['classification']]

    palette = {cls: CLASS_COLORS[cls] for cls in CLASS_ORDER if cls in sub['classification'].unique()}

    g = sns.PairGrid(sub, vars=plot_feats, hue='classification',
                     palette=palette, diag_sharey=False, height=2.2)
    g.map_upper(sns.scatterplot, alpha=0.4, s=12, edgecolor='none')
    g.map_diag(sns.kdeplot, fill=True, alpha=0.45)
    g.map_lower(sns.kdeplot, alpha=0.55, levels=4)
    g.add_legend(title='Class', fontsize=9)

    # Rename axes labels
    for ax in g.axes.flatten():
        if ax is None:
            continue
        xl = ax.get_xlabel()
        yl = ax.get_ylabel()
        if xl in FEATURE_LABELS:
            ax.set_xlabel(FEATURE_LABELS[xl].replace('\n', ' '), fontsize=7)
        if yl in FEATURE_LABELS:
            ax.set_ylabel(FEATURE_LABELS[yl].replace('\n', ' '), fontsize=7)

    g.figure.suptitle(f'{title_prefix} — Pairwise Feature Scatter (denovo)',
                      y=1.01, fontsize=11, fontweight='bold')

    out = os.path.join(output_dir, 'feature_pairwise_scatter.png')
    g.figure.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 5. Ablation analysis ──────────────────────────────────────────────────────

def _recompute_scores(row: pd.Series, zeroed_feat: str,
                      germline_feats: list, somatic_feats: list,
                      use_clone: bool) -> dict:
    """Recompute composite scores with one feature zeroed."""
    wg = WEIGHTS_GERMLINE
    ws = WEIGHTS_SOMATIC_CLONE if use_clone else WEIGHTS_SOMATIC_PURITY

    g_score = 0.0
    for f, w in wg.items():
        val = 0.0 if f == zeroed_feat else (row.get(f, 0.0) or 0.0)
        g_score += w * val

    s_score = 0.0
    for f, w in ws.items():
        val = 0.0 if f == zeroed_feat else (row.get(f, 0.0) or 0.0)
        s_score += w * val

    return {'germline_score': g_score, 'somatic_score': s_score}


def _classify(g: float, s: float, gt: float, st: float) -> str:
    if g > gt and s < st:
        return 'germline'
    elif s > st and g < gt:
        return 'somatic'
    return 'ambiguous'


def ablation_analysis(dn: pd.DataFrame, feats: list,
                       germline_threshold: float, somatic_threshold: float
                       ) -> pd.DataFrame:
    """
    For each feature: zero it out → recompute scores → count variants that
    change classification.  Returns a DataFrame sorted by impact.
    """
    use_clone = ('f_clone_enrichment' in feats and
                 dn['f_clone_enrichment'].notna().sum() > 5)

    g_feats = [f for f in GERMLINE_FEATURES if f in feats]
    s_feats = [f for f in SOMATIC_FEATURES  if f in feats]

    # Baseline classification from scores already in the file
    baseline = dn.apply(
        lambda r: _classify(r['germline_score'], r['somatic_score'],
                            germline_threshold, somatic_threshold), axis=1)

    records = []
    for feat in feats:
        ablated = dn.apply(
            lambda r: _classify(
                _recompute_scores(r, feat, g_feats, s_feats, use_clone)['germline_score'],
                _recompute_scores(r, feat, g_feats, s_feats, use_clone)['somatic_score'],
                germline_threshold, somatic_threshold), axis=1)

        flipped = (baseline != ablated).sum()
        total   = len(baseline)

        # Breakdown of flip directions
        flip_to_som = ((baseline == 'germline') & (ablated == 'somatic')).sum()
        flip_to_germ= ((baseline == 'somatic')  & (ablated == 'germline')).sum()
        flip_to_amb = ((baseline != 'ambiguous') & (ablated == 'ambiguous')).sum()

        records.append({
            'feature':             feat,
            'n_flipped':           int(flipped),
            'flip_rate_%':         round(100 * flipped / total, 2),
            'flips_to_somatic':    int(flip_to_som),
            'flips_to_germline':   int(flip_to_germ),
            'flips_to_ambiguous':  int(flip_to_amb),
        })

    abl_df = pd.DataFrame(records).set_index('feature')
    abl_df = abl_df.sort_values('n_flipped', ascending=False)
    return abl_df


def plot_ablation_bar(abl_df: pd.DataFrame, output_dir: str, title_prefix: str):
    df = abl_df.sort_values('n_flipped', ascending=True).copy()
    labels = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in df.index]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 2))

    # Stacked bar: flips_to_somatic | flips_to_germline | flips_to_ambiguous
    left = np.zeros(len(df))
    for col, color, label in [
        ('flips_to_somatic',   '#e74c3c', '→ Somatic'),
        ('flips_to_germline',  '#3498db', '→ Germline'),
        ('flips_to_ambiguous', '#95a5a6', '→ Ambiguous'),
    ]:
        vals = df[col].values
        ax.barh(labels, vals, left=left, color=color, label=label,
                edgecolor='white', linewidth=0.4, height=0.6)
        left += vals

    # Total label
    for i, (total, rate) in enumerate(zip(df['n_flipped'], df['flip_rate_%'])):
        ax.text(left[i] + 0.3, i, f' {total} ({rate}%)',
                va='center', ha='left', fontsize=9)

    ax.set_xlabel('Number of variants that change classification\nwhen feature is zeroed out',
                  fontsize=10)
    ax.set_title(f'{title_prefix}\nAblation: Feature Importance by Classification Impact',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_xlim(0, max(left) * 1.25 if max(left) > 0 else 1)

    plt.tight_layout()
    out = os.path.join(output_dir, 'feature_ablation.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 6. Redundancy / collinearity check ───────────────────────────────────────

def plot_feature_correlation_matrix(dn: pd.DataFrame, feats: list,
                                     output_dir: str, title_prefix: str):
    """
    Spearman rho between every pair of features.
    Highlights |r| > 0.7 to flag redundancy.
    """
    sub = dn[feats].dropna()
    if len(sub) < 5:
        print("  Skipping redundancy matrix (too few complete rows)")
        return

    rho_matrix = sub.corr(method='spearman')
    rho_matrix.index   = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in rho_matrix.index]
    rho_matrix.columns = [FEATURE_LABELS.get(f, f).replace('\n', ' ') for f in rho_matrix.columns]

    mask = np.zeros_like(rho_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # upper triangle only

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(rho_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, square=True,
                cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8})

    # Highlight highly correlated cells
    n = len(rho_matrix)
    for i in range(n):
        for j in range(i):
            val = rho_matrix.iloc[i, j]
            if abs(val) > 0.7:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                            fill=False, edgecolor='gold',
                                            linewidth=2.5, zorder=5))

    ax.set_title(f'{title_prefix}\nFeature–Feature Spearman ρ (lower triangle)\n'
                 f'Gold border = |ρ| > 0.7 (potential redundancy)',
                 fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    out = os.path.join(output_dir, 'feature_redundancy_matrix.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 7. Summary table ──────────────────────────────────────────────────────────

def build_summary_table(corr_df: pd.DataFrame, abl_df: pd.DataFrame,
                         output_dir: str) -> pd.DataFrame:
    """Join correlation stats + ablation impact into one summary TSV."""
    keep_corr = ['rho_vs_class', 'p_vs_class', 'kruskal_H', 'kruskal_p',
                 'cohens_d_germ_vs_som',
                 'median_germline', 'median_somatic', 'median_ambiguous']
    keep_corr = [c for c in keep_corr if c in corr_df.columns]

    keep_abl = ['n_flipped', 'flip_rate_%',
                'flips_to_somatic', 'flips_to_germline', 'flips_to_ambiguous']

    summary = corr_df[keep_corr].join(abl_df[keep_abl], how='outer')
    summary.index.name = 'feature'
    summary = summary.reset_index()
    summary['feature_label'] = summary['feature'].map(
        lambda f: FEATURE_LABELS.get(f, f).replace('\n', ' '))

    # Rank features by ablation impact (primary) + |rho| (secondary)
    summary['abs_rho_class'] = summary['rho_vs_class'].abs()
    summary = summary.sort_values(['n_flipped', 'abs_rho_class'],
                                   ascending=[False, False])

    out = os.path.join(output_dir, 'feature_summary.tsv')
    summary.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f"Saved: {out}")

    # Pretty print
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS SUMMARY  (sorted by ablation impact)")
    print("=" * 80)
    cols_print = ['feature_label', 'rho_vs_class', 'kruskal_p',
                  'cohens_d_germ_vs_som', 'n_flipped', 'flip_rate_%']
    cols_print = [c for c in cols_print if c in summary.columns]
    print(summary[cols_print].to_string(index=False))
    print("=" * 80 + "\n")

    return summary


# ── 8. Composite score distributions (defined vs denovo comparison) ───────────

def plot_composite_score_distributions(df: pd.DataFrame,
                                        output_dir: str, title_prefix: str,
                                        germline_threshold: float,
                                        somatic_threshold: float):
    """
    Two panels side by side:
    Left  — KDE of germline_score, split defined vs denovo
    Right — KDE of somatic_score,  split defined vs denovo
    Shows whether defined variants actually cluster near (1, 0) after
    the canonical override was removed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    race_styles = {'defined': ('#2ecc71', '-'),
                   'denovo':  ('#9b59b6', '--'),
                   'unknown': ('#95a5a6', ':')}

    for ax, score_col, threshold, label in [
        (axes[0], 'germline_score', germline_threshold, 'Germline Score'),
        (axes[1], 'somatic_score',  somatic_threshold,  'Somatic Score'),
    ]:
        for race, (color, ls) in race_styles.items():
            sub = df[df['race'] == race][score_col].dropna() if 'race' in df.columns \
                  else df[score_col].dropna()
            if len(sub) < 3:
                continue
            sns.kdeplot(sub, ax=ax, color=color, linestyle=ls,
                        linewidth=2, label=f'{race.capitalize()} (n={len(sub)})',
                        fill=True, alpha=0.12)

        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'Threshold ({threshold})')
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=':')

    fig.suptitle(f'{title_prefix} — Composite Score Distributions by Race',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(output_dir, 'composite_score_distributions.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Feature analysis for SPARCAL spatial SNV filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_analysis.py \\
    --input /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt \\
    --output_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/feature_analysis/ \\
    --title "P4 Tumor Section 1"
        """
    )
    parser.add_argument('--input', required=True,
                        help='Path to all_variant_scores.txt from run_spatial_filter_enhanced.py')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save all output figures and tables')
    parser.add_argument('--title', default='SPARCAL',
                        help='Title prefix for all figures (default: SPARCAL)')
    parser.add_argument('--germline_threshold', type=float,
                        default=DEFAULT_GERMLINE_THRESHOLD,
                        help=f'Germline score threshold (default: {DEFAULT_GERMLINE_THRESHOLD})')
    parser.add_argument('--somatic_threshold', type=float,
                        default=DEFAULT_SOMATIC_THRESHOLD,
                        help=f'Somatic score threshold (default: {DEFAULT_SOMATIC_THRESHOLD})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SPARCAL Feature Analysis")
    print("=" * 60)

    # Load
    df = load_scores(args.input)
    dn = get_denovo(df)
    feats = available_features(dn)
    print(f"  Available features : {feats}")

    if not feats:
        print("ERROR: No feature columns found in input file. "
              "Make sure to use all_variant_scores.txt (not the per-class files).")
        sys.exit(1)

    print("\n── 1. Feature distribution plots ──")
    try:
        plot_feature_distributions(dn, feats, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 1 (feature distributions): {e}")

    print("\n── 2. Correlation analysis ──")
    corr_df = None
    try:
        corr_df = correlation_analysis(dn, feats,
                                        args.germline_threshold,
                                        args.somatic_threshold)
        plot_correlation_bar(corr_df, args.output_dir, args.title)
        plot_correlation_heatmap(corr_df, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 2 (correlation analysis): {e}")

    print("\n── 3. Class mean heatmap ──")
    try:
        plot_class_mean_heatmap(dn, feats, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 3 (class mean heatmap): {e}")

    print("\n── 4. Pairwise scatter matrix ──")
    try:
        plot_pairwise_scatter(dn, feats, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 4 (pairwise scatter): {e}")

    print("\n── 5. Ablation analysis ──")
    abl_df = None
    try:
        abl_df = ablation_analysis(dn, feats,
                                    args.germline_threshold,
                                    args.somatic_threshold)
        plot_ablation_bar(abl_df, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 5 (ablation analysis): {e}")

    print("\n── 6. Feature redundancy matrix ──")
    try:
        plot_feature_correlation_matrix(dn, feats, args.output_dir, args.title)
    except Exception as e:
        print(f"  ERROR in step 6 (redundancy matrix): {e}")

    print("\n── 7. Summary table ──")
    try:
        if corr_df is not None and abl_df is not None:
            build_summary_table(corr_df, abl_df, args.output_dir)
        else:
            print("  Skipping summary table — upstream step(s) failed.")
    except Exception as e:
        print(f"  ERROR in step 7 (summary table): {e}")

    print("\n── 8. Composite score distributions (defined vs denovo) ──")
    try:
        plot_composite_score_distributions(df, args.output_dir, args.title,
                                            args.germline_threshold,
                                            args.somatic_threshold)
    except Exception as e:
        print(f"  ERROR in step 8 (composite score distributions): {e}")

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
