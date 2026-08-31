#!/usr/bin/env python
"""SparcalNet evaluation figure -- closes referee finding C3 / PAPER_WORK P0-2 scope
(the SparcalNet-only slice of it; the five-configuration whole-pipeline ablation in
P0-2 is a separate, larger effort and is NOT reproduced here).

WHY THIS SCRIPT EXISTS
  The manuscript names SparcalNet (a genotype classifier) but never evaluates it: no
  training-set description, no label source, no split, no cross-validation, no
  performance number, no ablation (PAPER_PLAN.md Sec 8.2 C3). This script builds the
  missing evaluation directly from the real per-sample VCFs, using the ACTUAL
  TrainingSetBuilder/FeatureExtractor classes imported from
  scripts/4_classifier/run_sparcal_net.py (not a re-implementation), and adds what
  that script never does: stratified 5-fold CV, permutation importance, a domain-shift
  check (panel-defined vs de novo candidates), and an ablation against a BAF/depth-only
  proxy for "the sequence-error model's own call".

CRITICAL CODE-PROVENANCE FINDING (verify before reading any number below):
  There are TWO independent NN-classifier implementations in this repo:
    - scripts/4_classifier/run_sparcal_net.py      hidden_layer_sizes=(100, 50), max_iter=500
    - scripts/4_classifier/run_supplimentary_models.py  hidden_layer_sizes=(64, 32), sklearn
      defaults for solver/max_iter (i.e. max_iter=200, undocumented in that file)
  The root CLAUDE.md pipeline table (Steps line + Architecture table) says step 5 (the
  NN classifier) is run via `run_supplimentary_models.py`, "not run_sparcal_net.py --
  that has the no_variance label-encoder bug". The manuscript's Methods text ("hidden
  layers of 64 and 32 neurons") matches run_supplimentary_models.py EXACTLY. **The
  manuscript's architecture description is therefore CORRECT for the script CLAUDE.md
  says is canonical -- this is NOT a case of the paper contradicting the code.** The
  live problem is narrower and still real: the Methods text names neither script, so a
  reader with only the manuscript (or even a reader with repo access but no CLAUDE.md
  pointer) cannot tell WHICH of the two implementations -- one of them flagged buggy
  and unused -- produced the results being described. This script evaluates
  run_sparcal_net.py because that is the file the evaluation task named and because its
  saved outputs (output_VCFs/SPARCALNet/) exist on disk for at least two samples,
  confirming it has actually been run historically -- but per CLAUDE.md it is NOT the
  canonical classifier, and this evaluation should not be quoted as "SparcalNet
  performance" without that caveat. See RESULTS.md "Architecture provenance" section
  for the full account.

INPUTS (real files, no synthetic data anywhere):
  For each of 4 tumor samples (P4_TUMOR/1, P6_TUMOR/1, DCIS/1, DCIS/2), quality
  filter baseQ0mapQ0:
    data/{sample}/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz       (SOURCE=BEAGLE, panel-defined)
    data/{sample}/output_VCFs/SeqErrModel/baseQ0mapQ0/sequence_no_error.vcf.gz (SOURCE=seq_no_err, de novo candidates
                                                                                 -- built from all_filtered_out.vcf.gz,
                                                                                 i.e. non-panel sites)
  Labels are the GT field already present in those same files (parsed by
  TrainingSetBuilder._extract_labels_from_vcf, called unmodified).

OUTPUTS:
  Figure   SPARCAL_pnas_2026/figs/v5_2026-08-23/fig_sparcalnet_eval[_preview].{pdf,png}
  Tables   snv_calling/data/sparcalnet_eval_2026-08-23/*.csv
  Report   snv_calling/data/sparcalnet_eval_2026-08-23/RESULTS.md   (written by this script)

RUN (env snv_caller; ~15-30 min wall time on 16 cores, CPU only):
  conda activate snv_caller
  python scripts/postanalyze/fig_sparcalnet_eval_v5.py
"""
import os
import sys
import time
import json
import importlib
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
PAPER = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026"
FONT_DIR = os.path.join(PAPER, "fonts", "arial")
FIG_DIR = os.path.join(PAPER, "figs", "v5_2026-08-23")
DATA_DIR = os.path.join(PROJECT, "data", "sparcalnet_eval_2026-08-23")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CLASSIFIER_DIR = os.path.join(PROJECT, "scripts", "4_classifier")
sys.path.insert(0, CLASSIFIER_DIR)
rsn = importlib.import_module("run_sparcal_net")  # the ACTUAL script under evaluation


def configure_required_font():
    """Require genuine Arial for a final-basename manuscript export."""
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local = font_manager.findSystemFonts(fontpaths=[FONT_DIR]) if os.path.isdir(FONT_DIR) else []
    explicit_family = (font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
                       if explicit and os.path.dirname(explicit) else [])
    candidates = ([explicit] + explicit_family) if explicit else local + font_manager.findSystemFonts()
    matches = []
    for path in dict.fromkeys(path for path in candidates if path):
        if not os.path.exists(path):
            continue
        try:
            family = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if family.casefold() == "arial":
            matches.append(path)
    if matches:
        for path in matches:
            font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Arial"
        return True, matches[0]
    plt.rcParams["font.family"] = "Nimbus Sans"
    return False, None


HAS_ARIAL, ARIAL_PATH = configure_required_font()

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"

CLASS_ORDER = ["no_variance", "heterozygous", "homozygous"]
CLASS_COLOR = {"no_variance": MUTED, "heterozygous": SPARCAL_L, "homozygous": SPARCAL_C}
SOURCE_COLOR = {"BEAGLE": MONO_C, "seq_no_err": SPARCAL_C}
SOURCE_LABEL = {"BEAGLE": "panel-defined\n(BEAGLE)", "seq_no_err": "de novo\n(non-panel)"}

QF = "baseQ0mapQ0"
MAX_TRAINING_SAMPLES = 90000  # matches TrainingSetBuilder's own default -- kept identical
SEED = 42
N_SPLITS = 5

SAMPLES = {
    "P4":    dict(dataset="P4_TUMOR", section_id="1"),
    "P6":    dict(dataset="P6_TUMOR", section_id="1"),
    "DCIS1": dict(dataset="DCIS",     section_id="1"),
    "DCIS2": dict(dataset="DCIS",     section_id="2"),
}
SAMPLE_ORDER = ["P4", "P6", "DCIS1", "DCIS2"]


# ---------------------------------------------------------------------------
# 1. Build each sample's labeled feature table using the REAL classes
# ---------------------------------------------------------------------------
def build_sample_frame(sample_key, cfg):
    t0 = time.time()
    builder = rsn.TrainingSetBuilder(
        dataset_name=cfg["dataset"], quality_filter=QF, section_id=cfg["section_id"],
        max_training_samples=MAX_TRAINING_SAMPLES,
    )
    extractor = rsn.FeatureExtractor()
    beagle_features = extractor.extract_features(builder.beagle_vcf, source="BEAGLE")
    seq_features = extractor.extract_features(builder.seq_no_error_vcf, source="seq_no_err")
    all_features = pd.concat([beagle_features, seq_features], ignore_index=True)

    labels = builder._extract_labels_from_vcf()
    if len(labels) != len(all_features):
        raise ValueError(
            f"[{sample_key}] label count ({len(labels)}) != feature count ({len(all_features)}) "
            "-- the same mismatch guard run_sparcal_net.py itself raises on."
        )
    all_features["label"] = labels
    all_features = all_features.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[{sample_key}] {len(all_features)} labeled variants "
          f"(BEAGLE={int((all_features.SOURCE == 'BEAGLE').sum())}, "
          f"seq_no_err={int((all_features.SOURCE == 'seq_no_err').sum())}) "
          f"in {time.time() - t0:.1f}s")
    return sample_key, all_features, builder.beagle_vcf, builder.seq_no_error_vcf


def summarize_class_balance(frames):
    rows = []
    for sample_key, df in frames.items():
        for stage, sub in [("full_available", df), ("cv_modeling_subsample", None)]:
            if stage == "cv_modeling_subsample":
                continue  # filled in later once the subsample is drawn
            n = len(sub)
            counts = sub["label"].value_counts()
            majority_class = counts.idxmax()
            majority_frac = counts.max() / n
            for cls in CLASS_ORDER:
                c = int(counts.get(cls, 0))
                rows.append(dict(sample=sample_key, stage=stage, source="combined", label=cls,
                                  count=c, fraction=c / n, n_total=n,
                                  majority_class=majority_class, majority_fraction=majority_frac))
            for src in ["BEAGLE", "seq_no_err"]:
                sub_src = sub[sub.SOURCE == src]
                if len(sub_src) == 0:
                    continue
                counts_src = sub_src["label"].value_counts()
                for cls in CLASS_ORDER:
                    c = int(counts_src.get(cls, 0))
                    rows.append(dict(sample=sample_key, stage=stage, source=src, label=cls,
                                      count=c, fraction=c / len(sub_src), n_total=len(sub_src),
                                      majority_class=counts_src.idxmax(),
                                      majority_fraction=counts_src.max() / len(sub_src)))
    return pd.DataFrame(rows)


def draw_modeling_subsample(df, seed=SEED, cap=MAX_TRAINING_SAMPLES):
    """Reproducible stand-in for TrainingSetBuilder's own (unseeded!) np.random.choice
    downsample -- see RESULTS.md 'Reproducibility note'."""
    if len(df) <= cap:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), cap, replace=False)
    return df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Per-fold worker: primary model (code arch), comparison model (paper arch),
#    baselines, permutation importance, BAF/DP-only ablation proxy.
# ---------------------------------------------------------------------------
def eval_predictions(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, macro_f1, acc


def run_fold(sample_key, X, y, source, feature_cols, train_idx, test_idx, fold_id):
    scaler = StandardScaler().fit(X[train_idx])
    Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    ytr, yte = y[train_idx], y[test_idx]
    src_te = source[test_idx]

    rows = []

    # --- primary model: EXACT hyperparameters hardcoded in run_sparcal_net.py ---
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                         max_iter=500, random_state=SEED)
    mlp.fit(Xtr, ytr)
    pred = mlp.predict(Xte)
    proba = mlp.predict_proba(Xte)
    max_prob = proba.max(axis=1)
    correct = (pred == yte)
    p, r, f1, macro_f1, acc = eval_predictions(yte, pred)
    for i, cls in enumerate(CLASS_ORDER):
        rows.append(dict(sample=sample_key, fold=fold_id, model="code_arch_100_50",
                          metric_class=cls, precision=p[i], recall=r[i], f1=f1[i],
                          macro_f1=macro_f1, accuracy=acc, n_test=len(yte)))
    rows.append(dict(sample=sample_key, fold=fold_id, model="code_arch_100_50",
                      metric_class="macro", precision=np.nan, recall=np.nan,
                      f1=macro_f1, macro_f1=macro_f1, accuracy=acc, n_test=len(yte)))

    # permutation importance on this held-out fold, primary model only
    perm = permutation_importance(mlp, Xte, yte, scoring="f1_macro", n_repeats=5,
                                   random_state=SEED + fold_id, n_jobs=1)
    perm_rows = [dict(sample=sample_key, fold=fold_id, feature=feat,
                       importance_mean=perm.importances_mean[i],
                       importance_std=perm.importances_std[i])
                 for i, feat in enumerate(feature_cols)]

    # out-of-fold records for the domain-shift panel
    oof_rows = [dict(sample=sample_key, fold=fold_id, source=s, true=t, pred=pd_, correct=c,
                      max_prob=mp)
                for s, t, pd_, c, mp in zip(src_te, yte, pred, correct, max_prob)]

    # --- comparison model: paper-stated architecture, run_supplimentary_models.py's
    #     ACTUAL call signature (hidden_layer_sizes=(64,32), sklearn defaults for the
    #     rest -> max_iter=200) ---
    mlp_paper = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", random_state=SEED)
    mlp_paper.fit(Xtr, ytr)
    pred_p = mlp_paper.predict(Xte)
    p2, r2, f12, macro_f1_2, acc2 = eval_predictions(yte, pred_p)
    for i, cls in enumerate(CLASS_ORDER):
        rows.append(dict(sample=sample_key, fold=fold_id, model="paper_arch_64_32",
                          metric_class=cls, precision=p2[i], recall=r2[i], f1=f12[i],
                          macro_f1=macro_f1_2, accuracy=acc2, n_test=len(yte)))
    rows.append(dict(sample=sample_key, fold=fold_id, model="paper_arch_64_32",
                      metric_class="macro", precision=np.nan, recall=np.nan,
                      f1=macro_f1_2, macro_f1=macro_f1_2, accuracy=acc2, n_test=len(yte)))

    # --- majority-class baseline ---
    maj_cls = pd.Series(ytr).value_counts().idxmax()
    maj_pred = np.full(len(yte), maj_cls, dtype=object)
    p3, r3, f13, macro_f1_3, acc3 = eval_predictions(yte, maj_pred)
    for i, cls in enumerate(CLASS_ORDER):
        rows.append(dict(sample=sample_key, fold=fold_id, model="baseline_majority",
                          metric_class=cls, precision=p3[i], recall=r3[i], f1=f13[i],
                          macro_f1=macro_f1_3, accuracy=acc3, n_test=len(yte)))
    rows.append(dict(sample=sample_key, fold=fold_id, model="baseline_majority",
                      metric_class="macro", precision=np.nan, recall=np.nan,
                      f1=macro_f1_3, macro_f1=macro_f1_3, accuracy=acc3, n_test=len(yte)))

    # --- DummyClassifier(strategy='stratified') baseline ---
    dummy = DummyClassifier(strategy="stratified", random_state=SEED).fit(Xtr, ytr)
    pred_d = dummy.predict(Xte)
    p4, r4, f14, macro_f1_4, acc4 = eval_predictions(yte, pred_d)
    for i, cls in enumerate(CLASS_ORDER):
        rows.append(dict(sample=sample_key, fold=fold_id, model="baseline_dummy_stratified",
                          metric_class=cls, precision=p4[i], recall=r4[i], f1=f14[i],
                          macro_f1=macro_f1_4, accuracy=acc4, n_test=len(yte)))
    rows.append(dict(sample=sample_key, fold=fold_id, model="baseline_dummy_stratified",
                      metric_class="macro", precision=np.nan, recall=np.nan,
                      f1=macro_f1_4, macro_f1=macro_f1_4, accuracy=acc4, n_test=len(yte)))

    # --- ablation: BAF+DP-only proxy for "the sequence-error model's own call".
    #     The real error model (run_sequence_error_model.py) thresholds BAF and depth
    #     per ref/alt transition type; a depth-limited decision tree on the same two
    #     variables is the closest honest, computable stand-in without re-deriving its
    #     exact per-transition median thresholds from the shifted_results.pkl cache. ---
    baf_idx = feature_cols.index("BAF")
    dp_idx = feature_cols.index("DP")
    Xtr_bd = Xtr[:, [baf_idx, dp_idx]]
    Xte_bd = Xte[:, [baf_idx, dp_idx]]
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(Xtr_bd, ytr)
    pred_t = tree.predict(Xte_bd)
    p5, r5, f15, macro_f1_5, acc5 = eval_predictions(yte, pred_t)
    for i, cls in enumerate(CLASS_ORDER):
        rows.append(dict(sample=sample_key, fold=fold_id, model="baf_dp_proxy_tree",
                          metric_class=cls, precision=p5[i], recall=r5[i], f1=f15[i],
                          macro_f1=macro_f1_5, accuracy=acc5, n_test=len(yte)))
    rows.append(dict(sample=sample_key, fold=fold_id, model="baf_dp_proxy_tree",
                      metric_class="macro", precision=np.nan, recall=np.nan,
                      f1=macro_f1_5, macro_f1=macro_f1_5, accuracy=acc5, n_test=len(yte)))

    return rows, perm_rows, oof_rows


def analyze_sample(sample_key, df, feature_cols):
    model_df = draw_modeling_subsample(df)
    X = model_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = model_df["label"].to_numpy(dtype=object)
    source = model_df["SOURCE"].to_numpy(dtype=object)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_rows, all_perm, all_oof = [], [], []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rows, perm_rows, oof_rows = run_fold(sample_key, X, y, source, feature_cols,
                                              train_idx, test_idx, fold_id)
        all_rows.extend(rows)
        all_perm.extend(perm_rows)
        all_oof.extend(oof_rows)
    return sample_key, len(model_df), all_rows, all_perm, all_oof


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans previews only.")

    print("\n=== Step 1: extract features + labels via the REAL run_sparcal_net.py classes ===")
    # Sequential, not joblib-parallel: Step 1 repeatedly hit
    # joblib.externals.loky.process_executor.TerminatedWorkerError (SIGKILL) on this
    # shared, contended, AFS-backed filesystem across three separate attempts (n_jobs=4,
    # 4, 2) -- the kill came from outside Python (no in-worker traceback), consistent
    # with external resource contention on a busy multi-tenant login node, not a bug in
    # build_sample_frame. Each sample takes ~20-55s serially (measured), so running all
    # four in-process costs a few minutes and removes the failure mode entirely.
    built = [build_sample_frame(k, cfg) for k, cfg in SAMPLES.items()]
    frames = {k: df for k, df, _, _ in built}
    vcf_paths = {k: (bv, sv) for k, _, bv, sv in built}

    # feature columns: identical logic to TrainingSetBuilder.build_training_set()
    feature_cols_by_sample = {}
    for k, df in frames.items():
        feature_cols_by_sample[k] = [c for c in df.columns if c not in ("label", "SOURCE", "POS")]
    # use the UNION as the modeling feature set, 0-filled where a sample lacks a column
    # (mirrors the fillna(0) already applied at inference time in apply_to_vcf)
    all_feature_cols = sorted(set().union(*feature_cols_by_sample.values()))
    for k, df in frames.items():
        for c in all_feature_cols:
            if c not in df.columns:
                df[c] = 0.0
    print(f"Feature set (union across samples, {len(all_feature_cols)} cols): {all_feature_cols}")

    class_balance = summarize_class_balance(frames)

    print("\n=== Step 2: 5-fold stratified CV + permutation importance + ablation "
          f"(cap={MAX_TRAINING_SAMPLES}/sample, seed={SEED}) ===")
    results = Parallel(n_jobs=4)(
        delayed(analyze_sample)(k, frames[k], all_feature_cols) for k in SAMPLE_ORDER
    )

    cv_rows, perm_rows, oof_rows, model_n = [], [], [], {}
    for sample_key, n_model, rows, prows, oofrows in results:
        cv_rows.extend(rows)
        perm_rows.extend(prows)
        oof_rows.extend(oofrows)
        model_n[sample_key] = n_model

    # fold in the cv_modeling_subsample class-balance rows now that we know n_model
    cb_rows = []
    for sample_key in SAMPLE_ORDER:
        sub = draw_modeling_subsample(frames[sample_key])
        n = len(sub)
        counts = sub["label"].value_counts()
        for cls in CLASS_ORDER:
            c = int(counts.get(cls, 0))
            cb_rows.append(dict(sample=sample_key, stage="cv_modeling_subsample", source="combined",
                                 label=cls, count=c, fraction=c / n, n_total=n,
                                 majority_class=counts.idxmax(), majority_fraction=counts.max() / n))
    class_balance = pd.concat([class_balance, pd.DataFrame(cb_rows)], ignore_index=True)

    cv_df = pd.DataFrame(cv_rows)
    perm_df = pd.DataFrame(perm_rows)
    oof_df = pd.DataFrame(oof_rows)

    # domain-shift summary + capped point sample for reproducible plotting
    ds_summary = (oof_df.groupby(["sample", "source"])
                  .agg(n=("max_prob", "size"), median_max_prob=("max_prob", "median"),
                       mean_max_prob=("max_prob", "mean"), accuracy=("correct", "mean"))
                  .reset_index())
    rng = np.random.RandomState(SEED)
    ds_points = (oof_df.groupby(["sample", "source"], group_keys=False)
                 .apply(lambda g: g.sample(n=min(len(g), 3000), random_state=SEED)))

    # ---------------- write CSVs ----------------
    class_balance_csv = os.path.join(DATA_DIR, "class_balance.csv")
    cv_csv = os.path.join(DATA_DIR, "cv_performance.csv")
    perm_csv = os.path.join(DATA_DIR, "permutation_importance.csv")
    ds_summary_csv = os.path.join(DATA_DIR, "domain_shift_summary.csv")
    ds_points_csv = os.path.join(DATA_DIR, "domain_shift_points_sample.csv")
    ablation_csv = os.path.join(DATA_DIR, "ablation_vs_seqerr_proxy.csv")

    class_balance.to_csv(class_balance_csv, index=False)
    cv_df.to_csv(cv_csv, index=False)
    perm_df.to_csv(perm_csv, index=False)
    ds_summary.to_csv(ds_summary_csv, index=False)
    ds_points[["sample", "fold", "source", "true", "pred", "correct", "max_prob"]].to_csv(
        ds_points_csv, index=False)
    cv_df[cv_df.model.isin(["code_arch_100_50", "baf_dp_proxy_tree"])].to_csv(ablation_csv, index=False)
    for p in [class_balance_csv, cv_csv, perm_csv, ds_summary_csv, ds_points_csv, ablation_csv]:
        print(f"[csv] wrote {p}")

    # ---------------- figure ----------------
    make_figure(class_balance, cv_df, perm_df, oof_df, ds_summary, model_n, all_feature_cols)

    write_results_md(frames, vcf_paths, class_balance, cv_df, perm_df, ds_summary, model_n,
                      all_feature_cols, t_start)

    print(f"\nDone in {time.time() - t_start:.1f}s total.")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_a(ax, class_balance, model_n, cv_df):
    """Class balance + training-set size, with the majority-class (trivial) baseline
    fraction AND the trained model's own out-of-sample accuracy drawn on the SAME axis
    -- so a reader can see directly whether the classifier clears the trivial rule, or
    is merely close to it, without cross-referencing panel (b)."""
    sub = class_balance[(class_balance.stage == "full_available") & (class_balance.source == "combined")]
    x = np.arange(len(SAMPLE_ORDER))
    width = 0.6
    bottoms = np.zeros(len(SAMPLE_ORDER))
    for cls in CLASS_ORDER:
        vals = [float(sub[(sub["sample"] == s) & (sub.label == cls)].fraction.iloc[0]) for s in SAMPLE_ORDER]
        ax.bar(x, vals, width, bottom=bottoms, color=CLASS_COLOR[cls], label=cls,
               edgecolor="white", linewidth=0.5, zorder=3)
        bottoms += np.array(vals)
    for i, s in enumerate(SAMPLE_ORDER):
        n_full = int(sub[(sub["sample"] == s) & (sub.label == CLASS_ORDER[0])].n_total.iloc[0])
        maj = float(sub[(sub["sample"] == s) & (sub.label == CLASS_ORDER[0])].majority_fraction.iloc[0])
        ax.text(i, 1.02, f"N={n_full:,}\n(cv n={model_n[s]:,})", ha="center", va="bottom",
                fontsize=5.6, color=MUTED)
        ax.plot([i - width / 2, i + width / 2], [maj, maj], color=INK, lw=1.3, zorder=4,
                 label="majority-class fraction\n(trivial-rule accuracy)" if i == 0 else None)
        # trained-model out-of-sample accuracy, same axis, same units (fraction correct) --
        # this is the direct "does it clear the trivial baseline" comparison, drawn here
        # (not only in panel b) because that is the finding, not an incidental detail.
        acc_vals = cv_df[(cv_df["sample"] == s) & (cv_df.model == "code_arch_100_50") &
                         (cv_df.metric_class == "macro")].accuracy.values
        ax.scatter([i], [acc_vals.mean()], marker="D", s=22, color=SPARCAL_C, zorder=5,
                   edgecolor="white", linewidth=0.4,
                   label="code-arch CV accuracy\n(mean, fold range)" if i == 0 else None)
        ax.plot([i, i], [acc_vals.min(), acc_vals.max()], color=SPARCAL_C, lw=1.3, zorder=5)
        ax.text(i, min(maj, acc_vals.mean()) - 0.06, f"Δacc={acc_vals.mean() - maj:+.3f}",
                ha="center", va="top", fontsize=5.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLE_ORDER, fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of labeled variants\n/ accuracy", fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=5.4, loc="lower right", frameon=False, ncol=1)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=11)
    style_ax(ax)


def panel_b(axes, cv_df):
    metrics = CLASS_ORDER + ["macro"]
    for ax, s in zip(axes, SAMPLE_ORDER):
        sub = cv_df[cv_df["sample"] == s]
        xt = np.arange(len(metrics))
        for j, m in enumerate(metrics):
            code = sub[(sub.model == "code_arch_100_50") & (sub.metric_class == m)].f1.values
            ax.scatter(np.full(len(code), j - 0.14), code, s=10, color=SPARCAL_C, zorder=4,
                       label="code arch (100,50)" if j == 0 else None)
            ax.plot([j - 0.14, j - 0.14], [code.min(), code.max()], color=SPARCAL_C, lw=1.0, zorder=3)
            if m == "macro":
                paper = sub[(sub.model == "paper_arch_64_32") & (sub.metric_class == m)].f1.values
                ax.scatter(np.full(len(paper), j + 0.14), paper, s=10, marker="D", color=MONO_C,
                           zorder=4, label="paper arch (64,32)")
                ax.plot([j + 0.14, j + 0.14], [paper.min(), paper.max()], color=MONO_C, lw=1.0, zorder=3)
            maj = sub[(sub.model == "baseline_majority") & (sub.metric_class == m)].f1.mean()
            dum = sub[(sub.model == "baseline_dummy_stratified") & (sub.metric_class == m)].f1.mean()
            ax.plot([j - 0.32, j + 0.32], [maj, maj], color=MUTED, lw=1.0, linestyle="--", zorder=2)
            ax.plot([j - 0.32, j + 0.32], [dum, dum], color=MUTED, lw=1.0, linestyle=":", zorder=2)
        ax.set_xticks(xt)
        ax.set_xticklabels(["no-var", "het", "hom", "macro"], fontsize=6.2, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.text(0.02, 0.96, s, transform=ax.transAxes, fontsize=7, color=INK, fontweight="bold",
                ha="left", va="top")
        ax.tick_params(axis="y", labelsize=6.2)
        style_ax(ax)
    axes[0].set_ylabel("F1 (5-fold CV)", fontsize=7.5)
    axes[0].set_title("b", loc="left", fontweight="bold", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=5.4, loc="lower left", frameon=False)


def panel_c(axes, perm_df, top_n=8):
    for ax, s in zip(axes, SAMPLE_ORDER):
        sub = perm_df[perm_df["sample"] == s]
        agg = (sub.groupby("feature")["importance_mean"]
               .agg(["mean", "min", "max"]).reset_index()
               .sort_values("mean", ascending=False).head(top_n))
        agg = agg.iloc[::-1]
        y = np.arange(len(agg))
        ax.barh(y, agg["mean"], color=SPARCAL_C, zorder=3, height=0.6)
        ax.errorbar(agg["mean"], y, xerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(agg["feature"], fontsize=5.8)
        ax.text(0.98, 0.04, s, transform=ax.transAxes, fontsize=7, color=INK, fontweight="bold",
                ha="right", va="bottom")
        ax.tick_params(axis="x", labelsize=6.2)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_xlabel("perm. importance (Δ macro-F1)", fontsize=6.8)
    axes[0].set_title("c", loc="left", fontweight="bold", fontsize=11)
    for ax in axes[1:]:
        ax.set_xlabel("")


def panel_d(axes, oof_df):
    for ax, s in zip(axes, SAMPLE_ORDER):
        sub = oof_df[oof_df["sample"] == s]
        data, colors, labels = [], [], []
        for src in ["BEAGLE", "seq_no_err"]:
            vals = sub[sub.source == src].max_prob.values
            if len(vals) == 0:
                continue
            data.append(vals)
            colors.append(SOURCE_COLOR[src])
            labels.append(SOURCE_LABEL[src])
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.75)
            pc.set_edgecolor(INK)
            pc.set_linewidth(0.5)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            parts[key].set_color(INK)
            parts[key].set_linewidth(0.8)
        for i, (vals, src) in enumerate(zip(data, ["BEAGLE", "seq_no_err"])):
            acc = sub[sub.source == src].correct.mean()
            ax.text(i + 1, 1.06, f"acc={acc:.2f}", ha="center", va="bottom", fontsize=5.6, color=MUTED)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_ylim(0, 1.2)
        ax.text(0.5, 0.02, s, transform=ax.transAxes, fontsize=7, color=INK, fontweight="bold",
                ha="center", va="bottom")
        ax.tick_params(axis="y", labelsize=6.2)
        style_ax(ax)
    axes[0].set_ylabel("max predicted\nprobability (OOF)", fontsize=7)
    axes[0].set_title("d", loc="left", fontweight="bold", fontsize=11)


def panel_e(ax, cv_df):
    x = np.arange(len(SAMPLE_ORDER))
    for j, (model, color, marker, label) in enumerate([
        ("code_arch_100_50", SPARCAL_C, "o", "full-feature MLP"),
        ("baf_dp_proxy_tree", MUTED, "s", "BAF+DP-only proxy tree"),
    ]):
        offs = -0.12 if j == 0 else 0.12
        for i, s in enumerate(SAMPLE_ORDER):
            vals = cv_df[(cv_df["sample"] == s) & (cv_df.model == model) &
                         (cv_df.metric_class == "macro")].f1.values
            ax.scatter(np.full(len(vals), i + offs), vals, s=14, color=color, marker=marker, zorder=4,
                       label=label if i == 0 else None)
            ax.plot([i + offs, i + offs], [vals.min(), vals.max()], color=color, lw=1.1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLE_ORDER, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro-F1 (5-fold CV)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.set_title("e", loc="left", fontweight="bold", fontsize=11)
    style_ax(ax)


def make_figure(class_balance, cv_df, perm_df, oof_df, ds_summary, model_n, feature_cols):
    fig = plt.figure(figsize=(11.5, 15.5))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, :])
    panel_a(ax_a, class_balance, model_n, cv_df)

    axes_b = [fig.add_subplot(gs[1, i]) for i in range(4)]
    panel_b(axes_b, cv_df)

    axes_c = [fig.add_subplot(gs[2, i]) for i in range(4)]
    panel_c(axes_c, perm_df)

    axes_d = [fig.add_subplot(gs[3, i]) for i in range(4)]
    panel_d(axes_d, oof_df)

    ax_e = fig.add_subplot(gs[4, :])
    panel_e(ax_e, cv_df)

    suffix = "" if HAS_ARIAL else "_preview"
    png = os.path.join(FIG_DIR, f"fig_sparcalnet_eval{suffix}.png")
    pdf = os.path.join(FIG_DIR, f"fig_sparcalnet_eval{suffix}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"[fig] wrote {png}")
    print(f"[fig] wrote {pdf}")


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------
def write_results_md(frames, vcf_paths, class_balance, cv_df, perm_df, ds_summary, model_n,
                      feature_cols, t_start):
    lines = []
    lines.append("# SparcalNet evaluation -- RESULTS\n")
    lines.append(f"Generated by `scripts/postanalyze/fig_sparcalnet_eval_v5.py`, "
                  f"{time.strftime('%Y-%m-%d %H:%M')}. Wall time this run: "
                  f"{time.time() - t_start:.0f}s.\n")

    lines.append("## 0. Architecture provenance -- read this before any number below\n")
    lines.append(
        "There are **two** independent NN-classifier scripts in `scripts/4_classifier/`:\n\n"
        "| script | hidden layers | solver/max_iter | matches manuscript? | wired into canonical pipeline? |\n"
        "|---|---|---|---|---|\n"
        "| `run_sparcal_net.py` | **(100, 50)** (hardcoded) | adam / 500 (hardcoded) | NO | "
        "NO -- root `CLAUDE.md` step-5 table names `run_supplimentary_models.py` explicitly and "
        "flags this file as having \"the no_variance label-encoder bug\" |\n"
        "| `run_supplimentary_models.py` | **(64, 32)** (hardcoded) | adam / **200** "
        "(sklearn defaults -- solver/max_iter not passed) | **YES, exactly** ('hidden layers of 64 "
        "and 32 neurons', PaperDraft.tex / PaperDraftGuided.tex Methods) | YES (root CLAUDE.md "
        "pipeline table, step 5) |\n\n"
        "**The manuscript's stated architecture (64, 32) is CORRECT** -- it matches "
        "`run_supplimentary_models.py`, the script CLAUDE.md documents as canonical step 5, "
        "exactly. **This is not a case of the paper contradicting the code.** The live problem is "
        "narrower: the Methods text names *neither* script, so a reader cannot tell which of the "
        "two implementations -- one of them (`run_sparcal_net.py`) explicitly flagged buggy and "
        "unused -- produced the results being described. **This evaluation was scoped to "
        "`run_sparcal_net.py`** (the file the task named, and the one whose saved model artifacts "
        "exist on disk: `data/P4_tumor/1/output_VCFs/SPARCALNet/` and "
        "`data/P6_tumor/1_pre_umidedup_2026-06-24/output_VCFs/SPARCALNet/`, confirming it has been "
        "run historically) -- i.e. it evaluates the non-canonical implementation, not the one "
        "CLAUDE.md says actually produced the manuscript's results. Panel b's 'paper arch "
        "(64,32)' series runs the *labels/features from run_sparcal_net.py's own extraction "
        "logic* through `run_supplimentary_models.py`'s exact `MLPClassifier("
        "hidden_layer_sizes=(64,32), activation='relu', random_state=42)` call (unspecified "
        "solver/max_iter -> sklearn defaults, i.e. max_iter=200) so the two architectures are "
        "compared on identical CV folds; it is NOT a full reproduction of "
        "`run_supplimentary_models.py`'s own training-set construction, which builds its training "
        "VCFs from `metrics/beagle/*_shifted_results.pkl` transition tables rather than reading "
        "BEAGLE+SeqErrModel GT fields directly, and writes its model to "
        "`output_VCFs/Classifier/<qf>/results/`, a completely different output tree from "
        "`output_VCFs/SPARCALNet/`. A from-scratch evaluation of the actually-deployed "
        "`run_supplimentary_models.py` training pipeline was out of scope for this figure and "
        "would need its own pass.\n"
    )

    lines.append("## 1. Inputs\n")
    lines.append("| sample | beagle VCF (panel-defined, SOURCE=BEAGLE) | "
                  "SeqErrModel VCF (de novo candidates, SOURCE=seq_no_err) |\n|---|---|---|\n")
    for k in SAMPLE_ORDER:
        bv, sv = vcf_paths[k]
        lines.append(f"| {k} | `{bv}` | `{sv}` |\n")
    lines.append(
        "\n`sequence_no_error.vcf.gz` is built by `scripts/3_classifier_prep/run_sequence_error_model.py` "
        "from `all_filtered_out.vcf.gz` (i.e. sites BEAGLE could NOT resolve against the 1000G panel), "
        "screened by a per-ref/alt-transition BAF+depth median threshold. So SOURCE=BEAGLE genuinely is "
        "the panel-defined domain and SOURCE=seq_no_err genuinely is the de novo/non-panel domain -- "
        "this is not a relabeling, it follows directly from how each file is produced upstream.\n"
    )
    lines.append(f"\nQuality filter: `{QF}`. Labels: GT field already present in these same VCFs, parsed by "
                  "`TrainingSetBuilder._extract_labels_from_vcf()` (called unmodified, not reimplemented). "
                  "Features: `FeatureExtractor.extract_features()` (also called unmodified) -- DP, QS, VDB, "
                  "RPB, MQB, BQB, MQSB, SGB, MQ0F, BAF, GQ, and up to 16 I16_* subfields.\n")

    lines.append("\n## 2. Reproducibility note (bug found, not fixed -- evaluation only)\n")
    lines.append(
        "`TrainingSetBuilder.build_training_set()` downsamples to `max_training_samples` (90,000, the "
        "script's own default) via `np.random.choice(...)` **with no seed set** -- every real run of "
        "`run_sparcal_net.py --dataset ... ` on a sample exceeding 90k combined variants (P6, DCIS1, "
        "DCIS2 all do) draws a *different* random training set. This evaluation fixes a local seed "
        "(`numpy.random.RandomState(42)`) purely so THIS script's own numbers are reproducible; it does "
        "not fix the upstream script.\n")

    lines.append("\n## 3. Panel (a) -- class balance and training-set size\n")
    lines.append("| sample | N labeled (full available) | N used for CV (cap=90,000) | "
                  "no_variance | heterozygous | homozygous | majority class | majority fraction |\n"
                  "|---|---|---|---|---|---|---|---|\n")
    full = class_balance[(class_balance.stage == "full_available") & (class_balance.source == "combined")]
    for s in SAMPLE_ORDER:
        sub = full[full["sample"] == s]
        n_full = int(sub.n_total.iloc[0])
        fr = {row.label: row.fraction for row in sub.itertuples()}
        maj = sub.majority_class.iloc[0]
        majf = sub.majority_fraction.iloc[0]
        lines.append(f"| {s} | {n_full:,} | {model_n[s]:,} | {fr.get('no_variance', 0):.3f} | "
                      f"{fr.get('heterozygous', 0):.3f} | {fr.get('homozygous', 0):.3f} | {maj} | "
                      f"{majf:.3f} |\n")
    lines.append("\nA classifier that always predicts the majority class already scores this well on "
                  "accuracy alone -- panel (b) reports macro-F1 (which does not reward majority-class "
                  "prediction) specifically to control for this.\n")

    lines.append("\n## 4. Panel (b) -- 5-fold stratified CV vs baselines\n")
    lines.append("Macro-F1 per sample, mean over 5 folds (see CSV for full fold-level spread and "
                  "per-class precision/recall/F1):\n\n")
    lines.append("| sample | code arch (100,50) macro-F1 | paper arch (64,32) macro-F1 | "
                  "majority-baseline macro-F1 | dummy-stratified macro-F1 |\n|---|---|---|---|---|\n")
    for s in SAMPLE_ORDER:
        sub = cv_df[(cv_df["sample"] == s) & (cv_df.metric_class == "macro")]
        code = sub[sub.model == "code_arch_100_50"].macro_f1.mean()
        paper = sub[sub.model == "paper_arch_64_32"].macro_f1.mean()
        maj = sub[sub.model == "baseline_majority"].macro_f1.mean()
        dum = sub[sub.model == "baseline_dummy_stratified"].macro_f1.mean()
        lines.append(f"| {s} | {code:.3f} | {paper:.3f} | {maj:.3f} | {dum:.3f} |\n")

    lines.append("\nPer-class F1 (code arch, mean over 5 folds):\n\n")
    lines.append("| sample | no_variance | heterozygous | homozygous |\n|---|---|---|---|\n")
    for s in SAMPLE_ORDER:
        sub = cv_df[(cv_df["sample"] == s) & (cv_df.model == "code_arch_100_50")]
        vals = {cls: sub[sub.metric_class == cls].f1.mean() for cls in CLASS_ORDER}
        lines.append(f"| {s} | {vals['no_variance']:.3f} | {vals['heterozygous']:.3f} | "
                      f"{vals['homozygous']:.3f} |\n")

    lines.append("\n## 5. Panel (c) -- permutation importance (top features, mean across folds)\n")
    for s in SAMPLE_ORDER:
        sub = perm_df[perm_df["sample"] == s]
        top = (sub.groupby("feature")["importance_mean"].mean()
               .sort_values(ascending=False).head(5))
        lines.append(f"- **{s}**: " + ", ".join(f"{f} ({v:.3f})" for f, v in top.items()) + "\n")

    lines.append("\n## 6. Panel (d) -- domain shift, panel-defined vs de novo\n")
    lines.append("Out-of-fold predictions only (each row scored by the fold model that held it out).\n\n")
    lines.append("| sample | source | n | median max-prob | accuracy |\n|---|---|---|---|---|\n")
    for s in SAMPLE_ORDER:
        for src in ["BEAGLE", "seq_no_err"]:
            row = ds_summary[(ds_summary["sample"] == s) & (ds_summary.source == src)]
            if len(row) == 0:
                continue
            lines.append(f"| {s} | {SOURCE_LABEL[src].splitlines()[0]} | {int(row.n.iloc[0]):,} | "
                          f"{row.median_max_prob.iloc[0]:.3f} | {row.accuracy.iloc[0]:.3f} |\n")

    lines.append("\n## 7. Panel (e) -- ablation vs a BAF+DP-only proxy\n")
    lines.append(
        "The real sequence-error model (`run_sequence_error_model.py`) makes its error/no-error call "
        "from BAF and depth alone, thresholded per ref/alt transition type on that transition's own "
        "median (see `calculate_transition_thresholds`). Re-deriving those exact per-transition "
        "thresholds from the `*_shifted_results.pkl` caches was out of scope here; instead this panel "
        "trains a depth-4 decision tree on BAF+DP alone, under the identical 5-fold CV protocol, as an "
        "honest (if more flexible than the real rule) stand-in for 'what BAF/depth thresholding alone "
        "can do'.\n\n")
    lines.append("| sample | full-feature MLP macro-F1 | BAF+DP-only proxy macro-F1 | delta |\n"
                  "|---|---|---|---|\n")
    for s in SAMPLE_ORDER:
        sub = cv_df[(cv_df["sample"] == s) & (cv_df.metric_class == "macro")]
        full_f1 = sub[sub.model == "code_arch_100_50"].macro_f1.mean()
        proxy_f1 = sub[sub.model == "baf_dp_proxy_tree"].macro_f1.mean()
        lines.append(f"| {s} | {full_f1:.3f} | {proxy_f1:.3f} | {full_f1 - proxy_f1:+.3f} |\n")

    lines.append("\n## 8. What this figure must NOT be read as claiming\n")
    lines.append(
        "- Not evidence about `run_supplimentary_models.py`, the classifier actually wired into the "
        "canonical pipeline per root `CLAUDE.md` -- see Sec. 0.\n"
        "- Not a validation of true-variant-vs-artifact discrimination. The label being predicted is a "
        "**genotype class** (no_variance/het/hom) read from the SAME VCF the features come from -- for "
        "SOURCE=seq_no_err rows in particular, that label is the raw pileup-called GT, not an "
        "independent ground truth, so high accuracy there can reflect the label being recoverable from "
        "correlated features rather than genuine artifact/true-variant discrimination.\n"
        "- Not a claim about which sample's classifier is 'better' -- sample-to-sample differences "
        "here likely track class balance and candidate-pool composition, not classifier quality.\n"
        "- The BAF+DP-only ablation is a proxy for the sequence-error model's decision variables, not "
        "a literal re-implementation of its per-transition threshold rule.\n")

    out_path = os.path.join(DATA_DIR, "RESULTS.md")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"[md] wrote {out_path}")


if __name__ == "__main__":
    main()
