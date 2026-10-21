#!/usr/bin/env python3
"""Deeper assessment of Ubiquitous Private Variants (UPVs).

This analysis treats the original BAF/purity GMM as a hypothesis generator.
It challenges that split with depth, a defined-germline BAF control, external
RNA-editing/PON masks, matched-normal WES, cross-section recurrence, parameter
stability, and spot-level purity association residualized for detection burden.
It never rewrites the production VCFs.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
QF = "baseQ0mapQ0"
CALICOST = Path("/data/maiziezhou_lab/leiy4/CalicoST")
SAMPLES = {
    "P4": {
        "build": "GRCh37",
        # The July production launch left its 0.3/0.2 example lines commented,
        # so the saved output used the executable defaults 0.5/0.4.  This pair
        # reproduces all 9,491 de-novo germline/UPV records exactly.
        "stage1_alpha": 0.5, "stage1_beta": 0.4,
        "root": PROJECT / "data/P4_tumor/1",
        "purity": CALICOST / "P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
        "clones": CALICOST / "P4_sec1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv",
        "filtered": PROJECT / "data/artifact_mask_benchmark/current_scomatic/P4",
    },
    "P6": {
        "build": "GRCh37",
        # The sample launch left its 0.4/0.4 example commented; executable
        # defaults were 0.5/0.4.  Alpha values from 0.3--0.6 happen to produce
        # the same 1,744 records at beta=0.4, so retain the actual default here.
        "stage1_alpha": 0.5, "stage1_beta": 0.4,
        "root": PROJECT / "data/P6_tumor/1",
        "purity": CALICOST / "P6_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
        "clones": CALICOST / "P6_sec1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv",
        "filtered": PROJECT / "data/artifact_mask_benchmark/current_scomatic/P6",
    },
    "DCIS1": {
        "build": "GRCh38",
        "stage1_alpha": 0.5, "stage1_beta": 0.2,
        "root": PROJECT / "data/dcis1",
        "purity": CALICOST / "DCIS1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
        "clones": CALICOST / "DCIS1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv",
        "filtered": PROJECT / "data/artifact_mask_benchmark/current_scomatic/DCIS1",
    },
    "DCIS2": {
        "build": "GRCh38",
        "stage1_alpha": 0.5, "stage1_beta": 0.2,
        "root": PROJECT / "data/dcis2",
        "purity": CALICOST / "DCIS2/estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
        "clones": CALICOST / "DCIS2/calicost/clone3_rectangle0_w1.0/clone_labels.tsv",
        "filtered": PROJECT / "data/artifact_mask_benchmark/current_scomatic/DCIS2",
    },
}


def load_helper():
    path = PROJECT / "scripts/6_spatial_filter/upv_baf_gmm_subfilter.py"
    spec = importlib.util.spec_from_file_location("upv_gmm_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canon(chrom):
    return str(chrom).removeprefix("chr")


def key(chrom, pos, ref, alt):
    return canon(chrom), int(pos), ref.upper(), alt.upper()


def key_string(k):
    return "%s_%d_%s_%s" % k


def parse_key_string(value):
    chrom, pos, ref, alt = value.split("_")
    return key(chrom, pos, ref, alt)


def load_vcf(path):
    rows = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip().split("\t")
            if len(f[3]) == 1 and len(f[4]) == 1 and "," not in f[4]:
                info = {}
                for item in f[7].split(";"):
                    if "=" in item:
                        a, b = item.split("=", 1)
                        info[a] = b
                rows[key(f[0], f[1], f[3], f[4])] = info
    return rows


def load_gmm(root):
    path = root / ("spatial_filter_purity/%s/germline/denovo/gmm_subfilter/"
                   "upv_gmm_classification.tsv" % QF)
    frame = pd.read_csv(path, sep="\t")
    frame["key"] = frame.chrom_pos_ref_alt.map(parse_key_string)
    return frame


def load_spot_data(cfg, upv_keys):
    purity = pd.read_csv(cfg["purity"], sep="\t")
    purity["barcode"] = purity.BARCODES.str.split("_").str[0]
    purity = purity.set_index("barcode").Tumor
    clones = pd.read_csv(cfg["clones"], sep="\t")
    clones["barcode"] = clones.BARCODES.str.split("_").str[0]
    clones = clones.set_index("barcode").clone_label
    spot_dir = cfg["root"] / ("spatial_filter_purity/%s/germline" % QF)
    records = []
    present = defaultdict(list)
    for path in spot_dir.glob("*.txt"):
        if path.name.endswith("_variants.txt"):
            continue
        barcode = path.stem
        raw_purity = purity.get(barcode, np.nan)
        # Production load_tumor_purity replaces NaN/missing CalicoST values by
        # zero for these samples.  Preserve that exact value and a complete-case
        # copy so the consequence of imputation can be measured explicitly.
        production_purity = 0.0 if pd.isna(raw_purity) else float(raw_purity)
        n_rows = 0
        with open(path) as handle:
            header = handle.readline().rstrip().split("\t")
            idx = {name: i for i, name in enumerate(header)}
            for line in handle:
                f = line.rstrip().split("\t")
                if len(f) < 4:
                    continue
                n_rows += 1
                race = f[idx.get("race", len(f) - 1)]
                if race != "denovo":
                    continue
                k = key(f[idx["chrom"]], f[idx["pos"]], f[idx["ref"]], f[idx["alt"]])
                if k in upv_keys:
                    present[k].append(barcode)
        records.append({
            "barcode": barcode,
            "purity": production_purity,
            "purity_complete": float(raw_purity) if pd.notna(raw_purity) else np.nan,
            "purity_observed": bool(pd.notna(raw_purity)),
            "clone": clones.get(barcode, np.nan),
            "germline_detection_burden": n_rows,
        })
    spots = pd.DataFrame(records).drop_duplicates("barcode").set_index("barcode")
    return spots, present


def standardized_residual(y, covariate):
    good = np.isfinite(y) & np.isfinite(covariate)
    design = np.column_stack([np.ones(good.sum()), covariate[good]])
    beta = np.linalg.lstsq(design, y[good], rcond=None)[0]
    residual = np.full(len(y), np.nan)
    residual[good] = y[good] - design @ beta
    residual[good] -= residual[good].mean()
    sd = residual[good].std(ddof=0)
    if sd:
        residual[good] /= sd
    return residual


def binary_corr(present_indices, x):
    good = np.isfinite(x)
    n = int(good.sum())
    if n < 4:
        return np.nan
    y = np.zeros(len(x), dtype=float)
    y[present_indices] = 1
    y = y[good]
    xv = x[good]
    if y.sum() == 0 or y.sum() == len(y) or xv.std() == 0:
        return np.nan
    return float(np.corrcoef(y, xv)[0, 1])


def partial_binary_corr(present_indices, purity, burden):
    good = np.isfinite(purity) & np.isfinite(burden)
    n = int(good.sum())
    if n < 5:
        return np.nan, np.nan, np.nan, n
    y = np.zeros(len(purity), dtype=float)
    y[present_indices] = 1
    y, p, b = y[good], purity[good], burden[good]
    if y.sum() == 0 or y.sum() == len(y) or p.std() == 0 or b.std() == 0:
        return np.nan, np.nan, np.nan, n
    r_yp = float(np.corrcoef(y, p)[0, 1])
    r_yb = float(np.corrcoef(y, b)[0, 1])
    r_pb = float(np.corrcoef(p, b)[0, 1])
    den = math.sqrt(max(0.0, (1 - r_yb * r_yb) * (1 - r_pb * r_pb)))
    partial = (r_yp - r_yb * r_pb) / den if den else np.nan
    return r_yp, r_yb, partial, n


def corr_pvalue(r, n, n_covariates=0):
    df = n - n_covariates - 2
    if not np.isfinite(r) or df <= 0 or abs(r) >= 1:
        return np.nan
    stat = r * math.sqrt(df / (1 - r * r))
    return float(2 * student_t.sf(abs(stat), df))


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    good = np.where(np.isfinite(values))[0]
    if not len(good):
        return out
    order = good[np.argsort(values[good])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1)
    return out


def add_spot_metrics(frame, spots, present):
    barcodes = list(spots.index)
    index = {bc: i for i, bc in enumerate(barcodes)}
    purity = spots.purity.to_numpy(float)
    purity_complete = spots.purity_complete.to_numpy(float)
    log_burden = np.log1p(spots.germline_detection_burden.to_numpy(float))
    clone_sizes = Counter(spots.clone.dropna().astype(int))
    raw_imputed, raw_complete, burden_corr, adjusted = [], [], [], []
    adjusted_imputed, clone_delta, nspots, n_complete = [], [], [], []
    for k in frame["key"]:
        idx = np.array([index[bc] for bc in present.get(k, []) if bc in index], dtype=int)
        idx = np.unique(idx)
        rpi, _, pai, ni = partial_binary_corr(idx, purity, log_burden)
        rpc, rbc, pac, nc = partial_binary_corr(idx, purity_complete, log_burden)
        raw_imputed.append(rpi)
        raw_complete.append(rpc)
        burden_corr.append(rbc)
        adjusted.append(pac)
        adjusted_imputed.append(pai)
        n_complete.append(nc)
        nspots.append(len(idx))
        if len(clone_sizes) == 2:
            clones = sorted(clone_sizes)
            p = []
            present_barcodes = {barcodes[i] for i in idx}
            for clone in clones:
                members = spots.index[spots.clone == clone]
                p.append(sum(bc in present_barcodes for bc in members) / len(members))
            clone_delta.append(p[1] - p[0])
        else:
            clone_delta.append(np.nan)
    frame["n_spots_with_upv"] = nspots
    frame["purity_corr_recomputed"] = raw_imputed
    frame["purity_corr_complete_case"] = raw_complete
    frame["presence_burden_corr_complete_case"] = burden_corr
    frame["purity_corr_burden_adjusted"] = adjusted
    frame["purity_corr_burden_adjusted_imputed"] = adjusted_imputed
    frame["n_complete_case_spots"] = n_complete
    frame["clone2_minus_clone1_prevalence"] = clone_delta
    pvals = [corr_pvalue(r, n, n_covariates=1) for r, n in zip(adjusted, n_complete)]
    pvals_imputed = [corr_pvalue(r, len(spots), n_covariates=1) for r in adjusted_imputed]
    frame["purity_adjusted_p"] = pvals
    frame["purity_adjusted_q"] = bh_adjust(pvals)
    frame["purity_adjusted_imputed_p"] = pvals_imputed
    frame["purity_adjusted_imputed_q"] = bh_adjust(pvals_imputed)
    return frame


def gmm_grid(sample, frame):
    rows = []
    x = frame[["baf", "purity_corr"]].to_numpy(float)
    dp = frame.dp.to_numpy(float)
    baf = frame.baf.to_numpy(float)
    for n_components in [2, 3, 4]:
        for seed in [0, 1, 2]:
            fit = dp >= 5
            model = GaussianMixture(
                n_components=n_components, covariance_type="full",
                n_init=3, random_state=seed,
            ).fit(x[fit])
            post = model.predict_proba(x)
            for pur_cut in [0.05, 0.10, 0.15]:
                comps = [
                    k for k in range(n_components)
                    if model.means_[k, 0] < 0.40 and model.means_[k, 1] > pur_cut
                ]
                probability = post[:, comps].sum(axis=1) if comps else np.zeros(len(frame))
                called = (probability > 0.5) & (baf < 0.35)
                rows.append({
                    "sample": sample, "n_components": n_components, "seed": seed,
                    "purity_cut": pur_cut, "n_called": int(called.sum()),
                    "fraction_called": float(called.mean()),
                    "somatic_components": ",".join(map(str, comps)) if comps else "none",
                    "component_means": ";".join(
                        "%.3f/%.3f" % tuple(v) for v in model.means_
                    ),
                })
    return rows


def stage1_threshold_grid(sample, cfg):
    """Reclassify the existing de-novo score table over a fixed alpha/beta grid.

    This tests the stability of UPV membership itself, upstream of the optional
    BAF/purity mixture.  The current trees contain no forced matched-normal
    de-novo records: the number of de-novo rows labeled germline is exactly the
    final UPV count in every sample.
    """
    path = cfg["root"] / ("spatial_filter_purity/%s/all_variant_scores.txt" % QF)
    scores = pd.read_csv(path, sep="\t", low_memory=False)
    scores = scores[scores.race.eq("denovo")].copy()
    scores["alpha"] = pd.to_numeric(scores.f_spatial_uniformity, errors="coerce")
    scores["beta"] = pd.to_numeric(scores.f_global_prevalence, errors="coerce")
    scores = scores.dropna(subset=["alpha", "beta"])
    production = set(scores.loc[scores.classification.eq("germline"), "variant"])
    rows = []
    for alpha in [0.3, 0.4, 0.5, 0.6]:
        for beta in [0.2, 0.3, 0.4, 0.5]:
            called = set(scores.loc[(scores.alpha > alpha) & (scores.beta > beta), "variant"])
            intersection = len(called & production)
            union = len(called | production)
            rows.append({
                "sample": sample,
                "alpha_threshold": alpha,
                "beta_threshold": beta,
                "is_production_setting": (
                    alpha == cfg["stage1_alpha"] and beta == cfg["stage1_beta"]
                ),
                "n_denovo_candidates": len(scores),
                "n_upv": len(called),
                "production_n_upv": len(production),
                "production_retained_fraction": (
                    intersection / len(production) if production else np.nan
                ),
                "jaccard_with_production": intersection / union if union else np.nan,
            })
    return rows


def defined_baf_control(sample, cfg, helper, upv_frame):
    base = cfg["root"]
    defined_vcf = base / ("spatial_filter_purity/%s/germline/defined/germline_defined.vcf.gz" % QF)
    merged = base / ("output_VCFs/mpileup_multi_bam/%s/merged_sorted_gt.vcf.gz" % QF)
    defined = load_vcf(defined_vcf)
    # The production helper preserves POS as a string in its tuple keys.  This
    # analysis canonicalizes POS to int, so convert explicitly at the boundary.
    helper_keys = [(chrom, str(pos), ref, alt) for chrom, pos, ref, alt in defined]
    baf_map = helper.lookup_baf(str(merged), helper_keys)
    d = pd.DataFrame([
        {"baf": baf, "dp": dp} for baf, dp in baf_map.values() if np.isfinite(baf)
    ], columns=["baf", "dp"])
    if d.empty:
        raise RuntimeError("no defined-germline BAF controls matched %s" % merged)
    rows = []
    for dbin, lo, hi in [("all", 0, np.inf), ("DP1-4", 1, 5), ("DP5-9", 5, 10),
                          ("DP10-29", 10, 30), ("DP30+", 30, np.inf)]:
        u = upv_frame[(upv_frame.dp >= lo) & (upv_frame.dp < hi)].baf.dropna()
        g = d[(d.dp >= lo) & (d.dp < hi)].baf.dropna()
        rows.append({
            "sample": sample, "depth_group": dbin,
            "n_upv": len(u), "n_defined": len(g),
            "upv_baf_median": u.median() if len(u) else np.nan,
            "defined_baf_median": g.median() if len(g) else np.nan,
            "wasserstein_baf": wasserstein_distance(u, g) if len(u) and len(g) else np.nan,
            "upv_low_baf_fraction": float((u < 0.35).mean()) if len(u) else np.nan,
            "defined_low_baf_fraction": float((g < 0.35).mean()) if len(g) else np.nan,
        })
    return rows


def summarize_spots(sample, cfg, spots):
    observed = spots.purity_observed.to_numpy(bool)
    purity = spots.loc[observed, "purity_complete"].to_numpy(float)
    burden = spots.germline_detection_burden.to_numpy(float)
    log_burden = np.log1p(burden)
    prod_purity = spots.purity.to_numpy(float)
    clone_counts = spots.clone.dropna().astype(int).value_counts().sort_index()
    return {
        "sample": sample, "build": cfg["build"], "n_spots": len(spots),
        "n_observed_purity": int(observed.sum()),
        "fraction_purity_imputed_zero": float(1 - observed.mean()),
        "purity_median_complete": float(np.median(purity)) if len(purity) else np.nan,
        "germline_detection_burden_median": float(np.median(burden)),
        "purity_log_burden_corr_complete": (
            float(np.corrcoef(purity, log_burden[observed])[0, 1]) if len(purity) > 2 else np.nan
        ),
        "purity_log_burden_corr_production_imputed": (
            float(np.corrcoef(prod_purity, log_burden)[0, 1]) if len(spots) > 2 else np.nan
        ),
        "clone_counts": ";".join("%d:%d" % (k, v) for k, v in clone_counts.items()),
    }


def pairwise_overlap(keysets, artifact_sets, editing_sets):
    rows = []
    for a, b in combinations(sorted(keysets), 2):
        if SAMPLES[a]["build"] != SAMPLES[b]["build"]:
            continue
        shared = keysets[a] & keysets[b]
        intersection = len(keysets[a] & keysets[b])
        union = len(keysets[a] | keysets[b])
        rows.append({
            "build": SAMPLES[a]["build"], "sample_a": a, "sample_b": b,
            "n_a": len(keysets[a]), "n_b": len(keysets[b]),
            "intersection": intersection, "jaccard": intersection / union if union else np.nan,
            "shared_artifact_masked_either": len(shared & (artifact_sets[a] | artifact_sets[b])),
            "shared_artifact_masked_both": len(shared & artifact_sets[a] & artifact_sets[b]),
            "shared_editing_compatible": len(shared & editing_sets[a] & editing_sets[b]),
        })
    return rows


def subgroup_comparison(all_variants):
    def finite_median(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else np.nan

    rows = []
    for sample, frame in all_variants.items():
        groups = {
            "all": np.ones(len(frame), dtype=bool),
            "low_baf_lt_0.35": frame.baf.to_numpy(float) < 0.35,
            "original_gmm_candidate": frame["class"].eq("somatic_candidate").to_numpy(),
            "adjusted_positive_q_lt_0.05": (
                (frame.purity_adjusted_q < 0.05) &
                (frame.purity_corr_burden_adjusted > 0)
            ).to_numpy(),
            "adjusted_negative_q_lt_0.05": (
                (frame.purity_adjusted_q < 0.05) &
                (frame.purity_corr_burden_adjusted < 0)
            ).to_numpy(),
        }
        for label, mask in groups.items():
            g = frame.loc[mask]
            rows.append({
                "sample": sample, "subgroup": label, "n": len(g),
                "artifact_mask_fraction": float(g.artifact_masked.mean()) if len(g) else np.nan,
                "editing_compatible_fraction": float(g.editing_compatible.mean()) if len(g) else np.nan,
                "median_adjusted_purity_corr": finite_median(g.purity_corr_burden_adjusted),
                "median_clone2_minus_clone1_prevalence": finite_median(g.clone2_minus_clone1_prevalence),
                "normal_wes_callable_n": int((g.normal_wes_callable == True).sum()),
                "normal_wes_alt_evidence_n": int((g.normal_wes_alt_evidence == True).sum()),
            })
    return pd.DataFrame(rows)


def wes_annotations(sample, frame, p1_dir):
    path = p1_dir / "sparcal_call_details.csv.gz"
    if not path.exists() or sample not in {"P4", "P6"}:
        frame["normal_wes_callable"] = np.nan
        frame["normal_wes_exact"] = np.nan
        frame["normal_wes_alt_evidence"] = np.nan
        return frame
    details = pd.read_csv(path)
    details = details[(details["sample"] == sample) & (details.panel == "upv")].copy()
    details["key"] = list(zip(
        details.chrom.map(canon), details.pos.astype(int),
        details.ref.str.upper(), details.alt.str.upper(),
    ))
    if details.key.duplicated().any():
        raise RuntimeError("duplicate exact alleles in P1 call details for %s" % sample)
    lookup = {row.key: row for row in details.itertuples(index=False)}
    callable_values, exact_values, evidence_values = [], [], []
    for k in frame["key"]:
        if k not in lookup:
            callable_values.append(False)
            exact_values.append(False)
            evidence_values.append(False)
            continue
        row = lookup[k]
        is_callable = row.wes_pileup_dp >= 10
        exact = bool(row.high_conf_wes_exact)
        evidence = exact or (
            is_callable and row.normal_wes_alt_count >= 3 and row.wes_pileup_vaf >= 0.10
        )
        callable_values.append(bool(is_callable))
        exact_values.append(exact)
        evidence_values.append(bool(evidence))
    frame["normal_wes_callable"] = callable_values
    frame["normal_wes_exact"] = exact_values
    frame["normal_wes_alt_evidence"] = evidence_values
    return frame


def build_summary(all_variants, baf_control, grid, stage1_grid):
    rows = []
    for sample, frame in all_variants.items():
        n = len(frame)
        callable_frame = frame[frame.normal_wes_callable == True]
        sig = frame.purity_adjusted_q < 0.05
        sig_imputed = frame.purity_adjusted_imputed_q < 0.05
        rows += [
            {"sample": sample, "metric": "n_upv", "value": n, "denominator": n},
            {"sample": sample, "metric": "median_baf", "value": frame.baf.median(), "denominator": n},
            {"sample": sample, "metric": "low_baf_fraction_lt_0.35",
             "value": float((frame.baf < 0.35).mean()), "denominator": n},
            {"sample": sample, "metric": "original_gmm_candidate_fraction",
             "value": float((frame["class"] == "somatic_candidate").mean()), "denominator": n},
            {"sample": sample, "metric": "artifact_mask_fraction",
             "value": float(frame.artifact_masked.mean()), "denominator": n},
            {"sample": sample, "metric": "editing_compatible_fraction",
             "value": float(frame.editing_compatible.mean()), "denominator": n},
            {"sample": sample, "metric": "burden_adjusted_purity_q_lt_0.05",
             "value": int((frame.purity_adjusted_q < 0.05).sum()), "denominator": n},
            {"sample": sample, "metric": "burden_adjusted_positive_q_lt_0.05",
             "value": int((sig & (frame.purity_corr_burden_adjusted > 0)).sum()), "denominator": n},
            {"sample": sample, "metric": "burden_adjusted_negative_q_lt_0.05",
             "value": int((sig & (frame.purity_corr_burden_adjusted < 0)).sum()), "denominator": n},
            {"sample": sample, "metric": "burden_adjusted_positive_q_lt_0.05_r_ge_0.10",
             "value": int((sig & (frame.purity_corr_burden_adjusted >= 0.10)).sum()), "denominator": n},
            {"sample": sample, "metric": "production_imputed_adjusted_q_lt_0.05",
             "value": int(sig_imputed.sum()), "denominator": n},
            {"sample": sample, "metric": "stored_vs_recomputed_imputed_corr",
             "value": frame[["purity_corr", "purity_corr_recomputed"]].corr().iloc[0, 1],
             "denominator": n},
        ]
        if len(callable_frame):
            rows += [
                {"sample": sample, "metric": "normal_wes_callable_upv",
                 "value": len(callable_frame), "denominator": n},
                {"sample": sample, "metric": "normal_wes_exact_fraction",
                 "value": float(callable_frame.normal_wes_exact.mean()),
                 "denominator": len(callable_frame)},
                {"sample": sample, "metric": "normal_wes_alt_evidence_fraction",
                 "value": float(callable_frame.normal_wes_alt_evidence.mean()),
                 "denominator": len(callable_frame)},
                {"sample": sample, "metric": "somatic_interpretation_upper_bound",
                 "value": float(1 - callable_frame.normal_wes_alt_evidence.mean()),
                 "denominator": len(callable_frame)},
            ]
    for sample, group in grid.groupby("sample"):
        rows += [
            {"sample": sample, "metric": "gmm_grid_min_called",
             "value": int(group.n_called.min()), "denominator": len(group)},
            {"sample": sample, "metric": "gmm_grid_max_called",
             "value": int(group.n_called.max()), "denominator": len(group)},
        ]
    for sample, group in stage1_grid.groupby("sample"):
        production = group[group.is_production_setting]
        rows += [
            {"sample": sample, "metric": "stage1_production_upv",
             "value": int(production.iloc[0].n_upv), "denominator": int(production.iloc[0].n_denovo_candidates)},
            {"sample": sample, "metric": "stage1_grid_min_upv",
             "value": int(group.n_upv.min()), "denominator": len(group)},
            {"sample": sample, "metric": "stage1_grid_max_upv",
             "value": int(group.n_upv.max()), "denominator": len(group)},
            {"sample": sample, "metric": "stage1_grid_min_jaccard_with_production",
             "value": float(group.jaccard_with_production.min()), "denominator": len(group)},
        ]
    return pd.DataFrame(rows)


def write_report(outdir, summary, overlap, baf_control, grid):
    lines = [
        "# UPV assessment: BAF, purity, matched normal, and artifact evidence", "",
        "## Executive result", "",
        "The original BAF/purity GMM does not define a stable somatic UPV subclass. "
        "Its output is challenged below with independent and negative-control evidence. "
        "UPV should remain a behavior-defined private/uniform class, not be relabelled as "
        "somatic or germline variant by variant from this GMM alone.", "",
        "## Summary metrics", "",
        "| sample | metric | value | denominator |", "|---|---|---:|---:|",
    ]
    for row in summary.itertuples():
        value = "%.4g" % row.value if isinstance(row.value, (float, np.floating)) else str(row.value)
        lines.append("| %s | %s | %s | %s |" % (row.sample, row.metric, value, row.denominator))
    lines += [
        "", "## Interpretation and claim boundaries", "",
        "- BAF in pseudobulk RNA is not a DNA VAF. Allele-specific expression, mapping bias, "
        "copy number, transcript abundance and depth can all move it away from 0.5.",
        "- PURITY_CORR is a correlation of binary spot detection with CalicoST purity, not a "
        "per-spot BAF-purity slope. Production replaces missing purity by zero and floors negative "
        "correlations. We therefore report both production-imputed and complete-case estimates.",
        "- The burden-adjusted statistic is a partial point-biserial correlation controlling for "
        "log per-spot germline detection burden. It addresses a first-order coverage proxy but does "
        "not make CalicoST independent of the RNA assay.",
        "- An exact high-confidence matched-normal WES allele establishes germline origin. "
        "Normal-WES ALT evidence outside the variant VCF is supportive but less stringent. "
        "Absence of either is only a conservative upper bound on somatic interpretation.",
        "- Editing/PON membership supports an RNA-artifact explanation; absence from those "
        "catalogues does not establish a biological variant.",
        "- The parameter grid is a stability diagnostic, not a model-selection exercise. "
        "No parameter was tuned against a somatic truth set.", "",
        "## Outputs", "",
        "- upv_variant_metrics.csv.gz: per-UPV evidence and adjusted associations.",
        "- upv_summary.csv: sample-level conclusions and the conservative somatic upper bound.",
        "- upv_gmm_sensitivity.csv: candidate counts across component/seed/purity-cut choices.",
        "- upv_stage1_threshold_sensitivity.csv: UPV counts and membership stability across alpha/beta thresholds.",
        "- upv_defined_baf_control.csv: depth-stratified UPV versus 1KGP-defined BAF distributions.",
        "- upv_pairwise_overlap.csv: exact-allele recurrence across sections.",
        "- upv_spot_covariates.csv: purity missingness, detection burden, and their correlation.",
        "- upv_subgroup_comparison.csv: artifact, editing, clone, and WES evidence by diagnostic subgroup.",
        "- RESULTS.md: this report.",
        "- Generator: %s" % Path(__file__).resolve(),
    ]
    (outdir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT / "data/upv_baf_assessment_2026-08-23")
    parser.add_argument("--p1-dir", type=Path,
                        default=PROJECT / "data/germline_concordance_2026-08-23")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    helper = load_helper()
    all_variants, grids, stage1_grids, controls, keysets = {}, [], [], [], {}
    artifact_sets, editing_sets, spot_rows = {}, {}, []
    for sample, cfg in SAMPLES.items():
        frame = load_gmm(cfg["root"])
        upv_vcf = cfg["root"] / ("spatial_filter_purity/%s/germline/denovo/germline_denovo.vcf.gz" % QF)
        filtered_vcf = cfg["filtered"] / ("spatial_filter_purity/%s/germline/denovo/germline_denovo.vcf.gz" % QF)
        original_keys = set(load_vcf(upv_vcf))
        kept_keys = set(load_vcf(filtered_vcf)) if filtered_vcf.exists() else set()
        frame["artifact_masked"] = frame["key"].map(lambda k: k not in kept_keys)
        frame["editing_compatible"] = [
            (ref == "A" and alt == "G") or (ref == "T" and alt == "C")
            for _, _, ref, alt in frame["key"]
        ]
        spots, present = load_spot_data(cfg, original_keys)
        spot_rows.append(summarize_spots(sample, cfg, spots))
        frame = add_spot_metrics(frame, spots, present)
        frame = wes_annotations(sample, frame, args.p1_dir)
        frame["sample"] = sample
        all_variants[sample] = frame
        keysets[sample] = original_keys
        artifact_sets[sample] = set(frame.loc[frame.artifact_masked, "key"])
        editing_sets[sample] = set(frame.loc[frame.editing_compatible, "key"])
        grids.extend(gmm_grid(sample, frame))
        stage1_grids.extend(stage1_threshold_grid(sample, cfg))
        controls.extend(defined_baf_control(sample, cfg, helper, frame))
    combined = pd.concat(all_variants.values(), ignore_index=True)
    grid = pd.DataFrame(grids)
    stage1_grid = pd.DataFrame(stage1_grids)
    control = pd.DataFrame(controls)
    overlap = pd.DataFrame(pairwise_overlap(keysets, artifact_sets, editing_sets))
    spot_summary = pd.DataFrame(spot_rows)
    subgroups = subgroup_comparison(all_variants)
    summary = build_summary(all_variants, control, grid, stage1_grid)
    combined = combined.drop(columns=["key"])
    combined.to_csv(args.outdir / "upv_variant_metrics.csv.gz", index=False, compression="gzip")
    grid.to_csv(args.outdir / "upv_gmm_sensitivity.csv", index=False)
    stage1_grid.to_csv(args.outdir / "upv_stage1_threshold_sensitivity.csv", index=False)
    control.to_csv(args.outdir / "upv_defined_baf_control.csv", index=False)
    overlap.to_csv(args.outdir / "upv_pairwise_overlap.csv", index=False)
    spot_summary.to_csv(args.outdir / "upv_spot_covariates.csv", index=False)
    subgroups.to_csv(args.outdir / "upv_subgroup_comparison.csv", index=False)
    summary.to_csv(args.outdir / "upv_summary.csv", index=False)
    write_report(args.outdir, summary, overlap, control, grid)
    print(summary.to_string(index=False))
    print("Wrote UPV assessment to %s" % args.outdir)


if __name__ == "__main__":
    main()
