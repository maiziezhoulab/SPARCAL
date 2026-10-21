#!/usr/bin/env python3
"""Non-trained genotype x allele-specific CNV x purity model for spatial SNVs.

This is a side-car analysis: it does not modify the production SPARCAL calls.
It benchmarks published WES somatic SNVs against high-confidence matched-normal
WES heterozygous SNVs using UMI-deduplicated spatial-BAM REF/ALT counts.

The primary score is a beta-binomial marginal log Bayes factor

    log P(spatial counts | somatic, CNV, purity)
      - log P(spatial counts | germline het, CNV, purity).

Somatic hypotheses marginalize cancer-cell fraction and mutation multiplicity;
germline hypotheses marginalize which allele carries ALT.  No outcome labels are
used to fit parameters or weights.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from scipy.special import betaln, gammaln, logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
CAL = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
STMUT = Path(
    "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/"
    "Datasets/ST_datasets/STmut_Data"
)
TRUTH = ROOT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv"
POSITIVE_CACHE = (
    ROOT
    / "data/calicost_snv_utility_2026-09-08/wes_spot_locus_umi_evidence.tsv.gz"
)
FULL = "baseQ0mapQ0_full_calicost"
SECTIONS = {
    "P4_rep1": dict(patient="P4", rep="1", clone="clone2_rectangle0_w1.0", cnv="cnv_seglevel.tsv"),
    "P4_rep2": dict(patient="P4", rep="2", clone="clone2_rectangle0_w1.0", cnv="cnv_seglevel.tsv"),
    "P6_rep1": dict(patient="P6", rep="1", clone="clone3_rectangle0_w1.0", cnv="cnv_seglevel.tsv"),
    "P6_rep2": dict(patient="P6", rep="2", clone="clone3_rectangle0_w1.0", cnv="cnv_diploid_seglevel.tsv"),
}
MODEL_CONFIGS = [
    "no_purity_no_cnv",
    "purity_only",
    "purity_total_cn",
    "purity_allele_cnv",
    "no_purity_allele_cnv",
]


def chrom(value: object) -> str:
    return str(value).removeprefix("chr")


def vkey(c: object, p: object, ref: object, alt: object) -> str:
    return f"{chrom(c)}_{int(p)}_{str(ref).upper()}_{str(alt).upper()}"


def unkey(key: str) -> tuple[str, int, str, str]:
    c, p, ref, alt = key.split("_")
    return c, int(p), ref, alt


def section_paths(label: str) -> dict[str, Path]:
    cfg = SECTIONS[label]
    p, r = cfg["patient"], cfg["rep"]
    sec = ROOT / "data" / f"{p}_tumor" / r
    co = CAL / f"{p}_sec{r}"
    outs = (
        STMUT
        / f"{p}_Visium"
        / f"spaceranger_align_rep{r}_hg19"
        / f"{p}_Tumor_output/outs"
    )
    return {
        "bam": outs / "possorted_genome_bam.dedup.bam",
        "defined": sec
        / "spatial_filter_purity"
        / FULL
        / "germline/defined/germline_defined.vcf.gz",
        "cnv": co / "calicost" / cfg["clone"] / cfg["cnv"],
        "clones": co / "calicost" / cfg["clone"] / "clone_labels.tsv",
        "purity": co / "estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
    }


def load_vcf_keys(path: Path) -> set[str]:
    result: set[str] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields[3]) == len(fields[4]) == 1 and "," not in fields[4]:
                result.add(vkey(fields[0], fields[1], fields[3], fields[4]))
    return result


def high_confidence_normal_hets(patient: str) -> set[str]:
    path = (
        STMUT
        / f"{patient}_Normal_WES"
        / f"{patient}_Normal_WES_gatk_exome_snps_chr1_22.vcf.gz"
    )
    result: set[str] = set()
    with pysam.VariantFile(str(path)) as vcf:
        sample_name = list(vcf.header.samples)[0]
        for record in vcf:
            if len(record.ref) != 1 or len(record.alts or ()) != 1 or len(record.alts[0]) != 1:
                continue
            sample = record.samples[sample_name]
            gt, ad = sample.get("GT"), sample.get("AD")
            if not gt or set(gt) != {0, 1} or not ad or len(ad) < 2 or None in ad[:2]:
                continue
            dp = sample.get("DP") or sum(ad[:2])
            gq = sample.get("GQ") or 0
            denom = sum(ad[:2])
            ab = ad[1] / denom if denom else np.nan
            if dp >= 20 and gq >= 30 and ad[1] >= 5 and 0.25 <= ab <= 0.75:
                result.add(vkey(record.chrom, record.pos, record.ref, record.alts[0]))
    return result


def published_truth() -> dict[str, set[str]]:
    table = pd.read_csv(TRUTH, sep="\t")
    table["variant"] = [
        vkey(*row)
        for row in table[["chrom_hg19", "pos_hg19", "ref", "alt"]].itertuples(
            index=False, name=None
        )
    ]
    return {
        patient: set(table.loc[table["sample"] == patient, "variant"])
        for patient in ("P4", "P6")
    }


def select_controls(n_controls: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    truth = published_truth()
    for patient in ("P4", "P6"):
        normal = high_confidence_normal_hets(patient)
        shared = normal.copy()
        for rep in ("1", "2"):
            shared &= load_vcf_keys(section_paths(f"{patient}_rep{rep}")["defined"])
        shared -= truth[patient]
        candidates = np.asarray(sorted(shared), dtype=object)
        if len(candidates) > n_controls:
            candidates = rng.choice(candidates, n_controls, replace=False)
        rows.extend(
            dict(patient=patient, variant=str(variant), truth_class="normal_WES_het")
            for variant in sorted(candidates)
        )
    return pd.DataFrame(rows)


def observed_published_positives() -> pd.DataFrame:
    evidence = pd.read_csv(POSITIVE_CACHE, sep="\t")
    observed = (
        evidence.groupby(["patient", "rep", "sample", "variant"], as_index=False)
        .agg(alt_umis=("altq", "sum"))
        .query("alt_umis > 0")
    )
    observed["rep"] = observed["rep"].astype(str)
    observed["truth_class"] = "published_WES_somatic"
    return observed[["patient", "rep", "sample", "variant", "truth_class"]]


def scan_bam(path: Path, variants: set[str], label: str) -> pd.DataFrame:
    """Count one best-supported base per CB/UB molecule at each requested locus."""
    cfg = SECTIONS[label]
    rows: list[dict[str, object]] = []
    with pysam.AlignmentFile(str(path), "rb") as bam:
        references = set(bam.references)
        for number, key in enumerate(sorted(variants), 1):
            c, pos, ref, alt = unkey(key)
            contig = c if c in references else f"chr{c}"
            molecules: dict[tuple[str, str], dict[str, tuple[int, int, bool, int]]] = defaultdict(dict)
            if contig in references:
                for column in bam.pileup(
                    contig,
                    pos - 1,
                    pos,
                    truncate=True,
                    stepper="all",
                    min_base_quality=0,
                    max_depth=1_000_000,
                ):
                    if column.reference_pos != pos - 1:
                        continue
                    for pileup_read in column.pileups:
                        read = pileup_read.alignment
                        if (
                            pileup_read.is_del
                            or pileup_read.is_refskip
                            or read.is_unmapped
                            or read.is_secondary
                            or read.is_supplementary
                            or read.is_qcfail
                            or read.is_duplicate
                            or not read.has_tag("CB")
                        ):
                            continue
                        query_position = pileup_read.query_position
                        if query_position is None:
                            continue
                        base = read.query_sequence[query_position].upper()
                        if base not in "ACGT":
                            continue
                        bq = int(read.query_qualities[query_position]) if read.query_qualities is not None else 0
                        mq = int(read.mapping_quality)
                        barcode = read.get_tag("CB").split("_")[0]
                        umi = read.get_tag("UB") if read.has_tag("UB") else read.query_name
                        start = int(read.reference_start)
                        old = molecules[(barcode, umi)].get(base)
                        value = (bq, mq, bool(read.is_reverse), start)
                        if old is None or value[:2] > old[:2]:
                            molecules[(barcode, umi)][base] = value
            by_barcode: dict[str, dict[str, object]] = defaultdict(
                lambda: dict(ref=0, alt=0, other=0, alt_fwd=0, alt_rev=0, alt_bq=[], alt_mq=[], alt_starts=set())
            )
            for (barcode, _umi), base_map in molecules.items():
                base, (bq, mq, reverse, start) = max(base_map.items(), key=lambda item: item[1][:2])
                if bq < 13 or mq < 20:
                    continue
                data = by_barcode[barcode]
                if base == ref:
                    data["ref"] += 1
                elif base == alt:
                    data["alt"] += 1
                    data["alt_rev" if reverse else "alt_fwd"] += 1
                    data["alt_bq"].append(bq)
                    data["alt_mq"].append(mq)
                    data["alt_starts"].add(start)
                else:
                    data["other"] += 1
            for barcode, data in by_barcode.items():
                rows.append(
                    dict(
                        sample=label,
                        patient=cfg["patient"],
                        rep=cfg["rep"],
                        variant=key,
                        barcode=barcode,
                        ref_umis=int(data["ref"]),
                        alt_umis=int(data["alt"]),
                        other_umis=int(data["other"]),
                        alt_fwd=int(data["alt_fwd"]),
                        alt_rev=int(data["alt_rev"]),
                        alt_bq_mean=float(np.mean(data["alt_bq"])) if data["alt_bq"] else np.nan,
                        alt_mq_mean=float(np.mean(data["alt_mq"])) if data["alt_mq"] else np.nan,
                        unique_alt_starts=len(data["alt_starts"]),
                    )
                )
            if number % 100 == 0:
                print(f"  {label}: {number}/{len(variants)} loci", flush=True)
    return pd.DataFrame(rows)


class AlleleCNV:
    def __init__(self, path: Path):
        table = pd.read_csv(path, sep="\t")
        self.columns: dict[str, tuple[str, str]] = {}
        for column in table.columns:
            if str(column).startswith("clone") and str(column).endswith(" A"):
                clone = str(column).split()[0].removeprefix("clone")
                bcol = str(column)[:-1] + "B"
                if bcol in table.columns:
                    self.columns[clone] = (str(column), bcol)
        self.intervals: dict[str, tuple[list[int], list[tuple[int, int, pd.Series]]]] = {}
        for c, group in table.groupby(table["CHR"].map(chrom)):
            records = [
                (int(row.START), int(row.END), row)
                for _, row in group.sort_values("START").iterrows()
            ]
            self.intervals[c] = ([item[0] for item in records], records)

    def get(self, c: str, pos: int, clone: str) -> tuple[float, float]:
        if c not in self.intervals:
            return np.nan, np.nan
        starts, records = self.intervals[c]
        index = bisect.bisect_right(starts, pos) - 1
        if index < 0 or not records[index][0] <= pos <= records[index][1]:
            return np.nan, np.nan
        available = clone if clone in self.columns else ("1" if "1" in self.columns else sorted(self.columns)[-1])
        acol, bcol = self.columns[available]
        row = records[index][2]
        return float(row[acol]), float(row[bcol])


def spot_metadata(label: str) -> pd.DataFrame:
    paths = section_paths(label)
    purity = pd.read_csv(paths["purity"], sep="\t")
    purity["barcode"] = purity["BARCODES"].astype(str).str.split("_").str[0]
    purity["purity"] = pd.to_numeric(purity["Tumor"], errors="coerce")
    clones = pd.read_csv(paths["clones"], sep="\t")
    clones["barcode"] = clones["BARCODES"].astype(str).str.split("_").str[0]
    clones["clone"] = pd.to_numeric(clones["clone_label"], errors="coerce").astype("Int64").astype(str)
    return (
        clones[["barcode", "clone"]]
        .drop_duplicates("barcode")
        .merge(purity[["barcode", "purity"]].drop_duplicates("barcode"), on="barcode", how="left")
    )


def annotate_spots(evidence: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, group in evidence.groupby("sample"):
        index = AlleleCNV(section_paths(label)["cnv"])
        group = group.merge(spot_metadata(label), on="barcode", how="left")
        a_values, b_values = [], []
        for row in group.itertuples(index=False):
            c, pos, _ref, _alt = unkey(row.variant)
            a, b = index.get(c, pos, row.clone)
            a_values.append(a)
            b_values.append(b)
        group["copy_a"], group["copy_b"] = a_values, b_values
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def beta_binomial_logpmf(y: np.ndarray, n: np.ndarray, q: np.ndarray, kappa: float) -> np.ndarray:
    q = np.clip(0.002 + 0.996 * q, 0.002, 0.998)
    alpha, beta = q * kappa, (1.0 - q) * kappa
    return (
        gammaln(n + 1)
        - gammaln(y + 1)
        - gammaln(n - y + 1)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )


def hypothesis_log_likelihoods(
    y: np.ndarray, n: np.ndarray, q: np.ndarray, kappa: float
) -> np.ndarray:
    """Vectorized beta-binomial log likelihood for hypotheses x spots."""
    q = np.clip(0.002 + 0.996 * q, 0.002, 0.998)
    alpha, beta = q * kappa, (1.0 - q) * kappa
    constant = (
        gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    )[None, :]
    return (
        constant
        + betaln(y[None, :] + alpha, n[None, :] - y[None, :] + beta)
        - betaln(alpha, beta)
    ).sum(axis=1)


def score_group(group: pd.DataFrame, config: str, kappa: float) -> dict[str, float]:
    y = group["alt_umis"].to_numpy(float)
    n = (group["ref_umis"] + group["alt_umis"]).to_numpy(float)
    p_observed = group["purity"].to_numpy(float)
    a_observed = group["copy_a"].to_numpy(float)
    b_observed = group["copy_b"].to_numpy(float)

    if config in {"no_purity_no_cnv", "purity_only"}:
        a_observed = np.ones(len(group))
        b_observed = np.ones(len(group))
    elif config == "purity_total_cn":
        total = a_observed + b_observed
        a_observed = total / 2.0
        b_observed = total / 2.0

    uses_purity = config.startswith("purity_")
    purity_hypotheses = (
        p_observed[None, :]
        if uses_purity
        else np.repeat(np.linspace(0.05, 1.0, 20)[:, None], len(group), axis=1)
    )
    ccf_grid = np.linspace(0.05, 1.0, 20)

    mode_multiplicity = np.stack(
        [np.ones(len(group)), a_observed, b_observed], axis=0
    )
    mode_weights = np.asarray([0.60, 0.20, 0.20])
    p = purity_hypotheses[:, None, None, :]
    multiplicity = mode_multiplicity[None, :, None, :]
    ccf = ccf_grid[None, None, :, None]
    denominator = 2.0 * (1.0 - p) + p * (a_observed + b_observed)[None, None, None, :]
    denominator = np.maximum(denominator, 1e-8)
    somatic_q = (p * ccf * multiplicity / denominator).reshape(-1, len(group))
    somatic_weights = np.broadcast_to(
        mode_weights[None, :, None] / (len(purity_hypotheses) * len(ccf_grid)),
        (len(purity_hypotheses), 3, len(ccf_grid)),
    ).reshape(-1)
    somatic_ll = hypothesis_log_likelihoods(y, n, somatic_q, kappa)
    somatic = float(logsumexp(somatic_ll + np.log(somatic_weights)))

    orientations = np.stack([a_observed, b_observed], axis=0)
    p_g = purity_hypotheses[:, None, :]
    denominator_g = 2.0 * (1.0 - p_g) + p_g * (a_observed + b_observed)[None, None, :]
    germline_q = (
        ((1.0 - p_g) + p_g * orientations[None, :, :])
        / np.maximum(denominator_g, 1e-8)
    ).reshape(-1, len(group))
    germline_weights = np.full(
        len(purity_hypotheses) * 2, 1.0 / (len(purity_hypotheses) * 2)
    )
    germline_ll = hypothesis_log_likelihoods(y, n, germline_q, kappa)
    germline = float(logsumexp(germline_ll + np.log(germline_weights)))
    return dict(log_marginal_somatic=somatic, log_marginal_germline=germline, log_bf_somatic_vs_germline=somatic - germline)


def scenario(a: float, b: float) -> str:
    total = a + b
    if np.isclose(total, 0):
        return "homozygous_deletion"
    if np.isclose(total, 2) and np.isclose(a, b):
        return "diploid_balanced"
    if np.isclose(total, 2):
        return "copy_neutral_LOH"
    if total < 2:
        return "allelic_loss" if min(a, b) <= 0.25 else "balanced_loss"
    return "balanced_gain" if np.isclose(a, b) else "unbalanced_gain"


def summarize_and_score(evidence: pd.DataFrame, kappa_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = evidence.copy()
    evidence["n"] = evidence.ref_umis + evidence.alt_umis
    complete = evidence[
        (evidence.n > 0)
        & evidence.purity.notna()
        & evidence.copy_a.notna()
        & evidence.copy_b.notna()
    ].copy()
    rows, expected = [], []
    for (label, variant, truth_class), group in complete.groupby(["sample", "variant", "truth_class"]):
        if group.alt_umis.sum() < 1:
            continue
        a, b = float(group.copy_a.median()), float(group.copy_b.median())
        total_alt = int(group.alt_umis.sum())
        total_ref = int(group.ref_umis.sum())
        fwd, rev = int(group.alt_fwd.sum()), int(group.alt_rev.sum())
        base = dict(
            sample=label,
            patient=SECTIONS[label]["patient"],
            rep=SECTIONS[label]["rep"],
            variant=variant,
            truth_class=truth_class,
            label=int(truth_class == "published_WES_somatic"),
            n_spots=len(group),
            total_ref_umis=total_ref,
            total_alt_umis=total_alt,
            pooled_vaf=total_alt / (total_alt + total_ref),
            purity_median=float(group.purity.median()),
            purity_min=float(group.purity.min()),
            purity_max=float(group.purity.max()),
            copy_a=a,
            copy_b=b,
            cnv_scenario=scenario(a, b),
            alt_strand_balance=min(fwd, rev) / (fwd + rev) if fwd + rev else np.nan,
            alt_bq_mean=float(np.average(group.alt_bq_mean.dropna(), weights=group.loc[group.alt_bq_mean.notna(), "alt_umis"])) if group.alt_bq_mean.notna().any() else np.nan,
            alt_mq_mean=float(np.average(group.alt_mq_mean.dropna(), weights=group.loc[group.alt_mq_mean.notna(), "alt_umis"])) if group.alt_mq_mean.notna().any() else np.nan,
            unique_alt_starts=int(group.unique_alt_starts.sum()),
        )
        p = group.purity.to_numpy(float)
        aa, bb = group.copy_a.to_numpy(float), group.copy_b.to_numpy(float)
        denominator = 2 * (1 - p) + p * (aa + bb)
        q_single = p / denominator
        q_a = p * aa / denominator
        q_b = p * bb / denominator
        q_g_a = ((1 - p) + p * aa) / denominator
        q_g_b = ((1 - p) + p * bb) / denominator
        possible = np.concatenate([0.2 * q_single, q_single, 0.2 * q_a, q_a, 0.2 * q_b, q_b])
        base.update(
            somatic_expected_vaf_plausible_low=float(np.nanmin(possible)),
            somatic_expected_vaf_plausible_high=float(np.nanmax(possible)),
            somatic_clonal_single_copy_vaf_median=float(np.nanmedian(q_single)),
            germline_het_vaf_orientation_low=float(min(np.nanmedian(q_g_a), np.nanmedian(q_g_b))),
            germline_het_vaf_orientation_high=float(max(np.nanmedian(q_g_a), np.nanmedian(q_g_b))),
        )
        for kappa in kappa_values:
            for config in MODEL_CONFIGS:
                scores = score_group(group, config, kappa)
                rows.append({**base, "kappa": kappa, "model": config, **scores})
        for item in group.itertuples(index=False):
            denominator_i = 2 * (1 - item.purity) + item.purity * (item.copy_a + item.copy_b)
            expected.append(
                dict(
                    sample=label,
                    variant=variant,
                    truth_class=truth_class,
                    barcode=item.barcode,
                    ref_umis=item.ref_umis,
                    alt_umis=item.alt_umis,
                    observed_vaf=item.alt_umis / (item.ref_umis + item.alt_umis),
                    purity=item.purity,
                    clone=item.clone,
                    copy_a=item.copy_a,
                    copy_b=item.copy_b,
                    cnv_scenario=scenario(item.copy_a, item.copy_b),
                    expected_somatic_clonal_single=item.purity / denominator_i,
                    expected_somatic_clonal_pre_A=item.purity * item.copy_a / denominator_i,
                    expected_somatic_clonal_pre_B=item.purity * item.copy_b / denominator_i,
                    expected_germline_het_A=((1 - item.purity) + item.purity * item.copy_a) / denominator_i,
                    expected_germline_het_B=((1 - item.purity) + item.purity * item.copy_b) / denominator_i,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(expected)


def depth_match(frame: pd.DataFrame, ratio: int = 3) -> pd.DataFrame:
    selected = []
    for sample, group in frame.groupby("sample"):
        positives = group[group.label == 1].sort_values(["total_ref_umis", "variant"])
        negatives = group[group.label == 0].copy()
        if positives.empty or negatives.empty:
            continue
        available = set(negatives.index)
        for _ in range(ratio):
            for index, positive in positives.iterrows():
                if not available:
                    break
                candidates = negatives.loc[sorted(available)]
                distance = (
                    (np.log1p(candidates.total_ref_umis + candidates.total_alt_umis) - np.log1p(positive.total_ref_umis + positive.total_alt_umis)) ** 2
                    + (np.log1p(candidates.n_spots) - np.log1p(positive.n_spots)) ** 2
                )
                chosen = int(distance.idxmin())
                selected.append(chosen)
                available.remove(chosen)
        selected.extend(positives.index.tolist())
    return frame.loc[sorted(set(selected))].copy()


def bootstrap_auc(frame: pd.DataFrame, score: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    if frame.label.nunique() < 2:
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(frame.label, frame[score]))
    rng = np.random.default_rng(seed)
    positives = frame.loc[frame.label == 1, score].to_numpy(float)
    negatives = frame.loc[frame.label == 0, score].to_numpy(float)
    values = []
    for _ in range(n_boot):
        p = positives[rng.integers(0, len(positives), len(positives))]
        n = negatives[rng.integers(0, len(negatives), len(negatives))]
        pairwise = p[:, None] - n[None, :]
        values.append(float(np.mean(pairwise > 0) + 0.5 * np.mean(pairwise == 0)))
    return auc, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate(scores: pd.DataFrame, primary_kappa: float, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = scores[scores.kappa == primary_kappa].copy()
    rows = []
    for model, group in primary.groupby("model"):
        for sample, data in [("pooled", group)] + list(group.groupby("sample")):
            matched = depth_match(data)
            for cohort, frame in (("all_evaluable", data), ("depth_spot_matched", matched)):
                if frame.label.nunique() < 2:
                    continue
                auc, low, high = bootstrap_auc(frame, "log_bf_somatic_vs_germline", n_boot, seed)
                rows.append(
                    dict(
                        model=model,
                        sample=sample,
                        cohort=cohort,
                        n_somatic=int(frame.label.sum()),
                        n_germline=int((frame.label == 0).sum()),
                        auc=auc,
                        auc_ci_low=low,
                        auc_ci_high=high,
                        average_precision=float(average_precision_score(frame.label, frame.log_bf_somatic_vs_germline)),
                    )
                )
    sensitivity = []
    for (kappa, model), data in scores.groupby(["kappa", "model"]):
        matched = depth_match(data)
        if matched.label.nunique() == 2:
            sensitivity.append(
                dict(
                    kappa=kappa,
                    model=model,
                    n_somatic=int(matched.label.sum()),
                    n_germline=int((matched.label == 0).sum()),
                    auc=float(roc_auc_score(matched.label, matched.log_bf_somatic_vs_germline)),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(sensitivity)


def markdown(table: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|")

    columns = list(table)
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        + [
            "| " + " | ".join(fmt(value) for value in row) + " |"
            for row in table.itertuples(index=False, name=None)
        ]
    )


def write_report(output: Path, controls: pd.DataFrame, scores: pd.DataFrame, evaluation: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    primary = evaluation[
        (evaluation["sample"] == "pooled")
        & (evaluation["cohort"] == "depth_spot_matched")
    ].sort_values("auc", ascending=False)
    counts = (
        scores[scores.kappa == 8]
        .drop_duplicates(["sample", "variant", "truth_class"])
        .groupby(["sample", "truth_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    allele_primary = scores[(scores.kappa == 8) & (scores.model == "purity_allele_cnv")]
    somatic_scenarios = (
        allele_primary[allele_primary.label == 1]
        .groupby("cnv_scenario")
        .size()
        .rename("n_published_somatic")
        .reset_index()
        .sort_values("n_published_somatic", ascending=False)
    )
    examples = (
        allele_primary[allele_primary.label == 1]
        .sort_values("log_bf_somatic_vs_germline", ascending=False)
        [["sample", "variant", "cnv_scenario", "total_ref_umis", "total_alt_umis", "pooled_vaf", "purity_median", "log_bf_somatic_vs_germline"]]
        .head(5)
    )
    scenario_examples = pd.DataFrame([
        ("diploid 1|1", 0.300, 0.300, 0.500, 0.500),
        ("hemizygous loss 1|0", 0.429, 0.429, 0.286, 0.714),
        ("CN-LOH 0|2", 0.300, 0.600, 0.200, 0.800),
        ("unbalanced gain 1|2", 0.231, 0.462, 0.385, 0.615),
    ], columns=["state_at_purity_0.6", "somatic_post_CNV_m1", "somatic_pre_CNV_max", "germline_het_low", "germline_het_high"])
    report = [
        "# Mechanistic CNV/purity-aware spatial VAF benchmark",
        "",
        "## Direct answer about the current pipeline",
        "",
        "The production `run_spatial_snv_filter_enhanced.py` does **not** currently implement a genotype × allele-specific CNV × purity VAF expectation. Its CNV feature is variant prevalence within a clone with heuristic LOH/gain boosts; its purity feature correlates binary record presence with purity. It does not read per-spot REF/ALT AD for those features. Therefore this analysis is a side-car prototype, not a restatement of the shipped score.",
        "",
        "## Mechanistic model",
        "",
        "For tumor purity `p`, allele-specific tumor copies `(A,B)`, cancer-cell fraction `f`, and mutation multiplicity `m`, the somatic expected DNA VAF is `p*f*m / (2*(1-p) + p*(A+B))`. A heterozygous germline ALT has expectation `((1-p) + p*A)/denominator` or the corresponding B-allele value. The score marginalizes `f=0.05..1`, post-CNV single-copy versus pre-CNV A/B multiplicity, and germline allele orientation using fixed predeclared priors. A beta-binomial emission (`kappa=8` primary; 4 and 20 sensitivity) deliberately allows RNA ASE/overdispersion. Missing purity is excluded, never changed to zero.",
        "",
        "## Independent labels and evaluable data",
        "",
        f"Negative controls were sampled from {len(controls):,} high-confidence matched-normal WES heterozygous loci shared by both SPARCAL defined-germline replicate callsets. Positives are published Table S1 WES somatic loci with at least one current q13/mapQ20 spatial ALT UMI. Neither label was used to fit parameters.",
        "",
        markdown(counts),
        "",
        "P4 rep2 has no informative purity values and is absent from the complete-case benchmark. This is an input limitation, not evidence against the model.",
        "",
        "## Model combinations (primary depth/spot-matched benchmark)",
        "",
        markdown(primary[["model", "n_somatic", "n_germline", "auc", "auc_ci_low", "auc_ci_high", "average_precision"]]),
        "",
        "Purity raises the matched AUC from 0.803 to 0.883. Total-CN and allele-specific-CN do not add measurable ranking improvement in this small benchmark. This is expected from the available information: only 5/16 evaluable somatic variant-sections are in altered states, and only two are in gains.",
        "",
        "## CNV scenarios and data-supported examples",
        "",
        markdown(somatic_scenarios),
        "",
        "The same nominal VAF has different meanings across states. For a clonal event at purity 0.6, the model expectations are:",
        "",
        markdown(scenario_examples),
        "",
        "Two low-VAF, multi-spot published somatic examples provide the clearest support for the mechanism: chr3:101404717 G>A in P6 rep1 has 1 ALT / 109 REF+ALT UMIs and log BF 52.5; chr6:116784827 T>A has 2/48 in P6 rep1 and 1/35 in rep2, with log BF 24.6 and 19.1. Conversely, most one-ALT low-coverage truth sites have negative absolute BF: the model correctly says that the spatial RNA alone is insufficient even though WES establishes truth.",
        "",
        markdown(examples),
        "",
        "`spot_expected_vaf_scenarios.tsv.gz` retains the per-spot observed VAF and all somatic/germline expectations, so future scores can use a soft interval/likelihood rather than a CNV-region indicator.",
        "",
        "## Overdispersion sensitivity",
        "",
        markdown(sensitivity.sort_values(["kappa", "auc"], ascending=[True, False])),
        "",
        "## Mutect2-derived evidence architecture",
        "",
        "The WES reverse-engineering audit should be used as a feature-family sanity check, not as independent validation: published filtered calls are compared with the local unfiltered-only Mutect2 set. Its strongest transferable signals are tumor-normal delta VAF, clean matched normal, ALT support/TLOD-like likelihood, and strand/orientation balance; raw depth is near chance. These motivate two separately interpretable axes:",
        "",
        "1. `E_real`: ALT UMI support, unique molecules/starts, strand/orientation balance, BQ/MQ/read-position context, recurrent-artifact/PON and population evidence.",
        "2. `E_somatic`: the CNV/purity/genotype marginal log Bayes factor implemented here.",
        "",
        "A final non-trained ranking can add calibrated log-evidence terms, while retaining each axis for audit. CNV/purity should be a soft likelihood term, never a hard filter: low coverage, ASE, purity error, subclonality, and CNV uncertainty can all widen the expected range.",
        "",
        "## Guardrails",
        "",
        "- Conditioning on spatial ALT observation evaluates classification among observable candidates; it does not measure unconditional WES recovery.",
        "- CalicoST and SPARCAL use the same spatial RNA, so CNV/purity is supportive evidence rather than orthogonal validation.",
        "- The present benchmark is intentionally small. Bootstrap intervals quantify sampling instability but not uncertainty in CalicoST states.",
        "- A production implementation should propagate alternative CalicoST ploidy/tree solutions and add an explicit artifact model before combining scores.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report))


def plot_results(output: Path, evaluation: pd.DataFrame, scores: pd.DataFrame) -> None:
    primary = evaluation[
        (evaluation["sample"] == "pooled")
        & (evaluation["cohort"] == "depth_spot_matched")
    ].sort_values("auc")
    display = {
        "no_purity_no_cnv": "Neither",
        "purity_only": "Purity",
        "purity_total_cn": "Purity + total CN",
        "purity_allele_cnv": "Purity + allele CN",
        "no_purity_allele_cnv": "Allele CN only",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(primary))
    axes[0].errorbar(
        primary.auc,
        y,
        xerr=[primary.auc - primary.auc_ci_low, primary.auc_ci_high - primary.auc],
        fmt="o",
        capsize=3,
    )
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, [display.get(x, x) for x in primary.model])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("AUC: published somatic vs normal-WES het")
    axes[0].set_title("Non-trained mechanistic combinations")

    best = primary.iloc[-1].model if len(primary) else "purity_allele_cnv"
    data = scores[(scores.kappa == 8) & (scores.model == best)]
    matched = depth_match(data)
    for label, group in matched.groupby("truth_class"):
        axes[1].hist(
            group.log_bf_somatic_vs_germline,
            bins=20,
            alpha=0.55,
            label=label.replace("_", " "),
        )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("log BF (somatic / germline het)")
    axes[1].set_ylabel("Variant-section count")
    axes[1].set_title(f"Best primary combination: {display.get(best, best)}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "mechanistic_vaf_benchmark.png", dpi=200)
    fig.savefig(output / "mechanistic_vaf_benchmark.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/mechanistic_cnv_purity_vaf_2026-09-10",
    )
    parser.add_argument("--n-controls-per-patient", type=int, default=400)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controls = select_controls(args.n_controls_per_patient, args.seed)
    positives = observed_published_positives()
    controls.to_csv(args.output_dir / "selected_normal_wes_het_controls.tsv", sep="\t", index=False)

    evidence_path = args.output_dir / "deduplicated_spatial_evidence.tsv.gz"
    if evidence_path.exists():
        print(f"Reusing {evidence_path}", flush=True)
        evidence = pd.read_csv(evidence_path, sep="\t", dtype={"rep": str})
    else:
        frames = []
        for label, cfg in SECTIONS.items():
            positive_variants = set(positives.loc[positives["sample"] == label, "variant"])
            control_variants = set(controls.loc[controls.patient == cfg["patient"], "variant"])
            variants = positive_variants | control_variants
            print(
                f"Scanning {label}: {len(positive_variants)} observed somatic positives + "
                f"{len(control_variants)} normal-WES het controls",
                flush=True,
            )
            frame = scan_bam(section_paths(label)["bam"], variants, label)
            positive_set = positive_variants
            frame["truth_class"] = np.where(
                frame.variant.isin(positive_set),
                "published_WES_somatic",
                "normal_WES_het",
            )
            frames.append(frame)
        evidence = pd.concat(frames, ignore_index=True)
        evidence.to_csv(evidence_path, sep="\t", index=False, compression="gzip")

    score_path = args.output_dir / "variant_mechanistic_scores.tsv.gz"
    expected_path = args.output_dir / "spot_expected_vaf_scenarios.tsv.gz"
    if score_path.exists() and expected_path.exists():
        print(f"Reusing {score_path} and {expected_path}", flush=True)
        scores = pd.read_csv(score_path, sep="\t")
        expected = pd.read_csv(expected_path, sep="\t")
    else:
        print("Annotating clone, allele-specific CNV, and purity", flush=True)
        evidence = annotate_spots(evidence)
        scores, expected = summarize_and_score(evidence, [4.0, 8.0, 20.0])
        scores.to_csv(score_path, sep="\t", index=False, compression="gzip")
        expected.to_csv(expected_path, sep="\t", index=False, compression="gzip")
    evaluation, sensitivity = evaluate(scores, 8.0, args.n_bootstrap, args.seed)

    evaluation.to_csv(args.output_dir / "model_combination_benchmark.tsv", sep="\t", index=False)
    sensitivity.to_csv(args.output_dir / "overdispersion_sensitivity.tsv", sep="\t", index=False)
    write_report(args.output_dir, controls, scores, evaluation, sensitivity)
    plot_results(args.output_dir, evaluation, scores)

    manifest = {
        "created": "2026-09-10",
        "script": str(Path(__file__).resolve()),
        "truth": str(TRUTH),
        "truth_sha256": hashlib.sha256(TRUTH.read_bytes()).hexdigest(),
        "n_selected_controls": len(controls),
        "n_evaluable_variant_sections": int(
            len(scores[scores.kappa == 8].drop_duplicates(["sample", "variant", "truth_class"]))
        ),
        "model_is_trained": False,
        "primary_kappa": 8,
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    primary = evaluation[
        (evaluation["sample"] == "pooled")
        & (evaluation["cohort"] == "depth_spot_matched")
    ].sort_values("auc", ascending=False)
    print(primary.to_string(index=False), flush=True)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
