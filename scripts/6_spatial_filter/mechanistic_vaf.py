#!/usr/bin/env python3
"""Training-free CNV/purity-aware VAF evidence for the spatial SNV filter.

The model compares two biological hypotheses with a beta-binomial observation
model.  Somatic hypotheses integrate over cancer-cell fraction (CCF), mutated
allele and mutation multiplicity.  Germline hypotheses integrate over the two
constitutional alleles.  Missing purity and CNV values are marginalized over
explicit priors rather than being converted to normal/diploid point estimates.

This module intentionally contains no project-path logic, which keeps the core
calculation independently testable and reusable by downstream audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import betaln, expit, gammaln, logsumexp


DEFAULT_PURITY_PRIOR = np.asarray([0.05, 0.15, 0.30, 0.50, 0.75, 0.95])
DEFAULT_CNV_PRIOR = (
    (1.0, 1.0, 0.70),
    (1.0, 0.0, 0.075),
    (0.0, 1.0, 0.075),
    (0.0, 2.0, 0.050),
    (2.0, 0.0, 0.050),
    (1.0, 2.0, 0.025),
    (2.0, 1.0, 0.025),
)
DEFAULT_CCF_GRID = np.asarray([0.10, 0.25, 0.50, 0.75, 1.00])


@dataclass(frozen=True)
class SampleCounts:
    """Read counts parsed from one single-sample VCF record."""

    genotype: str
    ref_count: int
    alt_count: int
    depth: int


def parse_sample_counts(format_field: str, sample_field: str) -> SampleCounts | None:
    """Parse GT/AD/DP and return biallelic counts, or ``None`` without usable AD.

    Presence is deliberately based on ``AD[1] > 0``.  A VCF record alone is not
    evidence that the ALT allele was observed; callers commonly emit reference
    or zero-depth records at candidate loci.
    """

    keys = str(format_field).split(":")
    values = str(sample_field).split(":")
    fields = dict(zip(keys, values))
    ad = fields.get("AD")
    if not ad or ad == ".":
        return None
    tokens = ad.split(",")
    if len(tokens) != 2 or any(token in {"", "."} for token in tokens):
        return None
    try:
        ref_count, alt_count = (int(tokens[0]), int(tokens[1]))
    except ValueError:
        return None
    if ref_count < 0 or alt_count < 0:
        return None
    ad_depth = ref_count + alt_count
    try:
        reported_depth = int(fields.get("DP", ad_depth))
    except (TypeError, ValueError):
        reported_depth = ad_depth
    # AD is the relevant biallelic depth; retain a larger reported DP only as
    # metadata because reads assigned to other alleles cannot enter this model.
    return SampleCounts(
        genotype=fields.get("GT", "."),
        ref_count=ref_count,
        alt_count=alt_count,
        depth=max(ad_depth, reported_depth, 0),
    )


def expected_somatic_vaf(
    purity: np.ndarray,
    copy_a: np.ndarray,
    copy_b: np.ndarray,
    ccf: float,
    multiplicity: np.ndarray,
) -> np.ndarray:
    """Expected VAF for a tumor mutation under allele-specific copy number."""

    total_cn = copy_a + copy_b
    denominator = 2.0 * (1.0 - purity) + purity * total_cn
    return np.divide(
        purity * ccf * multiplicity,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def expected_germline_vaf(
    purity: np.ndarray,
    copy_a: np.ndarray,
    copy_b: np.ndarray,
    allele: str,
) -> np.ndarray:
    """Expected VAF for a heterozygous constitutional allele A or B."""

    tumor_alt = copy_a if allele == "A" else copy_b
    total_cn = copy_a + copy_b
    denominator = 2.0 * (1.0 - purity) + purity * total_cn
    numerator = (1.0 - purity) + purity * tumor_alt
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, 0.5, dtype=float),
        where=denominator > 0,
    )


def _beta_binomial_logpmf(y: np.ndarray, n: np.ndarray, q: np.ndarray, kappa: float) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 1e-5, 1.0 - 1e-5)
    alpha = q * kappa
    beta = (1.0 - q) * kappa
    return (
        gammaln(n + 1.0)
        - gammaln(y + 1.0)
        - gammaln(n - y + 1.0)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )


def _state_prior_arrays(
    copy_a: np.ndarray,
    copy_b: np.ndarray,
    cnv_prior: Sequence[tuple[float, float, float]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    missing = ~(np.isfinite(copy_a) & np.isfinite(copy_b))
    if not np.any(missing):
        return [(copy_a, copy_b, 0.0)]
    states = []
    for prior_a, prior_b, weight in cnv_prior:
        a = np.where(missing, prior_a, copy_a)
        b = np.where(missing, prior_b, copy_b)
        states.append((a, b, float(np.log(weight))))
    return states


def _marginal_spot_loglik(
    y: np.ndarray,
    n: np.ndarray,
    purity: np.ndarray,
    purity_prior: np.ndarray,
    q_builder,
    kappa: float,
) -> float:
    """Sum spot likelihoods while integrating each missing purity value."""

    known = np.isfinite(purity)
    result = 0.0
    if np.any(known):
        known_indices = np.flatnonzero(known)
        q = q_builder(purity[known], known_indices)
        result += float(np.sum(_beta_binomial_logpmf(y[known], n[known], q, kappa)))
    missing_indices = np.flatnonzero(~known)
    if missing_indices.size:
        # Matrix shape = missing spots x purity-grid points.  Vectorization is
        # essential for P6, where most CalicoST purity entries can be missing.
        purity_matrix = np.broadcast_to(
            purity_prior, (len(missing_indices), len(purity_prior))
        )
        q = q_builder(purity_matrix, missing_indices)
        emissions = _beta_binomial_logpmf(
            y[missing_indices, None], n[missing_indices, None], q, kappa
        )
        result += float(np.sum(logsumexp(emissions, axis=1) - np.log(len(purity_prior))))
    return result


def score_mechanistic_vaf(
    alt_counts: Iterable[int],
    total_counts: Iterable[int],
    purities: Iterable[float],
    copy_a: Iterable[float],
    copy_b: Iterable[float],
    *,
    purity_prior: Iterable[float] = DEFAULT_PURITY_PRIOR,
    cnv_prior: Sequence[tuple[float, float, float]] = DEFAULT_CNV_PRIOR,
    ccf_grid: Iterable[float] = DEFAULT_CCF_GRID,
    kappa: float = 8.0,
    score_scale: float = 10.0,
) -> Mapping[str, float | str]:
    """Return a marginal somatic-vs-germline VAF log Bayes factor.

    Missing purity is marginalized independently using ``purity_prior``.
    Missing allele-specific copy number is marginalized over ``cnv_prior``.
    The returned bounded score is shrunk toward 0.5 when depth, purity or CNV
    context is weak; ``log_bf`` is retained for transparent downstream audits.
    """

    y = np.asarray(list(alt_counts), dtype=float)
    n = np.asarray(list(total_counts), dtype=float)
    p = np.asarray(list(purities), dtype=float)
    a0 = np.asarray(list(copy_a), dtype=float)
    b0 = np.asarray(list(copy_b), dtype=float)
    if not (len(y) == len(n) == len(p) == len(a0) == len(b0)):
        raise ValueError("All mechanistic VAF arrays must have equal length")
    valid = np.isfinite(y) & np.isfinite(n) & (n > 0) & (y >= 0) & (y <= n)
    y, n, p, a0, b0 = y[valid], n[valid], p[valid], a0[valid], b0[valid]
    if len(y) == 0:
        return {
            "log_bf": float("nan"), "score": 0.5, "information": 0.0,
            "context_reliability": 0.0, "observed_vaf": float("nan"),
            "somatic_vaf_low": float("nan"), "somatic_vaf_high": float("nan"),
            "germline_vaf_low": float("nan"), "germline_vaf_high": float("nan"),
            "cnv_scenario": "unobserved", "covered_spots": 0.0, "total_depth": 0.0,
        }

    purity_prior = np.clip(np.asarray(list(purity_prior), dtype=float), 0.0, 1.0)
    purity_prior = purity_prior[np.isfinite(purity_prior)]
    if purity_prior.size == 0:
        purity_prior = DEFAULT_PURITY_PRIOR.copy()
    ccf_grid = np.asarray(list(ccf_grid), dtype=float)
    ccf_grid = ccf_grid[(ccf_grid > 0) & (ccf_grid <= 1)]
    if ccf_grid.size == 0:
        raise ValueError("ccf_grid must contain values in (0, 1]")

    state_hypotheses = _state_prior_arrays(a0, b0, cnv_prior)
    somatic_loglikes: list[float] = []
    somatic_expectations: list[float] = []
    somatic_logweights: list[float] = []
    germline_loglikes: list[float] = []
    germline_expectations: list[float] = []
    germline_logweights: list[float] = []

    def representative_purity() -> np.ndarray:
        return np.where(np.isfinite(p), p, float(np.median(purity_prior)))

    for copy_a_state, copy_b_state, state_logweight in state_hypotheses:
        # Collapse biologically equivalent mutation multiplicities while
        # retaining allele/orientation prior mass.
        mutation_modes: dict[tuple[float, ...], float] = {}
        for allele_copy in (copy_a_state, copy_b_state):
            if np.nanmax(allele_copy) <= 0:
                continue
            max_copy = max(1, int(np.nanmax(allele_copy)))
            for multiplicity in range(1, max_copy + 1):
                # A deleted allele contributes zero mutant copies in the spots
                # carrying that state; an orientation deleted everywhere is
                # skipped above.
                mult = np.minimum(float(multiplicity), np.maximum(allele_copy, 0.0))
                key = tuple(mult.tolist())
                mutation_modes[key] = mutation_modes.get(key, 0.0) + 1.0
        total_mode_mass = sum(mutation_modes.values())

        for mult_key, mode_mass in mutation_modes.items():
            multiplicity = np.asarray(mult_key, dtype=float)
            for ccf in ccf_grid:
                def somatic_q(p_values, mask, ccf=ccf, multiplicity=multiplicity):
                    selected_a = copy_a_state[mask]
                    selected_b = copy_b_state[mask]
                    selected_m = multiplicity[mask]
                    if np.ndim(p_values) > 1:
                        selected_a = selected_a[:, None]
                        selected_b = selected_b[:, None]
                        selected_m = selected_m[:, None]
                    return expected_somatic_vaf(
                        p_values,
                        np.broadcast_to(selected_a, np.shape(p_values)),
                        np.broadcast_to(selected_b, np.shape(p_values)),
                        float(ccf),
                        np.broadcast_to(selected_m, np.shape(p_values)),
                    )

                somatic_loglikes.append(
                    _marginal_spot_loglik(y, n, p, purity_prior, somatic_q, kappa)
                )
                somatic_logweights.append(
                    state_logweight + np.log(mode_mass / total_mode_mass) - np.log(len(ccf_grid))
                )
                rep_p = representative_purity()
                somatic_expectations.extend(
                    expected_somatic_vaf(rep_p, copy_a_state, copy_b_state, float(ccf), multiplicity).tolist()
                )

        for allele in ("A", "B"):
            def germline_q(p_values, mask, allele=allele):
                selected_a = copy_a_state[mask]
                selected_b = copy_b_state[mask]
                if np.ndim(p_values) > 1:
                    selected_a = selected_a[:, None]
                    selected_b = selected_b[:, None]
                return expected_germline_vaf(
                    p_values,
                    np.broadcast_to(selected_a, np.shape(p_values)),
                    np.broadcast_to(selected_b, np.shape(p_values)),
                    allele,
                )

            germline_loglikes.append(
                _marginal_spot_loglik(y, n, p, purity_prior, germline_q, kappa)
            )
            germline_logweights.append(state_logweight - np.log(2.0))
            rep_p = representative_purity()
            germline_expectations.extend(
                expected_germline_vaf(rep_p, copy_a_state, copy_b_state, allele).tolist()
            )

    somatic_marginal = float(logsumexp(np.asarray(somatic_loglikes) + somatic_logweights))
    germline_marginal = float(logsumexp(np.asarray(germline_loglikes) + germline_logweights))
    log_bf = somatic_marginal - germline_marginal

    total_depth = float(np.sum(n))
    covered_spots = float(len(n))
    data_information = (1.0 - np.exp(-total_depth / 10.0)) * min(1.0, covered_spots / 3.0)
    purity_fraction = float(np.mean(np.isfinite(p)))
    cnv_fraction = float(np.mean(np.isfinite(a0) & np.isfinite(b0)))
    context_reliability = 0.5 * purity_fraction + 0.5 * cnv_fraction
    raw_score = float(expit(log_bf / max(float(score_scale), 1e-6)))
    shrinkage = data_information * context_reliability
    bounded_score = 0.5 + (raw_score - 0.5) * shrinkage

    known_states = list(zip(a0[np.isfinite(a0) & np.isfinite(b0)], b0[np.isfinite(a0) & np.isfinite(b0)]))
    if not known_states:
        cnv_scenario = "missing"
    elif any(a == 0 or b == 0 for a, b in known_states):
        cnv_scenario = "loh"
    elif any(a + b > 2 for a, b in known_states):
        cnv_scenario = "gain"
    elif any(a + b < 2 for a, b in known_states):
        cnv_scenario = "loss"
    else:
        cnv_scenario = "diploid"

    return {
        "log_bf": log_bf,
        "score": float(bounded_score),
        "information": float(data_information),
        "context_reliability": float(context_reliability),
        "observed_vaf": float(np.sum(y) / total_depth),
        "somatic_vaf_low": float(np.quantile(somatic_expectations, 0.05)),
        "somatic_vaf_high": float(np.quantile(somatic_expectations, 0.95)),
        "germline_vaf_low": float(np.quantile(germline_expectations, 0.05)),
        "germline_vaf_high": float(np.quantile(germline_expectations, 0.95)),
        "cnv_scenario": cnv_scenario,
        "covered_spots": covered_spots,
        "total_depth": total_depth,
        "purity_fraction": purity_fraction,
        "cnv_fraction": cnv_fraction,
    }
