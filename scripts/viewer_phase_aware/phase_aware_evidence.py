#!/usr/bin/env python3
"""Training-free phase/CNV/purity evidence for tumor-spot selection.

The functions in this module operate on label-free molecular observations.
Missing purity and copy-number values are marginalized over explicit priors.
Phased allele labels are never assumed to match copy-number allele labels
unless the caller supplies an orientation probability greater than 0.5.
"""

from __future__ import annotations

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
DEFAULT_COMPONENT_CAPS = {
    "phase_cnv_log_bf": 8.0,
    "purity_log_prior": 6.0,
    "somatic_vaf_log_bf": 8.0,
    "molecular_log_bf": 6.0,
    "spatial_log_prior": 4.0,
}


def expected_phased_het_fraction(
    purity: np.ndarray | float,
    copy_a: np.ndarray | float,
    copy_b: np.ndarray | float,
    *,
    counted_allele: str = "B",
) -> np.ndarray:
    """Expected fraction of a phased constitutional allele in mixed tissue.

    Normal cells contribute one copy of each allele.  Tumor cells contribute
    the allele-specific copy number supplied by CalicoST (or another caller).
    """

    p, a, b = np.broadcast_arrays(
        np.asarray(purity, float), np.asarray(copy_a, float), np.asarray(copy_b, float)
    )
    if counted_allele not in {"A", "B"}:
        raise ValueError("counted_allele must be A or B")
    if np.any((p < 0) | (p > 1)) or np.any(a < 0) or np.any(b < 0):
        raise ValueError("purity must be in [0,1] and copy numbers nonnegative")
    tumor_counted = a if counted_allele == "A" else b
    denominator = 2.0 * (1.0 - p) + p * (a + b)
    return np.divide(
        (1.0 - p) + p * tumor_counted,
        denominator,
        out=np.full(p.shape, 0.5, dtype=float),
        where=denominator > 0,
    )


def beta_binomial_logpmf(
    successes: np.ndarray | float,
    totals: np.ndarray | float,
    expected_fraction: np.ndarray | float,
    *,
    kappa: float = 19.0,
) -> np.ndarray:
    """Beta-binomial log PMF with mean ``expected_fraction``."""

    y, n, q = np.broadcast_arrays(
        np.asarray(successes, float),
        np.asarray(totals, float),
        np.asarray(expected_fraction, float),
    )
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if np.any(y < 0) or np.any(n < y):
        raise ValueError("counts require 0 <= successes <= totals")
    q = np.clip(q, 1e-5, 1.0 - 1e-5)
    alpha, beta = q * float(kappa), (1.0 - q) * float(kappa)
    return (
        gammaln(n + 1.0)
        - gammaln(y + 1.0)
        - gammaln(n - y + 1.0)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )


def _normalize_weighted_states(
    values: Sequence[tuple[float, ...]], name: str
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError(f"{name} prior must not be empty")
    states = np.asarray([row[:-1] for row in values], float)
    weights = np.asarray([row[-1] for row in values], float)
    if not np.isfinite(states).all() or not np.isfinite(weights).all():
        raise ValueError(f"{name} prior must be finite")
    if np.any(weights <= 0):
        raise ValueError(f"{name} prior weights must be positive")
    weights /= weights.sum()
    return states, weights


def phase_cnv_log_bayes_factor(
    counted_alleles: Iterable[int],
    total_counts: Iterable[int],
    purities: Iterable[float],
    copy_a: Iterable[float],
    copy_b: Iterable[float],
    *,
    orientation_probability: Iterable[float] | None = None,
    purity_prior: Iterable[float] = DEFAULT_PURITY_PRIOR,
    cnv_prior: Sequence[tuple[float, float, float]] = DEFAULT_CNV_PRIOR,
    kappa: float = 19.0,
    min_depth: int = 1,
) -> Mapping[str, np.ndarray]:
    """Per-observation log BF for tumor CNV/purity versus balanced normal.

    ``orientation_probability`` is P(the counted phased allele corresponds to
    copy-number allele B). Use 1 (or 0 for known correspondence to A) only
    when the count and copy-number outputs
    share an upstream phase convention (for example native CalicoST B counts
    and CalicoST A/B copy numbers).  Use 0.5 for Beagle labels that have not
    been aligned to the copy-number phase; this safely marginalizes A/B.

    Missing purity and/or copy number is integrated over the supplied priors.
    Unusable counts contribute exactly zero evidence and are marked invalid.
    """

    y = np.asarray(list(counted_alleles), float)
    n = np.asarray(list(total_counts), float)
    p = np.asarray(list(purities), float)
    a = np.asarray(list(copy_a), float)
    b = np.asarray(list(copy_b), float)
    arrays = (y, n, p, a, b)
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("all phase/CNV arrays must have equal length")
    if orientation_probability is None:
        orientation = np.full(len(y), 0.5)
    else:
        orientation = np.asarray(list(orientation_probability), float)
        if len(orientation) != len(y):
            raise ValueError("orientation_probability length differs")
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("known purity must be in [0,1]")
    if np.any(np.isfinite(a) & (a < 0)) or np.any(np.isfinite(b) & (b < 0)):
        raise ValueError("known copy number must be nonnegative")
    if np.any(~np.isfinite(orientation)) or np.any((orientation < 0) | (orientation > 1)):
        raise ValueError("orientation_probability must be finite in [0,1]")

    valid = (
        np.isfinite(y)
        & np.isfinite(n)
        & (n >= int(min_depth))
        & (y >= 0)
        & (y <= n)
    )
    out = np.zeros(len(y), float)
    expected_mean = np.full(len(y), np.nan)
    purity_grid = np.asarray(list(purity_prior), float)
    purity_grid = purity_grid[np.isfinite(purity_grid) & (purity_grid >= 0) & (purity_grid <= 1)]
    if purity_grid.size == 0:
        raise ValueError("purity_prior must contain values in [0,1]")
    cn_states, cn_weights = _normalize_weighted_states(cnv_prior, "CNV")
    if cn_states.shape[1] != 2 or np.any(cn_states < 0):
        raise ValueError("CNV prior states require nonnegative (copy_a, copy_b, weight)")

    for i in np.flatnonzero(valid):
        p_values = np.asarray([p[i]]) if np.isfinite(p[i]) else purity_grid
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            copy_values = np.asarray([[a[i], b[i]]])
            copy_weights = np.ones(1)
        else:
            # Preserve a known allele when only its partner is missing, then
            # collapse duplicate conditional states while retaining prior mass.
            conditional = cn_states.copy()
            if np.isfinite(a[i]):
                conditional[:, 0] = a[i]
            if np.isfinite(b[i]):
                conditional[:, 1] = b[i]
            collapsed: dict[tuple[float, float], float] = {}
            for state, weight in zip(conditional, cn_weights):
                key = (float(state[0]), float(state[1]))
                collapsed[key] = collapsed.get(key, 0.0) + float(weight)
            copy_values = np.asarray(list(collapsed), float)
            copy_weights = np.asarray(list(collapsed.values()), float)
        hypotheses = []
        weights = []
        for pi in p_values:
            for (ai, bi), cw in zip(copy_values, copy_weights):
                q_b = float(expected_phased_het_fraction(pi, ai, bi, counted_allele="B"))
                q_a = 1.0 - q_b
                hypotheses.extend([q_b, q_a])
                base = float(cw) / len(p_values)
                weights.extend([base * orientation[i], base * (1.0 - orientation[i])])
        hypotheses = np.asarray(hypotheses)
        weights = np.asarray(weights)
        positive = weights > 0
        weights = weights[positive]
        hypotheses = hypotheses[positive]
        weights /= weights.sum()
        tumor_ll = float(
            logsumexp(
                beta_binomial_logpmf(y[i], n[i], hypotheses, kappa=kappa)
                + np.log(weights)
            )
        )
        normal_ll = float(beta_binomial_logpmf(y[i], n[i], 0.5, kappa=kappa))
        out[i] = tumor_ll - normal_ll
        expected_mean[i] = float(np.sum(weights * hypotheses))

    information = np.zeros(len(y), float)
    information[valid] = 1.0 - np.exp(-n[valid] / 8.0)
    phase_available = valid.copy()
    phase_aligned = valid & (np.abs(orientation - 0.5) > 1e-12)
    purity_available = valid & np.isfinite(p)
    cnv_available = valid & np.isfinite(a) & np.isfinite(b)
    return {
        "log_bf": out,
        "information": information,
        "expected_fraction": expected_mean,
        "valid": valid,
        "phase_available": phase_available,
        "phase_aligned": phase_aligned,
        "purity_available": purity_available,
        "cnv_available": cnv_available,
    }


def aggregate_phase_evidence(
    barcodes: Iterable[str],
    log_bf: Iterable[float],
    information: Iterable[float],
    *,
    effective_observation_cap: float = 8.0,
) -> Mapping[str, Mapping[str, float]]:
    """Robustly aggregate possibly correlated phase observations per spot.

    The mean informative log BF is multiplied by at most
    ``effective_observation_cap``.  This prevents thousands of correlated
    bins from creating arbitrarily certain posteriors while retaining the
    sign and strength of replicated evidence.
    """

    names = np.asarray(list(barcodes), str)
    values = np.asarray(list(log_bf), float)
    info = np.asarray(list(information), float)
    if not (len(names) == len(values) == len(info)):
        raise ValueError("phase aggregation arrays must have equal length")
    if effective_observation_cap <= 0:
        raise ValueError("effective_observation_cap must be positive")
    result = {}
    for barcode in dict.fromkeys(names.tolist()):
        mask = (names == barcode) & np.isfinite(values) & np.isfinite(info) & (info > 0)
        if not mask.any():
            result[barcode] = {"log_bf": 0.0, "information": 0.0, "n_observations": 0.0}
            continue
        weights = info[mask]
        effective_n = min(float(weights.sum()), float(effective_observation_cap))
        result[barcode] = {
            "log_bf": float(np.average(values[mask], weights=weights) * effective_n),
            "information": float(1.0 - np.exp(-weights.sum() / 2.0)),
            "n_observations": float(mask.sum()),
        }
    return result


def purity_log_prior_shift(
    purities: Iterable[float],
    *,
    prior_probability: float = 0.20,
    epsilon: float = 1e-3,
) -> Mapping[str, np.ndarray]:
    """Express estimated tumor fraction as an operational occupancy-prior shift.

    This transparent mapping makes the posterior equal the supplied purity when
    it is the only fully informative component. It treats tumor fraction as a
    proxy for tumor-spot probability, not as an independently observed label.
    Missing values contribute zero shift and remain explicitly unavailable.
    """

    values = np.asarray(list(purities), float)
    available = np.isfinite(values)
    if np.any(available & ((values < 0) | (values > 1))):
        raise ValueError("known purity must be in [0,1]")
    if not 0 < prior_probability < 1 or not 0 < epsilon < 0.5:
        raise ValueError("prior_probability and epsilon must be valid probabilities")
    clipped = np.clip(values[available], epsilon, 1.0 - epsilon)
    baseline = np.log(prior_probability / (1.0 - prior_probability))
    shifts = np.zeros(len(values), float)
    shifts[available] = np.log(clipped / (1.0 - clipped)) - baseline
    return {"log_prior": shifts, "available": available}


def combine_tumor_evidence(
    phase_cnv_log_bf: Iterable[float],
    purity_log_prior: Iterable[float],
    somatic_vaf_log_bf: Iterable[float],
    molecular_log_bf: Iterable[float],
    spatial_log_prior: Iterable[float],
    information: Iterable[float],
    *,
    prior_probability: float = 0.20,
    component_caps: Mapping[str, float] = DEFAULT_COMPONENT_CAPS,
) -> Mapping[str, np.ndarray]:
    """Combine calibrated evidence without fitting labels or feature weights.

    Components are interpreted on the natural-log odds scale and therefore
    receive coefficient 1.  Fixed symmetric caps are robustness guards, not
    fitted weights.  Missing component values are neutral (zero evidence).
    ``information`` shrinks the combined evidence, but not the stated prior.
    """

    names = tuple(DEFAULT_COMPONENT_CAPS)
    arrays = {
        name: np.asarray(list(value), float)
        for name, value in zip(
            names,
            (
                phase_cnv_log_bf,
                purity_log_prior,
                somatic_vaf_log_bf,
                molecular_log_bf,
                spatial_log_prior,
            ),
        )
    }
    info = np.asarray(list(information), float)
    if len({len(info), *(len(value) for value in arrays.values())}) != 1:
        raise ValueError("tumor-evidence arrays must have equal length")
    if not 0 < prior_probability < 1:
        raise ValueError("prior_probability must be in (0,1)")
    if np.any(~np.isfinite(info)) or np.any((info < 0) | (info > 1)):
        raise ValueError("information must be finite in [0,1]")
    used = {}
    combined = np.zeros(len(info), float)
    for name, values in arrays.items():
        cap = float(component_caps.get(name, DEFAULT_COMPONENT_CAPS[name]))
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"invalid cap for {name}")
        finite = np.where(np.isfinite(values), values, 0.0)
        used[name] = np.clip(finite, -cap, cap)
        combined += used[name]
    prior_log_odds = float(np.log(prior_probability / (1.0 - prior_probability)))
    posterior_log_odds = prior_log_odds + info * combined
    return {
        "tumor_probability": expit(posterior_log_odds),
        "posterior_log_odds": posterior_log_odds,
        "combined_log_evidence": combined,
        **used,
    }
