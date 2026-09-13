#!/usr/bin/env python3
"""Vectorized, training-free likelihoods and evidence transformations.

The state model deliberately has no learned coefficient.  Every coefficient is
declared here and is shared by all sections.  WES and pathology labels are not
accepted by this module.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, expit, gammaln, logsumexp


STATE_NAMES = (
    "artifact",
    "germline_het",
    "germline_hom_alt",
    "somatic_subclonal",
    "somatic_clonal",
)


def beta_binomial_logpmf(y, n, q, kappa: float = 8.0):
    """Broadcasting beta-binomial log PMF with bounded expectations."""

    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), 1e-5, 1 - 1e-5)
    alpha = q * float(kappa)
    beta = (1.0 - q) * float(kappa)
    return (
        gammaln(n + 1)
        - gammaln(y + 1)
        - gammaln(n - y + 1)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )


def mixture_loglik(y, n, expectations, kappa: float = 8.0):
    """Equal-weight marginal likelihood over the last expectation axis."""

    q = np.asarray(expectations, dtype=float)
    ll = beta_binomial_logpmf(
        np.asarray(y, dtype=float)[..., None],
        np.asarray(n, dtype=float)[..., None],
        q,
        kappa=kappa,
    )
    return logsumexp(ll, axis=-1) - np.log(q.shape[-1])


def _bounded_interval(low, high, fallback):
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    lo = np.where(ok, low, fallback)
    hi = np.where(ok, high, fallback)
    return np.clip(lo, 1e-4, 1 - 1e-4), np.clip(hi, 1e-4, 1 - 1e-4)


def vaf_state_logliks(y, n, somatic_low, somatic_high, germline_low, germline_high):
    """Five biological observation models from pooled ALT/total depth.

    CNV/purity-aware expectation intervals come from the independently tested
    spot-level mechanistic model.  Missing intervals fall back to broad,
    biologically explicit values; they are never replaced by purity=0 or CN=2.
    """

    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    slo, shi = _bounded_interval(somatic_low, somatic_high, 0.25)
    glo, ghi = _bounded_interval(germline_low, germline_high, 0.50)

    artifact_q = np.broadcast_to(
        np.asarray([0.001, 0.005, 0.01, 0.02, 0.05]), (len(y), 5)
    )
    germline_het_q = np.column_stack([glo, (glo + ghi) / 2, ghi, np.full(len(y), 0.5)])
    germline_hom_q = np.broadcast_to(
        np.asarray([0.90, 0.97, 0.99, 0.999]), (len(y), 4)
    )
    somatic_sub_q = np.column_stack([
        slo,
        slo + 0.25 * (shi - slo),
        slo + 0.50 * (shi - slo),
        slo + 0.75 * (shi - slo),
    ])
    somatic_clonal_q = np.column_stack([
        np.maximum(slo, 0.75 * shi),
        shi,
        np.minimum(0.999, 1.15 * shi),
    ])
    return np.column_stack([
        mixture_loglik(y, n, artifact_q, kappa=20.0),
        mixture_loglik(y, n, germline_het_q, kappa=8.0),
        mixture_loglik(y, n, germline_hom_q, kappa=12.0),
        mixture_loglik(y, n, somatic_sub_q, kappa=8.0),
        mixture_loglik(y, n, somatic_clonal_q, kappa=8.0),
    ])


def state_log_priors(is_panel):
    """Population-panel membership is a soft prior, never a hard gate."""

    panel = np.asarray(is_panel, dtype=bool)
    denovo = np.asarray([0.72, 0.16, 0.02, 0.08, 0.02])
    defined = np.asarray([0.08, 0.72, 0.16, 0.03, 0.01])
    return np.log(np.where(panel[:, None], defined[None, :], denovo[None, :]))


def posterior_from_logits(logits):
    logits = np.asarray(logits, dtype=float)
    norm = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - norm)


def effective_somatic_probability(posterior, total_depth, positive_spots):
    """Somatic posterior shrunk to uncertainty when observation information is low."""

    posterior = np.asarray(posterior, dtype=float)
    raw = posterior[:, 3] + posterior[:, 4]
    depth_info = 1.0 - np.exp(-np.asarray(total_depth, dtype=float) / 10.0)
    spot_info = np.minimum(1.0, np.asarray(positive_spots, dtype=float) / 3.0)
    information = depth_info * spot_info
    return 0.5 + (raw - 0.5) * information, information


def entropy(posterior):
    p = np.clip(np.asarray(posterior, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def bounded_evidence(value, scale: float = 1.0):
    """Bound arbitrary signed evidence without allowing one feature to dominate."""

    return 2.0 * expit(np.asarray(value, dtype=float) / float(scale)) - 1.0

