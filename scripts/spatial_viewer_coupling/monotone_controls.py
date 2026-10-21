"""Label-free monotone controls and explicitly declared detector score paths."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata
from . import model

NULLS = ('rank_coverage', 'spline_coverage')
KNOT_PROBABILITIES = np.array([0., .10, .25, .50, .75, .90, 1.])


def coverage_controls(coverage):
    c = np.asarray(coverage, dtype=float)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('coverage must be a finite nonnegative vector')
    x = np.log1p(c)
    raw_knots = np.quantile(x, KNOT_PROBABILITIES)
    knots, inverse = np.unique(raw_knots, return_inverse=True)
    ordinates = np.bincount(inverse, weights=KNOT_PROBABILITIES) / np.bincount(inverse)
    spline = (np.full_like(x, ordinates[0]) if len(knots) == 1 else
              PchipInterpolator(knots, ordinates, extrapolate=False)(x))
    ranks = rankdata(c, method='average')
    if not np.array_equal(rankdata(spline, method='average'), ranks):
        raise ValueError('spline did not preserve coverage ranks')
    return dict(rank_coverage=ranks, spline_coverage=spline), dict(
        input='log1p(smoothed_coverage)', knot_probabilities=KNOT_PROBABILITIES.tolist(),
        raw_abscissae=raw_knots.tolist(), merged_abscissae=knots.tolist(),
        merged_ordinates=ordinates.tolist(), interpolation='PCHIP', gt_fit=False)


def is_burden(name):
    return name.endswith('_matched_burden') or (name not in ('smooth_coverage', *NULLS) and '_k' not in name)


def transform_chain(name):
    burden = is_burden(name)
    native = ('ALT_positive_selected_call_burden' if burden else
              {'smooth_coverage': 'H(UMI)', 'rank_coverage': 'rankdata(H(UMI))',
               'spline_coverage': 'PCHIP_CDF(log1p(H(UMI)))'}.get(name, 'frozen_B.tumor_prior'))
    return dict(native_score=native, raw_chain='H(x)' if burden else 'x',
                residual_chain='H(OLS_residual(x ~ 1 + raw_UMI))')


def detector_scores(name, score, umi, graph):
    x = np.asarray(score, dtype=float)
    if x.shape != np.asarray(umi).shape or not np.isfinite(x).all():
        raise ValueError('score shape/nonfinite')
    return dict(raw=model.smooth(x, graph) if is_burden(name) else x.copy(),
                coverage_residual=model.smooth(model.residual(x, umi), graph))


def prepare_scores(scores, umi, graph):
    if set(NULLS) & set(scores):
        raise ValueError('null names already present')
    coverage = model.smooth(umi, graph)
    if not np.array_equal(coverage, scores['smooth_coverage']):
        raise ValueError('cached coverage disagrees with frozen UMI/graph/runtime')
    nulls, spline = coverage_controls(coverage)
    complete = {**scores, **nulls}
    processed = {name: detector_scores(name, x, umi, graph) for name, x in complete.items()}
    return complete, processed, spline
