import numpy as np

from scripts.spatial_filter_rebuild.statistics import (
    STATE_NAMES,
    effective_somatic_probability,
    posterior_from_logits,
    state_log_priors,
    vaf_state_logliks,
)


def test_state_likelihoods_prefer_expected_vaf_regimes():
    y = np.array([0, 50, 98, 10, 45])
    n = np.full(5, 100)
    low = np.full(5, 0.08)
    high = np.full(5, 0.45)
    gl = np.full(5, 0.40)
    gh = np.full(5, 0.60)
    ll = vaf_state_logliks(y, n, low, high, gl, gh)
    assert ll.shape == (5, len(STATE_NAMES))
    assert STATE_NAMES[int(ll[0].argmax())] == "artifact"
    assert STATE_NAMES[int(ll[1].argmax())] == "germline_het"
    assert STATE_NAMES[int(ll[2].argmax())] == "germline_hom_alt"
    assert STATE_NAMES[int(ll[3].argmax())] == "somatic_subclonal"
    assert STATE_NAMES[int(ll[4].argmax())] == "somatic_clonal"


def test_panel_is_soft_germline_prior():
    priors = np.exp(state_log_priors([False, True]))
    assert priors[1, 1] > priors[0, 1]
    assert priors[1, 3] > 0
    assert np.allclose(priors.sum(axis=1), 1)


def test_low_information_shrinks_to_unresolved_half():
    posterior = posterior_from_logits(np.array([[0, 0, 0, 8, 8]], dtype=float))
    score, information = effective_somatic_probability(posterior, [1], [1])
    assert 0 < information[0] < 0.1
    assert 0.5 < score[0] < 0.55
