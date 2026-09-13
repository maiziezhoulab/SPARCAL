import pandas as pd

from scripts.spatial_filter_rebuild.score_quality_balanced import constrained_selection


def test_constrained_selection_respects_depth_budget_and_xmhc_veto():
    frame = pd.DataFrame({
        "variant_key": ["a", "b", "c", "d", "e"],
        "race": ["denovo"] * 5,
        "ref": ["A"] * 5,
        "alt": ["T"] * 5,
        "total_alt": [1] * 5,
        "total_depth_cache": [1, 1, 8, 10, 12],
        "xmhc": [False, False, True, False, False],
        "consensus_score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "consensus_reliability_score": [0.1, 0.2, 0.9, 0.8, 0.7],
    })
    selected = constrained_selection(
        frame, {"a", "b", "c"}, "consensus_score", "consensus_reliability_score", 3, 1
    )
    assert len(selected) == 3
    assert selected["xmhc"].sum() == 0
    assert selected["total_depth_cache"].lt(3).sum() <= 1
    assert "a" in set(selected["variant_key"])
