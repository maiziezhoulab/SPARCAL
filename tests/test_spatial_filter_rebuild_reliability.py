import pandas as pd

from scripts.spatial_filter_rebuild.score_reliability import reliability_scores


def _frame():
    data = {
        "total_alt": [1, 4],
        "positive_spots": [1, 3],
    }
    for model in ("molecular", "spatial", "calicost", "integrated"):
        data[f"{model}_p_somatic_subclonal"] = [0.30, 0.30]
        data[f"{model}_p_somatic_clonal"] = [0.10, 0.10]
        data[f"{model}_information"] = [0.05, 0.80]
    return pd.DataFrame(data)


def test_reliability_discount_places_informative_candidate_first():
    frame = _frame()
    scored = reliability_scores(frame)
    assert scored.loc[1, "integrated_reliability_score"] > scored.loc[0, "integrated_reliability_score"]


def test_production_labels_cannot_change_reliability_scores():
    frame = _frame()
    first = reliability_scores(frame)
    frame["source_classification"] = ["somatic", "germline"]
    second = reliability_scores(frame)
    pd.testing.assert_frame_equal(first, second)
