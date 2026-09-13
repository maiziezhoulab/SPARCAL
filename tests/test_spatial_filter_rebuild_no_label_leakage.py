import numpy as np
import pandas as pd

from scripts.spatial_filter_rebuild.score_models import score_frame


def test_source_class_does_not_change_new_scores():
    frame = pd.DataFrame({
        "race": ["denovo", "denovo", "defined"],
        "source_classification": ["somatic", "ambiguous", "germline"],
        "total_alt": [3, 2, 8],
        "total_depth_cache": [20, 20, 16],
        "positive_spots": [3, 2, 5],
        "local_density_10bp": [1, 2, 1],
        "rna_editing_change": [False, True, False],
        "xmhc": [False, False, False],
        "patient_rep_count": [2, 1, 2],
        "cross_patient_count": [1, 2, 1],
        "spatial_information": [0.8, 0.5, 0.8],
        "spatial_log_enrichment": [1.0, -0.5, 0.1],
        "spatial_neighbor_precision": [0.8, 0.1, 0.3],
        "clone_information": [0.8, 0.2, 0.8],
        "clone_log_odds": [1.0, -1.0, 0.0],
        "vaf_log_bf": [4.0, -2.0, -5.0],
        "vaf_information": [0.8, 0.5, 0.9],
        "vaf_context_reliability": [1.0, 1.0, 1.0],
        "vaf_somatic_low": [0.05, 0.05, 0.05],
        "vaf_somatic_high": [0.35, 0.35, 0.35],
        "vaf_germline_low": [0.4, 0.4, 0.4],
        "vaf_germline_high": [0.6, 0.6, 0.6],
    })
    first = score_frame(frame)
    shuffled = frame.copy()
    shuffled["source_classification"] = ["germline", "somatic", "ambiguous"]
    second = score_frame(shuffled)
    pd.testing.assert_frame_equal(first, second)
