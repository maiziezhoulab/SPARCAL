import numpy as np
from scipy import sparse

from scripts.spatial_filter_rebuild.compute_features import (
    coverage_conditioned_spatial,
    local_density,
)


def test_coverage_conditioned_join_count_prefers_adjacent_pattern():
    graph = sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    )
    coverage = sparse.csr_matrix(np.ones((2, 4), dtype=np.uint8))
    presence = sparse.csr_matrix(
        np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.uint8)
    )
    result = coverage_conditioned_spatial(presence, coverage, graph)
    assert result["spatial_join_count"].tolist() == [1.0, 0.0]
    assert result["spatial_log_enrichment"][0] > result["spatial_log_enrichment"][1]


def test_local_density_is_chromosome_specific():
    values = local_density(["1", "1", "1", "2"], [10, 12, 100, 11], 5)
    assert values.tolist() == [2, 2, 1, 1]
