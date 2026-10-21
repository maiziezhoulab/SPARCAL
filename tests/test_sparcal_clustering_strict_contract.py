import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_ROOT = ROOT / "clustering_benchmark"
if str(CLUSTERING_ROOT) not in sys.path:
    sys.path.insert(0, str(CLUSTERING_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "sparcal_clustering_strict_contract", CLUSTERING_ROOT / "SPARCAL_clustering.py"
)
clustering = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clustering)


def test_embedding_rank_is_raw_and_need_not_be_at_least_g():
    coordinate = np.linspace(-1.0, 1.0, 30)
    embedding = np.zeros((30, 30), dtype=float)
    embedding[:, 0] = coordinate
    embedding[:, 1] = coordinate ** 2
    projected, spectrum = clustering.prepare_mclust_embedding(embedding, 1e-6)
    assert spectrum["effective_rank"] == 2
    assert projected.shape == (30, 2)
    assert spectrum["embedding_shape"] == [30, 30]
    assert spectrum["relative_eigenvalue_threshold"] == 1e-6


def test_rank_one_is_not_artificially_floored():
    embedding = np.zeros((30, 30), dtype=float)
    embedding[:, 0] = np.linspace(-1.0, 1.0, 30)
    projected, spectrum = clustering.prepare_mclust_embedding(embedding, 1e-6)
    assert spectrum["effective_rank"] == 1
    assert projected.shape == (30, 1)


def test_strict_source_has_fixed_model_and_auditable_artifacts():
    source = inspect.getsource(clustering)
    assert 'strict_mclust_contract' in source
    assert 'mclust_fit_contract.npz' in source
    assert 'embedding_spectrum.json' in source
    assert 'failure_metadata.json' in source
    assert 'ari.txt is the cache/completion marker and must be written last' in source
    assert 'strict mclust EEE fit failed:' in source
    assert 'strict mclust EEE validation failed:' in source
    assert 'n_keep < int(num_cluster)' not in source
    assert 'max(int(keep.sum()), 2)' not in source


def test_legacy_call_signature_remains_backward_compatible():
    parameters = inspect.signature(clustering.mclust_R).parameters
    assert parameters["model_name"].default == "EEE"
    assert parameters["random_seed"].default == 0
    assert parameters["strict_mclust_contract"].default is False
