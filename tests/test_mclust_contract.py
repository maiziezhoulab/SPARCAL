import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clustering_benchmark"
    / "mclust_contract.py"
)
SPEC = importlib.util.spec_from_file_location("mclust_contract", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_fit():
    n, g, d = 8, 3, 2
    posterior = np.array([
        [0.90, 0.08, 0.02],
        [0.80, 0.15, 0.05],
        [0.10, 0.85, 0.05],
        [0.05, 0.90, 0.05],
        [0.02, 0.08, 0.90],
        [0.05, 0.10, 0.85],
        [0.70, 0.20, 0.10],
        [0.10, 0.20, 0.70],
    ])
    covariance = np.array([[2.0, 0.2], [0.2, 1.0]])
    return {
        "requested_g": g,
        "n_spots": n,
        "n_dimensions": d,
        "model_name": "EEE",
        "fitted_g": g,
        "fitted_n": n,
        "fitted_d": d,
        "return_code": 0,
        "loglik": -100.0,
        "df": 17.0,
        "bic": -240.0,
        "icl": -250.0,
        "selected_bic": -240.0,
        "proportions": np.array([0.375, 0.25, 0.375]),
        "means": np.arange(d * g, dtype=float).reshape(d, g),
        "covariance": covariance,
        "component_covariances": np.repeat(covariance[:, :, None], g, axis=2),
        "posterior": posterior,
        "classification": np.argmax(posterior, axis=1) + 1,
        "uncertainty": 1.0 - posterior.max(axis=1),
    }


class MclustContractTests(unittest.TestCase):
    def test_more_components_than_dimensions_is_valid(self):
        fit = valid_fit()
        self.assertLess(fit["n_dimensions"], fit["requested_g"])
        diagnostics = MODULE.validate_eee_fit(**fit)
        self.assertEqual(diagnostics["return_code"], 0)
        self.assertEqual(diagnostics["minimum_cluster_size"], 2)
        self.assertGreater(diagnostics["covariance_condition_number"], 1)

    def test_nonzero_mclust_return_code_is_rejected(self):
        fit = valid_fit()
        fit["return_code"] = -1
        with self.assertRaisesRegex(RuntimeError, "return code"):
            MODULE.validate_eee_fit(**fit)

    def test_non_positive_definite_covariance_is_rejected(self):
        fit = valid_fit()
        covariance = np.array([[1.0, 2.0], [2.0, 1.0]])
        fit["covariance"] = covariance
        fit["component_covariances"] = np.repeat(
            covariance[:, :, None], fit["requested_g"], axis=2
        )
        with self.assertRaisesRegex(RuntimeError, "positive definite"):
            MODULE.validate_eee_fit(**fit)

    def test_missing_component_is_rejected(self):
        fit = valid_fit()
        posterior = np.asarray(fit["posterior"]).copy()
        posterior[:, 2] = 0
        posterior[:, :2] /= posterior[:, :2].sum(axis=1, keepdims=True)
        fit["posterior"] = posterior
        fit["classification"] = np.argmax(posterior, axis=1) + 1
        fit["uncertainty"] = 1.0 - posterior.max(axis=1)
        with self.assertRaisesRegex(RuntimeError, "exactly G non-empty"):
            MODULE.validate_eee_fit(**fit)

    def test_classification_must_equal_posterior_argmax(self):
        fit = valid_fit()
        labels = np.asarray(fit["classification"]).copy()
        labels[0] = 2
        fit["classification"] = labels
        with self.assertRaisesRegex(RuntimeError, "posterior argmax"):
            MODULE.validate_eee_fit(**fit)


if __name__ == "__main__":
    unittest.main()
