import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = (
    ROOT / "clustering_benchmark"
    / "beagle_pnonref_strict_noninferiority_2026_09_13"
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATE = load_module("strict_dlpfc_validate", "validate_strict_run.py")
VERIFY = load_module("strict_dlpfc_verify", "verify_strict_inputs.py")


def write_successful_run(run_root: Path) -> dict[str, object]:
    n_spots, n_clusters = 14, 7
    truth = np.tile(np.arange(1, n_clusters + 1), 2)
    labels = pd.DataFrame(
        {"original_clusters": truth, "mclust": truth},
        index=[f"spot{i}" for i in range(n_spots)],
    )
    labels.to_csv(run_root / "cluster_labels.csv")
    (run_root / "ari.txt").write_text("1.0\n")

    coordinate = np.linspace(-1.0, 1.0, n_spots)
    embedding = np.zeros((n_spots, 30), dtype=float)
    embedding[:, 0] = coordinate
    embedding[:, 1] = coordinate ** 2
    np.save(run_root / "embedding.npy", embedding)
    spectrum = VALIDATE.recompute_embedding_spectrum(embedding)
    spectrum["eigenvalues"] = spectrum["eigenvalues"].tolist()
    (run_root / "embedding_spectrum.json").write_text(json.dumps(spectrum))

    posterior = np.full((n_spots, n_clusters), 0.01, dtype=float)
    posterior[np.arange(n_spots), truth - 1] = 0.94
    covariance = np.array([[2.0, 0.2], [0.2, 1.0]])
    fit = {
        "requested_g": n_clusters,
        "n_spots": n_spots,
        "n_dimensions": 2,
        "model_name": "EEE",
        "fitted_g": n_clusters,
        "fitted_n": n_spots,
        "fitted_d": 2,
        "return_code": 0,
        "loglik": -100.0,
        "df": 30.0,
        "bic": -250.0,
        "icl": -260.0,
        "selected_bic": -250.0,
        "proportions": np.full(n_clusters, 1.0 / n_clusters),
        "means": np.arange(2 * n_clusters, dtype=float).reshape(2, n_clusters),
        "covariance": covariance,
        "component_covariances": np.repeat(
            covariance[:, :, None], n_clusters, axis=2
        ),
        "posterior": posterior,
        "classification": truth,
        "uncertainty": 1.0 - posterior.max(axis=1),
    }
    np.savez(run_root / "mclust_fit_contract.npz", **fit)
    diagnostics = VALIDATE.validate_eee_fit(**fit)
    metadata = {
        "modality": "legacy_any_panel",
        "run": 2,
        "seed": 2,
        "requested_model": "EEE",
        "selected_model": "EEE",
        "fallback_used": False,
        "mclust_fit_contract": "fixed_eee_numerical_v1",
        "strict_mclust_contract": True,
        "effective_rank": spectrum["effective_rank"],
        "embedding_dimensions": 30,
        "rank_relative_eigenvalue_threshold": 1e-6,
        "projected_condition_number": spectrum["projected_condition_number"],
        **{f"mclust_{key}": value for key, value in diagnostics.items()},
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata))
    return metadata


class StrictDlpfcFreezeTests(unittest.TestCase):
    def test_successful_run_contract_recomputes_fit_with_rank_below_g(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            metadata = write_successful_run(run_root)
            self.assertEqual(metadata["effective_rank"], 2)
            self.assertLess(metadata["effective_rank"], 7)
            ari, n_labels = VALIDATE.validate_successful_run(
                run_root, "legacy_any_panel", 2
            )
            self.assertEqual(ari, 1.0)
            self.assertEqual(n_labels, 14)
            metadata["fallback_used"] = True
            (run_root / "run_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "fallback_used"):
                VALIDATE.validate_successful_run(
                    run_root, "legacy_any_panel", 2
                )

    def test_spectrum_and_fit_contract_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            write_successful_run(run_root)
            spectrum_path = run_root / "embedding_spectrum.json"
            spectrum = json.loads(spectrum_path.read_text())
            spectrum["eigenvalues"][0] *= 2
            spectrum_path.write_text(json.dumps(spectrum))
            with self.assertRaisesRegex(ValueError, "eigenvalues mismatch"):
                VALIDATE.validate_successful_run(
                    run_root, "legacy_any_panel", 2
                )

            write_successful_run(run_root)
            fit_path = run_root / "mclust_fit_contract.npz"
            with np.load(fit_path, allow_pickle=False) as archive:
                fit = {key: archive[key].copy() for key in archive.files}
            fit["return_code"] = np.array(-1)
            np.savez(fit_path, **fit)
            with self.assertRaisesRegex(RuntimeError, "return code"):
                VALIDATE.validate_successful_run(
                    run_root, "legacy_any_panel", 2
                )

    def test_only_audited_strict_eee_failures_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            embedding = np.zeros((14, 30), dtype=float)
            embedding[:, 0] = np.linspace(-1.0, 1.0, 14)
            np.save(run_root / "embedding.npy", embedding)
            spectrum = VALIDATE.recompute_embedding_spectrum(embedding)
            spectrum["eigenvalues"] = spectrum["eigenvalues"].tolist()
            (run_root / "embedding_spectrum.json").write_text(json.dumps(spectrum))
            status = "strict mclust EEE fit failed: singular covariance"
            failure = {
                "modality": "legacy_any_panel", "run": 2, "seed": 2,
                "mclust_fit_contract": "fixed_eee_numerical_v1",
                "strict_mclust_contract": True, "requested_model": "EEE",
                "fallback_used": False,
                "failure_stage": "mclust_eee_fit_or_validation",
                "error": status, "effective_rank": spectrum["effective_rank"],
                "embedding_dimensions": 30,
                "rank_relative_eigenvalue_threshold": 1e-6,
                "projected_condition_number": spectrum["projected_condition_number"],
            }
            failure_path = run_root / "failure_metadata.json"
            failure_path.write_text(json.dumps(failure))
            paths = VALIDATE.validate_failed_run(
                run_root, "legacy_any_panel", 2, status, 14
            )
            self.assertEqual(len(paths), 3)
            (run_root / "ari.txt").write_text("0.0\n")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                VALIDATE.validate_failed_run(
                    run_root, "legacy_any_panel", 2, status, 14
                )
            (run_root / "ari.txt").unlink()
            with self.assertRaisesRegex(ValueError, "unregistered run failure"):
                VALIDATE.validate_failed_run(
                    run_root, "legacy_any_panel", 2, "arbitrary error", 14
                )
            failure_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "failure metadata"):
                VALIDATE.validate_failed_run(
                    run_root, "legacy_any_panel", 2, status, 14
                )

    def test_success_cannot_carry_failure_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            write_successful_run(run_root)
            (run_root / "failure_metadata.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                VALIDATE.validate_successful_run(
                    run_root, "legacy_any_panel", 2
                )

    def test_section_only_smoke_selection_preserves_global_task_identity(self):
        self.assertEqual(
            VALIDATE.validate_requested_sections(["151508"]), ("151508",)
        )
        self.assertEqual(VALIDATE.SECTIONS.index("151508"), 1)
        with self.assertRaisesRegex(ValueError, "unique subset"):
            VALIDATE.validate_requested_sections(["151508", "151508"])

    def test_matrix_symlink_hash_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            source.write_bytes(b"matrix")
            link = root / "matrix.pkl"
            link.symlink_to(source)
            record = {
                "link_path": link.name,
                "absolute_path": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": hashlib.sha256(b"matrix").hexdigest(),
            }
            VERIFY.verify_record(record, root, link=True)
            source.write_bytes(b"drift!")
            with self.assertRaisesRegex(ValueError, "size drift|sha256 drift"):
                VERIFY.verify_record(record, root, link=True)

    def test_slurm_wrapper_uses_frozen_snapshot_and_both_methods(self):
        text = (TOOLS / "run_clustering.slurm").read_text()
        self.assertIn("RUN_ROOT/code_snapshot/SPARCAL_clustering.py", text)
        self.assertIn("RUN_ROOT/code_snapshot/benchmark_tools", text)
        self.assertIn(
            "--modalities legacy_any_panel,beagle_pnonref090_allalleles", text
        )
        self.assertIn("--n_runs 10", text)
        self.assertNotIn("clustering_benchmark/SPARCAL_clustering.py", text)

    def test_preparation_is_explicitly_lognorm_and_wes_free(self):
        text = (TOOLS / "prepare_strict_inputs.py").read_text()
        self.assertEqual(text.count('"preprocess": "lognorm"'), 2)
        self.assertIn('"wes_used_as_input": False', text)
        self.assertIn('"assess_strict_noninferiority.py"', text)
        self.assertIn('"validate_strict_run.py"', text)
        self.assertIn('"mclust_contract.py"', text)
        self.assertIn('"source_mclust_contract"', text)
        self.assertIn('"mclust_numerical_contract"', text)
        self.assertIn('"fixed_eee_numerical_v1"', text)
        validator = (TOOLS / "validate_strict_run.py").read_text()
        self.assertIn('"--sections"', validator)
        self.assertIn("smoke validation is read-only", validator)


if __name__ == "__main__":
    unittest.main()
