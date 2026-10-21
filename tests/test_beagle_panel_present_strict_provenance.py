import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = (
    ROOT / "clustering_benchmark"
    / "beagle_panel_present_strict_regression_2026_09_13"
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROVENANCE = load_module("panel_present_strict_provenance", "strict_provenance.py")
VERIFY = load_module("panel_present_strict_verify_external", "verify_strict_inputs.py")
VALIDATE = load_module("panel_present_strict_validate_more", "validate_strict_run.py")
FINALIZE = load_module("panel_present_strict_finalize", "finalize_strict_assessment.py")


class PanelPresentStrictProvenanceTests(unittest.TestCase):
    def test_candidate_semantic_provenance_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            section = "151507"
            candidate_root = root / "candidate"
            section_root = candidate_root / section
            matrix_dir = section_root / "matrix"
            snapshot_dir = candidate_root / "code_snapshot"
            matrix_dir.mkdir(parents=True)
            snapshot_dir.mkdir()

            baseline = root / "baseline.pkl"
            candidate = matrix_dir / "candidate.pkl"
            route_audit = root / "audit.tsv.gz"
            frame = pd.DataFrame(
                [[1, 0], [0, 1]],
                index=pd.Index(["spot_b", "spot_a"], name="barcode"),
                columns=pd.Index(["1_10", "2_20"], name="locus"),
                dtype="int8",
            )
            frame.to_pickle(baseline)
            frame.copy().to_pickle(candidate)
            route_audit.write_bytes(b"audit")
            baseline_sha = PROVENANCE.sha256(baseline)
            candidate_sha = PROVENANCE.sha256(candidate)
            route_sha = PROVENANCE.sha256(route_audit)
            dataframe_sha = "d" * 64
            routes = sorted(PROVENANCE.ROUTES)

            sidecar = {
                "selection": {"routes": routes},
                "multiallelic_policy": "any",
                "operation": "column subset only",
                "output_matrix": str(candidate.resolve()),
                "output_matrix_sha256": candidate_sha,
                "source_matrix": str(baseline.resolve()),
                "source_matrix_sha256": baseline_sha,
                "route_audit": str(route_audit.resolve()),
                "route_audit_sha256": route_sha,
                "source_shape": [2, 2],
                "output_shape": [2, 2],
                "source_nonzero": 2,
                "output_nonzero": 2,
                "selected_loci_observed_in_source_matrix": 2,
            }
            sidecar_path = Path(f"{candidate}.manifest.json")
            sidecar_path.write_text(json.dumps(sidecar))
            equality = {
                "assertion":
                    "pandas.testing.assert_frame_equal(check_exact=True)",
                "all_checks_passed": True,
                "index_equal": True,
                "columns_equal": True,
                "dtypes_equal": True,
                "shape_equal": True,
                "values_equal": True,
                "source_shape": [2, 2],
                "output_shape": [2, 2],
                "source_nonzero": 2,
                "output_nonzero": 2,
                "source_dataframe_sha256": dataframe_sha,
                "output_dataframe_sha256": dataframe_sha,
            }
            prep_manifest = {
                "schema_version": PROVENANCE.PREP_SCHEMA,
                "section": section,
                "scientific_role": PROVENANCE.COMPATIBILITY_ROLE,
                "not_a_claim_of": [
                    "germline status", "variant-caller accuracy",
                    "non-reference genotype status",
                ],
                "wes_input": False,
                "source_matrix": str(baseline.resolve()),
                "source_matrix_sha256": baseline_sha,
                "route_audit": str(route_audit.resolve()),
                "route_audit_sha256": route_sha,
                "selected_routes": routes,
                "position_projection":
                    "retain if any exact allele at the position is selected",
                "multiallelic_policy": "any",
                "output_matrix": str(candidate.resolve()),
                "output_matrix_sha256": candidate_sha,
                "independent_exact_dataframe_equality": equality,
            }
            prep_path = section_root / "PREP_MANIFEST.json"
            prep_path.write_text(json.dumps(prep_manifest))
            (section_root / "PREP_COMPLETE").write_text("complete\n")
            builder = snapshot_dir / "build_beagle_route_matrix.py"
            prep_builder = snapshot_dir / "build_panel_present_compatibility.py"
            builder.write_text("# frozen builder\n")
            prep_builder.write_text("# frozen prep\n")
            root_manifest = {
                "schema_version": PROVENANCE.PREP_SCHEMA,
                "output_root": str(candidate_root.resolve()),
                "sections": list(PROVENANCE.SECTIONS),
                "scientific_role": PROVENANCE.COMPATIBILITY_ROLE,
                "interpretation": (
                    "This is not a germline call set and must not be used "
                    "as evidence of caller accuracy."
                ),
                "wes_input": False,
                "selected_routes": routes,
                "multiallelic_policy": "any",
                "all_sections_exactly_equal_to_legacy": True,
                "code_snapshot": {
                    str(builder.relative_to(candidate_root)):
                        PROVENANCE.sha256(builder),
                    str(prep_builder.relative_to(candidate_root)):
                        PROVENANCE.sha256(prep_builder),
                },
                "section_records": {
                    section: {
                        "matrix": str(candidate.relative_to(candidate_root)),
                        "matrix_sha256": candidate_sha,
                        "shape": [2, 2],
                        "nonzero": 2,
                        "dataframe_sha256": dataframe_sha,
                        "exactly_equal_to_legacy": True,
                        "prep_manifest":
                            str(prep_path.relative_to(candidate_root)),
                        "prep_manifest_sha256": PROVENANCE.sha256(prep_path),
                    },
                },
            }
            root_manifest_path = candidate_root / "MANIFEST.json"
            root_manifest_path.write_text(json.dumps(root_manifest))
            checksummed = (
                root_manifest_path, prep_path, sidecar_path, candidate,
                builder, prep_builder,
            )
            (candidate_root / "CHECKSUMS.sha256").write_text("".join(
                f"{PROVENANCE.sha256(path)}  "
                f"{path.relative_to(candidate_root)}\n"
                for path in checksummed
            ))
            (candidate_root / "PREP_COMPLETE").write_text("complete\n")

            records = PROVENANCE.candidate_provenance_records(
                section, candidate_root, baseline, candidate,
                baseline_sha, candidate_sha, root,
            )
            self.assertEqual(len(records), 9)
            strict_root = root / "strict_root"
            strict_root.mkdir()
            VERIFY.verify_record(records[0], strict_root, external=True)
            VALIDATE.verify_record(records[0], strict_root, external=True)

            sidecar["selection"] = {"routes": ["beagle_uncertain"]}
            sidecar_path.write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "sidecar selection"):
                PROVENANCE.candidate_provenance_records(
                    section, candidate_root, baseline, candidate,
                    baseline_sha, candidate_sha, root,
                )

    def test_exact_dataframe_recheck_rejects_value_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.pkl"
            candidate = root / "candidate.pkl"
            pd.DataFrame([[1]], dtype="int8").to_pickle(baseline)
            pd.DataFrame([[0]], dtype="int8").to_pickle(candidate)
            with self.assertRaisesRegex(ValueError, "not exactly equal"):
                PROVENANCE.require_exact_dataframe_equality(
                    baseline, candidate, "151507"
                )

    def test_mclust_contract_manifest_is_pinned_in_both_verifiers(self):
        digest = "a" * 64
        manifest = {
            "source_mclust_contract": {"sha256": digest},
            "code_snapshot": [{
                "path": "code_snapshot/mclust_contract.py", "sha256": digest,
            }],
            "mclust_numerical_contract": {
                "name": "fixed_eee_numerical_v1",
                "model": "EEE",
                "fallback_allowed": False,
                "relative_eigenvalue_threshold": 1e-6,
                "effective_rank_definition": (
                    "count thresholded centered eigenvalues; no rank>=G rule"
                ),
                "success_artifacts": [
                    "embedding.npy", "embedding_spectrum.json",
                    "mclust_fit_contract.npz", "run_metadata.json",
                    "cluster_labels.csv", "ari.txt",
                ],
                "allowed_failure_prefixes": [
                    "strict mclust EEE fit failed:",
                    "strict mclust EEE validation failed:",
                ],
                "required_failure_artifacts": [
                    "embedding.npy", "embedding_spectrum.json",
                    "failure_metadata.json",
                ],
            },
        }
        VERIFY.validate_manifest_numerical_contract(manifest)
        VALIDATE.validate_manifest_numerical_contract(manifest)
        manifest["mclust_numerical_contract"]["fallback_allowed"] = True
        with self.assertRaisesRegex(ValueError, "fallback_allowed"):
            VERIFY.validate_manifest_numerical_contract(manifest)
        with self.assertRaisesRegex(ValueError, "fallback_allowed"):
            VALIDATE.validate_manifest_numerical_contract(manifest)

    def test_manifest_semantics_and_provenance_grid_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            candidate_root = (
                project / "data" / "experiments"
                / "beagle_dlpfc_panel_present_compatibility_20260913_v1"
            )
            candidate_root.mkdir(parents=True)
            contract = {
                "method": "beagle_panel_present_compat",
                "candidate_root": str(candidate_root),
                "scientific_role":
                    "Beagle-present feature-support compatibility only",
                "selected_routes": [
                    "beagle_nonref_highconf", "beagle_ref_highconf",
                    "beagle_uncertain",
                ],
                "multiallelic_policy": "any",
                "exact_dataframe_equal_to_baseline": True,
                "wes_used_as_input": False,
                "interpretation_scope": (
                    "backward-compatibility regression only; not a germline "
                    "call set and not evidence of SNV-caller accuracy"
                ),
            }
            provenance = [
                {"section": section, "kind": kind}
                for section in VALIDATE.SECTIONS
                for kind in VALIDATE.CANDIDATE_PROVENANCE_KINDS
            ]
            manifest = {
                "project_root": str(project),
                "candidate_semantic_contract": contract,
                "candidate_provenance": provenance,
            }
            for module in (VERIFY, VALIDATE):
                module.validate_candidate_semantic_contract(manifest)
                module.validate_candidate_provenance_grid(manifest)
            contract["multiallelic_policy"] = "all"
            for module in (VERIFY, VALIDATE):
                with self.assertRaisesRegex(ValueError, "multiallelic_policy"):
                    module.validate_candidate_semantic_contract(manifest)
            contract["multiallelic_policy"] = "any"
            provenance.pop()
            for module in (VERIFY, VALIDATE):
                with self.assertRaisesRegex(ValueError, "section/kind grid"):
                    module.validate_candidate_provenance_grid(manifest)

    def test_runtime_identity_and_candidate_failure_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            section = "151507"
            section_root = root / section
            section_root.mkdir()
            common = {
                "schema_version": "1.0",
                "section": section,
                "run_root": str(root),
                "hostname": "gpu-node",
                "platform": "linux",
                "python_executable": "/env/python",
                "python_version": "3.10",
                "packages": {"torch": "2.2.2"},
                "torch_runtime": {
                    "cuda_available": True,
                    "cuda_version": "12.1",
                    "cudnn_version": 8902,
                    "device_name": "NVIDIA RTX A6000",
                },
                "r_runtime": {"r_version": "4.5.1", "mclust_version": "6.1.2"},
                "nvidia_smi": ["NVIDIA RTX A6000, GPU-id, driver"],
                "slurm": {"SLURM_JOB_ID": "1", "SLURM_ARRAY_TASK_ID": "0"},
            }
            for phase, time in (
                ("start", "2026-09-13T00:00:00+00:00"),
                ("end", "2026-09-13T00:01:00+00:00"),
            ):
                record = dict(common, phase=phase, time_utc=time)
                (section_root / f"RUNTIME_{phase.upper()}.json").write_text(
                    json.dumps(record)
                )
            VALIDATE.validate_runtime(root, section)
            masks = {
                "legacy_any_panel": ["151507/run0"],
                "beagle_panel_present_compat": [],
            }
            self.assertEqual(
                VALIDATE.find_candidate_only_failures(masks), ["151507/run0"]
            )

    def _write_finalizer_inputs(
        self, assessment_root: Path, run_root: Path, *,
        accepted: bool = True, result_manifest: str | None = None,
        validation_manifest: str | None = None,
        validated_sections: list[str] | None = None,
        result_root: Path | None = None,
    ) -> None:
        manifest = run_root / "STRICT_INPUT_MANIFEST.json"
        if not manifest.exists():
            manifest.write_bytes(b"pinned strict manifest\n")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result_digest = digest if result_manifest is None else result_manifest
        validation_digest = digest if validation_manifest is None else validation_manifest
        bound_root = run_root if result_root is None else result_root
        gates = {gate: True for gate in FINALIZE.EXPECTED_GATES}
        (assessment_root / "NONINFERIORITY_RESULT.json").write_text(json.dumps({
            "schema_version": "2.0",
            "protocol_version": FINALIZE.PROTOCOL_VERSION,
            "baseline_method": FINALIZE.BASELINE_METHOD,
            "candidate_method": FINALIZE.CANDIDATE_METHOD,
            "interpretation_scope": FINALIZE.INTERPRETATION_SCOPE,
            "baseline_root": str(bound_root),
            "candidate_root": str(bound_root),
            "strict_input_manifest_sha256": result_digest,
            "gates": gates,
            "accept_replacement": accepted,
        }))
        (assessment_root / "STRICT_RUN_VALIDATION.json").write_text(json.dumps({
            "status": "all_checks_passed",
            "root": str(run_root),
            "strict_input_manifest_sha256": validation_digest,
            "validated_sections": (
                list(FINALIZE.SECTIONS)
                if validated_sections is None else validated_sections
            ),
        }))

    def _run_finalizer(self, root: Path) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["finalize", "--assessment-root", str(root)]
            FINALIZE.main()
        finally:
            sys.argv = old_argv

    def test_finalizer_writes_decision_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "strict_run"
            run_root.mkdir()
            self._write_finalizer_inputs(root, run_root)
            self._run_finalizer(root)
            self.assertTrue((root / "ACCEPTED").is_file())
            self.assertTrue((root / "ASSESSMENT_COMPLETE").is_file())
            self.assertTrue((root / "OUTPUT_SHA256SUMS").is_file())

    def test_finalizer_rejects_forged_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "strict_run"
            run_root.mkdir()
            self._write_finalizer_inputs(root, run_root, accepted=False)
            with self.assertRaisesRegex(ValueError, "disagrees with registered gates"):
                self._run_finalizer(root)

    def test_finalizer_requires_exact_boolean_gate_set(self):
        for mutation, expected_message in (
            ("missing", "exact registered gate set"),
            ("non_boolean", "must be booleans"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    run_root = root / "strict_run"
                    run_root.mkdir()
                    self._write_finalizer_inputs(root, run_root)
                    result_path = root / "NONINFERIORITY_RESULT.json"
                    result = json.loads(result_path.read_text())
                    gate = next(iter(FINALIZE.EXPECTED_GATES))
                    if mutation == "missing":
                        result["gates"].pop(gate)
                    else:
                        result["gates"][gate] = 1
                    result_path.write_text(json.dumps(result))
                    with self.assertRaisesRegex((ValueError, TypeError), expected_message):
                        self._run_finalizer(root)

    def test_finalizer_binds_methods_and_interpretation_scope(self):
        mutations = (
            ("baseline_method", "other", "baseline method"),
            ("candidate_method", "other", "candidate method"),
            ("interpretation_scope", "caller accuracy", "interpretation scope"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    run_root = root / "strict_run"
                    run_root.mkdir()
                    self._write_finalizer_inputs(root, run_root)
                    result_path = root / "NONINFERIORITY_RESULT.json"
                    result = json.loads(result_path.read_text())
                    result[key] = value
                    result_path.write_text(json.dumps(result))
                    with self.assertRaisesRegex(ValueError, message):
                        self._run_finalizer(root)

    def test_finalizer_rejects_root_manifest_or_section_mismatch(self):
        cases = (
            ("run root", {"result_root_name": "other_run"}),
            ("manifest hash", {"validation_manifest": "b" * 64}),
            (
                "manifest hash",
                {"result_manifest": "b" * 64, "validation_manifest": "b" * 64},
            ),
            ("all 12", {"validated_sections": list(FINALIZE.SECTIONS[:-1])}),
        )
        for expected_message, changes in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    run_root = root / "strict_run"
                    run_root.mkdir()
                    result_root_name = changes.pop("result_root_name", None)
                    if result_root_name is not None:
                        result_root = root / result_root_name
                        result_root.mkdir()
                        changes["result_root"] = result_root
                    self._write_finalizer_inputs(root, run_root, **changes)
                    with self.assertRaisesRegex(ValueError, expected_message):
                        self._run_finalizer(root)

    def test_wrappers_require_explicit_output_roots(self):
        cluster = (TOOLS / "run_clustering.slurm").read_text()
        assess = (TOOLS / "run_assessment.slurm").read_text()
        self.assertIn("RUN_ROOT=${RUN_ROOT:?", cluster)
        self.assertIn("RUN_ROOT=${RUN_ROOT:?", assess)
        self.assertIn("ASSESSMENT_ROOT=${ASSESSMENT_ROOT:?", assess)


if __name__ == "__main__":
    unittest.main()
