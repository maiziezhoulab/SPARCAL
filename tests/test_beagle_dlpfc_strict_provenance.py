import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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


PROVENANCE = load_module("strict_dlpfc_provenance", "strict_provenance.py")
VERIFY = load_module("strict_dlpfc_verify_external", "verify_strict_inputs.py")
VALIDATE = load_module("strict_dlpfc_validate_more", "validate_strict_run.py")
FINALIZE = load_module("strict_dlpfc_finalize", "finalize_strict_assessment.py")


class StrictDlpfcProvenanceTests(unittest.TestCase):
    def test_candidate_semantic_provenance_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            section = "151507"
            candidate_root = root / "candidate"
            section_root = candidate_root / section
            matrix_dir = section_root / "matrix"
            code_dir = section_root / "code_snapshot"
            matrix_dir.mkdir(parents=True)
            code_dir.mkdir()
            baseline = root / "baseline.pkl"
            candidate = matrix_dir / "candidate.pkl"
            route_audit = root / "audit.tsv.gz"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")
            route_audit.write_bytes(b"audit")
            baseline_sha = hashlib.sha256(b"baseline").hexdigest()
            candidate_sha = hashlib.sha256(b"candidate").hexdigest()
            sidecar = {
                "selection": {"minimum": 0.9, "score": "beagle_p_nonref"},
                "multiallelic_policy": "all",
                "operation": "column subset only",
                "output_matrix": str(candidate),
                "output_matrix_sha256": candidate_sha,
                "source_matrix": str(baseline),
                "source_matrix_sha256": baseline_sha,
                "route_audit": str(route_audit),
                "route_audit_sha256": hashlib.sha256(b"audit").hexdigest(),
            }
            sidecar_path = Path(f"{candidate}.manifest.json")
            sidecar_path.write_text(json.dumps(sidecar))
            (section_root / "PREP_MANIFEST.txt").write_text(
                "wes_input=NONE\nselection=beagle_p_nonref>=0.90\n"
                "multiallelic_projection=all_exact_alleles_at_position_must_pass\n"
                "matrix_operation=column_subset_only\n"
            )
            (section_root / "PREP_COMPLETE").write_text("complete\n")
            (code_dir / "build_beagle_route_matrix.py").write_text("# frozen\n")
            records = PROVENANCE.candidate_provenance_records(
                section, candidate_root, baseline, candidate,
                baseline_sha, candidate_sha, root,
            )
            self.assertEqual(len(records), 5)
            strict_root = root / "strict_root"
            strict_root.mkdir()
            VERIFY.verify_record(records[0], strict_root, external=True)
            VALIDATE.verify_record(records[0], strict_root, external=True)
            sidecar["selection"]["minimum"] = 0.8
            sidecar_path.write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "selection mismatch"):
                PROVENANCE.candidate_provenance_records(
                    section, candidate_root, baseline, candidate,
                    baseline_sha, candidate_sha, root,
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
                "beagle_pnonref090_allalleles": [],
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
