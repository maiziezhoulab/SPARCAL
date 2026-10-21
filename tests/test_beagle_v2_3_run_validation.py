import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "2_beagle_filtering"
    / "beagle_v2_3_candidate_2026_09_13"
    / "contracts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beagle_v2_3_run_validation", MODULE_PATH
)
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)


class BeagleV23RunValidationTests(unittest.TestCase):
    def make_run(self, root: Path, *, mismatch=False):
        quality = "baseQ0mapQ0"
        output = root / "output_VCFs" / "beagle" / quality
        output.mkdir(parents=True)
        raw = root / "raw.vcf.gz"
        raw.write_text("raw\n")
        raw_index = root / "raw.vcf.gz.tbi"
        raw_index.write_text("raw index\n")
        code_dir = root / "code_snapshot"
        code_dir.mkdir()
        code_paths = []
        for index in range(4):
            path = code_dir / f"source_{index}.py"
            path.write_text(f"# source {index}\n")
            code_paths.append(path)
        executable_dir = root / "synthetic_executables"
        executable_dir.mkdir()
        executables = {}
        for name in ("beagle_jar", "java", "bgzip", "tabix"):
            path = executable_dir / name
            path.write_text(f"{name}\n")
            executables[name] = CONTRACTS.file_identity(path)
        reference = root / "reference"
        reference.mkdir()
        for name in ("MANIFEST.txt", "REFERENCE_SHA256SUMS.txt", "OUTPUT_SHA256SUMS.txt"):
            (reference / name).write_text(f"{name}\n")
        panel = reference / "chr1.vcf.gz"
        panel.write_text("panel\n")
        panel_index = reference / "chr1.vcf.gz.tbi"
        panel_index.write_text("panel index\n")
        reference_sums = {
            str(panel.resolve()): CONTRACTS.sha256_file(panel),
            str(panel_index.resolve()): CONTRACTS.sha256_file(panel_index),
        }
        digest = "a" * 64
        marker = {
            "schema": CONTRACTS.MARKER_CONTRACT_SCHEMA,
            "contig": "chr1",
            "status": "completed",
            "production_inputs_include_wes": False,
            "exact_identity_pass": True,
            "tabix_probe": "chr1:10:A:G",
            "raw_exact_alleles": {"count": 1, "sha256": digest},
            "raw_panel_exact_alleles": {"count": 1, "sha256": digest},
            "raw_beagle_exact_alleles": {
                "count": 1,
                "sha256": "b" * 64 if mismatch else digest,
            },
        }
        status = {
            "schema": "sparcal_beagle_contig_status_v2_3",
            "production_inputs_include_wes": False,
            "raw_record_count": 1,
            "expected_contigs": ["chr1"],
            "contigs": [
                {
                    "chromosome": "chr1",
                    "status": "completed",
                    "marker_contract": marker,
                }
            ],
        }
        status_path = output / "BEAGLE_CONTIG_STATUS.json"
        status_path.write_text(json.dumps(status) + "\n")
        router = {
            "schema": "sparcal_beagle_evidence_v2_2",
            "production_inputs_include_wes": False,
            "raw_record_count": 1,
            "anchor_score": "p_nonref",
            "min_anchor_probability": 0.90,
            "hard_anchor_route": "beagle_nonref_highconf",
            "upstream_contig_status_sha256": CONTRACTS.sha256_file(
                status_path
            ),
            "compatibility": {
                "all_panel_present.vcf.gz": "three represented routes"
            },
            "counts": {
                "beagle_nonref_highconf": 1,
                "beagle_ref_highconf": 0,
                "beagle_uncertain": 0,
                "panel_absent": 0,
                "total": 1,
            },
        }
        router_path = output / "MANIFEST.json"
        router_path.write_text(json.dumps(router) + "\n")
        manifest = {
            "status": "outputs_ready_for_checksum",
            "min_anchor_probability": 0.90,
            "beagle_parameters": {
                "IMPUTE": False,
                "GPROBS": True,
                "ITERATIONS": 0,
                "SEED": 17,
                "MODEL_SCALE": 2,
                "MAX_LR": 5000,
                "REFERENCE_LEDGER": str(reference.resolve()),
            },
            "raw_vcf": CONTRACTS.file_identity(raw),
            "raw_vcf_index": CONTRACTS.file_identity(raw_index),
            "code": [CONTRACTS.file_identity(path) for path in code_paths],
            "executables": executables,
            "reference_panels": [
                {
                    "contig": "chr1",
                    "panel": CONTRACTS.file_identity(panel),
                    "panel_index": CONTRACTS.file_identity(panel_index),
                }
            ],
            "schema": CONTRACTS.RUN_CONTRACT_SCHEMA,
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "impute_effective": False,
            "anchor_score": "p_nonref",
            "quality_filter": quality,
            "reference_ledger": {
                "root": str(reference.resolve()),
                "manifest": CONTRACTS.file_identity(reference / "MANIFEST.txt"),
                "reference_sha256sums": CONTRACTS.file_identity(
                    reference / "REFERENCE_SHA256SUMS.txt"
                ),
                "output_sha256sums": CONTRACTS.file_identity(
                    reference / "OUTPUT_SHA256SUMS.txt"
                ),
            },
            "contig_status_file": str(status_path.relative_to(root)),
            "contig_status_file_sha256": CONTRACTS.sha256_file(
                status_path
            ),
            "completed_beagle_contigs": ["chr1"],
        }
        manifest_path = root / "V2_3_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest) + "\n")
        checksums = CONTRACTS.write_sha256sums(
            root, [status_path, router_path, manifest_path]
        )
        CONTRACTS.write_completion(
            root,
            manifest_path=manifest_path,
            checksum_path=checksums,
        )
        return reference_sums

    def test_valid_run_passes_semantic_and_content_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_sums = self.make_run(root)
            with mock.patch.object(
                CONTRACTS,
                "verify_reference_ledger",
                return_value=reference_sums,
            ):
                result = CONTRACTS.validate_v2_3_run(root)
            self.assertEqual(result["raw_record_count"], 1)
            self.assertEqual(result["completed_contig_count"], 1)

    def test_marker_digest_mismatch_fails_after_valid_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_sums = self.make_run(root, mismatch=True)
            with mock.patch.object(
                CONTRACTS,
                "verify_reference_ledger",
                return_value=reference_sums,
            ), self.assertRaisesRegex(
                ValueError, "marker identity digest/count mismatch"
            ):
                CONTRACTS.validate_v2_3_run(root)

    def test_snapshotted_source_mutation_fails_identity_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_sums = self.make_run(root)
            (root / "code_snapshot" / "source_0.py").write_text(
                "# mutated source with different size\n"
            )
            with mock.patch.object(
                CONTRACTS,
                "verify_reference_ledger",
                return_value=reference_sums,
            ), self.assertRaisesRegex(
                ValueError, "source file 0 (metadata|content) changed"
            ):
                CONTRACTS.validate_v2_3_run(root)

    def test_post_completion_mutation_fails_before_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            status_path = (
                root
                / "output_VCFs"
                / "beagle"
                / "baseQ0mapQ0"
                / "BEAGLE_CONTIG_STATUS.json"
            )
            with status_path.open("a") as handle:
                handle.write("mutated\n")
            with self.assertRaisesRegex(
                ValueError, "artifact checksum mismatch"
            ):
                CONTRACTS.validate_v2_3_run(root)


if __name__ == "__main__":
    unittest.main()
