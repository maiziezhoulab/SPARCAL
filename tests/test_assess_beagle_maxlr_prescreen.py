import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_maxlr_prescreen.py"
)
SPEC = importlib.util.spec_from_file_location(
    "assess_beagle_maxlr_prescreen", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def audit_row(
    position,
    *,
    p_nonref="0.5",
    route="beagle_uncertain",
    raw_gt="0/1",
):
    panel_absent = route == "panel_absent"
    return {
        "schema": "sparcal_beagle_evidence_v2_2",
        "chrom": "chr22",
        "pos": str(position),
        "ref": "A",
        "alt": "G",
        "raw_gt": raw_gt,
        "beagle_gt": "." if panel_absent else "0/1",
        "beagle_gp": "." if panel_absent else "0.5,0.4,0.1",
        "beagle_called_gp": "." if panel_absent else "0.5",
        "beagle_p_nonref": "." if panel_absent else str(p_nonref),
        "beagle_gp_entropy": "." if panel_absent else "0.85",
        "beagle_gp_margin": "." if panel_absent else "0.1",
        "beagle_af": "." if panel_absent else "0.2",
        "gt_transition": f"{raw_gt}>{'.' if panel_absent else '0/1'}",
        "route": route,
        "hard_anchor": "1" if route == "beagle_nonref_highconf" else "0",
    }


def write_audit(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MODULE.AUDIT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def file_identity(name, digest):
    return {
        "path": f"/different/run/{name}",
        "size_bytes": 17,
        "mtime_ns": 123,
        "sha256": digest,
    }


def manifest(maxlr, *, path_prefix="/comparator"):
    code = []
    for index in range(4):
        item = file_identity(
            f"source_{index}.py",
            f"{index + 1:064x}",
        )
        item["path"] = f"{path_prefix}/code/source_{index}.py"
        code.append(item)
    return {
        "schema": "sparcal_beagle_v2_3_run_v1",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "impute_effective": False,
        "anchor_score": "p_nonref",
        "min_anchor_probability": 0.90,
        "dataset": "P4_TUMOR",
        "quality_filter": "baseQ0mapQ0",
        "completed_beagle_contigs": ["chr22"],
        "raw_vcf": file_identity(
            f"{path_prefix}/raw.vcf.gz", "a" * 64
        ),
        "raw_vcf_index": file_identity(
            f"{path_prefix}/raw.vcf.gz.tbi", "b" * 64
        ),
        "code": code,
        "executables": {
            name: file_identity(name, digest * 64)
            for name, digest in (
                ("beagle_jar", "c"),
                ("java", "d"),
                ("bgzip", "e"),
                ("tabix", "f"),
            )
        },
        "reference_ledger": {
            "root": "/frozen/reference",
            "manifest": file_identity("reference_manifest", "1" * 64),
            "reference_sha256sums": file_identity("reference_sums", "2" * 64),
            "output_sha256sums": file_identity("output_sums", "3" * 64),
        },
        "reference_panels": [
            {
                "contig": "chr22",
                "panel": file_identity("panel", "4" * 64),
                "panel_index": file_identity("panel_index", "5" * 64),
                "raw_panel_exact_alleles": {
                    "count": 1,
                    "sha256": "6" * 64,
                },
            }
        ],
        "beagle_parameters": {
            "IMPUTE": False,
            "GPROBS": True,
            "ITERATIONS": 0,
            "SEED": 20260915,
            "MODEL_SCALE": 2.0,
            "MAX_LR": float(maxlr),
            "GENETIC_MAP": None,
            "THREADS": 30,
            "PARALLEL_CHROMOSOMES": 1,
            "MEMORY": "10g",
            "REFERENCE_LEDGER": "/frozen/reference",
        },
    }


def pair_manifests():
    comparator = manifest(5000, path_prefix="/comparator")
    candidate = manifest(50000, path_prefix="/candidate")
    for field in (
        "raw_vcf",
        "raw_vcf_index",
        "code",
        "executables",
    ):
        def change_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "path":
                        value[key] = str(value[key]).replace(
                            "/comparator", "/candidate"
                        )
                        value["mtime_ns"] = 456
                    else:
                        change_paths(item)
            elif isinstance(value, list):
                for item in value:
                    change_paths(item)

        change_paths(candidate[field])
    return comparator, candidate


class MaxlrPrescreenTests(unittest.TestCase):
    def test_identical_routes_and_subthreshold_delta_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator = root / "comparator.tsv.gz"
            candidate = root / "candidate.tsv.gz"
            write_audit(
                comparator,
                [
                    audit_row(10, p_nonref="0.5000"),
                    audit_row(20, route="panel_absent"),
                ],
            )
            write_audit(
                candidate,
                [
                    audit_row(10, p_nonref="0.5009"),
                    audit_row(20, route="panel_absent"),
                ],
            )
            result = MODULE.compare_audits(
                comparator, candidate, expected_rows=2
            )
            self.assertEqual(result["route_disagreement_count"], 0)
            self.assertEqual(result["anchor_disagreement_count"], 0)
            self.assertEqual(result["p_nonref_compared_n"], 1)
            self.assertAlmostEqual(
                result["p_nonref_max_absolute_delta"], 0.0009
            )
            self.assertFalse(result["advance_to_full_genome"])
            self.assertEqual(
                result["decision"], "stop_after_chr22_no_full_genome"
            )

    def test_delta_at_boundary_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator = root / "comparator.tsv.gz"
            candidate = root / "candidate.tsv.gz"
            write_audit(comparator, [audit_row(10, p_nonref="0.100")])
            write_audit(candidate, [audit_row(10, p_nonref="0.101")])
            result = MODULE.compare_audits(comparator, candidate)
            self.assertGreaterEqual(
                result["p_nonref_max_absolute_delta"], 0.001
            )
            self.assertTrue(result["advance_to_full_genome"])

    def test_route_or_anchor_change_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator = root / "comparator.tsv.gz"
            candidate = root / "candidate.tsv.gz"
            write_audit(
                comparator,
                [
                    audit_row(
                        10,
                        p_nonref="0.95",
                        route="beagle_nonref_highconf",
                    )
                ],
            )
            write_audit(
                candidate,
                [
                    audit_row(
                        10,
                        p_nonref="0.95",
                        route="beagle_uncertain",
                    )
                ],
            )
            result = MODULE.compare_audits(comparator, candidate)
            self.assertEqual(result["route_disagreement_count"], 1)
            self.assertEqual(result["anchor_disagreement_count"], 1)
            self.assertTrue(result["advance_to_full_genome"])

    def test_allele_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator = root / "comparator.tsv.gz"
            candidate = root / "candidate.tsv.gz"
            write_audit(comparator, [audit_row(10), audit_row(20)])
            write_audit(candidate, [audit_row(20), audit_row(10)])
            with self.assertRaisesRegex(ValueError, "universe/order"):
                MODULE.compare_audits(comparator, candidate)

    def test_manifest_pair_allows_only_maxlr_and_run_local_paths(self):
        comparator, candidate = pair_manifests()
        MODULE.validate_manifest_pair(comparator, candidate)
        candidate["beagle_parameters"]["THREADS"] = 29
        with self.assertRaisesRegex(ValueError, "more than MAX_LR"):
            MODULE.validate_manifest_pair(comparator, candidate)

    def test_input_content_change_is_rejected(self):
        comparator, candidate = pair_manifests()
        candidate["raw_vcf"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw_vcf"):
            MODULE.validate_manifest_pair(comparator, candidate)

    def test_run_local_bcftools_header_is_the_only_semantic_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator = root / "comparator.vcf.gz"
            candidate = root / "candidate.vcf.gz"
            stable = (
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "chr22\t10\t.\tA\tG\t.\tPASS\t.\n"
            )
            with gzip.open(comparator, "wt") as handle:
                handle.write(
                    "##bcftools_viewCommand=view -r chr22 -Oz -o "
                    "old.vcf.gz; Date=one\n"
                )
                handle.write(stable)
            with gzip.open(candidate, "wt") as handle:
                handle.write(
                    "##bcftools_viewCommand=view -r chr22 -Oz -o "
                    "new.vcf.gz; Date=two\n"
                )
                handle.write(stable)
            self.assertEqual(
                MODULE.vcf_semantic_identity(comparator),
                MODULE.vcf_semantic_identity(candidate),
            )
            with gzip.open(candidate, "wt") as handle:
                handle.write(
                    "##bcftools_viewCommand=view -r chr22 -Oz -o "
                    "new.vcf.gz; Date=two\n"
                )
                handle.write(stable.replace("chr22\t10", "chr22\t11"))
            self.assertNotEqual(
                MODULE.vcf_semantic_identity(comparator),
                MODULE.vcf_semantic_identity(candidate),
            )

    def test_run_pair_calls_contracts_for_both_and_finds_canonical_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator_root = root / "comparator"
            candidate_root = root / "candidate"
            comparator_manifest, candidate_manifest = pair_manifests()
            for run_root, payload in (
                (comparator_root, comparator_manifest),
                (candidate_root, candidate_manifest),
            ):
                run_root.mkdir()
                raw_vcf = run_root / "raw.vcf.gz"
                with gzip.open(raw_vcf, "wt") as handle:
                    handle.write(
                        "##fileformat=VCFv4.2\n"
                        "##bcftools_viewCommand=view -r chr22 -Oz -o "
                        "test.vcf.gz; Date=test\n"
                        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                        "chr22\t10\t.\tA\tG\t.\tPASS\t.\n"
                    )
                raw_index = run_root / "raw.vcf.gz.tbi"
                raw_index.write_bytes(b"test-index\n")
                payload["raw_vcf"] = MODULE.identity(raw_vcf)
                payload["raw_vcf_index"] = MODULE.identity(raw_index)
                (run_root / "V2_3_MANIFEST.json").write_text(
                    json.dumps(payload) + "\n"
                )
                write_audit(
                    MODULE.canonical_audit(run_root, payload),
                    [audit_row(10)],
                )
            validator = mock.Mock(
                side_effect=[
                    {"status": "verified", "raw_record_count": 1},
                    {"status": "verified", "raw_record_count": 1},
                ]
            )
            contracts = SimpleNamespace(validate_v2_3_run=validator)
            result = MODULE.validate_run_pair(
                comparator_root, candidate_root, contracts
            )
            self.assertEqual(validator.call_count, 2)
            self.assertEqual(result[2], MODULE.canonical_audit(
                comparator_root, comparator_manifest
            ).resolve())
            self.assertEqual(result[3], MODULE.canonical_audit(
                candidate_root, candidate_manifest
            ).resolve())

    def test_bundle_is_bound_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparator_root = root / "comparator"
            candidate_root = root / "candidate"
            comparator_manifest, candidate_manifest = pair_manifests()
            for run_root, payload in (
                (comparator_root, comparator_manifest),
                (candidate_root, candidate_manifest),
            ):
                run_root.mkdir()
                raw_vcf = run_root / "raw.vcf.gz"
                with gzip.open(raw_vcf, "wt") as handle:
                    handle.write(
                        "##fileformat=VCFv4.2\n"
                        "##bcftools_viewCommand=view -r chr22 -Oz -o "
                        "test.vcf.gz; Date=test\n"
                        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                        "chr22\t10\t.\tA\tG\t.\tPASS\t.\n"
                    )
                raw_index = run_root / "raw.vcf.gz.tbi"
                raw_index.write_bytes(b"test-index\n")
                payload["raw_vcf"] = MODULE.identity(raw_vcf)
                payload["raw_vcf_index"] = MODULE.identity(raw_index)
                (run_root / "V2_3_MANIFEST.json").write_text(
                    json.dumps(payload) + "\n"
                )
                (run_root / "RUN_COMPLETE.json").write_text(
                    '{"status":"complete"}\n'
                )
                write_audit(
                    MODULE.canonical_audit(run_root, payload),
                    [audit_row(10)],
                )
            contracts_path = root / "contracts.py"
            protocol_path = root / "protocol.md"
            contracts_path.write_text("# frozen\n")
            protocol_path.write_text("# frozen\n")
            comparator_audit = MODULE.canonical_audit(
                comparator_root, comparator_manifest
            )
            candidate_audit = MODULE.canonical_audit(
                candidate_root, candidate_manifest
            )
            result = MODULE.compare_audits(
                comparator_audit, candidate_audit
            )
            output = root / "assessment"
            keyword_args = {
                "result": result,
                "comparator_root": comparator_root,
                "candidate_root": candidate_root,
                "comparator_manifest": comparator_manifest,
                "candidate_manifest": candidate_manifest,
                "comparator_audit": comparator_audit,
                "candidate_audit": candidate_audit,
                "comparator_validation": {
                    "status": "verified",
                    "raw_record_count": 1,
                },
                "candidate_validation": {
                    "status": "verified",
                    "raw_record_count": 1,
                },
                "contracts_path": contracts_path,
                "protocol_path": protocol_path,
            }
            MODULE.write_bundle(output, **keyword_args)
            for name in (
                "maxlr_prescreen_summary.tsv",
                "PROVENANCE.json",
                "CODE_AND_INPUT_SHA256SUMS.txt",
                "OUTPUT_SHA256SUMS.txt",
                "RUN_COMPLETE",
                "RUN_COMPLETE.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            completion = json.loads(
                (output / "RUN_COMPLETE.json").read_text()
            )
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(
                completion["output_sha256sums"]["sha256"],
                MODULE.sha256(output / "OUTPUT_SHA256SUMS.txt"),
            )
            provenance = json.loads((output / "PROVENANCE.json").read_text())
            self.assertFalse(provenance["production_inputs_include_wes"])
            self.assertFalse(provenance["evaluation_uses_wes"])
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                MODULE.write_bundle(output, **keyword_args)


if __name__ == "__main__":
    unittest.main()
