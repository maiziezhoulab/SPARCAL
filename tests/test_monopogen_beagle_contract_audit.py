import csv
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "postanalyze"
    / "monopogen_beagle_contract_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "monopogen_beagle_contract_audit", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_vcf(path: Path, format_name: str, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        )
        for chrom, pos, ref, alt, sample in rows:
            handle.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\t"
                f"{format_name}\t{sample}\n"
            )


class MonopogenBeagleContractAuditTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> Path:
        run_dir = root / "synthetic_run"
        germline = run_dir / "out" / "germline"
        write_vcf(
            germline / "chr1.gl.vcf.gz",
            "GT:PL",
            [
                ("chr1", 100, "C", "T", "./.:0,20,40"),
                ("chr1", 200, "A", "G", "./.:20,0,30"),
                ("chr1", 300, "G", "A", "./.:40,20,0"),
                ("chr1", 400, "T", "C", "./.:40,0,20"),
                ("chr1", 500, "A", "C", "./.:0,10,20"),
                ("chr1", 600, "G", "T", "./.:20,0,0"),
            ],
        )
        write_vcf(
            germline / "chr1.gp.vcf.gz",
            "GT:GP",
            [
                ("chr1", 100, "C", "T", "0/0:0.99,0.01,0"),
                ("chr1", 200, "A", "G", "0/1:0.05,0.90,0.05"),
                ("chr1", 300, "G", "A", "0/1:0.05,0.92,0.03"),
                ("chr1", 400, "T", "C", "1/1:0.05,0.05,0.90"),
                ("chr1", 600, "G", "T", "0/1:0.05,0.90,0.05"),
            ],
        )
        write_vcf(
            germline / "chr1.phased.vcf.gz",
            "GT",
            [
                ("chr1", 100, "C", "T", "0|0"),
                ("chr1", 200, "A", "G", "1|0"),
                ("chr1", 300, "G", "A", "0|1"),
                ("chr1", 400, "T", "C", "1|1"),
                ("chr1", 600, "G", "T", "1|0"),
            ],
        )
        return run_dir

    def test_streaming_contract_metrics_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self.build_fixture(root)
            output_dir = root / "audit"
            summaries = MODULE.run_audit(
                [("fixture", run_dir)], (1,), 0.90, output_dir
            )
            summary = summaries[0]

            self.assertEqual(summary["raw_records"], 6)
            self.assertEqual(summary["gp_records"], 5)
            self.assertEqual(summary["phased_records"], 5)
            self.assertFalse(summary["raw_gp_record_sets_equal"])
            self.assertTrue(summary["gp_phased_record_sets_equal"])
            self.assertFalse(summary["all_three_record_sets_equal"])
            self.assertEqual(summary["raw_keys_absent_from_gp"], 1)
            self.assertEqual(summary["gp_keys_absent_from_raw"], 0)
            self.assertEqual(summary["raw_gp_comparable"], 5)
            self.assertEqual(summary["raw_gp_pl_ties"], 1)
            self.assertEqual(summary["raw_gp_concordant"], 3)
            self.assertEqual(summary["raw_gp_class_concordance"], 0.6)
            self.assertEqual(summary["monopogen_downstream_beagle_records"], 5)
            self.assertEqual(
                summary["monopogen_paper_raw_beagle_gt_concordant_nonref"], 2
            )
            self.assertEqual(
                summary["diagnostic_paper_unique_raw_pl_concordant_nonref"], 1
            )
            self.assertEqual(summary["sparcal_derived_pnonref_threshold_count"], 4)
            self.assertEqual(summary["diagnostic_final_phased_nonref_gt"], 4)
            self.assertNotIn("released_code_nonref", summary)
            self.assertEqual(summary["raw_1_1_to_gp_0_1"], 1)
            self.assertEqual(summary["gp_phased_unordered_gt_changes"], 0)
            self.assertEqual(summary["phased_0_pipe_1"], 1)
            self.assertEqual(summary["phased_1_pipe_0"], 2)

            with (output_dir / "method_intersections.tsv").open() as handle:
                intersections = list(csv.DictReader(handle, delimiter="\t"))
            downstream_paper = next(
                row
                for row in intersections
                if row["left_method"] == MODULE.DOWNSTREAM_RECORD_METHOD
                and row["right_method"] == MODULE.PAPER_CONCORDANCE_METHOD
            )
            self.assertEqual(downstream_paper["intersection_count"], "2")
            self.assertEqual(
                float(downstream_paper["fraction_of_left_in_intersection"]), 0.4
            )
            self.assertEqual(
                float(downstream_paper["fraction_of_right_in_intersection"]), 1.0
            )
            downstream_pnonref = next(
                row
                for row in intersections
                if row["left_method"] == MODULE.DOWNSTREAM_RECORD_METHOD
                and row["right_method"] == MODULE.PNONREF_METHOD
            )
            self.assertEqual(downstream_pnonref["intersection_count"], "4")
            self.assertEqual(
                float(downstream_pnonref["fraction_of_left_in_intersection"]), 0.8
            )
            self.assertEqual(
                float(downstream_pnonref["fraction_of_right_in_intersection"]),
                1.0,
            )

            with (output_dir / "transition_counts.tsv").open() as handle:
                transitions = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(transitions), 9)
            one_one_to_het = next(
                row
                for row in transitions
                if row["raw_pl_gt"] == "1/1" and row["beagle_gp_gt"] == "0/1"
            )
            self.assertEqual(one_one_to_het["count"], "1")

            with (output_dir / "input_manifest.tsv").open() as handle:
                inputs = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(inputs), 3)
            for item in inputs:
                self.assertEqual(item["sha256"], MODULE.sha256(Path(item["path"])))

            provenance = json.loads((output_dir / "PROVENANCE.json").read_text())
            self.assertFalse(provenance["wes_used"])
            self.assertEqual(provenance["audit_schema_version"], "3.0")
            self.assertEqual(
                provenance["monopogen_public_commit"],
                MODULE.PINNED_MONOPOGEN_COMMIT,
            )
            self.assertEqual(
                provenance["public_source_contract"]
                ["downstream_exact_phased_record_lookup_and_read_support"],
                MODULE.MONOPOGEN_DOWNSTREAM_SOURCE,
            )
            self.assertEqual(provenance["runs"][0]["path"], str(run_dir.resolve()))
            self.assertEqual(
                provenance["software_versions"]["external_python_packages_used"],
                {},
            )
            self.assertIn("raw read support", provenance["scope_limit"])
            self.assertNotIn("released_code_nonref", json.dumps(provenance))

            self.assertEqual(
                (output_dir / "code_snapshot" / MODULE_PATH.name).read_bytes(),
                MODULE_PATH.read_bytes(),
            )
            test_path = Path(__file__).resolve()
            self.assertEqual(
                (output_dir / "code_snapshot" / test_path.name).read_bytes(),
                test_path.read_bytes(),
            )
            self.assertTrue((output_dir / "RUN_COMPLETE").is_file())

            checksum_path = output_dir / "OUTPUT_SHA256SUMS.txt"
            observed = {}
            for line in checksum_path.read_text().splitlines():
                digest, relative = line.split("  ", 1)
                observed[relative] = digest
            expected = {
                path.relative_to(output_dir).as_posix()
                for path in output_dir.rglob("*")
                if path.is_file() and path != checksum_path
            }
            self.assertEqual(set(observed), expected)
            for relative, digest in observed.items():
                self.assertEqual(digest, MODULE.sha256(output_dir / relative))

    def test_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "already_exists"
            output_dir.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.run_audit(
                    [("missing", Path(tmp) / "missing")], (1,), 0.90, output_dir
                )

    def test_raw_pl_tie_uses_vcf_order_and_is_reportable(self):
        self.assertEqual(MODULE.raw_gt_from_pl("20,0,0"), "0/1")
        self.assertTrue(MODULE.raw_pl_is_tied("20,0,0"))
        self.assertIsNone(MODULE.unique_raw_gt_from_pl("20,0,0"))


if __name__ == "__main__":
    unittest.main()
