import gzip
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
    "beagle_v2_3_contracts", MODULE_PATH
)
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)


def write_vcf(path: Path, records, *, gzip_output=False):
    text = "".join(
        [
            "##fileformat=VCFv4.2\n",
            "##contig=<ID=chr1>\n",
            (
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
                "\tFORMAT\tsample\n"
            ),
            *(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t."
                "\tGT:GP\t0/1:0.01,0.98,0.01\n"
                for chrom, pos, ref, alt in records
            ),
        ]
    )
    if gzip_output:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)


class BeagleV23MarkerContractTests(unittest.TestCase):
    def test_digest_is_order_independent_and_allele_exact(self):
        first = {
            ("chr1", 20, "A", "C"),
            ("chr1", 10, "G", "T"),
        }
        second = list(reversed(sorted(first)))
        self.assertEqual(
            CONTRACTS.digest_keys(first), CONTRACTS.digest_keys(second)
        )
        self.assertNotEqual(
            CONTRACTS.digest_keys(first),
            CONTRACTS.digest_keys(
                {
                    ("chr1", 20, "A", "G"),
                    ("chr1", 10, "G", "T"),
                }
            ),
        )

    def test_raw_and_beagle_multiallelic_records_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.vcf"
            write_vcf(path, [("chr1", 10, "A", "C,G")])
            with self.assertRaisesRegex(ValueError, "split to biallelic"):
                CONTRACTS.read_exact_keys(path, contig="chr1")

    def test_reference_multiallelic_record_does_not_match_biallelic_raw(self):
        raw = {("chr1", 10, "A", "G")}
        lines = ["chr1\t10\t.\tA\tC,G\t.\tPASS\t."]
        self.assertEqual(
            CONTRACTS._panel_overlap_from_lines(
                lines, raw, source="panel"
            ),
            set(),
        )

    def test_panel_query_is_exact_and_removes_temp_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel.vcf.gz"
            panel.touch()
            Path(str(panel) + ".tbi").touch()
            completed = mock.Mock(
                stdout=(
                    "chr1\t10\t.\tA\tC,G\t.\tPASS\t.\n"
                    "chr1\t20\t.\tC\tT\t.\tPASS\t.\n"
                )
            )
            raw = {
                ("chr1", 10, "A", "G"),
                ("chr1", 20, "C", "T"),
            }
            with mock.patch.object(
                CONTRACTS.subprocess, "run", return_value=completed
            ) as run:
                overlap = CONTRACTS.query_reference_overlap(
                    raw, panel, contig="chr1", temp_dir=root
                )
            self.assertEqual(overlap, {("chr1", 20, "C", "T")})
            region_path = Path(run.call_args.args[0][3])
            self.assertFalse(region_path.exists())

    def test_complete_contract_requires_full_set_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "chr1.vcf.gz"
            write_vcf(
                output,
                [("chr1", 10, "A", "G")],
                gzip_output=True,
            )
            Path(str(output) + ".tbi").write_text("index")
            raw = {
                ("chr1", 10, "A", "G"),
                ("chr1", 20, "C", "T"),
            }
            panel = set(raw)
            with self.assertRaisesRegex(
                ValueError, "raw∩panel != raw∩Beagle"
            ):
                CONTRACTS.build_contig_contract(
                    contig="chr1",
                    raw_keys=raw,
                    panel_overlap_keys=panel,
                    reference_panel=root / "panel.vcf.gz",
                    beagle_vcf=output,
                )

    def test_complete_contract_accepts_imputed_nonraw_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "chr1.vcf.gz"
            write_vcf(
                output,
                [
                    ("chr1", 10, "A", "G"),
                    ("chr1", 999, "C", "A"),
                ],
                gzip_output=True,
            )
            Path(str(output) + ".tbi").write_text("index")
            raw = {
                ("chr1", 10, "A", "G"),
                ("chr1", 20, "C", "T"),
            }
            tabix_result = mock.Mock(
                stdout=(
                    "chr1\t10\t.\tA\tG\t.\tPASS\t."
                    "\tGT\t0/1\n"
                )
            )
            with mock.patch.object(
                CONTRACTS.subprocess,
                "run",
                return_value=tabix_result,
            ):
                contract = CONTRACTS.build_contig_contract(
                    contig="chr1",
                    raw_keys=raw,
                    panel_overlap_keys={
                        ("chr1", 10, "A", "G"),
                    },
                    reference_panel=root / "panel.vcf.gz",
                    beagle_vcf=output,
                )
            self.assertEqual(contract["status"], "completed")
            self.assertTrue(contract["exact_identity_pass"])
            self.assertEqual(
                contract["raw_panel_exact_alleles"]["count"], 1
            )
            self.assertEqual(
                contract["raw_beagle_exact_alleles"]["count"], 1
            )

    def test_empty_and_panel_absent_states_reject_fake_outputs(self):
        cases = (
            (set(), set(), "raw_empty"),
            (
                {("chr1", 1, "A", "G")},
                set(),
                "panel_absent",
            ),
        )
        for raw, panel, state in cases:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                contract = CONTRACTS.build_contig_contract(
                    contig="chr1",
                    raw_keys=raw,
                    panel_overlap_keys=panel,
                    reference_panel=root / "panel.vcf.gz",
                )
                self.assertEqual(contract["status"], state)
                fake = root / "fake.vcf.gz"
                fake.touch()
                with self.assertRaisesRegex(ValueError, "must not have"):
                    CONTRACTS.build_contig_contract(
                        contig="chr1",
                        raw_keys=raw,
                        panel_overlap_keys=panel,
                        reference_panel=root / "panel.vcf.gz",
                        beagle_vcf=fake,
                    )


class BeagleV23BundleContractTests(unittest.TestCase):
    def test_completion_verifies_then_detects_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "routes.tsv"
            manifest = root / "MANIFEST.json"
            artifact.write_text("route\tcount\nanchor\t1\n")
            manifest.write_text(json.dumps({"schema": "test"}) + "\n")
            checksums = CONTRACTS.write_sha256sums(
                root, [artifact, manifest]
            )
            CONTRACTS.write_completion(
                root,
                manifest_path=manifest,
                checksum_path=checksums,
            )
            result = CONTRACTS.verify_complete_bundle(root)
            self.assertEqual(result["artifact_count"], 2)
            artifact.write_text("mutated\n")
            with self.assertRaisesRegex(
                ValueError, "artifact checksum mismatch"
            ):
                CONTRACTS.verify_complete_bundle(root)

    def test_completion_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completion = {
                "schema": CONTRACTS.COMPLETION_SCHEMA,
                "status": "complete",
                "production_inputs_include_wes": False,
                "manifest": "../outside.json",
                "manifest_sha256": "0" * 64,
                "checksum_file": "checksums.txt",
                "checksum_file_sha256": "0" * 64,
            }
            (root / "RUN_COMPLETE.json").write_text(
                json.dumps(completion)
            )
            with self.assertRaisesRegex(
                ValueError, "escapes bundle root"
            ):
                CONTRACTS.verify_complete_bundle(root)


if __name__ == "__main__":
    unittest.main()
