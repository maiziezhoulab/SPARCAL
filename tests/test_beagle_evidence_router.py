import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "2_beagle_filtering"
    / "beagle_evidence_router.py"
)
SPEC = importlib.util.spec_from_file_location("beagle_evidence_router", MODULE_PATH)
ROUTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ROUTER
SPEC.loader.exec_module(ROUTER)


def write_vcf(
    path: Path,
    sample_id: str = "sample",
    records=(("chr1", 100, "A", "G"),),
    contigs=("chr1",),
    extra_sample_ids=(),
) -> None:
    lines = ["##fileformat=VCFv4.2\n"]
    lines.extend(f"##contig=<ID={contig}>\n" for contig in contigs)
    samples = (sample_id, *extra_sample_ids)
    lines.append(
        "\t".join((
            "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
            "INFO", "FORMAT", *samples,
        )) + "\n"
    )
    for chrom, position, ref, alt in records:
        sample_fields = ("0/1:0.01,0.98,0.01",) * len(samples)
        lines.append(
            "\t".join((
                chrom, str(position), ".", ref, alt, ".", "PASS", "AF=0.1",
                "GT:GP", *sample_fields,
            )) + "\n"
        )
    path.write_text("".join(lines))


class BeagleEvidenceRouterTests(unittest.TestCase):
    def test_genotype_normalization(self):
        self.assertEqual(ROUTER.normalize_gt("1|0"), "0/1")
        self.assertEqual(ROUTER.normalize_gt("1/1"), "1/1")
        self.assertIsNone(ROUTER.normalize_gt(".|."))

    def test_gp_is_normalized_and_summarized(self):
        evidence = ROUTER.summarize_evidence("0|1", "0.02,0.96,0.02", "0.1")
        self.assertAlmostEqual(evidence.called_gp, 0.96)
        self.assertAlmostEqual(evidence.p_nonref, 0.98)
        self.assertAlmostEqual(evidence.beagle_af, 0.1)
        self.assertGreater(evidence.margin, 0.9)
        self.assertGreaterEqual(evidence.entropy, 0)
        self.assertLessEqual(evidence.entropy, 1)

    def test_reported_gp_is_not_shifted_by_rounding_sum(self):
        evidence = ROUTER.summarize_evidence("0/1", "0.01,0.90,0.10")
        self.assertAlmostEqual(evidence.called_gp, 0.90)
        self.assertAlmostEqual(evidence.p_nonref, 1.00)
        self.assertEqual(
            ROUTER.route_evidence(evidence, 0.90), "beagle_nonref_highconf"
        )

    def test_nonref_probability_separates_anchor_status_from_exact_gt(self):
        evidence = ROUTER.summarize_evidence("0/1", "0.08,0.50,0.42")
        self.assertEqual(
            ROUTER.route_evidence(evidence, 0.90), "beagle_nonref_highconf"
        )
        self.assertEqual(
            ROUTER.route_evidence(evidence, 0.90, anchor_score="called_gp"),
            "beagle_uncertain",
        )

    def test_nonref_decimal_sum_at_threshold_is_inclusive(self):
        evidence = ROUTER.summarize_evidence("0/1", "0.10,0.29,0.61")
        self.assertLess(evidence.p_nonref, 0.90)
        self.assertEqual(
            ROUTER.route_evidence(evidence, 0.90), "beagle_nonref_highconf"
        )

    def test_threshold_uses_absolute_tolerance_only(self):
        self.assertTrue(ROUTER.meets_threshold(0.90 - 0.5e-12, 0.90))
        self.assertFalse(ROUTER.meets_threshold(0.90 - 2e-12, 0.90))
        self.assertFalse(ROUTER.meets_threshold(0.90 - 0.5e-9, 0.90))

    def test_four_routes_do_not_require_raw_concordance(self):
        nonref = ROUTER.summarize_evidence("0/1", "0.001,0.998,0.001")
        ref = ROUTER.summarize_evidence("0/0", "0.99,0.01,0")
        uncertain = ROUTER.summarize_evidence("0/1", "0.08,0.89,0.03")
        self.assertEqual(
            ROUTER.route_evidence(nonref, 0.90), "beagle_nonref_highconf"
        )
        self.assertEqual(
            ROUTER.route_evidence(ref, 0.90), "beagle_ref_highconf"
        )
        self.assertEqual(
            ROUTER.route_evidence(uncertain, 0.90, anchor_score="called_gp"),
            "beagle_uncertain",
        )
        self.assertEqual(ROUTER.route_evidence(None, 0.90), "panel_absent")

    def test_missing_or_bad_gp_is_uncertain(self):
        for gp in (None, ".", "bad", "0.1,0.9"):
            evidence = ROUTER.summarize_evidence("0/1", gp)
            self.assertEqual(
                ROUTER.route_evidence(evidence, 0.90), "beagle_uncertain"
            )

    def test_compatibility_outputs_preserve_downstream_semantics(self):
        self.assertEqual(
            ROUTER.compatibility_outputs("beagle_nonref_highconf"),
            ("filtered_in", "panel_present"),
        )
        self.assertEqual(
            ROUTER.compatibility_outputs("beagle_ref_highconf"),
            ("filtered_out", "panel_discordant", "panel_present"),
        )
        self.assertEqual(
            ROUTER.compatibility_outputs("beagle_uncertain"),
            ("filtered_out", "panel_discordant", "panel_present"),
        )
        self.assertEqual(
            ROUTER.compatibility_outputs("panel_absent"),
            ("filtered_out",),
        )

    def test_success_preserves_compatibility_routes_and_records_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "beagle.vcf"
            output = root / "routes"
            write_vcf(
                raw,
                records=(
                    ("chr1", 100, "A", "G"),
                    ("chr1", 200, "C", "T"),
                ),
            )
            write_vcf(beagle, records=(("chr1", 100, "A", "G"),))

            with mock.patch.object(ROUTER.subprocess, "run") as run_mock:
                counts = ROUTER.route_vcf(
                    raw,
                    [beagle],
                    output,
                    expected_contigs=("chr1",),
                )

            self.assertEqual(counts["beagle_nonref_highconf"], 1)
            self.assertEqual(counts["panel_absent"], 1)
            self.assertEqual(run_mock.call_count, 16)

            def data_lines(name):
                return [
                    line
                    for line in (output / name).read_text().splitlines()
                    if not line.startswith("#")
                ]

            self.assertEqual(len(data_lines("all_filtered_in.vcf")), 1)
            self.assertEqual(len(data_lines("all_filtered_out.vcf")), 1)
            self.assertEqual(len(data_lines("all_panel_discordant.vcf")), 0)
            self.assertEqual(len(data_lines("all_panel_present.vcf")), 1)
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(
                manifest["schema"], "sparcal_beagle_evidence_v2_2"
            )
            self.assertIn("all_panel_present.vcf.gz", manifest["compatibility"])
            self.assertEqual(manifest["raw_sample_id"], "sample")
            self.assertEqual(manifest["beagle_sample_id"], "sample")
            self.assertEqual(manifest["expected_contigs"], ["chr1"])
            self.assertEqual(manifest["expected_contigs_source"], "explicit")
            self.assertEqual(manifest["exact_overlap_count"], 1)
            self.assertEqual(manifest["counts"]["total"], 2)
            self.assertTrue((output / "RUN_COMPLETE").is_file())

    def test_output_directory_must_be_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "already_here"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                ROUTER.route_vcf(
                    Path(tmp) / "raw.vcf.gz", [], output, min_called_gp=0.90
                )

    def test_existing_beagle_directory_is_allowed_but_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_output = Path(tmp) / "empty_beagle_dir"
            empty_output.mkdir()
            with self.assertRaises(FileNotFoundError):
                ROUTER.route_vcf(
                    Path(tmp) / "missing.vcf.gz", [], empty_output,
                    allow_existing_output_dir=True,
                )

            protected_output = Path(tmp) / "protected_beagle_dir"
            protected_output.mkdir()
            (protected_output / "all_filtered_in.vcf.gz").touch()
            with self.assertRaisesRegex(
                FileExistsError, "refusing to overwrite route outputs"
            ):
                ROUTER.route_vcf(
                    Path(tmp) / "missing.vcf.gz", [], protected_output,
                    allow_existing_output_dir=True,
                )

    def test_malformed_beagle_data_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            beagle = Path(tmp) / "beagle.vcf"
            write_vcf(beagle)
            lines = beagle.read_text().splitlines()
            lines[-1] = "chr1\t100\t.\tA\tG\t.\tPASS\tAF=0.1\tGT:GP"
            beagle.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "expected 10 columns"):
                ROUTER.collect_beagle_evidence([beagle])

    def test_multi_sample_beagle_vcf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            beagle = Path(tmp) / "beagle.vcf"
            write_vcf(beagle, extra_sample_ids=("second_sample",))
            with self.assertRaisesRegex(ValueError, "exactly one sample"):
                ROUTER.collect_beagle_evidence([beagle])

    def test_multi_sample_raw_vcf_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "beagle.vcf"
            output = root / "routes"
            write_vcf(raw, extra_sample_ids=("second_sample",))
            write_vcf(beagle)
            with self.assertRaisesRegex(ValueError, "exactly one sample"):
                ROUTER.route_vcf(raw, [beagle], output)
            self.assertFalse(output.exists())

    def test_raw_and_beagle_sample_ids_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "beagle.vcf"
            output = root / "routes"
            write_vcf(raw, sample_id="raw_sample")
            write_vcf(beagle, sample_id="beagle_sample")
            with self.assertRaisesRegex(
                ValueError,
                "raw='raw_sample', Beagle='beagle_sample'",
            ):
                ROUTER.route_vcf(raw, [beagle], output)
            self.assertFalse(output.exists())

    def test_beagle_inputs_must_share_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "chr1.vcf"
            second = root / "chr2.vcf"
            write_vcf(first, sample_id="first", contigs=("chr1",))
            write_vcf(
                second,
                sample_id="second",
                records=(("chr2", 200, "C", "T"),),
                contigs=("chr2",),
            )
            with self.assertRaisesRegex(ValueError, "sample ID mismatch"):
                ROUTER.collect_beagle_evidence([first, second])

    def test_expected_contigs_catch_a_missing_beagle_chromosome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "chr1.vcf"
            output = root / "routes"
            write_vcf(
                raw,
                records=(
                    ("chr1", 100, "A", "G"),
                    ("chr2", 200, "C", "T"),
                ),
                contigs=("chr1", "chr2"),
            )
            write_vcf(beagle, contigs=("chr1", "chr2"))
            with self.assertRaisesRegex(
                ValueError, "per-contig exact raw-allele coverage"
            ):
                ROUTER.route_vcf(
                    raw,
                    [beagle],
                    output,
                    expected_contigs=("chr1", "chr2"),
                )
            self.assertFalse(output.exists())

    def test_zero_exact_overlap_is_rejected_for_nonempty_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "beagle.vcf"
            output = root / "routes"
            write_vcf(raw, records=(("chr1", 100, "A", "G"),))
            write_vcf(beagle, records=(("chr1", 200, "A", "G"),))
            with self.assertRaisesRegex(ValueError, "per-contig exact raw-allele coverage"):
                ROUTER.route_vcf(raw, [beagle], output)
            self.assertFalse(output.exists())

    def test_header_only_beagle_bundle_is_zero_overlap_for_nonempty_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "beagle.vcf"
            output = root / "routes"
            write_vcf(raw)
            write_vcf(beagle, records=())
            with self.assertRaisesRegex(ValueError, "per-contig exact raw-allele coverage"):
                ROUTER.route_vcf(raw, [beagle], output)
            self.assertFalse(output.exists())

    def test_cli_expected_contig_parser_accepts_spaces_and_commas(self):
        self.assertEqual(
            ROUTER.parse_expected_contigs(["chr1,chr2", "chr2", "chrX"]),
            ("chr1", "chr2", "chrX"),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            ROUTER.parse_expected_contigs([",", "  "])

    def test_omitted_expected_contigs_infers_and_validates_valid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "chr1.vcf"
            output = root / "routes"
            write_vcf(raw, records=(("chr1", 100, "A", "G"),))
            write_vcf(beagle, records=(("chr1", 100, "A", "G"),))
            with mock.patch.object(ROUTER.subprocess, "run"):
                ROUTER.route_vcf(raw, [beagle], output, min_called_gp=0.90)
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(manifest["expected_contigs"], ["chr1"])
            self.assertEqual(manifest["expected_contigs_source"], "inferred")
            self.assertEqual(manifest["anchor_score"], "called_gp")
            self.assertEqual(manifest["exact_overlap_by_contig"], {"chr1": 1})

    def test_omitted_expected_contigs_rejects_header_only_and_wrong_contig(self):
        cases = (
            ((), ("chr1",)),
            ((("chr2", 200, "C", "T"),), ("chr1", "chr2")),
        )
        for records, contigs in cases:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = root / "raw.vcf"
                beagle = root / "chr1.vcf"
                output = root / "routes"
                write_vcf(raw, records=(("chr1", 100, "A", "G"),))
                write_vcf(beagle, records=records, contigs=contigs)
                with self.assertRaisesRegex(
                    ValueError,
                    "per-contig exact raw-allele coverage|outside expected_contigs",
                ):
                    ROUTER.route_vcf(raw, [beagle], output)
                self.assertFalse(output.exists())

    def test_inferred_mode_refuses_unproven_panel_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle = root / "chr1.vcf"
            output = root / "routes"
            write_vcf(
                raw,
                records=(
                    ("chr1", 100, "A", "G"),
                    ("chr2", 200, "C", "T"),
                ),
                contigs=("chr1", "chr2"),
            )
            write_vcf(beagle, records=(("chr1", 100, "A", "G"),))
            with self.assertRaisesRegex(
                ValueError, "per-contig exact raw-allele coverage for: chr2"
            ):
                ROUTER.route_vcf(raw, [beagle], output)
            self.assertFalse(output.exists())

    def test_inferred_mode_uses_trusted_panel_absent_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            beagle_dir = root / "beagle"
            beagle_dir.mkdir()
            beagle = beagle_dir / "chr1.vcf"
            output = root / "routes"
            write_vcf(
                raw,
                records=(
                    ("chr1", 100, "A", "G"),
                    ("chr2", 200, "C", "T"),
                ),
                contigs=("chr1", "chr2"),
            )
            write_vcf(beagle, records=(("chr1", 100, "A", "G"),))
            status_path = beagle_dir / "BEAGLE_CONTIG_STATUS.json"
            status_path.write_text(json.dumps({
                "production_inputs_include_wes": False,
                "raw_sample_id": "sample",
                "raw_record_count": 2,
                "expected_contigs": ["chr1", "chr2"],
                "contigs": [
                    {
                        "chromosome": "chr1",
                        "status": "completed",
                        "raw_marker_count": 1,
                        "has_reference_marker": True,
                        "beagle_required": True,
                        "route_expectation": "beagle_evidence_required",
                        "output_file": str(beagle.resolve()),
                    },
                    {
                        "chromosome": "chr2",
                        "status": "panel_absent",
                        "raw_marker_count": 1,
                        "has_reference_marker": False,
                        "beagle_required": False,
                        "route_expectation": "all_raw_alleles_panel_absent",
                        "output_file": None,
                    },
                ],
            }))
            with mock.patch.object(ROUTER.subprocess, "run"):
                counts = ROUTER.route_vcf(raw, [beagle], output)
            self.assertEqual(counts["beagle_nonref_highconf"], 1)
            self.assertEqual(counts["panel_absent"], 1)
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(manifest["expected_contigs_source"], "inferred")
            self.assertEqual(
                manifest["upstream_contig_status_file"],
                str(status_path.resolve()),
            )

    def test_frozen_called_gp_cli_without_expected_contigs_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beagle_dir = root / "beagle"
            beagle_dir.mkdir()
            (beagle_dir / "chr1.vcf.gz").touch()
            argv = [
                "beagle_evidence_router.py",
                "--raw-vcf", str(root / "raw.vcf.gz"),
                "--beagle-dir", str(beagle_dir),
                "--output-dir", str(root / "routes"),
                "--min-called-gp", "0.90",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                ROUTER, "route_vcf", return_value={}
            ) as route_mock:
                ROUTER.main()
            self.assertIsNone(route_mock.call_args.kwargs["expected_contigs"])
            self.assertEqual(route_mock.call_args.kwargs["min_called_gp"], 0.90)


if __name__ == "__main__":
    unittest.main()
