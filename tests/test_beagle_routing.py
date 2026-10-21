import importlib.util
import gzip
import json
from pathlib import Path
import sys
from unittest import mock
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "2_beagle_filtering"
    / "run_beagle.py"
)
SPEC = importlib.util.spec_from_file_location("run_beagle", MODULE_PATH)
RUN_BEAGLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_BEAGLE)

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "postanalyze"
    / "beagle_evidence_benchmark.py"
)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "beagle_evidence_benchmark", BENCHMARK_PATH
)
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)


def write_vcf_gz(path, records=(), contigs=("chr1", "chr2"), sample="sample"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        for contig in contigs:
            handle.write(f"##contig=<ID={contig}>\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            f"{sample}\n"
        )
        for chrom, pos, ref, alt in records:
            handle.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\tAF=0.1\t"
                "GT:GP\t0/1:0.01,0.98,0.01\n"
            )


def make_minimal_runner(root, regions=("chr1", "chr2")):
    runner = object.__new__(RUN_BEAGLE.BeaglePipeline)
    runner.dataset_name = "DLPFC"
    runner.section_id = "test"
    runner.quality_filter = "baseQ0mapQ0"
    runner.anchor_score = "p_nonref"
    runner.min_anchor_probability = 0.90
    runner.regions = list(regions)
    runner.output_dirs = {
        "input_vcf_dir": str(root / "input"),
        "output_vcf_dir": str(root / "output"),
        "log_dir": str(root / "logs"),
        "metrics_dir": str(root / "metrics"),
    }
    for path in runner.output_dirs.values():
        Path(path).mkdir(parents=True)
    return runner


class BeagleRoutingTests(unittest.TestCase):
    def test_main_runner_loads_authoritative_pnonref_router(self):
        router = RUN_BEAGLE._load_beagle_router()
        evidence = router.summarize_evidence("0/1", "0.08,0.50,0.42")
        self.assertEqual(
            router.route_evidence(evidence, 0.90),
            "beagle_nonref_highconf",
        )
        self.assertEqual(
            router.ROUTER_SCHEMAS["p_nonref"],
            "sparcal_beagle_evidence_v2_2",
        )

    def test_runner_claims_empty_tree_and_refuses_in_place_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = {name: Path(tmp) / name for name in ("output_vcf_dir", "log_dir", "metrics_dir")}
            for root in roots.values():
                root.mkdir()
            runner = object.__new__(RUN_BEAGLE.BeaglePipeline)
            runner.output_dirs = {name: str(path) for name, path in roots.items()}
            runner.dataset_name = "DLPFC"
            runner.section_id = "test"
            runner.quality_filter = "baseQ0mapQ0"
            runner.anchor_score = "p_nonref"
            runner.min_anchor_probability = 0.90
            marker = runner._claim_fresh_output_tree({"ITERATIONS": 0})
            self.assertTrue(marker.is_file())
            with self.assertRaisesRegex(FileExistsError, "non-empty Beagle run tree"):
                runner._claim_fresh_output_tree({"ITERATIONS": 0})

    def test_legacy_called_gp_rule_remains_available_for_ablation(self):
        router = RUN_BEAGLE._load_beagle_router()
        evidence = router.summarize_evidence("0/1", "0.08,0.50,0.42")
        self.assertEqual(
            router.route_evidence(evidence, 0.90, anchor_score="called_gp"),
            "beagle_uncertain",
        )

    def test_monopogen_raw_pl_argmax_uses_vcf_order_for_ties(self):
        self.assertEqual(BENCHMARK.genotype_from_pl("90,33,0"), "1/1")
        self.assertEqual(BENCHMARK.genotype_from_pl("0,40,82"), "0/0")
        self.assertEqual(BENCHMARK.genotype_from_pl("30,0,40"), "0/1")
        self.assertEqual(BENCHMARK.genotype_from_pl("0,0,20"), "0/0")
        self.assertIsNone(BENCHMARK.genotype_from_pl("bad"))

    def test_monopogen_paper_and_released_code_are_separate_methods(self):
        key = ("chr1", 100, "A", "G")
        decisions = {key: {}}
        evidence = {
            key: BENCHMARK.MonopogenEvidence(
                raw_gl_gt="0/0",
                beagle_gp_gt="0/1",
                final_phased_gt="0/1",
                called_gp=0.89,
                p_nonref=0.96,
                final_phased_present=True,
            )
        }
        BENCHMARK.add_monopogen_decisions(decisions, evidence)
        self.assertTrue(
            decisions[key]["Monopogen released-code phased-record membership component"].anchor
        )
        self.assertTrue(
            decisions[key]["Monopogen final-phased nonref-GT diagnostic"].anchor
        )
        self.assertFalse(
            decisions[key]["Monopogen paper concordant nonref"].anchor
        )
        self.assertTrue(
            decisions[key]["Monopogen P(nonref)>=0.90 (derived)"].anchor
        )


    def test_released_membership_does_not_require_nonreference_final_gt(self):
        key = ("chr1", 100, "A", "G")
        decisions = {key: {}}
        evidence = {
            key: BENCHMARK.MonopogenEvidence(
                raw_gl_gt="0/0",
                beagle_gp_gt="0/0",
                final_phased_gt="0/0",
                called_gp=0.99,
                p_nonref=0.01,
                final_phased_present=True,
            )
        }
        BENCHMARK.add_monopogen_decisions(decisions, evidence)
        self.assertTrue(decisions[key][BENCHMARK.MONO_RELEASED_COMPONENT].anchor)
        self.assertFalse(
            decisions[key][BENCHMARK.MONO_PHASED_GT_DIAGNOSTIC].anchor
        )

    def test_legacy_benchmark_label_matches_frozen_called_gp_contract(self):
        self.assertIn("called-GP>=0.90", BENCHMARK.LEGACY_CONCORDANCE_METHOD)
        definitions = {
            row["method"]: row["definition"]
            for row in BENCHMARK.method_definitions()
        }
        definition = definitions[BENCHMARK.LEGACY_CONCORDANCE_METHOD]
        self.assertIn("raw non-reference GT", definition)
        self.assertIn("called Beagle GT is at least 0.90", definition)

    def test_future_completion_marker_is_covered_by_output_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result.tsv").write_text("value\n1\n")
            payload = b'{"status": "complete"}\n'
            BENCHMARK.write_output_checksums(root, {"RUN_COMPLETE": payload})
            (root / "RUN_COMPLETE").write_bytes(payload)
            entries = BENCHMARK.verify_checksum_tree(
                root, "OUTPUT_SHA256SUMS.txt"
            )
            self.assertIn("RUN_COMPLETE", entries)
            self.assertEqual(
                entries["RUN_COMPLETE"], BENCHMARK.sha256_bytes(payload)
            )

    def test_monopogen_derived_pnonref_matches_sweep_contract(self):
        nonref_boundary = ("chr1", 101, "A", "C")
        reference_high = ("chr1", 102, "G", "T")
        decisions = {nonref_boundary: {}, reference_high: {}}
        evidence = {
            nonref_boundary: BENCHMARK.MonopogenEvidence(
                None, "0/1", "0/1", 0.5, 0.90 - 5e-13
            ),
            reference_high: BENCHMARK.MonopogenEvidence(
                "0/0", "0/0", "0/0", 0.96, 0.96
            ),
        }
        BENCHMARK.add_monopogen_decisions(decisions, evidence)
        method = "Monopogen P(nonref)>=0.90 (derived)"
        self.assertTrue(decisions[nonref_boundary][method].anchor)
        self.assertFalse(decisions[reference_high][method].anchor)
    def test_process_chromosome_distinguishes_empty_and_panel_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = make_minimal_runner(Path(tmp))
            with mock.patch.object(
                runner,
                "chromosome_has_reference_marker",
                side_effect=((False, 0), (False, 3)),
            ):
                raw_empty = runner.process_chromosome("chr1", {})
                panel_absent = runner.process_chromosome("chr2", {})

            self.assertEqual(raw_empty["status"], "raw_empty")
            self.assertEqual(raw_empty["route_expectation"], "no_raw_records")
            self.assertEqual(panel_absent["status"], "panel_absent")
            self.assertEqual(panel_absent["raw_marker_count"], 3)
            self.assertFalse(panel_absent["beagle_required"])

    def test_runner_passes_every_completed_contig_to_router_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = make_minimal_runner(root)
            raw = Path(runner.output_dirs["input_vcf_dir"]) / "merged_sorted_gt.vcf.gz"
            write_vcf_gz(raw, records=(("chr1", 100, "A", "G"),))

            def fake_process(chromosome, params):
                if chromosome == "chr2":
                    return {
                        "chromosome": chromosome,
                        "status": "raw_empty",
                        "raw_marker_count": 0,
                        "has_reference_marker": False,
                        "beagle_required": False,
                        "route_expectation": "no_raw_records",
                        "output_file": None,
                    }
                output = Path(runner.output_dirs["output_vcf_dir"]) / "chr1.vcf.gz"
                write_vcf_gz(output, records=(("chr1", 100, "A", "G"),))
                Path(str(output) + ".tbi").touch()
                return {
                    "chromosome": chromosome,
                    "status": "completed",
                    "raw_marker_count": 1,
                    "has_reference_marker": True,
                    "beagle_required": True,
                    "route_expectation": "beagle_evidence_required",
                    "output_file": str(output),
                }

            router = RUN_BEAGLE._load_beagle_router()
            route_counts = {route: 0 for route in router.ROUTES}
            with mock.patch.object(runner, "process_chromosome", side_effect=fake_process), mock.patch.object(
                router, "route_vcf", return_value=route_counts
            ) as route_mock:
                runner.run_pipeline({"PARALLEL_CHROMOSOMES": 1})

            self.assertEqual(route_mock.call_args.kwargs["expected_contigs"], ["chr1", "chr2"])
            self.assertTrue(route_mock.call_args.kwargs["contig_status_path"].is_file())
            status = json.loads(
                (Path(runner.output_dirs["output_vcf_dir"]) / "BEAGLE_CONTIG_STATUS.json").read_text()
            )
            self.assertEqual(status["expected_contigs"], ["chr1", "chr2"])
            self.assertEqual(
                status["status_counts"],
                {"completed": 1, "panel_absent": 0, "raw_empty": 1},
            )

    def test_completed_header_only_or_wrong_contig_output_fails_closed(self):
        for beagle_records in ((), (("chr2", 200, "C", "T"),), (("chr1", 999, "C", "T"),)):
            with self.subTest(beagle_records=beagle_records), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner = make_minimal_runner(root)
                raw = Path(runner.output_dirs["input_vcf_dir"]) / "merged_sorted_gt.vcf.gz"
                write_vcf_gz(raw, records=(("chr1", 100, "A", "G"),))

                def fake_process(chromosome, params):
                    if chromosome == "chr2":
                        return {
                            "chromosome": chromosome,
                            "status": "raw_empty",
                            "raw_marker_count": 0,
                            "has_reference_marker": False,
                            "output_file": None,
                        }
                    output = Path(runner.output_dirs["output_vcf_dir"]) / "chr1.vcf.gz"
                    write_vcf_gz(output, records=beagle_records)
                    Path(str(output) + ".tbi").touch()
                    return {
                        "chromosome": chromosome,
                        "status": "completed",
                        "raw_marker_count": 1,
                        "has_reference_marker": True,
                        "output_file": str(output),
                    }

                with mock.patch.object(runner, "process_chromosome", side_effect=fake_process):
                    with self.assertRaisesRegex(ValueError, "completed Beagle output"):
                        runner.run_pipeline({"PARALLEL_CHROMOSOMES": 1})
                self.assertFalse(
                    (Path(runner.output_dirs["output_vcf_dir"]) / "RUN_COMPLETE.json").exists()
                )

    def test_all_panel_absent_sample_routes_without_fake_beagle_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = make_minimal_runner(root)
            raw = Path(runner.output_dirs["input_vcf_dir"]) / "merged_sorted_gt.vcf.gz"
            write_vcf_gz(raw, records=(("chr1", 100, "A", "G"),))

            def fake_process(chromosome, params):
                if chromosome == "chr1":
                    return {
                        "chromosome": chromosome,
                        "status": "panel_absent",
                        "raw_marker_count": 1,
                        "has_reference_marker": False,
                        "beagle_required": False,
                        "route_expectation": "all_raw_alleles_panel_absent",
                        "output_file": None,
                    }
                return {
                    "chromosome": chromosome,
                    "status": "raw_empty",
                    "raw_marker_count": 0,
                    "has_reference_marker": False,
                    "beagle_required": False,
                    "route_expectation": "no_raw_records",
                    "output_file": None,
                }

            with mock.patch.object(runner, "process_chromosome", side_effect=fake_process):
                metrics = runner.run_pipeline({"PARALLEL_CHROMOSOMES": 1})

            self.assertEqual(set(metrics["status"]), {"panel_absent", "raw_empty"})
            output = Path(runner.output_dirs["output_vcf_dir"])
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(manifest["routing_mode"], "all_panel_absent_no_beagle")
            self.assertEqual(manifest["beagle_vcfs"], [])
            self.assertEqual(manifest["beagle_vcf_count"], 0)
            self.assertEqual(manifest["counts"]["panel_absent"], 1)
            self.assertEqual(manifest["counts"]["total"], 1)
            self.assertFalse((output / "chr1.vcf.gz").exists())
            with gzip.open(output / "all_filtered_out.vcf.gz", "rt") as handle:
                records = [line for line in handle if not line.startswith("#")]
            self.assertEqual(len(records), 1)
            with gzip.open(output / "all_filtered_in.vcf.gz", "rt") as handle:
                self.assertFalse(any(not line.startswith("#") for line in handle))
            with gzip.open(output / "all_panel_present.vcf.gz", "rt") as handle:
                self.assertFalse(any(not line.startswith("#") for line in handle))
            completion = json.loads((output / "RUN_COMPLETE.json").read_text())
            self.assertEqual(completion["routing_mode"], "all_panel_absent_no_beagle")



if __name__ == "__main__":
    unittest.main()
