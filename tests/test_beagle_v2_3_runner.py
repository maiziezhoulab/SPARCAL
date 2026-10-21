import importlib.util
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
    / "run_beagle_v2_3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beagle_v2_3_runner", MODULE_PATH
)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class BeagleV23RunnerTests(unittest.TestCase):
    def pipeline_without_initialization(self):
        pipeline = RUNNER.BeagleV23Pipeline.__new__(
            RUNNER.BeagleV23Pipeline
        )
        pipeline.get_1000genome_reference = mock.Mock(
            return_value="/reference/chr1.vcf.gz"
        )
        pipeline.merge_vcf_fields = mock.Mock()
        return pipeline

    def test_gl_first_pass_rejects_misleading_imputation_flag(self):
        pipeline = self.pipeline_without_initialization()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "has no effect"):
                pipeline.run_beagle_command(
                    "raw.vcf.gz",
                    str(root / "chr1"),
                    "chr1",
                    {
                        "IMPUTE": True,
                        "MEMORY": "1g",
                        "MODEL_SCALE": 2,
                        "MAX_LR": 5000,
                        "THREADS": 1,
                        "GPROBS": True,
                        "ITERATIONS": 0,
                        "SEED": 1,
                        "GENETIC_MAP": None,
                    },
                    str(root / "chr1.log"),
                )

    def test_command_pins_influential_parameters_and_optional_map(self):
        pipeline = self.pipeline_without_initialization()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_prefix = root / "chr1"
            Path(f"{output_prefix}.temp.vcf.gz").touch()
            map_path = root / "map_chr1.txt"
            map_path.touch()
            params = {
                "IMPUTE": False,
                "MEMORY": "1g",
                "MODEL_SCALE": 1.25,
                "MAX_LR": 1000,
                "THREADS": 3,
                "GPROBS": True,
                "ITERATIONS": 5,
                "SEED": 77,
                "GENETIC_MAP": str(root / "map_{chrom}.txt"),
            }
            with mock.patch.object(
                RUNNER.subprocess, "run"
            ) as run_mock:
                pipeline.run_beagle_command(
                    "raw.vcf.gz",
                    str(output_prefix),
                    "chr1",
                    params,
                    str(root / "chr1.log"),
                )
            command = run_mock.call_args.args[0]
            for expected in (
                "impute=false",
                "modelscale=1.25",
                "maxlr=1000",
                "nthreads=3",
                "gprobs=true",
                "niterations=5",
                "seed=77",
                f"map={map_path}",
            ):
                self.assertIn(expected, command)
            pipeline.merge_vcf_fields.assert_called_once()

    def test_reference_provenance_preserves_logical_ledger_paths(self):
        pipeline = self.pipeline_without_initialization()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "physical_panel.vcf.gz"
            target.write_text("panel\n")
            target_index = root / "physical_panel.vcf.gz.tbi"
            target_index.write_text("index\n")
            logical = root / "logical_panel.vcf.gz"
            logical.symlink_to(target)
            logical_index = Path(str(logical) + ".tbi")
            logical_index.symlink_to(target_index)
            pipeline.regions = ["chr1"]
            pipeline.get_1000genome_reference = mock.Mock(
                return_value=str(logical)
            )
            pipeline._reference_sums = {
                str(logical): RUNNER.CONTRACTS.sha256_file(target),
                str(logical_index): RUNNER.CONTRACTS.sha256_file(target_index),
            }
            pipeline._panel_keys_by_contig = {
                "chr1": {("chr1", 10, "A", "G")}
            }
            row = pipeline._reference_provenance()[0]
            self.assertEqual(row["panel"]["path"], str(logical.absolute()))
            self.assertEqual(
                row["panel_index"]["path"], str(logical_index.absolute())
            )
            self.assertNotEqual(row["panel"]["path"], str(target.resolve()))

    def test_log_and_output_are_not_reused(self):
        pipeline = self.pipeline_without_initialization()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_prefix = root / "chr1"
            Path(f"{output_prefix}.temp.vcf.gz").touch()
            log = root / "chr1.log"
            log.write_text("existing\n")
            with self.assertRaises(FileExistsError):
                pipeline.run_beagle_command(
                    "raw.vcf.gz",
                    str(output_prefix),
                    "chr1",
                    {
                        "IMPUTE": False,
                        "MEMORY": "1g",
                        "MODEL_SCALE": 2,
                        "MAX_LR": 5000,
                        "THREADS": 1,
                        "GPROBS": True,
                        "ITERATIONS": 0,
                        "SEED": 1,
                        "GENETIC_MAP": None,
                    },
                    str(log),
                )


if __name__ == "__main__":
    unittest.main()
