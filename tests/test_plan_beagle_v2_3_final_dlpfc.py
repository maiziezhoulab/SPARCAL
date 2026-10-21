from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/plan_beagle_v2_3_final_dlpfc.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plan_beagle_v2_3_final_dlpfc", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_code_ledger(path: Path) -> None:
    path.write_text(f"{MODULE.sha256(MODULE_PATH)}  {MODULE_PATH.resolve()}\n")


def finalize_bundle(root: Path, output_names, completion) -> None:
    ledger = root / "OUTPUT_SHA256SUMS.txt"
    ledger.write_text(
        "".join(
            f"{MODULE.sha256(root / name)}  {name}\n" for name in output_names
        )
    )
    completion["output_sha256sums"] = MODULE.identity(ledger)
    write_json(root / "RUN_COMPLETE.json", completion)
    (root / "RUN_COMPLETE").write_text("complete\n")


class FinalDlpfcPlanTests(unittest.TestCase):
    def build_inputs(
        self,
        root: Path,
        *,
        representation: str = "single_seed",
        p6_modelscale: float | None = None,
    ):
        base = "opaque_selected_setting"
        params = {
            "THREADS": 30,
            "MEMORY": "10g",
            "MODEL_SCALE": 1.75,
            "ITERATIONS": 4,
            "IMPUTE": False,
            "GPROBS": True,
            "SEED": MODULE.SINGLE_SEED,
            "PARALLEL_CHROMOSOMES": 1,
            "MAX_LR": 12345.0,
            "GENETIC_MAP": None,
            "REFERENCE_LEDGER": str((root / "reference_ledger").resolve()),
        }
        manifests = []
        for sample in MODULE.SAMPLES:
            run_root = root / "source_runs" / sample
            run_root.mkdir(parents=True)
            sample_params = dict(params)
            if sample == "P6" and p6_modelscale is not None:
                sample_params["MODEL_SCALE"] = p6_modelscale
            manifest = {
                "schema": "sparcal_beagle_run_v2_3_candidate_1",
                "status": "outputs_ready_for_checksum",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": False,
                "dataset": f"{sample}_TUMOR",
                "section_id": "1",
                "quality_filter": "baseQ0mapQ0",
                "anchor_score": "p_nonref",
                "min_anchor_probability": 0.90,
                "beagle_parameters": sample_params,
                "reference_ledger": {"root": params["REFERENCE_LEDGER"]},
            }
            manifest_path = run_root / "V2_3_MANIFEST.json"
            write_json(manifest_path, manifest)
            manifests.append({
                "config": f"{base}_seed{MODULE.SINGLE_SEED}",
                "sample": sample,
                "run_root": str(run_root),
                "manifest": MODULE.identity(manifest_path),
            })

        base_root = root / "base_assessment"
        base_root.mkdir()
        for name in (
            "replicated_configuration_metrics_long.tsv",
            "replicated_configuration_summary.tsv",
            "replicated_pairwise_stability.tsv",
            "replication_gate_results.tsv",
        ):
            (base_root / name).write_text(f"{name}\n")
        write_json(
            base_root / "FINAL_BASE_CONFIGURATION.json",
            {
                "schema": "sparcal_beagle_v2_3_final_base_selection_v1",
                "selected_base_configuration": base,
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "final_ensemble_selected": False,
            },
        )
        write_json(
            base_root / "PROVENANCE.json",
            {
                "schema": "sparcal_beagle_v2_3_replication_assessment_v1",
                "status": "base_configuration_selected",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "selected_base_configuration": base,
                "selected_configurations": [base],
                "seeds": list(MODULE.ALL_SEEDS),
                "samples": list(MODULE.SAMPLES),
                "validated_runs": manifests,
            },
        )
        write_code_ledger(base_root / "CODE_AND_INPUT_SHA256SUMS.txt")
        finalize_bundle(
            base_root,
            (
                "replicated_configuration_metrics_long.tsv",
                "replicated_configuration_summary.tsv",
                "replicated_pairwise_stability.tsv",
                "replication_gate_results.tsv",
                "FINAL_BASE_CONFIGURATION.json",
                "PROVENANCE.json",
                "CODE_AND_INPUT_SHA256SUMS.txt",
            ),
            {
                "schema": "sparcal_beagle_v2_3_replication_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "selected_base_configuration": base,
            },
        )

        final_root = root / "final_selection"
        final_root.mkdir()
        with (final_root / "ensemble_gate_results.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                delimiter="\t",
                fieldnames=("method", "selected_final_representation"),
            )
            writer.writeheader()
            for method in ("ensemble_mean", "ensemble_median"):
                writer.writerow({
                    "method": method,
                    "selected_final_representation": str(
                        method == representation
                    ).lower(),
                })
        write_json(
            final_root / "FINAL_BEAGLE_REPRESENTATION.json",
            {
                "schema": "sparcal_beagle_v2_3_final_representation_v1",
                "base_configuration": base,
                "selected_representation": representation,
                "selected_seed": (
                    MODULE.SINGLE_SEED if representation == "single_seed" else None
                ),
                "beagle_run_multiplier": (
                    1 if representation == "single_seed" else 3
                ),
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
            },
        )
        write_json(
            final_root / "PROVENANCE.json",
            {
                "schema": "sparcal_beagle_v2_3_ensemble_selection_v1",
                "status": "final_representation_selected",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "base_configuration": base,
                "selected_representation": representation,
                "frozen_seeds": list(MODULE.ALL_SEEDS),
            },
        )
        write_code_ledger(final_root / "CODE_AND_INPUT_SHA256SUMS.txt")
        finalize_bundle(
            final_root,
            (
                "ensemble_gate_results.tsv",
                "FINAL_BEAGLE_REPRESENTATION.json",
                "PROVENANCE.json",
                "CODE_AND_INPUT_SHA256SUMS.txt",
            ),
            {
                "schema": "sparcal_beagle_v2_3_ensemble_selection_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "base_configuration": base,
                "selected_representation": representation,
            },
        )
        return final_root, base_root, base, params

    def test_single_seed_plan_uses_manifest_parameters_and_exact_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, base, params = self.build_inputs(root)
            run_root = root / "fresh_runs"
            output = root / "plan"
            result = MODULE.build_plan(final_root, base_root, run_root, output)
            self.assertEqual(result["task_count"], 12)
            self.assertEqual(result["array_max"], 11)
            self.assertFalse(run_root.exists())
            with (output / "dlpfc_tasks.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["section"] for row in rows], list(MODULE.SECTIONS))
            self.assertEqual({row["seed"] for row in rows}, {"20260915"})
            self.assertEqual({row["modelscale"] for row in rows}, {"1.75"})
            self.assertEqual({row["niterations"] for row in rows}, {"4"})
            self.assertEqual({row["maxlr"] for row in rows}, {"12345.0"})
            self.assertEqual({row["base_config"] for row in rows}, {base})
            provenance = json.loads((output / "PROVENANCE.json").read_text())
            completion = json.loads((output / "RUN_COMPLETE.json").read_text())
            self.assertEqual(provenance["beagle_parameters"], params)
            self.assertFalse(provenance["production_inputs_include_wes"])
            self.assertFalse(provenance["evaluation_uses_wes"])
            self.assertEqual(
                provenance["selected_seed20260915_manifests"],
                completion["selected_seed20260915_manifests"],
            )
            self.assertEqual(
                [row["sample"] for row in provenance["selected_seed20260915_manifests"]],
                ["P4", "P6"],
            )

    def test_ensemble_plan_has_exact_three_seed_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, base, _ = self.build_inputs(
                root, representation="ensemble_median"
            )
            output = root / "plan"
            run_root = root / "runs"
            result = MODULE.build_plan(final_root, base_root, run_root, output)
            self.assertEqual(result["task_count"], 36)
            with (output / "dlpfc_tasks.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            observed = [(int(row["seed"]), row["section"]) for row in rows]
            expected = [
                (seed, section)
                for seed in MODULE.ALL_SEEDS
                for section in MODULE.SECTIONS
            ]
            self.assertEqual(observed, expected)
            self.assertEqual(
                rows[-1]["section_root"],
                str(run_root / f"{base}_seed20260917" / "151676"),
            )

    def test_manifest_parameter_disagreement_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, _, _ = self.build_inputs(
                root, p6_modelscale=1.8
            )
            with self.assertRaisesRegex(ValueError, "different parameters"):
                MODULE.build_plan(
                    final_root, base_root, root / "runs", root / "plan"
                )

    def test_upstream_mutation_invalidates_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, _, _ = self.build_inputs(root)
            output = root / "plan"
            MODULE.build_plan(final_root, base_root, root / "runs", output)
            (final_root / "FINAL_BEAGLE_REPRESENTATION.json").write_text("mutated\n")
            with self.assertRaises(ValueError):
                MODULE.validate_frozen_plan(output)

    def test_noncanonical_section_root_fails_even_with_refreshed_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, _, _ = self.build_inputs(root)
            output = root / "plan"
            MODULE.build_plan(final_root, base_root, root / "runs", output)
            task_path = output / "dlpfc_tasks.tsv"
            rows = task_path.read_text().splitlines()
            fields = rows[1].split("\t")
            fields[-1] = str(root / "escape")
            rows[1] = "\t".join(fields)
            task_path.write_text("\n".join(rows) + "\n")
            provenance_path = output / "PROVENANCE.json"
            ledger = output / "OUTPUT_SHA256SUMS.txt"
            ledger.write_text(
                f"{MODULE.sha256(task_path)}  dlpfc_tasks.tsv\n"
                f"{MODULE.sha256(provenance_path)}  PROVENANCE.json\n"
            )
            completion_path = output / "RUN_COMPLETE.json"
            completion = json.loads(completion_path.read_text())
            completion["output_sha256sums"] = MODULE.identity(ledger)
            write_json(completion_path, completion)
            with self.assertRaisesRegex(ValueError, "frozen grid"):
                MODULE.validate_frozen_plan(output)

    def test_refuses_existing_plan_or_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_root, base_root, _, _ = self.build_inputs(root)
            existing_plan = root / "existing_plan"
            existing_plan.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.build_plan(
                    final_root, base_root, root / "new_runs", existing_plan
                )
            existing_run = root / "existing_runs"
            existing_run.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.build_plan(
                    final_root, base_root, existing_run, root / "new_plan"
                )


if __name__ == "__main__":
    unittest.main()
