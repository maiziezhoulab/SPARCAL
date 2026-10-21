from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/plan_beagle_v2_3_replication.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plan_beagle_v2_3_replication", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_ledger(path: Path, files) -> None:
    path.write_text(
        "".join(f"{MODULE.sha256(source)}  {source.name}\n" for source in files)
    )


def refresh_output_ledger(
    root: Path, gate_name: str, extra_names: tuple[str, ...] = ()
) -> None:
    ledger = root / "OUTPUT_SHA256SUMS.txt"
    write_ledger(
        ledger,
        (
            root / gate_name,
            root / "PROVENANCE.json",
            root / "CODE_AND_INPUT_SHA256SUMS.txt",
            *(root / name for name in extra_names),
        ),
    )
    completion_path = root / "RUN_COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["output_sha256sums"] = MODULE.identity(ledger)
    write_json(completion_path, completion)


def write_selection(
    root: Path,
    *,
    kind: str,
    selected: list[str],
    extension_advance: bool | None = None,
) -> None:
    root.mkdir()
    (root / "RUN_COMPLETE").write_text("complete\n")
    code_ledger = root / "CODE_AND_INPUT_SHA256SUMS.txt"
    code_ledger.write_text(f"{MODULE.sha256(MODULE_PATH)}  {MODULE_PATH.resolve()}\n")

    common = {
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
    }
    if kind == "factorial":
        extra_names: tuple[str, ...] = ()
        gate_name = "screen_gate_results.tsv"
        configs = [
            f"{base}_seed{MODULE.SCREEN_SEED}"
            for base in MODULE.FACTORIAL_CONFIGURATIONS
        ]
        with (root / gate_name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("config", "advance_to_replication"),
                delimiter="\t",
            )
            writer.writeheader()
            for config in configs:
                writer.writerow({
                    "config": config,
                    "advance_to_replication": str(config in selected).lower(),
                })
        provenance = {
            "schema": "sparcal_beagle_v2_3_screen_selection_v1",
            **common,
            "final_default_selected": False,
            "comparator": f"ms2_niter0_seed{MODULE.SCREEN_SEED}",
            "replicate_configurations": selected,
        }
        completion = {
            "schema": "sparcal_beagle_v2_3_screen_selection_completion_v1",
            "status": "complete",
            **common,
            "replicate_configurations": selected,
        }
    elif kind == "extension":
        extra_names = ()
        if extension_advance is None:
            raise AssertionError("extension fixture requires an advancement decision")
        gate_name = "extension_gate_result.tsv"
        extension = f"ms4_niter0_seed{MODULE.SCREEN_SEED}"
        with (root / gate_name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("config", "advance_to_replication"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "config": extension,
                "advance_to_replication": str(extension_advance).lower(),
            })
        provenance = {
            "schema": "sparcal_beagle_v2_3_modelscale_extension_selection_v1",
            **common,
            "final_default_selected": False,
            "extension": extension,
            "comparator": f"ms2_niter0_seed{MODULE.SCREEN_SEED}",
            "advance_to_replication": extension_advance,
        }
        completion = {
            "schema": "sparcal_beagle_v2_3_modelscale_extension_completion_v1",
            "status": "complete",
            **common,
            "replicate_configurations": [extension] if extension_advance else [],
        }
    elif kind == "maxlr":
        extra_names = (
            "matched_configuration_metrics_long.tsv",
            "matched_configuration_summary.tsv",
        )
        for name in extra_names:
            (root / name).write_text(name + "\n")
        if extension_advance is None:
            raise AssertionError("maxlr fixture requires an advancement decision")
        gate_name = "maxlr_gate_result.tsv"
        candidate = f"ms2_niter0_maxlr50000_seed{MODULE.SCREEN_SEED}"
        comparator = f"ms2_niter0_seed{MODULE.SCREEN_SEED}"
        with (root / gate_name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("config", "comparator", "advance_to_replication"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow({
                "config": candidate,
                "comparator": comparator,
                "advance_to_replication": str(extension_advance).lower(),
            })
        replicate = [candidate] if extension_advance else []
        provenance = {
            "schema": "sparcal_beagle_v2_3_maxlr_extension_selection_v1",
            **common,
            "final_default_selected": False,
            "candidate": candidate,
            "comparator": comparator,
            "advance_to_replication": extension_advance,
            "replicate_configurations": replicate,
        }
        completion = {
            "schema": "sparcal_beagle_v2_3_maxlr_extension_completion_v1",
            "status": "complete",
            **common,
            "replicate_configurations": replicate,
        }
    else:
        raise AssertionError(kind)

    write_json(root / "PROVENANCE.json", provenance)
    write_json(root / "RUN_COMPLETE.json", completion)
    refresh_output_ledger(root, gate_name, extra_names)


def write_pre_replication_audit(
    root: Path,
    *,
    extension_advance: bool,
    maxlr_advance: bool,
) -> list[str]:
    audit = MODULE.PREAUDIT
    root.mkdir()
    rows = []
    blocked_factorial = {
        f"ms2_niter5_seed{audit.SEED}",
        f"ms0p8_niter5_seed{audit.SEED}",
    }
    for config in audit.SETTINGS:
        for sample in audit.SAMPLES:
            for metric, numerator, denominator in (
                ("normal_wes_anchor_sensitivity", 96, 100),
                ("selected_anchor_normal_wes_support", 94, 100),
                ("wes_somatic_anchor_leakage", 5, 50),
            ):
                if metric == "wes_somatic_anchor_leakage":
                    if config in blocked_factorial and sample == "P4":
                        numerator = 6
                    elif config == audit.MODELSCALE and extension_advance:
                        numerator = 4
                    elif config == audit.MAXLR and maxlr_advance:
                        numerator = 4
                rows.append({
                    "config": config,
                    "sample": sample,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": numerator / denominator,
                    "ci95_low": 0.0,
                    "ci95_high": 1.0,
                })
    metrics = pd.DataFrame(rows)
    gates = audit.compute_gate_tables(metrics)
    factorial, modelscale, maxlr = gates
    admitted = [
        str(audit.SETTINGS[config]["base"])
        for config in factorial.loc[
            factorial["advance_to_replication"], "config"
        ]
    ]
    for config, gate in (
        (audit.MODELSCALE, modelscale),
        (audit.MAXLR, maxlr),
    ):
        if bool(gate.iloc[0]["advance_to_replication"]):
            admitted.append(str(audit.SETTINGS[config]["base"]))

    metrics.to_csv(
        root / "recomputed_gate_metrics_long.tsv", sep="\t", index=False
    )
    for name, frame in zip(
        (
            "factorial_gate_results_recomputed.tsv",
            "modelscale4_gate_result_recomputed.tsv",
            "maxlr_gate_result_recomputed.tsv",
        ),
        gates,
    ):
        frame.to_csv(root / name, sep="\t", index=False)
    write_json(root / "ADMITTED_CONFIGURATIONS.json", {
        "schema": (
            "sparcal_beagle_v2_3_pre_replication_"
            "admitted_configurations_v1"
        ),
        "status": "screen_decisions_independently_recomputed",
        "screen_seed": audit.SEED,
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "admitted_base_configurations": admitted,
    })
    write_json(root / "PROVENANCE.json", {
        "schema": "sparcal_beagle_v2_3_pre_replication_audit_v1",
        "status": "selector_decisions_verified",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "screen_seed": audit.SEED,
        "samples": list(audit.SAMPLES),
        "expected_configurations": list(audit.SETTINGS),
        "admitted_base_configurations": admitted,
        "audit": audit.identity(Path(audit.__file__)),
    })
    audit.write_ledger(
        [Path(audit.__file__)], root / "CODE_AND_INPUT_SHA256SUMS.txt"
    )
    audit.write_ledger(
        [root / name for name in audit.AUDIT_MEMBER_NAMES],
        root / "OUTPUT_SHA256SUMS.txt",
        root,
    )
    (root / "RUN_COMPLETE").write_text("complete\n")
    write_json(root / "RUN_COMPLETE.json", {
        "schema": (
            "sparcal_beagle_v2_3_pre_replication_audit_completion_v1"
        ),
        "status": "complete",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "screen_seed": audit.SEED,
        "admitted_base_configurations": admitted,
        "admitted_configurations": audit.identity(
            root / "ADMITTED_CONFIGURATIONS.json"
        ),
        "output_sha256sums": audit.identity(root / "OUTPUT_SHA256SUMS.txt"),
    })
    return admitted


class ReplicationPlanTests(unittest.TestCase):
    def build_inputs(
        self,
        root: Path,
        extension_advance=True,
        maxlr_advance=True,
        audit_extension_advance=None,
        audit_maxlr_advance=None,
    ):
        selected = [
            f"ms2_niter0_seed{MODULE.SCREEN_SEED}",
            f"ms0p8_niter0_seed{MODULE.SCREEN_SEED}",
        ]
        factorial = root / "factorial"
        write_selection(factorial, kind="factorial", selected=selected)
        extension = root / "extension"
        write_selection(
            extension,
            kind="extension",
            selected=[],
            extension_advance=extension_advance,
        )
        maxlr = root / "maxlr"
        write_selection(
            maxlr,
            kind="maxlr",
            selected=[],
            extension_advance=maxlr_advance,
        )
        audit = root / "pre_replication_audit"
        write_pre_replication_audit(
            audit,
            extension_advance=(
                extension_advance
                if audit_extension_advance is None
                else audit_extension_advance
            ),
            maxlr_advance=(
                maxlr_advance
                if audit_maxlr_advance is None
                else audit_maxlr_advance
            ),
        )
        return factorial, extension, maxlr, audit

    def test_plan_contains_only_selected_configs_two_seeds_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(root)
            output = root / "plan"
            result = MODULE.build_plan(
                factorial, extension, maxlr, audit, output
            )
            self.assertEqual(
                result["selected_configurations"],
                [
                    "ms2_niter0",
                    "ms0p8_niter0",
                    "ms4_niter0",
                    "ms2_niter0_maxlr50000",
                ],
            )
            self.assertEqual(result["task_count"], 16)
            rows = (output / "replication_tasks.tsv").read_text().splitlines()
            self.assertEqual(len(rows), 17)
            self.assertEqual(
                rows[0],
                "task_id\tconfig\tmodelscale\tniterations\tmaxlr\tseed\t"
                "sample\tsample_root\tdataset",
            )
            maxlr_rows = [row.split("\t") for row in rows[1:] if "maxlr50000" in row]
            self.assertEqual(len(maxlr_rows), 4)
            self.assertEqual({row[4] for row in maxlr_rows}, {"50000.0"})
            self.assertTrue((output / "RUN_COMPLETE.json").is_file())
            provenance = json.loads((output / "PROVENANCE.json").read_text())
            self.assertEqual(
                len(provenance["pre_replication_audit_inputs"]), 10
            )
            self.assertTrue(all(
                item in provenance["selection_inputs"]
                for item in provenance["pre_replication_audit_inputs"]
            ))
            with self.assertRaises(FileExistsError):
                MODULE.build_plan(factorial, extension, maxlr, audit, output)

    def test_extension_is_omitted_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(
                root, extension_advance=False
            )
            result = MODULE.build_plan(
                factorial, extension, maxlr, audit, root / "plan"
            )
            self.assertEqual(
                result["selected_configurations"],
                ["ms2_niter0", "ms0p8_niter0", "ms2_niter0_maxlr50000"],
            )
            self.assertEqual(result["task_count"], 12)

    def test_maxlr_is_omitted_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(
                root, maxlr_advance=False
            )
            result = MODULE.build_plan(
                factorial, extension, maxlr, audit, root / "plan"
            )
            self.assertNotIn(
                "ms2_niter0_maxlr50000", result["selected_configurations"]
            )

    def test_comparator_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(root)
            selected = [f"ms0p8_niter0_seed{MODULE.SCREEN_SEED}"]
            completion_path = factorial / "RUN_COMPLETE.json"
            completion = json.loads(completion_path.read_text())
            completion["replicate_configurations"] = selected
            write_json(completion_path, completion)
            provenance_path = factorial / "PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["replicate_configurations"] = selected
            write_json(provenance_path, provenance)
            with (factorial / "screen_gate_results.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            for row in rows:
                row["advance_to_replication"] = str(row["config"] in selected).lower()
            with (factorial / "screen_gate_results.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            refresh_output_ledger(factorial, "screen_gate_results.tsv")
            with self.assertRaisesRegex(ValueError, "comparator"):
                MODULE.build_plan(
                    factorial, extension, maxlr, audit, root / "plan"
                )

    def test_semantic_gate_mutation_is_rejected_even_when_rechecksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(root)
            gate = factorial / "screen_gate_results.tsv"
            with gate.open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            rows[0]["advance_to_replication"] = "false"
            with gate.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            refresh_output_ledger(factorial, gate.name)
            with self.assertRaisesRegex(ValueError, "decisions disagree"):
                MODULE.build_plan(
                    factorial, extension, maxlr, audit, root / "plan"
                )

    def test_provenance_mutation_is_rejected_even_when_rechecksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(root)
            provenance_path = extension / "PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["advance_to_replication"] = False
            write_json(provenance_path, provenance)
            refresh_output_ledger(extension, "extension_gate_result.tsv")
            with self.assertRaisesRegex(ValueError, "decisions disagree"):
                MODULE.build_plan(
                    factorial, extension, maxlr, audit, root / "plan"
                )

    def test_gate_checksum_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(root)
            with (factorial / "screen_gate_results.tsv").open("a") as handle:
                handle.write("# mutation\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.build_plan(
                    factorial, extension, maxlr, audit, root / "plan"
                )

    def test_audit_checksum_mutation_and_extra_file_are_rejected(self):
        for mutation in ("checksum", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                factorial, extension, maxlr, audit = self.build_inputs(root)
                if mutation == "checksum":
                    with (audit / "factorial_gate_results_recomputed.tsv").open("a") as handle:
                        handle.write("# mutation\n")
                    expected = "checksum mismatch"
                else:
                    (audit / "unexpected.txt").write_text("unexpected\n")
                    expected = "file set is not exact"
                with self.assertRaisesRegex(ValueError, expected):
                    MODULE.build_plan(
                        factorial, extension, maxlr, audit, root / "plan"
                    )

    def test_audit_and_selector_admissions_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factorial, extension, maxlr, audit = self.build_inputs(
                root,
                extension_advance=True,
                audit_extension_advance=False,
            )
            with self.assertRaisesRegex(
                ValueError, "selector admissions differ"
            ):
                MODULE.build_plan(
                    factorial, extension, maxlr, audit, root / "plan"
                )


if __name__ == "__main__":
    unittest.main()
