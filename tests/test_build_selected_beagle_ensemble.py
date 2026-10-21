import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/build_selected_beagle_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_selected_beagle_ensemble", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


OUTPUT_NAMES = (
    "replicated_configuration_metrics_long.tsv",
    "replicated_configuration_summary.tsv",
    "replicated_pairwise_stability.tsv",
    "replication_gate_results.tsv",
    "FINAL_BASE_CONFIGURATION.json",
    "PROVENANCE.json",
    "CODE_AND_INPUT_SHA256SUMS.txt",
)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def refresh_output_ledger(root: Path) -> None:
    ledger = root / "OUTPUT_SHA256SUMS.txt"
    ledger.write_text(
        "".join(
            f"{MODULE.sha256(root / name)}  {name}\n" for name in OUTPUT_NAMES
        )
    )
    completion_path = root / "RUN_COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["output_sha256sums"] = MODULE.identity(ledger)
    write_json(completion_path, completion)


def build_assessment(root: Path, omit=None):
    root.mkdir()
    base = "ms2_niter0"
    selected = [base]
    validated = []
    audits = {}
    for sample in MODULE.SAMPLES:
        for seed in MODULE.SEEDS:
            config = f"{base}_seed{seed}"
            run_root = root.parent / "runs" / config / sample
            run_root.mkdir(parents=True)
            files = {}
            for field, name in (
                ("manifest", "V2_3_MANIFEST.json"),
                ("completion", "RUN_COMPLETE.json"),
                ("audit", "beagle_evidence_audit.tsv.gz"),
                ("runtime_metrics", "beagle_contig_metrics.tsv"),
            ):
                path = run_root / name
                path.write_bytes(f"{field}/{config}/{sample}\n".encode())
                files[field] = path
            audits[(sample, seed)] = files["audit"]
            if omit != (sample, seed):
                validated.append({
                    "base_config": base,
                    "config": config,
                    "sample": sample,
                    "run_root": str(run_root.resolve()),
                    **{
                        field: MODULE.identity(path)
                        for field, path in files.items()
                    },
                })

    (root / "replicated_configuration_metrics_long.tsv").write_text(
        "config\tsample\tmetric\tnumerator\tdenominator\tvalue\n"
        f"{base}_seed{MODULE.SEEDS[0]}\tP4\tdummy\t1\t1\t1\n"
    )
    (root / "replicated_configuration_summary.tsv").write_text(
        "config\tsample\tbeagle_runtime_seconds\n"
        f"{base}_seed{MODULE.SEEDS[0]}\tP4\t1\n"
    )
    (root / "replicated_pairwise_stability.tsv").write_text(
        "sample\tconfig_a\tconfig_b\tanchor_jaccard\n"
    )
    with (root / "replication_gate_results.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "base_config",
                "eligible_as_base_default",
                "selected_base_default",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({
            "base_config": base,
            "eligible_as_base_default": "true",
            "selected_base_default": "true",
        })
    selection = {
        "schema": "sparcal_beagle_v2_3_final_base_selection_v1",
        "selected_base_configuration": base,
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "final_ensemble_selected": False,
    }
    provenance = {
        "schema": "sparcal_beagle_v2_3_replication_assessment_v1",
        "selected_base_configuration": base,
        "selected_configurations": selected,
        "seeds": list(MODULE.SEEDS),
        "samples": list(MODULE.SAMPLES),
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "validated_runs": validated,
    }
    write_json(root / "FINAL_BASE_CONFIGURATION.json", selection)
    write_json(root / "PROVENANCE.json", provenance)
    (root / "RUN_COMPLETE").write_text("complete\n")
    code_ledger = root / "CODE_AND_INPUT_SHA256SUMS.txt"
    code_ledger.write_text(f"{MODULE.sha256(MODULE_PATH)}  {MODULE_PATH.resolve()}\n")
    write_json(
        root / "RUN_COMPLETE.json",
        {
            "schema": "sparcal_beagle_v2_3_replication_completion_v1",
            "selected_base_configuration": base,
            "status": "complete",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": True,
        },
    )
    refresh_output_ledger(root)
    return audits


class SelectedEnsembleBuilderTests(unittest.TestCase):
    def test_loads_exact_selected_three_seed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assessment = root / "assessment"
            audits = build_assessment(assessment)
            base, sources, _ = MODULE.load_selected_sources(assessment)
            self.assertEqual(base, "ms2_niter0")
            for sample in MODULE.SAMPLES:
                self.assertEqual(set(sources[sample]), set(MODULE.SEEDS))
                for seed in MODULE.SEEDS:
                    self.assertEqual(sources[sample][seed], audits[(sample, seed)])

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            build_assessment(assessment, omit=("P6", 20260917))
            with self.assertRaisesRegex(ValueError, "validated-run grid"):
                MODULE.load_selected_sources(assessment)

    def test_mutated_source_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            audits = build_assessment(assessment)
            audits[("P4", 20260916)].write_bytes(b"mutated\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                MODULE.load_selected_sources(assessment)

    def test_gate_selection_mutation_is_rejected_when_rechecksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            build_assessment(assessment)
            gate = assessment / "replication_gate_results.tsv"
            with gate.open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            rows[0]["selected_base_default"] = "false"
            with gate.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            refresh_output_ledger(assessment)
            with self.assertRaisesRegex(ValueError, "gate/selection/provenance"):
                MODULE.load_selected_sources(assessment)

    def test_selection_mutation_is_rejected_when_rechecksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            build_assessment(assessment)
            selection_path = assessment / "FINAL_BASE_CONFIGURATION.json"
            selection = json.loads(selection_path.read_text())
            selection["selected_base_configuration"] = "ms0p8_niter0"
            write_json(selection_path, selection)
            refresh_output_ledger(assessment)
            with self.assertRaisesRegex(ValueError, "selection/status contract"):
                MODULE.load_selected_sources(assessment)

    def test_provenance_grid_mutation_is_rejected_when_rechecksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            build_assessment(assessment)
            provenance_path = assessment / "PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["seeds"] = list(MODULE.SEEDS[:-1])
            write_json(provenance_path, provenance)
            refresh_output_ledger(assessment)
            with self.assertRaisesRegex(ValueError, "selection grid"):
                MODULE.load_selected_sources(assessment)

    def test_metric_file_mutation_is_rejected_by_output_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            assessment = Path(tmp) / "assessment"
            build_assessment(assessment)
            with (assessment / "replicated_configuration_metrics_long.tsv").open(
                "a"
            ) as handle:
                handle.write("mutated\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.load_selected_sources(assessment)


if __name__ == "__main__":
    unittest.main()
