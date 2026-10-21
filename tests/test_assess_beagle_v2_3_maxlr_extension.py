from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_v2_3_maxlr_extension.py"
)
SPEC = importlib.util.spec_from_file_location(
    "assess_beagle_v2_3_maxlr_extension", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metrics(*, sensitivity=965, support=945, p4_leakage=4, p6_leakage=1):
    rows = []
    for config in MODULE.SETTINGS:
        candidate = config == MODULE.CANDIDATE
        for sample in MODULE.SAMPLES:
            values = {
                "normal_wes_anchor_sensitivity": (
                    sensitivity if candidate else 960,
                    1000,
                ),
                "selected_anchor_normal_wes_support": (
                    support if candidate else 940,
                    1000,
                ),
                "wes_somatic_anchor_leakage": (
                    (p4_leakage if sample == "P4" else p6_leakage)
                    if candidate
                    else (4 if sample == "P4" else 1),
                    34 if sample == "P4" else 22,
                ),
            }
            for metric, (numerator, denominator) in values.items():
                rows.append({
                    "config": config,
                    "sample": sample,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": numerator / denominator,
                })
    return pd.DataFrame(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_prescreen(root: Path) -> None:
    root.mkdir()
    dependencies = []
    for name in (
        "assessor.py",
        "contracts.py",
        "protocol.md",
        "comparator_manifest.json",
        "comparator_completion.json",
        "comparator_audit.tsv.gz",
        "candidate_manifest.json",
        "candidate_completion.json",
        "candidate_audit.tsv.gz",
    ):
        path = root.parent / name
        path.write_text(name + "\n")
        dependencies.append(path)
    code_ledger = root / "CODE_AND_INPUT_SHA256SUMS.txt"
    code_ledger.write_text(
        "".join(f"{MODULE.sha256(path)}  {path.resolve()}\n" for path in dependencies)
    )
    (root / "maxlr_prescreen_summary.tsv").write_text(
        "advance_to_full_genome\ntrue\n"
    )
    provenance = {
        "schema": "sparcal_beagle_maxlr_prescreen_v1",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "advance_to_full_genome": True,
        "decision": "advance_to_frozen_full_genome_comparison",
        "assessor": MODULE.identity(dependencies[0]),
        "contracts": MODULE.identity(dependencies[1]),
        "protocol": MODULE.identity(dependencies[2]),
        "inputs": {
            "comparator": {
                "manifest": MODULE.identity(dependencies[3]),
                "completion": MODULE.identity(dependencies[4]),
                "audit": MODULE.identity(dependencies[5]),
            },
            "candidate": {
                "manifest": MODULE.identity(dependencies[6]),
                "completion": MODULE.identity(dependencies[7]),
                "audit": MODULE.identity(dependencies[8]),
            },
        },
    }
    write_json(root / "PROVENANCE.json", provenance)
    output_ledger = root / "OUTPUT_SHA256SUMS.txt"
    output_ledger.write_text(
        "".join(
            f"{MODULE.sha256(root / name)}  {name}\n"
            for name in (
                "maxlr_prescreen_summary.tsv",
                "PROVENANCE.json",
                "CODE_AND_INPUT_SHA256SUMS.txt",
            )
        )
    )
    (root / "RUN_COMPLETE").write_text("complete\n")
    write_json(
        root / "RUN_COMPLETE.json",
        {
            "schema": "sparcal_beagle_maxlr_prescreen_completion_v1",
            "status": "complete",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "advance_to_full_genome": True,
            "decision": "advance_to_frozen_full_genome_comparison",
            "output_sha256sums": MODULE.identity(output_ledger),
        },
    )


class MaxlrExtensionTests(unittest.TestCase):
    def test_candidate_advances_with_no_harm_and_material_gain(self):
        result = MODULE.assess(metrics())
        self.assertTrue(result["advance_to_replication"])
        self.assertTrue(result["material_improvement"])

    def test_any_sample_leakage_increase_rejects(self):
        result = MODULE.assess(metrics(p4_leakage=5))
        self.assertFalse(result["leakage_nonincrease_both"])
        self.assertFalse(result["advance_to_replication"])

    def test_no_material_gain_rejects(self):
        result = MODULE.assess(metrics(sensitivity=960, support=940))
        self.assertFalse(result["material_improvement"])
        self.assertFalse(result["advance_to_replication"])

    def test_incomplete_metric_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            MODULE.assess(metrics().iloc[:-1])

    def test_prescreen_output_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prescreen"
            build_prescreen(root)
            self.assertEqual(len(MODULE.validate_prescreen(root)), 6)
            with (root / "maxlr_prescreen_summary.tsv").open("a") as handle:
                handle.write("mutated\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.validate_prescreen(root)


if __name__ == "__main__":
    unittest.main()
