from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "scripts/postanalyze/materialize_beagle_v2_3_final_dlpfc.py"
ENSEMBLE_PATH = PROJECT / "scripts/postanalyze/beagle_posterior_ensemble.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = load(MODULE_PATH, "materialize_beagle_v2_3_final_dlpfc")
ENSEMBLE = load(ENSEMBLE_PATH, "materialize_test_ensemble")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit_row(
    *, chrom="chr1", pos="10", ref="A", alt="C",
    gp="0.05,0.95,0", route="beagle_nonref_highconf",
):
    p_nonref = "." if gp == "." else str(sum(float(x) for x in gp.split(",")[1:]))
    anchor = route == "beagle_nonref_highconf"
    return {
        "schema": "sparcal_beagle_evidence_v2_2",
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "raw_gt": "0/1",
        "beagle_gt": "." if gp == "." else "0/1",
        "beagle_gp": gp,
        "beagle_called_gp": "." if gp == "." else "0.95",
        "beagle_p_nonref": p_nonref,
        "beagle_gp_entropy": "." if gp == "." else "0.2",
        "beagle_gp_margin": "." if gp == "." else "0.9",
        "beagle_af": "." if gp == "." else "0.1",
        "gt_transition": "0/1>0/1",
        "route": route,
        "hard_anchor": "1" if anchor else "0",
    }


def write_audit(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENSEMBLE.AUDIT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class FinalDlpfcMaterializerTests(unittest.TestCase):
    def make_plan(self, root: Path, representation: str):
        plan = root / "plan"
        run_root = root / "runs"
        final_root = root / "final"
        base_root = root / "base"
        plan.mkdir()
        final_root.mkdir()
        base_root.mkdir()
        final_names = (
            "ensemble_gate_results.tsv", "FINAL_BEAGLE_REPRESENTATION.json",
            "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
            "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE", "RUN_COMPLETE.json",
        )
        base_names = (
            "replicated_configuration_metrics_long.tsv",
            "replicated_configuration_summary.tsv",
            "replicated_pairwise_stability.tsv", "replication_gate_results.tsv",
            "FINAL_BASE_CONFIGURATION.json", "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt", "OUTPUT_SHA256SUMS.txt",
            "RUN_COMPLETE", "RUN_COMPLETE.json",
        )
        for directory, names in ((final_root, final_names), (base_root, base_names)):
            for name in names:
                (directory / name).write_text(f"{name}\n")
        source_manifests = []
        for sample in ("P4", "P6"):
            path = root / f"{sample}_V2_3_MANIFEST.json"
            path.write_text(f"{sample}\n")
            source_manifests.append({"sample": sample, "manifest": MODULE.identity(path)})
        protocol = (
            PROJECT / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
            / "BENCHMARK_PROTOCOL_DLPFC_FINAL.md"
        )
        params = {
            "THREADS": 30, "MEMORY": "10g", "MODEL_SCALE": 2.0,
            "ITERATIONS": 0, "IMPUTE": False, "GPROBS": True,
            "SEED": 20260915, "PARALLEL_CHROMOSOMES": 1,
            "MAX_LR": 5000.0, "GENETIC_MAP": None,
            "REFERENCE_LEDGER": str((root / "reference").resolve()),
        }
        seeds = (
            MODULE.SINGLE_SEEDS if representation == "single_seed"
            else MODULE.ENSEMBLE_SEEDS
        )
        rows = []
        for seed in seeds:
            for section in MODULE.SECTIONS:
                rows.append({
                    "task_id": len(rows), "section": section, "seed": seed,
                    "dataset": "DLPFC", "modelscale": 2.0,
                    "niterations": 0, "maxlr": 5000.0,
                    "base_config": "opaque", "representation": representation,
                    "section_root": str(run_root / f"opaque_seed{seed}" / section),
                })
        with (plan / "dlpfc_tasks.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MODULE.PLAN_HEADER, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        shared = {
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "base_config": "opaque", "representation": representation,
            "seeds": list(seeds), "sections": list(MODULE.SECTIONS),
            "task_count": len(rows), "array_max": len(rows) - 1,
            "final_selection_inputs": [MODULE.identity(final_root / name) for name in final_names],
            "base_assessment_inputs": [MODULE.identity(base_root / name) for name in base_names],
            "selected_seed20260915_manifests": source_manifests,
        }
        provenance = {
            "schema": MODULE.PLAN_SCHEMA, "status": "tasks_frozen", **shared,
            "dataset": "DLPFC", "quality_filter": MODULE.QUALITY_FILTER,
            "anchor_score": "p_nonref", "min_anchor_probability": 0.90,
            "run_root": str(run_root.resolve()),
            "final_selection_root": str(final_root.resolve()),
            "base_assessment_root": str(base_root.resolve()),
            "beagle_parameters": params,
            "reference_ledger_root": params["REFERENCE_LEDGER"],
            "planner": MODULE.identity(PROJECT / "scripts/postanalyze/plan_beagle_v2_3_final_dlpfc.py"),
            "integrity_module": MODULE.identity(PROJECT / "scripts/postanalyze/beagle_selection_integrity.py"),
            "protocol": MODULE.identity(protocol),
            "launcher": MODULE.identity(MODULE.CANONICAL_LAUNCHER),
            "execution_code": [MODULE.identity(path) for path in MODULE.EXECUTION_CODE],
        }
        write_json(plan / "PROVENANCE.json", provenance)
        ledger = plan / "OUTPUT_SHA256SUMS.txt"
        ledger.write_text(
            f"{MODULE.sha256(plan / 'dlpfc_tasks.tsv')}  dlpfc_tasks.tsv\n"
            f"{MODULE.sha256(plan / 'PROVENANCE.json')}  PROVENANCE.json\n"
        )
        (plan / "RUN_COMPLETE").write_text("complete\n")
        completion = {
            "schema": MODULE.PLAN_COMPLETION_SCHEMA, "status": "complete",
            **shared, "output_sha256sums": MODULE.identity(ledger),
        }
        write_json(plan / "RUN_COMPLETE.json", completion)
        return plan, run_root, protocol

    def refresh_plan_ledger(self, plan: Path) -> None:
        ledger = plan / "OUTPUT_SHA256SUMS.txt"
        ledger.write_text(
            f"{MODULE.sha256(plan / 'dlpfc_tasks.tsv')}  dlpfc_tasks.tsv\n"
            f"{MODULE.sha256(plan / 'PROVENANCE.json')}  PROVENANCE.json\n"
        )
        completion_path = plan / "RUN_COMPLETE.json"
        completion = json.loads(completion_path.read_text())
        completion["output_sha256sums"] = MODULE.identity(ledger)
        write_json(completion_path, completion)

    def test_plan_requires_exact_single_or_ensemble_grid(self):
        for representation, expected in (("single_seed", 12), ("ensemble_median", 36)):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as tmp:
                plan, run_root, protocol = self.make_plan(Path(tmp), representation)
                verified = MODULE.verify_plan(plan, run_root, protocol)
                self.assertEqual(len(verified["tasks"]), expected)
                self.assertEqual(tuple(verified["seeds"]), (
                    MODULE.SINGLE_SEEDS if expected == 12 else MODULE.ENSEMBLE_SEEDS
                ))

    def test_plan_rejects_noncanonical_root_even_after_rechecksumming(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, run_root, protocol = self.make_plan(Path(tmp), "single_seed")
            lines = (plan / "dlpfc_tasks.tsv").read_text().splitlines()
            fields = lines[1].split("\t")
            fields[-1] = str(Path(tmp) / "escape")
            lines[1] = "\t".join(fields)
            (plan / "dlpfc_tasks.tsv").write_text("\n".join(lines) + "\n")
            self.refresh_plan_ledger(plan)
            with self.assertRaisesRegex(ValueError, "section_root is not canonical"):
                MODULE.verify_plan(plan, run_root, protocol)

    def test_plan_rejects_routing_score_drift_even_after_rechecksumming(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, run_root, protocol = self.make_plan(Path(tmp), "single_seed")
            provenance_path = plan / "PROVENANCE.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["anchor_score"] = "called_gp"
            write_json(provenance_path, provenance)
            self.refresh_plan_ledger(plan)
            with self.assertRaisesRegex(ValueError, "WES/dataset/run-root contract"):
                MODULE.verify_plan(plan, run_root, protocol)

    def test_run_snapshot_execution_code_drift_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planned = []
            observed = []
            for name in ("run_beagle.py", "beagle_evidence_router.py", "contracts.py", "run_beagle_v2_3.py"):
                source = root / "planned" / name
                snapshot = root / "run_snapshot" / name
                source.parent.mkdir(exist_ok=True)
                snapshot.parent.mkdir(exist_ok=True)
                source.write_text(f"{name}\n")
                snapshot.write_text(f"{name}\n")
                planned.append(MODULE.identity(source))
                observed.append(MODULE.identity(snapshot))
            MODULE.require_execution_code_match(observed, planned)
            changed = root / "run_snapshot" / "contracts.py"
            changed.write_text("mutated execution code\n")
            observed[2] = MODULE.identity(changed)
            with self.assertRaisesRegex(ValueError, "content drift"):
                MODULE.require_execution_code_match(observed, planned)

    def test_multiallelic_any_projection_preserves_order_dtype_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            frame = pd.DataFrame(
                [[1, 0, 2], [0, 1, 0]],
                index=pd.Index(["spot_b", "spot_a"], name="barcode"),
                columns=pd.Index(["1_10", "1_20", "1_30"], name="locus"),
                dtype="int8",
            )
            frame.to_pickle(source)
            audit = root / "audit.tsv.gz"
            write_audit(audit, [
                audit_row(pos="10", alt="C"),
                audit_row(pos="10", alt="G", gp=".", route="panel_absent"),
                audit_row(pos="20", gp="0.95,0.05,0", route="beagle_ref_highconf"),
                audit_row(pos="30", gp=".", route="panel_absent"),
            ])
            primary = root / "out" / "primary.pkl"
            anchor = root / "out" / "anchor.pkl"
            differences = root / "differences.tsv.gz"
            result = MODULE.build_matrix_views(
                source, audit, primary, anchor, differences, ENSEMBLE.AUDIT_COLUMNS
            )
            primary_frame = pd.read_pickle(primary)
            anchor_frame = pd.read_pickle(anchor)
            self.assertEqual(list(primary_frame.columns), ["1_10", "1_20"])
            self.assertEqual(list(anchor_frame.columns), ["1_10"])
            self.assertTrue(primary_frame.index.equals(frame.index))
            self.assertTrue(primary_frame.dtypes.equals(frame.iloc[:, :2].dtypes))
            pd.testing.assert_frame_equal(primary_frame, frame.iloc[:, :2], check_exact=True)
            self.assertEqual(result["multiallelic_policy"], "any")
            self.assertEqual(result["panel_mixed_multiallelic_locus_count"], 1)
            self.assertEqual(result["primary_candidate_only_feature_count"], 0)
            self.assertEqual(result["primary_legacy_only_feature_count"], 1)
            self.assertFalse(result["primary_exact_dataframe_equal_to_legacy"])

    def test_candidate_panel_position_absent_from_source_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            pd.DataFrame([[1]], columns=["1_10"], dtype="int8").to_pickle(source)
            audit = root / "audit.tsv.gz"
            write_audit(audit, [audit_row(pos="99")])
            with self.assertRaisesRegex(ValueError, "cannot be represented"):
                MODULE.build_matrix_views(
                    source, audit, root / "out" / "primary.pkl",
                    root / "out" / "anchor.pkl", root / "diff.tsv.gz",
                    ENSEMBLE.AUDIT_COLUMNS,
                )
            self.assertFalse((root / "out").exists())

    def test_empty_secondary_anchor_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            frame = pd.DataFrame([[1], [0]], columns=["1_10"], dtype="int8")
            frame.to_pickle(source)
            audit = root / "audit.tsv.gz"
            write_audit(audit, [
                audit_row(pos="10", gp="0.95,0.05,0", route="beagle_ref_highconf")
            ])
            primary = root / "out" / "primary.pkl"
            anchor = root / "out" / "anchor.pkl"
            result = MODULE.build_matrix_views(
                source, audit, primary, anchor, root / "diff.tsv.gz",
                ENSEMBLE.AUDIT_COLUMNS,
            )
            pd.testing.assert_frame_equal(pd.read_pickle(primary), frame, check_exact=True)
            anchor_frame = pd.read_pickle(anchor)
            self.assertEqual(anchor_frame.shape, (2, 0))
            self.assertTrue(anchor_frame.index.equals(frame.index))
            self.assertEqual(result["anchor_status"], "empty_no_pnonref_gp090_loci")

    def test_selected_mean_and_median_are_streamed_without_loo_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_audits = {}
            gps = ("0.11,0.89,0", "0.11,0.89,0", "0.01,0.99,0")
            for seed, gp in zip(MODULE.ENSEMBLE_SEEDS, gps):
                path = root / f"{seed}.tsv.gz"
                route = "beagle_nonref_highconf" if gp.startswith("0.01") else "beagle_uncertain"
                write_audit(path, [audit_row(gp=gp, route=route)])
                source_audits[seed] = path
            mean_path = root / "mean" / "audit.tsv.gz"
            mean = MODULE.stream_selected_ensemble(
                source_audits, mean_path, "ensemble_mean", ENSEMBLE
            )
            self.assertEqual(mean["row_count"], 1)
            with gzip.open(mean_path, "rt") as handle:
                mean_row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(mean_row["route"], "beagle_nonref_highconf")
            self.assertEqual(list((root / "mean").iterdir()), [mean_path])

            median_path = root / "median" / "audit.tsv.gz"
            MODULE.stream_selected_ensemble(
                source_audits, median_path, "ensemble_median", ENSEMBLE
            )
            with gzip.open(median_path, "rt") as handle:
                median_row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(median_row["route"], "beagle_uncertain")

    def test_ensemble_rejects_allele_or_panel_status_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audits = {}
            for seed in MODULE.ENSEMBLE_SEEDS:
                path = root / f"{seed}.tsv.gz"
                value = (
                    audit_row(pos="11") if seed == 20260917 else audit_row()
                )
                write_audit(path, [value])
                audits[seed] = path
            with self.assertRaisesRegex(ValueError, "allele order/content"):
                MODULE.stream_selected_ensemble(
                    audits, root / "output" / "audit.tsv.gz",
                    "ensemble_mean", ENSEMBLE,
                )

    def test_materializer_refuses_existing_output_before_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                MODULE.materialize(
                    plan_root=root / "missing_plan", run_root=root / "runs",
                    output_root=output, project=root,
                    contracts_path=root / "contracts.py",
                    ensemble_path=root / "ensemble.py",
                    protocol_path=root / "protocol.md",
                )


if __name__ == "__main__":
    unittest.main()
