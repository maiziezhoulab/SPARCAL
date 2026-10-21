import hashlib
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/plot_beagle_module_supplement.py"
)
SPEC = importlib.util.spec_from_file_location("plot_beagle_module_supplement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest(path),
    }


def write_ledger(root: Path, members: set[str], name: str) -> None:
    lines = []
    for member in sorted(members):
        path = root / member
        assert path.is_file(), path
        lines.append(f"{digest(path)}  {member}\n")
    (root / name).write_text("".join(lines))


def fraction(numerator: int, denominator: int = 100) -> dict:
    value = numerator / denominator
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "ci95_low": max(0.0, value - 0.02),
        "ci95_high": min(1.0, value + 0.02),
    }


def build_monopogen(root: Path) -> None:
    root.mkdir()
    (root / "code_snapshot").mkdir()
    (root / "code_snapshot/monopogen_beagle_contract_audit.py").write_text("# frozen\n")
    (root / "code_snapshot/test_monopogen_beagle_contract_audit.py").write_text("# frozen\n")
    method_names = [
        "monopogen_downstream_beagle_record_membership",
        "monopogen_paper_raw_beagle_gt_concordant_nonref",
        "sparcal_derived_pnonref_threshold",
        "diagnostic_final_phased_nonref_gt",
    ]
    count_fields = {
        method_names[0]: ("monopogen_downstream_beagle_records", 100),
        method_names[1]: ("monopogen_paper_raw_beagle_gt_concordant_nonref", 90),
        method_names[2]: ("sparcal_derived_pnonref_threshold_count", 85),
        method_names[3]: ("diagnostic_final_phased_nonref_gt", 85),
    }
    summary = []
    method_sets = []
    for run in MODULE.MONO_RUNS:
        row = {
            "run": run, "gp_records": 100, "phased_records": 100,
            "gp_phased_record_sets_equal": True,
            "gp_keys_absent_from_phased": 0, "phased_keys_absent_from_gp": 0,
            "gp_phased_comparable": 100, "gp_phased_unordered_gt_changes": 0,
            "phased_heterozygous": 40, "phased_0_pipe_1": 19,
            "phased_1_pipe_0": 21,
        }
        for method in method_names:
            field, count = count_fields[method]
            row[field] = count
            method_sets.append({
                "run": run, "method": method, "method_role": "synthetic_test",
                "count": count, "definition": "synthetic immutable contract row",
            })
        summary.append(row)
    pd.DataFrame(summary).to_csv(root / "run_summary.tsv", sep="\t", index=False)
    pd.DataFrame(method_sets).to_csv(root / "method_sets.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"run": run, "left_method": method_names[0], "right_method": method_names[1],
         "intersection_count": 90, "jaccard": 0.9}
        for run in MODULE.MONO_RUNS
    ]).to_csv(root / "method_intersections.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"run": run, "raw_pl_gt": "0/1", "beagle_gp_gt": "0/1", "count": 90}
        for run in MODULE.MONO_RUNS
    ]).to_csv(root / "transition_counts.tsv", sep="\t", index=False)
    manifest = []
    for index in range(264):
        manifest.append({
            "run": MODULE.MONO_RUNS[index % 4], "chromosome": index % 22 + 1,
            "kind": "vcf", "path": f"/synthetic/input/{index}.vcf.gz",
            "size_bytes": 10, "mtime_ns": index + 1, "sha256": "0" * 64,
        })
    pd.DataFrame(manifest).to_csv(root / "input_manifest.tsv", sep="\t", index=False)
    dump_json(root / "PROVENANCE.json", {
        "audit_schema_version": "3.0", "wes_used": False,
        "monopogen_public_commit": MODULE.MONO_COMMIT, "pnonref_threshold": 0.90,
        "scope_limit": (
            "This audit does not claim that every phased record is a final "
            "Monopogen somatic call"
        ),
    })
    (root / "RUN_COMPLETE").write_text(
        "Monopogen Beagle contract audit completed successfully at "
        "2026-09-13T00:00:00+00:00.\n"
    )
    write_ledger(root, MODULE.MONO_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS.txt")


def build_evidence(root: Path, mono: Path) -> None:
    root.mkdir()
    for member in MODULE.EVIDENCE_LEDGER_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("synthetic\n")
    external = [
        ("Monopogen released-code phased-record membership component", "external_monopogen_execution", "released_downstream_beagle_component"),
        ("Monopogen final-phased nonref-GT diagnostic", "external_monopogen_execution", "diagnostic_only"),
        ("Monopogen paper concordant nonref", "external_monopogen_execution", "paper_described_concordance_rule"),
        ("Monopogen P(nonref)>=0.90 (derived)", "external_monopogen_execution", "sparcal_derived_rule_not_native_monopogen"),
    ]
    roles = [
        (MODULE.EVIDENCE_METHODS[0], "same_sparcal_input", "historical_broad_panel_anchor"),
        (MODULE.EVIDENCE_METHODS[1], "same_sparcal_input", "paper_specified_legacy_rule"),
        (MODULE.EVIDENCE_METHODS[2], "same_sparcal_input", "ablation"),
        (MODULE.EVIDENCE_METHODS[3], "same_sparcal_input", "production_candidate"),
        *external,
    ]
    pd.DataFrame([
        {"method": method, "comparison_scope": scope, "role": role,
         "definition": "synthetic immutable contract row"}
        for method, scope, role in roles
    ]).to_csv(root / "METHOD_DEFINITIONS.tsv", sep="\t", index=False)
    metric_rows = []
    count_rows = []
    for sample_i, sample in enumerate(MODULE.SAMPLES):
        for method_i, method in enumerate(MODULE.EVIDENCE_METHODS):
            count_rows.append({
                "sample": sample, "method": method,
                "anchor_count": 1000 + sample_i * 10 + method_i,
                "comparison_scope": "same_sparcal_input",
            })
            for metric_i, metric in enumerate(MODULE.EVIDENCE_ALL_METRICS):
                metric_rows.append({
                    "sample": sample, "method": method, "metric": metric,
                    **fraction(70 + sample_i + method_i + metric_i),
                    "comparison_scope": "same_sparcal_input",
                })
    metrics = pd.DataFrame(metric_rows)
    counts = pd.DataFrame(count_rows)
    metrics.to_csv(root / "same_input_method_metrics.tsv", sep="\t", index=False)
    counts.to_csv(root / "same_input_anchor_counts.tsv", sep="\t", index=False)
    metric_lookup = metrics.set_index(["sample", "method", "metric"])
    count_lookup = counts.set_index(["sample", "method"])
    sweep = []
    for sample in MODULE.SAMPLES:
        for source, method in (
            ("SPARCAL Beagle called-GP", MODULE.EVIDENCE_METHODS[2]),
            ("SPARCAL Beagle P(nonref)", MODULE.EVIDENCE_METHODS[3]),
        ):
            sweep.append({
                "sample": sample, "source": source, "threshold": 0.90,
                "anchor_count": int(count_lookup.loc[(sample, method), "anchor_count"]),
                "normal_wes_anchor_sensitivity": float(metric_lookup.loc[(sample, method, "normal_wes_anchor_sensitivity"), "value"]),
                "normal_wes_gt_accuracy": float(metric_lookup.loc[(sample, method, "normal_wes_anchor_gt_accuracy"), "value"]),
                "wes_somatic_leakage": 1 - float(metric_lookup.loc[(sample, method, "wes_somatic_retention"), "value"]),
                "comparison_scope": "same_sparcal_input",
            })
    pd.DataFrame(sweep).to_csv(root / "gp_threshold_sweep.tsv", sep="\t", index=False)
    mono_sets = pd.read_csv(mono / "method_sets.tsv", sep="\t")
    binding_counts = [
        {"run": str(row.run), "method": str(row.method), "count": int(row.count)}
        for row in mono_sets.itertuples(index=False)
    ]
    dump_json(root / "PROVENANCE.json", {
        "benchmark_schema_version": "11.0",
        "production_inputs_include_wes": False, "evaluation_uses_wes": True,
        "matched_wes_role": "internal development evidence, not untouched held-out evaluation",
        "code_snapshot_created_before_computation": True,
        "monopogen_audit_validation": {
            "path": str(mono.resolve()), "audit_schema_version": "3.0",
            "public_commit": MODULE.MONO_COMMIT,
            "input_manifest_sha256": digest(mono / "input_manifest.tsv"),
            "output_manifest_sha256": digest(mono / "OUTPUT_SHA256SUMS.txt"),
            "full_run_method_counts": binding_counts,
        },
    })
    dump_json(root / "VALIDATION.json", {
        "benchmark_schema_version": "11.0", "status": "all_checks_passed",
    })
    dump_json(root / "RUN_COMPLETE", {
        "benchmark_schema_version": "11.0", "status": "complete",
    })
    write_ledger(root, MODULE.EVIDENCE_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS.txt")


def build_quality(root: Path) -> None:
    root.mkdir()
    rows = []
    summary = []
    for config_i, config in enumerate(MODULE.QUALITY_CONFIGS):
        for sample_i, sample in enumerate(MODULE.SAMPLES):
            row = {"config": config, "sample": sample}
            for metric_i, metric in enumerate(MODULE.QUALITY_PLOT_METRICS):
                values = fraction(82 + config_i + sample_i + metric_i)
                rows.append({"config": config, "sample": sample, "metric": metric, **values})
                row[metric] = values["value"]
            summary.append(row)
    pd.DataFrame(rows).to_csv(root / "configuration_metrics_long.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(root / "configuration_summary.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"sample": sample, "config_a": MODULE.QUALITY_CONFIGS[0],
         "config_b": MODULE.QUALITY_CONFIGS[1], "anchor_jaccard": 0.98}
        for sample in MODULE.SAMPLES
    ]).to_csv(root / "pairwise_stability.tsv", sep="\t", index=False)
    dump_json(root / "PROVENANCE.json", {
        "production_inputs_include_wes": False, "evaluation_uses_wes": True,
        "anchor_score": "p_nonref", "min_anchor_probability": 0.90,
    })
    (root / "RUN_COMPLETE").write_text("complete\n")


def build_replication(root: Path) -> tuple[str, tuple[int, ...]]:
    root.mkdir()
    base = "ne2.0_niter17"
    seeds = (101, 202, 303)
    configs = tuple(f"{base}_seed{seed}" for seed in seeds)
    metrics = []
    for config in configs:
        for sample in MODULE.SAMPLES:
            for metric, numerator in (
                ("normal_wes_anchor_sensitivity", 90),
                ("selected_anchor_normal_wes_support", 80),
                ("wes_somatic_anchor_leakage", 2),
            ):
                metrics.append({"config": config, "sample": sample, "metric": metric, **fraction(numerator)})
    pd.DataFrame(metrics).to_csv(root / "replicated_configuration_metrics_long.tsv", sep="\t", index=False)
    pd.DataFrame([{"config": config, "sample": sample} for config in configs for sample in MODULE.SAMPLES]).to_csv(
        root / "replicated_configuration_summary.tsv", sep="\t", index=False,
    )
    pairwise = []
    for sample in MODULE.SAMPLES:
        for index, (left, right) in enumerate(itertools.combinations(configs, 2)):
            pairwise.append({
                "sample": sample, "config_a": left, "config_b": right,
                "anchor_jaccard": (0.97, 0.98, 0.99)[index],
                "route_agreement": (0.96, 0.97, 0.98)[index],
                "p_nonref_mae": (0.01, 0.02, 0.03)[index],
            })
    pd.DataFrame(pairwise).to_csv(root / "replicated_pairwise_stability.tsv", sep="\t", index=False)
    gate = {
        "base_config": base, "comparator": False,
        "mean_normal_wes_anchor_sensitivity": 0.9,
        "delta_mean_sensitivity_vs_comparator": 0.0,
        "replicated_sensitivity_no_harm": True,
        "mean_selected_anchor_normal_wes_support": 0.8,
        "delta_mean_support_vs_comparator": 0.0,
        "replicated_support_no_harm": True,
        "total_wes_somatic_anchor_leakage": 12,
        "total_leakage_reduction_vs_comparator": 0,
        "replicated_material_improvement": False,
        "minimum_anchor_jaccard": 0.97, "minimum_anchor_jaccard_pass": True,
        "minimum_route_agreement": 0.96, "minimum_route_agreement_pass": True,
        "median_anchor_jaccard": 0.98, "median_anchor_jaccard_vs_comparator_pass": True,
        "median_p_nonref_mae": 0.02, "median_p_nonref_mae_vs_comparator_pass": True,
        "matched_seed_sample_leakage_nonincrease": True,
        "mean_beagle_runtime_seconds": 12.0, "simplicity_rank": 1,
        "eligible_as_base_default": True, "selected_base_default": True,
    }
    pd.DataFrame([gate]).to_csv(root / "replication_gate_results.tsv", sep="\t", index=False)
    dump_json(root / "FINAL_BASE_CONFIGURATION.json", {
        "schema": "sparcal_beagle_v2_3_final_base_selection_v1",
        "selected_base_configuration": base,
        "production_inputs_include_wes": False, "evaluation_uses_wes": True,
        "final_ensemble_selected": False,
    })
    dump_json(root / "PROVENANCE.json", {
        "schema": "sparcal_beagle_v2_3_replication_assessment_v1",
        "status": "base_configuration_selected", "production_inputs_include_wes": False,
        "evaluation_uses_wes": True, "development_set_evaluation": True,
        "samples": list(MODULE.SAMPLES), "selected_base_configuration": base,
        "selected_configurations": [base], "seeds": list(seeds),
    })
    (root / "CODE_AND_INPUT_SHA256SUMS.txt").write_text("synthetic inputs\n")
    (root / "RUN_COMPLETE").write_text("complete\n")
    write_ledger(root, MODULE.REPLICATION_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS.txt")
    dump_json(root / "RUN_COMPLETE.json", {
        "schema": "sparcal_beagle_v2_3_replication_completion_v1",
        "status": "complete", "production_inputs_include_wes": False,
        "evaluation_uses_wes": True, "selected_base_configuration": base,
        "output_sha256sums": identity(root / "OUTPUT_SHA256SUMS.txt"),
    })
    return base, seeds


def build_ensemble(root: Path, base: str, seeds: tuple[int, ...]) -> None:
    root.mkdir()
    rows = []
    for method_i, method in enumerate(MODULE.ENSEMBLE_METHODS):
        rows.append({
            "method": method, "single_seed_comparator": f"{base}_seed{seeds[0]}",
            "mean_normal_wes_anchor_sensitivity": 0.9,
            "delta_mean_sensitivity": -0.001 * (method_i + 1),
            "sensitivity_no_harm": True,
            "mean_selected_anchor_normal_wes_support": 0.8,
            "delta_mean_support": -0.001 * (method_i + 1),
            "support_no_harm": True, "P4_somatic_leakage_n": 1,
            "P6_somatic_leakage_n": 1, "total_somatic_leakage_n": 2,
            "leakage_nonincrease_both": True, "total_leakage_reduction": 0,
            "material_improvement": False,
            "minimum_leave_one_out_anchor_jaccard": 0.99,
            "leave_one_out_stability_pass": True, "beagle_run_multiplier": 3,
            "advance_over_single_seed": False, "selected_final_representation": False,
        })
    pd.DataFrame(rows).to_csv(root / "ensemble_gate_results.tsv", sep="\t", index=False)
    dump_json(root / "FINAL_BEAGLE_REPRESENTATION.json", {
        "schema": "sparcal_beagle_v2_3_final_representation_v1",
        "base_configuration": base, "selected_representation": "single_seed",
        "selected_seed": seeds[0], "beagle_run_multiplier": 1,
        "production_inputs_include_wes": False, "evaluation_uses_wes": True,
    })
    dump_json(root / "PROVENANCE.json", {
        "schema": "sparcal_beagle_v2_3_ensemble_selection_v1",
        "status": "final_representation_selected", "production_inputs_include_wes": False,
        "evaluation_uses_wes": True, "development_set_evaluation": True,
        "base_configuration": base, "selected_representation": "single_seed",
        "frozen_seeds": list(seeds),
    })
    (root / "CODE_AND_INPUT_SHA256SUMS.txt").write_text("synthetic inputs\n")
    (root / "RUN_COMPLETE").write_text("complete\n")
    write_ledger(root, MODULE.ENSEMBLE_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS.txt")
    dump_json(root / "RUN_COMPLETE.json", {
        "schema": "sparcal_beagle_v2_3_ensemble_selection_completion_v1",
        "status": "complete", "production_inputs_include_wes": False,
        "evaluation_uses_wes": True, "base_configuration": base,
        "selected_representation": "single_seed",
        "output_sha256sums": identity(root / "OUTPUT_SHA256SUMS.txt"),
    })


def build_dlpfc(root: Path) -> None:
    run_root = root.parent / "strict_run"
    run_root.mkdir()
    semantic = {
        "method": MODULE.STRICT_METHOD,
        "materializer_schema": "sparcal_beagle_v2_3_final_dlpfc_materialization_v1",
        "scientific_role": "final all_panel_present feature-support compatibility",
        "selected_routes": ["beagle_nonref_highconf", "beagle_ref_highconf", "beagle_uncertain"],
        "multiallelic_policy": "any", "all_candidate_panel_positions_representable": True,
        "complete_column_subset_of_baseline": True,
        "exact_dataframe_equal_to_baseline_required": False,
        "production_inputs_include_wes": False, "evaluation_uses_wes": False,
    }
    dump_json(run_root / "STRICT_INPUT_MANIFEST.json", {
        "schema_version": "1.0", "status": "prepared",
        "protocol_version": MODULE.STRICT_PROTOCOL,
        "sections": list(MODULE.SECTIONS),
        "methods": ["legacy_any_panel", MODULE.STRICT_METHOD],
        "runs_per_section": 10, "wes_used_as_input": False,
        "mandatory_execution_even_if_exactly_equal": True,
        "candidate_semantic_contract": semantic,
    })
    manifest_sha = digest(run_root / "STRICT_INPUT_MANIFEST.json")
    root.mkdir()
    section_rows = []
    seed_rows = []
    for index, section in enumerate(MODULE.SECTIONS):
        delta_ari = 0.001 + index * 0.0001
        delta_nmi = 0.0005 + index * 0.0001
        section_rows.append({
            "section": section, "delta_ari": delta_ari, "delta_nmi": delta_nmi,
            "donor": MODULE.DONOR_BY_SECTION[section],
        })
        for run in range(10):
            seed_rows.append({
                "section": section, "run": run,
                "delta_ari": delta_ari, "delta_nmi": delta_nmi,
            })
    sections = pd.DataFrame(section_rows)
    sections.to_csv(root / "section_paired_deltas.tsv", sep="\t", index=False)
    pd.DataFrame(seed_rows).to_csv(root / "seed_paired_deltas.tsv", sep="\t", index=False)
    donors = sections.groupby("donor", sort=False)[["delta_ari", "delta_nmi"]].mean().reset_index()
    donors.to_csv(root / "donor_paired_deltas.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"section": section, "legacy_any_panel": 10, MODULE.STRICT_METHOD: 10}
        for section in MODULE.SECTIONS
    ]).to_csv(root / "method_success_counts.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"section": MODULE.SECTIONS[0], "method": "legacy_any_panel", "run": 0,
         "status": "ok", "ari": 0.5, "nmi": 0.5},
        {"section": MODULE.SECTIONS[0], "method": MODULE.STRICT_METHOD, "run": 0,
         "status": "ok", "ari": 0.501, "nmi": 0.5005},
    ]).to_csv(root / "clustering_metrics_long.tsv", sep="\t", index=False)
    gates = {gate: True for gate in MODULE.STRICT_GATES}
    result = {
        "schema_version": "3.0", "protocol_version": MODULE.STRICT_PROTOCOL,
        "baseline_method": "legacy_any_panel", "candidate_method": MODULE.STRICT_METHOD,
        "mandatory_execution_even_if_inputs_exactly_equal": True,
        "expected_section_count": 12, "expected_runs_per_section": 10,
        "expected_paired_run_count": 120,
        "interpretation_scope": "functional no-harm; not SNV-caller accuracy",
        "candidate_only_failures": [], "baseline_root": str(run_root.resolve()),
        "candidate_root": str(run_root.resolve()),
        "strict_input_manifest_sha256": manifest_sha,
        "gates": gates, "accept_replacement": True,
        "mean_delta_ari": float(sections["delta_ari"].mean()),
        "mean_delta_nmi": float(sections["delta_nmi"].mean()),
        "section_bootstrap_delta_ari_ci95_descriptive": [
            float(sections["delta_ari"].mean()) - 0.0005,
            float(sections["delta_ari"].mean()) + 0.0005,
        ],
        "donor_mean_delta_ari_range_descriptive": [
            float(donors["delta_ari"].min()), float(donors["delta_ari"].max()),
        ],
        "minimum_donor_mean_delta_ari": float(donors["delta_ari"].min()),
        "baseline_successful_runs": 120, "candidate_successful_runs": 120,
    }
    dump_json(root / "NONINFERIORITY_RESULT.json", result)
    dump_json(root / "STRICT_RUN_VALIDATION.json", {
        "status": "all_checks_passed", "protocol_version": MODULE.STRICT_PROTOCOL,
        "validated_sections": list(MODULE.SECTIONS), "candidate_only_failures": [],
        "mandatory_execution_even_if_exactly_equal": True,
        "root": str(run_root.resolve()), "strict_input_manifest_sha256": manifest_sha,
    })
    write_ledger(root, MODULE.STRICT_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS")
    (root / "ACCEPTED").write_text("all registered final-DLPFC gates passed\n")
    (root / "ASSESSMENT_COMPLETE").write_text("complete\n")


def refresh_dlpfc_binding(root: Path) -> None:
    run_root = root.parent / "strict_run"
    manifest_sha = digest(run_root / "STRICT_INPUT_MANIFEST.json")
    for name in ("NONINFERIORITY_RESULT.json", "STRICT_RUN_VALIDATION.json"):
        path = root / name
        payload = json.loads(path.read_text())
        payload["strict_input_manifest_sha256"] = manifest_sha
        dump_json(path, payload)
    write_ledger(root, MODULE.STRICT_LEDGER_MEMBERS, "OUTPUT_SHA256SUMS")


class BeagleSupplementFigureTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> dict[str, Path]:
        paths = {name: root / name for name in (
            "evidence", "replication", "ensemble", "quality", "dlpfc", "monopogen",
        )}
        build_monopogen(paths["monopogen"])
        build_evidence(paths["evidence"], paths["monopogen"])
        build_quality(paths["quality"])
        base, seeds = build_replication(paths["replication"])
        build_ensemble(paths["ensemble"], base, seeds)
        build_dlpfc(paths["dlpfc"])
        return paths

    @staticmethod
    def load(paths: dict[str, Path]):
        return MODULE.load_inputs(
            paths["evidence"], paths["replication"], paths["ensemble"],
            paths["quality"], paths["dlpfc"], paths["monopogen"],
        )

    def test_six_dossier_contract_and_machine_readable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_inputs(Path(tmp))
            inputs = self.load(paths)
            self.assertEqual(inputs.replication.seeds, (101, 202, 303))
            plotted = MODULE.build_plotted_data(inputs)
            self.assertEqual(set(plotted["panel"]), set("ABCDEF"))
            self.assertEqual(len(plotted.query("panel == 'A'")), 24)
            self.assertEqual(len(plotted.query("panel == 'E' and element == 'registered_gate'")), 8)
            self.assertEqual(len(plotted.query("panel == 'E' and element == 'donor_mean'")), 6)
            self.assertEqual(len(plotted.query("panel == 'E' and element == 'registered_interval'")), 1)
            self.assertEqual(len(plotted.query("panel == 'F'")), 16)
            legacy = plotted.loc[plotted["method"].eq(MODULE.EVIDENCE_METHODS[1])]
            self.assertFalse(legacy.empty)
            self.assertIn("raw GT = Beagle GT", legacy.iloc[0]["method_label"])
            dlpfc = plotted.query("panel == 'E'")
            self.assertEqual(set(dlpfc["method"].dropna()), {MODULE.STRICT_METHOD})
            self.assertTrue(dlpfc["note"].str.contains("not SNV-caller accuracy").any())
            fig, axes = MODULE.compose_figure(inputs)
            self.assertEqual(
                [axis.get_title(loc="left") for axis in axes.ravel()],
                list("abcdef"),
            )
            self.assertIsNone(fig._suptitle)
            self.assertEqual(fig.texts, [])
            MODULE.plt.close(fig)

    def test_checksum_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_inputs(Path(tmp))
            with (paths["monopogen"] / "run_summary.tsv").open("a") as handle:
                handle.write("tamper\n")
            with self.assertRaisesRegex(ValueError, "checksum-mismatched"):
                self.load(paths)

    def test_wrong_dlpfc_scientific_role_is_rejected_even_when_rebound(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_inputs(Path(tmp))
            manifest_path = Path(tmp) / "strict_run/STRICT_INPUT_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["candidate_semantic_contract"]["scientific_role"] = "germline"
            dump_json(manifest_path, manifest)
            refresh_dlpfc_binding(paths["dlpfc"])
            with self.assertRaisesRegex(ValueError, "all_panel_present semantic mismatch"):
                self.load(paths)

    def test_bundle_is_deterministic_hashed_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_inputs(root)
            kwargs = {
                "evidence_dir": paths["evidence"],
                "replication_dir": paths["replication"],
                "ensemble_dir": paths["ensemble"], "quality_dir": paths["quality"],
                "dlpfc_dir": paths["dlpfc"], "monopogen_dir": paths["monopogen"],
                "font_family": "DejaVu Sans",
            }
            first, second = root / "bundle1", root / "bundle2"
            MODULE.generate_bundle(**kwargs, output_dir=first)
            MODULE.generate_bundle(**kwargs, output_dir=second)
            expected = {
                "beagle_v2_3_supplement.pdf", "beagle_v2_3_supplement.png",
                "plotted_data.tsv", "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
                "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE",
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected)
            for name in ("beagle_v2_3_supplement.pdf", "beagle_v2_3_supplement.png", "plotted_data.tsv", "PROVENANCE.json"):
                self.assertEqual(digest(first / name), digest(second / name), name)
            provenance = json.loads((first / "PROVENANCE.json").read_text())
            self.assertFalse(provenance["production_inputs_include_wes"])
            self.assertEqual(provenance["dlpfc_primary_interface"], "final all_panel_present feature-support compatibility")
            self.assertEqual(provenance["font"]["resolved_family"], "DejaVu Sans")
            self.assertGreaterEqual(len(provenance["font"]["faces"]), 1)
            ledger_members = {
                line.split("  ", 1)[1] for line in
                (first / "OUTPUT_SHA256SUMS.txt").read_text().splitlines()
            }
            self.assertEqual(ledger_members, {
                "beagle_v2_3_supplement.pdf", "beagle_v2_3_supplement.png",
                "plotted_data.tsv", "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
            })
            with self.assertRaises(FileExistsError):
                MODULE.generate_bundle(**kwargs, output_dir=first)

    def test_project_arial_gate_loads_all_required_faces(self):
        if not MODULE.DEFAULT_ARIAL_DIR.is_dir():
            self.skipTest("licensed project Arial directory is unavailable")
        family, paths = MODULE.resolve_font("Arial", None)
        self.assertEqual(family, "Arial")
        styles = {MODULE.FT2Font(str(path)).style_name for path in paths}
        self.assertTrue(MODULE.REQUIRED_ARIAL_STYLES.issubset(styles))
        self.assertEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()
