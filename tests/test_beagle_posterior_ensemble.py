import csv
import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/beagle_posterior_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beagle_posterior_ensemble", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(
    *,
    chrom="chr1",
    pos="10",
    raw_gt="0/1",
    gp="0.11,0.89,0",
    route="beagle_uncertain",
):
    values = {
        "schema": "sparcal_beagle_evidence_v2_2",
        "chrom": chrom,
        "pos": pos,
        "ref": "A",
        "alt": "C",
        "raw_gt": raw_gt,
        "beagle_gt": "0/1",
        "beagle_gp": gp,
        "beagle_called_gp": "0.89",
        "beagle_p_nonref": "0.89",
        "beagle_gp_entropy": "0.1",
        "beagle_gp_margin": "0.78",
        "beagle_af": "0.2",
        "gt_transition": f"{raw_gt}>0/1",
        "route": route,
        "hard_anchor": "0",
    }
    return values


def write_audit(path: Path, values) -> None:
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MODULE.AUDIT_COLUMNS, delimiter="\t"
        )
        writer.writeheader()
        writer.writerow(values)


class BeaglePosteriorEnsembleTests(unittest.TestCase):
    def test_fixed_mean_and_median_can_differ_at_boundary(self):
        rows = {
            20260915: row(gp="0.11,0.89,0"),
            20260916: row(gp="0.11,0.89,0"),
            20260917: row(
                gp="0.01,0.99,0", route="beagle_nonref_highconf"
            ),
        }
        mean_fields, mean_route = MODULE.summarize_row(
            rows, "mean", MODULE.SEEDS
        )
        median_fields, median_route = MODULE.summarize_row(
            rows, "median", MODULE.SEEDS
        )
        mean = dict(zip(MODULE.AUDIT_COLUMNS, mean_fields))
        median = dict(zip(MODULE.AUDIT_COLUMNS, median_fields))
        self.assertEqual(mean_route, "beagle_nonref_highconf")
        self.assertEqual(median_route, "beagle_uncertain")
        self.assertAlmostEqual(float(mean["beagle_p_nonref"]), 0.923333333333)
        self.assertAlmostEqual(float(median["beagle_p_nonref"]), 0.89)

    def test_panel_absent_is_preserved(self):
        rows = {
            seed: row(gp=".", route="panel_absent")
            for seed in MODULE.SEEDS
        }
        fields, route = MODULE.summarize_row(rows, "mean", MODULE.SEEDS)
        output = dict(zip(MODULE.AUDIT_COLUMNS, fields))
        self.assertEqual(route, "panel_absent")
        self.assertEqual(output["beagle_gp"], ".")
        self.assertEqual(output["hard_anchor"], "0")

    def test_median_routes_direct_pnonref_not_sum_of_component_medians(self):
        rows = {
            20260915: row(gp="0.1,0.9,0"),
            20260916: row(gp="0.1,0,0.9"),
            20260917: row(gp="0.2,0.4,0.4"),
        }
        fields, route = MODULE.summarize_row(rows, "median", MODULE.SEEDS)
        output = dict(zip(MODULE.AUDIT_COLUMNS, fields))
        displayed_gp = [
            float(value) for value in output["beagle_gp"].split(",")
        ]
        self.assertAlmostEqual(sum(displayed_gp[1:]), 0.8)
        self.assertAlmostEqual(float(output["beagle_p_nonref"]), 0.9)
        self.assertEqual(route, "beagle_nonref_highconf")

    def test_any_invalid_included_gp_routes_conservatively(self):
        rows = {
            20260915: row(gp="0.01,0.99,0", route="beagle_nonref_highconf"),
            20260916: row(gp="bad"),
            20260917: row(gp="0.01,0.99,0", route="beagle_nonref_highconf"),
        }
        fields, route = MODULE.summarize_row(rows, "mean", MODULE.SEEDS)
        output = dict(zip(MODULE.AUDIT_COLUMNS, fields))
        self.assertEqual(route, "beagle_uncertain")
        self.assertEqual(output["beagle_gp"], ".")
        self.assertEqual(output["beagle_gt"], ".")

    def test_key_or_panel_presence_mismatch_is_rejected(self):
        key_mismatch = {seed: row() for seed in MODULE.SEEDS}
        key_mismatch[20260917] = row(pos="11")
        with self.assertRaisesRegex(ValueError, "allele order/content"):
            MODULE.summarize_row(key_mismatch, "mean", MODULE.SEEDS)

        presence_mismatch = {seed: row() for seed in MODULE.SEEDS}
        presence_mismatch[20260917] = row(gp=".", route="panel_absent")
        with self.assertRaisesRegex(ValueError, "panel presence"):
            MODULE.summarize_row(presence_mismatch, "mean", MODULE.SEEDS)

    def test_parser_requires_exact_frozen_seed_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for seed in MODULE.SEEDS:
                path = root / f"{seed}.tsv.gz"
                write_audit(path, row())
                paths[seed] = path
            parsed = MODULE.parse_seed_audits(
                [f"{seed}={paths[seed]}" for seed in reversed(MODULE.SEEDS)]
            )
            self.assertEqual(set(parsed), set(MODULE.SEEDS))
            with self.assertRaisesRegex(ValueError, "exactly the frozen seeds"):
                MODULE.parse_seed_audits(
                    [f"{seed}={paths[seed]}" for seed in MODULE.SEEDS[:2]]
                )

    def test_mutated_aggregation_audit_is_rejected_by_full_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_rows = {
                20260915: row(gp="0.11,0.89,0"),
                20260916: row(gp="0.11,0.89,0"),
                20260917: row(
                    gp="0.01,0.99,0", route="beagle_nonref_highconf"
                ),
            }
            source_audits = {}
            for seed, source_row in source_rows.items():
                path = root / f"source_{seed}.tsv.gz"
                write_audit(path, source_row)
                source_audits[seed] = path

            ensemble_audits = {}
            for method, (aggregation, included_seeds) in MODULE.METHODS.items():
                fields, _ = MODULE.summarize_row(
                    source_rows, aggregation, included_seeds
                )
                path = root / f"{method}.tsv.gz"
                write_audit(path, dict(zip(MODULE.AUDIT_COLUMNS, fields)))
                ensemble_audits[method] = path

            verified = MODULE.validate_ensemble_audits(
                audits=ensemble_audits,
                source_audits=source_audits,
            )
            self.assertEqual(verified["ensemble_mean"]["raw_record_count"], 1)

            mutated = dict(
                zip(
                    MODULE.AUDIT_COLUMNS,
                    MODULE.summarize_row(
                        source_rows, "mean", MODULE.SEEDS
                    )[0],
                )
            )
            mutated["beagle_p_nonref"] = "0.5"
            write_audit(ensemble_audits["ensemble_mean"], mutated)
            with self.assertRaisesRegex(ValueError, "prescribed ensemble_mean"):
                MODULE.validate_ensemble_audits(
                    audits=ensemble_audits,
                    source_audits=source_audits,
                )

    def test_build_writes_completed_discoverable_bundles_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audits = {}
            for seed in MODULE.SEEDS:
                path = root / f"source_{seed}.tsv.gz"
                value = row(
                    gp=(
                        "0.01,0.99,0"
                        if seed == MODULE.SEEDS[-1]
                        else "0.11,0.89,0"
                    ),
                    route=(
                        "beagle_nonref_highconf"
                        if seed == MODULE.SEEDS[-1]
                        else "beagle_uncertain"
                    ),
                )
                write_audit(path, value)
                audits[seed] = path

            contracts_path = (
                Path(__file__).resolve().parents[1]
                / "scripts/2_beagle_filtering/"
                "beagle_v2_3_candidate_2026_09_13/contracts.py"
            )
            protocol_path = (
                Path(__file__).resolve().parents[1]
                / "scripts/2_beagle_filtering/"
                "beagle_v2_3_candidate_2026_09_13/"
                "BENCHMARK_PROTOCOL_ENSEMBLE.md"
            )
            manifests = {
                seed: {"beagle_parameters": {"SEED": seed}}
                for seed in MODULE.SEEDS
            }
            roots = {seed: root / f"run_{seed}" for seed in MODULE.SEEDS}
            output = root / "ensemble"
            with mock.patch.object(
                MODULE,
                "validate_sources",
                return_value=(manifests, roots),
            ):
                result = MODULE.build(
                    audits=audits,
                    output_root=output,
                    sample="P4",
                    contracts_path=contracts_path,
                    protocol_path=protocol_path,
                )
                self.assertEqual(result["status"], "complete")
                for name in MODULE.METHODS:
                    sample_root = output / name / "P4_sec1"
                    self.assertTrue((sample_root / "RUN_COMPLETE.json").is_file())
                    self.assertTrue((sample_root / "OUTPUT_SHA256SUMS.txt").is_file())
                mean_audit = (
                    output
                    / "ensemble_mean/P4_sec1/beagle_evidence_v2_3_ensemble/"
                    "beagle_evidence_audit.tsv.gz"
                )
                with gzip.open(mean_audit, "rt") as handle:
                    output_rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertEqual(
                    output_rows[0]["route"], "beagle_nonref_highconf"
                )
                with self.assertRaises(FileExistsError):
                    MODULE.build(
                        audits=audits,
                        output_root=output,
                        sample="P4",
                        contracts_path=contracts_path,
                        protocol_path=protocol_path,
                    )


if __name__ == "__main__":
    unittest.main()
