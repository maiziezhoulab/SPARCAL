import gzip
import importlib.util
import tempfile
import unittest

import pandas as pd

from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/postanalyze/beagle_configuration_benchmark.py"
SPEC = importlib.util.spec_from_file_location("beagle_configuration_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

HEADER = "chrom\tpos\tref\talt\tbeagle_gt\tbeagle_called_gp\troute\n"


def write_audit(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER)
        handle.writelines(rows)


class BeagleConfigurationBenchmarkTests(unittest.TestCase):
    def test_discovers_nested_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / (
                "niter0_seed1/P4_sec1/beagle_evidence_v1_gp090/"
                "beagle_evidence_audit.tsv.gz"
            )
            second = root / (
                "niter0_seed1/P6_sec1/beagle_evidence_v1_gp090/"
                "beagle_evidence_audit.tsv.gz"
            )
            write_audit(first, [])
            write_audit(second, [])
            (root / "niter0_seed1/P4_sec1/RUN_COMPLETE").touch()
            (root / "niter0_seed1/P6_sec1/RUN_COMPLETE.json").write_text("{}\n")
            self.assertEqual(
                MODULE.discover_audits(root),
                {
                    ("niter0_seed1", "P4"): first,
                    ("niter0_seed1", "P6"): second,
                },
            )

    def test_ignores_incomplete_run_and_prefers_explicit_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / (
                "cfg0/P4_sec1/beagle_evidence_v1_gp090/"
                "beagle_evidence_audit.tsv.gz"
            )
            integrated = root / (
                "cfg1/P4_sec1/output_VCFs/beagle/baseQ0mapQ0/"
                "beagle_evidence_audit.tsv.gz"
            )
            explicit = root / (
                "cfg1/P4_sec1/beagle_evidence_v1_gp090/"
                "beagle_evidence_audit.tsv.gz"
            )
            for path in (incomplete, integrated, explicit):
                write_audit(path, [])
            (root / "cfg1/P4_sec1/RUN_COMPLETE").touch()
            self.assertEqual(
                MODULE.discover_audits(root),
                {("cfg1", "P4"): explicit},
            )

    def test_pairwise_anchor_and_route_stability(self):
        key1, key2, key3 = (
            ("chr1", 10, "A", "C"), ("chr1", 20, "G", "T"), ("chr1", 30, "C", "G")
        )
        configs = {
            "niter0_seed1": {
                key1: {"route": "beagle_nonref_highconf", "gt": "0/1", "called_gp": 0.95},
                key2: {"route": "panel_absent", "gt": None, "called_gp": None},
            },
            "niter0_seed2": {
                key1: {"route": "beagle_nonref_highconf", "gt": "1/0", "called_gp": 0.90},
                key2: {"route": "beagle_uncertain", "gt": "0/1", "called_gp": 0.60},
                key3: {"route": "beagle_nonref_highconf", "gt": "1/1", "called_gp": 0.99},
            },
        }
        row = MODULE.pairwise_stability(configs, "P4")[0]
        self.assertTrue(row["same_settings_except_seed"])
        self.assertAlmostEqual(row["raw_jaccard"], 2 / 3)
        self.assertAlmostEqual(row["route_agreement"], 0.5)
        self.assertAlmostEqual(row["anchor_jaccard"], 0.5)
        self.assertEqual(row["anchor_symmetric_difference_n"], 1)
        self.assertEqual(row["gt_concordance"], 1.0)
        self.assertAlmostEqual(row["called_gp_mae"], 0.05)

    def test_pnonref_rerouting_ignores_recorded_called_gp_route(self):
        key = ("chr1", 10, "A", "C")
        evidence = {
            key: {
                "route": "beagle_uncertain",
                "gt": "0/1",
                "called_gp": 0.50,
                "p_nonref": 0.92,
                "p_ref": 0.08,
            }
        }
        rerouted = MODULE.reroute_evidence(evidence, "p_nonref", 0.90)
        self.assertEqual(
            rerouted[key]["route"], "beagle_nonref_highconf"
        )
        self.assertEqual(evidence[key]["route"], "beagle_uncertain")

    def test_pnonref_decimal_boundary_is_inclusive(self):
        key = ("chr1", 20, "G", "T")
        evidence = {
            key: {
                "route": "beagle_uncertain", "gt": "0/1",
                "called_gp": 0.29, "p_nonref": 0.29 + 0.61, "p_ref": 0.10,
            }
        }
        rerouted = MODULE.reroute_evidence(evidence, "p_nonref", 0.90)
        self.assertEqual(rerouted[key]["route"], "beagle_nonref_highconf")

    def test_end_to_end_metrics_penalize_upstream_candidate_loss(self):
        def row(chrom, pos, ref, alt, **extra):
            return {"sample": "P4", "chrom": chrom, "pos": pos,
                    "ref": ref, "alt": alt, **extra}

        normal = pd.DataFrame([
            row("chr1", 10, "A", "C", wes_gt="0/1"),
            row("chr1", 20, "G", "T", wes_gt="0/1"),
        ])
        support = pd.DataFrame(columns=[
            "sample", "chrom", "pos", "ref", "alt",
            "wes_pileup_dp", "high_conf_wes_exact",
        ])
        somatic = pd.DataFrame([
            row("chr1", 30, "C", "G", rna_covered=True),
            row("chr1", 40, "T", "A", rna_covered=True),
        ])
        table_s1 = pd.DataFrame(columns=[
            "sample", "chrom_hg19", "pos_hg19", "ref", "alt",
        ])
        evidence = {
            ("chr1", 10, "A", "C"): {
                "route": "beagle_nonref_highconf", "gt": "0/1",
            },
            ("chr1", 30, "C", "G"): {
                "route": "beagle_uncertain", "gt": "0/1",
            },
        }
        metrics, _ = MODULE.evaluate(
            "cfg", "P4", evidence, normal, support, somatic, table_s1
        )
        values = {item["metric"]: item["value"] for item in metrics}
        self.assertEqual(values["raw_candidate_normal_wes_recall"], 0.5)
        self.assertEqual(values["normal_wes_end_to_end_anchor_recall"], 0.5)
        self.assertEqual(values["normal_wes_anchor_sensitivity"], 1.0)
        self.assertEqual(values["raw_candidate_wes_somatic_recall"], 0.5)
        self.assertEqual(values["wes_somatic_end_to_end_retention"], 0.5)
        self.assertEqual(values["wes_somatic_retention"], 1.0)

    def test_duplicate_config_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_audit(root / "cfg/P4_sec1/a/beagle_evidence_audit.tsv.gz", [])
            write_audit(root / "cfg/P4_sec1/b/beagle_evidence_audit.tsv.gz", [])
            (root / "cfg/P4_sec1/RUN_COMPLETE").touch()
            with self.assertRaises(ValueError):
                MODULE.discover_audits(root)


if __name__ == "__main__":
    unittest.main()
