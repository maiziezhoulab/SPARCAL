import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "scripts/2_beagle_filtering/freeze_beagle_default.py"
SPEC = importlib.util.spec_from_file_location("freeze_beagle_default", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def profile():
    return dict(schema=M.SCHEMA, status="architecture_frozen", representation="single_seed",
        parameters=dict(M.PARAMETERS), anchor_score="p_nonref", min_anchor_probability=0.9,
        quality_filter="baseQ0mapQ0", production_inputs_include_wes=False,
        scientific_final_validation_complete=False, reference_ledger="/frozen/panel")


class FrozenDefaultTests(unittest.TestCase):
    def test_inference_and_scientific_status_cannot_silently_change(self):
        for key, value in [("representation", "ensemble_median"),
                           ("production_inputs_include_wes", True),
                           ("scientific_final_validation_complete", True),
                           ("anchor_score", "called_gp"), ("min_anchor_probability", 0.8)]:
            p = profile()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_profile(p)
        p = profile()
        p["parameters"]["SEED"] = 20260917
        with self.assertRaises(ValueError):
            M.validate_profile(p)

    def test_command_pins_biological_and_execution_settings(self):
        M.validate_profile(profile())
        cmd = M.run_command(profile(), Path("/snapshot"), "P4_TUMOR", "1", Path("/out"))
        pairs = dict(zip(cmd[2::2], cmd[3::2]))
        for key, value in {"--iterations": "0", "--seed": "20260915",
                           "--modelscale": "2.0", "--maxlr": "5000.0",
                           "--threads": "30", "--parallel-chromosomes": "1",
                           "--anchor-score": "p_nonref", "--min-anchor-probability": "0.9"}.items():
            self.assertEqual(pairs[key], value)
        self.assertNotIn("--wes", cmd)

    def test_launch_copies_complete_runtime_inside_run_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, output = root / "bundle", root / "output"
            for name in M.SOURCES:
                target = bundle / "runtime" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("runtime\n")
            (bundle / "FREEZE_COMPLETE.json").write_text("{}\n")
            raw = root / "raw.vcf.gz"
            raw.write_bytes(b"raw input")
            Path(str(raw) + ".tbi").write_bytes(b"index")
            with mock.patch.object(M, "verify", return_value=(profile(), None)), mock.patch.object(M.subprocess, "run") as call:
                M.run(bundle, raw, "P4_TUMOR", "1", output)
                command = call.call_args.args[0]
                self.assertTrue(Path(command[1]).is_file())
                self.assertTrue(str(Path(command[1])).startswith(str(output)))
                for name in M.SOURCES:
                    self.assertTrue((output / "code_snapshot/scripts/2_beagle_filtering" / name).is_file())
                with self.assertRaises(FileExistsError):
                    M.run(bundle, raw, "P4_TUMOR", "1", output)
            self.assertEqual(raw.read_bytes(), b"raw input")
            self.assertFalse(json.loads((output / "FROZEN_PROFILE.json").read_text())["scientific_final_validation_complete"])

    def make_partition(self, root):
        raw = root / "raw.vcf.gz"
        output = root / "routes"
        output.mkdir()
        rows = [["chr1", str(i + 1), ".", "A", "C", "60", "PASS", "DP=8", "GT:DP", "0/1:8"] for i in range(4)]
        with gzip.open(raw, "wt") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.writelines("\t".join(row) + "\n" for row in rows)
        columns = ["chrom", "pos", "ref", "alt", "route", "hard_anchor"]
        with gzip.open(output / "beagle_evidence_audit.tsv.gz", "wt") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(columns)
            for row, route in zip(rows, M.ROUTES):
                writer.writerow([row[0], row[1], row[3], row[4], route, int(route == M.ROUTES[0])])
        for name, accepted in M.VIEWS.items():
            with gzip.open(output / f"all_{name}.vcf.gz", "wt") as f:
                for row, route in zip(rows, M.ROUTES):
                    if route in accepted:
                        copy = list(row)
                        copy[7] += ";BEAGLE_ROUTE=" + route
                        f.write("\t".join(copy) + "\n")
        return raw, output

    def test_all_four_states_and_eight_views_conserve_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = self.make_partition(Path(tmp))
            result = M.verify_partition(raw, out)
            self.assertEqual(result["raw_candidates"], 4)
            self.assertEqual(result["views_checked"], 8)
            self.assertEqual(result["route_counts"], dict.fromkeys(M.ROUTES, 1))

    def test_lost_allele_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = self.make_partition(Path(tmp))
            with gzip.open(out / "all_panel_present.vcf.gz", "wt") as f:
                f.write("")
            with self.assertRaises(ValueError):
                M.verify_partition(raw, out)

    def test_extra_allele_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = self.make_partition(Path(tmp))
            p = out / "all_beagle_ref_highconf.vcf.gz"
            with gzip.open(p, "rt") as f:
                content = f.read()
            with gzip.open(p, "wt") as f:
                f.write(content + content)
            with self.assertRaises(ValueError):
                M.verify_partition(raw, out)

    def test_changed_genotype_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = self.make_partition(Path(tmp))
            p = out / "all_filtered_in.vcf.gz"
            with gzip.open(p, "rt") as f:
                content = f.read()
            with gzip.open(p, "wt") as f:
                f.write(content.replace("0/1:8", "1/1:8"))
            with self.assertRaises(ValueError):
                M.verify_partition(raw, out)

    def test_raw_audit_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = self.make_partition(Path(tmp))
            with gzip.open(raw, "at") as f:
                f.write("chr1\t9\t.\tG\tT\t60\tPASS\tDP=8\tGT:DP\t0/1:8\n")
            with self.assertRaises(ValueError):
                M.verify_partition(raw, out)

    def test_modified_bundle_is_rejected_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required = ["PROFILE.json", "beagle.py", "DECISION.md", "PROVENANCE.json"]
            required += ["runtime/" + name for name in M.SOURCES]
            for name in required:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            ledger = root / "SHA256SUMS.txt"
            ledger.write_text("".join(f"{M.digest(root / name)}  {name}\n" for name in required))
            M.write_json(root / "FREEZE_COMPLETE.json", dict(schema=M.SCHEMA, status="architecture_frozen",
                                                           ledger_sha256=M.digest(ledger)))
            (root / "runtime/beagle_evidence_router.py").write_text("changed\n")
            with mock.patch.object(M, "load") as load, self.assertRaises(ValueError):
                M.verify(root)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
