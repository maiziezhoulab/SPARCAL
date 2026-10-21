import csv
import gzip
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = load(
    ROOT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
    "materialize_ensemble_routes.py",
    "_test_materialize_beagle_ensemble_routes",
)
ROUTER = load(
    ROOT / "scripts/2_beagle_filtering/beagle_evidence_router.py",
    "_test_materialize_beagle_ensemble_router",
)
ENSEMBLE = load(
    ROOT / "scripts/postanalyze/beagle_posterior_ensemble.py",
    "_test_materialize_beagle_ensemble_aggregation",
)
CONTRACTS = load(
    ROOT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
    "contracts.py",
    "_test_materialize_beagle_ensemble_contracts",
)


RECORDS = (
    ("chr1", "100", "A", "G", "0/1:7,3"),
    ("chr1", "200", "C", "T", "1/1:0,9"),
    ("chr1", "300", "G", "A", "1/0:4,5"),
    ("chr1", "400", "T", "C", "0/0:8,0"),
)


def write_raw(path: Path, records=RECORDS, info="DP=10") -> None:
    with path.open("w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##contig=<ID=chr1>\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
        )
        for chrom, pos, ref, alt, sample in records:
            handle.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT:AD\t{sample}\n"
            )


def posterior_row(record, gp, p_nonref, route, af="0.2") -> dict[str, str]:
    chrom, pos, ref, alt, sample = record
    raw_gt = ROUTER.normalize_gt(sample.split(":", 1)[0]) or "."
    gt_index = max(range(3), key=lambda index: (gp[index], -index))
    gt = ("0/0", "0/1", "1/1")[gt_index]
    total = sum(gp)
    normalized = tuple(value / total for value in gp)
    entropy = -sum(
        value * math.log(value) for value in normalized if value > 0
    ) / math.log(3)
    ordered = sorted(gp, reverse=True)
    return {
        "schema": ENSEMBLE.AUDIT_SCHEMA,
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "raw_gt": raw_gt,
        "beagle_gt": gt,
        "beagle_gp": ",".join(ENSEMBLE.number(value) for value in gp),
        "beagle_called_gp": ENSEMBLE.number(gp[gt_index]),
        "beagle_p_nonref": ENSEMBLE.number(p_nonref),
        "beagle_gp_entropy": ENSEMBLE.number(entropy),
        "beagle_gp_margin": ENSEMBLE.number(ordered[0] - ordered[1]),
        "beagle_af": af,
        "gt_transition": f"{raw_gt}>{gt}",
        "route": route,
        "hard_anchor": "1" if route == "beagle_nonref_highconf" else "0",
    }


def default_rows() -> list[dict[str, str]]:
    rows = [
        # Directly aggregated median P(nonref)=0.90 while the componentwise
        # displayed GP01+GP11 is 0.89. The former must decide the route.
        posterior_row(
            RECORDS[0], (0.11, 0.44, 0.45), 0.90,
            "beagle_nonref_highconf",
        ),
        posterior_row(
            RECORDS[1], (0.95, 0.03, 0.02), 0.05,
            "beagle_ref_highconf",
        ),
        posterior_row(
            RECORDS[2], (0.40, 0.30, 0.30), 0.60,
            "beagle_uncertain",
        ),
    ]
    chrom, pos, ref, alt, sample = RECORDS[3]
    raw_gt = ROUTER.normalize_gt(sample.split(":", 1)[0]) or "."
    rows.append({
        "schema": ENSEMBLE.AUDIT_SCHEMA,
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "raw_gt": raw_gt,
        "beagle_gt": ".",
        "beagle_gp": ".",
        "beagle_called_gp": ".",
        "beagle_p_nonref": ".",
        "beagle_gp_entropy": ".",
        "beagle_gp_margin": ".",
        "beagle_af": ".",
        "gt_transition": f"{raw_gt}>.",
        "route": "panel_absent",
        "hard_anchor": "0",
    })
    return rows


def write_audit(path: Path, rows) -> None:
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENSEMBLE.AUDIT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def data_lines(path: Path) -> list[str]:
    with ROUTER.open_text(path) as handle:
        return [line for line in handle if not line.startswith("#")]


class EnsembleRouteMaterializerTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        raw = root / "raw.vcf"
        audit = root / "ensemble.tsv.gz"
        write_raw(raw)
        write_audit(audit, default_rows())
        return raw, audit

    def test_alignment_conserves_all_four_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, audit = self.make_inputs(Path(tmp))
            expected = {
                "beagle_nonref_highconf": 1,
                "beagle_ref_highconf": 1,
                "beagle_uncertain": 1,
                "panel_absent": 1,
                "total": 4,
            }
            counts, total = MATERIALIZER.validate_alignment(
                raw,
                audit,
                router=ROUTER,
                ensemble=ENSEMBLE,
                expected_counts=expected,
            )
            self.assertEqual(total, 4)
            self.assertEqual(counts, {key: expected[key] for key in ROUTER.ROUTES})

    def test_writes_exact_routes_and_compatibility_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self.make_inputs(root)
            output = root / "routes"
            counts, paths = MATERIALIZER.write_uncompressed_routes(
                raw_vcf=raw,
                audit=audit,
                output_dir=output,
                router=ROUTER,
                ensemble=ENSEMBLE,
            )
            self.assertEqual(set(counts.values()), {1})
            self.assertEqual(len(data_lines(paths["filtered_in"])), 1)
            self.assertEqual(len(data_lines(paths["filtered_out"])), 3)
            self.assertEqual(len(data_lines(paths["panel_discordant"])), 2)
            self.assertEqual(len(data_lines(paths["panel_present"])), 3)

            anchor = data_lines(paths["beagle_nonref_highconf"])[0]
            self.assertIn("BEAGLE_P_NONREF=0.9", anchor)
            self.assertIn("BEAGLE_GP=0.11,0.44,0.45", anchor)
            self.assertEqual(anchor.rstrip().split("\t")[9], RECORDS[0][4])
            self.assertEqual(
                MATERIALIZER.validate_materialized_routes(
                    raw_vcf=raw,
                    audit=audit,
                    route_paths=paths,
                    router=ROUTER,
                    ensemble=ENSEMBLE,
                ),
                {
                    "beagle_nonref_highconf": 1,
                    "beagle_ref_highconf": 1,
                    "beagle_uncertain": 1,
                    "panel_absent": 1,
                    "filtered_in": 1,
                    "filtered_out": 3,
                    "panel_discordant": 2,
                    "panel_present": 3,
                },
            )

    def test_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self.make_inputs(root)
            output = root / "routes"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                MATERIALIZER.write_uncompressed_routes(
                    raw_vcf=raw,
                    audit=audit,
                    output_dir=output,
                    router=ROUTER,
                    ensemble=ENSEMBLE,
                )

    def test_rejects_raw_audit_allele_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            audit = root / "ensemble.tsv.gz"
            write_raw(raw)
            rows = default_rows()
            rows[1]["pos"] = "201"
            write_audit(audit, rows)
            with self.assertRaisesRegex(ValueError, "raw/audit allele mismatch"):
                MATERIALIZER.validate_alignment(
                    raw, audit, router=ROUTER, ensemble=ENSEMBLE
                )

    def test_rejects_route_that_disagrees_with_direct_p_nonref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            audit = root / "ensemble.tsv.gz"
            write_raw(raw)
            rows = default_rows()
            rows[0]["route"] = "beagle_uncertain"
            rows[0]["hard_anchor"] = "0"
            write_audit(audit, rows)
            with self.assertRaisesRegex(
                ValueError, "route disagrees with ensemble posterior"
            ):
                MATERIALIZER.validate_alignment(
                    raw, audit, router=ROUTER, ensemble=ENSEMBLE
                )

    def test_rejects_preannotated_raw_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.vcf"
            audit = root / "ensemble.tsv.gz"
            write_raw(raw, info="BEAGLE_ROUTE=legacy")
            write_audit(audit, default_rows())
            with self.assertRaisesRegex(ValueError, "already contains Beagle"):
                MATERIALIZER.validate_alignment(
                    raw, audit, router=ROUTER, ensemble=ENSEMBLE
                )

    def test_final_vcf_validation_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self.make_inputs(root)
            output = root / "routes"
            _, paths = MATERIALIZER.write_uncompressed_routes(
                raw_vcf=raw,
                audit=audit,
                output_dir=output,
                router=ROUTER,
                ensemble=ENSEMBLE,
            )
            panel_present = paths["panel_present"]
            text = panel_present.read_text().replace(
                "BEAGLE_P_NONREF=0.9", "BEAGLE_P_NONREF=0.8", 1
            )
            panel_present.write_text(text)
            with self.assertRaisesRegex(ValueError, "differs from raw/audit"):
                MATERIALIZER.validate_materialized_routes(
                    raw_vcf=raw,
                    audit=audit,
                    route_paths=paths,
                    router=ROUTER,
                    ensemble=ENSEMBLE,
                )

    def synthetic_validated_bundle(self, root: Path, raw: Path, audit: Path):
        evidence = root / "source_manifest.json"
        evidence.write_text("{}\n")
        return {
            "contracts": CONTRACTS,
            "ensemble": ENSEMBLE,
            "manifest": {
                "method": "ensemble_median",
                "aggregation": "median",
                "frozen_source_seeds": [20260915, 20260916, 20260917],
                "counts": {
                    "beagle_nonref_highconf": 1,
                    "beagle_ref_highconf": 1,
                    "beagle_uncertain": 1,
                    "panel_absent": 1,
                    "total": 4,
                },
            },
            "manifest_path": evidence,
            "audit": audit,
            "raw_vcf": raw,
            "evidence_paths": (evidence,),
        }

    @staticmethod
    def fake_compression(command, check):
        del check
        executable = command[0]
        if executable == "fake-bgzip":
            source = Path(command[-1])
            with source.open("rb") as input_handle, gzip.open(
                str(source) + ".gz", "wb"
            ) as output_handle:
                output_handle.write(input_handle.read())
            source.unlink()
        elif executable == "fake-tabix":
            Path(str(command[-1]) + ".tbi").write_text("synthetic index\n")
        else:
            raise AssertionError(f"unexpected executable: {executable}")
        return mock.Mock(returncode=0)

    def test_full_materialization_writes_a_verified_completion_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self.make_inputs(root)
            bundle = self.synthetic_validated_bundle(root, raw, audit)
            output = root / "complete_routes"
            with mock.patch.object(
                MATERIALIZER,
                "validate_ensemble_bundle",
                return_value=bundle,
            ), mock.patch.object(
                MATERIALIZER.subprocess,
                "run",
                side_effect=self.fake_compression,
            ):
                result = MATERIALIZER.materialize(
                    ensemble_root=root / "synthetic_ensemble",
                    output_dir=output,
                    bgzip="fake-bgzip",
                    tabix="fake-tabix",
                )
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["production_inputs_include_wes"])
            self.assertEqual(result["method"], "ensemble_median")
            verified = CONTRACTS.verify_complete_bundle(output)
            self.assertEqual(verified["status"], "verified")
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertFalse(manifest["emits_phased_target_gt"])
            self.assertFalse(manifest["impute_effective"])
            self.assertEqual(manifest["counts"]["total"], 4)
            self.assertEqual(len(manifest["route_outputs"]), 8)
            for filename in MATERIALIZER.ROUTE_FILES.values():
                self.assertTrue((output / f"{filename}.gz").is_file())
                self.assertTrue((output / f"{filename}.gz.tbi").is_file())

    def test_compression_failure_never_writes_completion_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self.make_inputs(root)
            bundle = self.synthetic_validated_bundle(root, raw, audit)
            output = root / "failed_routes"
            with mock.patch.object(
                MATERIALIZER,
                "validate_ensemble_bundle",
                return_value=bundle,
            ), mock.patch.object(
                MATERIALIZER.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "fake-bgzip"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    MATERIALIZER.materialize(
                        ensemble_root=root / "synthetic_ensemble",
                        output_dir=output,
                        bgzip="fake-bgzip",
                        tabix="fake-tabix",
                    )
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "RUN_COMPLETE").exists())
            self.assertFalse((output / "RUN_COMPLETE.json").exists())

    def test_release_protocol_checksum_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "RELEASE_PROTOCOL.md"
            checksum = root / "RELEASE_PROTOCOL_SHA256.txt"
            protocol.write_text("frozen rule\n")
            checksum.write_text(f"{'0' * 64}  RELEASE_PROTOCOL.md\n")
            with self.assertRaisesRegex(ValueError, "protocol checksum mismatch"):
                MATERIALIZER.validate_protocol_checksum(protocol, checksum)


if __name__ == "__main__":
    unittest.main()
