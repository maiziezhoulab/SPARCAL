import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pysam


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "5_refilter_bam"
    / "run_filter_bams_by_snv_pools.py"
)
SPEC = importlib.util.spec_from_file_location("run_filter_bams_by_snv_pools", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
SCHEMA = "sparcal_beagle_evidence_v2_2"
AUDIT_COLUMNS = [
    "schema", "chrom", "pos", "ref", "alt", "raw_gt", "beagle_gt",
    "beagle_gp", "beagle_called_gp", "beagle_p_nonref", "beagle_gp_entropy",
    "beagle_gp_margin", "beagle_af", "gt_transition", "route", "hard_anchor",
]


def _record(route, pos, gt="0/1"):
    return {
        "route": route,
        "chrom": "chr1",
        "pos": pos,
        "ref": "A",
        "alt": "G",
        "gt": gt,
    }


def write_routed_vcf(path, records=(), schema=SCHEMA, probability=0.9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            f'##SPARCAL_BEAGLE_ROUTER=<Schema="{schema}",'
            f'AnchorScore="p_nonref",MinAnchorProbability={probability:g},'
            'HardAnchor="beagle_nonref_highconf">\n'
        )
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for record in records:
            info = (
                f"BEAGLE_EVIDENCE_SCHEMA={schema};BEAGLE_ROUTE={record['route']};"
                f"BEAGLE_GT={record['gt']};BEAGLE_GP=0.01,0.98,0.01;"
                "BEAGLE_CALLED_GP=0.98;BEAGLE_P_NONREF=0.99"
            )
            handle.write(
                f"{record['chrom']}\t{record['pos']}\t.\t{record['ref']}\t"
                f"{record['alt']}\t.\tPASS\t{info}\tGT\t{record['gt']}\n"
            )
    Path(str(path) + ".tbi").touch()


def write_complete_bundle(section_root, route_records=None, probability=0.9, schema=SCHEMA):
    bundle = section_root / "output_VCFs/beagle/baseQ0mapQ0"
    if route_records is None:
        route_records = {route: [] for route in MODULE.ROUTE_TO_FILENAME}
    for route, filename in MODULE.ROUTE_TO_FILENAME.items():
        write_routed_vcf(
            bundle / filename, route_records.get(route, ()),
            schema=schema, probability=probability,
        )
    compatibility = {
        "all_filtered_in.vcf.gz": list(route_records.get("beagle_nonref_highconf", ())),
        "all_filtered_out.vcf.gz": [
            *route_records.get("beagle_ref_highconf", ()),
            *route_records.get("beagle_uncertain", ()),
            *route_records.get("panel_absent", ()),
        ],
        "all_panel_discordant.vcf.gz": [
            *route_records.get("beagle_ref_highconf", ()),
            *route_records.get("beagle_uncertain", ()),
        ],
    }
    if schema == MODULE.PANEL_PRESENT_SCHEMA:
        compatibility[MODULE.PANEL_PRESENT_VCF_NAME] = [
            *route_records.get("beagle_nonref_highconf", ()),
            *route_records.get("beagle_ref_highconf", ()),
            *route_records.get("beagle_uncertain", ()),
        ]
    for filename, records in compatibility.items():
        write_routed_vcf(
            bundle / filename, records, schema=schema, probability=probability
        )

    counts = {
        route: len(route_records.get(route, ())) for route in MODULE.ROUTE_TO_FILENAME
    }
    total = sum(counts.values())
    with gzip.open(bundle / "beagle_evidence_audit.tsv.gz", "wt") as handle:
        handle.write("\t".join(AUDIT_COLUMNS) + "\n")
        for route in MODULE.ROUTE_TO_FILENAME:
            for record in route_records.get(route, ()):
                handle.write("\t".join([
                    schema, record["chrom"], str(record["pos"]), record["ref"],
                    record["alt"], record["gt"], record["gt"],
                    "0.01,0.98,0.01", "0.98", "0.99", "0.1", "0.97",
                    "0.2", f"{record['gt']}>{record['gt']}", route,
                    "1" if route == "beagle_nonref_highconf" else "0",
                ]) + "\n")
    with (bundle / "beagle_evidence_summary.tsv").open("w") as handle:
        handle.write(
            "schema\troute\tcount\tanchor_score\t"
            "min_anchor_probability\thard_anchor\n"
        )
        for route in MODULE.ROUTE_TO_FILENAME:
            handle.write(
                f"{schema}\t{route}\t{counts[route]}\tp_nonref\t"
                f"{probability:g}\t{int(route == 'beagle_nonref_highconf')}\n"
            )
        handle.write(
            f"{schema}\ttotal\t{total}\tp_nonref\t{probability:g}\t.\n"
        )
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "schema": schema,
        "compatibility": {name: "declared test union" for name in compatibility},
        "anchor_score": "p_nonref",
        "min_anchor_probability": probability,
        "hard_anchor_route": "beagle_nonref_highconf",
        "production_inputs_include_wes": False,
        "raw_record_count": total,
        "counts": {**counts, "total": total},
    }))
    (bundle / "RUN_COMPLETE").write_text("complete\n")
    return bundle


def write_empty_classifier(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
    Path(str(path) + ".tbi").touch()


def make_generator(section_root):
    return MODULE.SNVMatrixGenerator(
        dataset_name="P4_TUMOR",
        section_id="1",
        quality_filter="baseQ0mapQ0",
        section_root=str(section_root),
    )


class ExactAlleleCountingTests(unittest.TestCase):
    def test_shared_position_alleles_are_counted_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            bam = Path(tmp) / "spot.bam"
            header = {"HD": {"VN": "1.6", "SO": "coordinate"},
                      "SQ": [{"SN": "chr1", "LN": 100}]}
            with pysam.AlignmentFile(bam, "wb", header=header) as handle:
                for index, sequence in enumerate(("AAAAAAAAAA", "AAAAGAAAAA")):
                    read = pysam.AlignedSegment()
                    read.query_name = f"read{index}"
                    read.query_sequence = sequence
                    read.flag = 0
                    read.reference_id = 0
                    read.reference_start = 0
                    read.mapping_quality = 60
                    read.cigar = ((0, 10),)
                    read.query_qualities = pysam.qualitystring_to_array("I" * 10)
                    handle.write(read)
            pysam.index(str(bam))

            alleles = {5: [("A", "G"), ("A", "C")], 11: [("A", "T")]}
            reads, detected, counts = MODULE.filter_bam_one_chrom(
                str(bam), "chr1", alleles
            )

            self.assertEqual(len(reads), 2)
            self.assertEqual(detected, {(5, "A", "C"), (5, "A", "G")})
            self.assertEqual(counts[(5, "A", "C")], [1, 0])
            self.assertEqual(counts[(5, "A", "G")], [1, 1])
            self.assertNotIn((11, "A", "T"), detected)

            _, reverse_detected, reverse_counts = MODULE.filter_bam_one_chrom(
                str(bam), "chr1", {5: list(reversed(alleles[5]))}
            )
            self.assertEqual(reverse_detected, detected)
            self.assertEqual(reverse_counts, counts)


class RouteBundleValidationTests(unittest.TestCase):
    def test_complete_pnonref090_bundle_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bundle(root)
            manifest = make_generator(root).validate_beagle_route_bundle()
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertEqual(manifest["min_anchor_probability"], 0.9)

    def test_v21_bundle_without_panel_present_remains_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bundle(
                root, schema="sparcal_beagle_evidence_v2_1"
            )
            manifest = make_generator(root).validate_beagle_route_bundle()
            self.assertEqual(manifest["schema"], "sparcal_beagle_evidence_v2_1")

    def test_v22_requires_panel_present_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_complete_bundle(root)
            (bundle / MODULE.PANEL_PRESENT_VCF_NAME).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "panel-present"):
                make_generator(root).validate_beagle_route_bundle()

    def test_missing_completion_sentinel_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_complete_bundle(root)
            (bundle / "RUN_COMPLETE").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "incomplete Beagle four-route"):
                make_generator(root).validate_beagle_route_bundle()

    def test_nonproduction_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bundle(root, probability=0.91)
            with self.assertRaisesRegex(ValueError, "must be P\\(nonref\\)>=0.90"):
                make_generator(root).validate_beagle_route_bundle()

    def test_vcf_schema_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_complete_bundle(root)
            write_routed_vcf(
                bundle / "all_beagle_uncertain.vcf.gz", schema="sparcal_beagle_evidence_v1"
            )
            with self.assertRaisesRegex(ValueError, "contract disagrees"):
                make_generator(root).validate_beagle_route_bundle()

    def test_compatibility_union_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes = {route: [] for route in MODULE.ROUTE_TO_FILENAME}
            routes["beagle_nonref_highconf"] = [_record("beagle_nonref_highconf", 10)]
            bundle = write_complete_bundle(root, routes)
            write_routed_vcf(bundle / "all_filtered_in.vcf.gz")
            with self.assertRaisesRegex(ValueError, "not the declared route union"):
                make_generator(root).validate_beagle_route_bundle()

    def test_panel_present_union_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes = {route: [] for route in MODULE.ROUTE_TO_FILENAME}
            routes["beagle_nonref_highconf"] = [
                _record("beagle_nonref_highconf", 10)
            ]
            bundle = write_complete_bundle(root, routes)
            write_routed_vcf(bundle / MODULE.PANEL_PRESENT_VCF_NAME)
            with self.assertRaisesRegex(ValueError, "not the declared route union"):
                make_generator(root).validate_beagle_route_bundle()

    def test_route_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes = {route: [] for route in MODULE.ROUTE_TO_FILENAME}
            routes["beagle_nonref_highconf"] = [_record("beagle_nonref_highconf", 10)]
            routes["beagle_ref_highconf"] = [_record("beagle_ref_highconf", 10)]
            write_complete_bundle(root, routes)
            with self.assertRaisesRegex(ValueError, "not disjoint"):
                make_generator(root).validate_beagle_route_bundle()


class GenotypeAndInputContractTests(unittest.TestCase):
    def test_canonical_full_rerun_uses_pnonref_anchor_contract(self):
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "run_slurm"
            / "run_beagle_panel_routing_experiment.sh"
        ).read_text()
        self.assertIn(
            "--anchor-score p_nonref --min-anchor-probability 0.90",
            wrapper,
        )
        self.assertNotIn("--min-anchor-gp", wrapper)
        self.assertIn("weak_label_score=called_gp", wrapper)

    def test_phased_and_unphased_nonref_genotypes_are_normalized(self):
        generator = MODULE.SNVMatrixGenerator.__new__(MODULE.SNVMatrixGenerator)
        line = "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t1|0\n"
        self.assertTrue(generator._is_valid_genotype(line))
        self.assertEqual(generator._normalized_genotype(line), (0, 1))

    def test_authoritative_route_is_not_vetoed_by_raw_reference_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes = {route: [] for route in MODULE.ROUTE_TO_FILENAME}
            routes["beagle_nonref_highconf"] = [
                _record("beagle_nonref_highconf", 10, gt="0/0")
            ]
            write_complete_bundle(root, routes)
            generator = make_generator(root)
            write_empty_classifier(Path(generator.classifier_homo_vcf))
            write_empty_classifier(Path(generator.classifier_hetero_vcf))
            snvs = generator.collect_snvs()
            self.assertEqual({snv.key for snv in snvs}, {"1_10_A_G"})

    def test_missing_required_classifier_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_bundle(root)
            with self.assertRaisesRegex(FileNotFoundError, "required classifier"):
                make_generator(root).collect_snvs()


class OutputPreflightTests(unittest.TestCase):
    def test_constructor_does_not_create_outputs_and_preflight_refuses_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator = make_generator(root)
            self.assertFalse(Path(generator.filtered_bam_dir).exists())
            self.assertFalse(Path(generator.log_dir).exists())

            generator.preflight_output_paths()
            generator.initialize_output_paths()
            self.assertTrue(Path(generator.bam_by_spot_dir).is_dir())
            self.assertTrue(Path(generator.vcf_by_spot_dir).is_dir())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                generator.preflight_output_paths()


if __name__ == "__main__":
    unittest.main()
