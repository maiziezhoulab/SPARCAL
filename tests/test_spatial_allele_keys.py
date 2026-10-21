import gzip
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_DIR = PROJECT_ROOT / "scripts" / "6_spatial_filter"


def _load_module(name, filename):
    sys.path.insert(0, str(SPATIAL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, SPATIAL_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


matrix_module = _load_module("generate_sparcal_matrices_test", "generate_sparcal_matrices.py")
spatial_module = _load_module(
    "run_spatial_snv_filter_enhanced_test", "run_spatial_snv_filter_enhanced.py"
)


def _write_spot_table(path, rows):
    pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "race"]).to_csv(
        path, sep="\t", index=False
    )


def test_matrix_keeps_same_position_alt_alleles_distinct_and_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    rows = [
        ("chr1", "100", "A", "G", "defined"),
        ("1", "100", "A", "C", "defined"),
    ]
    _write_spot_table(first / "spot-b.txt", rows)
    _write_spot_table(first / "spot-a.txt", list(reversed(rows)))
    _write_spot_table(second / "spot-a.txt", rows)
    _write_spot_table(second / "spot-b.txt", list(reversed(rows)))

    loaded_first = matrix_module.load_category(first, None)
    loaded_second = matrix_module.load_category(second, None)
    matrix_first = matrix_module.build_matrix(loaded_first, sorted(loaded_first))
    matrix_second = matrix_module.build_matrix(loaded_second, sorted(loaded_second))

    assert list(matrix_first.columns) == ["1_100_A_C", "1_100_A_G"]
    pd.testing.assert_frame_equal(matrix_first, matrix_second)


def test_matrix_legacy_position_mode_is_explicit(tmp_path):
    _write_spot_table(
        tmp_path / "spot.txt",
        [("1", "100", "A", "C", "defined"), ("1", "100", "A", "G", "defined")],
    )

    loaded = matrix_module.load_category(tmp_path, None, legacy_position_only=True)

    assert loaded == {"spot": {"1_100"}}


def test_matrix_rejects_duplicate_exact_rows(tmp_path):
    row = ("1", "100", "A", "C", "defined")
    _write_spot_table(tmp_path / "spot.txt", [row, row])

    with pytest.raises(ValueError, match="duplicate allele-exact"):
        matrix_module.load_category(tmp_path, None)


def test_matrix_preflight_refuses_existing_output(tmp_path):
    output = tmp_path / "matrix.pkl"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        matrix_module.assert_outputs_absent([output])

    assert output.read_bytes() == b"existing"


def _load_spatial_records(tmp_path, lines):
    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir(parents=True)
    with gzip.open(vcf_dir / "spot.vcf.gz", "wt") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tspot\n")
        for line in lines:
            handle.write(line + "\n")

    instance = spatial_module.EnhancedSpatialSNVFilter.__new__(
        spatial_module.EnhancedSpatialSNVFilter
    )
    instance.snv_pos_dir = str(vcf_dir)
    instance.spot_positions = {"spot": (0.0, 0.0)}
    instance.spot_snvs = defaultdict(set)
    instance.snv_ref_alt_map = {}
    instance.snv_race = {}
    instance.snv_ref_counts = defaultdict(dict)
    instance.snv_alt_counts = defaultdict(dict)
    instance.snv_depth_counts = defaultdict(dict)
    instance.vcf_evidence_stats = defaultdict(int)
    instance.alt_presence_min_count = spatial_module.ALT_PRESENCE_MIN_COUNT
    instance.load_snv_data()
    return instance


def test_spatial_loader_keeps_alt_alleles_and_resolves_duplicates_deterministically(tmp_path):
    records = [
        "chr1\t100\t.\tA\tC\t.\tPASS\tRACE=denovo\tGT:AD:DP\t0/1:4,3:7",
        "chr1\t100\t.\tA\tG\t.\tPASS\tRACE=denovo\tGT:AD:DP\t0/1:5,2:7",
        "chr1\t100\t.\tA\tC\t.\tPASS\tRACE=denovo\tGT:AD:DP\t0/1:2,5:7",
    ]
    forward = _load_spatial_records(tmp_path / "forward", records)
    reverse = _load_spatial_records(tmp_path / "reverse", list(reversed(records)))

    assert forward.spot_snvs["spot"] == {"1_100_A_C", "1_100_A_G"}
    assert dict(forward.snv_alt_counts) == dict(reverse.snv_alt_counts)
    assert forward.snv_alt_counts["1_100_A_C"]["spot"] == 5
    assert forward.vcf_evidence_stats["records_exact_duplicate"] == 1


def test_spatial_output_preflight_preserves_existing_file(tmp_path):
    output = tmp_path / "spatial"
    output.mkdir()
    artifact = output / "summary.txt"
    artifact.write_text("keep me")
    instance = spatial_module.EnhancedSpatialSNVFilter.__new__(
        spatial_module.EnhancedSpatialSNVFilter
    )
    instance.output_dir = str(output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        instance._assert_output_is_empty()

    assert artifact.read_text() == "keep me"



def test_spatial_loader_fails_on_zero_required_vcfs(tmp_path):
    instance = spatial_module.EnhancedSpatialSNVFilter.__new__(
        spatial_module.EnhancedSpatialSNVFilter
    )
    instance.snv_pos_dir = str(tmp_path)
    instance.spot_positions = {}
    instance.vcf_evidence_stats = defaultdict(int)
    with pytest.raises(FileNotFoundError, match="zero required"):
        instance.load_snv_data()


def test_spatial_loader_fails_on_malformed_record(tmp_path):
    with pytest.raises(ValueError, match="malformed single-sample VCF"):
        _load_spatial_records(
            tmp_path,
            ["chr1\t100\t.\tA\tG\t.\tPASS\tRACE=denovo\tGT:AD:DP"],
        )


def test_spatial_loader_fails_on_corrupt_gzip(tmp_path):
    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    (vcf_dir / "spot.vcf.gz").write_bytes(b"not gzip")
    instance = spatial_module.EnhancedSpatialSNVFilter.__new__(
        spatial_module.EnhancedSpatialSNVFilter
    )
    instance.snv_pos_dir = str(vcf_dir)
    instance.spot_positions = {"spot": (0.0, 0.0)}
    instance.vcf_evidence_stats = defaultdict(int)
    instance.alt_presence_min_count = spatial_module.ALT_PRESENCE_MIN_COUNT
    with pytest.raises(ValueError, match="could not read required"):
        instance.load_snv_data()


def test_requested_optional_pool_missing_fails_closed(tmp_path):
    instance = spatial_module.EnhancedSpatialSNVFilter.__new__(
        spatial_module.EnhancedSpatialSNVFilter
    )
    instance.snv_ref_alt_map = {}
    with pytest.raises(FileNotFoundError, match="requested optional VCF"):
        instance.load_vcf_variants(str(tmp_path / "missing.vcf.gz"))


def test_noncanonical_legacy_min_expression_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="deprecated"):
        spatial_module.EnhancedSpatialSNVFilter(
            dataset="p4_tumor",
            section_id="1",
            quality_filter="baseQ0mapQ0",
            tumor_purity_file=None,
            output_dir=str(tmp_path / "fresh"),
            min_expression_germline=3,
            min_expression_somatic=1,
        )


def _write_spatial_manifest(sf_root, germline_files, somatic_files):
    import json

    (sf_root / "germline").mkdir(parents=True)
    (sf_root / "somatic").mkdir()
    manifest = {
        "schema": matrix_module.SPATIAL_FILTER_SCHEMA,
        "status": "complete",
        "dataset": "P4_TUMOR",
        "section_id": "1",
        "quality_filter": "baseQ0mapQ0",
        "per_barcode_files": {
            "germline": sorted(germline_files),
            "somatic": sorted(somatic_files),
        },
        "per_barcode_file_counts": {
            "germline": len(germline_files),
            "somatic": len(somatic_files),
        },
    }
    (sf_root / "MANIFEST.json").write_text(json.dumps(manifest))
    (sf_root / "RUN_COMPLETE").write_text("complete\n")


def test_matrix_loader_fails_on_missing_category(tmp_path):
    with pytest.raises(FileNotFoundError, match="category is missing"):
        matrix_module.load_category(tmp_path / "missing", None)


def test_matrix_loader_fails_on_read_error(tmp_path):
    (tmp_path / "spot.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="could not read required category"):
        matrix_module.load_category(tmp_path, None)


def test_matrix_requires_spatial_completion_sentinel(tmp_path):
    sf_root = tmp_path / "spatial_filter_purity" / "baseQ0mapQ0"
    _write_spatial_manifest(sf_root, [], [])
    (sf_root / "RUN_COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete spatial-filter"):
        matrix_module.validate_spatial_filter_bundle(
            sf_root, "P4_TUMOR", "1", "baseQ0mapQ0", {"germline"}
        )


def test_matrix_main_writes_manifest_and_completion_sidecar(tmp_path, monkeypatch):
    section_root = tmp_path / "section"
    sf_root = section_root / "spatial_filter_purity" / "baseQ0mapQ0"
    _write_spatial_manifest(sf_root, ["spot.txt"], ["spot.txt"])
    _write_spot_table(
        sf_root / "germline" / "spot.txt",
        [("chr1", "100", "A", "G", "defined")],
    )
    _write_spot_table(
        sf_root / "somatic" / "spot.txt",
        [("chr1", "200", "C", "T", "denovo")],
    )
    monkeypatch.setattr(sys, "argv", [
        "generate_sparcal_matrices.py",
        "--dataset", "P4_TUMOR",
        "--section_id", "1",
        "--quality_filter", "baseQ0mapQ0",
        "--section-root", str(section_root),
    ])
    matrix_module.main()

    matrix_dir = section_root / "matrix"
    for cls in ("1000G", "germline", "somatic", "merged"):
        output = matrix_dir / f"P4_TUMOR_1_SPARCAL_{cls}_matrix.pkl"
        assert output.is_file()
        assert Path(str(output) + ".manifest.json").is_file()
        assert Path(str(output) + ".complete").read_text().strip() == "complete"

    # A second invocation cannot replace either the matrices or their contracts.
    with pytest.raises(SystemExit):
        matrix_module.main()
