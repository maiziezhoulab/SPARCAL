#!/usr/bin/env python3
"""
generate_sparcal_matrices.py

Canonical SPARCAL spot x SNV matrix builder (replaces the fragile
run_generate_matrix.py / final_snv_mat.py path juggling that failed repeatedly).

It reads the per-barcode SNV lists produced by the spatial filter (step 7)
directly from

    {output_base}/{section_id}/spatial_filter_purity/{quality_filter}/
        germline/{barcode}.txt   (cols: chrom pos ref alt race[defined|denovo])
        somatic/{barcode}.txt     (cols: chrom pos ref alt race[denovo])

and emits FOUR binary (int8) spot x SNV matrices, one per class:

    1000G     germline rows with race == defined   (present in the 1000G panel)
    germline  all germline rows                    (1000G + UPV; == old "normal")
    somatic   all somatic rows
    merged    germline U somatic

All four share the SAME row index (the union of every barcode that has a
germline or somatic file), so they are directly comparable. Columns are the
per-class unique allele-exact SNVs, keyed ``chrom_pos_ref_alt``. The old
position-only representation is available only through the explicit
``--legacy-position-only`` compatibility flag.

Output (matrix dir is created if missing):
    {output_base}/{section_id}/matrix/{STUDY}_{section_id}_SPARCAL_{class}_matrix.pkl

Usage:
    python generate_sparcal_matrices.py --dataset P6_TUMOR --section_id 1
    python generate_sparcal_matrices.py --dataset DCIS     --section_id dcis1
"""

import argparse
import hashlib
import json
import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# Per-dataset pipeline output roots (case-sensitive: data/P6_tumor != data/p6_tumor).
# Kept in sync with run_spatial_snv_filter_enhanced.py "output_base".
PROJECT_BASE_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
# DLPFC_PREDEDUP=1 -> read the pre-dedup tree (data/dlpfc_prededup) so the
# post-dedup matrices under data/dlpfc are left untouched. Default: unchanged.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"
DATASET_OUTPUT_BASE = {
    "DLPFC":    "data/dlpfc_prededup" if _PREDEDUP else "data/dlpfc",
    "P4_TUMOR": "data/P4_tumor",
    "P6_TUMOR": "data/P6_tumor",
    "DCIS":     "data",        # section_id already carries the "dcis1"/"dcis2" prefix
    "OVAR_P5":  "data/ovar_p5",
    "NCCE":     "data/ncce",
}

# class token -> (source dir, race filter or None for "all rows in that dir")
# merged is handled specially (union of germline + somatic).
# "normal" is an alias of "germline" (all germline rows) for normal-tissue
# datasets like DLPFC — it is the direct successor of the old bcftools "normal".
CLASS_SPECS = {
    "1000G":    ("germline", "defined"),
    "UPV":      ("germline", "denovo"),
    "germline": ("germline", None),
    "normal":   ("germline", None),
    "somatic":  ("somatic",  None),
}

SPATIAL_FILTER_SCHEMA = "sparcal_spatial_filter_v1"
MATRIX_SCHEMA = "sparcal_spot_variant_matrix_v1"



def make_feature_id(chrom, pos, ref=None, alt=None, *, legacy_position_only=False):
    """Build a deterministic matrix feature ID from one per-barcode row."""
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    try:
        normalized_pos = str(int(str(pos)))
    except ValueError as error:
        raise ValueError(f"Invalid matrix position: {pos!r}") from error
    if int(normalized_pos) <= 0:
        raise ValueError(f"Invalid matrix position: {pos!r}")
    fields = (chrom, normalized_pos) if legacy_position_only else (
        chrom, normalized_pos, str(ref), str(alt)
    )
    if any(not field or field in {"nan", "."} or "_" in field for field in fields):
        raise ValueError(f"Invalid feature-ID fields: {fields!r}")
    if not legacy_position_only and "," in fields[3]:
        raise ValueError("Matrix input must contain one ALT allele per row")
    return "_".join(fields)


def _barcode_txt_files(cat_dir):
    """Return sorted per-barcode files, excluding aggregate class tables."""
    files = []
    for item in Path(cat_dir).glob("*.txt"):
        name = item.name
        if name.startswith("germline") or name.startswith("somatic"):
            continue
        files.append(item)
    return sorted(files)


def load_category(
    cat_dir,
    race_filter,
    *,
    legacy_position_only=False,
    expected_files=None,
):
    """Load a completed category using allele-exact, fail-closed parsing."""
    cat_dir = Path(cat_dir)
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"required spatial-filter category is missing: {cat_dir}")
    paths = _barcode_txt_files(cat_dir)
    actual_names = [path.name for path in paths]
    if expected_files is not None and actual_names != sorted(expected_files):
        raise ValueError(
            f"category inventory disagrees with Stage 6 manifest for {cat_dir}: "
            f"declared={len(expected_files)}, actual={len(paths)}"
        )
    spot_snvs = {}
    for path in paths:
        barcode = path.stem
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                dtype={"chrom": str, "pos": str, "ref": str, "alt": str, "race": str},
            )
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"empty per-barcode category file: {path}") from error
        except Exception as error:
            raise ValueError(f"could not read required category file {path}: {error}") from error
        required = {"chrom", "pos", "race"}
        if not legacy_position_only:
            required.update({"ref", "alt"})
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        if df.empty:
            raise ValueError(f"per-barcode category file has no variant rows: {path}")
        if df["race"].isna().any() or (df["race"].str.len() == 0).any():
            raise ValueError(f"{path} contains missing race values")
        invalid_races = sorted(
            set(df["race"])
            - {"defined", "panel_discordant", "denovo", "unknown"}
        )
        if invalid_races:
            raise ValueError(f"{path} contains invalid race values: {invalid_races}")
        if race_filter is not None:
            df = df[df["race"] == race_filter]
        feature_ids = [
            make_feature_id(
                row.chrom,
                row.pos,
                getattr(row, "ref", None),
                getattr(row, "alt", None),
                legacy_position_only=legacy_position_only,
            )
            for row in df.itertuples(index=False)
        ]
        if not legacy_position_only and len(feature_ids) != len(set(feature_ids)):
            duplicates = sorted(
                feature_id for feature_id, count in Counter(feature_ids).items()
                if count > 1
            )
            raise ValueError(
                f"{path} contains duplicate allele-exact rows: {duplicates[:5]}"
            )
        spot_snvs[barcode] = set(feature_ids)
    return spot_snvs


def validate_spatial_filter_bundle(sf_root, dataset, section_id, quality_filter, categories):
    """Require the Stage 6 manifest/sentinel and exact per-category inventories."""
    sf_root = Path(sf_root)
    manifest_path = sf_root / "MANIFEST.json"
    complete_path = sf_root / "RUN_COMPLETE"
    missing = [str(path) for path in (manifest_path, complete_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "incomplete spatial-filter input; missing: " + ", ".join(missing)
        )
    if complete_path.read_text().strip() != "complete":
        raise ValueError(f"invalid spatial-filter RUN_COMPLETE: {complete_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid spatial-filter manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != SPATIAL_FILTER_SCHEMA or manifest.get("status") != "complete":
        raise ValueError(f"unsupported/incomplete spatial-filter manifest: {manifest_path}")
    if (
        manifest.get("dataset") != dataset
        or str(manifest.get("section_id")) != str(section_id)
        or manifest.get("quality_filter") != quality_filter
    ):
        raise ValueError("spatial-filter manifest dataset/section/quality mismatch")
    declared = manifest.get("per_barcode_files")
    declared_counts = manifest.get("per_barcode_file_counts")
    if not isinstance(declared, dict) or not isinstance(declared_counts, dict):
        raise ValueError("spatial-filter manifest has no per-barcode inventory")
    for category in sorted(categories):
        if category not in {"germline", "somatic"}:
            raise ValueError(f"unknown spatial-filter category: {category}")
        if category not in declared or category not in declared_counts:
            raise ValueError(f"spatial-filter manifest is missing {category} inventory")
        names = declared[category]
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError(f"invalid {category} inventory in spatial-filter manifest")
        if int(declared_counts[category]) != len(names):
            raise ValueError(f"invalid {category} file count in spatial-filter manifest")
        category_dir = sf_root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"required category directory is missing: {category_dir}")
        actual = [path.name for path in _barcode_txt_files(category_dir)]
        if actual != sorted(names):
            raise ValueError(
                f"{category} files disagree with spatial-filter manifest: "
                f"declared={len(names)}, actual={len(actual)}"
            )
    return manifest, manifest_path


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_matrix(spot_snvs, all_barcodes):
    """Binary int8 DataFrame: rows=all_barcodes, cols=sorted unique SNVs."""
    all_snvs = set()
    for s in spot_snvs.values():
        all_snvs.update(s)
    snv_list = sorted(all_snvs)
    col_idx = {snv: j for j, snv in enumerate(snv_list)}

    mat = np.zeros((len(all_barcodes), len(snv_list)), dtype=np.int8)
    for i, bc in enumerate(all_barcodes):
        for snv in spot_snvs.get(bc, ()):  # bc may only exist in the other class
            j = col_idx.get(snv)
            if j is not None:
                mat[i, j] = 1
    return pd.DataFrame(mat, index=all_barcodes, columns=snv_list)


def assert_outputs_absent(paths):
    """Raise before matrix construction if any planned output already exists."""
    collisions = sorted(str(path) for path in paths if os.path.lexists(path))
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing matrix output(s):\n  "
            + "\n  ".join(collisions)
            + "\nUse a new --section-root or --model token."
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="P4_TUMOR, P6_TUMOR, DCIS, DLPFC, OVAR_P5")
    ap.add_argument("--section_id", required=True, help="e.g. 1, dcis1, 151507")
    ap.add_argument("--quality_filter", default="baseQ0mapQ0",
                    help="Upstream spot-profile quality recorded in the Stage 6 manifest")
    ap.add_argument("--spatial-filter-tag", default=None,
                    help="Directory name under spatial_filter_purity/ when it differs "
                         "from --quality_filter (for tagged experimental arms)")
    ap.add_argument("--section-root", default=None,
                    help="Override the section directory holding spatial_filter_purity/. "
                         "Redirects BOTH input and matrix output, so an experimental "
                         "tree (e.g. data/artifact_mask_benchmark/<label>/<sample>) "
                         "cannot overwrite the canonical matrices. Default: derived "
                         "from --dataset and --section_id.")
    ap.add_argument("--model", default="SPARCAL",
                    help="model token in the output name (default: SPARCAL)")
    ap.add_argument("--classes", nargs="+",
                    default=["1000G", "germline", "somatic", "merged"],
                    help="subset of classes to build (default: all four)")
    ap.add_argument(
        "--legacy-position-only",
        action="store_true",
        help="Compatibility mode: collapse REF/ALT alleles into chrom_pos columns. "
             "The default preserves chrom_pos_ref_alt allele identity.",
    )
    args = ap.parse_args()

    dataset = args.dataset.upper()
    if dataset not in DATASET_OUTPUT_BASE:
        print(f"ERROR: unknown dataset '{args.dataset}'", file=sys.stderr)
        sys.exit(1)

    # --section-root redirects BOTH the input spatial-filter tree and the matrix
    # output directory, so a masked/experimental tree can never overwrite the
    # canonical matrices: they move together into the parallel tree.
    if args.section_root:
        section_root = os.path.abspath(args.section_root)
    else:
        section_root = os.path.join(PROJECT_BASE_DIR, DATASET_OUTPUT_BASE[dataset],
                                    str(args.section_id))
    spatial_filter_tag = args.spatial_filter_tag or args.quality_filter
    sf_root = os.path.join(section_root, "spatial_filter_purity", spatial_filter_tag)
    if not os.path.isdir(sf_root):
        print(f"ERROR: spatial filter output not found: {sf_root}\n"
              f"       run step 7 (spatial filter) first.", file=sys.stderr)
        sys.exit(1)

    matrix_dir = os.path.join(section_root, "matrix")
    valid_classes = []
    for cls in args.classes:
        if cls not in CLASS_SPECS and cls != "merged":
            print(f"ERROR: unknown class '{cls}'", file=sys.stderr)
            sys.exit(1)
        if cls in valid_classes:
            print(f"ERROR: duplicate class requested: {cls}", file=sys.stderr)
            sys.exit(1)
        valid_classes.append(cls)
    required_categories = set()
    for cls in valid_classes:
        if cls == "merged":
            required_categories.update({"germline", "somatic"})
        else:
            required_categories.add(CLASS_SPECS[cls][0])
    try:
        source_manifest, source_manifest_path = validate_spatial_filter_bundle(
            sf_root, dataset, args.section_id, args.quality_filter,
            required_categories,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    planned_outputs = {
        cls: os.path.join(
            matrix_dir,
            f"{dataset}_{args.section_id}_{args.model}_{cls}_matrix.pkl",
        )
        for cls in valid_classes
    }
    planned_sidecars = [
        sidecar
        for output in planned_outputs.values()
        for sidecar in (output + ".manifest.json", output + ".complete")
    ]
    try:
        assert_outputs_absent([*planned_outputs.values(), *planned_sidecars])
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(matrix_dir, exist_ok=True)

    print(f"Reading per-barcode SNVs from {sf_root}")
    load_options = {"legacy_position_only": args.legacy_position_only}
    declared = source_manifest["per_barcode_files"]
    germline_defined = germline_denovo = germline_all = somatic_all = {}
    if "germline" in required_categories:
        category_options = {
            **load_options, "expected_files": declared["germline"]
        }
        germline_defined = load_category(
            os.path.join(sf_root, "germline"), "defined", **category_options)
        germline_denovo = load_category(
            os.path.join(sf_root, "germline"), "denovo", **category_options)
        germline_all = load_category(
            os.path.join(sf_root, "germline"), None, **category_options)
    if "somatic" in required_categories:
        somatic_all = load_category(
            os.path.join(sf_root, "somatic"), None,
            **load_options, expected_files=declared["somatic"],
        )

    # Shared row index remains the historical union of barcodes represented in
    # the requested source categories.
    all_barcodes = sorted(set(germline_all) | set(somatic_all))
    if not all_barcodes:
        print("ERROR: no per-barcode SNV files found — nothing to build.", file=sys.stderr)
        sys.exit(1)
    print(f"  germline barcodes: {len(germline_all):,}  "
          f"somatic barcodes: {len(somatic_all):,}  "
          f"union (rows): {len(all_barcodes):,}")

    sources = {
        "1000G":    germline_defined,
        "UPV":      germline_denovo,
        "germline": germline_all,
        "normal":   germline_all,   # alias for normal-tissue datasets (DLPFC)
        "somatic":  somatic_all,
    }

    outputs = []
    for cls in valid_classes:
        if cls == "merged":
            merged = {bc: (germline_all.get(bc, set()) | somatic_all.get(bc, set()))
                      for bc in all_barcodes}
            spot_snvs = merged
        else:
            spot_snvs = sources[cls]
        df = build_matrix(spot_snvs, all_barcodes)
        out_path = planned_outputs[cls]
        out_name = os.path.basename(out_path)
        df.to_pickle(out_path)
        outputs.append((cls, df.shape, out_path))
        print(f"  [{cls:8s}] {df.shape[0]:>6,} spots x {df.shape[1]:>8,} SNVs  ->  {out_name}")

    # Only after every requested matrix has been serialized do we publish each
    # per-matrix manifest and its completion sentinel.
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    for cls, shape, out_path in outputs:
        matrix_manifest = {
            "schema": MATRIX_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "dataset": dataset,
            "section_id": args.section_id,
            "quality_filter": args.quality_filter,
            "model": args.model,
            "matrix_class": cls,
            "matrix_path": str(Path(out_path).resolve()),
            "matrix_sha256": _sha256_file(out_path),
            "shape": {"spots": int(shape[0]), "variants": int(shape[1])},
            "feature_identity": (
                "chrom_pos" if args.legacy_position_only else "chrom_pos_ref_alt"
            ),
            "source_spatial_filter_schema": source_manifest["schema"],
            "source_spatial_filter_manifest": str(source_manifest_path.resolve()),
            "source_spatial_filter_manifest_sha256": source_manifest_sha256,
        }
        manifest_path = Path(out_path + ".manifest.json")
        complete_path = Path(out_path + ".complete")
        with manifest_path.open("x") as handle:
            json.dump(matrix_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with complete_path.open("x") as handle:
            handle.write("complete\n")

    print("\nDone. Matrices written to:", matrix_dir)
    for cls, shape, output_path in outputs:
        print(f"  {cls:8s} {shape}  {output_path}")


if __name__ == "__main__":
    main()
