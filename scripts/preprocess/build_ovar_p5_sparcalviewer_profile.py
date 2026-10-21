#!/usr/bin/env python3
"""Build the current OVAR_P5 merged-SPARCAL package for SPARCALViewer.

All generated files are written below ``data/ovar_p5/viewer_profiles``.  The
SpaceRanger and pipeline inputs are read only.  This supersedes the older
germline-only OVAR viewer bundle with the current merged SPARCAL matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_ncce_sparcalviewer_profiles import PROJECT_ROOT, sha256, total_umi_per_spot, viewer_config


SECTION_ID = "P5_sr13"
STUDY_NAME = "OVAR_P5_SPARCAL"
SPACERANGER_OUTS = Path(
    "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs/P5_sr13/outs"
)
POSITION_COLUMNS = [
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
]


def read_tissue_positions(path: Path) -> pd.DataFrame:
    """Normalize SpaceRanger's headerless tissue_positions_list.csv."""
    first_line = path.read_text().splitlines()[0]
    if first_line.startswith("barcode,"):
        positions = pd.read_csv(path)
    else:
        positions = pd.read_csv(path, header=None, names=POSITION_COLUMNS)
    missing = set(POSITION_COLUMNS) - set(positions.columns)
    if missing:
        raise ValueError(f"Missing position columns in {path}: {sorted(missing)}")
    return positions[POSITION_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "data" / "ovar_p5" / "viewer_profiles",
        help="Project-local directory for the viewer folder and release asset",
    )
    args = parser.parse_args()

    matrix_path = (
        PROJECT_ROOT / "data" / "ovar_p5" / SECTION_ID / "matrix"
        / "OVAR_P5_P5_sr13_SPARCAL_merged_matrix.pkl"
    )
    spatial_dir = SPACERANGER_OUTS / "spatial"
    raw_positions = spatial_dir / "tissue_positions_list.csv"
    source_files = {
        "tissue_hires_image.png": spatial_dir / "tissue_hires_image.png",
        "scalefactors_json.json": spatial_dir / "scalefactors_json.json",
    }
    required = [matrix_path, raw_positions, *source_files.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required OVAR_P5 viewer inputs are missing:\n  " + "\n  ".join(missing))

    package_dir = args.output_root / STUDY_NAME
    package_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_pickle(matrix_path)
    if matrix.empty:
        raise ValueError(f"Merged SPARCAL matrix is empty: {matrix_path}")
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = (matrix > 0).astype(np.uint8)
    matrix.to_pickle(package_dir / "snv_matrix.pkl")

    read_tissue_positions(raw_positions).to_csv(package_dir / "tissue_positions.csv", index=False)
    for output_name, source_path in source_files.items():
        shutil.copy2(source_path, package_dir / output_name)
    total_umi_per_spot(SPACERANGER_OUTS).to_csv(package_dir / "spot_coverage.csv", index=False)

    # No OVAR_P5 tumor/cell-type annotations were supplied for this package.
    # Leave valid, empty viewer profile tables for interactive region creation.
    pd.DataFrame(columns=["profile", "region_name", "barcode"]).to_csv(
        package_dir / "tumor_groups.csv", index=False
    )
    pd.DataFrame(columns=["profile", "region_name", "group_name", "group_type", "snv_key"]).to_csv(
        package_dir / "variant_groups.csv", index=False
    )
    pd.DataFrame(columns=["profile", "region_name", "barcode"]).to_csv(
        package_dir / "tumor_centers.csv", index=False
    )
    (package_dir / f"{STUDY_NAME}.config").write_text(viewer_config(STUDY_NAME))

    package_manifest = {
        "study": STUDY_NAME,
        "matrix_source": "data/ovar_p5/P5_sr13/matrix/OVAR_P5_P5_sr13_SPARCAL_merged_matrix.pkl",
        "matrix_class": "merged",
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_dtype": str(matrix.dtypes.iloc[0]),
        "profiles": "empty; no OVAR_P5 tumor or cell-type annotation was supplied",
        "source_data_read_only": True,
    }
    (package_dir / "MANIFEST.json").write_text(json.dumps(package_manifest, indent=2) + "\n")

    release_dir = args.output_root / "release_assets"
    release_dir.mkdir(parents=True, exist_ok=True)
    archive_path = release_dir / f"{STUDY_NAME}.zip"
    package_files = [
        package_dir / "snv_matrix.pkl",
        package_dir / "tissue_positions.csv",
        package_dir / "tissue_hires_image.png",
        package_dir / "scalefactors_json.json",
        package_dir / "spot_coverage.csv",
        package_dir / "tumor_groups.csv",
        package_dir / "variant_groups.csv",
        package_dir / "tumor_centers.csv",
        package_dir / f"{STUDY_NAME}.config",
        package_dir / "MANIFEST.json",
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_files:
            archive.write(path, arcname=f"{STUDY_NAME}/{path.name}")

    release_manifest = pd.DataFrame([{
        "study": STUDY_NAME,
        "spots": matrix.shape[0],
        "snvs": matrix.shape[1],
        "matrix_class": "merged",
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
    }])
    manifest_path = release_dir / "OVAR_P5_SPARCAL_manifest.tsv"
    release_manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"Package: {package_dir}")
    print(f"Archive: {archive_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
