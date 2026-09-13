#!/usr/bin/env python3
"""Materialize independent model matrices and collision-safe viewer studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.spatial_filter_rebuild import config


def model_token(model: str, selection: str) -> str:
    return f"SPARCAL_{config.viewer_label(model, selection)}"


def output_tree(section: str, model: str, selection: str) -> Path:
    return config.section_dir(section) / "materialized" / f"{model}__{selection}"


def compatible_matrix_path(section: str, model: str, selection: str) -> Path:
    cfg = config.SECTIONS[section]
    dataset = f"{cfg['patient']}_TUMOR"
    return (
        output_tree(section, model, selection) / "matrix"
        / f"{dataset}_{cfg['rep']}_{model_token(model, selection)}_somatic_matrix.pkl"
    )


def materialize_matrix(section: str, model: str, selection: str, overwrite: bool) -> dict[str, object]:
    cache = config.cache_dir(section)
    candidates = pd.read_csv(cache / "candidates.tsv.gz", sep="\t", usecols=["variant"])
    spots = pd.read_csv(cache / "spots.tsv", sep="\t").sort_values("spot_index")
    presence = sparse.load_npz(cache / "presence.npz").tocsr()
    selected_path = config.score_dir(section) / f"selected__{model}__{selection}.tsv.gz"
    selected = pd.read_csv(selected_path, sep="\t", usecols=["variant"])
    index = pd.Series(np.arange(len(candidates)), index=candidates["variant"].astype(str))
    missing = sorted(set(selected["variant"].astype(str)) - set(index.index))
    if missing:
        raise ValueError(f"{len(missing)} selected variants absent from cache")
    row_indices = index.reindex(selected["variant"].astype(str)).to_numpy(dtype=np.int64)
    matrix_sparse = presence[row_indices, :].T.tocsr()
    matrix = pd.DataFrame(
        matrix_sparse.toarray().astype(np.uint8, copy=False),
        index=spots["barcode"].astype(str).tolist(),
        columns=selected["variant"].astype(str).tolist(),
    )
    path = compatible_matrix_path(section, model, selection)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_pickle(path)
    result = {
        "section": section,
        "model": model,
        "selection": selection,
        "selected_source": str(selected_path),
        "matrix": str(path),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "positive_entries": int(matrix_sparse.nnz),
    }
    (path.parent.parent / "matrix_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_viewer(section: str, model: str, selection: str, viewer_root: Path, overwrite: bool) -> None:
    cfg = config.SECTIONS[section]
    command = [
        sys.executable,
        str(config.ROOT / "scripts/preprocess/build_sparcalviewer_profile.py"),
        "--dataset", f"{cfg['patient']}_TUMOR",
        "--section-id", str(cfg["rep"]),
        "--section-root", str(output_tree(section, model, selection)),
        "--quality-filter", config.QUALITY_FILTER,
        "--model", model_token(model, selection),
        "--matrix-class", "somatic",
        "--run-label", config.viewer_label(model, selection),
        "--output-root", str(viewer_root),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=config.ROOT, check=True)
    study_name = (
        f"{cfg['patient']}_rep{cfg['rep']}_SPARCAL_somatic_"
        f"{config.viewer_label(model, selection)}"
    )
    study_dir = viewer_root / study_name
    sys.path.insert(0, str(config.BENCHMARKING / "viewer"))
    from sparcal_viewer.config import load_config
    from sparcal_viewer.data import StudyData

    study = StudyData(load_config(str(study_dir / f"{study_name}.config")))
    region_rows = []
    center_rows = []
    summaries = []
    for profile, normalize in (("AutoFixedRaw", False), ("AutoFixedCoverageResidual", True)):
        result = study.auto_tumor_regions(
            seed_pct=90, grow_pct=80, min_size=5, normalize=normalize, split_depth=0.40
        )
        for index, (region, centers) in enumerate(zip(result["regions"], result["centers"]), 1):
            name = f"Tumor_{index}"
            region_rows.extend((profile, name, barcode) for barcode in region)
            center_rows.extend((profile, name, barcode) for barcode in centers)
        summaries.append({
            "profile": profile,
            "normalize_by_umi": normalize,
            "n_regions": len(result["regions"]),
            "n_tumor_spots": sum(len(region) for region in result["regions"]),
            "parameters": {"seed_pct": 90, "grow_pct": 80, "min_size": 5, "split_depth": 0.40},
        })
    pd.DataFrame(region_rows, columns=["profile", "region_name", "barcode"]).to_csv(
        study_dir / "tumor_groups.csv", index=False
    )
    pd.DataFrame(center_rows, columns=["profile", "region_name", "barcode"]).to_csv(
        study_dir / "tumor_centers.csv", index=False
    )
    (study_dir / "auto_tumor_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    archive_path = viewer_root / f"{study_name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in study_dir.rglob("*") if item.is_file()):
            archive.write(path, arcname=f"{study_name}/{path.relative_to(study_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    parser.add_argument("--models", nargs="+", default=["baseline", *config.MODEL_NAMES])
    parser.add_argument("--selections", nargs="+", default=["matched"], choices=config.SELECTIONS)
    parser.add_argument("--viewer-root", type=Path, default=config.OUTPUT_ROOT / "viewer_studies")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = []
    for model in args.models:
        for selection in args.selections:
            if model == "baseline" and selection != "matched":
                continue
            result = materialize_matrix(args.section, model, selection, args.overwrite)
            results.append(result)
            if not args.skip_viewer and result["shape"][1] > 0:
                build_viewer(args.section, model, selection, args.viewer_root, args.overwrite)
    manifest = config.section_dir(args.section) / "materialized/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
