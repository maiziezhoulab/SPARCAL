#!/usr/bin/env python3
"""Build a phase/CNV/purity tumor sidecar directly from one CalicoST run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from phase_aware_evidence import (
    aggregate_phase_evidence,
    combine_tumor_evidence,
    phase_cnv_log_bayes_factor,
    purity_log_prior_shift,
)


SCHEMA = "SPARCAL_tumor_evidence_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_barcode(value: object) -> str:
    return str(value).split("_")[0]


def _load_calicost_rectangle(
    parsed_inputs: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    bins = pd.read_csv(parsed_inputs / "table_bininfo.csv.gz", sep="\t")
    required_bins = {"CHR", "START", "END"}
    if missing := required_bins - set(bins.columns):
        raise ValueError(f"table_bininfo missing columns: {sorted(missing)}")
    counts = pd.read_csv(
        parsed_inputs / "table_rdrbaf.csv.gz",
        sep="\t",
        usecols=["BARCODES", "TOT", "B"],
    )
    native = counts.BARCODES.drop_duplicates().astype(str).tolist()
    n_spots, n_bins = len(native), len(bins)
    if len(counts) != n_spots * n_bins:
        raise ValueError("CalicoST BAF table is not a complete spots x bins rectangle")
    names = counts.BARCODES.astype(str).to_numpy().reshape(n_spots, n_bins)
    if not np.all(names == np.asarray(native)[:, None]):
        raise ValueError("CalicoST BAF rows are not grouped in stable spot/bin order")
    barcodes = np.asarray([_clean_barcode(value) for value in native])
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("CalicoST barcodes are not unique after sample-suffix removal")
    total = (
        pd.to_numeric(counts.TOT, errors="raise")
        .to_numpy(float)
        .reshape(n_spots, n_bins)
    )
    counted = (
        pd.to_numeric(counts.B, errors="raise")
        .to_numpy(float)
        .reshape(n_spots, n_bins)
    )
    if (
        not np.isfinite(total).all()
        or not np.isfinite(counted).all()
        or np.any(counted < 0)
        or np.any(total < counted)
    ):
        raise ValueError("CalicoST B/TOT counts are invalid")
    return bins, barcodes, counted, total


def _load_spot_values(path: Path, value_column: str) -> dict[str, float]:
    frame = pd.read_csv(path, sep="\t")
    if "BARCODES" not in frame or value_column not in frame:
        raise ValueError(f"{path.name} requires BARCODES and {value_column}")
    clean = frame.BARCODES.map(_clean_barcode)
    if clean.duplicated().any():
        raise ValueError(f"duplicate cleaned barcodes in {path}")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return dict(zip(clean, values))


def _copy_number_by_clone(
    bins: pd.DataFrame, cnv_path: Path, clone_ids: np.ndarray
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    cnv = pd.read_csv(cnv_path, sep="\t")
    coordinates = ["CHR", "START", "END"]
    if any(column not in cnv for column in coordinates):
        raise ValueError("CNV table requires CHR, START, and END")
    left = bins[coordinates].copy()
    left["CHR"] = left.CHR.astype(str).str.removeprefix("chr")
    right = cnv.copy()
    right["CHR"] = right.CHR.astype(str).str.removeprefix("chr")
    merged = left.merge(right, on=coordinates, how="left", validate="one_to_one")
    result = {}
    for clone in dict.fromkeys(clone_ids.tolist()):
        token = str(int(float(clone)))
        acol, bcol = f"clone{token} A", f"clone{token} B"
        if acol not in merged or bcol not in merged:
            raise ValueError(f"CNV table lacks {acol}/{bcol}")
        result[token] = (
            pd.to_numeric(merged[acol], errors="coerce").to_numpy(float),
            pd.to_numeric(merged[bcol], errors="coerce").to_numpy(float),
        )
    clone_columns = merged.filter(regex=r"^clone.* [AB]$")
    matched = int(clone_columns.notna().any(axis=1).sum())
    return result, matched


def _load_optional_components(
    path: Path | None, barcodes: np.ndarray
) -> dict[str, np.ndarray]:
    names = ("somatic_vaf_log_bf", "molecular_log_bf", "spatial_log_prior")
    result = {name: np.zeros(len(barcodes), float) for name in names}
    if path is None:
        return result
    sep = "," if path.name.lower().endswith((".csv", ".csv.gz")) else "\t"
    frame = pd.read_csv(path, sep=sep)
    forbidden = {
        "wes",
        "truth",
        "ground_truth",
        "silver_standard",
        "truth_class",
        "outcome",
        "is_positive",
        "class_label",
        "validated",
    }
    columns = {str(column).strip().lower() for column in frame.columns}
    if found := forbidden & columns:
        raise ValueError(f"component table contains forbidden GT columns: {sorted(found)}")
    if "barcode" not in frame or frame.barcode.astype(str).duplicated().any():
        raise ValueError("component table requires unique barcode rows")
    frame.index = frame.barcode.astype(str)
    for name in names:
        if name not in frame:
            continue
        mapped = (
            pd.to_numeric(frame[name], errors="coerce")
            .reindex(barcodes)
            .to_numpy(float)
        )
        result[name] = np.where(np.isfinite(mapped), mapped, 0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parsed_inputs", type=Path)
    parser.add_argument("purity", type=Path, help="loh_estimator_tumor_prop.tsv")
    parser.add_argument("clones", type=Path, help="clone_labels.tsv")
    parser.add_argument("cnv", type=Path, help="selected cnv*_seglevel.tsv")
    parser.add_argument("output", type=Path, help="new tumor_evidence.npz")
    parser.add_argument("--spot-components", type=Path)
    parser.add_argument("--prior-probability", type=float, default=0.20)
    parser.add_argument("--min-bin-depth", type=int, default=1)
    parser.add_argument("--model-name", default="calicost_phase_cnv_purity_v1")
    parser.add_argument("--confirm-label-free", action="store_true", required=True)
    parser.add_argument(
        "--acknowledge-in-sample-calicost-evidence",
        action="store_true",
        required=True,
        help="acknowledge CN/purity were inferred from the same CalicoST BAF/RDR run",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.min_bin_depth < 1:
        raise ValueError("min-bin-depth must be positive")

    bins, barcodes, counted, total = _load_calicost_rectangle(args.parsed_inputs)
    purity_map = _load_spot_values(args.purity, "Tumor")
    clone_map = _load_spot_values(args.clones, "clone_label")
    purities = np.asarray([purity_map.get(barcode, np.nan) for barcode in barcodes])
    clones = np.asarray([clone_map.get(barcode, np.nan) for barcode in barcodes])
    if np.any(~np.isfinite(clones)):
        raise ValueError("some CalicoST BAF barcodes lack clone assignments")
    copy_by_clone, matched_bins = _copy_number_by_clone(bins, args.cnv, clones)

    phase_log_bf = np.zeros(len(barcodes), float)
    phase_information = np.zeros(len(barcodes), float)
    phase_available = np.zeros(len(barcodes), bool)
    observations_per_spot = np.zeros(len(barcodes), int)
    for row, (barcode, clone) in enumerate(zip(barcodes, clones)):
        token = str(int(float(clone)))
        copy_a, copy_b = copy_by_clone[token]
        use = total[row] >= args.min_bin_depth
        if not use.any():
            continue
        scored = phase_cnv_log_bayes_factor(
            counted[row, use],
            total[row, use],
            np.full(int(use.sum()), purities[row]),
            copy_a[use],
            copy_b[use],
            orientation_probability=np.ones(int(use.sum())),
            min_depth=args.min_bin_depth,
        )
        aggregate = aggregate_phase_evidence(
            [barcode] * int(use.sum()), scored["log_bf"], scored["information"]
        )[barcode]
        phase_log_bf[row] = aggregate["log_bf"]
        phase_information[row] = aggregate["information"]
        observations_per_spot[row] = int(aggregate["n_observations"])
        phase_available[row] = observations_per_spot[row] > 0

    purity_evidence = purity_log_prior_shift(
        purities, prior_probability=args.prior_probability
    )
    purity_available = purity_evidence["available"]
    information = np.maximum(phase_information, purity_available.astype(float))
    valid = information > 0
    extras = _load_optional_components(args.spot_components, barcodes)
    combined = combine_tumor_evidence(
        phase_log_bf,
        purity_evidence["log_prior"],
        extras["somatic_vaf_log_bf"],
        extras["molecular_log_bf"],
        extras["spatial_log_prior"],
        information,
        prior_probability=args.prior_probability,
    )

    inputs = [
        args.parsed_inputs / "table_bininfo.csv.gz",
        args.parsed_inputs / "table_rdrbaf.csv.gz",
        args.purity,
        args.clones,
        args.cnv,
    ]
    if args.spot_components:
        inputs.append(args.spot_components)
    provenance = {
        "schema": SCHEMA,
        "model_name": args.model_name,
        "inputs": {str(path.resolve()): _sha256(path) for path in inputs},
        "phase_cn_orientation": "aligned_native_CalicoST_B_to_CalicoST_AB",
        "cnv_bins_matched": matched_bins,
        "cnv_bins_total": len(bins),
        "phase_observations_total": int(observations_per_spot.sum()),
        "phase_effective_observation_cap": 8.0,
        "prior_probability": args.prior_probability,
        "missing_purity_cnv_policy": "explicit_prior_marginalization",
        "calicost_in_sample_descriptive_evidence": True,
        "independent_validation": False,
        "production_inputs_include_ground_truth": False,
        "evaluation_uses_ground_truth": False,
        "posterior_is_frozen": True,
        "phase_scope": "CalicoST-native phased bins; SPARCAL candidate PS_BLOCK not used",
    }
    payload = {
        "schema": np.asarray(SCHEMA),
        "model_name": np.asarray(args.model_name),
        "provenance": np.asarray(json.dumps(provenance, sort_keys=True)),
        "barcodes": barcodes,
        "valid": valid,
        "phase_available": phase_available & valid,
        "purity_available": purity_available & valid,
        "tumor_probability": combined["tumor_probability"],
        "phase_cnv_log_bf": combined["phase_cnv_log_bf"],
        "purity_log_prior": combined["purity_log_prior"],
        "somatic_vaf_log_bf": combined["somatic_vaf_log_bf"],
        "molecular_log_bf": combined["molecular_log_bf"],
        "spatial_log_prior": combined["spatial_log_prior"],
        "information": information,
        "posterior_log_odds": combined["posterior_log_odds"],
        "phase_information": phase_information,
        "phase_observation_count": observations_per_spot,
        "purity_raw": purities,
        "clone_label": clones,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=args.output.name + ".",
            suffix=".npz",
            dir=args.output.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, args.output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"Wrote {args.output}: spots={len(barcodes)} valid={int(valid.sum())} "
        f"phase={int(phase_available.sum())} purity={int(purity_available.sum())} "
        f"CN bins={matched_bins}/{len(bins)}"
    )


if __name__ == "__main__":
    main()
