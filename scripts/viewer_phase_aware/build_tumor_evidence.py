#!/usr/bin/env python3
"""Build a frozen, label-free SPARCALviewer tumor-evidence sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase_aware_evidence import (
    DEFAULT_COMPONENT_CAPS,
    aggregate_phase_evidence,
    combine_tumor_evidence,
    phase_cnv_log_bayes_factor,
)


SCHEMA = "SPARCAL_tumor_evidence_v1"
COMPONENTS = (
    "phase_cnv_log_bf",
    "purity_log_prior",
    "somatic_vaf_log_bf",
    "molecular_log_bf",
    "spatial_log_prior",
)
FORBIDDEN_COLUMNS = {
    "wes", "truth", "ground_truth", "silver_standard", "truth_class",
    "outcome", "is_positive", "class_label", "validated",
}


def _read(path: Path) -> pd.DataFrame:
    lowered = path.name.lower()
    separator = "," if lowered.endswith((".csv", ".csv.gz")) else "\t"
    return pd.read_csv(path, sep=separator)


def _reject_ground_truth_columns(frame: pd.DataFrame, source: str) -> None:
    found = FORBIDDEN_COLUMNS & {str(column).strip().lower() for column in frame.columns}
    if found:
        raise ValueError(f"{source} contains forbidden evaluation/GT columns: {sorted(found)}")


def _numeric(frame: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _bool(frame: pd.DataFrame, name: str, default: bool) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, bool)
    values = frame[name]
    if values.dtype == bool:
        return values.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    if not normalized.isin(allowed).all():
        raise ValueError(f"{name} must contain booleans")
    return normalized.isin({"true", "1", "yes"}).to_numpy(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spots", help="unique per-spot CSV/TSV; required column: barcode")
    parser.add_argument("output", help="new tumor_evidence.npz path (refuses overwrite)")
    parser.add_argument(
        "--phase-observations",
        help=(
            "optional long CSV/TSV with barcode,B,TOT,purity,copy_a,copy_b; "
            "multiple observations per barcode are allowed"
        ),
    )
    parser.add_argument(
        "--phase-cn-orientation", choices=("marginalize", "aligned"), default="marginalize",
        help="use aligned only when B counts and CN A/B share the same phase convention",
    )
    parser.add_argument("--prior-probability", type=float, default=0.20)
    parser.add_argument("--model-name", default="phase_cnv_purity_label_free_v1")
    parser.add_argument("--confirm-label-free", action="store_true", required=True)
    args = parser.parse_args()

    spots_path, output = Path(args.spots), Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    spots = _read(spots_path)
    _reject_ground_truth_columns(spots, "spots")
    if "barcode" not in spots:
        raise ValueError("spots table requires barcode")
    spots["barcode"] = spots.barcode.astype(str)
    if spots.barcode.duplicated().any():
        raise ValueError("spots table barcodes must be unique")

    components = {name: _numeric(spots, name, 0.0) for name in COMPONENTS}
    provided = np.zeros(len(spots), bool)
    for name in COMPONENTS:
        if name in spots:
            provided |= np.isfinite(pd.to_numeric(spots[name], errors="coerce"))
    information = (
        _numeric(spots, "information", 0.0)
        if "information" in spots
        else provided.astype(float)
    )
    if np.any(~np.isfinite(information)) or np.any((information < 0) | (information > 1)):
        raise ValueError("information must be finite in [0,1]")
    explicit_valid = "valid" in spots
    valid = _bool(spots, "valid", False)
    phase_available = _bool(spots, "phase_available", False)
    purity_available = _bool(spots, "purity_available", False)

    phase_source = None
    if args.phase_observations:
        phase_path = Path(args.phase_observations)
        observations = _read(phase_path)
        _reject_ground_truth_columns(observations, "phase observations")
        required = {"barcode", "B", "TOT", "purity", "copy_a", "copy_b"}
        missing = required - set(observations.columns)
        if missing:
            raise ValueError(f"phase observations missing columns: {sorted(missing)}")
        observations["barcode"] = observations.barcode.astype(str)
        orientation = np.full(
            len(observations), 1.0 if args.phase_cn_orientation == "aligned" else 0.5
        )
        scored = phase_cnv_log_bayes_factor(
            pd.to_numeric(observations.B, errors="coerce"),
            pd.to_numeric(observations.TOT, errors="coerce"),
            pd.to_numeric(observations.purity, errors="coerce"),
            pd.to_numeric(observations.copy_a, errors="coerce"),
            pd.to_numeric(observations.copy_b, errors="coerce"),
            orientation_probability=orientation,
        )
        aggregated = aggregate_phase_evidence(
            observations.barcode, scored["log_bf"], scored["information"]
        )
        index = {barcode: row for row, barcode in enumerate(spots.barcode)}
        for barcode, values in aggregated.items():
            row = index.get(barcode)
            if row is None:
                continue
            components["phase_cnv_log_bf"][row] = values["log_bf"]
            information[row] = max(information[row], values["information"])
        observed_phase = {
            str(barcode)
            for barcode, ok in zip(observations.barcode, scored["valid"])
            if ok
        }
        observed_purity = {
            str(barcode)
            for barcode, ok in zip(observations.barcode, scored["purity_available"])
            if ok
        }
        phase_available |= spots.barcode.isin(observed_phase).to_numpy()
        purity_available |= spots.barcode.isin(observed_purity).to_numpy()
        phase_source = str(phase_path.resolve())

    if not explicit_valid:
        valid = information > 0
    if not valid.any():
        raise ValueError("no valid spots: provide positive information or an explicit valid mask")
    if np.any(phase_available & ~valid) or np.any(purity_available & ~valid):
        raise ValueError("phase_available and purity_available must be subsets of valid")
    combined = combine_tumor_evidence(
        components["phase_cnv_log_bf"],
        components["purity_log_prior"],
        components["somatic_vaf_log_bf"],
        components["molecular_log_bf"],
        components["spatial_log_prior"],
        information,
        prior_probability=args.prior_probability,
    )
    provenance = {
        "schema": SCHEMA,
        "model_name": args.model_name,
        "spots_source": str(spots_path.resolve()),
        "phase_source": phase_source,
        "phase_cn_orientation": args.phase_cn_orientation,
        "prior_probability": args.prior_probability,
        "component_coefficients": {name: 1.0 for name in COMPONENTS},
        "component_caps": DEFAULT_COMPONENT_CAPS,
        "missing_component_policy": "zero_log_evidence",
        "missing_purity_cnv_policy": "explicit_prior_marginalization",
        "phase_scope": "local_or_upstream_validated; never chromosome-wide by PS alone",
        "production_inputs_include_ground_truth": False,
        "evaluation_uses_ground_truth": False,
        "posterior_is_frozen": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(SCHEMA),
        model_name=np.asarray(args.model_name),
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
        barcodes=spots.barcode.to_numpy(str),
        valid=valid,
        phase_available=phase_available,
        purity_available=purity_available,
        tumor_probability=combined["tumor_probability"],
        phase_cnv_log_bf=combined["phase_cnv_log_bf"],
        purity_log_prior=combined["purity_log_prior"],
        somatic_vaf_log_bf=combined["somatic_vaf_log_bf"],
        molecular_log_bf=combined["molecular_log_bf"],
        spatial_log_prior=combined["spatial_log_prior"],
        information=information,
    )
    print(
        f"Wrote {output}: spots={len(spots)} valid={int(valid.sum())} "
        f"phase={int(phase_available.sum())} purity={int(purity_available.sum())}"
    )


if __name__ == "__main__":
    main()
