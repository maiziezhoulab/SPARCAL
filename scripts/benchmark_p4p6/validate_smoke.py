#!/usr/bin/env python3
from __future__ import annotations
import gzip
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling/data/p4p6_caller_panel_2026-10-06")
SMOKE = ROOT / "_smoke_chr21"
SECTION = "P4_rep2"


def vcf_counts(path: Path) -> dict:
    out = {"records": 0, "pass": 0, "pass_biallelic_snvs": 0}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            out["records"] += 1
            f = line.rstrip("\n").split("\t")
            if f[6] == "PASS":
                out["pass"] += 1
                if len(f[3]) == len(f[4]) == 1 and "," not in f[4]:
                    out["pass_biallelic_snvs"] += 1
    return out


def matrix_stats(path: Path) -> dict:
    frame = pd.read_pickle(path)
    values = frame.to_numpy(copy=False)
    valid_keys = all(not str(c).startswith("chr") and len(str(c).rsplit("_", 3)) == 4 for c in frame.columns)
    return {
        "shape": list(frame.shape), "presences": int(values.sum()),
        "all_uint8": all(str(x) == "uint8" for x in frame.dtypes),
        "unique_rows": bool(frame.index.is_unique), "unique_columns": bool(frame.columns.is_unique),
        "key_contract": valid_keys,
    }


def main():
    result = {"section": SECTION, "region": "chr21", "status": "PASS", "callers": {}}
    expected_rows = None
    for caller in ("strelka2", "gatk"):
        directory = SMOKE / SECTION / caller
        pass_vcf = directory / f"{SECTION}_{caller}_pass_snvs.vcf.gz"
        matrix = directory / f"{SECTION}_{caller}_presence_matrix.pkl"
        no1kgp = directory / f"{SECTION}_{caller}_no1kgp_presence_matrix.pkl"
        raw_stats, clean_stats = matrix_stats(matrix), matrix_stats(no1kgp)
        if expected_rows is None:
            expected_rows = raw_stats["shape"][0]
        checks = {
            "nonzero_vcf": vcf_counts(pass_vcf)["pass_biallelic_snvs"] > 0,
            "nonzero_matrix": raw_stats["shape"][1] > 0,
            "same_spot_count": raw_stats["shape"][0] == expected_rows == clean_stats["shape"][0],
            "no1kgp_subset": clean_stats["shape"][1] <= raw_stats["shape"][1],
            "uint8": raw_stats["all_uint8"] and clean_stats["all_uint8"],
            "key_contract": raw_stats["key_contract"] and clean_stats["key_contract"],
        }
        result["callers"][caller] = {
            "pass_vcf": vcf_counts(pass_vcf), "matrix": raw_stats,
            "no1kgp_matrix": clean_stats, "checks": checks,
        }
        if not all(checks.values()):
            result["status"] = "FAIL"

    spatial_dir = SMOKE / SECTION / "spatialsnv"
    result["callers"]["spatialsnv"] = {
        "chr21_vcf": vcf_counts(spatial_dir / "P4_sec2.chr21.vcf.gz"),
        "presence_shape": list(pd.read_pickle(spatial_dir / f"{SECTION}_spatialsnv_presence_matrix.pkl").shape),
        "filtered_shape": list(pd.read_pickle(spatial_dir / f"{SECTION}_spatialsnv_presence_filtered_matrix.pkl").shape),
    }
    rep1 = json.loads((SMOKE / "_spatialsnv_rep1_builder_verification/verification.json").read_text())
    result["spatialsnv_rep1_reproduction"] = rep1
    if not all(rep1[k] for k in ("shape_match", "index_equal", "columns_equal", "values_equal", "dtype_all_uint8")):
        result["status"] = "FAIL"
    output = SMOKE / "smoke_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit("Smoke validation failed")


if __name__ == "__main__":
    main()
