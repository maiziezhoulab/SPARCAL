#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--existing", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    rerun = pd.read_pickle(args.rerun)
    existing = pd.read_pickle(args.existing)
    result = {
        "rerun_shape": list(rerun.shape),
        "existing_shape": list(existing.shape),
        "expected_shape": [4855, 45038],
        "shape_match": rerun.shape == existing.shape == (4855, 45038),
        "index_equal": rerun.index.equals(existing.index),
        "columns_equal": rerun.columns.equals(existing.columns),
        "values_equal": rerun.equals(existing),
        "dtype_all_uint8": all(str(dtype) == "uint8" for dtype in rerun.dtypes),
    }
    Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(result[key] for key in ("shape_match", "index_equal", "columns_equal", "values_equal", "dtype_all_uint8")):
        raise SystemExit("rep1 SpatialSNV builder reproduction failed")


if __name__ == "__main__":
    main()
