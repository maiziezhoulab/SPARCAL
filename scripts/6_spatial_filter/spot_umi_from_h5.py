#!/usr/bin/env python3
"""Extract per-barcode total UMI counts from a 10x filtered matrix HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def spot_umi_from_h5(path: str) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = np.asarray(matrix["data"])
        indptr = np.asarray(matrix["indptr"], dtype=np.int64)
        barcodes = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in matrix["barcodes"][:]
        ]
    if len(indptr) != len(barcodes) + 1:
        raise ValueError("invalid 10x CSC matrix: indptr/barcode dimensions disagree")
    cumulative = np.concatenate(([0], np.cumsum(data, dtype=np.int64)))
    totals = cumulative[indptr[1:]] - cumulative[indptr[:-1]]
    return pd.DataFrame({"barcode": barcodes, "total_umi": totals})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    spot_umi_from_h5(args.h5).to_csv(output, sep="\t", index=False, mode="x")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
