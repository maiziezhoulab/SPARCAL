#!/usr/bin/env python3
"""Path adapter around the exact established SpatialSNV CallBack builder."""
import argparse
import importlib.util
from pathlib import Path

SOURCE = Path("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/callback_to_pkl.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("established_callback_to_pkl", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Defaults 1/1 reproduce the established FILTERED = PASS intersect biallelic-SNV rule.
    module.build(args.matrix_dir, args.sample, args.vcf, args.outdir, 1, 1)


if __name__ == "__main__":
    main()
