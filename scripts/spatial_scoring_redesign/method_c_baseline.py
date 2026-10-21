#!/usr/bin/env python
"""
method_c_baseline.py
=====================
Method C = the shipped vote-quota cascade, AS-IS. No recomputation: Stage 2's
per-variant vote fraction and final classification are already persisted in
all_variant_scores.txt (see data/somatic_calibration_2026-08-23/RESULTS.md,
which verified floor(n_nongermline*0.10) reproduces the shipped somatic count
bit-for-bit). This script just reads that straight out of the candidates
cache built by build_inputs.py (which itself is a read-only re-derivation of
the same table) and writes it out in this project's own selected-set schema.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def run_section(section):
    outdir = common.cache_dir(section)
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t")
    selected = candidates[candidates["shipped_classification"] == "somatic"].copy()
    selected = selected.sort_values("shipped_vote_fraction", ascending=False)

    sel_dir = os.path.join(common.OUT_ROOT, "selected_sets")
    os.makedirs(sel_dir, exist_ok=True)
    out_path = os.path.join(sel_dir, f"{section}_methodC_shipped.tsv")
    selected[["variant", "chrom", "pos", "shipped_vote_fraction", "n_pos_spots"]].to_csv(
        out_path, sep="\t", index=False)

    print(f"{section}: Method C (shipped quota) selected {len(selected)} / {len(candidates)} "
          f"candidates ({100.0 * len(selected) / len(candidates):.2f}%) -> {out_path}")
    return selected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    args = ap.parse_args()
    run_section(args.section)
