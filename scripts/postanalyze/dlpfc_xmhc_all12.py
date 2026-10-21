#!/usr/bin/env python3
"""Extend the 4-section DLPFC xMHC enrichment calculation to all 12 sections.

Reads the SPARCAL "somatic" (spatially-clustered) class spot x SNV matrix for
each of the 12 DLPFC sections and counts how many of its called variants fall
inside the extended MHC (chr6:28,000,000-34,000,000, GRCh38 coordinates,
matching cosmic_mhc_partition.py / fig4ab_cosmic_xmhc_v4.py convention).

Read-only on data/dlpfc/*/matrix/*.pkl. Writes only under --out-dir.

CAVEAT (record plainly, do not treat this as an exact null): in DLPFC there is
no tumour-purity file, so the spatial filter's delta (purity-correlation) and
epsilon (clone-specific) descriptors are identically zero / degenerate for
every spot, and the "somatic" (spatially-clustered) class is effectively
selected by the zeta (pure spatial-clustering) descriptor alone. In the tumour
sections (P4/P6/DCIS1/DCIS2) all 3-4 descriptors (gamma, delta, epsilon, zeta)
contribute. The DLPFC construct is therefore not an exact null model for the
tumour pipeline's class -- it is a normal-tissue run of the SAME spatial-
clustering machinery, missing the purity terms.

Run (env snv_caller):
    python scripts/postanalyze/dlpfc_xmhc_all12.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")
SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]
MHC_CHROM = "6"
MHC_LO, MHC_HI = 28_000_000, 34_000_000
GENOME_LEN_BP = 3_088_269_832  # GRCh38 primary-assembly autosomes+X+Y+MT, matches
# the "0.194% of the genome" figure quoted in the task (6,000,000 / 3,088,269,832
# would give 0.1943%, i.e. this constant reproduces the caller's own number).
XMHC_LEN_BP = MHC_HI - MHC_LO
GENOME_FRAC_PCT = 100.0 * XMHC_LEN_BP / GENOME_LEN_BP


def matrix_path(section: str) -> Path:
    return REPO / f"data/dlpfc/{section}/matrix/DLPFC_{section}_SPARCAL_somatic_matrix.pkl"


def count_section(section: str) -> dict:
    path = matrix_path(section)
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    cols = list(df.columns)
    n_calls = len(cols)
    n_xmhc = 0
    for c in cols:
        chrom, pos = c.split("_", 1)
        if chrom == MHC_CHROM:
            try:
                p = int(pos)
            except ValueError:
                continue
            if MHC_LO <= p <= MHC_HI:
                n_xmhc += 1
    pct = 100.0 * n_xmhc / n_calls if n_calls else float("nan")
    fold = pct / GENOME_FRAC_PCT if GENOME_FRAC_PCT else float("nan")
    return dict(section=section, n_calls=n_calls, n_xmhc=n_xmhc, pct_xmhc=pct,
                fold_enrichment=fold, matrix_path=str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [count_section(s) for s in SECTIONS]
    df = pd.DataFrame(rows)
    out_path = out_dir / "dlpfc_xmhc_all12.csv"
    df.to_csv(out_path, index=False)

    print(f"genome-length expectation used: {GENOME_FRAC_PCT:.4f}% "
          f"({XMHC_LEN_BP:,} bp / {GENOME_LEN_BP:,} bp)")
    print(df.to_string(index=False))
    print(f"\nWROTE {out_path}")
    print(f"\nsummary: n={len(df)} sections, pct_xmhc range "
          f"[{df.pct_xmhc.min():.3f}, {df.pct_xmhc.max():.3f}]%, "
          f"fold range [{df.fold_enrichment.min():.2f}, {df.fold_enrichment.max():.2f}]x, "
          f"mean fold {df.fold_enrichment.mean():.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
