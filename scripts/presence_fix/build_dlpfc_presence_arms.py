#!/usr/bin/env python3
"""Build DLPFC presence-rule arms for the germline clustering benchmark (2026-10-09).

G0 (reference, not rebuilt) = DLPFC_{s}_SPARCAL_in_panel_germline_matrix.pkl from the 2026-09-19 frozen Beagle
experiment: Beagle-routed in-panel germline loci; a spot is 1 wherever step 6 wrote a record, i.e. wherever a
read covered the locus, whatever base it carried.

  G1  same values as G0, but keep only loci with >= 1 ALT molecule in at least one in-tissue spot of the section
      ("a germline variant the individual carries is present wherever the locus is observed")
  G2  same loci as G0; a spot is 1 only if it has >= 1 ALT molecule at that locus
Molecules come from scripts/presence_fix/recount_spot_alleles.py (UMI-collapsed, primary alignments).

Each arm is also binned at 250 kb with the 2026-09-19 rule (sum of binary presence per bin). Comparator matrices
(GATK, Strelka2, Strelka2-RNA) are symlinked unbinned and rebinned with the same rule, so every binned modality
shares one binning function. Rows and row order are kept from each source matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
BEAGLE = PROJECT / "data/experiments/beagle_default_dlpfc_20260919_v3_orderfix"
LEGACY = PROJECT / "data/dlpfc"
COMPARATORS = {  # modality file stem -> unbinned source in data/dlpfc/{s}/matrix
    "gatk_germline_6": "DLPFC_{s}_gatk_germline_6_matrix.pkl",
    "strelka2_germline_6": "DLPFC_{s}_strelka2_germline_6_matrix.pkl",
    "strelka2rna_germline_6": "DLPFC_{s}_strelka2rna_germline_6_matrix.pkl",
}


def bin_matrix(matrix: pd.DataFrame, bp: int = 250_000) -> pd.DataFrame:
    """Copied from beagle_default_dlpfc_2026_09_19/experiment.py:bin_matrix (binary presence summed per bin)."""
    keys = np.array([f"{c.rsplit('_', 1)[0]}_{(int(c.rsplit('_', 1)[1]) // bp) * bp}" for c in matrix.columns])
    unique, inverse = np.unique(keys, return_inverse=True)
    result = np.zeros((len(unique), len(matrix)), dtype=np.int32)
    np.add.at(result, inverse, (matrix.to_numpy().T > 0).astype(np.int32))
    return pd.DataFrame(result.T, index=matrix.index, columns=unique)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True)
    ap.add_argument("--root", required=True, type=Path)
    a = ap.parse_args()
    s = a.section
    mdir = a.root / s / "matrix"
    mdir.mkdir(parents=True, exist_ok=True)

    g0 = pd.read_pickle(BEAGLE / s / "matrix" / f"DLPFC_{s}_SPARCAL_in_panel_germline_matrix.pkl")
    rc = pd.read_csv(PROJECT / f"data/presence_fix_2026-10-09/recount/DLPFC_{s}.tsv.gz", sep="\t",
                     usecols=["barcode", "chrom", "pos", "alt_mol", "alt_read"], dtype={"chrom": str})
    rc["key"] = rc["chrom"].str.replace("chr", "", regex=False) + "_" + rc["pos"].astype(str)
    alt = rc[rc["alt_mol"] > 0][["barcode", "key"]].drop_duplicates()

    cols = pd.Index([str(c) for c in g0.columns])
    g0.columns = cols
    col_idx = pd.Series(np.arange(len(cols)), index=cols)
    row_idx = pd.Series(np.arange(len(g0)), index=g0.index)
    alt = alt[alt["barcode"].isin(row_idx.index) & alt["key"].isin(col_idx.index)]

    g2v = np.zeros(g0.shape, dtype=np.int8)
    g2v[row_idx[alt["barcode"]].to_numpy(), col_idx[alt["key"]].to_numpy()] = 1
    g0v = (g0.to_numpy() > 0)
    if (g2v.astype(bool) & ~g0v).any():
        raise RuntimeError("ALT molecule observed where G0 has no record; recount and G0 disagree")
    g2 = pd.DataFrame(g2v, index=g0.index, columns=cols)
    keep = g2v.any(axis=0)
    g1 = g0.loc[:, keep]

    out = {}
    for name, m in (("SPARCAL_g1", g1), ("SPARCAL_g2", g2)):
        m.to_pickle(mdir / f"DLPFC_{s}_{name}_matrix.pkl")
        bin_matrix(m).to_pickle(mdir / f"DLPFC_{s}_{name}_bin250kb_matrix.pkl")
        out[name] = list(m.shape)
    for stem, src in COMPARATORS.items():
        p = LEGACY / s / "matrix" / src.format(s=s)
        link = mdir / f"DLPFC_{s}_{stem}_matrix.pkl"
        if not link.exists():
            os.symlink(p, link)
        m = pd.read_pickle(p)
        m.columns = [str(c).replace("chr", "") for c in m.columns]
        caller, filt, grouping = stem.split("_")
        bin_matrix(m).to_pickle(mdir / f"DLPFC_{s}_{caller}_{filt}_bin250kb_{grouping}_matrix.pkl")
        out[stem] = list(m.shape)

    stats = {
        "section": s,
        "g0_shape": list(g0.shape),
        "g0_ones": int(g0v.sum()),
        "g0_ones_with_alt_molecule": int(g2v.sum()),
        "frac_g0_ones_with_alt_molecule": float(g2v.sum() / g0v.sum()),
        "g0_loci_with_any_alt_molecule": int(keep.sum()),
        "frac_loci_kept_g1": float(keep.mean()),
        "shapes": out,
    }
    (a.root / s / "PRESENCE_ARMS.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
