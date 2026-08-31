#!/usr/bin/env python3
"""dlpfc_density_confound.py (2026-08-28) -- P0-1 density confound table.

Computes nonzero-fraction and row-sum summary stats for every DLPFC matrix
relevant to the P0-1 negative-control battery: the real somtop25_bin250kb
representation and all five controls (coverage_only, detection_only,
allele_permuted, smoothed_random, detection_downsampled), across all 12
sections. This is the source table for fig_dlpfc_negative_controls.py's new
density panel, and is also how the detection_only-vs-real density confound
(3.8-5.9x denser, 4.9-6.5x more signal per spot across sections) was
verified for the fig-dlpfc-negative-controls.md dossier.

Read-only against data/dlpfc/ and data/dlpfc_negative_controls_2026-08-27/
(including its density_matched/ subdir). Writes only
data/dlpfc_negative_controls_2026-08-27/density_matched/matrix_density_comparison.csv.

Usage:
    python scripts/postanalyze/dlpfc_density_confound.py
"""
import os
import numpy as np
import pandas as pd

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
REAL_BASE = os.path.join(PROJECT, "data", "dlpfc")
CTRL_BASE = os.path.join(PROJECT, "data", "dlpfc_negative_controls_2026-08-27")
DM_BASE = os.path.join(CTRL_BASE, "density_matched")

SECTIONS = ["151507", "151508", "151509", "151510", "151669", "151670",
            "151671", "151672", "151673", "151674", "151675", "151676"]
DONOR = {}
for s in ("151507", "151508", "151509", "151510"):
    DONOR[s] = "Br5292"
for s in ("151669", "151670", "151671", "151672"):
    DONOR[s] = "Br5595"
for s in ("151673", "151674", "151675", "151676"):
    DONOR[s] = "Br8100"

MODALITIES = {
    "real_somtop25_bin250kb": (REAL_BASE, "DLPFC_{s}_SPARCAL_somtop25_bin250kb_matrix.pkl"),
    "coverage_only": (CTRL_BASE, "DLPFC_{s}_SPARCAL_coverage_only_matrix.pkl"),
    "detection_only": (CTRL_BASE, "DLPFC_{s}_SPARCAL_detection_only_matrix.pkl"),
    "allele_permuted": (CTRL_BASE, "DLPFC_{s}_SPARCAL_allele_permuted_matrix.pkl"),
    "smoothed_random": (CTRL_BASE, "DLPFC_{s}_SPARCAL_smoothed_random_matrix.pkl"),
    "detection_downsampled": (DM_BASE, "DLPFC_{s}_SPARCAL_detection_downsampled_matrix.pkl"),
}


def main():
    rows = []
    for s in SECTIONS:
        for mod, (base, tmpl) in MODALITIES.items():
            path = os.path.join(base, s, "matrix", tmpl.format(s=s))
            df = pd.read_pickle(path)
            v = df.values.astype(np.float64)
            nz = v > 0
            row_sums = v.sum(axis=1)
            rows.append(dict(
                section=s, donor=DONOR[s], modality=mod,
                n_spots=v.shape[0], n_bins=v.shape[1],
                nonzero_frac=float(nz.mean()),
                mean_row_sum=float(row_sums.mean()),
                median_row_sum=float(np.median(row_sums)),
                p25_row_sum=float(np.quantile(row_sums, 0.25)),
                p75_row_sum=float(np.quantile(row_sums, 0.75)),
                min_row_sum=float(row_sums.min()),
                max_row_sum=float(row_sums.max()),
            ))
            print(f"{s:8} {mod:26} nz_frac={nz.mean():.4f}  mean_row_sum={row_sums.mean():.1f}")

    out = pd.DataFrame(rows)
    os.makedirs(DM_BASE, exist_ok=True)
    out_path = os.path.join(DM_BASE, "matrix_density_comparison.csv")
    out.to_csv(out_path, index=False)
    print("wrote", out_path)

    real = out[out.modality.eq("real_somtop25_bin250kb")].set_index("section")
    det = out[out.modality.eq("detection_only")].set_index("section")
    ratio_nz = (det.nonzero_frac / real.nonzero_frac)
    ratio_sum = (det.mean_row_sum / real.mean_row_sum)
    print(f"\ndetection_only/real nonzero_frac ratio: {ratio_nz.min():.2f}-{ratio_nz.max():.2f}x")
    print(f"detection_only/real mean_row_sum ratio: {ratio_sum.min():.2f}-{ratio_sum.max():.2f}x")


if __name__ == "__main__":
    main()
