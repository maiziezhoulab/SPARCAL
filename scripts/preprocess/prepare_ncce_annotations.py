#!/usr/bin/env python3
"""Prepare NCCE spot metadata and SPARCAL-compatible copyKAT tumor labels.

The source annotations are intentionally read-only.  This script copies only
the normalized, per-in-tissue-spot metadata into ``data/ncce/<sample>/``:

* ``annotations/spot_metadata.tsv`` — spatial coordinates, cell type, and
  copyKAT call for every in-tissue spot;
* ``annotations/copykat_tumor_labels.tsv`` — the labelled subset in the
  ``BARCODES``/``Tumor`` format consumed by the SPARCAL spatial filter;
* ``annotations/input_qc.tsv`` — annotation coverage and source paths.

``Tumor`` is a hard copyKAT-derived probability: aneuploid=1 and diploid=0.
Unlabelled spots are deliberately omitted from the filter input rather than
being called normal.  The NCCE filter configuration uses the labelled-sample
mean as a neutral value for these unlabelled spots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SAMPLES = [
    "6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C",
    "11A", "11B", "11C", "14A", "14B", "14C",
]

SPACERANGER_ROOT = Path("/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs")
CELLTYPE_ROOT = Path("/data/maiziezhou_lab/Weiman/ST_CNV/data/JapanNCCE_finalCT")
COPYKAT_ROOT = Path("/data/maiziezhou_lab/Weiman/ST_CNV/copyKAT")


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def read_unique(frame: pd.DataFrame, barcode_col: str, path: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame[barcode_col] = frame[barcode_col].astype(str)
    duplicated = frame[barcode_col].duplicated(keep=False)
    if duplicated.any():
        examples = ", ".join(frame.loc[duplicated, barcode_col].head(5))
        raise ValueError(f"{path} has duplicate barcodes (for example: {examples})")
    return frame


def prepare_sample(sample: str, output_root: Path) -> dict[str, object]:
    outs = SPACERANGER_ROOT / f"NCCE_{sample}" / "outs"
    positions_path = outs / "spatial" / "tissue_positions.csv"
    celltype_path = CELLTYPE_ROOT / f"{sample}.csv"
    copykat_path = COPYKAT_ROOT / f"NCCE_{sample}" / "prediction_aligned.txt"
    required_paths = [positions_path, celltype_path, copykat_path, outs / "possorted_genome_bam.bam"]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing NCCE input: {path}")

    positions = pd.read_csv(positions_path)
    require_columns(positions, {"barcode", "in_tissue"}, positions_path)
    positions = read_unique(positions, "barcode", positions_path)
    positions = positions.loc[positions["in_tissue"].astype(int) == 1].copy()
    if positions.empty:
        raise ValueError(f"No in-tissue spots in {positions_path}")

    celltypes = pd.read_csv(celltype_path)
    require_columns(celltypes, {"pixel", "cell_type"}, celltype_path)
    celltypes = read_unique(celltypes[["pixel", "cell_type"]], "pixel", celltype_path)
    celltypes = celltypes.rename(columns={"pixel": "barcode"})

    copykat = pd.read_csv(copykat_path, sep="\t")
    require_columns(copykat, {"cell.names", "copykat.pred"}, copykat_path)
    copykat = read_unique(copykat[["cell.names", "copykat.pred"]], "cell.names", copykat_path)
    copykat = copykat.rename(columns={"cell.names": "barcode", "copykat.pred": "copykat_prediction"})
    copykat["copykat_prediction"] = copykat["copykat_prediction"].astype(str).str.strip().str.lower()
    # Some source runs retain copyKAT's cluster/low-confidence decoration
    # (for example ``c2:aneuploid:low.conf``).  It still carries the same
    # diploid/aneuploid tumor call, so preserve the original text in metadata
    # and normalize the class only for the SPARCAL tumor-probability column.
    copykat["copykat_class"] = copykat["copykat_prediction"].str.extract(r"(aneuploid|diploid)", expand=False)
    copykat["copykat_low_confidence"] = copykat["copykat_prediction"].str.contains("low.conf", regex=False)
    copykat["Tumor"] = copykat["copykat_class"].map({"aneuploid": 1.0, "diploid": 0.0})
    unknown_labels = sorted(copykat.loc[copykat["Tumor"].isna(), "copykat_prediction"].unique())
    if unknown_labels:
        raise ValueError(f"Unexpected copyKAT labels in {copykat_path}: {unknown_labels}")

    metadata = positions.merge(celltypes, on="barcode", how="left", validate="one_to_one")
    metadata = metadata.merge(copykat, on="barcode", how="left", validate="one_to_one")
    metadata = metadata.sort_values("barcode").reset_index(drop=True)

    annotation_dir = output_root / sample / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(annotation_dir / "spot_metadata.tsv", sep="\t", index=False)

    tumor_labels = metadata.loc[metadata["Tumor"].notna(), ["barcode", "Tumor"]].copy()
    tumor_labels = tumor_labels.rename(columns={"barcode": "BARCODES"})
    tumor_labels.to_csv(annotation_dir / "copykat_tumor_labels.tsv", sep="\t", index=False)

    return {
        "sample": sample,
        "in_tissue_spots": len(metadata),
        "celltype_annotated": int(metadata["cell_type"].notna().sum()),
        "copykat_annotated": len(tumor_labels),
        "copykat_aneuploid": int((tumor_labels["Tumor"] == 1).sum()),
        "celltype_coverage": metadata["cell_type"].notna().mean(),
        "copykat_coverage": metadata["Tumor"].notna().mean(),
        "space_ranger_outs": str(outs),
        "celltype_source": str(celltype_path),
        "copykat_source": str(copykat_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/ncce"),
                        help="Project-relative or absolute NCCE output root.")
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=SAMPLES,
                        help="Subset of samples to prepare (default: all matched samples).")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    rows = [prepare_sample(sample, output_root) for sample in args.samples]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    print(manifest[["sample", "in_tissue_spots", "celltype_annotated", "copykat_annotated", "copykat_aneuploid"]].to_string(index=False))
    print(f"Wrote {output_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
