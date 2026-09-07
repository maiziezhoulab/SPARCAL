#!/usr/bin/env python3
"""Finish CalicoST LOH phylogeny and spatial projection for P4/P6 hg19 runs.

Startle is required when the observed LOH matrix needs homoplasy correction.
For the current outputs, three sections contain a single realized clone and the
only two-clone section has an exact perfect phylogeny, so no optimization solver
is needed. The method is recorded beside every result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
CALICOST_CODE = Path("/data/maiziezhou_lab/leiy4/CalicoST")
OUTPUT_ROOT = CALICOST_CODE / "hg19_rerun_20260906"
MANIFEST = PROJECT_ROOT / "run_slurm" / "calicost_hg19" / "samples.tsv"
SAMPLE_CONFIG = {
    "P4_sec1": ("clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P4_sec2": ("clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P6_sec1": ("clone3_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P6_sec2": ("clone3_rectangle0_w1.0", "cnv_diploid_seglevel.tsv"),
}


def clean_barcode(value: object) -> str:
    barcode = str(value)
    if "_" in barcode:
        return barcode.rsplit("_", 1)[0]
    return barcode


def read_positions(outs: Path) -> pd.DataFrame:
    path = outs / "spatial" / "tissue_positions.csv"
    columns = [
        "barcode", "in_tissue", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres",
    ]
    with path.open() as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("barcode,"):
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, header=None, names=columns)
    frame["barcode"] = frame["barcode"].astype(str)
    return frame[frame["in_tissue"].astype(int) == 1].copy()


def realized_clone_table(labels_path: Path, positions: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, sep="\t")
    labels = labels[labels["BARCODES"].astype(str) != "BARCODES"].copy()
    labels["barcode"] = labels["BARCODES"].map(clean_barcode)
    labels["clone_label"] = pd.to_numeric(labels["clone_label"], errors="raise").astype(int)
    labels["clone_name"] = labels["clone_label"].map(lambda value: f"clone{value}")
    merged = positions.merge(
        labels[["barcode", "clone_label", "clone_name"]], on="barcode", how="inner"
    )
    if merged.empty:
        raise ValueError(f"No shared barcodes between {labels_path} and tissue positions")
    return merged


def infer_tree(
    clone_names: list,
    binary: pd.DataFrame,
    phylogeny_module,
) -> Tuple[str, str]:
    if len(clone_names) == 1:
        return f"{clone_names[0]};", "singleton_no_inference_required"

    binary = binary.reindex(clone_names).fillna(0).astype(int)
    try:
        _, cell_tree = phylogeny_module.generate_perfect_phylogeny(binary)
    except Exception as exc:
        raise RuntimeError(
            "The realized LOH matrix is not a perfect phylogeny. Run the official "
            "Startle solver with CPLEX before spatial projection."
        ) from exc
    newick = phylogeny_module.tree_to_newick(cell_tree) + ";"
    return newick, "calicost_exact_perfect_phylogeny_on_observed_loh"


def node_locations(
    newick_path: Path,
    merged: pd.DataFrame,
    clone_names: list,
    phylogeography_module,
) -> Tuple[pd.DataFrame, object]:
    if len(clone_names) == 1:
        center_x = float(merged["pxl_col_in_fullres"].mean())
        center_y = float(merged["pxl_row_in_fullres"].mean())
        nodes = pd.DataFrame([{
            "node": clone_names[0],
            "is_leaf": True,
            "x_fullres": center_x,
            "y_fullres": center_y,
        }])
        return nodes, None

    coords = merged[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    tree = phylogeography_module.project_phylogeneny_space(
        str(newick_path), coords, merged["clone_name"].to_numpy()
    )
    records = [{
        "node": node.name,
        "is_leaf": bool(node.is_leaf()),
        "x_fullres": float(node.x),
        "y_fullres": float(node.y),
    } for node in tree.traverse()]
    return pd.DataFrame(records), tree


def plot_spatial_tree(
    sample: str,
    outs: Path,
    merged: pd.DataFrame,
    nodes: pd.DataFrame,
    tree: object,
    output_dir: Path,
) -> None:
    spatial = outs / "spatial"
    scale = json.loads((spatial / "scalefactors_json.json").read_text())["tissue_hires_scalef"]
    image = plt.imread(spatial / "tissue_hires_image.png")
    nodes = nodes.copy()
    nodes["x_hires"] = nodes["x_fullres"] * scale
    nodes["y_hires"] = nodes["y_fullres"] * scale
    nodes.to_csv(output_dir / "phylogeography_nodes.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(image)
    clone_names = sorted(merged["clone_name"].unique())
    colors = dict(zip(clone_names, plt.cm.tab10(np.linspace(0, 1, len(clone_names)))))
    for clone_name, group in merged.groupby("clone_name"):
        ax.scatter(
            group["pxl_col_in_fullres"] * scale,
            group["pxl_row_in_fullres"] * scale,
            s=12, alpha=0.55, color=colors[clone_name], label=clone_name,
        )

    location = nodes.set_index("node")
    if tree is not None:
        for node in tree.traverse():
            if node.is_root():
                continue
            parent = node.up
            ax.annotate(
                "",
                xy=(location.loc[node.name, "x_hires"], location.loc[node.name, "y_hires"]),
                xytext=(location.loc[parent.name, "x_hires"], location.loc[parent.name, "y_hires"]),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
            )

    for record in nodes.itertuples(index=False):
        ax.scatter(
            record.x_hires, record.y_hires, marker="D", s=70,
            facecolor="white", edgecolor="black", linewidth=1.5, zorder=5,
        )
        ax.text(record.x_hires + 5, record.y_hires - 5, record.node, fontsize=8)

    ax.set_title(f"{sample}: CalicoST LOH phylogeography")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "phylogeography.pdf")
    fig.savefig(output_dir / "phylogeography.png", dpi=200)
    plt.close(fig)


def process_sample(
    row: pd.Series,
    phylogeny_module,
    phylogeography_module,
) -> Dict[str, object]:
    sample = str(row["sample"])
    clone_dir_name, cnv_name = SAMPLE_CONFIG[sample]
    clone_dir = OUTPUT_ROOT / sample / "calicost" / clone_dir_name
    cnv_path = clone_dir / cnv_name
    labels_path = clone_dir / "clone_labels.tsv"
    outs = Path(str(row["spaceranger_outs"]))
    output_dir = OUTPUT_ROOT / sample / "calicost" / f"phylogeny_{clone_dir_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = read_positions(outs)
    merged = realized_clone_table(labels_path, positions)
    clone_names = sorted(merged["clone_name"].unique())
    cnv = pd.read_csv(cnv_path, sep="\t")
    loh = phylogeny_module.get_LoH_for_phylogeny(cnv, min_segments=3)
    binary = phylogeny_module.get_binary_matrix(loh)
    loh.to_csv(output_dir / "loh_matrix.tsv", sep="\t")
    binary.to_csv(output_dir / "loh_binary_matrix.tsv", sep="\t")

    newick, method = infer_tree(clone_names, binary, phylogeny_module)
    newick_path = output_dir / "loh_tree.newick"
    newick_path.write_text(newick + "\n")
    nodes, tree = node_locations(
        newick_path, merged, clone_names, phylogeography_module
    )
    plot_spatial_tree(sample, outs, merged, nodes, tree, output_dir)

    metadata = {
        "sample": sample,
        "requested_clone_count": int(
            clone_dir_name.removeprefix("clone").split("_", 1)[0]
        ),
        "realized_clones": clone_names,
        "realized_clone_count": len(clone_names),
        "tree_kind": "branching" if len(clone_names) > 1 else "singleton",
        "method": method,
        "startle_used": False,
        "startle_reason": (
            "not required for singleton or exact perfect LOH phylogeny"
        ),
        "cnv_source": str(cnv_path),
        "spatial_spots": int(len(merged)),
        "loh_characters": int(loh.shape[1]),
        "binary_characters": int(binary.shape[1]),
        "newick": newick,
    }
    (output_dir / "METHOD.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLE_CONFIG))
    args = parser.parse_args()
    unknown = sorted(set(args.samples) - set(SAMPLE_CONFIG))
    if unknown:
        parser.error(f"Unknown sample(s): {', '.join(unknown)}")

    sys.path.insert(0, str(CALICOST_CODE / "src"))
    import calicost.phylogeography as phylogeography
    import calicost.phylogeny_startle as phylogeny

    manifest = pd.read_csv(MANIFEST, sep="\t").set_index("sample", drop=False)
    results = [
        process_sample(manifest.loc[sample], phylogeny, phylogeography)
        for sample in args.samples
    ]
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_ROOT / "phylogeography_summary.tsv", sep="\t", index=False)
    print(summary[[
        "sample", "requested_clone_count", "realized_clone_count",
        "tree_kind", "method", "newick",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
