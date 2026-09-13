#!/usr/bin/env python3
"""Compute coverage-conditioned spatial and molecular evidence from one cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.spatial_filter_rebuild import config


def binary(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    result = matrix.copy().tocsr()
    result.data = np.ones_like(result.data, dtype=np.uint8)
    result.eliminate_zeros()
    return result


def coverage_conditioned_spatial(
    presence: sparse.csr_matrix,
    coverage: sparse.csr_matrix,
    graph: sparse.csr_matrix,
) -> dict[str, np.ndarray]:
    """Analytic join-count enrichment within each locus's covered spot graph."""

    x = binary(presence)
    c = binary(coverage)
    graph = binary(graph)
    xw = x @ graph
    cw = c @ graph
    observed = 0.5 * np.asarray(xw.multiply(x).sum(axis=1)).ravel()
    covered_edges = 0.5 * np.asarray(cw.multiply(c).sum(axis=1)).ravel()
    positive_to_covered = np.asarray(xw.multiply(c).sum(axis=1)).ravel()
    positive_spots = np.diff(x.indptr).astype(float)
    covered_spots = np.diff(c.indptr).astype(float)
    pair_probability = np.divide(
        positive_spots * np.maximum(positive_spots - 1, 0),
        covered_spots * np.maximum(covered_spots - 1, 0),
        out=np.zeros_like(positive_spots),
        where=covered_spots >= 2,
    )
    expected = covered_edges * pair_probability
    variance = covered_edges * pair_probability * (1.0 - pair_probability)
    z_score = np.divide(
        observed - expected,
        np.sqrt(variance + 0.25),
        out=np.zeros_like(observed),
        where=covered_edges > 0,
    )
    log_enrichment = np.log((observed + 0.5) / (expected + 0.5))
    neighbor_precision = np.divide(
        2.0 * observed,
        positive_to_covered,
        out=np.zeros_like(observed),
        where=positive_to_covered > 0,
    )
    information = (
        (1.0 - np.exp(-positive_spots / 3.0))
        * (1.0 - np.exp(-covered_edges / 5.0))
    )
    return {
        "spatial_join_count": observed,
        "spatial_expected_join_count": expected,
        "spatial_covered_edges": covered_edges,
        "spatial_z": z_score,
        "spatial_log_enrichment": log_enrichment,
        "spatial_neighbor_precision": neighbor_precision,
        "spatial_information": information,
    }


def local_density(chromosomes, positions, radius: int) -> np.ndarray:
    """Number of candidate coordinates within +/- radius, including self."""

    chrom = np.asarray(chromosomes, dtype=str)
    pos = np.asarray(positions, dtype=np.int64)
    output = np.ones(len(pos), dtype=np.int32)
    for value in np.unique(chrom):
        indices = np.flatnonzero(chrom == value)
        order = indices[np.argsort(pos[indices], kind="stable")]
        ordered = pos[order]
        left = np.searchsorted(ordered, ordered - radius, side="left")
        right = np.searchsorted(ordered, ordered + radius, side="right")
        output[order] = right - left
    return output


def clone_enrichment(
    presence: sparse.csr_matrix,
    coverage: sparse.csr_matrix,
    barcodes: list[str],
    clone_path: Path,
) -> dict[str, np.ndarray]:
    clones = pd.read_csv(clone_path, sep="\t")
    clones["barcode"] = clones["BARCODES"].astype(str).str.split("_").str[0]
    mapping = dict(zip(clones["barcode"], pd.to_numeric(clones["clone_label"], errors="coerce")))
    labels = np.asarray([mapping.get(barcode, np.nan) for barcode in barcodes], dtype=float)
    tumor = np.isfinite(labels) & (labels > 0)
    normal = np.isfinite(labels) & (labels == 0)
    x = binary(presence)
    c = binary(coverage)

    def sums(matrix, mask):
        if not np.any(mask):
            return np.zeros(matrix.shape[0], dtype=float)
        return np.asarray(matrix[:, mask].sum(axis=1)).ravel().astype(float)

    tp, tc = sums(x, tumor), sums(c, tumor)
    np_, nc = sums(x, normal), sums(c, normal)
    log_odds = (
        np.log((tp + 0.5) / (np.maximum(tc - tp, 0) + 0.5))
        - np.log((np_ + 0.5) / (np.maximum(nc - np_, 0) + 0.5))
    )
    information = (1.0 - np.exp(-(tc + nc) / 10.0)) * (tc > 0) * (nc > 0)
    return {
        "clone_tumor_positive": tp,
        "clone_tumor_covered": tc,
        "clone_normal_positive": np_,
        "clone_normal_covered": nc,
        "clone_log_odds": log_odds,
        "clone_information": information,
    }


def compute(section: str, cache_dir: Path, output_path: Path) -> None:
    started = time.time()
    candidates = pd.read_csv(cache_dir / "candidates.tsv.gz", sep="\t", low_memory=False)
    spots = pd.read_csv(cache_dir / "spots.tsv", sep="\t")
    depth = sparse.load_npz(cache_dir / "depth.npz").tocsr()
    presence = sparse.load_npz(cache_dir / "presence.npz").tocsr()
    graph = sparse.load_npz(cache_dir / "viewer_graph.npz").tocsr()
    if depth.shape != presence.shape or depth.shape[0] != len(candidates):
        raise ValueError("cache matrix/candidate shape mismatch")

    spatial = coverage_conditioned_spatial(presence, depth, graph)
    for name, values in spatial.items():
        candidates[name] = values
    clones = clone_enrichment(
        presence, depth, spots.sort_values("spot_index")["barcode"].astype(str).tolist(),
        Path(config.SECTIONS[section]["clones"]),
    )
    for name, values in clones.items():
        candidates[name] = values

    split = candidates["variant"].astype(str).str.split("_", n=1, expand=True)
    chromosomes = split[0].str.removeprefix("chr")
    positions = pd.to_numeric(split[1], errors="raise").astype(np.int64)
    for radius in (1, 5, 10, 50):
        candidates[f"local_density_{radius}bp"] = local_density(chromosomes, positions, radius)
    candidates["rna_editing_change"] = (
        (candidates["ref"].eq("A") & candidates["alt"].eq("G"))
        | (candidates["ref"].eq("T") & candidates["alt"].eq("C"))
    )
    candidates["xmhc"] = (
        chromosomes.eq("6") & positions.between(28_000_000, 34_000_000)
    )
    candidates["snp"] = candidates["ref"].astype(str).str.len().eq(1) & candidates["alt"].astype(str).str.len().eq(1)
    candidates["coverage_prevalence"] = np.divide(
        candidates["positive_spots"], candidates["covered_spots_cache"],
        out=np.zeros(len(candidates), dtype=float),
        where=candidates["covered_spots_cache"].to_numpy() > 0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, sep="\t", index=False)
    manifest = {
        "section": section,
        "n_candidates": len(candidates),
        "n_alt_observed": int((candidates["total_alt"] > 0).sum()),
        "n_spatial_informative": int((candidates["spatial_information"] > 0).sum()),
        "n_clone_informative": int((candidates["clone_information"] > 0).sum()),
        "elapsed_seconds": time.time() - started,
        "gt_used": False,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cache = args.cache_dir or config.cache_dir(args.section)
    output = args.output or config.score_dir(args.section) / "features.tsv.gz"
    if output.exists():
        raise FileExistsError(f"feature output exists: {output}")
    compute(args.section, cache, output)


if __name__ == "__main__":
    main()
