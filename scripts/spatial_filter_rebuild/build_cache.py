#!/usr/bin/env python3
"""Build an all-class sparse AD/DP cache without touching production outputs.

Unlike the historical Stage-2 cache, this includes every candidate in
``all_variant_scores.txt``.  A row in a spot VCF is coverage, not ALT presence:
presence is strictly ``FORMAT/AD[1] > 0`` and covered depth is ``DP > 0``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from scripts.spatial_filter_rebuild import config


_VARIANT_INDEX: dict[str, int] | None = None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def chrom(value: object) -> str:
    text = str(value)
    return text[3:] if text.startswith("chr") else text


def parse_info(text: str) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for token in str(text).split(";"):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
        else:
            result[token] = True
    return result


def parse_counts(format_field: str, sample_field: str) -> tuple[int, int] | None:
    values = dict(zip(str(format_field).split(":"), str(sample_field).split(":")))
    ad = values.get("AD")
    if not ad or ad == ".":
        return None
    tokens = ad.split(",")
    if len(tokens) != 2:
        return None
    try:
        ref_count, alt_count = int(tokens[0]), int(tokens[1])
    except ValueError:
        return None
    if ref_count < 0 or alt_count < 0:
        return None
    ad_depth = ref_count + alt_count
    try:
        reported = int(values.get("DP", ad_depth))
    except (TypeError, ValueError):
        reported = ad_depth
    return max(ad_depth, reported, 0), alt_count


def pooled_metadata(path: Path, candidate_index: dict[str, int]) -> pd.DataFrame:
    """Read section-level INFO once; it is repeated in every spot VCF."""

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or "," in fields[4]:
                continue
            key = f"{chrom(fields[0])}_{fields[1]}"
            if key not in candidate_index or key in seen:
                continue
            seen.add(key)
            info = parse_info(fields[7])
            row: dict[str, object] = {
                "variant": key,
                "ref": fields[3].upper(),
                "alt": fields[4].upper(),
                "info_dp": pd.to_numeric(info.get("DP"), errors="coerce"),
                "info_af": pd.to_numeric(info.get("AF"), errors="coerce"),
                "info_race": info.get("RACE"),
                "nn_class": info.get("NEURAL_NETWORK_CLASS"),
                "nn_homo": pd.to_numeric(info.get("NEURAL_NETWORK_HOMO"), errors="coerce"),
                "nn_hetero": pd.to_numeric(info.get("NEURAL_NETWORK_HETERO"), errors="coerce"),
                "nn_novar": pd.to_numeric(info.get("NEURAL_NETWORK_NOVAR"), errors="coerce"),
            }
            for tag in ("VDB", "SGB", "RPB", "MQB", "MQSB", "BQB", "MQ0F"):
                row[f"info_{tag.lower()}"] = pd.to_numeric(info.get(tag), errors="coerce")
            i16_text = info.get("I16")
            try:
                i16 = [float(value) for value in str(i16_text).split(",")]
            except ValueError:
                i16 = []
            if len(i16) == 16:
                row.update(
                    ref_forward=i16[0], ref_reverse=i16[1],
                    alt_forward=i16[2], alt_reverse=i16[3],
                    ref_baseq_sum=i16[4], alt_baseq_sum=i16[6],
                    ref_mapq_sum=i16[8], alt_mapq_sum=i16[10],
                    ref_tail_sum=i16[12], alt_tail_sum=i16[14],
                )
            rows.append(row)
    return pd.DataFrame(rows)
METADATA_COLUMNS = (
    "variant", "ref", "alt", "info_dp", "info_af", "info_race", "nn_class",
    "nn_homo", "nn_hetero", "nn_novar", "info_vdb", "info_sgb", "info_rpb",
    "info_mqb", "info_mqsb", "info_bqb", "info_mq0f", "ref_forward",
    "ref_reverse", "alt_forward", "alt_reverse", "ref_baseq_sum",
    "alt_baseq_sum", "ref_mapq_sum", "alt_mapq_sum", "ref_tail_sum",
    "alt_tail_sum",
)


def metadata_tuple(fields: list[str], key: str) -> tuple[object, ...]:
    """Compact pooled INFO record; the first occurrence of a locus is enough."""
    info = parse_info(fields[7])
    numeric = lambda name: pd.to_numeric(info.get(name), errors="coerce")
    try:
        i16 = [float(value) for value in str(info.get("I16", "")).split(",")]
    except ValueError:
        i16 = []
    if len(i16) != 16:
        i16 = [np.nan] * 16
    return (
        key, fields[3].upper(), fields[4].upper(), numeric("DP"), numeric("AF"),
        info.get("RACE"), info.get("NEURAL_NETWORK_CLASS"), numeric("NEURAL_NETWORK_HOMO"),
        numeric("NEURAL_NETWORK_HETERO"), numeric("NEURAL_NETWORK_NOVAR"), numeric("VDB"),
        numeric("SGB"), numeric("RPB"), numeric("MQB"), numeric("MQSB"), numeric("BQB"),
        numeric("MQ0F"), i16[0], i16[1], i16[2], i16[3], i16[4], i16[6], i16[8],
        i16[10], i16[12], i16[14],
    )


def load_positions(path: Path) -> dict[str, tuple[float, float]]:
    with path.open() as handle:
        first = handle.readline().strip().split(",")[0]
    has_header = not bool(re.fullmatch(r"[ACGTacgt]+-\d+", first))
    frame = pd.read_csv(path, header=0 if has_header else None)
    if has_header:
        columns = {str(name): idx for idx, name in enumerate(frame.columns)}
        required = {
            "barcode": columns.get("barcode", 0),
            "in_tissue": columns.get("in_tissue", 1),
            "row": columns.get("pxl_row_in_fullres", 4),
            "col": columns.get("pxl_col_in_fullres", 5),
        }
        values = frame.iloc[:, list(required.values())].copy()
    else:
        values = frame.iloc[:, [0, 1, 4, 5]].copy()
    values.columns = ["barcode", "in_tissue", "row", "col"]
    values = values[pd.to_numeric(values["in_tissue"], errors="coerce").eq(1)]
    return {
        str(row.barcode): (float(row.col), float(row.row))
        for row in values.itertuples(index=False)
    }


def viewer_graph(barcodes: list[str], positions: dict[str, tuple[float, float]]):
    """Six-neighbour graph matching SPARCALViewer's 1.5x-pitch rule."""

    points = np.asarray([positions[barcode] for barcode in barcodes], dtype=float)
    n = len(points)
    if n < 2:
        return sparse.csr_matrix((n, n), dtype=np.uint8), np.nan, np.nan
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(7, n))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    nearest = distances[:, 1]
    pitch = float(np.median(nearest[np.isfinite(nearest)]))
    cutoff = 1.5 * pitch
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            if np.isfinite(distance) and distance <= cutoff and i != int(j):
                rows.append(i)
                cols.append(int(j))
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n)
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph, pitch, cutoff


def _parse_chunk(tasks):
    candidate_index = _VARIANT_INDEX
    assert candidate_index is not None
    row_indices: list[int] = []
    col_indices: list[int] = []
    depths: list[int] = []
    alt_counts: list[int] = []
    spot_totals: list[tuple[int, int]] = []
    failures: list[str] = []
    metadata: dict[int, tuple[object, ...]] = {}
    for barcode, spot_index, path in tasks:
        total = 0
        try:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 10 or "," in fields[4]:
                        continue
                    parsed = parse_counts(fields[8], fields[9])
                    if parsed is None:
                        continue
                    depth, alt_count = parsed
                    total += depth
                    key = f"{chrom(fields[0])}_{fields[1]}"
                    row_index = candidate_index.get(key)
                    if row_index is not None and row_index not in metadata:
                        metadata[row_index] = metadata_tuple(fields, key)
                    if depth <= 0:
                        continue
                    if row_index is None:
                        continue
                    row_indices.append(row_index)
                    col_indices.append(spot_index)
                    depths.append(depth)
                    alt_counts.append(alt_count)
        except Exception as exc:  # pragma: no cover - corrupt external file
            failures.append(f"{barcode}\t{path}\t{exc!r}")
        spot_totals.append((spot_index, total))
    return (
        np.asarray(row_indices, dtype=np.int32),
        np.asarray(col_indices, dtype=np.int32),
        np.asarray(depths, dtype=np.int32),
        np.asarray(alt_counts, dtype=np.int32),
        spot_totals,
        failures,
        metadata,
    )


def build(section: str, output_dir: Path, workers: int, limit_spots: int | None, overwrite: bool) -> None:
    global _VARIANT_INDEX
    cfg = config.SECTIONS[section]
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "cache_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"cache exists: {output_dir}; pass --overwrite or use a new run label")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(cfg["scores"], sep="\t", dtype={"variant": str})
    if scores["variant"].duplicated().any():
        raise ValueError("all_variant_scores contains duplicate coordinate keys")
    scores = scores.rename(columns={
        "classification": "source_classification",
        "somatic_score": "source_somatic_score",
        "germline_score": "source_germline_score",
    })
    _VARIANT_INDEX = {value: idx for idx, value in enumerate(scores["variant"].astype(str))}

    vcf_dir = Path(cfg["vcf_dir"])
    positions = load_positions(Path(cfg["positions"]))
    vcf_files = {
        path.name[:-7]: path for path in vcf_dir.glob("*.vcf.gz")
    }
    barcodes = sorted(set(positions) & set(vcf_files))
    if limit_spots:
        barcodes = barcodes[:limit_spots]
    if not barcodes:
        raise FileNotFoundError(f"no in-tissue VCF files under {vcf_dir}")
    log(f"{section}: {len(scores):,} all-class candidates; {len(barcodes):,} in-tissue spot VCFs")

    graph, pitch, cutoff = viewer_graph(barcodes, positions)
    spot_index = {barcode: idx for idx, barcode in enumerate(barcodes)}
    tasks = [(barcode, spot_index[barcode], str(vcf_files[barcode])) for barcode in barcodes]
    chunks = [tasks[index::workers] for index in range(workers)]

    started = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_parse_chunk, chunks)
    log(f"{section}: parsed spot VCFs in {time.time() - started:.1f}s")

    rows = np.concatenate([result[0] for result in results])
    cols = np.concatenate([result[1] for result in results])
    depth_values = np.concatenate([result[2] for result in results])
    alt_values = np.concatenate([result[3] for result in results])
    shape = (len(scores), len(barcodes))
    depth = sparse.coo_matrix((depth_values, (rows, cols)), shape=shape).tocsr()
    depth.sum_duplicates()
    positive = alt_values > 0
    alt = sparse.coo_matrix(
        (alt_values[positive], (rows[positive], cols[positive])), shape=shape
    ).tocsr()
    alt.sum_duplicates()
    presence = alt.copy()
    presence.data = np.ones_like(presence.data, dtype=np.uint8)
    presence.eliminate_zeros()

    total_dp = np.zeros(len(barcodes), dtype=np.int64)
    failures: list[str] = []
    metadata_by_index: dict[int, tuple[object, ...]] = {}
    for result in results:
        for idx, value in result[4]:
            total_dp[idx] = value
        failures.extend(result[5])
        for row_index, values in result[6].items():
            metadata_by_index.setdefault(row_index, values)

    meta = pd.DataFrame(metadata_by_index.values(), columns=METADATA_COLUMNS)
    log(f"{section}: pooled INFO recovered for {len(meta):,}/{len(scores):,} candidates")


    scores = scores.merge(meta, on="variant", how="left", validate="one_to_one")
    scores["total_alt"] = np.asarray(alt.sum(axis=1)).ravel().astype(np.int64)
    scores["total_depth_cache"] = np.asarray(depth.sum(axis=1)).ravel().astype(np.int64)
    scores["positive_spots"] = np.diff(presence.indptr).astype(np.int32)
    scores["covered_spots_cache"] = np.diff(depth.indptr).astype(np.int32)
    scores["vaf_cache"] = np.divide(
        scores["total_alt"], scores["total_depth_cache"],
        out=np.full(len(scores), np.nan), where=scores["total_depth_cache"].to_numpy() > 0,
    )

    alt_strand_total = scores["alt_forward"] + scores["alt_reverse"]
    scores["alt_strand_balance"] = np.divide(
        np.minimum(scores["alt_forward"], scores["alt_reverse"]), alt_strand_total,
        out=np.full(len(scores), np.nan), where=alt_strand_total.to_numpy() > 0,
    )
    scores["alt_mean_baseq"] = np.divide(
        scores["alt_baseq_sum"], alt_strand_total,
        out=np.full(len(scores), np.nan), where=alt_strand_total.to_numpy() > 0,
    )
    scores["alt_mean_mapq"] = np.divide(
        scores["alt_mapq_sum"], alt_strand_total,
        out=np.full(len(scores), np.nan), where=alt_strand_total.to_numpy() > 0,
    )
    scores["variant_key"] = (
        scores["variant"].astype(str) + "_" + scores["ref"].fillna("N").astype(str)
        + "_" + scores["alt"].fillna("N").astype(str)
    )

    spot_frame = pd.DataFrame({
        "barcode": barcodes,
        "spot_index": np.arange(len(barcodes), dtype=np.int32),
        "x": [positions[barcode][0] for barcode in barcodes],
        "y": [positions[barcode][1] for barcode in barcodes],
        "total_dp": total_dp,
    })
    rank = pd.Series(total_dp).rank(method="first")
    spot_frame["depth_decile"] = pd.qcut(rank, 10, labels=False, duplicates="drop").astype(int)

    scores.to_csv(output_dir / "candidates.tsv.gz", sep="\t", index=False)
    spot_frame.to_csv(output_dir / "spots.tsv", sep="\t", index=False)
    sparse.save_npz(output_dir / "depth.npz", depth, compressed=True)
    sparse.save_npz(output_dir / "alt.npz", alt, compressed=True)
    sparse.save_npz(output_dir / "presence.npz", presence, compressed=True)
    sparse.save_npz(output_dir / "viewer_graph.npz", graph, compressed=True)
    if failures:
        (output_dir / "parse_failures.tsv").write_text("\n".join(failures) + "\n")

    manifest = {
        "section": section,
        "run_label": config.RUN_LABEL,
        "source_scores": str(cfg["scores"]),
        "source_vcf_dir": str(vcf_dir),
        "source_positions": str(cfg["positions"]),
        "n_candidates": len(scores),
        "n_spots": len(barcodes),
        "depth_nnz": int(depth.nnz),
        "alt_nnz": int(alt.nnz),
        "graph_edges": int(graph.nnz // 2),
        "median_pitch": pitch,
        "neighbor_cutoff": cutoff,
        "parse_failures": len(failures),
        "presence_definition": "FORMAT/AD[1] > 0",
        "coverage_definition": "max(sum(FORMAT/AD), FORMAT/DP) > 0",
        "elapsed_seconds": time.time() - started,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    log(f"{section}: cache complete at {output_dir}; alt nnz={alt.nnz:,}, depth nnz={depth.nnz:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", required=True, choices=sorted(config.SECTIONS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit-spots", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or config.cache_dir(args.section)
    build(args.section, output, max(1, args.workers), args.limit_spots, args.overwrite)


if __name__ == "__main__":
    main()
