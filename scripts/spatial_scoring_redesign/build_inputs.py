#!/usr/bin/env python
"""
build_inputs.py
================
Side-car, READ-ONLY input builder for the Stage-2 spatial-scoring redesign.
Run once per section; caches everything Method A / Method B / Method C need
under data/spatial_scoring_redesign_2026-08-31/_cache/{section}/.

What it reads (all read-only, nothing shipped is modified or re-run):
  - all_variant_scores.txt          -- the Stage-2 candidate pool + shipped vote/classification
  - output_VCFs/spotprofiles/{qf}/vcf_by_spot/{barcode}.vcf.gz  -- per-spot local genotype calls
  - the SpaceRanger tissue_positions file                        -- spot (x,y)

Presence definition (IMPORTANT, documented departure from the shipped
pipeline's own `spot_snvs`): the shipped `load_snv_data()` in
run_spatial_snv_filter_enhanced.py adds a variant to a spot's set merely
because a ROW for it exists in that spot's vcf_by_spot file -- regardless of
that spot's own GT/AD (verified directly: most rows in a given spot's file
are GT=0/0, AD=0,0, i.e. "this position was locally genotyped, ref-only", not
"this spot has the ALT allele"). Reusing that convention here would make
Method A's "spatial autocorrelation of presence" actually measure spatial
autocorrelation of LOCAL COVERAGE/GENOTYPING BREADTH -- exactly the coverage
confound Method A is supposed to control for, defeating its own purpose. This
script therefore defines presence as FORMAT/AD alt-count >= 1 (this spot has
>=1 locally observed ALT-supporting read at this exact position), which is
the correct, non-circular notion of "detected here" for a clustering
statistic. This is a deliberate, disclosed choice -- flagged again in RESULTS.md.

Depth proxy per spot ("total UMI depth" in the task spec): sum of FORMAT/DP
(this spot's own local read depth) over EVERY row in that spot's vcf_by_spot
file (not just candidate rows) -- a direct, already-computed sequencing-depth/
library-complexity proxy, no BAM rescan needed. (INFO/DP in these files is a
GLOBAL, position-level total across the whole section's pooled multi-BAM
mpileup, verified identical across spot files for the same position -- used
separately below as the per-variant "read depth" for endpoint 2, again with
no BAM rescan.)

Outputs (data/spatial_scoring_redesign_2026-08-31/_cache/{section}/):
  spots.tsv        barcode, x, y, total_dp, decile, spot_idx
  candidates.tsv   variant, chrom, pos, ref, alt, shipped_classification,
                    shipped_vote_fraction, info_dp, n_pos_spots
  graph_W.npz       sparse spots x spots symmetric 0/1 adjacency (scipy npz)
  presence.npz      sparse candidates x spots binary (AD_alt>=1), int8 (npz)
  altdepth.npz      sparse candidates x spots AD_alt counts, int32 (npz)
"""
import argparse
import gzip
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

_CANDIDATE_INDEX = None  # set as a global before Pool() so fork-COW shares it


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _parse_one_file(barcode, path):
    """Parse a single spot's vcf_by_spot file. Returns
    (barcode, total_dp:int, triples:list[(row_idx,int alt_count)], info_updates:dict[row_idx]->(ref,alt,info_dp))
    """
    total_dp = 0
    triples = []
    info_updates = {}
    cand_idx = _CANDIDATE_INDEX
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
                pos = parts[1]
                ref = parts[3]
                alt = parts[4]
                if "," in alt:
                    continue
                sample = parts[9]
                sub = sample.split(":")
                # Observed FORMAT order in every sampled file: GT:AD:DP[:GQ:PL]
                try:
                    ad = sub[1]
                    dp_s = sub[2]
                except IndexError:
                    fmt_keys = parts[8].split(":")
                    d = dict(zip(fmt_keys, sub))
                    ad = d.get("AD", "0,0")
                    dp_s = d.get("DP", "0")
                try:
                    dp_val = int(dp_s)
                except ValueError:
                    dp_val = 0
                total_dp += dp_val

                key = chrom + "_" + pos
                row_idx = cand_idx.get(key)
                if row_idx is not None:
                    try:
                        alt_count = int(ad.split(",")[1])
                    except (IndexError, ValueError):
                        alt_count = 0
                    if alt_count >= 1:
                        triples.append((row_idx, alt_count))
                    if row_idx not in info_updates:
                        info_dp = None
                        for field in parts[7].split(";"):
                            if field.startswith("DP="):
                                info_dp = field[3:]
                                break
                        info_updates[row_idx] = (ref, alt, info_dp)
    except Exception as e:
        log(f"WARNING: failed parsing {path}: {e}")
    return barcode, total_dp, triples, info_updates


def _worker_chunk(args):
    chunk = args
    out_dp = {}
    out_triples = []  # (row_idx, spot_idx, alt_count)
    out_info = {}
    for barcode, path, spot_idx in chunk:
        bc, dp, triples, info_updates = _parse_one_file(barcode, path)
        out_dp[bc] = dp
        for row_idx, alt_count in triples:
            out_triples.append((row_idx, spot_idx, alt_count))
        for row_idx, tup in info_updates.items():
            if row_idx not in out_info:
                out_info[row_idx] = tup
    return out_dp, out_triples, out_info


def build_section(section, n_workers=12, limit_spots=None):
    global _CANDIDATE_INDEX
    cfg = common.SECTIONS[section]
    outdir = common.cache_dir(section)
    log(f"=== {section} === build_dir={outdir}")

    log("loading candidate pool from all_variant_scores.txt ...")
    candidates = common.load_candidate_pool(section)
    n_cand = len(candidates)
    log(f"{section}: {n_cand} Stage-2 candidates (race=denovo, shipped classification in somatic/ambiguous)")
    _CANDIDATE_INDEX = {v: i for i, v in enumerate(candidates["variant"].values)}

    log("loading spot positions ...")
    positions = common.load_spot_positions(section)
    log(f"{section}: {len(positions)} in-tissue spots with positions")

    vcf_files = {}
    for path in os.listdir(cfg["vcf_by_spot_dir"]):
        if path.endswith(".vcf.gz"):
            bc = path[:-7]
            vcf_files[bc] = os.path.join(cfg["vcf_by_spot_dir"], path)
    log(f"{section}: {len(vcf_files)} vcf_by_spot files")

    barcodes = sorted(set(positions.keys()) & set(vcf_files.keys()))
    if limit_spots:
        barcodes = barcodes[:limit_spots]
    n_spots = len(barcodes)
    log(f"{section}: {n_spots} spots with BOTH position and vcf_by_spot file (final spot set)")
    spot_idx_of = {bc: i for i, bc in enumerate(barcodes)}

    log("building hexagonal neighbour graph (kNN=13 + median-pitch threshold, ported read-only) ...")
    W, median_pitch, thresh = common.build_neighbor_graph(barcodes, positions)
    log(f"{section}: median NN pitch={median_pitch:.1f}px, neighbor_threshold={thresh:.1f}px, "
        f"mean degree={W.nnz / n_spots:.2f}")

    log(f"parsing {n_spots} per-spot VCFs across {n_workers} workers "
        f"(presence:=FORMAT/AD alt-count>=1, depth proxy:=sum FORMAT/DP per spot) ...")
    tasks = [(bc, vcf_files[bc], spot_idx_of[bc]) for bc in barcodes]
    chunks = [tasks[i::n_workers] for i in range(n_workers)]

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_worker_chunk, chunks)
    log(f"parse done in {time.time() - t0:.1f}s")

    total_dp = np.zeros(n_spots, dtype=np.int64)
    rows_list = []
    cols_list = []
    vals_list = []
    info_map = {}
    for out_dp, out_triples, out_info in results:
        for bc, dp in out_dp.items():
            total_dp[spot_idx_of[bc]] = dp
        for row_idx, spot_idx, alt_count in out_triples:
            rows_list.append(row_idx)
            cols_list.append(spot_idx)
            vals_list.append(alt_count)
        for row_idx, tup in out_info.items():
            if row_idx not in info_map:
                info_map[row_idx] = tup

    rows_arr = np.asarray(rows_list, dtype=np.int64)
    cols_arr = np.asarray(cols_list, dtype=np.int64)
    vals_arr = np.asarray(vals_list, dtype=np.int32)

    altdepth = sparse.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_cand, n_spots)).tocsr()
    altdepth.sum_duplicates()
    presence = altdepth.copy()
    presence.data = np.ones_like(presence.data, dtype=np.int8)

    n_pos_spots = np.asarray(presence.sum(axis=1)).ravel()
    log(f"{section}: presence matrix nnz={presence.nnz}, "
        f"median positive-spots/candidate={np.median(n_pos_spots):.1f}, "
        f"candidates with >=2 positive spots: {(n_pos_spots >= 2).sum()}")

    ref_col = []
    alt_col = []
    info_dp_col = []
    for i in range(n_cand):
        tup = info_map.get(i)
        if tup is None:
            ref_col.append(None)
            alt_col.append(None)
            info_dp_col.append(np.nan)
        else:
            ref_col.append(tup[0])
            alt_col.append(tup[1])
            try:
                info_dp_col.append(float(tup[2]) if tup[2] is not None else np.nan)
            except ValueError:
                info_dp_col.append(np.nan)

    candidates = candidates.copy()
    candidates["ref"] = ref_col
    candidates["alt"] = alt_col
    candidates["info_dp"] = info_dp_col
    candidates["n_pos_spots"] = n_pos_spots

    n_missing_info = candidates["info_dp"].isna().sum()
    if n_missing_info:
        log(f"NOTE: {n_missing_info}/{n_cand} candidates never appeared as a row in ANY spot's "
            f"vcf_by_spot file at all (zero local coverage anywhere) -- info_dp/ref/alt unknown for these; "
            f"they get n_pos_spots=0 and are excluded from any BAM-based endpoint by construction.")

    # depth deciles from the per-spot total_dp proxy
    decile = pd.qcut(pd.Series(total_dp).rank(method="first"), 10, labels=False, duplicates="drop")
    decile = decile.values.astype(np.int32)

    spots_df = pd.DataFrame({
        "barcode": barcodes,
        "x": [positions[b][0] for b in barcodes],
        "y": [positions[b][1] for b in barcodes],
        "total_dp": total_dp,
        "decile": decile,
        "spot_idx": np.arange(n_spots),
    })

    spots_df.to_csv(os.path.join(outdir, "spots.tsv"), sep="\t", index=False)
    candidates.to_csv(os.path.join(outdir, "candidates.tsv"), sep="\t", index=False)
    sparse.save_npz(os.path.join(outdir, "graph_W.npz"), W)
    sparse.save_npz(os.path.join(outdir, "presence.npz"), presence)
    sparse.save_npz(os.path.join(outdir, "altdepth.npz"), altdepth)

    with open(os.path.join(outdir, "build_summary.txt"), "w") as f:
        f.write(f"section={section}\n")
        f.write(f"n_spots={n_spots}\n")
        f.write(f"n_candidates={n_cand}\n")
        f.write(f"median_nn_pitch_px={median_pitch}\n")
        f.write(f"neighbor_threshold_px={thresh}\n")
        f.write(f"mean_graph_degree={W.nnz / n_spots}\n")
        f.write(f"presence_nnz={presence.nnz}\n")
        f.write(f"n_candidates_ge2_pos_spots={(n_pos_spots >= 2).sum()}\n")
        f.write(f"n_candidates_zero_local_coverage_anywhere={n_missing_info}\n")
        f.write(f"decile_bin_counts={pd.Series(decile).value_counts().sort_index().to_dict()}\n")

    log(f"{section}: wrote cache to {outdir}")
    return outdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-spots", type=int, default=None, help="debug: only parse the first N spots")
    args = ap.parse_args()
    build_section(args.section, n_workers=args.workers, limit_spots=args.limit_spots)
    log("DONE")
