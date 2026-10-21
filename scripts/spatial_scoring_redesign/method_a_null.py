#!/usr/bin/env python
"""
method_a_null.py
=================
Method A: calibrated spatial statistic with a DETECTION-CONDITIONED
permutation null (the redesign's core proposal).

Statistic (primary): join-count on binary ALT-presence (AD alt-count >= 1
per spot, see build_inputs.py docstring for why this differs from the
shipped pipeline's coverage-based `spot_snvs`), on the same symmetrized
hexagonal neighbour graph the pipeline builds.
  JC(v) = 0.5 * sum_i sum_j W_ij * X_v(i) * X_v(j)   (W binary, symmetric, W_ii=0)
        = number of graph edges whose BOTH endpoints have the variant.

Statistic (secondary, reported for robustness): Moran's I on ALT-read-depth
(continuous AD alt-count per spot, from the same source, no BAM rescan).

Null (the redesign's key step): permute a variant's positive spots WITHIN
its own depth-decile strata (10 bins on each spot's total local FORMAT/DP,
see build_inputs.py), holding the number of positive spots PER DECILE fixed
for every variant simultaneously -- not just holding the total count k fixed.
This is implemented with one efficient identity, verified by hand:

  Column-permuting the presence matrix X by sigma, where sigma restricted to
  each decile's spot-index set is an independent random permutation of THAT
  SAME set, preserves -- for every row (variant) at once -- the exact count
  of positive spots within every decile (a column permutation only reorders
  which spot-slot holds a 0/1 value; it cannot change how many 1s fall in a
  fixed index subset that sigma maps to itself). So ONE shared random
  within-decile column permutation per round gives a valid detection-
  conditioned null draw for the ENTIRE candidate pool simultaneously, instead
  of requiring a separate permutation loop per variant.

>=999 permutations -> one-sided empirical p (JC_perm >= JC_obs) -> BH q over
the FULL Stage-2 candidate pool (candidates with <2 positive spots can never
have a nonzero join-count and are assigned p=1 without spending permutation
budget on them).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def joincount_all(X_csr, W_csr):
    """X: (n_variants x n_spots) sparse 0/1. W: (n_spots x n_spots) sparse 0/1 symmetric.
    Returns length-n_variants array of join counts."""
    XW = X_csr @ W_csr           # (n_variants x n_spots), XW[v,i] = # neighbors of i that have v
    prod = XW.multiply(X_csr)    # keep only spots where v is ALSO present
    return 0.5 * np.asarray(prod.sum(axis=1)).ravel()


def morans_i_all(Xc_csr, W_csr, spot_mask=None):
    """Moran's I on a continuous n_variants x n_spots matrix (ALT depth), computed
    over ALL n_spots (absence -> 0, a legitimate 'zero ALT reads observed' value,
    see build_inputs.py). Vectorized per-row Moran's I with a shared W."""
    n = Xc_csr.shape[1]
    row_mean = np.asarray(Xc_csr.mean(axis=1)).ravel()
    # center: (X - mean) as dense row operations would be too costly; use sparse-safe algebra.
    # I = (n/S0) * sum_ij w_ij (x_i-mx)(x_j-mx) / sum_i (x_i-mx)^2
    # sum_ij w_ij (x_i-mx)(x_j-mx) = sum_ij w_ij x_i x_j - mx*sum_ij w_ij x_i - mx*sum_ij w_ij x_j + mx^2*sum_ij w_ij
    # = (X W X^T)_diag - 2*mx*(X W).sum(axis=1)... but since W row sums vary, do it exactly per row:
    S0 = W_csr.sum()
    XW = Xc_csr @ W_csr  # n_variants x n_spots
    num_cross = np.asarray(XW.multiply(Xc_csr).sum(axis=1)).ravel()   # sum_ij w_ij x_i x_j
    Wi_row_sum = np.asarray(W_csr.sum(axis=1)).ravel()                # length n_spots, sum_j w_ij
    sum_wij_xi = np.asarray(Xc_csr.multiply(Wi_row_sum).sum(axis=1)).ravel()  # sum_i x_i * (sum_j w_ij)
    Wj_col_sum = np.asarray(W_csr.sum(axis=0)).ravel()                # symmetric -> same as row sum
    sum_wij_xj = np.asarray(Xc_csr.multiply(Wj_col_sum).sum(axis=1)).ravel()
    numerator = num_cross - row_mean * sum_wij_xi - row_mean * sum_wij_xj + (row_mean ** 2) * S0
    sq = Xc_csr.multiply(Xc_csr)
    sum_x2 = np.asarray(sq.sum(axis=1)).ravel()
    denom = sum_x2 - n * (row_mean ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = (n / S0) * (numerator / denom)
    I[denom <= 1e-12] = 0.0
    return I


def make_sigma(decile, rng):
    sigma = np.arange(len(decile))
    for d in np.unique(decile):
        idx = np.where(decile == d)[0]
        sigma[idx] = rng.permutation(idx)
    return sigma


def run_section(section, n_perm=999, seed=0, workers_note="single-process, vectorized sparse"):
    outdir = common.cache_dir(section)
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t")
    spots = pd.read_csv(os.path.join(outdir, "spots.tsv"), sep="\t")
    W = sparse.load_npz(os.path.join(outdir, "graph_W.npz")).tocsr()
    presence = sparse.load_npz(os.path.join(outdir, "presence.npz")).tocsr()
    altdepth = sparse.load_npz(os.path.join(outdir, "altdepth.npz")).tocsr()

    n_cand = presence.shape[0]
    n_spots = presence.shape[1]
    decile = spots.sort_values("spot_idx")["decile"].values
    assert len(decile) == n_spots

    n_pos = candidates["n_pos_spots"].values
    testable = np.where(n_pos >= 2)[0]
    log(f"{section}: {n_cand} candidates total, {len(testable)} testable (n_pos_spots>=2), "
        f"{n_cand - len(testable)} auto-assigned p=1 (statistic trivially 0)")

    X = presence[testable, :].tocsr()
    Xc = altdepth[testable, :].astype(np.float64).tocsr()

    log("computing observed join-count statistic ...")
    jc_obs = joincount_all(X, W)
    log("computing observed Moran's I (ALT-depth, secondary statistic) ...")
    moran_obs = morans_i_all(Xc, W)

    log(f"running {n_perm} detection-conditioned permutations "
        f"(within-decile spot relabeling, shared across all candidates per round) ...")
    rng = np.random.default_rng(seed)
    count_ge_jc = np.zeros(len(testable), dtype=np.int64)
    count_ge_moran = np.zeros(len(testable), dtype=np.int64)
    Xcsc = X.tocsc()
    Xc_csc = Xc.tocsc()

    t0 = time.time()
    for r in range(n_perm):
        sigma = make_sigma(decile, rng)
        Xp = Xcsc[:, sigma].tocsr()
        jc_p = joincount_all(Xp, W)
        count_ge_jc += (jc_p >= jc_obs)

        Xcp = Xc_csc[:, sigma].tocsr()
        moran_p = morans_i_all(Xcp, W)
        count_ge_moran += (moran_p >= moran_obs)

        if (r + 1) % 100 == 0 or r == 0:
            elapsed = time.time() - t0
            rate = (r + 1) / elapsed
            eta = (n_perm - r - 1) / rate if rate > 0 else float("nan")
            log(f"  permutation {r + 1}/{n_perm}  ({elapsed:.1f}s elapsed, "
                f"{rate:.2f} perm/s, ETA {eta:.0f}s)")
    log(f"permutation loop done in {time.time() - t0:.1f}s")

    p_jc_testable = (1 + count_ge_jc) / (1 + n_perm)
    p_moran_testable = (1 + count_ge_moran) / (1 + n_perm)

    p_jc_full = np.ones(n_cand)
    p_moran_full = np.ones(n_cand)
    jc_obs_full = np.zeros(n_cand)
    moran_obs_full = np.zeros(n_cand)
    p_jc_full[testable] = p_jc_testable
    p_moran_full[testable] = p_moran_testable
    jc_obs_full[testable] = jc_obs
    moran_obs_full[testable] = moran_obs

    q_jc_full = common.bh_qvalues(p_jc_full)
    q_moran_full = common.bh_qvalues(p_moran_full)

    out = candidates.copy()
    out["join_count_obs"] = jc_obs_full
    out["p_joincount"] = p_jc_full
    out["q_joincount"] = q_jc_full
    out["moran_i_obs"] = moran_obs_full
    out["p_moran"] = p_moran_full
    out["q_moran"] = q_moran_full

    scores_path = os.path.join(outdir, f"methodA_{section}_scores.tsv")
    out.to_csv(scores_path, sep="\t", index=False)
    log(f"{section}: wrote {scores_path}")

    counts = {}
    for q_thresh in (0.05, 0.10, 0.20):
        counts[q_thresh] = {
            "joincount": int((out["q_joincount"] < q_thresh).sum()),
            "moran": int((out["q_moran"] < q_thresh).sum()),
        }
        log(f"{section}: join-count q<{q_thresh} -> {counts[q_thresh]['joincount']} variants; "
            f"Moran's I q<{q_thresh} -> {counts[q_thresh]['moran']} variants")

    n_shipped_somatic = int((candidates["shipped_classification"] == "somatic").sum())
    log(f"{section}: for comparison, shipped 10%% quota selected {n_shipped_somatic} variants")

    summary_row = dict(section=section, n_candidates=n_cand, n_testable=len(testable),
                        n_permutations=n_perm, seed=seed,
                        n_shipped_quota=n_shipped_somatic,
                        q05_joincount=counts[0.05]["joincount"], q10_joincount=counts[0.10]["joincount"],
                        q20_joincount=counts[0.20]["joincount"],
                        q05_moran=counts[0.05]["moran"], q10_moran=counts[0.10]["moran"],
                        q20_moran=counts[0.20]["moran"],
                        min_achievable_p=1.0 / (1 + n_perm))
    calib_path = os.path.join(common.OUT_ROOT, "fdr_calibration.csv")
    row_df = pd.DataFrame([summary_row])
    if os.path.exists(calib_path):
        prev = pd.read_csv(calib_path)
        prev = prev[prev["section"] != section]
        row_df = pd.concat([prev, row_df], ignore_index=True)
    row_df.to_csv(calib_path, index=False)
    log(f"{section}: appended to {calib_path}")

    return out, summary_row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_section(args.section, n_perm=args.n_perm, seed=args.seed)
    log("DONE")
