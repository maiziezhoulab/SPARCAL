#!/usr/bin/env python
"""
method_b_gae.py
================
Method B: unsupervised graph-autoencoder (GAE) alternative to Method A,
tested fairly against it (per task spec: hyperparameters fixed a priori,
never tuned against any of the four evaluation endpoints).

Environment: run under conda env `snv_clustering` (has torch 2.2.2+cu121 +
torch_geometric 2.5.2 -- confirmed present, so no numpy/scipy fallback was
needed; CPU only in this session, fine at this graph size).

Design (deliberately structural-only, NO per-variant information as node
features -- see rationale below):
  - Graph: the SAME symmetrized hexagonal spot graph Method A uses
    (graph_W.npz from build_inputs.py), converted to a PyG edge_index.
  - Node features (4-dim, z-scored): spatial x, spatial y, log1p(per-spot
    total local depth proxy), node degree. NOT the candidate-presence
    matrix. Using presence as node features would let the encoder's spot
    embeddings directly encode which candidates are near which spots before
    any scoring happens -- a form of leakage into "unsupervised" training
    that would make Method B's comparison to Method A/C less honest. Using
    only generic structural/depth covariates keeps the model exactly what
    the task asks for: "a graph autoencoder / unsupervised embedding over
    the spot graph" with no labels and, additionally here, no per-candidate
    information at all during training.
  - Encoder: 2-layer GCNConv (4 -> 16 -> 8), ReLU between layers, wrapped in
    torch_geometric.nn.GAE (standard inner-product edge decoder).
  - Training objective: GAE.recon_loss (reconstruct the OBSERVED spot graph
    edges against random negative pairs) -- pure structure learning, no
    variant labels anywhere in the loss.
  - Fixed hyperparameters (chosen before looking at ANY endpoint result,
    not tuned): hidden=16, out=8, epochs=200, lr=0.01, Adam, seed=0.
  - Per-variant score (computed AFTER training, from the frozen embeddings):
    for a candidate present in spots P_v (|P_v|>=2), score = -(mean
    Euclidean distance of each embedded positive spot to their centroid) --
    i.e. how tightly the candidate's positive spots cluster in the learned
    structural embedding. Higher score = tighter = more spatial structure.
    Candidates with <2 positive spots get score = -inf (ranked last, same
    convention as Method A's "not testable").
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

HIDDEN = 16
OUT_DIM = 8
EPOCHS = 200
LR = 0.01
SEED = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def build_features(spots_df, W):
    x = spots_df["x"].values.astype(np.float64)
    y = spots_df["y"].values.astype(np.float64)
    dp = np.log1p(spots_df["total_dp"].values.astype(np.float64))
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)

    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v - v.mean()

    feats = np.stack([z(x), z(y), z(dp), z(degree)], axis=1)
    return feats


def run_section(section):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GAE, GCNConv

    n_threads = int(os.environ.get("GAE_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
    log(f"{section}: torch.set_num_threads({n_threads})")

    outdir = common.cache_dir(section)
    spots_df = pd.read_csv(os.path.join(outdir, "spots.tsv"), sep="\t").sort_values("spot_idx")
    W = sparse.load_npz(os.path.join(outdir, "graph_W.npz")).tocsr()
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t")
    presence = sparse.load_npz(os.path.join(outdir, "presence.npz")).tocsr()

    n_spots = W.shape[0]
    log(f"{section}: {n_spots} spots, building node features (structural only, no variant info) ...")
    feats = build_features(spots_df, W)

    Wcoo = W.tocoo()
    edge_index = torch.tensor(np.vstack([Wcoo.row, Wcoo.col]), dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float32)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    class Encoder(torch.nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            z = self.conv2(h, edge_index)
            return z

    model = GAE(Encoder(x.shape[1], HIDDEN, OUT_DIM))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log(f"{section}: training GAE (hidden={HIDDEN}, out={OUT_DIM}, epochs={EPOCHS}, lr={LR}, seed={SEED}) ...")
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        te = time.time()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        if epoch < 5 or (epoch + 1) % 20 == 0:
            log(f"  epoch {epoch + 1}/{EPOCHS}  loss={loss.item():.4f}  ({time.time() - te:.2f}s/epoch, "
                f"{time.time() - t0:.1f}s total)")
    log(f"{section}: training done in {time.time() - t0:.1f}s, final loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z = model.encode(x, edge_index).numpy()

    log(f"{section}: scoring {presence.shape[0]} candidates by embedding-space compactness ...")
    n_pos = candidates["n_pos_spots"].values
    scores = np.full(len(candidates), -np.inf, dtype=np.float64)
    presence_csr = presence.tocsr()
    testable = np.where(n_pos >= 2)[0]
    for row_idx in testable:
        spot_idx = presence_csr[row_idx, :].indices
        emb = Z[spot_idx, :]
        centroid = emb.mean(axis=0)
        mean_dist = np.linalg.norm(emb - centroid, axis=1).mean()
        scores[row_idx] = -mean_dist

    out = candidates.copy()
    out["gae_score"] = scores
    out["gae_rank"] = (-out["gae_score"]).rank(method="first")

    scores_path = os.path.join(outdir, f"methodB_{section}_scores.tsv")
    out.to_csv(scores_path, sep="\t", index=False)
    log(f"{section}: wrote {scores_path}")

    hp_path = os.path.join(outdir, f"methodB_{section}_hyperparams.txt")
    with open(hp_path, "w") as f:
        f.write(f"encoder=2-layer GCNConv\nhidden={HIDDEN}\nout_dim={OUT_DIM}\nepochs={EPOCHS}\n"
                f"lr={LR}\noptimizer=Adam\nseed={SEED}\n"
                f"node_features=[x_z,y_z,log1p(total_dp)_z,degree_z] (structural only, no per-variant data)\n"
                f"loss=torch_geometric.nn.GAE.recon_loss (edge reconstruction, random negative sampling)\n"
                f"final_loss={loss.item():.4f}\n"
                f"score=-(mean euclidean distance of positive spots' embeddings to their centroid)\n"
                f"fixed_a_priori=True (not tuned against any evaluation endpoint)\n")
    log(f"{section}: wrote {hp_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    args = ap.parse_args()
    run_section(args.section)
    log("DONE")
