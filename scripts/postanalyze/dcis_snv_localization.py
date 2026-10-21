#!/usr/bin/env python3
"""Depth-adjusted tumor localization of called SNVs against a pathology annotation.

Question, per called variant: is its ALT allele observed preferentially inside the annotated
tumor, at matched depth? A raw "fraction of ALT spots inside the tumor" is inflated because tumor
spots carry more UMIs, so every statistic here conditions on per-spot molecule depth, and each
SPARCAL class is read against the 1000G germline class as the internal control.

Per variant: spots in the labelled set that carry >= 1 UB-collapsed molecule at the locus
(S6+ table); positive = >= 1 ALT molecule; strata by molecule depth {1,2,3,4,5,6-7,8-10,11-15,16-25,>=26}; Mantel-
Haenszel common OR (0.5 added to each cell of strata that contain both groups) and the CMH
chi-square test; "informative" = >= 3 covered spots in each of tumor and non-tumor. Per class:
pooled MH OR over (variant x stratum) cells with a spatial block bootstrap CI.

Labels, the DCIS2 labelled window and spot coordinates come from the region-benchmark harness
(SPARCAL_Benchmarking/.../snv_informed_regions_2026-09-12/harness/data.py:load_section), so this
uses exactly the evaluation set of the region benchmark. Pathology labels are read ONLY for the
localization statistics; the confident tier never uses them.

    python scripts/postanalyze/dcis_snv_localization.py --sections DCIS1 DCIS2 \
        --out data/dcis_snv_justification_2026-10-20
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
HARNESS = Path("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/analysis/region_method_benchmark/"
               "snv_informed_regions_2026-09-12")
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(PROJECT / "scripts/6_spatial_filter"))
from harness.data import load_section  # noqa: E402
from upv_v1_annotate import load_bed, in_bed  # noqa: E402

SECTIONS = {
    "DCIS1": {"root": PROJECT / "data/dcis1", "recount": "DCIS_dcis1.tsv.gz", "gate": "DCIS1"},
    "DCIS2": {"root": PROJECT / "data/dcis2", "recount": "DCIS_dcis2.tsv.gz", "gate": "DCIS2"},
}
RECOUNT_DIR = PROJECT / "data/presence_fix_2026-10-09/recount"
EDIT_BED = PROJECT / "resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz"
GTF = Path("/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/genes/gene_annotations.gtf.gz")
CGC = Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_CancerGeneCensus_v103_GRCh38.tsv")
COSMIC_VCF = Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz")

# Confident somatic tier (pathology-free).
TIER_MIN_ALT_MOL = 2
TIER_MIN_ALT_SPOTS = 2
XMHC = ("6", 28_000_000, 34_000_000)          # GRCh38
MIN_COVERED_PER_GROUP = 3
DEPTH_BREAKS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 15), (16, 25), (26, np.inf)]
BLOCK_ROWS, BLOCK_COLS = 6, 12                 # Visium array: col index steps by 2 per spot
BOOT_REPS, BOOT_SEED, BOOT_CHUNK = 1000, 20261020, 50
FDR = 0.10


def key(chrom, pos, ref, alt):
    return f"{str(chrom).removeprefix('chr')}_{int(pos)}_{str(ref).upper()}_{str(alt).upper()}"


def read_class(path, race=None):
    d = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    if race is not None:
        d = d[d["race"] == race]
    return set(key(*r) for r in d[["chrom", "pos", "ref", "alt"]].itertuples(index=False))


def load_classes(root, tag):
    sf = root / "spatial_filter_purity" / tag
    g = sf / "germline/germline_variants.txt"
    out = {
        "1000G": read_class(g, "defined"),
        "UPV": read_class(g, "denovo"),
        "somatic": read_class(sf / "somatic/somatic_variants.txt"),
        "ambiguous": read_class(sf / "ambiguous/ambiguous_variants.txt"),
    }
    som = pd.read_csv(sf / "somatic/somatic_variants.txt", sep="\t", dtype={"chrom": str})
    return out, som


def stratum_of(depth):
    s = np.full(depth.shape, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(DEPTH_BREAKS):
        s[(depth >= lo) & (depth <= hi)] = i
    return s


def bh(p):
    p = np.asarray(p, float); q = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * len(v) / np.arange(1, len(v) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(v); out[o] = np.clip(r, 0, 1)
        q[ok] = out
    return q


def cell_table(rows):
    """rows: DataFrame(key, stratum, tumor, pos) -> per (key, stratum) a,b,c,d (raw counts)."""
    g = rows.groupby(["key", "stratum", "tumor"], observed=True)["pos"].agg(["sum", "size"]).reset_index()
    g["neg"] = g["size"] - g["sum"]
    t = g[g.tumor == 1].set_index(["key", "stratum"])
    n = g[g.tumor == 0].set_index(["key", "stratum"])
    cells = pd.DataFrame({"a": t["sum"], "b": t["neg"]}).join(
        pd.DataFrame({"c": n["sum"], "d": n["neg"]}), how="outer").fillna(0)
    return cells.reset_index()


def per_variant(cells):
    both = cells[((cells.a + cells.b) > 0) & ((cells.c + cells.d) > 0)].copy()
    # MH OR with 0.5 continuity in strata that contain both groups.
    for x in "abcd":
        both[x + "5"] = both[x] + 0.5
    both["n5"] = both[["a5", "b5", "c5", "d5"]].sum(axis=1)
    both["num"] = both.a5 * both.d5 / both.n5
    both["den"] = both.b5 * both.c5 / both.n5
    # CMH chi-square on raw counts.
    n = both[["a", "b", "c", "d"]].sum(axis=1)
    both["ea"] = (both.a + both.b) * (both.a + both.c) / n
    both["va"] = np.where(n > 1, (both.a + both.b) * (both.c + both.d) * (both.a + both.c)
                          * (both.b + both.d) / (n ** 2 * (n - 1)), 0.0)
    agg = both.groupby("key").agg(num=("num", "sum"), den=("den", "sum"), a=("a", "sum"),
                                  ea=("ea", "sum"), va=("va", "sum"))
    cov = cells.groupby("key").agg(t_cov=("a", "sum"), t_neg=("b", "sum"),
                                   n_cov=("c", "sum"), n_neg=("d", "sum"))
    cov["tumor_covered"] = cov.t_cov + cov.t_neg; cov["normal_covered"] = cov.n_cov + cov.n_neg
    cov["tumor_alt_spots"] = cov.t_cov; cov["normal_alt_spots"] = cov.n_cov
    out = cov[["tumor_covered", "normal_covered", "tumor_alt_spots", "normal_alt_spots"]].join(agg, how="left")
    out["log_or"] = np.log(out.num / out.den)
    stat = (out.a - out.ea) ** 2 / out.va.replace(0, np.nan)
    out["p_cmh"] = chi2.sf(stat, 1)
    out["informative"] = (out.tumor_covered >= MIN_COVERED_PER_GROUP) & (out.normal_covered >= MIN_COVERED_PER_GROUP)
    out.loc[~out.informative, ["log_or", "p_cmh"]] = np.nan
    return out.drop(columns=["num", "den", "a", "ea", "va"])


def pooled_or(cells):
    c = cells[((cells.a + cells.b) > 0) & ((cells.c + cells.d) > 0)]
    n = c[["a", "b", "c", "d"]].sum(axis=1)
    num = (c.a * c.d / n).sum(); den = (c.b * c.c / n).sum()
    return float(num / den) if den > 0 else np.nan


def bootstrap_pooled(rows, n_blocks, reps=BOOT_REPS, seed=BOOT_SEED):
    """Spatial block bootstrap of the pooled MH OR; rows carry an integer 'block' column."""
    cells = rows.groupby(["key", "stratum"], observed=True).ngroup().to_numpy()
    n_cells = cells.max() + 1 if len(cells) else 0
    mats = {}
    for name, sel in (("a", (rows.tumor == 1) & (rows.pos == 1)), ("b", (rows.tumor == 1) & (rows.pos == 0)),
                      ("c", (rows.tumor == 0) & (rows.pos == 1)), ("d", (rows.tumor == 0) & (rows.pos == 0))):
        sel = sel.to_numpy()
        mats[name] = sp.csr_matrix((np.ones(sel.sum()), (rows.block.to_numpy()[sel], cells[sel])),
                                   shape=(n_blocks, n_cells))
    rng = np.random.default_rng(seed); out = []
    for start in range(0, reps, BOOT_CHUNK):
        k = min(BOOT_CHUNK, reps - start)
        w = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=k).astype(float)
        A, B, C, D = (np.asarray(mats[x].T @ w.T).T for x in "abcd")   # (k, n_cells), dense
        n = A + B + C + D
        ok = ((A + B) > 0) & ((C + D) > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            num = np.where(ok, A * D / n, 0).sum(1); den = np.where(ok, B * C / n, 0).sum(1)
        out.extend((num / den).tolist())
    out = np.asarray(out); out = out[np.isfinite(out)]
    return (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))) if len(out) else (np.nan, np.nan)


def load_gene_index():
    rows = []
    with gzip.open(GTF, "rt") as h:
        for line in h:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            name = f[8].split('gene_name "', 1)[1].split('"', 1)[0] if 'gene_name "' in f[8] else ""
            rows.append((f[0].removeprefix("chr"), int(f[3]), int(f[4]), name))
    g = pd.DataFrame(rows, columns=["chrom", "start", "end", "gene"])
    return {c: d.sort_values("start").reset_index(drop=True) for c, d in g.groupby("chrom")}


def genes_at(index, chrom, pos):
    d = index.get(chrom)
    if d is None:
        return ""
    hit = d[(d.start <= pos) & (d.end >= pos)]
    return ",".join(sorted(set(hit.gene)))


def cosmic_hit(vcf, chrom, pos, ref, alt):
    try:
        for rec in vcf.fetch(chrom, pos - 1, pos):
            if rec.pos == pos and rec.ref == ref and alt in (rec.alts or ()):
                return rec.id or "yes"
    except ValueError:
        return ""
    return ""


def run_section(name, tag, out_dir, edit_bed, gene_index, cgc_genes, cosmic):
    cfg = SECTIONS[name]
    sec = load_section(name)
    classes, som_table = load_classes(cfg["root"], tag)
    gate_path = PROJECT / f"data/calicost_gate_2026-10-19/{cfg['gate']}.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None

    mol = pd.read_csv(RECOUNT_DIR / cfg["recount"], sep="\t",
                      usecols=["barcode", "chrom", "pos", "ref", "alt", "ref_mol", "alt_mol"],
                      dtype={"chrom": str, "barcode": "category"})
    idx = {b: i for i, b in enumerate(sec.barcodes)}
    mol = mol[mol.barcode.isin(idx)].copy()
    mol["key"] = [key(*r) for r in mol[["chrom", "pos", "ref", "alt"]].itertuples(index=False)]
    mol["spot"] = mol.barcode.map(idx).astype(int)
    mol["depth"] = mol.ref_mol + mol.alt_mol

    # ---- confident somatic tier (pathology-free; all in-tissue spots) ----
    alt_tot = mol.groupby("key").alt_mol.sum()
    alt_spots = mol[mol.alt_mol >= 1].groupby("key").size()
    som = som_table.copy()
    som["key"] = [key(*r) for r in som[["chrom", "pos", "ref", "alt"]].itertuples(index=False)]
    som["alt_molecules"] = som.key.map(alt_tot).fillna(0).astype(int)
    som["alt_spots"] = som.key.map(alt_spots).fillna(0).astype(int)
    som["f_alt_support"] = (som.alt_molecules >= TIER_MIN_ALT_MOL) & (som.alt_spots >= TIER_MIN_ALT_SPOTS)
    som["f_not_panel"] = som.race != "defined"
    pos_i = som.pos.astype(int)
    som["f_not_xmhc"] = ~((som.chrom.str.removeprefix("chr") == XMHC[0]) & (pos_i >= XMHC[1]) & (pos_i <= XMHC[2]))
    som["f_not_editing"] = [not in_bed(edit_bed, "chr" + c.removeprefix("chr"), p - 1)
                            for c, p in zip(som.chrom, pos_i)]
    cnv_ok = bool(gate and gate.get("cnv_pass"))
    # CNV-normal-clone exclusion applies only when the gate certified a reference clone.
    som["f_no_alt_in_reference_clone"] = True
    if cnv_ok:
        ref = gate["reference_clone"]; lab = gate["effective_labels"]
        ref_spots = {idx[b] for b, l in lab.items() if l == ref and b in idx}
        hit = mol[(mol.alt_mol >= 1) & mol.spot.isin(ref_spots)].key.unique()
        som["f_no_alt_in_reference_clone"] = ~som.key.isin(set(hit))
    flags = [c for c in som.columns if c.startswith("f_")]
    som["confident"] = som[flags].all(axis=1)
    classes["somatic_confident"] = set(som.loc[som.confident, "key"])
    filt = {"section": name, "somatic_total": len(som)}
    running = np.ones(len(som), bool)
    for f in flags:
        running &= som[f].to_numpy(); filt[f"after_{f}"] = int(running.sum())
    filt["cnv_reference_filter_applied"] = cnv_ok

    # ---- localization statistics on the labelled set ----
    lab = sec.labels; mask = sec.eval_mask & np.isfinite(lab)
    ev = mol[mask[mol.spot.to_numpy()] & (mol.depth > 0)].copy()
    ev["tumor"] = (lab[ev.spot.to_numpy()] == 1).astype(np.int8)
    ev["pos"] = (ev.alt_mol >= 1).astype(np.int8)
    ev["stratum"] = stratum_of(ev.depth.to_numpy())
    br = sec.array_row // BLOCK_ROWS; bc = sec.array_col // BLOCK_COLS
    block_codes, block_ids = np.unique(np.column_stack([br, bc]), axis=0, return_inverse=True)
    ev["block"] = block_ids.ravel()[ev.spot.to_numpy()]
    n_blocks = int(ev.block.max()) + 1

    per_var, summary = [], []
    for cls, keys in classes.items():
        rows = ev[ev.key.isin(keys)]
        if rows.empty:
            summary.append({"section": name, "class": cls, "n_calls": len(keys)}); continue
        cells = cell_table(rows)
        pv = per_variant(cells)
        pv["q_bh"] = bh(pv.p_cmh.to_numpy())
        pv["class"] = cls; pv["section"] = name
        per_var.append(pv.reset_index())
        inf = pv[pv.informative]
        lo, hi = bootstrap_pooled(rows[rows.key.isin(inf.index)], n_blocks)
        summary.append({
            "section": name, "class": cls, "n_calls": len(keys),
            "n_with_labelled_coverage": int(len(pv)), "n_informative": int(len(inf)),
            "frac_informative": len(inf) / len(keys) if keys else np.nan,
            "median_log_or": float(inf.log_or.median()) if len(inf) else np.nan,
            "iqr_log_or": float(inf.log_or.quantile(0.75) - inf.log_or.quantile(0.25)) if len(inf) else np.nan,
            "frac_tumor_enriched_fdr10": float(((inf.q_bh < FDR) & (inf.log_or > 0)).mean()) if len(inf) else np.nan,
            "frac_normal_enriched_fdr10": float(((inf.q_bh < FDR) & (inf.log_or < 0)).mean()) if len(inf) else np.nan,
            "pooled_mh_or": pooled_or(cell_table(rows[rows.key.isin(inf.index)])) if len(inf) else np.nan,
            "pooled_or_ci_low": lo, "pooled_or_ci_high": hi,
        })

    # ---- top confident variants (candidates only; not interpreted) ----
    pvd = pd.concat(per_var, ignore_index=True) if per_var else pd.DataFrame()
    conf = pvd[(pvd["class"] == "somatic_confident") & pvd.informative].sort_values("log_or", ascending=False).head(30)
    top = []
    for r in conf.itertuples():
        c, p, ref, alt = r.key.split("_"); p = int(p)
        g = genes_at(gene_index, c, p)
        top.append({"section": name, "variant": r.key, "genes": g,
                    "cgc_gene": ",".join(x for x in g.split(",") if x in cgc_genes),
                    "cosmic_id": cosmic_hit(cosmic, c, p, ref, alt) if cosmic else "",
                    "log_or": r.log_or, "q_bh": r.q_bh, "tumor_covered": r.tumor_covered,
                    "normal_covered": r.normal_covered, "tumor_alt_spots": r.tumor_alt_spots,
                    "normal_alt_spots": r.normal_alt_spots})
    meta = {"section": name, "labelled_spots": int(mask.sum()), "tumor_spots": int(np.nansum(lab[mask] == 1)),
            "blocks": n_blocks, "eval_rows": int(len(ev)), "gate_cnv_pass": cnv_ok}
    return pvd, pd.DataFrame(summary), filt, pd.DataFrame(top), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=["DCIS1", "DCIS2"], choices=sorted(SECTIONS))
    ap.add_argument("--tag", default="baseQ0mapQ0", help="spatial_filter_purity tag to read classes from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    edit_bed = load_bed(str(EDIT_BED))
    gene_index = load_gene_index()
    cgc = set(pd.read_csv(CGC, sep="\t", usecols=["GENE_SYMBOL"]).GENE_SYMBOL)
    try:
        import pysam
        cosmic = pysam.VariantFile(str(COSMIC_VCF))
    except Exception as e:  # noqa: BLE001
        print(f"COSMIC VCF unavailable: {e}", file=sys.stderr); cosmic = None

    pv, sm, ft, tp, meta = [], [], [], [], []
    for s in a.sections:
        print(f"[{s}] running", file=sys.stderr)
        r = run_section(s, a.tag, out, edit_bed, gene_index, cgc, cosmic)
        pv.append(r[0]); sm.append(r[1]); ft.append(r[2]); tp.append(r[3]); meta.append(r[4])
    pd.concat(pv).to_csv(out / "per_variant.csv.gz", index=False)
    pd.concat(sm).to_csv(out / "class_summary.csv", index=False)
    pd.DataFrame(ft).to_csv(out / "confident_tier_filters.csv", index=False)
    pd.concat(tp).to_csv(out / "top_confident.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps({"tag": a.tag, "sections": meta,
        "tier": {"min_alt_mol": TIER_MIN_ALT_MOL, "min_alt_spots": TIER_MIN_ALT_SPOTS,
                 "xmhc_grch38": XMHC, "editing_bed": str(EDIT_BED)},
        "depth_strata": [list(map(float, b)) for b in DEPTH_BREAKS],
        "block": [BLOCK_ROWS, BLOCK_COLS], "bootstrap": [BOOT_REPS, BOOT_SEED]}, indent=1))
    print(pd.concat(sm).to_string(index=False))


if __name__ == "__main__":
    main()
