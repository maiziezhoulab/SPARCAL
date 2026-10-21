#!/usr/bin/env python3
"""xMHC (chr6:28-34 Mb) enrichment across three independent spatial-RNA callers.

Tests whether the extended-MHC concentration seen in SPARCAL's spatially-clustered
class is caller-specific or a shared property of reference-based variant calling on
spatial RNA.  Compares SPARCAL, SpatialSNV (Mutect2 tumour-only) and Monopogen
(LD-based, >=4 REF / >=4 ALT depth floor), plus SPARCAL on tumour-free DLPFC.

Read-only.  Writes data/xmhc_artifact_mechanism_2026-08-31/cross_caller_xmhc.csv
"""
import csv, glob, gzip, os, pickle, re, sys

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
LO, HI, GENOME = 28_000_000, 34_000_000, 3.1e9
EXPECTED = (HI - LO) / GENOME          # 0.00194
OUT = os.path.join(REPO, "data/xmhc_artifact_mechanism_2026-08-31/cross_caller_xmhc.csv")

def in_xmhc(chrom, pos):
    return chrom == "6" and LO <= pos <= HI

def norm(c):
    m = re.match(r"^(?:chr)?(\w+)$", str(c))
    return m.group(1) if m else None

rows = []
def add(caller, sample, build, tot, inm, src):
    pct = 100.0 * inm / tot if tot else 0.0
    rows.append(dict(caller=caller, sample=sample, build=build, n_calls=tot,
                     n_in_xmhc=inm, pct_in_xmhc=round(pct, 4),
                     fold_vs_genome=round(pct / (100 * EXPECTED), 3) if tot else "",
                     source=src))

# ---- SpatialSNV: features.tsv.gz rows look like "12_54858388:T>A"
for s, build in [("p4", "GRCh37"), ("p6", "GRCh37"), ("dcis1", "GRCh38"), ("dcis2", "GRCh38")]:
    f = f"{REPO}/SpatialSNV/results/{s}/matrix/{s}_alt/features.tsv.gz"
    if not os.path.exists(f):
        add("SpatialSNV", s.upper(), build, 0, 0, "MISSING"); continue
    tot = inm = 0
    with gzip.open(f, "rt") as fh:
        for line in fh:
            m = re.match(r"^(?:chr)?(\w+)_(\d+):", line)
            if not m: continue
            tot += 1
            if in_xmhc(m.group(1), int(m.group(2))): inm += 1
    add("SpatialSNV", s.upper(), build, tot, inm, os.path.relpath(f, REPO))

# ---- Monopogen somatic module: per-chromosome putativeSNVs.csv
for d, s, build in [("Monopogen_P4_rep1", "P4", "GRCh37"),
                    ("Monopogen_P6_rep1", "P6", "GRCh37"),
                    ("Monopogen_DCIS1", "DCIS1", "GRCh38"),
                    ("Monopogen_DCIS2", "DCIS2", "GRCh38")]:
    files = sorted(glob.glob(f"{REPO}/{d}/out/somatic/chr*.putativeSNVs.csv")) or \
            sorted(glob.glob(f"{REPO}/{d}/out/*.putativeSNVs.csv"))
    if not files:
        add("Monopogen", s, build, 0, 0, f"MISSING ({d})"); continue
    tot = inm = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                p = re.split(r"[,\t]", line.strip())
                if len(p) < 2 or not p[1].isdigit(): continue
                c = norm(p[0])
                if not c: continue
                tot += 1
                if in_xmhc(c, int(p[1])): inm += 1
    add("Monopogen", s, build, tot, inm, f"{d}/out ({len(files)} files)")

# ---- SpaceTracer (results-v220).  P4/P6 results/ dirs are EMPTY -> run incomplete.
for s, build in [("P4","GRCh37"), ("P6","GRCh37"), ("DCIS1","GRCh38"), ("DCIS2","GRCh38")]:
    for tag, fn in [("all",  f"{s}_spatial_free_model_total_pred_truesites.vcf"),
                    ("PASS", f"{s}_spatial_free_model_total_pred_truesites_PASS.vcf")]:
        f = f"{REPO}/SpaceTracer/results-v220/{s}/mutation_prediction/results/{fn}"
        if not os.path.exists(f):
            add(f"SpaceTracer_{tag}", s, build, 0, 0, "RUN INCOMPLETE (empty results/)")
            continue
        tot = inm = 0
        with open(f) as fh:
            for line in fh:
                if line.startswith("#"): continue
                q = line.split("\t")
                if len(q) < 2 or not q[1].isdigit(): continue
                c = norm(q[0])
                if not c: continue
                tot += 1
                if in_xmhc(c, int(q[1])): inm += 1
        add(f"SpaceTracer_{tag}", s, build, tot, inm, os.path.relpath(f, REPO))

# ---- SPARCAL on tumour-free DLPFC (all 12 sections), read-only pkl columns
for sec in ["151507","151508","151509","151510","151669","151670",
            "151671","151672","151673","151674","151675","151676"]:
    p = f"{REPO}/data/dlpfc/{sec}/matrix/DLPFC_{sec}_SPARCAL_somatic_matrix.pkl"
    if not os.path.exists(p):
        add("SPARCAL_DLPFC", sec, "GRCh38", 0, 0, "MISSING"); continue
    with open(p, "rb") as fh: M = pickle.load(fh)
    tot = inm = 0
    for col in (M.columns if hasattr(M, "columns") else []):
        m = re.match(r"^(?:chr)?(\w+)[_:](\d+)", str(col))
        if not m: continue
        tot += 1
        if in_xmhc(m.group(1), int(m.group(2))): inm += 1
    add("SPARCAL_DLPFC", sec, "GRCh38", tot, inm, os.path.relpath(p, REPO))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"genome-length expectation = {100*EXPECTED:.3f}%\n")
print(f"{'caller':16}{'sample':9}{'build':9}{'n_calls':>9}{'xMHC':>7}{'pct':>9}{'fold':>8}")
for r in rows:
    print(f"{r['caller']:16}{r['sample']:9}{r['build']:9}{r['n_calls']:>9}"
          f"{r['n_in_xmhc']:>7}{r['pct_in_xmhc']:>8.3f}%{str(r['fold_vs_genome']):>8}")
print(f"\nwrote {OUT}")
