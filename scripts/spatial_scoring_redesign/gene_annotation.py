#!/usr/bin/env python
"""
gene_annotation.py
====================
READ-ONLY gene-body annotation for endpoint 4 (cross-section gene-level
overlap). No package beyond what's already in `snv_caller` is installed
(`intervaltree` is NOT present -- confirmed -- so this implements a small
sweep-line interval-stabbing lookup with the stdlib `heapq` instead).

GTF sources (both pre-existing, read-only, belonging to other projects on
this filesystem -- NOT written to):
  GRCh37 (P4, P6):     /data/maiziezhou_lab/leiy4/CalicoST/hg19_resources/... is
                        lifted-over; use the same GENCODE hg19 GTF SpaceTracer
                        already downloaded instead (more standard, non-lifted):
                        /data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg19.gtf.gz
  GRCh38 (DCIS1,DCIS2): /data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg38.gtf.gz

Only `feature == "gene"` rows are used (gene body, not exon-restricted --
appropriate here since a handful of intronic/near-exon-boundary candidate
SNVs are expected in spatial RNA-seq calling, and endpoint 4 wants a stable,
generous "which gene locus" label, not exon-precision).
"""
import gzip
import heapq
import os
import re
import sys
from collections import defaultdict

GTF_PATHS = {
    "GRCh37": "/data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg19.gtf.gz",
    "GRCh38": "/data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg38.gtf.gz",
}

GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')

_CACHE = {}


def load_genes(build):
    """Returns dict: chrom(bare) -> list of (start,end,gene_name) sorted by start (1-based, inclusive, GTF convention)."""
    if build in _CACHE:
        return _CACHE[build]
    path = GTF_PATHS[build]
    genes_by_chrom = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
            start = int(parts[3])
            end = int(parts[4])
            m = GENE_NAME_RE.search(parts[8])
            name = m.group(1) if m else None
            if name is None:
                continue
            genes_by_chrom[chrom].append((start, end, name))
    for c in genes_by_chrom:
        genes_by_chrom[c].sort(key=lambda t: t[0])
    _CACHE[build] = dict(genes_by_chrom)
    return _CACHE[build]


def annotate_positions(build, chrom_pos_pairs):
    """chrom_pos_pairs: iterable of (chrom_bare:str, pos:int, 1-based).
    Returns dict (chrom,pos) -> frozenset of gene names overlapping that position
    (empty frozenset if intergenic / chrom not in GTF)."""
    genes_by_chrom = load_genes(build)
    by_chrom = defaultdict(list)
    for chrom, pos in chrom_pos_pairs:
        by_chrom[chrom].append(pos)

    result = {}
    for chrom, positions in by_chrom.items():
        genes = genes_by_chrom.get(str(chrom), [])
        if not genes:
            for p in positions:
                result[(chrom, p)] = frozenset()
            continue
        order = sorted(range(len(positions)), key=lambda i: positions[i])
        sorted_pos = [positions[i] for i in order]

        gi = 0
        n_genes = len(genes)
        active = []  # heap of (end, name)
        active_names = defaultdict(int)  # name -> count currently active (handles duplicate gene records)
        pos_result = [None] * len(sorted_pos)
        for qi, p in enumerate(sorted_pos):
            while gi < n_genes and genes[gi][0] <= p:
                s, e, name = genes[gi]
                if e >= p:
                    heapq.heappush(active, (e, name))
                    active_names[name] += 1
                gi += 1
            while active and active[0][0] < p:
                _, name = heapq.heappop(active)
                active_names[name] -= 1
                if active_names[name] <= 0:
                    del active_names[name]
            pos_result[qi] = frozenset(active_names.keys())
        for qi, orig_i in enumerate(order):
            result[(chrom, positions[orig_i])] = pos_result[qi]
    return result


if __name__ == "__main__":
    # tiny smoke test
    import time
    t0 = time.time()
    genes = load_genes("GRCh38")
    print(f"loaded {sum(len(v) for v in genes.values())} genes in {time.time()-t0:.1f}s", file=sys.stderr)
    test = [("1", 1000000), ("6", 29500000), ("17", 7676000)]  # last one near TP53
    ann = annotate_positions("GRCh38", test)
    for k, v in ann.items():
        print(k, sorted(v))
