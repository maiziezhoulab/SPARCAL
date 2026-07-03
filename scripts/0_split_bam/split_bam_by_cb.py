#!/usr/bin/env python
"""
Split a 10x possorted BAM into per-barcode BAMs by the CB tag.

Written for datasets (e.g. OVAR_P5 / Ovar_P5) whose source outs/ is read-only and
whose BAM carries a single @RG line, so `samtools split` (RG-based) cannot produce
per-barcode files. This does a single streaming pass over the BAM with pysam and
writes one <CB>.bam per (in-tissue) barcode into an output directory in OUR project.

Robust to the open-file-descriptor limit: it raises RLIMIT_NOFILE to what is needed
when possible, and otherwise falls back to processing barcodes in chunks (one BAM
pass per chunk).

Usage:
    python split_bam_by_cb.py \
        --bam    /path/to/possorted_genome_bam.bam \
        --positions /path/to/spatial/tissue_positions_list.csv \
        --out-dir data/ovar_p5/P5_sr13/split_BAM \
        --in-tissue-only --index --threads 4
"""

import argparse
import os
import resource
import sys

import pysam


def load_barcodes(positions_file, in_tissue_only, in_tissue_col=1):
    """Read barcodes from a headerless Visium tissue_positions_list.csv."""
    barcodes = set()
    with open(positions_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            bc = parts[0]
            # Skip a header row if present.
            if bc.lower() in ("barcode", "barcodes"):
                continue
            if in_tissue_only:
                try:
                    if int(parts[in_tissue_col]) != 1:
                        continue
                except (IndexError, ValueError):
                    continue
            barcodes.add(bc)
    return barcodes


def raise_fd_limit(target):
    """Try to raise the soft NOFILE limit toward `target`; return the achieved soft cap."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(hard, target)
    if want > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            soft = want
        except (ValueError, OSError):
            pass
    return soft


def get_cb(read):
    try:
        return read.get_tag("CB")
    except KeyError:
        return None


def split_single_pass(bam_path, barcodes, out_dir, header, threads):
    """Open one output BAM per barcode and stream the input once."""
    writers = {
        bc: pysam.AlignmentFile(
            os.path.join(out_dir, f"{bc}.bam"), "wb", header=header, threads=threads
        )
        for bc in barcodes
    }
    counts = {bc: 0 for bc in barcodes}
    n_total = n_written = 0
    with pysam.AlignmentFile(bam_path, "rb", threads=threads) as bam:
        for read in bam:
            n_total += 1
            cb = get_cb(read)
            w = writers.get(cb) if cb is not None else None
            if w is not None:
                w.write(read)
                counts[cb] += 1
                n_written += 1
    for w in writers.values():
        w.close()
    return n_total, n_written, counts


def split_chunked(bam_path, barcodes, out_dir, header, threads, chunk_size):
    """Fallback: one BAM pass per chunk of barcodes (bounded open FDs)."""
    barcodes = sorted(barcodes)
    counts = {}
    n_total = n_written = 0
    n_chunks = (len(barcodes) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = set(barcodes[ci * chunk_size:(ci + 1) * chunk_size])
        writers = {
            bc: pysam.AlignmentFile(
                os.path.join(out_dir, f"{bc}.bam"), "wb", header=header, threads=threads
            )
            for bc in chunk
        }
        with pysam.AlignmentFile(bam_path, "rb", threads=threads) as bam:
            for read in bam:
                if ci == 0:
                    n_total += 1
                cb = get_cb(read)
                w = writers.get(cb) if cb is not None else None
                if w is not None:
                    w.write(read)
                    counts[cb] = counts.get(cb, 0) + 1
                    n_written += 1
        for w in writers.values():
            w.close()
        print(f"  chunk {ci + 1}/{n_chunks} done ({len(chunk)} barcodes)", flush=True)
    return n_total, n_written, counts


def main():
    ap = argparse.ArgumentParser(description="Split a 10x BAM into per-barcode BAMs by CB tag.")
    ap.add_argument("--bam", required=True, help="Input possorted_genome_bam.bam")
    ap.add_argument("--positions", required=True, help="tissue_positions_list.csv")
    ap.add_argument("--out-dir", required=True, help="Output dir for <CB>.bam files")
    ap.add_argument("--in-tissue-only", action="store_true",
                    help="Only split in-tissue barcodes (col[1]==1)")
    ap.add_argument("--in-tissue-col", type=int, default=1)
    ap.add_argument("--index", action="store_true", help="samtools-index each output BAM")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    barcodes = load_barcodes(args.positions, args.in_tissue_only, args.in_tissue_col)
    if not barcodes:
        sys.exit(f"ERROR: no barcodes loaded from {args.positions}")
    print(f"Loaded {len(barcodes)} barcodes "
          f"({'in-tissue only' if args.in_tissue_only else 'all'})", flush=True)

    needed = len(barcodes) + 128
    soft = raise_fd_limit(needed)
    print(f"NOFILE soft limit: {soft} (need ~{needed})", flush=True)

    with pysam.AlignmentFile(args.bam, "rb") as bam:
        header = bam.header.to_dict()

    if soft >= needed:
        print("Splitting in a single pass...", flush=True)
        n_total, n_written, counts = split_single_pass(
            args.bam, barcodes, args.out_dir, header, args.threads)
    else:
        chunk_size = max(64, soft - 128)
        print(f"FD limit too low for single pass; chunking (chunk_size={chunk_size})...",
              flush=True)
        n_total, n_written, counts = split_chunked(
            args.bam, barcodes, args.out_dir, header, args.threads, chunk_size)

    n_nonempty = sum(1 for c in counts.values() if c > 0)
    print(f"Reads: {n_total} total, {n_written} written to barcodes", flush=True)
    print(f"Barcodes with >=1 read: {n_nonempty}/{len(barcodes)}", flush=True)

    if args.index:
        print("Indexing output BAMs...", flush=True)
        for bc in barcodes:
            path = os.path.join(args.out_dir, f"{bc}.bam")
            if os.path.exists(path):
                pysam.index(path)
        print("Indexing done.", flush=True)


if __name__ == "__main__":
    main()
