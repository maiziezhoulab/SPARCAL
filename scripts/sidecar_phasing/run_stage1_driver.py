#!/usr/bin/env python3
"""
run_stage1_driver.py — orchestrates run_stage1_genomewide.py across chr1-22
for all four samples (P4, P6, DCIS1, DCIS2), bounded concurrency, writes a
manifest with per-(sample,chrom) timing/success/phasing-yield.

Does not touch scripts/1_calling..6_spatial_filter or data/<sample>/output_VCFs.
Writes only under data/confident_set_phasing_2026-08-24/ (this task's own
deliverable directory; see run_stage1_genomewide.py docstring for why this
tree is used instead of data_sidecar_phased/).

Usage:
    conda activate snv_caller
    python scripts/sidecar_phasing/run_stage1_driver.py --workers 4 --threads-per-job 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
SCRIPT = os.path.join(REPO, "scripts/sidecar_phasing/run_stage1_genomewide.py")
OUT_ROOT = os.path.join(REPO, "data/confident_set_phasing_2026-08-24")
LOG_DIR = os.path.join(OUT_ROOT, "genomewide_beagle_gt", "_logs")
MANIFEST_PATH = os.path.join(OUT_ROOT, "genomewide_beagle_gt", "_manifest.json")

# (label used in reporting, --dataset, --section_id)
SAMPLES = [
    ("P4", "P4_TUMOR", "1"),
    ("P6", "P6_TUMOR", "1"),
    ("DCIS1", "DCIS", "1"),
    ("DCIS2", "DCIS", "2"),
]
CHROMS = [f"chr{i}" for i in range(1, 23)]


def run_job(label, dataset, section_id, chrom, threads):
    log_path = os.path.join(LOG_DIR, f"{label}_{chrom}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    cmd = [
        sys.executable, SCRIPT,
        "--dataset", dataset, "--section_id", section_id, "--chrom", chrom,
        "--quality-filter", "baseQ0mapQ0", "--threads", str(threads),
    ]
    start = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - start
    result = {
        "label": label, "dataset": dataset, "section_id": section_id, "chrom": chrom,
        "returncode": proc.returncode, "wall_seconds": wall, "log_path": log_path,
    }
    if proc.returncode == 0:
        # pull the last JSON blob (stats) out of the log for the manifest
        try:
            with open(log_path) as f:
                text = f.read()
            json_start = text.index("{\n  \"dataset\"")
            payload = json.loads(text[json_start:])
            result["stats"] = payload.get("stats")
            result["reused_existing_run"] = payload.get("reused_existing_run")
            result["beagle_wall_seconds"] = payload.get("beagle_wall_seconds")
            result["marker_check_seconds"] = payload.get("marker_check_seconds")
            result["total_wall_seconds"] = payload.get("total_wall_seconds")
        except Exception as e:
            result["parse_error"] = str(e)
    else:
        with open(log_path) as f:
            result["tail"] = f.read()[-4000:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--only-sample", default=None, help="restrict to one sample label (P4/P6/DCIS1/DCIS2)")
    ap.add_argument("--only-chrom", default=None, help="restrict to one chrom (e.g. chr1)")
    args = ap.parse_args()

    jobs = []
    for label, dataset, section_id in SAMPLES:
        if args.only_sample and label != args.only_sample:
            continue
        for chrom in CHROMS:
            if args.only_chrom and chrom != args.only_chrom:
                continue
            jobs.append((label, dataset, section_id, chrom))

    os.makedirs(LOG_DIR, exist_ok=True)
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    done_keys = {(m["label"], m["chrom"]) for m in manifest if m.get("returncode") == 0}

    print(f"[driver] {len(jobs)} total jobs queued, {len(done_keys)} already recorded done, "
          f"workers={args.workers} threads/job={args.threads_per_job}", flush=True)

    to_run = [(l, d, s, c) for (l, d, s, c) in jobs if (l, c) not in done_keys]
    print(f"[driver] {len(to_run)} jobs to actually launch this pass.", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_job, label, dataset, section_id, chrom, args.threads_per_job): (label, chrom)
            for (label, dataset, section_id, chrom) in to_run
        }
        for fut in as_completed(futures):
            label, chrom = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"label": label, "chrom": chrom, "returncode": -1, "exception": str(e)}
            manifest.append(res)
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            status = "OK" if res.get("returncode") == 0 else "FAIL"
            phased = None
            if res.get("stats"):
                phased = res["stats"].get("n_het_phased")
            print(f"[driver] {status} {label} {chrom} rc={res.get('returncode')} "
                  f"wall={res.get('wall_seconds', 0):.1f}s n_het_phased={phased}", flush=True)

    n_ok = sum(1 for m in manifest if m.get("returncode") == 0)
    n_fail = sum(1 for m in manifest if m.get("returncode") != 0)
    print(f"[driver] FINISHED pass. manifest total={len(manifest)} ok={n_ok} fail={n_fail}", flush=True)
    print(f"[driver] manifest -> {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
