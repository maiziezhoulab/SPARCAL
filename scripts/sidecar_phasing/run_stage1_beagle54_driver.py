#!/usr/bin/env python3
"""
run_stage1_beagle54_driver.py — orchestrates run_stage1_beagle54.py across
chr1-22 for a configurable set of samples. Used for:
  (a) the 29 DCIS (sample,chrom) pairs that failed under Beagle 4.1
  (b) ALL 44 P4/P6 chromosomes (engine-consistency + 4.1-vs-5.4 concordance)

Writes/updates a SEPARATE manifest file from the 4.1 driver's
(_manifest_beagle54.json), so neither manifest is ever overwritten by the
other engine's run. Every entry carries "engine": "beagle5.4" explicitly.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
SCRIPT = os.path.join(REPO, "scripts/sidecar_phasing/run_stage1_beagle54.py")
OUT_ROOT = os.path.join(REPO, "data/confident_set_phasing_2026-08-24")
LOG_DIR = os.path.join(OUT_ROOT, "genomewide_beagle_gt", "_logs_beagle54")
MANIFEST_PATH = os.path.join(OUT_ROOT, "genomewide_beagle_gt", "_manifest_beagle54.json")

SAMPLES_ALL = [
    ("P4", "P4_TUMOR", "1"),
    ("P6", "P6_TUMOR", "1"),
    ("DCIS1", "DCIS", "1"),
    ("DCIS2", "DCIS", "2"),
]
CHROMS = [f"chr{i}" for i in range(1, 23)]

# The 29 (sample, chrom) pairs that failed under Beagle 4.1 (from
# genomewide_beagle_gt/_manifest.json, returncode != 0), for --only-failed-4.1.
DCIS_FAILED_4_1 = {
    ("DCIS1", "chr4"), ("DCIS1", "chr3"), ("DCIS1", "chr1"), ("DCIS1", "chr2"),
    ("DCIS1", "chr8"), ("DCIS1", "chr6"), ("DCIS1", "chr7"), ("DCIS1", "chr12"),
    ("DCIS1", "chr14"), ("DCIS1", "chr15"), ("DCIS1", "chr18"), ("DCIS1", "chr20"),
    ("DCIS1", "chr22"), ("DCIS1", "chr19"),
    ("DCIS2", "chr2"), ("DCIS2", "chr1"), ("DCIS2", "chr5"), ("DCIS2", "chr8"),
    ("DCIS2", "chr6"), ("DCIS2", "chr7"), ("DCIS2", "chr9"), ("DCIS2", "chr13"),
    ("DCIS2", "chr10"), ("DCIS2", "chr12"), ("DCIS2", "chr14"), ("DCIS2", "chr15"),
    ("DCIS2", "chr16"), ("DCIS2", "chr22"), ("DCIS2", "chr19"),
}


def run_job(label, dataset, section_id, chrom, threads):
    log_path = os.path.join(LOG_DIR, f"{label}_{chrom}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    cmd = [sys.executable, SCRIPT, "--dataset", dataset, "--section_id", section_id,
           "--chrom", chrom, "--quality-filter", "baseQ0mapQ0", "--threads", str(threads)]
    start = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - start
    result = {"label": label, "dataset": dataset, "section_id": section_id, "chrom": chrom,
              "engine": "beagle5.4", "returncode": proc.returncode, "wall_seconds": wall, "log_path": log_path}
    if proc.returncode == 0:
        try:
            with open(log_path) as f:
                text = f.read()
            json_start = text.index('{\n  "engine"')
            payload = json.loads(text[json_start:])
            result["stats"] = payload.get("stats")
            result["reused_existing_run"] = payload.get("reused_existing_run")
            result["beagle_wall_seconds"] = payload.get("beagle_wall_seconds")
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
    ap.add_argument("--samples", default="P4,P6,DCIS1,DCIS2", help="comma-separated labels to include")
    ap.add_argument("--only-failed-4.1", dest="only_failed", action="store_true",
                    help="restrict DCIS1/DCIS2 to exactly the 29 pairs that failed under Beagle 4.1")
    ap.add_argument("--explicit-pairs", default=None,
                    help="comma-separated LABEL:CHROM pairs, e.g. DCIS1:chr5,DCIS1:chr9 -- "
                         "if given, OVERRIDES --samples/--only-failed-4.1 entirely and runs "
                         "exactly this job list.")
    args = ap.parse_args()

    label_to_ds = {l: (d, s) for l, d, s in SAMPLES_ALL}
    if args.explicit_pairs:
        jobs = []
        for tok in args.explicit_pairs.split(","):
            label, chrom = tok.split(":")
            dataset, section_id = label_to_ds[label]
            jobs.append((label, dataset, section_id, chrom))
    else:
        wanted_samples = set(args.samples.split(","))
        jobs = []
        for label, dataset, section_id in SAMPLES_ALL:
            if label not in wanted_samples:
                continue
            for chrom in CHROMS:
                if args.only_failed and label in ("DCIS1", "DCIS2") and (label, chrom) not in DCIS_FAILED_4_1:
                    continue
                jobs.append((label, dataset, section_id, chrom))

    os.makedirs(LOG_DIR, exist_ok=True)
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    done_keys = {(m["label"], m["chrom"]) for m in manifest if m.get("returncode") == 0}

    print(f"[driver54] {len(jobs)} total jobs queued, {len(done_keys)} already recorded done, "
          f"workers={args.workers} threads/job={args.threads_per_job}", flush=True)
    to_run = [(l, d, s, c) for (l, d, s, c) in jobs if (l, c) not in done_keys]
    print(f"[driver54] {len(to_run)} jobs to actually launch this pass.", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, label, dataset, section_id, chrom, args.threads_per_job): (label, chrom)
                   for (label, dataset, section_id, chrom) in to_run}
        for fut in as_completed(futures):
            label, chrom = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"label": label, "chrom": chrom, "engine": "beagle5.4", "returncode": -1, "exception": str(e)}
            manifest.append(res)
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            status = "OK" if res.get("returncode") == 0 else "FAIL"
            phased = res.get("stats", {}).get("n_het_phased") if res.get("stats") else None
            print(f"[driver54] {status} {label} {chrom} rc={res.get('returncode')} "
                  f"wall={res.get('wall_seconds', 0):.1f}s n_het_phased={phased}", flush=True)

    n_ok = sum(1 for m in manifest if m.get("returncode") == 0)
    n_fail = sum(1 for m in manifest if m.get("returncode") != 0)
    print(f"[driver54] FINISHED pass. manifest total={len(manifest)} ok={n_ok} fail={n_fail}", flush=True)
    print(f"[driver54] manifest -> {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
