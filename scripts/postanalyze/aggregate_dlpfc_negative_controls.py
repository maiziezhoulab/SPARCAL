#!/usr/bin/env python3
"""Aggregate the P0-1 DLPFC negative-control clustering runs.

Reads   data/dlpfc_negative_controls_2026-08-27/<section>/clustering/summary.csv
Writes  control_ari_long.csv, control_summary.csv, donor_level_tests.csv

Section->donor map: spatialLIBD DLPFC is 12 sections from THREE donors,
4 sections each, adjacent pairs 10um apart.  Treating n=12 as independent
is pseudoreplication (referee finding C7).
"""
import csv, glob, os, statistics, collections, itertools, math

D = "data/dlpfc_negative_controls_2026-08-27"
DONOR = {}
for s in ("151507","151508","151509","151510"): DONOR[s] = "Br5292"
for s in ("151669","151670","151671","151672"): DONOR[s] = "Br5595"
for s in ("151673","151674","151675","151676"): DONOR[s] = "Br8100"

rows = []
for f in sorted(glob.glob(os.path.join(D, "*", "clustering", "summary.csv"))):
    for r in csv.DictReader(open(f)):
        a = (r.get("ari") or "").strip()
        rows.append({
            "section": r["section"], "donor": DONOR.get(r["section"], "?"),
            "modality": r["modality"], "run": r["run"], "seed": r.get("seed",""),
            "ari": float(a) if a else None,
            "n_spots": r.get("n_spots",""), "n_snvs": r.get("n_snvs",""),
            "degenerate": "" if a else "1",
            "status": (r.get("status") or "")[:120],
        })

with open(os.path.join(D, "control_ari_long.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# per modality x section
per_sec = collections.defaultdict(list)
for r in rows:
    if r["ari"] is not None:
        per_sec[(r["modality"], r["section"])].append(r["ari"])

summ = []
mods = sorted({r["modality"] for r in rows})
for m in mods:
    allv = [v for r in rows if r["modality"] == m and r["ari"] is not None for v in [r["ari"]]]
    ndeg = sum(1 for r in rows if r["modality"] == m and r["ari"] is None)
    secmeans = [statistics.mean(per_sec[(m, s)]) for s in sorted(DONOR) if per_sec.get((m, s))]
    summ.append({
        "modality": m, "n_runs": sum(1 for r in rows if r["modality"] == m),
        "n_valid": len(allv), "n_degenerate": ndeg,
        "n_sections_with_valid": len(secmeans),
        "mean_ari_over_runs": round(statistics.mean(allv), 4) if allv else "",
        "sd_ari_over_runs": round(statistics.pstdev(allv), 4) if len(allv) > 1 else "",
        "mean_of_section_means": round(statistics.mean(secmeans), 4) if secmeans else "",
        "sd_of_section_means": round(statistics.pstdev(secmeans), 4) if len(secmeans) > 1 else "",
    })
with open(os.path.join(D, "control_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

# donor level
don = []
for m in mods:
    dmeans = {}
    for d in sorted(set(DONOR.values())):
        secs = [s for s in DONOR if DONOR[s] == d and per_sec.get((m, s))]
        if secs:
            dmeans[d] = statistics.mean(statistics.mean(per_sec[(m, s)]) for s in secs)
    if dmeans:
        don.append({
            "modality": m, "n_donors": len(dmeans),
            **{f"donor_{k}_mean_ari": round(v, 4) for k, v in dmeans.items()},
            "mean_over_donors": round(statistics.mean(dmeans.values()), 4),
            "sd_over_donors": round(statistics.pstdev(list(dmeans.values())), 4) if len(dmeans) > 1 else "",
            "note": "n=3 donors is the correct unit of replication, not n=12 sections (C7)",
        })
with open(os.path.join(D, "donor_level_tests.csv"), "w", newline="") as fh:
    keys = sorted({k for r in don for k in r})
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(don)

print("modality            runs valid  degen  meanARI    sd   sections")
for r in summ:
    print(f"{r['modality']:20}{r['n_runs']:>4}{r['n_valid']:>6}{r['n_degenerate']:>7}"
          f"{str(r['mean_ari_over_runs']):>9}{str(r['sd_ari_over_runs']):>7}{r['n_sections_with_valid']:>10}")
print("\nwrote control_ari_long.csv, control_summary.csv, donor_level_tests.csv")
