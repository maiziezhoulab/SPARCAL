#!/usr/bin/env python3
"""Freeze and execute a Beagle engineering default; scientific gates stay deferred."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

SCHEMA = "sparcal_beagle_engineering_freeze_v1"
CANDIDATE = "beagle_v2_3_candidate_2026_09_13"
SOURCES = ("run_beagle.py", "beagle_evidence_router.py",
           f"{CANDIDATE}/run_beagle_v2_3.py", f"{CANDIDATE}/contracts.py")
PARAMETERS = dict(MODEL_SCALE=2.0, ITERATIONS=0, MAX_LR=5000.0,
                  SEED=20260915, IMPUTE=False, GPROBS=True, GENETIC_MAP=None,
                  THREADS=30, PARALLEL_CHROMOSOMES=1, MEMORY="10g")
ROUTES = ("beagle_nonref_highconf", "beagle_ref_highconf",
          "beagle_uncertain", "panel_absent")
VIEWS = {r: (r,) for r in ROUTES}
VIEWS.update(filtered_in=ROUTES[:1], filtered_out=ROUTES[1:],
             panel_discordant=ROUTES[1:3], panel_present=ROUTES[:3])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def validate_profile(profile):
    if (profile.get("schema") != SCHEMA
            or profile.get("status") != "architecture_frozen"
            or profile.get("representation") != "single_seed"
            or profile.get("production_inputs_include_wes") is not False
            or profile.get("scientific_final_validation_complete") is not False
            or profile.get("parameters") != PARAMETERS
            or profile.get("anchor_score") != "p_nonref"
            or profile.get("min_anchor_probability") != 0.90
            or profile.get("quality_filter") != "baseQ0mapQ0"):
        raise ValueError("profile differs from the frozen engineering decision")


def verify(bundle):
    bundle = Path(bundle).resolve()
    complete = json.loads((bundle / "FREEZE_COMPLETE.json").read_text())
    if complete.get("schema") != SCHEMA or complete.get("status") != "architecture_frozen":
        raise ValueError("not a completed engineering freeze")
    ledger = bundle / "SHA256SUMS.txt"
    if digest(ledger) != complete.get("ledger_sha256"):
        raise ValueError("bundle ledger changed")
    seen = set()
    for line in ledger.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (bundle / name).resolve()
        path.relative_to(bundle)
        if name in seen or digest(path) != expected:
            raise ValueError(f"bundle checksum mismatch/duplicate: {name}")
        seen.add(name)
    required = {"PROFILE.json", "beagle.py", "DECISION.md", "PROVENANCE.json"}
    required.update(f"runtime/{name}" for name in SOURCES)
    if seen != required:
        raise ValueError("bundle ledger does not cover the exact runtime set")
    profile = json.loads((bundle / "PROFILE.json").read_text())
    validate_profile(profile)
    contracts = load(bundle / "runtime" / CANDIDATE / "contracts.py", "_frozen_contracts")
    contracts.verify_reference_ledger(Path(profile["reference_ledger"]))
    for name, record in profile["executables"].items():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"executable changed: {name}")
    return profile, contracts


def source_manifest(root, contracts):
    contracts.validate_v2_3_run(root)
    manifest = json.loads((root / "V2_3_MANIFEST.json").read_text())
    params = dict(manifest["beagle_parameters"])
    params.pop("REFERENCE_LEDGER")
    if params != PARAMETERS or manifest["min_anchor_probability"] != 0.90:
        raise ValueError("source run does not use the frozen default")
    return manifest


def freeze(project, bundle, decision):
    project, bundle = Path(project).resolve(), Path(bundle).resolve()
    if bundle.exists():
        raise FileExistsError(bundle)
    contracts = load(project / "scripts/2_beagle_filtering" / CANDIDATE / "contracts.py",
                     "_freeze_source_contracts")
    screen = project / "data/experiments/beagle_v2_3_factorial_screen_20260913_v1/ms2_niter0_seed20260915"
    roots = {sample: screen / f"{sample}_sec1" for sample in ("P4", "P6")}
    manifests = {sample: source_manifest(root, contracts) for sample, root in roots.items()}
    p4, p6 = manifests["P4"], manifests["P6"]
    codes = [{Path(row["path"]).name: row["sha256"] for row in m["code"]} for m in (p4, p6)]
    if codes[0] != codes[1] or p4["executables"] != p6["executables"]:
        raise ValueError("P4/P6 runtime identities differ")
    if p4["reference_ledger"] != p6["reference_ledger"]:
        raise ValueError("P4/P6 reference ledgers differ")
    bundle.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        destination = bundle / "runtime" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(roots["P4"] / "code_snapshot/scripts/2_beagle_filtering" / name, destination)
        if digest(destination) != codes[0][Path(name).name]:
            raise ValueError("snapshot differs from validated source")
    shutil.copy2(__file__, bundle / "beagle.py")
    shutil.copy2(decision, bundle / "DECISION.md")
    profile = dict(schema=SCHEMA, status="architecture_frozen", representation="single_seed",
        parameters=PARAMETERS, anchor_score="p_nonref", min_anchor_probability=0.90,
        quality_filter="baseQ0mapQ0", production_inputs_include_wes=False,
        scientific_final_validation_complete=False,
        reference_ledger=p4["reference_ledger"]["root"], executables=p4["executables"],
        source_runs={k: str(v) for k, v in roots.items()},
        inference_view="all_beagle_nonref_highconf.vcf.gz",
        spatial_feature_view="all_panel_present.vcf.gz",
        deferred=["three_seed_selection", "posterior_ensemble", "final_module_dlpfc"])
    write_json(bundle / "PROFILE.json", profile)
    write_json(bundle / "PROVENANCE.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), decision=identity(decision),
        freezer=identity(__file__),
        evidence={sample: [identity(root / name) for name in
            ("V2_3_MANIFEST.json", "RUN_COMPLETE.json", "OUTPUT_SHA256SUMS.txt")]
            for sample, root in roots.items()},
        selection_basis="user-directed architecture freeze; prespecified fail-safe comparator",
        registered_selection_protocol_completed=False))
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    with (bundle / "SHA256SUMS.txt").open("x") as handle:
        for path in files:
            handle.write(f"{digest(path)}  {path.relative_to(bundle)}\n")
    write_json(bundle / "FREEZE_COMPLETE.json", dict(schema=SCHEMA,
        status="architecture_frozen", ledger_sha256=digest(bundle / "SHA256SUMS.txt")))
    verify(bundle)
    print(f"Frozen: {bundle}", flush=True)


def text_digest(path):
    h = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(handle):
    return (line.rstrip("\n").split("\t") for line in handle if not line.startswith("#"))


def verify_partition(raw, output):
    """Independently check every raw/audit/view row for conservation and equality."""
    with ExitStack() as stack:
        raw_rows = records(stack.enter_context(gzip.open(raw, "rt")))
        audit = csv.DictReader(stack.enter_context(
            gzip.open(output / "beagle_evidence_audit.tsv.gz", "rt")), delimiter="\t")
        views = {name: records(stack.enter_context(
            gzip.open(output / f"all_{name}.vcf.gz", "rt"))) for name in VIEWS}
        seen, counts = set(), dict.fromkeys(ROUTES, 0)
        for fields, row in zip_longest(raw_rows, audit):
            if fields is None or row is None:
                raise ValueError("raw/audit length mismatch")
            key = (fields[0], fields[1], fields[3], fields[4])
            if key in seen or key != tuple(row[k] for k in ("chrom", "pos", "ref", "alt")):
                raise ValueError("duplicate or mismatched raw/audit allele")
            seen.add(key)
            route = row["route"]
            if route not in ROUTES or row["hard_anchor"] != str(int(route == ROUTES[0])):
                raise ValueError("invalid audit route/anchor")
            counts[route] += 1
            for name, admitted in VIEWS.items():
                if route not in admitted:
                    continue
                emitted = next(views[name], None)
                if emitted is None or emitted[:7] + emitted[8:] != fields[:7] + fields[8:]:
                    raise ValueError(f"{name}: record missing, altered, or out of order")
                info = dict(item.split("=", 1) for item in emitted[7].split(";") if "=" in item)
                if info.get("BEAGLE_ROUTE") != route:
                    raise ValueError(f"{name}: route annotation disagrees with audit")
        if any(next(view, None) is not None for view in views.values()):
            raise ValueError("extra record in route/compatibility view")
    return dict(raw_candidates=len(seen), route_counts=counts, views_checked=len(VIEWS))


def replay(bundle, sample, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    profile, contracts = verify(bundle)
    root = Path(profile["source_runs"][sample])
    manifest = source_manifest(root, contracts)
    for row in manifest["code"]:
        matches = [bundle / "runtime" / name for name in SOURCES
                   if Path(name).name == Path(row["path"]).name]
        if len(matches) != 1 or digest(matches[0]) != row["sha256"]:
            raise ValueError("source run code differs from frozen runtime")
    original = root / "output_VCFs/beagle" / profile["quality_filter"]
    routing = json.loads((original / "MANIFEST.json").read_text())
    router = load(bundle / "runtime/beagle_evidence_router.py", "_frozen_replay_router")
    print(f"{sample}: source verified; replaying {routing['raw_record_count']} candidates", flush=True)
    counts = router.route_vcf(Path(manifest["raw_vcf"]["path"]),
        [Path(p) for p in routing["beagle_vcfs"]], output,
        anchor_score=profile["anchor_score"], min_anchor_probability=profile["min_anchor_probability"],
        bgzip=profile["executables"]["bgzip"]["path"], tabix=profile["executables"]["tabix"]["path"],
        expected_contigs=routing["expected_contigs"],
        contig_status_path=root / manifest["contig_status_file"])
    comparisons = {}
    for name in ["beagle_evidence_audit.tsv.gz"] + [f"all_{name}.vcf.gz" for name in VIEWS]:
        expected, observed = text_digest(original / name), text_digest(output / name)
        if observed != expected:
            raise ValueError(f"{sample}: replay differs from frozen source: {name}")
        comparisons[name] = observed
    partition = verify_partition(Path(manifest["raw_vcf"]["path"]), output)
    if counts != partition["route_counts"]:
        raise ValueError("independent partition counts disagree")
    contracts.validate_v2_3_run(root)
    verify(bundle)
    result = dict(status="passed", sample=sample, verification="saved-posterior routing replay",
        reran_beagle=False, production_inputs_include_wes=False,
        source_completion=identity(root / "RUN_COMPLETE.json"),
        bundle_completion=identity(bundle / "FREEZE_COMPLETE.json"),
        content_sha256=comparisons, **partition)
    write_json(output / "REPLAY_VALIDATION.json", result)
    with (output / "REPLAY_SHA256SUMS.txt").open("x") as handle:
        for path in sorted(p for p in output.iterdir()
                           if p.is_file() and p.name != "REPLAY_SHA256SUMS.txt"):
            handle.write(f"{digest(path)}  {path.name}\n")
    write_json(output / "REPLAY_COMPLETE.json", dict(status="passed",
        ledger_sha256=digest(output / "REPLAY_SHA256SUMS.txt")))
    print(json.dumps(result, sort_keys=True), flush=True)


def run_command(profile, runtime, dataset, section, output):
    p = profile["parameters"]
    return [sys.executable, str(runtime / CANDIDATE / "run_beagle_v2_3.py"),
        "--dataset", dataset, "--section-id", section, "--section-root", str(output),
        "--reference-ledger-root", profile["reference_ledger"],
        "--quality-filter", profile["quality_filter"], "--threads", str(p["THREADS"]),
        "--memory", p["MEMORY"], "--parallel-chromosomes", str(p["PARALLEL_CHROMOSOMES"]),
        "--iterations", str(p["ITERATIONS"]), "--seed", str(p["SEED"]),
        "--modelscale", str(p["MODEL_SCALE"]), "--maxlr", str(p["MAX_LR"]),
        "--anchor-score", profile["anchor_score"],
        "--min-anchor-probability", str(profile["min_anchor_probability"])]


def run(bundle, raw, dataset, section, output, dry_run=False):
    bundle, raw, output = Path(bundle).resolve(), Path(raw).resolve(), Path(output).resolve()
    profile, _ = verify(bundle)
    if output.exists():
        raise FileExistsError(output)
    if not raw.is_file() or not Path(str(raw) + ".tbi").is_file():
        raise FileNotFoundError("raw VCF and tabix index are required")
    runtime = output / "code_snapshot/scripts/2_beagle_filtering"
    command = run_command(profile, runtime, dataset, section, output)
    if dry_run:
        print(json.dumps(command))
        return
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "runtime" / name, target)
    inputs = output / "output_VCFs/mpileup_multi_bam" / profile["quality_filter"]
    inputs.mkdir(parents=True)
    (inputs / "merged_sorted_gt.vcf.gz").symlink_to(raw)
    (inputs / "merged_sorted_gt.vcf.gz.tbi").symlink_to(str(raw) + ".tbi")
    write_json(output / "FROZEN_PROFILE.json", profile)
    write_json(output / "FREEZE_ORIGIN.json", identity(bundle / "FREEZE_COMPLETE.json"))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--decision", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    p = sub.add_parser("replay")
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--sample", choices=("P4", "P6"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--raw-vcf", type=Path, required=True)
    p.add_argument("--dataset", choices=("DLPFC", "P4_TUMOR", "P6_TUMOR", "DCIS", "OVAR_P5", "NCCE"), required=True)
    p.add_argument("--section-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.project, args.bundle, args.decision)
    elif args.command == "verify":
        profile, _ = verify(args.bundle)
        print(json.dumps(profile, indent=2, sort_keys=True))
    elif args.command == "replay":
        replay(args.bundle, args.sample, args.output)
    else:
        run(args.bundle, args.raw_vcf, args.dataset, args.section_id, args.output, args.dry_run)


if __name__ == "__main__":
    main()
