#!/usr/bin/env python3
"""Materialize a validated Beagle-v2.3 posterior ensemble as route VCFs.

The existing ensemble builder intentionally emits a lossless aggregate audit.
This release-side adapter turns a selected mean/median audit into the same four
mutually exclusive VCF routes and compatibility views as a single v2.3 run.
It revalidates the three source runs and every aggregate row, never accepts WES,
and writes only to a fresh output directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


PROJECT = Path(__file__).resolve().parents[3]
CANDIDATE = Path(__file__).resolve().parent
DEFAULT_ROUTER = CANDIDATE.parent / "beagle_evidence_router.py"
DEFAULT_CONTRACTS = CANDIDATE / "contracts.py"
DEFAULT_PROTOCOL = CANDIDATE / "RELEASE_PROTOCOL.md"
DEFAULT_PROTOCOL_CHECKSUM = CANDIDATE / "RELEASE_PROTOCOL_SHA256.txt"

SCHEMA = "sparcal_beagle_v2_3_ensemble_route_bundle_v1"
PRODUCTION_METHODS = ("ensemble_mean", "ensemble_median")
ROUTE_FILES = {
    "beagle_nonref_highconf": "all_beagle_nonref_highconf.vcf",
    "beagle_ref_highconf": "all_beagle_ref_highconf.vcf",
    "beagle_uncertain": "all_beagle_uncertain.vcf",
    "panel_absent": "all_panel_absent.vcf",
    "filtered_in": "all_filtered_in.vcf",
    "filtered_out": "all_filtered_out.vcf",
    "panel_discordant": "all_panel_discordant.vcf",
    "panel_present": "all_panel_present.vcf",
}


def load_module(path: Path, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def content_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"path", "mtime_ns"}
    }


def strict_optional_float(text: object, label: str) -> Optional[float]:
    if text in {None, "", "."}:
        return None
    try:
        value = float(str(text))
    except ValueError as error:
        raise ValueError(f"invalid {label}: {text!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {text!r}")
    return value


def close_number(observed: Optional[float], expected: float, label: str) -> None:
    if observed is None or not math.isclose(
        observed, expected, rel_tol=1e-10, abs_tol=5e-12
    ):
        raise ValueError(f"{label} disagrees with aggregate GP")


def iter_raw_records(raw_vcf: Path, router):
    header_seen = False
    with router.open_text(raw_vcf) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    if header_seen:
                        raise ValueError(f"{raw_vcf}: duplicate #CHROM header")
                    header_seen = True
                continue
            if not header_seen:
                raise ValueError(f"{raw_vcf}: data precedes #CHROM header")
            fields = line.rstrip("\n").split("\t")
            router._validate_vcf_data_fields(fields, raw_vcf, line_number)
            yield line_number, line, fields
    if not header_seen:
        raise ValueError(f"{raw_vcf}: missing #CHROM header")


def iter_audit_records(audit: Path, ensemble):
    with gzip.open(audit, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(ensemble.AUDIT_COLUMNS):
            raise ValueError("ensemble audit has a non-exact schema")
        yield from enumerate(reader, start=2)


def validate_row(
    fields: Sequence[str], row: Mapping[str, str], *, router, ensemble
) -> tuple[str, Mapping[str, str]]:
    key = (fields[0], fields[1], fields[3], fields[4])
    audit_key = tuple(row[name] for name in ("chrom", "pos", "ref", "alt"))
    if key != audit_key:
        raise ValueError(f"raw/audit allele mismatch: {key} != {audit_key}")
    if row.get("schema") != ensemble.AUDIT_SCHEMA:
        raise ValueError(f"invalid ensemble audit row schema at {key}")

    raw_gt = router.normalize_gt(
        router.extract_format_value(fields[8], fields[9], "GT")
    )
    if (raw_gt or ".") != row.get("raw_gt"):
        raise ValueError(f"raw GT differs from audit at {key}")
    for info_item in fields[7].split(";"):
        if info_item.startswith("BEAGLE_"):
            raise ValueError(f"raw VCF already contains Beagle annotations at {key}")

    route = str(row.get("route", ""))
    if route not in router.ROUTES:
        raise ValueError(f"invalid ensemble route at {key}: {route!r}")
    expected_hard_anchor = "1" if route == "beagle_nonref_highconf" else "0"
    if row.get("hard_anchor") != expected_hard_anchor:
        raise ValueError(f"hard-anchor flag disagrees with route at {key}")

    gp_text = row.get("beagle_gp", ".")
    gp = ensemble.parse_probability_triplet(gp_text)
    gt_text = row.get("beagle_gt", ".")
    called_gp = strict_optional_float(row.get("beagle_called_gp"), "called GP")
    p_nonref = strict_optional_float(row.get("beagle_p_nonref"), "P(nonref)")
    entropy = strict_optional_float(row.get("beagle_gp_entropy"), "GP entropy")
    margin = strict_optional_float(row.get("beagle_gp_margin"), "GP margin")
    beagle_af = strict_optional_float(row.get("beagle_af"), "Beagle AF")

    derived = (called_gp, p_nonref, entropy, margin)
    if gp is None:
        if gp_text != "." or gt_text != "." or any(value is not None for value in derived):
            raise ValueError(f"incomplete/malformed missing-GP fields at {key}")
        if beagle_af is not None:
            raise ValueError(f"missing-GP row unexpectedly contains Beagle AF at {key}")
        if route not in {"beagle_uncertain", "panel_absent"}:
            raise ValueError(f"missing-GP row has a confident route at {key}")
    else:
        if any(value < 0 for value in gp):
            raise ValueError(f"negative aggregate GP at {key}")
        gt_index = max(range(3), key=lambda index: (gp[index], -index))
        expected_gt = ("0/0", "0/1", "1/1")[gt_index]
        if gt_text != expected_gt:
            raise ValueError(f"aggregate GT is not the GP argmax at {key}")
        close_number(called_gp, gp[gt_index], f"called GP at {key}")
        if p_nonref is None or not -1e-12 <= p_nonref <= 1 + 1e-12:
            raise ValueError(f"invalid P(nonref) range at {key}")
        total = sum(gp)
        normalized = tuple(value / total for value in gp)
        expected_entropy = -sum(
            value * math.log(value) for value in normalized if value > 0
        ) / math.log(3)
        ordered = sorted(gp, reverse=True)
        close_number(entropy, expected_entropy, f"entropy at {key}")
        close_number(margin, ordered[0] - ordered[1], f"margin at {key}")
        if beagle_af is not None and not -1e-12 <= beagle_af <= 1 + 1e-12:
            raise ValueError(f"invalid Beagle AF range at {key}")
        if ensemble.meets_threshold(p_nonref):
            expected_route = "beagle_nonref_highconf"
        elif ensemble.meets_threshold(gp[0]):
            expected_route = "beagle_ref_highconf"
        else:
            expected_route = "beagle_uncertain"
        if route != expected_route:
            raise ValueError(f"route disagrees with ensemble posterior at {key}")

    transition = f"{raw_gt or '.'}>{gt_text}"
    if row.get("gt_transition") != transition:
        raise ValueError(f"GT transition disagrees with raw/ensemble GT at {key}")

    annotations = {
        "BEAGLE_EVIDENCE_SCHEMA": ensemble.AUDIT_SCHEMA,
        "BEAGLE_ROUTE": route,
        "BEAGLE_GT": gt_text,
        "BEAGLE_GP": gp_text,
        "BEAGLE_CALLED_GP": row.get("beagle_called_gp", "."),
        # Preserve the directly aggregated P(nonref). For the median method it
        # need not equal median(GP01)+median(GP11).
        "BEAGLE_P_NONREF": row.get("beagle_p_nonref", "."),
        "BEAGLE_GP_ENTROPY": row.get("beagle_gp_entropy", "."),
        "BEAGLE_GP_MARGIN": row.get("beagle_gp_margin", "."),
        "BEAGLE_AF": row.get("beagle_af", "."),
        "BEAGLE_GT_TRANSITION": transition,
    }
    return route, annotations


def validate_alignment(
    raw_vcf: Path,
    audit: Path,
    *,
    router,
    ensemble,
    expected_counts: Optional[Mapping[str, int]] = None,
) -> tuple[Mapping[str, int], int]:
    router._scan_single_sample_vcf(raw_vcf)
    counts = {route: 0 for route in router.ROUTES}
    sentinel = object()
    paired: Iterable[tuple[object, object]] = zip_longest(
        iter_raw_records(raw_vcf, router),
        iter_audit_records(audit, ensemble),
        fillvalue=sentinel,
    )
    for raw_item, audit_item in paired:
        if raw_item is sentinel or audit_item is sentinel:
            raise ValueError("raw VCF and ensemble audit have different row counts")
        _, _, fields = raw_item  # type: ignore[misc]
        _, row = audit_item  # type: ignore[misc]
        route, _ = validate_row(fields, row, router=router, ensemble=ensemble)
        counts[route] += 1
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("raw VCF and ensemble audit contain no records")
    if expected_counts is not None:
        expected = {route: int(expected_counts[route]) for route in router.ROUTES}
        expected_total = int(expected_counts["total"])
        if counts != expected or total != expected_total:
            raise ValueError("ensemble manifest counts disagree with raw/audit rows")
    return counts, total


def annotate_raw_line(
    line: str, annotations: Mapping[str, str]
) -> str:
    fields = line.rstrip("\n").split("\t")
    addition = ";".join(f"{key}={value}" for key, value in annotations.items())
    fields[7] = addition if fields[7] == "." else fields[7] + ";" + addition
    return "\t".join(fields) + "\n"


def write_uncompressed_routes(
    *,
    raw_vcf: Path,
    audit: Path,
    output_dir: Path,
    router,
    ensemble,
) -> tuple[Mapping[str, int], Mapping[str, Path]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {name: output_dir / filename for name, filename in ROUTE_FILES.items()}
    counts = {route: 0 for route in router.ROUTES}
    handles = {name: path.open("x") for name, path in paths.items()}
    try:
        audit_iter = iter(iter_audit_records(audit, ensemble))
        header_seen = False
        with router.open_text(raw_vcf) as raw_handle:
            for line_number, line in enumerate(raw_handle, start=1):
                if line.startswith("#"):
                    if line.startswith("#CHROM"):
                        if header_seen:
                            raise ValueError(f"{raw_vcf}: duplicate #CHROM header")
                        header_seen = True
                        for handle in handles.values():
                            handle.writelines(
                                router._meta_headers(
                                    ensemble.AUDIT_SCHEMA, "p_nonref", ensemble.THRESHOLD
                                )
                            )
                    for handle in handles.values():
                        handle.write(line)
                    continue
                if not header_seen:
                    raise ValueError(f"{raw_vcf}: data precedes #CHROM header")
                fields = line.rstrip("\n").split("\t")
                router._validate_vcf_data_fields(fields, raw_vcf, line_number)
                try:
                    _, row = next(audit_iter)
                except StopIteration as error:
                    raise ValueError("ensemble audit ends before raw VCF") from error
                route, annotations = validate_row(
                    fields, row, router=router, ensemble=ensemble
                )
                routed_line = annotate_raw_line(line, annotations)
                handles[route].write(routed_line)
                for compatibility in router.compatibility_outputs(route):
                    handles[compatibility].write(routed_line)
                counts[route] += 1
        try:
            next(audit_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("ensemble audit contains rows after raw VCF ends")
    finally:
        for handle in handles.values():
            handle.close()
    return counts, paths


def validate_materialized_routes(
    *,
    raw_vcf: Path,
    audit: Path,
    route_paths: Mapping[str, Path],
    router,
    ensemble,
) -> Mapping[str, int]:
    """Prove every compressed VCF is the exact routed projection of raw+audit."""
    if set(route_paths) != set(ROUTE_FILES):
        raise ValueError("materialized route-file set is not exact")
    expected_by_route = {
        route: {route, *router.compatibility_outputs(route)}
        for route in router.ROUTES
    }

    def data_lines(path: Path):
        header_seen = False
        schema_header_seen = False
        with router.open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.startswith("##SPARCAL_BEAGLE_ROUTER="):
                    schema_header_seen = ensemble.AUDIT_SCHEMA in line
                if line.startswith("#"):
                    if line.startswith("#CHROM"):
                        if header_seen:
                            raise ValueError(f"{path}: duplicate #CHROM header")
                        header_seen = True
                    continue
                if not header_seen or not schema_header_seen:
                    raise ValueError(f"{path}: incomplete ensemble-route header")
                fields = line.rstrip("\n").split("\t")
                router._validate_vcf_data_fields(fields, path, line_number)
                yield line
        if not header_seen or not schema_header_seen:
            raise ValueError(f"{path}: incomplete ensemble-route header")

    output_counts = {name: 0 for name in ROUTE_FILES}
    sentinel = object()
    with ExitStack() as stack:
        # Generators own their gzip handles; closing them explicitly ensures
        # descriptors are released even if a comparison fails midway.
        streams = {
            name: stack.enter_context(_closing_generator(data_lines(path)))
            for name, path in route_paths.items()
        }
        aligned: Iterable[tuple[object, object]] = zip_longest(
            iter_raw_records(raw_vcf, router),
            iter_audit_records(audit, ensemble),
            fillvalue=sentinel,
        )
        for raw_item, audit_item in aligned:
            if raw_item is sentinel or audit_item is sentinel:
                raise ValueError("raw VCF and ensemble audit changed during VCF validation")
            _, raw_line, fields = raw_item  # type: ignore[misc]
            _, row = audit_item  # type: ignore[misc]
            route, annotations = validate_row(
                fields, row, router=router, ensemble=ensemble
            )
            expected_line = annotate_raw_line(raw_line, annotations)
            for name in expected_by_route[route]:
                try:
                    observed = next(streams[name])
                except StopIteration as error:
                    raise ValueError(f"materialized {name} VCF ends early") from error
                if observed != expected_line:
                    raise ValueError(f"materialized {name} VCF differs from raw/audit")
                output_counts[name] += 1
        for name, stream in streams.items():
            try:
                next(stream)
            except StopIteration:
                continue
            raise ValueError(f"materialized {name} VCF contains extra records")
    return output_counts


class _closing_generator:
    """Small context manager for a generator that may own an open VCF handle."""

    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        return self.generator

    def __exit__(self, exc_type, exc_value, traceback):
        self.generator.close()
        return False


def snapshot_ensemble_module(manifest: Mapping[str, object], contracts) -> Path:
    identities = manifest.get("code_snapshot")
    if not isinstance(identities, list) or len(identities) != 3:
        raise ValueError("ensemble bundle has an invalid code snapshot")
    observed = {}
    for index, record in enumerate(identities):
        path = contracts.verify_file_identity(record, label=f"ensemble snapshot {index}")
        observed[path.name] = path
    try:
        return observed["beagle_posterior_ensemble.py"]
    except KeyError as error:
        raise ValueError("ensemble snapshot lacks its aggregation module") from error


def validate_ensemble_bundle(
    root: Path, *, contracts_path: Path, ensemble_path: Optional[Path] = None
) -> Mapping[str, object]:
    root = Path(root).resolve()
    contracts = load_module(contracts_path, "_ensemble_route_contracts")
    verified = contracts.verify_complete_bundle(root)
    manifest_path = Path(str(verified["manifest"])).resolve()
    expected_manifest = root / "beagle_evidence_v2_3_ensemble/MANIFEST.json"
    if manifest_path != expected_manifest:
        raise ValueError("ensemble completion points to a noncanonical manifest")
    manifest = json.loads(manifest_path.read_text())
    method = str(manifest.get("method", ""))
    if (
        manifest.get("schema") != "sparcal_beagle_posterior_ensemble_v2_3_candidate_1"
        or manifest.get("status") != "outputs_ready_for_checksum"
        or method not in PRODUCTION_METHODS
        or manifest.get("aggregation") != method.removeprefix("ensemble_")
        or manifest.get("production_inputs_include_wes") is not False
        or manifest.get("evaluation_uses_wes") is not False
        or manifest.get("anchor_score") != "p_nonref"
        or float(manifest.get("min_anchor_probability", -1)) != 0.90
        or manifest.get("hard_anchor_route") != "beagle_nonref_highconf"
        or manifest.get("frozen_source_seeds") != [20260915, 20260916, 20260917]
    ):
        raise ValueError("invalid production ensemble manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "beagle_nonref_highconf", "beagle_ref_highconf",
        "beagle_uncertain", "panel_absent", "total",
    }:
        raise ValueError("ensemble manifest has an invalid count contract")

    snapshot_module = snapshot_ensemble_module(manifest, contracts)
    if ensemble_path is not None and sha256(ensemble_path) != sha256(snapshot_module):
        raise ValueError("requested ensemble module differs from immutable snapshot")
    ensemble = load_module(snapshot_module, "_ensemble_route_aggregation")

    source_audit_records = manifest.get("source_audits")
    source_roots = manifest.get("source_run_roots")
    if not isinstance(source_audit_records, dict) or not isinstance(source_roots, dict):
        raise ValueError("ensemble manifest lacks source runs/audits")
    seed_keys = {str(seed) for seed in ensemble.SEEDS}
    if set(source_audit_records) != seed_keys or set(source_roots) != seed_keys:
        raise ValueError("ensemble source seed grid is not exact")

    source_audits = {}
    raw_identities = []
    evidence_paths = [
        manifest_path,
        root / "RUN_COMPLETE.json",
        root / "OUTPUT_SHA256SUMS.txt",
        root / "RUN_COMPLETE",
    ]
    for seed in ensemble.SEEDS:
        key = str(seed)
        audit_path = contracts.verify_file_identity(
            source_audit_records[key], label=f"source audit seed {seed}"
        )
        source_audits[seed] = audit_path
        source_root = Path(str(source_roots[key])).resolve()
        contracts.validate_v2_3_run(source_root)
        source_manifest_path = source_root / "V2_3_MANIFEST.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        raw_path = contracts.verify_file_identity(
            source_manifest.get("raw_vcf"), label=f"source raw VCF seed {seed}"
        )
        raw_identities.append(source_manifest["raw_vcf"])
        evidence_paths.extend((
            audit_path,
            source_manifest_path,
            source_root / "RUN_COMPLETE.json",
            source_root / "OUTPUT_SHA256SUMS.txt",
        ))
    if len({json.dumps(content_identity(record), sort_keys=True) for record in raw_identities}) != 1:
        raise ValueError("ensemble sources do not share one raw VCF content identity")
    raw_vcf = Path(str(raw_identities[0]["path"])).resolve()

    audit = root / "beagle_evidence_v2_3_ensemble/beagle_evidence_audit.tsv.gz"
    if not audit.is_file():
        raise FileNotFoundError(audit)
    validation = ensemble.validate_ensemble_audit(
        audit=audit, source_audits=source_audits, method_name=method
    )
    if validation.get("counts") != counts:
        raise ValueError("recomputed ensemble counts disagree with manifest")
    return {
        "contracts": contracts,
        "ensemble": ensemble,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "audit": audit,
        "raw_vcf": raw_vcf,
        "evidence_paths": tuple(dict.fromkeys(Path(path).resolve() for path in evidence_paths)),
    }


def write_external_ledger(destination: Path, paths: Iterable[Path]) -> Path:
    rows = {}
    for value in paths:
        path = Path(value).resolve()
        rows[str(path)] = sha256(path)
    with destination.open("x") as handle:
        for path, digest in sorted(rows.items()):
            handle.write(f"{digest}  {path}\n")
    return destination


def validate_protocol_checksum(protocol: Path, checksum_file: Path) -> None:
    protocol = Path(protocol).resolve()
    checksum_file = Path(checksum_file).resolve()
    lines = checksum_file.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("release protocol checksum file must contain exactly one row")
    parts = lines[0].split("  ", 1)
    if (
        len(parts) != 2
        or len(parts[0]) != 64
        or parts[1] != protocol.name
        or sha256(protocol) != parts[0]
    ):
        raise ValueError("release protocol checksum mismatch")


def materialize(
    *,
    ensemble_root: Path,
    output_dir: Path,
    router_path: Path = DEFAULT_ROUTER,
    ensemble_path: Optional[Path] = None,
    contracts_path: Path = DEFAULT_CONTRACTS,
    protocol_path: Path = DEFAULT_PROTOCOL,
    protocol_checksum_path: Path = DEFAULT_PROTOCOL_CHECKSUM,
    bgzip: str = "bgzip",
    tabix: str = "tabix",
) -> Mapping[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    for path in (
        router_path, contracts_path, protocol_path, protocol_checksum_path,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    validate_protocol_checksum(protocol_path, protocol_checksum_path)
    bundle = validate_ensemble_bundle(
        ensemble_root, contracts_path=contracts_path, ensemble_path=ensemble_path
    )
    contracts = bundle["contracts"]
    ensemble = bundle["ensemble"]
    router = load_module(router_path, "_ensemble_route_router")
    raw_vcf = Path(bundle["raw_vcf"])
    audit = Path(bundle["audit"])
    manifest = bundle["manifest"]
    initial_raw_identity = identity(raw_vcf)
    initial_audit_identity = identity(audit)
    counts, total = validate_alignment(
        raw_vcf,
        audit,
        router=router,
        ensemble=ensemble,
        expected_counts=manifest["counts"],
    )

    try:
        written_counts, route_paths = write_uncompressed_routes(
            raw_vcf=raw_vcf,
            audit=audit,
            output_dir=output_dir,
            router=router,
            ensemble=ensemble,
        )
        if written_counts != counts:
            raise AssertionError("route counts changed between validation and writing")

        bgzip_path = shutil.which(bgzip) or bgzip
        tabix_path = shutil.which(tabix) or tabix
        compressed_paths = []
        for path in route_paths.values():
            subprocess.run([bgzip_path, "-f", str(path)], check=True)
            compressed = Path(str(path) + ".gz")
            subprocess.run([tabix_path, "-f", "-p", "vcf", str(compressed)], check=True)
            index = Path(str(compressed) + ".tbi")
            if not compressed.is_file() or not index.is_file():
                raise FileNotFoundError(f"compression/index output missing for {path}")
            compressed_paths.extend((compressed, index))

        compressed_route_paths = {
            name: Path(str(path) + ".gz") for name, path in route_paths.items()
        }

        audit_copy = output_dir / "beagle_evidence_audit.tsv.gz"
        shutil.copy2(audit, audit_copy)
        if sha256(audit_copy) != initial_audit_identity["sha256"]:
            raise ValueError("ensemble audit changed while it was copied")
        output_counts = validate_materialized_routes(
            raw_vcf=raw_vcf,
            audit=audit_copy,
            route_paths=compressed_route_paths,
            router=router,
            ensemble=ensemble,
        )
        expected_output_counts = {
            "beagle_nonref_highconf": counts["beagle_nonref_highconf"],
            "beagle_ref_highconf": counts["beagle_ref_highconf"],
            "beagle_uncertain": counts["beagle_uncertain"],
            "panel_absent": counts["panel_absent"],
            "filtered_in": counts["beagle_nonref_highconf"],
            "filtered_out": total - counts["beagle_nonref_highconf"],
            "panel_discordant": counts["beagle_ref_highconf"] + counts["beagle_uncertain"],
            "panel_present": total - counts["panel_absent"],
        }
        if output_counts != expected_output_counts:
            raise AssertionError("materialized compatibility VCF counts are inconsistent")

        summary = output_dir / "beagle_evidence_summary.tsv"
        ensemble.write_summary(summary, counts, total)

        if identity(raw_vcf) != initial_raw_identity:
            raise ValueError("raw VCF changed during ensemble route materialization")
        if identity(audit) != initial_audit_identity:
            raise ValueError("ensemble audit changed during route materialization")

        snapshot_dir = output_dir / "code_snapshot"
        snapshot_dir.mkdir(exist_ok=False)
        snapshot_files = []
        sources = (
            Path(__file__), Path(router_path), Path(bundle["ensemble"].__file__),
            Path(contracts_path), Path(protocol_path), Path(protocol_checksum_path),
        )
        for source in sources:
            destination = snapshot_dir / Path(source).name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
            snapshot_files.append(destination)

        code_input_ledger = write_external_ledger(
            output_dir / "CODE_AND_INPUT_SHA256SUMS.txt",
            (*sources, *bundle["evidence_paths"], raw_vcf, audit),
        )
        manifest_path = output_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps({
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "outputs_ready_for_checksum",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "emits_phased_target_gt": False,
            "impute_effective": False,
            "method": manifest["method"],
            "aggregation": manifest["aggregation"],
            "frozen_source_seeds": manifest["frozen_source_seeds"],
            "anchor_score": "p_nonref",
            "min_anchor_probability": 0.90,
            "hard_anchor_route": "beagle_nonref_highconf",
            "raw_vcf": identity(raw_vcf),
            "source_ensemble_audit": identity(audit),
            "source_ensemble_manifest": identity(bundle["manifest_path"]),
            "raw_record_count": total,
            "counts": {**counts, "total": total},
            "route_outputs": {
                name: identity(compressed_route_paths[name])
                for name in route_paths
            },
            "compatibility": {
                "all_filtered_in.vcf.gz": "beagle_nonref_highconf only",
                "all_filtered_out.vcf.gz": "union of the other three routes",
                "all_panel_discordant.vcf.gz": "union of reference-high-confidence and uncertain panel-present routes",
                "all_panel_present.vcf.gz": "union of all three panel-present routes; feature-support view only, not a germline callset",
            },
            "code_snapshot": [identity(path) for path in snapshot_files],
            "code_and_input_ledger": identity(code_input_ledger),
        }, indent=2, sort_keys=True) + "\n")
        simple_completion = output_dir / "RUN_COMPLETE"
        simple_completion.write_text("complete\n")
        output_ledger = contracts.write_sha256sums(
            output_dir,
            (
                *compressed_paths,
                audit_copy,
                summary,
                manifest_path,
                code_input_ledger,
                simple_completion,
                *snapshot_files,
            ),
        )
        completion = contracts.write_completion(
            output_dir,
            manifest_path=manifest_path,
            checksum_path=output_ledger,
        )
    except Exception:
        # Preserve partial forensic output. The absence of both completion
        # records makes it ineligible for discovery or release.
        raise
    return {
        "schema": SCHEMA,
        "status": "complete",
        "production_inputs_include_wes": False,
        "method": manifest["method"],
        "output_dir": str(output_dir.resolve()),
        "counts": {**counts, "total": total},
        "completion": str(completion.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument(
        "--ensemble-module",
        type=Path,
        default=None,
        help="optional expected module; immutable ensemble snapshot is always executed",
    )
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--protocol-checksum", type=Path, default=DEFAULT_PROTOCOL_CHECKSUM
    )
    parser.add_argument("--bgzip", default="bgzip")
    parser.add_argument("--tabix", default="tabix")
    args = parser.parse_args()
    result = materialize(
        ensemble_root=args.ensemble_root,
        output_dir=args.output_dir,
        router_path=args.router,
        ensemble_path=args.ensemble_module,
        contracts_path=args.contracts,
        protocol_path=args.protocol,
        protocol_checksum_path=args.protocol_checksum,
        bgzip=args.bgzip,
        tabix=args.tabix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
