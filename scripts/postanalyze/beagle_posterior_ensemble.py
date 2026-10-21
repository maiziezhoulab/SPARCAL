#!/usr/bin/env python3
"""Build immutable, WES-free posterior ensembles from three Beagle v2.3 runs.

The program streams the lossless routing audits in lockstep, so full P6 inputs
do not need to be loaded into memory.  It fails closed unless the three source
runs are valid v2.3 bundles with identical non-seed parameters, identical raw
input identity, and exactly aligned allele/panel-presence records.

One invocation writes the two prespecified three-seed summaries (componentwise
mean and median GP) plus the three leave-one-seed-out componentwise means used
only for the frozen stability gate.  Matched WES is neither accepted nor read.
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
import statistics
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


SCHEMA = "sparcal_beagle_posterior_ensemble_v2_3_candidate_1"
AUDIT_SCHEMA = "sparcal_beagle_evidence_v2_3_ensemble"
SEEDS = (20260915, 20260916, 20260917)
THRESHOLD = 0.90
ROUTES = (
    "beagle_nonref_highconf",
    "beagle_ref_highconf",
    "beagle_uncertain",
    "panel_absent",
)
AUDIT_COLUMNS = (
    "schema",
    "chrom",
    "pos",
    "ref",
    "alt",
    "raw_gt",
    "beagle_gt",
    "beagle_gp",
    "beagle_called_gp",
    "beagle_p_nonref",
    "beagle_gp_entropy",
    "beagle_gp_margin",
    "beagle_af",
    "gt_transition",
    "route",
    "hard_anchor",
)
METHODS = {
    "ensemble_mean": ("mean", SEEDS),
    "ensemble_median": ("median", SEEDS),
    f"ensemble_mean_loo_seed{SEEDS[0]}": ("mean", SEEDS[1:]),
    f"ensemble_mean_loo_seed{SEEDS[1]}": ("mean", (SEEDS[0], SEEDS[2])),
    f"ensemble_mean_loo_seed{SEEDS[2]}": ("mean", SEEDS[:2]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def load_contracts(path: Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "_sparcal_beagle_ensemble_contracts", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import v2.3 contracts module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_root(audit: Path) -> Path:
    for parent in Path(audit).resolve().parents:
        if (
            (parent / "V2_3_MANIFEST.json").is_file()
            and (parent / "RUN_COMPLETE.json").is_file()
        ):
            return parent
    raise ValueError(f"audit is not inside a completed v2.3 run: {audit}")


def parse_seed_audits(values: Sequence[str]) -> dict[int, Path]:
    audits: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"--audit must be SEED=PATH, found {value!r}") from error
        if seed in audits:
            raise ValueError(f"duplicate seed: {seed}")
        path = Path(path_text).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        audits[seed] = path
    if set(audits) != set(SEEDS):
        raise ValueError(f"exactly the frozen seeds {SEEDS} are required")
    return audits


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _content_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Compare copied artifacts by content, not run-specific path/timestamp."""
    return {
        key: item
        for key, item in value.items()
        if key not in {"path", "mtime_ns"}
    }


def validate_sources(
    audits: Mapping[int, Path], contracts
) -> tuple[dict[int, Mapping[str, object]], dict[int, Path]]:
    manifests: dict[int, Mapping[str, object]] = {}
    roots: dict[int, Path] = {}
    invariant_params: Optional[str] = None
    invariant_raw: Optional[str] = None
    invariant_dataset: Optional[object] = None
    invariant_quality: Optional[object] = None
    invariant_code: Optional[str] = None
    invariant_executables: Optional[str] = None
    invariant_reference: Optional[str] = None

    for seed in SEEDS:
        audit = audits[seed]
        root = source_root(audit)
        contracts.validate_v2_3_run(root)
        manifest_path = root / "V2_3_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        expected_audit = (
            root
            / "output_VCFs"
            / "beagle"
            / str(manifest.get("quality_filter"))
            / "beagle_evidence_audit.tsv.gz"
        ).resolve()
        if audit != expected_audit:
            raise ValueError(
                f"input must be the canonical v2.3 route audit: "
                f"{audit} != {expected_audit}"
            )
        params = dict(manifest["beagle_parameters"])
        observed_seed = params.pop("SEED", None)
        if observed_seed != seed:
            raise ValueError(
                f"source label seed {seed} differs from manifest seed {observed_seed}"
            )
        if manifest.get("production_inputs_include_wes") is not False:
            raise ValueError(f"source run unexpectedly uses WES: {root}")
        if manifest.get("evaluation_uses_wes") is not False:
            raise ValueError(f"source run unexpectedly evaluates with WES: {root}")
        if manifest.get("anchor_score") != "p_nonref":
            raise ValueError(f"source run has wrong biological router: {root}")
        if not math.isclose(
            float(manifest.get("min_anchor_probability", -1)),
            THRESHOLD,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"source run has wrong anchor threshold: {root}")

        raw = _content_identity(manifest["raw_vcf"])
        code = [
            _content_identity(row)
            for row in manifest["code"]
        ]
        executables = {
            name: _content_identity(row)
            for name, row in manifest["executables"].items()
        }
        reference = {
            "reference_ledger": manifest["reference_ledger"],
            "reference_panels": manifest["reference_panels"],
        }
        comparisons = (
            ("non-seed Beagle parameters", _stable_json(params), invariant_params),
            ("raw input identity", _stable_json(raw), invariant_raw),
            ("dataset", _stable_json(manifest.get("dataset")), invariant_dataset),
            (
                "quality filter",
                _stable_json(manifest.get("quality_filter")),
                invariant_quality,
            ),
            ("code identity", _stable_json(code), invariant_code),
            ("executable identity", _stable_json(executables), invariant_executables),
            ("reference identity", _stable_json(reference), invariant_reference),
        )
        if seed == SEEDS[0]:
            invariant_params = comparisons[0][1]
            invariant_raw = comparisons[1][1]
            invariant_dataset = comparisons[2][1]
            invariant_quality = comparisons[3][1]
            invariant_code = comparisons[4][1]
            invariant_executables = comparisons[5][1]
            invariant_reference = comparisons[6][1]
        else:
            for label, observed, expected in comparisons:
                if observed != expected:
                    raise ValueError(f"{label} differs across seeds at seed {seed}")

        manifests[seed] = manifest
        roots[seed] = root
    if len(set(roots.values())) != len(SEEDS):
        raise ValueError("each seed must come from a distinct immutable run root")
    return manifests, roots


def parse_probability_triplet(text: object) -> Optional[tuple[float, float, float]]:
    if text in {None, "", "."}:
        return None
    try:
        values = tuple(float(value) for value in str(text).split(","))
    except ValueError:
        return None
    if (
        len(values) != 3
        or any(not math.isfinite(value) or value < 0 for value in values)
        or sum(values) <= 0
    ):
        return None
    return values  # type: ignore[return-value]


def parse_optional_float(text: object) -> Optional[float]:
    if text in {None, "", "."}:
        return None
    try:
        value = float(str(text))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def aggregate(values: Sequence[float], method: str) -> float:
    if method == "mean":
        return sum(values) / len(values)
    if method == "median":
        return statistics.median(values)
    raise ValueError(f"unknown aggregation method: {method}")


def number(value: Optional[float]) -> str:
    return "." if value is None else f"{value:.12g}"


def meets_threshold(value: Optional[float], threshold: float = THRESHOLD) -> bool:
    return value is not None and (
        value >= threshold
        or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)
    )


def summarize_row(
    rows_by_seed: Mapping[int, Mapping[str, str]],
    method: str,
    included_seeds: Sequence[int],
) -> tuple[list[str], str]:
    first = rows_by_seed[SEEDS[0]]
    key = tuple(first[name] for name in ("chrom", "pos", "ref", "alt"))
    raw_gt = first["raw_gt"]
    panel_absent = first["route"] == "panel_absent"
    for seed in SEEDS[1:]:
        row = rows_by_seed[seed]
        observed_key = tuple(row[name] for name in ("chrom", "pos", "ref", "alt"))
        if observed_key != key:
            raise ValueError(
                f"allele order/content differs at {key} versus seed {seed}: "
                f"{observed_key}"
            )
        if row["raw_gt"] != raw_gt:
            raise ValueError(f"raw GT differs across seeds at {key}")
        if (row["route"] == "panel_absent") != panel_absent:
            raise ValueError(f"panel presence differs across seeds at {key}")

    if panel_absent:
        gp = None
        af = None
        gt = "."
        called_gp = p_nonref = entropy = margin = None
        route = "panel_absent"
    else:
        parsed = {
            seed: parse_probability_triplet(rows_by_seed[seed]["beagle_gp"])
            for seed in included_seeds
        }
        if any(value is None for value in parsed.values()):
            # Fail conservative at the allele level: malformed posterior evidence
            # cannot become either a positive or negative high-confidence route.
            gp = None
            af = None
            gt = "."
            called_gp = p_nonref = entropy = margin = None
            route = "beagle_uncertain"
        else:
            gp = tuple(
                aggregate(
                    [parsed[seed][index] for seed in included_seeds],  # type: ignore[index]
                    method,
                )
                for index in range(3)
            )
            # The frozen rule operates on P(nonref) itself.  This matters for
            # the median: median(GP01)+median(GP11) need not equal
            # median(GP01+GP11).  The displayed GP remains the componentwise
            # summary, while routing and this audit field use the direct class
            # posterior summary.
            p_nonref = aggregate(
                [
                    parsed[seed][1] + parsed[seed][2]  # type: ignore[index]
                    for seed in included_seeds
                ],
                method,
            )
            p_ref = aggregate(
                [parsed[seed][0] for seed in included_seeds],  # type: ignore[index]
                method,
            )
            af_values = [
                parse_optional_float(rows_by_seed[seed]["beagle_af"])
                for seed in included_seeds
            ]
            af = (
                aggregate([value for value in af_values if value is not None], method)
                if all(value is not None for value in af_values)
                else None
            )
            gt_index = max(range(3), key=lambda index: (gp[index], -index))
            gt = ("0/0", "0/1", "1/1")[gt_index]
            called_gp = gp[gt_index]
            total = sum(gp)
            normalized = tuple(value / total for value in gp)
            entropy = -sum(
                value * math.log(value)
                for value in normalized
                if value > 0
            ) / math.log(3)
            ordered = sorted(gp, reverse=True)
            margin = ordered[0] - ordered[1]
            if meets_threshold(p_nonref):
                route = "beagle_nonref_highconf"
            elif meets_threshold(p_ref):
                route = "beagle_ref_highconf"
            else:
                route = "beagle_uncertain"

    transition = f"{raw_gt}>{gt}"
    fields = [
        AUDIT_SCHEMA,
        key[0],
        key[1],
        key[2],
        key[3],
        raw_gt,
        gt,
        "." if gp is None else ",".join(number(value) for value in gp),
        number(called_gp),
        number(p_nonref),
        number(entropy),
        number(margin),
        number(af),
        transition,
        route,
        "1" if route == "beagle_nonref_highconf" else "0",
    ]
    return fields, route


def write_summary(path: Path, counts: Mapping[str, int], total: int) -> None:
    with path.open("x") as handle:
        handle.write(
            "schema\troute\tcount\tanchor_score\t"
            "min_anchor_probability\thard_anchor\n"
        )
        for route in ROUTES:
            handle.write(
                f"{AUDIT_SCHEMA}\t{route}\t{counts[route]}\t"
                f"p_nonref\t{THRESHOLD:g}\t"
                f"{int(route == 'beagle_nonref_highconf')}\n"
            )
        handle.write(
            f"{AUDIT_SCHEMA}\ttotal\t{total}\tp_nonref\t{THRESHOLD:g}\t.\n"
        )


def validate_ensemble_audit(
    *,
    audit: Path,
    source_audits: Mapping[int, Path],
    method_name: str,
) -> Mapping[str, object]:
    """Recompute every ensemble row from its three immutable source audits."""
    if method_name not in METHODS:
        raise ValueError(f"unknown ensemble method: {method_name}")
    if set(source_audits) != set(SEEDS):
        raise ValueError(f"source audits must contain exactly seeds {SEEDS}")
    method, included_seeds = METHODS[method_name]
    counts = {route: 0 for route in ROUTES}
    total = 0
    with ExitStack() as stack:
        readers = {}
        for seed in SEEDS:
            handle = stack.enter_context(gzip.open(source_audits[seed], "rt"))
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
                raise ValueError(f"source audit seed {seed} has a non-exact schema")
            readers[seed] = reader
        output_handle = stack.enter_context(gzip.open(audit, "rt"))
        output_reader = csv.DictReader(output_handle, delimiter="\t")
        if tuple(output_reader.fieldnames or ()) != AUDIT_COLUMNS:
            raise ValueError("ensemble audit has a non-exact schema")

        sentinel = object()
        streams: Iterable[tuple[object, ...]] = zip_longest(
            *(readers[seed] for seed in SEEDS),
            output_reader,
            fillvalue=sentinel,
        )
        for row_number, aligned in enumerate(streams, start=2):
            if any(row is sentinel for row in aligned):
                raise ValueError(
                    f"ensemble/source row counts differ near row {row_number}"
                )
            rows_by_seed = {
                seed: aligned[index]  # type: ignore[dict-item]
                for index, seed in enumerate(SEEDS)
            }
            expected, route = summarize_row(
                rows_by_seed, method, included_seeds
            )
            observed_row = aligned[-1]
            observed = [
                observed_row[column]  # type: ignore[index]
                for column in AUDIT_COLUMNS
            ]
            if observed != [str(value) for value in expected]:
                differing = [
                    column
                    for column, left, right in zip(
                        AUDIT_COLUMNS, observed, expected
                    )
                    if left != str(right)
                ]
                raise ValueError(
                    f"ensemble row {row_number} is not the prescribed "
                    f"{method_name} aggregation; fields={differing}"
                )
            counts[route] += 1
            total += 1
    if total <= 0 or sum(counts.values()) != total:
        raise ValueError("ensemble audit has no conserved records")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "method": method_name,
        "included_seeds": list(included_seeds),
        "raw_record_count": total,
        "counts": {**counts, "total": total},
    }


def validate_ensemble_audits(
    *,
    audits: Mapping[str, Path],
    source_audits: Mapping[int, Path],
) -> Mapping[str, Mapping[str, object]]:
    """Recompute all five prescribed ensemble audits in one streaming pass."""
    if set(audits) != set(METHODS):
        raise ValueError("ensemble audits must contain the exact prescribed methods")
    if set(source_audits) != set(SEEDS):
        raise ValueError(f"source audits must contain exactly seeds {SEEDS}")
    counts = {
        name: {route: 0 for route in ROUTES}
        for name in METHODS
    }
    total = 0
    with ExitStack() as stack:
        source_readers = {}
        for seed in SEEDS:
            handle = stack.enter_context(gzip.open(source_audits[seed], "rt"))
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
                raise ValueError(f"source audit seed {seed} has a non-exact schema")
            source_readers[seed] = reader
        output_readers = {}
        for name in METHODS:
            handle = stack.enter_context(gzip.open(audits[name], "rt"))
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
                raise ValueError(f"ensemble audit {name} has a non-exact schema")
            output_readers[name] = reader

        sentinel = object()
        ordered_streams = (
            *(source_readers[seed] for seed in SEEDS),
            *(output_readers[name] for name in METHODS),
        )
        for row_number, aligned in enumerate(
            zip_longest(*ordered_streams, fillvalue=sentinel), start=2
        ):
            if any(row is sentinel for row in aligned):
                raise ValueError(
                    f"ensemble/source row counts differ near row {row_number}"
                )
            rows_by_seed = {
                seed: aligned[index]  # type: ignore[dict-item]
                for index, seed in enumerate(SEEDS)
            }
            for offset, (name, (method, included_seeds)) in enumerate(
                METHODS.items(), start=len(SEEDS)
            ):
                expected, route = summarize_row(
                    rows_by_seed, method, included_seeds
                )
                observed_row = aligned[offset]
                observed = [
                    observed_row[column]  # type: ignore[index]
                    for column in AUDIT_COLUMNS
                ]
                if observed != [str(value) for value in expected]:
                    differing = [
                        column
                        for column, left, right in zip(
                            AUDIT_COLUMNS, observed, expected
                        )
                        if left != str(right)
                    ]
                    raise ValueError(
                        f"ensemble row {row_number} is not the prescribed "
                        f"{name} aggregation; fields={differing}"
                    )
                counts[name][route] += 1
            total += 1
    if total <= 0:
        raise ValueError("ensemble audits contain no records")
    return {
        name: {
            "schema": SCHEMA,
            "status": "verified",
            "method": name,
            "included_seeds": list(METHODS[name][1]),
            "raw_record_count": total,
            "counts": {**counts[name], "total": total},
        }
        for name in METHODS
    }


def build(
    *,
    audits: Mapping[int, Path],
    output_root: Path,
    sample: str,
    contracts_path: Path,
    protocol_path: Path,
) -> Mapping[str, object]:
    if sample not in {"P4", "P6"}:
        raise ValueError("sample must be P4 or P6")
    contracts = load_contracts(contracts_path)
    if not Path(protocol_path).is_file():
        raise FileNotFoundError(protocol_path)
    manifests, roots = validate_sources(audits, contracts)

    destinations = {
        name: Path(output_root).resolve() / name / f"{sample}_sec1"
        for name in METHODS
    }
    conflicts = [path for path in destinations.values() if path.exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite ensemble sample roots: "
            + ", ".join(str(path) for path in conflicts)
        )
    for sample_root in destinations.values():
        (sample_root / "beagle_evidence_v2_3_ensemble").mkdir(
            parents=True, exist_ok=False
        )

    counts = {name: {route: 0 for route in ROUTES} for name in METHODS}
    total = 0
    audit_paths = {
        name: sample_root
        / "beagle_evidence_v2_3_ensemble"
        / "beagle_evidence_audit.tsv.gz"
        for name, sample_root in destinations.items()
    }
    try:
        with ExitStack() as stack:
            readers = {}
            for seed in SEEDS:
                handle = stack.enter_context(gzip.open(audits[seed], "rt"))
                reader = csv.DictReader(handle, delimiter="\t")
                missing = set(AUDIT_COLUMNS).difference(reader.fieldnames or ())
                if missing:
                    raise ValueError(
                        f"source audit seed {seed} lacks columns: {sorted(missing)}"
                    )
                readers[seed] = reader
            writers = {
                name: stack.enter_context(gzip.open(path, "xt"))
                for name, path in audit_paths.items()
            }
            for writer in writers.values():
                writer.write("\t".join(AUDIT_COLUMNS) + "\n")

            sentinel = object()
            streams: Iterable[tuple[object, ...]] = zip_longest(
                *(readers[seed] for seed in SEEDS),
                fillvalue=sentinel,
            )
            for row_number, aligned in enumerate(streams, start=2):
                if any(row is sentinel for row in aligned):
                    raise ValueError(
                        f"source audits have different row counts near row {row_number}"
                    )
                rows_by_seed = {
                    seed: aligned[index]  # type: ignore[dict-item]
                    for index, seed in enumerate(SEEDS)
                }
                for name, (method, included_seeds) in METHODS.items():
                    fields, route = summarize_row(
                        rows_by_seed, method, included_seeds
                    )
                    writers[name].write("\t".join(fields) + "\n")
                    counts[name][route] += 1
                total += 1
    except Exception:
        # Deliberately leave incomplete roots for forensic inspection; the
        # missing completion markers prevent benchmark discovery.
        raise

    if total <= 0:
        raise ValueError("source audits contain no records")
    if any(sum(route_counts.values()) != total for route_counts in counts.values()):
        raise AssertionError("ensemble route counts do not conserve raw alleles")

    source_identities = {str(seed): identity(audits[seed]) for seed in SEEDS}
    contracts_identity = identity(contracts_path)
    self_identity = identity(Path(__file__))
    completed = {}
    for name, (method, included_seeds) in METHODS.items():
        sample_root = destinations[name]
        audit_dir = sample_root / "beagle_evidence_v2_3_ensemble"
        snapshot_dir = sample_root / "code_snapshot"
        snapshot_dir.mkdir(exist_ok=False)
        snapshot_files = []
        for source in (Path(__file__), contracts_path, protocol_path):
            destination = snapshot_dir / Path(source).name
            shutil.copy2(source, destination)
            snapshot_files.append(destination)
        summary_path = audit_dir / "beagle_evidence_summary.tsv"
        write_summary(summary_path, counts[name], total)
        manifest_path = audit_dir / "MANIFEST.json"
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "outputs_ready_for_checksum",
            "sample": sample,
            "method": name,
            "aggregation": method,
            "included_seeds": list(included_seeds),
            "frozen_source_seeds": list(SEEDS),
            "anchor_score": "p_nonref",
            "min_anchor_probability": THRESHOLD,
            "hard_anchor_route": "beagle_nonref_highconf",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "invalid_gp_policy": "beagle_uncertain",
            "panel_absent_policy": "must_agree_and_remain_panel_absent",
            "p_nonref_aggregation": "direct_across_seed_GP01_plus_GP11",
            "gp_aggregation": "componentwise_for_descriptive_genotype_fields",
            "raw_record_count": total,
            "counts": {**counts[name], "total": total},
            "source_run_roots": {
                str(seed): str(roots[seed]) for seed in SEEDS
            },
            "source_audits": source_identities,
            "source_beagle_parameters": {
                str(seed): manifests[seed]["beagle_parameters"] for seed in SEEDS
            },
            "source_contracts_module": contracts_identity,
            "source_ensemble_module": self_identity,
            "code_snapshot": [identity(path) for path in snapshot_files],
        }
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        checksum_path = contracts.write_sha256sums(
            sample_root,
            (audit_paths[name], summary_path, manifest_path, *snapshot_files),
        )
        completion_path = contracts.write_completion(
            sample_root,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
        )
        completed[name] = {
            "sample_root": str(sample_root),
            "audit": str(audit_paths[name]),
            "audit_sha256": sha256(audit_paths[name]),
            "completion": str(completion_path),
            "counts": {**counts[name], "total": total},
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "sample": sample,
        "production_inputs_include_wes": False,
        "methods": completed,
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    default_contracts = (
        project
        / "scripts/2_beagle_filtering/"
        "beagle_v2_3_candidate_2026_09_13/contracts.py"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="one v2.3 route audit; exactly seeds 20260915/16/17 are required",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample", choices=("P4", "P6"), required=True)
    parser.add_argument("--contracts", type=Path, default=default_contracts)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            project
            / "scripts/2_beagle_filtering/"
            "beagle_v2_3_candidate_2026_09_13/"
            "BENCHMARK_PROTOCOL_ENSEMBLE.md"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        audits=parse_seed_audits(args.audit),
        output_root=args.output_root,
        sample=args.sample,
        contracts_path=args.contracts.resolve(),
        protocol_path=args.protocol.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
