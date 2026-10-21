#!/usr/bin/env python3
"""Run the frozen, WES-free Beagle maxlr chromosome-22 relevance screen.

The two inputs must be complete Beagle v2.3 bundles that differ only in
MAX_LR (5000 versus 50000).  Their canonical routing audits are compared in
lockstep so allele identity and order are part of the decision contract.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARATOR = (
    PROJECT
    / "data/experiments/beagle_v2_3_smoke_20260913_v4/P4_chr22"
)
DEFAULT_CANDIDATE = (
    PROJECT
    / "data/experiments/beagle_v2_3_maxlr50000_smoke_20260913_v1/P4_chr22"
)
DEFAULT_CONTRACTS = (
    PROJECT
    / "scripts/2_beagle_filtering/"
    "beagle_v2_3_candidate_2026_09_13/contracts.py"
)
DEFAULT_PROTOCOL = (
    PROJECT
    / "scripts/2_beagle_filtering/"
    "beagle_v2_3_candidate_2026_09_13/"
    "BENCHMARK_PROTOCOL_MAXLR_PRESCREEN.md"
)

COMPARATOR_MAXLR = 5000.0
CANDIDATE_MAXLR = 50000.0
MAX_PNONREF_DELTA_STOP = 0.001
EXPECTED_SAMPLE = "P4_TUMOR"
EXPECTED_QUALITY = "baseQ0mapQ0"
EXPECTED_CONTIGS = ("chr22",)
EXPECTED_SEED = 20260915
EXPECTED_MODELSCALE = 2.0
EXPECTED_ITERATIONS = 0
ROUTES = (
    "beagle_nonref_highconf",
    "beagle_ref_highconf",
    "beagle_uncertain",
    "panel_absent",
)
AUDIT_SCHEMA = "sparcal_beagle_evidence_v2_2"
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


def require_file(path: Path, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def load_contracts(path: Path):
    path = require_file(path, "v2.3 contracts module")
    spec = importlib.util.spec_from_file_location(
        "_sparcal_beagle_maxlr_prescreen_contracts", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import v2.3 contracts module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_identity(value: object) -> object:
    """Remove only run-local identity metadata while preserving content."""
    if isinstance(value, Mapping):
        return {
            str(key): content_identity(item)
            for key, item in value.items()
            if key not in {"path", "mtime_ns"}
        }
    if isinstance(value, list):
        return [content_identity(item) for item in value]
    return value


def vcf_semantic_identity(path: Path) -> dict[str, object]:
    """Hash every VCF byte except the run-local bcftools command header.

    The chromosome smoke launcher materializes the same source slice inside
    each immutable run root.  bcftools records the destination path and wall
    clock time in ``##bcftools_viewCommand``; that one provenance header also
    changes BGZF block offsets and therefore the tabix bytes.  It is not a
    biological input difference.  All other headers and every variant record
    remain part of this exact semantic identity.
    """
    path = require_file(path, "raw VCF for semantic comparison")
    opener = gzip.open if path.name.endswith(".gz") else open
    digest = hashlib.sha256()
    line_count = 0
    ignored_run_local_headers = 0
    with opener(path, "rb") as handle:
        for line in handle:
            line_count += 1
            if line.startswith(
                b"##bcftools_viewCommand=view -r chr22 -Oz -o "
            ):
                ignored_run_local_headers += 1
                continue
            digest.update(line)
    if ignored_run_local_headers != 1:
        raise ValueError(
            "raw VCF does not have exactly one frozen chr22 materialization "
            f"header: {path}"
        )
    return {
        "sha256_excluding_bcftools_view_command": digest.hexdigest(),
        "line_count": line_count,
        "ignored_run_local_header_count": ignored_run_local_headers,
    }


def canonical_audit(root: Path, manifest: Mapping[str, object]) -> Path:
    return (
        Path(root).resolve()
        / "output_VCFs"
        / "beagle"
        / str(manifest.get("quality_filter"))
        / "beagle_evidence_audit.tsv.gz"
    )


def validate_manifest(
    manifest: Mapping[str, object], *, label: str, expected_maxlr: float
) -> None:
    if manifest.get("production_inputs_include_wes") is not False:
        raise ValueError(f"{label} unexpectedly uses WES as a production input")
    if manifest.get("evaluation_uses_wes") is not False:
        raise ValueError(f"{label} unexpectedly evaluates with WES")
    if manifest.get("impute_effective") is not False:
        raise ValueError(f"{label} unexpectedly enables reference-only imputation")
    if manifest.get("anchor_score") != "p_nonref":
        raise ValueError(f"{label} does not route on P(nonref)")
    if not math.isclose(
        float(manifest.get("min_anchor_probability", -1)),
        0.90,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} does not use the frozen 0.90 anchor threshold")
    if manifest.get("dataset") != EXPECTED_SAMPLE:
        raise ValueError(f"{label} is not the frozen P4 smoke sample")
    if manifest.get("quality_filter") != EXPECTED_QUALITY:
        raise ValueError(f"{label} has the wrong quality-filter input")
    if tuple(manifest.get("completed_beagle_contigs", ())) != EXPECTED_CONTIGS:
        raise ValueError(f"{label} is not an exact chromosome-22 smoke run")

    parameters = manifest.get("beagle_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{label} lacks Beagle parameters")
    expected = {
        "IMPUTE": False,
        "GPROBS": True,
        "ITERATIONS": EXPECTED_ITERATIONS,
        "SEED": EXPECTED_SEED,
        "MODEL_SCALE": EXPECTED_MODELSCALE,
        "MAX_LR": expected_maxlr,
        "GENETIC_MAP": None,
    }
    for name, value in expected.items():
        observed = parameters.get(name)
        if isinstance(value, float):
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise ValueError(f"{label} parameter {name} is not numeric")
            matches = math.isclose(
                float(observed), value, rel_tol=0, abs_tol=1e-12
            )
        else:
            matches = observed == value
        if not matches:
            raise ValueError(
                f"{label} parameter {name} differs from the frozen value: "
                f"{observed!r} != {value!r}"
            )


def validate_manifest_pair(
    comparator: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    raw_vcf_semantically_equal: bool = False,
) -> None:
    """Require one input/configuration identity with only MAX_LR changed."""
    validate_manifest(
        comparator, label="comparator", expected_maxlr=COMPARATOR_MAXLR
    )
    validate_manifest(
        candidate, label="candidate", expected_maxlr=CANDIDATE_MAXLR
    )

    comparator_parameters = dict(comparator["beagle_parameters"])
    expected_candidate_parameters = dict(comparator_parameters)
    expected_candidate_parameters["MAX_LR"] = CANDIDATE_MAXLR
    if stable_json(candidate["beagle_parameters"]) != stable_json(
        expected_candidate_parameters
    ):
        raise ValueError(
            "Beagle parameter tables differ by more than MAX_LR=5000 vs 50000"
        )

    comparator_raw = content_identity(comparator.get("raw_vcf"))
    candidate_raw = content_identity(candidate.get("raw_vcf"))
    raw_bytes_equal = stable_json(comparator_raw) == stable_json(candidate_raw)
    comparator_index = content_identity(comparator.get("raw_vcf_index"))
    candidate_index = content_identity(candidate.get("raw_vcf_index"))
    index_bytes_equal = stable_json(comparator_index) == stable_json(
        candidate_index
    )
    if not raw_bytes_equal and not raw_vcf_semantically_equal:
        raise ValueError("input/configuration identity differs at raw_vcf")
    if raw_bytes_equal and not index_bytes_equal:
        raise ValueError("input/configuration identity differs at raw_vcf_index")

    invariant_fields = (
        "schema",
        "production_inputs_include_wes",
        "evaluation_uses_wes",
        "impute_effective",
        "anchor_score",
        "min_anchor_probability",
        "dataset",
        "quality_filter",
        "completed_beagle_contigs",
        "code",
        "executables",
        "reference_ledger",
        "reference_panels",
    )
    for field in invariant_fields:
        left = content_identity(comparator.get(field))
        right = content_identity(candidate.get(field))
        if stable_json(left) != stable_json(right):
            raise ValueError(f"input/configuration identity differs at {field}")


def validate_run_pair(
    comparator_root: Path,
    candidate_root: Path,
    contracts,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Path,
    Path,
    Mapping[str, object],
    Mapping[str, object],
]:
    comparator_root = Path(comparator_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    if comparator_root == candidate_root:
        raise ValueError("comparator and candidate must be distinct bundles")

    comparator_validation = contracts.validate_v2_3_run(comparator_root)
    candidate_validation = contracts.validate_v2_3_run(candidate_root)
    for label, validation in (
        ("comparator", comparator_validation),
        ("candidate", candidate_validation),
    ):
        if validation.get("status") != "verified":
            raise ValueError(f"{label} v2.3 bundle was not verified")

    comparator_manifest_path = require_file(
        comparator_root / "V2_3_MANIFEST.json", "comparator manifest"
    )
    candidate_manifest_path = require_file(
        candidate_root / "V2_3_MANIFEST.json", "candidate manifest"
    )
    comparator_manifest = json.loads(comparator_manifest_path.read_text())
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    comparator_raw_identity = content_identity(comparator_manifest.get("raw_vcf"))
    candidate_raw_identity = content_identity(candidate_manifest.get("raw_vcf"))
    raw_vcf_semantically_equal = False
    if stable_json(comparator_raw_identity) != stable_json(
        candidate_raw_identity
    ):
        comparator_semantic = vcf_semantic_identity(
            Path(str(comparator_manifest["raw_vcf"]["path"]))
        )
        candidate_semantic = vcf_semantic_identity(
            Path(str(candidate_manifest["raw_vcf"]["path"]))
        )
        raw_vcf_semantically_equal = (
            stable_json(comparator_semantic) == stable_json(candidate_semantic)
        )
        if not raw_vcf_semantically_equal:
            raise ValueError(
                "input/configuration identity differs at raw_vcf semantic content"
            )
    validate_manifest_pair(
        comparator_manifest,
        candidate_manifest,
        raw_vcf_semantically_equal=raw_vcf_semantically_equal,
    )

    comparator_audit = require_file(
        canonical_audit(comparator_root, comparator_manifest),
        "canonical comparator audit",
    )
    candidate_audit = require_file(
        canonical_audit(candidate_root, candidate_manifest),
        "canonical candidate audit",
    )
    return (
        comparator_manifest,
        candidate_manifest,
        comparator_audit,
        candidate_audit,
        comparator_validation,
        candidate_validation,
    )


def parse_probability(value: object, *, label: str) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if text in {"", ".", "nan", "NaN"}:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} is outside [0,1]: {value!r}")
    return number


def audit_rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != AUDIT_COLUMNS:
            raise ValueError(f"{path} has a non-exact audit header")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number} has extra columns")
            yield row


def allele_key(row: Mapping[str, str], *, label: str) -> tuple[str, int, str, str]:
    try:
        position = int(row["pos"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid allele position") from error
    if position <= 0:
        raise ValueError(f"{label} has a non-positive allele position")
    key = (row["chrom"], position, row["ref"], row["alt"])
    if any(not str(value) for value in key):
        raise ValueError(f"{label} has an empty exact-allele field")
    return key


def validate_audit_row(
    row: Mapping[str, str], *, label: str
) -> tuple[tuple[str, int, str, str], Optional[float]]:
    key = allele_key(row, label=label)
    if row["schema"] != AUDIT_SCHEMA:
        raise ValueError(f"{label} has an unexpected audit schema")
    route = row["route"]
    if route not in ROUTES:
        raise ValueError(f"{label} has an unknown route: {route!r}")
    expected_anchor = "1" if route == "beagle_nonref_highconf" else "0"
    if row["hard_anchor"] != expected_anchor:
        raise ValueError(f"{label} route and hard_anchor disagree at {key}")
    p_nonref = parse_probability(
        row["beagle_p_nonref"], label=f"{label} P(nonref) at {key}"
    )
    if (route == "panel_absent") != (p_nonref is None):
        raise ValueError(
            f"{label} panel-presence route and P(nonref) disagree at {key}"
        )
    return key, p_nonref


def compare_audits(
    comparator_audit: Path,
    candidate_audit: Path,
    *,
    expected_rows: Optional[int] = None,
) -> dict[str, object]:
    """Compare two exact audits in order and return the frozen gate summary."""
    seen: set[tuple[str, int, str, str]] = set()
    route_counts = {
        "comparator": Counter(),
        "candidate": Counter(),
    }
    transitions: Counter[tuple[str, str]] = Counter()
    row_count = 0
    anchor_disagreements = 0
    route_disagreements = 0
    p_nonref_compared = 0
    p_nonref_delta_sum = 0.0
    p_nonref_delta_max = 0.0

    for row_count, pair in enumerate(
        zip_longest(
            audit_rows(comparator_audit),
            audit_rows(candidate_audit),
        ),
        start=1,
    ):
        comparator_row, candidate_row = pair
        if comparator_row is None or candidate_row is None:
            raise ValueError("audit row counts differ")
        comparator_key, comparator_probability = validate_audit_row(
            comparator_row, label=f"comparator row {row_count + 1}"
        )
        candidate_key, candidate_probability = validate_audit_row(
            candidate_row, label=f"candidate row {row_count + 1}"
        )
        if comparator_key != candidate_key:
            raise ValueError(
                "audit allele universe/order differs at data row "
                f"{row_count}: {comparator_key} != {candidate_key}"
            )
        if comparator_key in seen:
            raise ValueError(f"duplicate exact allele in audits: {comparator_key}")
        seen.add(comparator_key)
        if comparator_row["raw_gt"] != candidate_row["raw_gt"]:
            raise ValueError(f"raw genotype differs at {comparator_key}")

        comparator_route = comparator_row["route"]
        candidate_route = candidate_row["route"]
        if (comparator_route == "panel_absent") != (
            candidate_route == "panel_absent"
        ):
            raise ValueError(f"panel presence differs at {comparator_key}")
        route_counts["comparator"][comparator_route] += 1
        route_counts["candidate"][candidate_route] += 1
        transitions[(comparator_route, candidate_route)] += 1
        if comparator_route != candidate_route:
            route_disagreements += 1
        if (
            comparator_route == "beagle_nonref_highconf"
        ) != (candidate_route == "beagle_nonref_highconf"):
            anchor_disagreements += 1

        if (comparator_probability is None) != (candidate_probability is None):
            raise ValueError(f"P(nonref) availability differs at {comparator_key}")
        if comparator_probability is not None:
            assert candidate_probability is not None
            delta = abs(comparator_probability - candidate_probability)
            p_nonref_compared += 1
            p_nonref_delta_sum += delta
            p_nonref_delta_max = max(p_nonref_delta_max, delta)

    if row_count == 0:
        raise ValueError("audits contain no alleles")
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"audit row count differs from v2.3 contract: "
            f"{row_count} != {expected_rows}"
        )
    if p_nonref_compared == 0:
        raise ValueError("audits contain no panel-present P(nonref) values")

    routes_identical = route_disagreements == 0
    anchors_identical = anchor_disagreements == 0
    delta_below_stop_threshold = (
        p_nonref_delta_max < MAX_PNONREF_DELTA_STOP
    )
    advance = not (routes_identical and delta_below_stop_threshold)
    decision = (
        "advance_to_frozen_full_genome_comparison"
        if advance
        else "stop_after_chr22_no_full_genome"
    )
    result: dict[str, object] = {
        "allele_count": row_count,
        "p_nonref_compared_n": p_nonref_compared,
        "p_nonref_mean_absolute_delta": (
            p_nonref_delta_sum / p_nonref_compared
        ),
        "p_nonref_max_absolute_delta": p_nonref_delta_max,
        "route_disagreement_count": route_disagreements,
        "anchor_disagreement_count": anchor_disagreements,
        "route_assignments_identical": routes_identical,
        "anchor_assignments_identical": anchors_identical,
        "max_p_nonref_delta_below_0p001": delta_below_stop_threshold,
        "advance_to_full_genome": advance,
        "decision": decision,
        "route_transition_counts": {
            f"{left}->{right}": count
            for (left, right), count in sorted(transitions.items())
        },
    }
    for side in ("comparator", "candidate"):
        for route in ROUTES:
            result[f"{side}_{route}_count"] = route_counts[side][route]
    return result


def write_checksum_ledger(
    paths: Sequence[Path],
    *,
    base: Path,
    destination: Path,
) -> None:
    base = Path(base).resolve()

    def label(path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            return str(resolved)

    rows = sorted(
        ((label(path), sha256(path)) for path in paths),
        key=lambda item: item[0],
    )
    with Path(destination).open("x") as handle:
        for name, digest in rows:
            handle.write(f"{digest}  {name}\n")


def summary_columns() -> tuple[str, ...]:
    return (
        "comparator_maxlr",
        "candidate_maxlr",
        "allele_count",
        "p_nonref_compared_n",
        "p_nonref_mean_absolute_delta",
        "p_nonref_max_absolute_delta",
        "route_disagreement_count",
        "anchor_disagreement_count",
        "route_assignments_identical",
        "anchor_assignments_identical",
        "max_p_nonref_delta_below_0p001",
        *(
            f"{side}_{route}_count"
            for side in ("comparator", "candidate")
            for route in ROUTES
        ),
        "advance_to_full_genome",
        "decision",
    )


def write_summary(path: Path, result: Mapping[str, object]) -> None:
    columns = summary_columns()
    row = {
        **result,
        "comparator_maxlr": COMPARATOR_MAXLR,
        "candidate_maxlr": CANDIDATE_MAXLR,
    }
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerow(row)


def write_bundle(
    output_dir: Path,
    *,
    result: Mapping[str, object],
    comparator_root: Path,
    candidate_root: Path,
    comparator_manifest: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    comparator_audit: Path,
    candidate_audit: Path,
    comparator_validation: Mapping[str, object],
    candidate_validation: Mapping[str, object],
    contracts_path: Path,
    protocol_path: Path,
) -> None:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    summary_path = output_dir / "maxlr_prescreen_summary.tsv"
    write_summary(summary_path, result)
    provenance_path = output_dir / "PROVENANCE.json"
    provenance = {
        "schema": "sparcal_beagle_maxlr_prescreen_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "outputs_ready_for_checksum",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "wes_free_decision": True,
        "scope": "P4 chromosome 22 computational relevance prescreen",
        "comparator_maxlr": COMPARATOR_MAXLR,
        "candidate_maxlr": CANDIDATE_MAXLR,
        "decision_threshold": {
            "stop_only_if_route_assignments_identical": True,
            "stop_only_if_max_absolute_p_nonref_delta_strictly_below": (
                MAX_PNONREF_DELTA_STOP
            ),
        },
        "decision": result["decision"],
        "advance_to_full_genome": result["advance_to_full_genome"],
        "metrics": dict(result),
        "raw_vcf_semantic_identity": {
            "comparator": vcf_semantic_identity(
                Path(str(comparator_manifest["raw_vcf"]["path"]))
            ),
            "candidate": vcf_semantic_identity(
                Path(str(candidate_manifest["raw_vcf"]["path"]))
            ),
            "definition": (
                "exact decompressed VCF bytes excluding only the single "
                "frozen chr22 materialization ##bcftools_viewCommand header"
            ),
        },
        "assessor": identity(Path(__file__)),
        "contracts": identity(contracts_path),
        "protocol": identity(protocol_path),
        "inputs": {
            "comparator": {
                "root": str(Path(comparator_root).resolve()),
                "manifest": identity(
                    Path(comparator_root).resolve() / "V2_3_MANIFEST.json"
                ),
                "completion": identity(
                    Path(comparator_root).resolve() / "RUN_COMPLETE.json"
                ),
                "audit": identity(comparator_audit),
                "manifest_beagle_parameters": comparator_manifest[
                    "beagle_parameters"
                ],
                "contract_validation": comparator_validation,
            },
            "candidate": {
                "root": str(Path(candidate_root).resolve()),
                "manifest": identity(
                    Path(candidate_root).resolve() / "V2_3_MANIFEST.json"
                ),
                "completion": identity(
                    Path(candidate_root).resolve() / "RUN_COMPLETE.json"
                ),
                "audit": identity(candidate_audit),
                "manifest_beagle_parameters": candidate_manifest[
                    "beagle_parameters"
                ],
                "contract_validation": candidate_validation,
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    code_and_input_paths = (
        Path(__file__).resolve(),
        Path(contracts_path).resolve(),
        Path(protocol_path).resolve(),
        Path(comparator_root).resolve() / "V2_3_MANIFEST.json",
        Path(comparator_root).resolve() / "RUN_COMPLETE.json",
        Path(comparator_audit).resolve(),
        Path(candidate_root).resolve() / "V2_3_MANIFEST.json",
        Path(candidate_root).resolve() / "RUN_COMPLETE.json",
        Path(candidate_audit).resolve(),
        Path(str(comparator_manifest["raw_vcf"]["path"])).resolve(),
        Path(str(comparator_manifest["raw_vcf_index"]["path"])).resolve(),
        Path(str(candidate_manifest["raw_vcf"]["path"])).resolve(),
        Path(str(candidate_manifest["raw_vcf_index"]["path"])).resolve(),
    )
    input_sums = output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_checksum_ledger(
        code_and_input_paths,
        base=PROJECT,
        destination=input_sums,
    )
    output_sums = output_dir / "OUTPUT_SHA256SUMS.txt"
    write_checksum_ledger(
        (summary_path, provenance_path, input_sums),
        base=output_dir,
        destination=output_sums,
    )
    (output_dir / "RUN_COMPLETE").write_text("complete\n")
    completion = {
        "schema": "sparcal_beagle_maxlr_prescreen_completion_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "decision": result["decision"],
        "advance_to_full_genome": result["advance_to_full_genome"],
        "output_sha256sums": identity(output_sums),
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparator-root", type=Path, default=DEFAULT_COMPARATOR
    )
    parser.add_argument(
        "--candidate-root", type=Path, default=DEFAULT_CANDIDATE
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    require_file(args.contracts, "v2.3 contracts module")
    require_file(args.protocol, "frozen maxlr prescreen protocol")
    contracts = load_contracts(args.contracts)
    (
        comparator_manifest,
        candidate_manifest,
        comparator_audit,
        candidate_audit,
        comparator_validation,
        candidate_validation,
    ) = validate_run_pair(
        args.comparator_root,
        args.candidate_root,
        contracts,
    )
    comparator_count = int(comparator_validation["raw_record_count"])
    candidate_count = int(candidate_validation["raw_record_count"])
    if comparator_count != candidate_count:
        raise ValueError(
            "raw record counts differ across validated v2.3 bundles: "
            f"{comparator_count} != {candidate_count}"
        )
    result = compare_audits(
        comparator_audit,
        candidate_audit,
        expected_rows=comparator_count,
    )
    write_bundle(
        args.output_dir,
        result=result,
        comparator_root=args.comparator_root,
        candidate_root=args.candidate_root,
        comparator_manifest=comparator_manifest,
        candidate_manifest=candidate_manifest,
        comparator_audit=comparator_audit,
        candidate_audit=candidate_audit,
        comparator_validation=comparator_validation,
        candidate_validation=candidate_validation,
        contracts_path=args.contracts,
        protocol_path=args.protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
