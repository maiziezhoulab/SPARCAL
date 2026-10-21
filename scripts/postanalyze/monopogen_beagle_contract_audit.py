#!/usr/bin/env python3
"""Audit the WES-independent Beagle contract in released Monopogen outputs.

The audit compares pooled raw genotype likelihoods (``*.gl.vcf.gz``), the
first Beagle pass (``*.gp.vcf.gz``), and the final phasing pass
(``*.phased.vcf.gz``).  VCFs are joined as position-sized streaming groups, so
whole callsets are never loaded into memory.  Output directories are immutable:
an existing path is always an error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT = Path(__file__).resolve().parents[2]
PINNED_MONOPOGEN_COMMIT = "2c0a4b0868cb86a77f196cba29b4d4dd3ce482f6"
PINNED_MONOPOGEN_REPOSITORY = "https://github.com/KChen-lab/Monopogen"
MONOPOGEN_PAPER = "https://doi.org/10.1038/s41587-023-01873-x"
MONOPOGEN_BEAGLE_SOURCE = (
    f"{PINNED_MONOPOGEN_REPOSITORY}/blob/{PINNED_MONOPOGEN_COMMIT}"
    "/src/Monopogen.py#L96-L120"
)
MONOPOGEN_DOWNSTREAM_SOURCE = (
    f"{PINNED_MONOPOGEN_REPOSITORY}/blob/{PINNED_MONOPOGEN_COMMIT}"
    "/src/somatic.py#L108-L156"
)
AUDIT_SCHEMA_VERSION = "3.0"
DEFAULT_RUNS = (
    ("P4_rep1", PROJECT / "Monopogen_P4_rep1"),
    ("P4_rep2", PROJECT / "Monopogen_P4_rep2"),
    ("P6_rep1", PROJECT / "Monopogen_P6_rep1"),
    ("P6_rep2", PROJECT / "Monopogen_P6_rep2"),
)
GT_BY_INDEX = ("0/0", "0/1", "1/1")
NONREF_GT = {"0/1", "1/1"}
Key = Tuple[str, int, str, str]

DOWNSTREAM_RECORD_METHOD = "monopogen_downstream_beagle_record_membership"
PAPER_CONCORDANCE_METHOD = "monopogen_paper_raw_beagle_gt_concordant_nonref"
PNONREF_METHOD = "sparcal_derived_pnonref_threshold"
PHASED_NONREF_DIAGNOSTIC = "diagnostic_final_phased_nonref_gt"
METHOD_ORDER = (
    DOWNSTREAM_RECORD_METHOD,
    PAPER_CONCORDANCE_METHOD,
    PNONREF_METHOD,
    PHASED_NONREF_DIAGNOSTIC,
)


@dataclass(frozen=True)
class VCFRecord:
    key: Key
    sample: Mapping[str, str]


@dataclass(frozen=True)
class InputFile:
    run: str
    chromosome: str
    kind: str
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


def normalize_gt(value: Optional[str]) -> Optional[str]:
    if value in {None, "", ".", "./.", ".|."}:
        return None
    alleles = str(value).replace("|", "/").split("/")
    if len(alleles) != 2 or any(allele not in {"0", "1"} for allele in alleles):
        return None
    return "/".join(sorted(alleles))


def parse_triplet(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if value in {None, "", "."}:
        return None
    try:
        parsed = tuple(float(item) for item in str(value).split(","))
    except ValueError:
        return None
    if len(parsed) != 3 or not all(math.isfinite(item) for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def raw_gt_from_pl(value: Optional[str]) -> Optional[str]:
    values = parse_triplet(value)
    if values is None:
        return None
    return GT_BY_INDEX[min(range(3), key=values.__getitem__)]


def unique_raw_gt_from_pl(value: Optional[str]) -> Optional[str]:
    values = parse_triplet(value)
    if values is None:
        return None
    best = min(values)
    winners = [index for index, item in enumerate(values) if item == best]
    if len(winners) != 1:
        return None
    return GT_BY_INDEX[winners[0]]


def raw_pl_is_tied(value: Optional[str]) -> bool:
    values = parse_triplet(value)
    return values is not None and values.count(min(values)) > 1


def read_vcf(path: Path) -> Iterator[VCFRecord]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        previous: Optional[Tuple[str, int]] = None
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"{path}:{line_number}: expected a single-sample VCF")
            chrom, pos_text, ref, alt = fields[0], fields[1], fields[3], fields[4]
            if "," in alt:
                raise ValueError(f"{path}:{line_number}: multiallelic record is not normalized")
            try:
                pos = int(pos_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid POS {pos_text!r}") from exc
            coordinate = (chrom, pos)
            if previous is not None and coordinate < previous:
                raise ValueError(f"{path}:{line_number}: records are not coordinate sorted")
            previous = coordinate
            names = fields[8].split(":")
            values = fields[9].split(":")
            yield VCFRecord((chrom, pos, ref, alt), dict(zip(names, values)))


def position_groups(path: Path) -> Iterator[Tuple[Tuple[str, int], List[VCFRecord]]]:
    coordinate: Optional[Tuple[str, int]] = None
    records: List[VCFRecord] = []
    for record in read_vcf(path):
        current = record.key[:2]
        if coordinate is not None and current != coordinate:
            yield coordinate, records
            records = []
        coordinate = current
        records.append(record)
    if coordinate is not None:
        yield coordinate, records


def record_map(records: Iterable[VCFRecord], path: Path) -> Dict[Key, VCFRecord]:
    out: Dict[Key, VCFRecord] = {}
    for record in records:
        if record.key in out:
            raise ValueError(f"{path}: duplicate normalized key {record.key}")
        out[record.key] = record
    return out


def next_group(
    iterator: Iterator[Tuple[Tuple[str, int], List[VCFRecord]]]
) -> Optional[Tuple[Tuple[str, int], List[VCFRecord]]]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def safe_fraction(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def jaccard(left: int, right: int, intersection: int) -> Optional[float]:
    return safe_fraction(intersection, left + right - intersection)


def parse_chromosomes(specification: str) -> Tuple[int, ...]:
    values: List[int] = []
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid chromosome interval {token!r}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    result = tuple(dict.fromkeys(values))
    if not result or any(value < 1 or value > 22 for value in result):
        raise ValueError("chromosomes must be autosomes 1-22")
    return result


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must have the form LABEL=PATH")
    label, path_text = value.split("=", 1)
    if not label or not path_text:
        raise argparse.ArgumentTypeError("--run must have the form LABEL=PATH")
    return label, Path(path_text)


def validate_inputs(
    runs: Sequence[Tuple[str, Path]], chromosomes: Sequence[int]
) -> List[InputFile]:
    labels = [label for label, _ in runs]
    if len(labels) != len(set(labels)):
        raise ValueError("run labels must be unique")
    manifest: List[InputFile] = []
    for label, run_dir in runs:
        germline = run_dir.resolve() / "out" / "germline"
        for chromosome in chromosomes:
            chrom = f"chr{chromosome}"
            for kind in ("gl", "gp", "phased"):
                path = germline / f"{chrom}.{kind}.vcf.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                stat = path.stat()
                manifest.append(
                    InputFile(
                        run=label,
                        chromosome=chrom,
                        kind=kind,
                        path=str(path),
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        sha256=sha256(path),
                    )
                )
    return manifest


def audit_run(
    label: str,
    run_dir: Path,
    chromosomes: Sequence[int],
    threshold: float,
) -> Tuple[Dict[str, object], Counter, Counter, Counter]:
    counts: Counter = Counter()
    transitions: Counter = Counter()
    method_counts: Counter = Counter()
    method_intersections: Counter = Counter()
    germline = run_dir.resolve() / "out" / "germline"

    for chromosome in chromosomes:
        chrom = f"chr{chromosome}"
        paths = {
            "raw": germline / f"{chrom}.gl.vcf.gz",
            "gp": germline / f"{chrom}.gp.vcf.gz",
            "phased": germline / f"{chrom}.phased.vcf.gz",
        }
        iterators = {kind: position_groups(path) for kind, path in paths.items()}
        groups = {kind: next_group(iterator) for kind, iterator in iterators.items()}

        while any(group is not None for group in groups.values()):
            coordinate = min(
                group[0] for group in groups.values() if group is not None
            )
            maps: Dict[str, Dict[Key, VCFRecord]] = {}
            for kind in ("raw", "gp", "phased"):
                group = groups[kind]
                if group is not None and group[0] == coordinate:
                    maps[kind] = record_map(group[1], paths[kind])
                    groups[kind] = next_group(iterators[kind])
                else:
                    maps[kind] = {}

            raw_keys, gp_keys, phased_keys = (
                set(maps["raw"]),
                set(maps["gp"]),
                set(maps["phased"]),
            )
            counts["raw_records"] += len(raw_keys)
            counts["gp_records"] += len(gp_keys)
            counts["phased_records"] += len(phased_keys)
            counts["gp_keys_absent_from_phased"] += len(gp_keys - phased_keys)
            counts["phased_keys_absent_from_gp"] += len(phased_keys - gp_keys)
            counts["raw_keys_absent_from_gp"] += len(raw_keys - gp_keys)
            counts["gp_keys_absent_from_raw"] += len(gp_keys - raw_keys)

            for key in raw_keys | gp_keys | phased_keys:
                raw = maps["raw"].get(key)
                gp = maps["gp"].get(key)
                phased = maps["phased"].get(key)
                raw_gt = raw_gt_from_pl(raw.sample.get("PL")) if raw else None
                raw_unique_gt = (
                    unique_raw_gt_from_pl(raw.sample.get("PL")) if raw else None
                )
                gp_gt = normalize_gt(gp.sample.get("GT")) if gp else None
                phased_gt = normalize_gt(phased.sample.get("GT")) if phased else None

                if raw is not None and raw_gt is None:
                    counts["raw_pl_unresolved"] += 1
                raw_tied = raw is not None and raw_pl_is_tied(raw.sample.get("PL"))
                if raw_tied:
                    counts["raw_pl_ties"] += 1

                probabilities = parse_triplet(gp.sample.get("GP")) if gp else None
                if gp is not None and probabilities is None:
                    counts["gp_probability_unresolved"] += 1
                if probabilities is not None and gp_gt is not None:
                    gt_index = GT_BY_INDEX.index(gp_gt)
                    maximum = max(probabilities)
                    winners = [
                        index
                        for index, value in enumerate(probabilities)
                        if value == maximum
                    ]
                    if len(winners) > 1:
                        counts["gp_printed_argmax_ties"] += 1
                    if probabilities[gt_index] < maximum:
                        counts["gp_gt_strictly_below_printed_max"] += 1

                if raw_gt is not None and gp_gt is not None:
                    counts["raw_gp_comparable"] += 1
                    counts["raw_gp_pl_ties"] += int(raw_tied)
                    transitions[(raw_gt, gp_gt)] += 1
                    if raw_gt == gp_gt:
                        counts["raw_gp_concordant"] += 1

                if gp_gt is not None and phased_gt is not None:
                    counts["gp_phased_comparable"] += 1
                    if gp_gt != phased_gt:
                        counts["gp_phased_unordered_gt_changes"] += 1
                if phased_gt == "0/1":
                    counts["phased_heterozygous"] += 1
                    literal_gt = phased.sample.get("GT") if phased else None
                    if literal_gt == "0|1":
                        counts["phased_0_pipe_1"] += 1
                    elif literal_gt == "1|0":
                        counts["phased_1_pipe_0"] += 1

                downstream_record = phased is not None
                paper_concordance = raw_gt in NONREF_GT and raw_gt == gp_gt
                paper_concordance_unique = (
                    raw_unique_gt in NONREF_GT and raw_unique_gt == gp_gt
                )
                p_nonref = (
                    probabilities[1] + probabilities[2]
                    if probabilities is not None
                    else None
                )
                derived_pnonref = (
                    p_nonref is not None and p_nonref >= threshold - 1e-12
                )
                memberships = {
                    DOWNSTREAM_RECORD_METHOD: downstream_record,
                    PAPER_CONCORDANCE_METHOD: paper_concordance,
                    PNONREF_METHOD: derived_pnonref,
                    PHASED_NONREF_DIAGNOSTIC: phased_gt in NONREF_GT,
                }
                for method, member in memberships.items():
                    method_counts[method] += int(member)
                for left, right in combinations(METHOD_ORDER, 2):
                    method_intersections[(left, right)] += int(
                        memberships[left] and memberships[right]
                    )
                counts[
                    "diagnostic_paper_unique_raw_pl_concordant_nonref"
                ] += int(paper_concordance_unique)

    summary: Dict[str, object] = {
        "run": label,
        "run_dir": str(run_dir.resolve()),
        "chromosome_count": len(chromosomes),
        "raw_records": counts["raw_records"],
        "gp_records": counts["gp_records"],
        "phased_records": counts["phased_records"],
        "raw_gp_record_sets_equal": (
            counts["raw_keys_absent_from_gp"] == 0
            and counts["gp_keys_absent_from_raw"] == 0
        ),
        "gp_phased_record_sets_equal": (
            counts["gp_keys_absent_from_phased"] == 0
            and counts["phased_keys_absent_from_gp"] == 0
        ),
        "all_three_record_sets_equal": (
            counts["raw_keys_absent_from_gp"] == 0
            and counts["gp_keys_absent_from_raw"] == 0
            and counts["gp_keys_absent_from_phased"] == 0
            and counts["phased_keys_absent_from_gp"] == 0
        ),
        "raw_keys_absent_from_gp": counts["raw_keys_absent_from_gp"],
        "gp_keys_absent_from_phased": counts["gp_keys_absent_from_phased"],
        "phased_keys_absent_from_gp": counts["phased_keys_absent_from_gp"],
        "gp_keys_absent_from_raw": counts["gp_keys_absent_from_raw"],
        "raw_pl_unresolved": counts["raw_pl_unresolved"],
        "raw_pl_ties": counts["raw_pl_ties"],
        "raw_gp_comparable": counts["raw_gp_comparable"],
        "raw_gp_pl_ties": counts["raw_gp_pl_ties"],
        "raw_gp_concordant": counts["raw_gp_concordant"],
        "raw_gp_class_concordance": safe_fraction(
            counts["raw_gp_concordant"], counts["raw_gp_comparable"]
        ),
        "monopogen_downstream_beagle_records": method_counts[
            DOWNSTREAM_RECORD_METHOD
        ],
        "monopogen_paper_raw_beagle_gt_concordant_nonref": method_counts[
            PAPER_CONCORDANCE_METHOD
        ],
        "diagnostic_paper_unique_raw_pl_concordant_nonref": counts[
            "diagnostic_paper_unique_raw_pl_concordant_nonref"
        ],
        "pnonref_threshold": threshold,
        "sparcal_derived_pnonref_threshold_count": method_counts[PNONREF_METHOD],
        "diagnostic_final_phased_nonref_gt": method_counts[
            PHASED_NONREF_DIAGNOSTIC
        ],
        "gp_probability_unresolved": counts["gp_probability_unresolved"],
        "gp_printed_argmax_ties": counts["gp_printed_argmax_ties"],
        "gp_gt_strictly_below_printed_max": counts[
            "gp_gt_strictly_below_printed_max"
        ],
        "gp_phased_comparable": counts["gp_phased_comparable"],
        "gp_phased_unordered_gt_changes": counts[
            "gp_phased_unordered_gt_changes"
        ],
        "phased_heterozygous": counts["phased_heterozygous"],
        "phased_0_pipe_1": counts["phased_0_pipe_1"],
        "phased_1_pipe_0": counts["phased_1_pipe_0"],
        "raw_1_1_to_gp_0_1": transitions[("1/1", "0/1")],
    }
    return summary, transitions, method_counts, method_intersections


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_definitions(threshold: float) -> Dict[str, Dict[str, str]]:
    return {
        DOWNSTREAM_RECORD_METHOD: {
            "role": "released_downstream_beagle_component",
            "definition": (
                "Exact CHROM,POS,REF,ALT membership in the final phased VCF, "
                "regardless of final phased GT. The full Monopogen somatic code "
                "also applies raw read-support conditions downstream; those are "
                "outside this Beagle-only audit."
            ),
        },
        PAPER_CONCORDANCE_METHOD: {
            "role": "paper_described_concordance_rule",
            "definition": (
                "Raw PL argmin (lowest standard VCF genotype index on ties) "
                "equals the first-pass Beagle GT and is non-reference."
            ),
        },
        PNONREF_METHOD: {
            "role": "sparcal_derived_rule_not_native_monopogen",
            "definition": (
                f"First-pass Beagle GP(0/1)+GP(1/1) >= {threshold:g}."
            ),
        },
        PHASED_NONREF_DIAGNOSTIC: {
            "role": "diagnostic_only",
            "definition": (
                "Unordered final phased GT is 0/1 or 1/1. This is not the "
                "released downstream Beagle record-membership contract."
            ),
        },
    }


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    shutil.copystat(source, destination)


def write_output_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "OUTPUT_SHA256SUMS.txt"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("x") as handle:
        for path in files:
            relative = path.relative_to(output_dir).as_posix()
            handle.write(f"{sha256(path)}  {relative}\n")
    return checksum_path


def run_audit(
    runs: Sequence[Tuple[str, Path]],
    chromosomes: Sequence[int],
    threshold: float,
    output_dir: Path,
) -> List[Dict[str, object]]:
    started_at = datetime.now(timezone.utc)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    script_path = Path(__file__).resolve()
    test_path = PROJECT / "tests" / "test_monopogen_beagle_contract_audit.py"
    snapshot_sources = (script_path, test_path)
    for source in snapshot_sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    manifest = validate_inputs(runs, chromosomes)
    output_dir.mkdir(parents=True, exist_ok=False)

    definitions = method_definitions(threshold)
    summaries: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []
    method_set_rows: List[Dict[str, object]] = []
    intersection_rows: List[Dict[str, object]] = []
    for label, run_dir in runs:
        summary, transitions, method_counts, intersections = audit_run(
            label, run_dir, chromosomes, threshold
        )
        summaries.append(summary)
        comparable = int(summary["raw_gp_comparable"])
        discordant_beagle_nonref = sum(
            count
            for (raw_gt, gp_gt), count in transitions.items()
            if gp_gt in NONREF_GT and raw_gt != gp_gt
        )
        for raw_gt, gp_gt in product(GT_BY_INDEX, repeat=2):
            count = transitions[(raw_gt, gp_gt)]
            transition_rows.append(
                {
                    "run": label,
                    "raw_pl_gt": raw_gt,
                    "beagle_gp_gt": gp_gt,
                    "count": count,
                    "fraction_of_comparable": safe_fraction(count, comparable),
                    "fraction_of_discordant_beagle_gp_nonref": (
                        safe_fraction(count, discordant_beagle_nonref)
                        if gp_gt in NONREF_GT and raw_gt != gp_gt
                        else None
                    ),
                }
            )

        for method in METHOD_ORDER:
            method_set_rows.append(
                {
                    "run": label,
                    "method": method,
                    "method_role": definitions[method]["role"],
                    "count": method_counts[method],
                    "pnonref_threshold": threshold
                    if method == PNONREF_METHOD
                    else None,
                    "definition": definitions[method]["definition"],
                }
            )
        for left, right in combinations(METHOD_ORDER, 2):
            left_count = method_counts[left]
            right_count = method_counts[right]
            intersection = intersections[(left, right)]
            intersection_rows.append(
                {
                    "run": label,
                    "left_method": left,
                    "right_method": right,
                    "left_count": left_count,
                    "right_count": right_count,
                    "intersection_count": intersection,
                    "union_count": left_count + right_count - intersection,
                    "fraction_of_left_in_intersection": safe_fraction(
                        intersection, left_count
                    ),
                    "fraction_of_right_in_intersection": safe_fraction(
                        intersection, right_count
                    ),
                    "jaccard": jaccard(left_count, right_count, intersection),
                }
            )

    write_tsv(output_dir / "run_summary.tsv", summaries)
    write_tsv(output_dir / "method_sets.tsv", method_set_rows)
    write_tsv(output_dir / "method_intersections.tsv", intersection_rows)
    write_tsv(output_dir / "transition_counts.tsv", transition_rows)
    write_tsv(
        output_dir / "input_manifest.tsv",
        [asdict(item) for item in manifest],
    )

    snapshot_dir = output_dir / "code_snapshot"
    for source in snapshot_sources:
        copy_exclusive(source, snapshot_dir / source.name)

    completed_at = datetime.now(timezone.utc)
    provenance = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "command": list(sys.argv),
        "working_directory": str(Path.cwd().resolve()),
        "software_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "external_python_packages_used": {},
            "dependency_note": "The audit uses only the Python standard library.",
        },
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "test": str(test_path.resolve()),
        "test_sha256": sha256(test_path),
        "wes_used": False,
        "chromosomes": list(chromosomes),
        "pnonref_threshold": threshold,
        "runs": [
            {"label": label, "path": str(path.resolve())} for label, path in runs
        ],
        "monopogen_public_repository": PINNED_MONOPOGEN_REPOSITORY,
        "monopogen_public_commit": PINNED_MONOPOGEN_COMMIT,
        "monopogen_public_commit_url": (
            f"{PINNED_MONOPOGEN_REPOSITORY}/commit/{PINNED_MONOPOGEN_COMMIT}"
        ),
        "monopogen_paper": MONOPOGEN_PAPER,
        "method_definitions": definitions,
        "public_source_contract": {
            "beagle_invocations_and_second_pass_input": MONOPOGEN_BEAGLE_SOURCE,
            "downstream_exact_phased_record_lookup_and_read_support": (
                MONOPOGEN_DOWNSTREAM_SOURCE
            ),
        },
        "scope_limit": (
            "This audit isolates the Beagle-derived component. It does not claim "
            "that every phased record is a final Monopogen somatic call, because "
            "the full downstream code additionally evaluates raw read support."
        ),
        "outputs": [
            "run_summary.tsv",
            "method_sets.tsv",
            "method_intersections.tsv",
            "transition_counts.tsv",
            "input_manifest.tsv",
            "code_snapshot/monopogen_beagle_contract_audit.py",
            "code_snapshot/test_monopogen_beagle_contract_audit.py",
            "PROVENANCE.json",
            "RUN_COMPLETE",
            "OUTPUT_SHA256SUMS.txt",
        ],
        "checksum_scope": (
            "OUTPUT_SHA256SUMS.txt covers every regular output file except itself."
        ),
    }
    with (output_dir / "PROVENANCE.json").open("x") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "RUN_COMPLETE").open("x") as handle:
        handle.write(
            "Monopogen Beagle contract audit completed successfully at "
            f"{completed_at.isoformat()}.\n"
        )
    write_output_checksums(output_dir)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Run specification LABEL=PATH; repeat as needed (default: four local P4/P6 replicates)",
    )
    parser.add_argument(
        "--chromosomes",
        default="1-22",
        help="Comma-separated autosomes or intervals (default: 1-22)",
    )
    parser.add_argument(
        "--pnonref-threshold",
        type=float,
        default=0.90,
        help="Derived P(non-reference) threshold (default: 0.90)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs = tuple(args.run) if args.run else DEFAULT_RUNS
    chromosomes = parse_chromosomes(args.chromosomes)
    summaries = run_audit(
        runs=runs,
        chromosomes=chromosomes,
        threshold=args.pnonref_threshold,
        output_dir=args.output_dir,
    )
    for summary in summaries:
        print(
            summary["run"],
            f"downstream_records={summary['monopogen_downstream_beagle_records']}",
            f"paper_concordance={summary['monopogen_paper_raw_beagle_gt_concordant_nonref']}",
            f"pnonref={summary['sparcal_derived_pnonref_threshold_count']}",
            f"phased_nonref_diagnostic={summary['diagnostic_final_phased_nonref_gt']}",
            f"GP/phased_equal={summary['gp_phased_record_sets_equal']}",
        )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
