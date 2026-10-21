#!/usr/bin/env python3
"""Route raw spatial SNVs using Beagle genotype-posterior evidence.

This module deliberately separates *allele origin* from the noisy zygosity in
the pooled spatial-RNA mpileup.  Beagle is used only as evidence that the ALT
allele is inherited (or that the panel strongly supports reference); raw read
counts, FORMAT fields, and the raw genotype are never replaced.

The production router is WES-free. Matched WES is used only for internal method
development/evaluation in a separate benchmark script, never as pipeline input.

Four mutually exclusive routes are written:

``beagle_nonref_highconf``
    P(non-reference)=P(0/1)+P(1/1) reaches ``--min-anchor-probability``. This is
    the only hard germline-anchor route. Exact zygosity and called-GP remain
    separate evidence fields and are not required for this class decision.
``beagle_ref_highconf``
    P(0/0) reaches the threshold. This is negative germline evidence, not proof
    of a somatic mutation; the locus remains a downstream candidate.
``beagle_uncertain``
    The allele is represented in the panel-aligned Beagle output, but the
    non-reference and reference class probabilities are both below threshold,
    or GP is missing/malformed.
``panel_absent``
    The exact CHROM/POS/REF/ALT allele is absent from Beagle output.

For compatibility, ``all_filtered_in.vcf.gz`` contains only high-confidence
non-reference anchors, ``all_filtered_out.vcf.gz`` contains the other three
routes, and ``all_panel_discordant.vcf.gz`` contains represented-but-nonanchor
alleles (reference-high-confidence plus uncertain).  The explicit
``all_panel_present.vcf.gz`` compatibility view is the union of all three
Beagle-present routes.  It preserves the historical DLPFC feature-support
contract, but is not a germline callset. Explicit route VCFs and a lossless
audit table are authoritative.

The command refuses to reuse an existing output directory.  This makes every
ablation immutable by default and prevents accidental replacement of legacy
Beagle outputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ROUTER_SCHEMAS = {
    "p_nonref": "sparcal_beagle_evidence_v2_2",
    "called_gp": "sparcal_beagle_evidence_v1",
}
ROUTES = (
    "beagle_nonref_highconf",
    "beagle_ref_highconf",
    "beagle_uncertain",
    "panel_absent",
)
COMPATIBILITY_OUTPUTS = {
    "beagle_nonref_highconf": ("filtered_in", "panel_present"),
    "beagle_ref_highconf": ("filtered_out", "panel_discordant", "panel_present"),
    "beagle_uncertain": ("filtered_out", "panel_discordant", "panel_present"),
    "panel_absent": ("filtered_out",),
}


def open_text(path: Path):
    """Open plain or gzip-compressed text based on the filename."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else path.open("rt")


def normalize_gt(gt: Optional[str]) -> Optional[str]:
    """Return an unphased, allele-order-normalized diploid genotype."""
    if not gt or gt in {".", "./.", ".|."}:
        return None
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) != 2 or any(allele == "." for allele in alleles):
        return None
    try:
        alleles = sorted(alleles, key=int)
    except ValueError:
        alleles = sorted(alleles)
    return "/".join(alleles)


def extract_format_value(
    format_string: str, sample_string: str, field_name: str
) -> Optional[str]:
    fields = format_string.split(":")
    values = sample_string.split(":")
    if field_name not in fields:
        return None
    index = fields.index(field_name)
    return values[index] if index < len(values) else None


def parse_gp(gp: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """Parse reported biallelic-diploid GP=(P00,P01,P11), or return None.

    Beagle 4.1 prints rounded values that occasionally sum to 0.99 or 1.01.
    Preserve the reported probabilities for thresholding; normalization is
    used only for entropy below.
    """
    if not gp or gp == ".":
        return None
    try:
        values = tuple(float(value) for value in gp.split(","))
    except ValueError:
        return None
    if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
        return None
    total = sum(values)
    if total <= 0:
        return None
    return values  # type: ignore[return-value]


def parse_info_value(info: str, field_name: str) -> Optional[str]:
    prefix = field_name + "="
    for item in info.split(";"):
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def variant_key(fields: Sequence[str]) -> Tuple[str, int, str, str]:
    return fields[0], int(fields[1]), fields[3], fields[4]


@dataclass(frozen=True)
class VcfContract:
    """Structural facts established while strictly scanning one VCF."""

    sample_id: str
    declared_contigs: Tuple[str, ...]
    data_contigs: Tuple[str, ...]
    records_by_contig: Mapping[str, int]
    record_count: int


def _declared_contig(line: str) -> Optional[str]:
    prefix = "##contig=<ID="
    if not line.startswith(prefix):
        return None
    value = line[len(prefix):].split(",", 1)[0].split(">", 1)[0]
    return value or None


def _validate_vcf_data_fields(
    fields: Sequence[str], path: Path, line_number: int
) -> None:
    if len(fields) != 10:
        raise ValueError(
            f"{path}:{line_number}: expected 10 columns for a single-sample "
            f"VCF record, found {len(fields)}"
        )
    if fields[0] in {"", "."}:
        raise ValueError(f"{path}:{line_number}: missing CHROM")
    try:
        position = int(fields[1])
    except ValueError as error:
        raise ValueError(
            f"{path}:{line_number}: invalid POS {fields[1]!r}"
        ) from error
    if position <= 0:
        raise ValueError(f"{path}:{line_number}: POS must be positive")
    if fields[3] in {"", "."} or fields[4] in {"", "."}:
        raise ValueError(f"{path}:{line_number}: REF and ALT must be present")
    if fields[8] in {"", "."}:
        raise ValueError(f"{path}:{line_number}: FORMAT must be present")


def _scan_single_sample_vcf(
    path: Path,
    on_record: Optional[Callable[[Sequence[str], int], None]] = None,
) -> VcfContract:
    """Validate a VCF and optionally consume each structurally valid record."""
    header_seen = False
    sample_id: Optional[str] = None
    declared_contigs: Dict[str, None] = {}
    records_by_contig: Dict[str, int] = {}
    record_count = 0
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                contig = _declared_contig(line)
                if contig is not None:
                    declared_contigs.setdefault(contig, None)
                continue
            if line.startswith("#CHROM"):
                if header_seen:
                    raise ValueError(f"{path}:{line_number}: duplicate #CHROM header")
                columns = line.rstrip("\n").split("\t")
                fixed = [
                    "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
                    "INFO", "FORMAT",
                ]
                if len(columns) != 10 or columns[:9] != fixed:
                    raise ValueError(
                        f"{path}:{line_number}: expected exactly one sample after "
                        "the nine standard VCF columns"
                    )
                if columns[9] in {"", "."}:
                    raise ValueError(f"{path}:{line_number}: missing sample ID")
                sample_id = columns[9]
                header_seen = True
                continue
            if line.startswith("#"):
                continue
            if not header_seen:
                raise ValueError(
                    f"{path}:{line_number}: VCF data encountered before #CHROM header"
                )
            fields = line.rstrip("\n").split("\t")
            _validate_vcf_data_fields(fields, path, line_number)
            records_by_contig[fields[0]] = records_by_contig.get(fields[0], 0) + 1
            record_count += 1
            if on_record is not None:
                on_record(fields, line_number)
    if not header_seen or sample_id is None:
        raise ValueError(f"{path}: missing #CHROM header")
    return VcfContract(
        sample_id=sample_id,
        declared_contigs=tuple(declared_contigs),
        data_contigs=tuple(records_by_contig),
        records_by_contig=dict(records_by_contig),
        record_count=record_count,
    )


@dataclass(frozen=True)
class BeagleEvidence:
    gt: Optional[str]
    gp_text: Optional[str]
    gp: Optional[Tuple[float, float, float]]
    called_gp: Optional[float]
    p_nonref: Optional[float]
    entropy: Optional[float]
    margin: Optional[float]
    beagle_af: Optional[float]


def summarize_evidence(
    gt: Optional[str], gp_text: Optional[str], beagle_af_text: Optional[str] = None
) -> BeagleEvidence:
    normalized_gt = normalize_gt(gt)
    gp = parse_gp(gp_text)
    gt_index = {"0/0": 0, "0/1": 1, "1/1": 2}.get(normalized_gt)
    called_gp = gp[gt_index] if gp is not None and gt_index is not None else None
    p_nonref = gp[1] + gp[2] if gp is not None else None
    if gp is None:
        entropy = None
        margin = None
    else:
        total = sum(gp)
        normalized_gp = tuple(value / total for value in gp)
        entropy = -sum(
            value * math.log(value) for value in normalized_gp if value > 0
        ) / math.log(3)
        ordered = sorted(gp, reverse=True)
        margin = ordered[0] - ordered[1]
    try:
        beagle_af = float(beagle_af_text) if beagle_af_text not in {None, "."} else None
    except ValueError:
        beagle_af = None
    return BeagleEvidence(
        gt=normalized_gt,
        gp_text=gp_text,
        gp=gp,
        called_gp=called_gp,
        p_nonref=p_nonref,
        entropy=entropy,
        margin=margin,
        beagle_af=beagle_af,
    )


def meets_threshold(value: Optional[float], threshold: float) -> bool:
    """Compare reported decimal probabilities without binary-float edge loss."""
    return value is not None and (value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12))


def route_evidence(
    evidence: Optional[BeagleEvidence], min_anchor_probability: float,
    anchor_score: str = "p_nonref",
) -> str:
    """Assign one of the four evidence routes without using WES."""
    if anchor_score not in ROUTER_SCHEMAS:
        raise ValueError(f"unknown anchor_score: {anchor_score}")
    if evidence is None:
        return "panel_absent"
    if anchor_score == "called_gp":
        if not meets_threshold(evidence.called_gp, min_anchor_probability):
            return "beagle_uncertain"
        if evidence.gt in {"0/1", "1/1"}:
            return "beagle_nonref_highconf"
        if evidence.gt == "0/0":
            return "beagle_ref_highconf"
        return "beagle_uncertain"
    if meets_threshold(evidence.p_nonref, min_anchor_probability):
        return "beagle_nonref_highconf"
    if evidence.gp is not None and meets_threshold(evidence.gp[0], min_anchor_probability):
        return "beagle_ref_highconf"
    return "beagle_uncertain"


def compatibility_outputs(route: str) -> Tuple[str, ...]:
    """Return legacy output aliases for a mutually exclusive evidence route."""
    try:
        return COMPATIBILITY_OUTPUTS[route]
    except KeyError as error:
        raise ValueError(f"unknown route: {route}") from error


def _collect_beagle_bundle(
    paths: Iterable[Path],
) -> Tuple[
    Dict[Tuple[str, int, str, str], BeagleEvidence],
    Optional[str],
    Tuple[VcfContract, ...],
]:
    """Strictly validate Beagle VCFs and collect exact-allele evidence."""
    evidence: Dict[Tuple[str, int, str, str], BeagleEvidence] = {}
    contracts = []
    sample_id: Optional[str] = None
    for path in paths:
        path = Path(path)

        def consume(fields: Sequence[str], line_number: int) -> None:
            key = variant_key(fields)
            if key in evidence:
                raise ValueError(
                    f"{path}:{line_number}: duplicate exact allele across "
                    f"Beagle inputs: {key}"
                )
            gt = extract_format_value(fields[8], fields[9], "GT")
            gp = extract_format_value(fields[8], fields[9], "GP")
            evidence[key] = summarize_evidence(
                gt, gp, parse_info_value(fields[7], "AF")
            )

        contract = _scan_single_sample_vcf(path, consume)
        if sample_id is None:
            sample_id = contract.sample_id
        elif contract.sample_id != sample_id:
            raise ValueError(
                "Beagle VCF sample ID mismatch: "
                f"expected {sample_id!r}, found {contract.sample_id!r} in {path}"
            )
        contracts.append(contract)
    return evidence, sample_id, tuple(contracts)


def collect_beagle_evidence(
    paths: Iterable[Path],
) -> Dict[Tuple[str, int, str, str], BeagleEvidence]:
    """Collect Beagle evidence after enforcing the single-sample contract."""
    evidence, _, _ = _collect_beagle_bundle(paths)
    return evidence


def _preflight_inputs(
    raw_vcf: Path,
    beagle_vcfs: Sequence[Path],
    expected_contigs: Optional[Iterable[str]],
    require_exact_overlap: bool,
    contig_status_path: Optional[Path] = None,
) -> Tuple[
    Dict[Tuple[str, int, str, str], BeagleEvidence],
    VcfContract,
    str,
    Tuple[str, ...],
    int,
    Mapping[str, object],
]:
    """Validate raw/Beagle coverage, with safe deterministic CLI inference.

    When expected contigs are omitted, they are inferred from the raw VCF's
    declared and observed contigs. A raw-nonempty contig still requires an exact
    Beagle overlap unless an upstream runner status file explicitly proves that
    the contig was panel-absent and therefore intentionally skipped.
    """
    evidence, beagle_sample_id, beagle_contracts = _collect_beagle_bundle(
        beagle_vcfs
    )
    if beagle_sample_id is None:
        raise ValueError("at least one Beagle VCF is required")

    exact_overlap_count = 0
    exact_overlap_by_contig: Dict[str, int] = {}

    def observe_raw(fields: Sequence[str], line_number: int) -> None:
        nonlocal exact_overlap_count
        if variant_key(fields) in evidence:
            exact_overlap_count += 1
            contig = fields[0]
            exact_overlap_by_contig[contig] = (
                exact_overlap_by_contig.get(contig, 0) + 1
            )

    raw_contract = _scan_single_sample_vcf(raw_vcf, observe_raw)
    if raw_contract.sample_id != beagle_sample_id:
        raise ValueError(
            "raw/Beagle sample ID mismatch: "
            f"raw={raw_contract.sample_id!r}, Beagle={beagle_sample_id!r}"
        )

    upstream_status = None
    upstream_by_contig = {}
    upstream_expected: Tuple[str, ...] = ()
    if contig_status_path is not None:
        contig_status_path = Path(contig_status_path)
        if not contig_status_path.is_file():
            raise FileNotFoundError(contig_status_path)
        try:
            upstream_status = json.loads(contig_status_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid upstream contig-status JSON: {contig_status_path}: {error}"
            ) from error
        if upstream_status.get("production_inputs_include_wes") is not False:
            raise ValueError(
                f"upstream contig status is not declared WES-free: {contig_status_path}"
            )
        if upstream_status.get("raw_sample_id") != raw_contract.sample_id:
            raise ValueError(
                f"upstream/raw sample ID mismatch: {contig_status_path}"
            )
        try:
            upstream_raw_count = int(upstream_status["raw_record_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid upstream raw_record_count: {contig_status_path}"
            ) from error
        if upstream_raw_count != raw_contract.record_count:
            raise ValueError(
                f"upstream/raw record-count mismatch: {upstream_raw_count} != "
                f"{raw_contract.record_count}"
            )
        raw_upstream_expected = upstream_status.get("expected_contigs")
        if not isinstance(raw_upstream_expected, list):
            raise ValueError(
                f"upstream expected_contigs must be a list: {contig_status_path}"
            )
        upstream_expected = tuple(dict.fromkeys(raw_upstream_expected))
        if (
            len(upstream_expected) != len(raw_upstream_expected)
            or any(not isinstance(contig, str) or not contig for contig in upstream_expected)
        ):
            raise ValueError(
                f"invalid upstream expected_contigs: {contig_status_path}"
            )
        status_rows = upstream_status.get("contigs")
        if not isinstance(status_rows, list):
            raise ValueError(
                f"upstream contigs must be a list: {contig_status_path}"
            )
        for row in status_rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid upstream contig row: {contig_status_path}")
            contig = row.get("chromosome")
            if contig not in upstream_expected:
                raise ValueError(
                    f"unexpected upstream contig status {contig!r}: {contig_status_path}"
                )
            if contig in upstream_by_contig:
                raise ValueError(
                    f"duplicate upstream contig status {contig!r}: {contig_status_path}"
                )
            upstream_by_contig[contig] = row
        missing_status = [
            contig for contig in upstream_expected if contig not in upstream_by_contig
        ]
        if missing_status:
            raise ValueError(
                "upstream status is missing contigs: " + ", ".join(missing_status)
            )

    if expected_contigs is None:
        normalized_expected = tuple(dict.fromkeys((
            *raw_contract.declared_contigs,
            *raw_contract.data_contigs,
            *upstream_expected,
        )))
        expected_contigs_source = "inferred"
    else:
        normalized_expected = tuple(dict.fromkeys(expected_contigs))
        expected_contigs_source = "explicit"
    if not normalized_expected or any(
        not isinstance(contig, str) or not contig for contig in normalized_expected
    ):
        raise ValueError("expected_contigs must contain at least one nonempty ID")

    unexpected_raw = sorted(
        set(raw_contract.data_contigs) - set(normalized_expected)
    )
    if unexpected_raw:
        raise ValueError(
            "raw VCF has nonempty contigs outside expected_contigs: "
            + ", ".join(unexpected_raw)
        )

    declared = {}
    data_contigs = {}
    for contract in beagle_contracts:
        for contig in contract.declared_contigs:
            declared.setdefault(contig, None)
        for contig in contract.data_contigs:
            data_contigs.setdefault(contig, None)
    covered_contigs = tuple(dict.fromkeys((*declared, *data_contigs)))
    unexpected_beagle_data = sorted(
        set(data_contigs) - set(normalized_expected)
    )
    if unexpected_beagle_data:
        raise ValueError(
            "Beagle inputs contain data outside expected_contigs: "
            + ", ".join(unexpected_beagle_data)
        )

    beagle_resolved_paths = {Path(path).resolve() for path in beagle_vcfs}
    invalid = []
    for contig in normalized_expected:
        raw_count = int(raw_contract.records_by_contig.get(contig, 0))
        exact_count = int(exact_overlap_by_contig.get(contig, 0))
        status_row = upstream_by_contig.get(contig)
        status = status_row.get("status") if status_row is not None else None
        if status_row is not None:
            try:
                reported_raw_count = int(status_row.get("raw_marker_count"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{contig}: invalid upstream raw_marker_count"
                ) from error
            if reported_raw_count != raw_count:
                raise ValueError(
                    f"{contig}: upstream raw_marker_count={reported_raw_count} "
                    f"does not match raw VCF count={raw_count}"
                )
            if status not in {"completed", "panel_absent", "raw_empty"}:
                raise ValueError(f"{contig}: invalid upstream status {status!r}")
            if status == "completed":
                output_file = status_row.get("output_file")
                if (
                    status_row.get("has_reference_marker") is not True
                    or status_row.get("beagle_required") is not True
                    or status_row.get("route_expectation") != "beagle_evidence_required"
                    or not isinstance(output_file, str)
                    or Path(output_file).resolve() not in beagle_resolved_paths
                ):
                    raise ValueError(
                        f"{contig}: invalid completed upstream status contract"
                    )
            elif status == "panel_absent":
                if (
                    status_row.get("has_reference_marker") is not False
                    or status_row.get("beagle_required") is not False
                    or status_row.get("route_expectation")
                    != "all_raw_alleles_panel_absent"
                    or status_row.get("output_file") is not None
                ):
                    raise ValueError(
                        f"{contig}: invalid panel_absent upstream status contract"
                    )
            elif status == "raw_empty":
                if (
                    status_row.get("has_reference_marker") is not False
                    or status_row.get("beagle_required") is not False
                    or status_row.get("route_expectation") != "no_raw_records"
                    or status_row.get("output_file") is not None
                ):
                    raise ValueError(
                        f"{contig}: invalid raw_empty upstream status contract"
                    )

        if raw_count == 0:
            if status not in {None, "raw_empty"}:
                raise ValueError(
                    f"{contig}: upstream status {status!r} contradicts an empty raw contig"
                )
            continue
        if status == "raw_empty":
            raise ValueError(
                f"{contig}: upstream raw_empty status contradicts {raw_count} raw records"
            )
        if status == "panel_absent":
            if contig in data_contigs or exact_count:
                raise ValueError(
                    f"{contig}: panel_absent status contradicts Beagle data"
                )
            continue
        if contig not in data_contigs or (require_exact_overlap and exact_count == 0):
            invalid.append(contig)

    if invalid:
        qualifier = (
            "no explicit upstream panel-absent status was available; "
            if contig_status_path is None else ""
        )
        raise ValueError(
            "Beagle inputs lack per-contig exact raw-allele coverage for: "
            + ", ".join(invalid)
            + "; " + qualifier
            + "supply --expected-contigs and trusted upstream contig status"
        )

    if (
        require_exact_overlap
        and raw_contract.record_count > 0
        and exact_overlap_count == 0
        and not all(
            upstream_by_contig.get(contig, {}).get("status") == "panel_absent"
            for contig in raw_contract.data_contigs
        )
    ):
        raise ValueError(
            "zero exact CHROM/POS/REF/ALT overlap between a nonempty raw VCF "
            "and the Beagle bundle; check genome build, contig names, and allele encoding"
        )

    validation = {
        "expected_contigs": list(normalized_expected),
        "expected_contigs_source": expected_contigs_source,
        "exact_overlap_by_contig": {
            contig: int(exact_overlap_by_contig.get(contig, 0))
            for contig in normalized_expected
        },
        "upstream_contig_status_file": (
            str(contig_status_path.resolve()) if contig_status_path is not None else None
        ),
        "upstream_contig_status_sha256": (
            _sha256(contig_status_path) if contig_status_path is not None else None
        ),
    }
    return (
        evidence,
        raw_contract,
        beagle_sample_id,
        covered_contigs,
        exact_overlap_count,
        validation,
    )


def _number(value: Optional[float]) -> str:
    return "." if value is None else f"{value:.8g}"


def _annotate_line(
    line: str,
    route: str,
    raw_gt: Optional[str],
    evidence: Optional[BeagleEvidence],
    schema: str,
) -> Tuple[str, Mapping[str, str]]:
    fields = line.rstrip("\n").split("\t")
    ev = evidence or BeagleEvidence(None, None, None, None, None, None, None, None)
    transition = f"{raw_gt or '.'}>{ev.gt or '.'}"
    annotations = {
        "BEAGLE_EVIDENCE_SCHEMA": schema,
        "BEAGLE_ROUTE": route,
        "BEAGLE_GT": ev.gt or ".",
        "BEAGLE_GP": ev.gp_text or ".",
        "BEAGLE_CALLED_GP": _number(ev.called_gp),
        "BEAGLE_P_NONREF": _number(ev.p_nonref),
        "BEAGLE_GP_ENTROPY": _number(ev.entropy),
        "BEAGLE_GP_MARGIN": _number(ev.margin),
        "BEAGLE_AF": _number(ev.beagle_af),
        "BEAGLE_GT_TRANSITION": transition,
    }
    addition = ";".join(f"{key}={value}" for key, value in annotations.items())
    fields[7] = addition if fields[7] == "." else fields[7] + ";" + addition
    return "\t".join(fields) + "\n", annotations


def _meta_headers(schema: str, anchor_score: str,
                  min_anchor_probability: float) -> Sequence[str]:
    return (
        f'##SPARCAL_BEAGLE_ROUTER=<Schema="{schema}",AnchorScore="{anchor_score}",MinAnchorProbability={min_anchor_probability:g},HardAnchor="beagle_nonref_highconf">\n',
        '##INFO=<ID=BEAGLE_EVIDENCE_SCHEMA,Number=1,Type=String,Description="Versioned SPARCAL Beagle evidence schema">\n',
        '##INFO=<ID=BEAGLE_ROUTE,Number=1,Type=String,Description="Beagle evidence route: beagle_nonref_highconf, beagle_ref_highconf, beagle_uncertain, or panel_absent">\n',
        '##INFO=<ID=BEAGLE_GT,Number=1,Type=String,Description="Unphased Beagle-refined genotype; raw FORMAT genotype is unchanged">\n',
        '##INFO=<ID=BEAGLE_GP,Number=G,Type=Float,Description="Beagle genotype posterior probabilities for 0/0,0/1,1/1">\n',
        '##INFO=<ID=BEAGLE_CALLED_GP,Number=1,Type=Float,Description="Posterior probability assigned to BEAGLE_GT">\n',
        '##INFO=<ID=BEAGLE_P_NONREF,Number=1,Type=Float,Description="Posterior probability P(0/1)+P(1/1) of a non-reference genotype">\n',
        '##INFO=<ID=BEAGLE_GP_ENTROPY,Number=1,Type=Float,Description="Normalized entropy of the three Beagle genotype probabilities (0 certain, 1 uniform)">\n',
        '##INFO=<ID=BEAGLE_GP_MARGIN,Number=1,Type=Float,Description="Difference between largest and second-largest Beagle genotype probabilities">\n',
        '##INFO=<ID=BEAGLE_AF,Number=1,Type=Float,Description="Estimated ALT allele frequency emitted by Beagle; not interpreted as reference-panel population AF">\n',
        '##INFO=<ID=BEAGLE_GT_TRANSITION,Number=1,Type=String,Description="Unphased raw genotype followed by Beagle genotype; raw genotype is not overwritten">\n',
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_vcf(
    raw_vcf: Path,
    beagle_vcfs: Sequence[Path],
    output_dir: Path,
    min_anchor_probability: float = 0.90,
    anchor_score: str = "p_nonref",
    bgzip: str = "bgzip",
    tabix: str = "tabix",
    min_called_gp: Optional[float] = None,
    allow_existing_output_dir: bool = False,
    expected_contigs: Optional[Iterable[str]] = None,
    require_exact_overlap: bool = True,
    contig_status_path: Optional[Path] = None,
) -> Mapping[str, int]:
    """Create a new immutable route bundle and enforce count conservation."""
    # The old keyword/CLI selects called-GP routing and reproduces v1 exactly.
    if min_called_gp is not None:
        anchor_score = "called_gp"
        min_anchor_probability = min_called_gp
    if anchor_score not in ROUTER_SCHEMAS:
        raise ValueError(f"unknown anchor_score: {anchor_score}")
    if not 0 <= min_anchor_probability <= 1:
        raise ValueError("min_anchor_probability must be between 0 and 1")
    schema = ROUTER_SCHEMAS[anchor_score]
    route_files = {route: output_dir / f"all_{route}.vcf" for route in ROUTES}
    route_files.update({
        "filtered_in": output_dir / "all_filtered_in.vcf",
        "filtered_out": output_dir / "all_filtered_out.vcf",
        "panel_discordant": output_dir / "all_panel_discordant.vcf",
        "panel_present": output_dir / "all_panel_present.vcf",
    })
    generated_paths = [
        output_dir / "beagle_evidence_audit.tsv.gz",
        output_dir / "beagle_evidence_summary.tsv",
        output_dir / "MANIFEST.json",
        output_dir / "RUN_COMPLETE",
    ]
    for path in route_files.values():
        generated_paths.extend((
            path,
            Path(str(path) + ".gz"),
            Path(str(path) + ".gz.tbi"),
        ))
    create_output_dir = not output_dir.exists()
    if output_dir.exists():
        if not allow_existing_output_dir:
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output_dir}"
            )
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        conflicts = [path for path in generated_paths if path.exists()]
        if conflicts:
            preview = ", ".join(str(path) for path in conflicts[:5])
            raise FileExistsError(f"refusing to overwrite route outputs: {preview}")
    if not raw_vcf.is_file():
        raise FileNotFoundError(raw_vcf)
    if not beagle_vcfs:
        raise ValueError("at least one Beagle VCF is required")
    for path in beagle_vcfs:
        if not path.is_file():
            raise FileNotFoundError(path)

    normalized_expected_contigs = (
        tuple(dict.fromkeys(expected_contigs))
        if expected_contigs is not None
        else None
    )
    if contig_status_path is None and beagle_vcfs:
        parent_dirs = {Path(path).resolve().parent for path in beagle_vcfs}
        if len(parent_dirs) == 1:
            candidate = next(iter(parent_dirs)) / "BEAGLE_CONTIG_STATUS.json"
            if candidate.is_file():
                contig_status_path = candidate
    (
        evidence,
        raw_contract,
        beagle_sample_id,
        covered_contigs,
        exact_overlap_count,
        contig_validation,
    ) = _preflight_inputs(
        raw_vcf,
        beagle_vcfs,
        normalized_expected_contigs,
        require_exact_overlap,
        contig_status_path,
    )
    if create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=False)
    handles = {name: path.open("wt") for name, path in route_files.items()}
    counts = {route: 0 for route in ROUTES}
    total = 0
    audit_path = output_dir / "beagle_evidence_audit.tsv.gz"
    audit_columns = (
        "schema", "chrom", "pos", "ref", "alt", "raw_gt", "beagle_gt",
        "beagle_gp", "beagle_called_gp", "beagle_p_nonref", "beagle_gp_entropy",
        "beagle_gp_margin", "beagle_af", "gt_transition", "route", "hard_anchor",
    )
    try:
        with gzip.open(audit_path, "wt") as audit, open_text(raw_vcf) as raw_handle:
            audit.write("\t".join(audit_columns) + "\n")
            for line_number, line in enumerate(raw_handle, start=1):
                if line.startswith("#"):
                    if line.startswith("#CHROM"):
                        for handle in handles.values():
                            handle.writelines(_meta_headers(schema, anchor_score, min_anchor_probability))
                    for handle in handles.values():
                        handle.write(line)
                    continue

                fields = line.rstrip("\n").split("\t")
                _validate_vcf_data_fields(fields, raw_vcf, line_number)
                key = variant_key(fields)
                raw_gt = None
                if len(fields) >= 10:
                    raw_gt = normalize_gt(extract_format_value(fields[8], fields[9], "GT"))
                ev = evidence.get(key)
                route = route_evidence(ev, min_anchor_probability, anchor_score)
                routed_line, annotations = _annotate_line(line, route, raw_gt, ev, schema)
                handles[route].write(routed_line)
                for compatibility in compatibility_outputs(route):
                    handles[compatibility].write(routed_line)
                counts[route] += 1
                total += 1
                audit.write("\t".join((
                    schema, fields[0], fields[1], fields[3], fields[4], raw_gt or ".",
                    annotations["BEAGLE_GT"], annotations["BEAGLE_GP"],
                    annotations["BEAGLE_CALLED_GP"], annotations["BEAGLE_P_NONREF"],
                    annotations["BEAGLE_GP_ENTROPY"], annotations["BEAGLE_GP_MARGIN"],
                    annotations["BEAGLE_AF"], annotations["BEAGLE_GT_TRANSITION"], route,
                    "1" if route == "beagle_nonref_highconf" else "0",
                )) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    if sum(counts.values()) != total:
        raise AssertionError("route counts do not conserve raw variants")

    bgzip_path = shutil.which(bgzip) or bgzip
    tabix_path = shutil.which(tabix) or tabix
    for path in route_files.values():
        subprocess.run([bgzip_path, "-f", str(path)], check=True)
        subprocess.run([tabix_path, "-f", "-p", "vcf", str(path) + ".gz"], check=True)

    with (output_dir / "beagle_evidence_summary.tsv").open("wt") as handle:
        handle.write("schema\troute\tcount\tanchor_score\tmin_anchor_probability\thard_anchor\n")
        for route in ROUTES:
            handle.write(
                f"{schema}\t{route}\t{counts[route]}\t{anchor_score}\t{min_anchor_probability:g}\t"
                f"{int(route == 'beagle_nonref_highconf')}\n"
            )
        handle.write(f"{schema}\ttotal\t{total}\t{anchor_score}\t{min_anchor_probability:g}\t.\n")

    manifest = {
        "schema": schema,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "raw_vcf": str(raw_vcf.resolve()),
        "raw_vcf_sha256": _sha256(raw_vcf),
        "raw_sample_id": raw_contract.sample_id,
        "raw_record_count": raw_contract.record_count,
        "beagle_vcfs": [str(path.resolve()) for path in beagle_vcfs],
        "beagle_vcf_count": len(beagle_vcfs),
        "beagle_sample_id": beagle_sample_id,
        "beagle_covered_contigs": list(covered_contigs),
        "expected_contigs": contig_validation["expected_contigs"],
        "expected_contigs_source": contig_validation["expected_contigs_source"],
        "exact_overlap_count": exact_overlap_count,
        "exact_overlap_by_contig": contig_validation["exact_overlap_by_contig"],
        "upstream_contig_status_file": contig_validation["upstream_contig_status_file"],
        "upstream_contig_status_sha256": contig_validation["upstream_contig_status_sha256"],
        "anchor_score": anchor_score,
        "min_anchor_probability": min_anchor_probability,
        "hard_anchor_route": "beagle_nonref_highconf",
        "compatibility": {
            "all_filtered_in.vcf.gz": "beagle_nonref_highconf only",
            "all_filtered_out.vcf.gz": "union of other three routes",
            "all_panel_discordant.vcf.gz": (
                "union of beagle_ref_highconf and beagle_uncertain"
            ),
            "all_panel_present.vcf.gz": (
                "union of beagle_nonref_highconf, beagle_ref_highconf, and "
                "beagle_uncertain; feature-support compatibility view only, "
                "not a germline callset"
            ),
        },
        "counts": {**counts, "total": total, "beagle_records": len(evidence)},
        "router_source": str(Path(__file__).resolve()),
        "router_source_sha256": _sha256(Path(__file__).resolve()),
    }
    with (output_dir / "MANIFEST.json").open("wt") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "RUN_COMPLETE").write_text("complete\n")
    return counts


def _chromosome_order(path: Path) -> Tuple[int, str]:
    stem = path.name.split(".", 1)[0].removeprefix("chr")
    try:
        return int(stem), path.name
    except ValueError:
        return 10_000, path.name


def parse_expected_contigs(values: Sequence[str]) -> Tuple[str, ...]:
    """Parse comma- or space-separated exact contig IDs for CLI validation."""
    contigs = []
    seen = set()
    for value in values:
        for item in value.split(","):
            contig = item.strip()
            if not contig:
                continue
            if contig not in seen:
                seen.add(contig)
                contigs.append(contig)
    if not contigs:
        raise ValueError("expected contigs must contain at least one nonempty ID")
    return tuple(contigs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-vcf", type=Path, required=True)
    parser.add_argument(
        "--beagle-dir", type=Path, required=True,
        help="Directory containing per-chromosome chr*.vcf.gz Beagle outputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-contigs", nargs="+", required=False, default=None,
        help=("Preferred explicit contig IDs; if omitted, infer deterministically "
              "from raw VCF header/data and validate each raw-nonempty contig"),
    )
    parser.add_argument(
        "--contig-status-file", type=Path, default=None,
        help=("Trusted upstream BEAGLE_CONTIG_STATUS.json; by default it is "
              "auto-detected beside the per-contig Beagle VCFs"),
    )
    parser.add_argument(
        "--anchor-score", choices=tuple(ROUTER_SCHEMAS), default="p_nonref",
    )
    parser.add_argument("--min-anchor-probability", type=float, default=0.90)
    parser.add_argument(
        "--min-called-gp", type=float, default=None,
        help="Deprecated v1 compatibility alias; selects called_gp routing",
    )
    parser.add_argument("--bgzip", default="bgzip")
    parser.add_argument("--tabix", default="tabix")
    parser.add_argument(
        "--allow-existing-output-dir", action="store_true",
        help=("Allow per-chromosome Beagle files in output-dir, but still "
              "refuse to overwrite any router-generated file"),
    )
    parser.add_argument(
        "--allow-zero-exact-overlap", action="store_true",
        help=("Permit nonempty raw and Beagle inputs with zero exact "
              "CHROM/POS/REF/ALT overlap (unsafe; intended for diagnostics)"),
    )
    args = parser.parse_args()
    beagle_vcfs = sorted(args.beagle_dir.glob("chr*.vcf.gz"), key=_chromosome_order)
    counts = route_vcf(
        args.raw_vcf,
        beagle_vcfs,
        args.output_dir,
        min_anchor_probability=args.min_anchor_probability,
        anchor_score=args.anchor_score,
        min_called_gp=args.min_called_gp,
        bgzip=args.bgzip,
        tabix=args.tabix,
        allow_existing_output_dir=args.allow_existing_output_dir,
        expected_contigs=(
            parse_expected_contigs(args.expected_contigs)
            if args.expected_contigs is not None else None
        ),
        require_exact_overlap=not args.allow_zero_exact_overlap,
        contig_status_path=args.contig_status_file,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
