#!/usr/bin/env python3
"""Fail-closed marker and artifact contracts for SPARCAL Beagle v2.3.

The biological router is deliberately kept separate from these checks. This
module proves that Beagle returned every raw exact allele represented in the
reference panel, and that a completed output bundle has not been truncated or
mutated. It never reads WES data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Set, Tuple


MARKER_CONTRACT_SCHEMA = "sparcal_beagle_marker_contract_v1"
COMPLETION_SCHEMA = "sparcal_beagle_bundle_completion_v1"
RUN_CONTRACT_SCHEMA = "sparcal_beagle_run_v2_3_candidate_1"
VariantKey = Tuple[str, int, str, str]


def open_text(path: Path):
    """Open a plain or gzip-compressed text file."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else path.open("rt")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_key(key: VariantKey) -> bytes:
    chrom, position, ref, alt = key
    if any("\t" in value or "\n" in value for value in (chrom, ref, alt)):
        raise ValueError(f"invalid control character in allele key: {key!r}")
    return f"{chrom}\t{position}\t{ref}\t{alt}\n".encode("utf-8")


def digest_keys(keys: Iterable[VariantKey]) -> str:
    """Hash a set of exact alleles in a stable genomic/textual order."""
    digest = hashlib.sha256()
    for key in sorted(
        set(keys), key=lambda value: (value[0], value[1], value[2], value[3])
    ):
        digest.update(_canonical_key(key))
    return digest.hexdigest()


def _record_keys(
    fields: Sequence[str], *, split_alt: bool
) -> Iterator[VariantKey]:
    if len(fields) < 5:
        raise ValueError(f"VCF record has fewer than five fields: {fields!r}")
    chrom, pos_text, ref, alt_text = (
        fields[0],
        fields[1],
        fields[3],
        fields[4],
    )
    try:
        position = int(pos_text)
    except ValueError as error:
        raise ValueError(f"invalid VCF position: {pos_text!r}") from error
    if position <= 0 or chrom in {"", "."} or ref in {"", "."}:
        raise ValueError(f"invalid VCF allele fields: {fields[:5]!r}")
    alts = alt_text.split(",") if split_alt else [alt_text]
    if not split_alt and "," in alt_text:
        raise ValueError(
            "raw/Beagle VCF must be split to biallelic records before routing: "
            f"{chrom}:{position}:{ref}:{alt_text}"
        )
    for alt in alts:
        if alt in {"", ".", "*"} or alt.startswith("<"):
            continue
        yield chrom, position, ref, alt


def read_exact_keys(
    path: Path,
    *,
    contig: str,
    split_alt: bool = False,
) -> Set[VariantKey]:
    """Read unique exact alleles for one contig from a VCF-like file."""
    keys: Set[VariantKey] = set()
    with open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] != contig:
                continue
            for key in _record_keys(fields, split_alt=split_alt):
                if key in keys:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate exact allele {key!r}"
                    )
                keys.add(key)
    return keys


def _panel_overlap_from_lines(
    lines: Iterable[str],
    raw_keys: Set[VariantKey],
    *,
    source: str,
) -> Set[VariantKey]:
    """Return panel records with Beagle-corresponding full allele lists.

    Beagle requires corresponding target/reference markers to have identical
    REF and ALT allele sequences. A biallelic raw record therefore does not
    correspond to one ALT extracted from a multi-ALT reference record.
    """
    overlap: Set[VariantKey] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 5:
            raise ValueError(
                f"{source}:{line_number}: VCF record has fewer than five fields"
            )
        try:
            position = int(fields[1])
        except ValueError as error:
            raise ValueError(
                f"{source}:{line_number}: invalid VCF position {fields[1]!r}"
            ) from error
        chrom, ref, alt = fields[0], fields[3], fields[4]
        if (
            position <= 0
            or chrom in {"", "."}
            or ref in {"", "."}
            or alt in {"", ".", "*"}
            or alt.startswith("<")
        ):
            continue
        key = (chrom, position, ref, alt)
        if key in raw_keys:
            overlap.add(key)
    return overlap


def query_reference_overlap(
    raw_keys: Set[VariantKey],
    reference_panel: Path,
    *,
    contig: str,
    bcftools: str = "bcftools",
    temp_dir: Optional[Path] = None,
) -> Set[VariantKey]:
    """Return raw exact alleles intersected with the panel for one contig.

    bcftools view -R limits decompression to raw positions. The returned panel
    records are checked by exact full REF/ALT lists; position overlap and one
    ALT extracted from a multi-ALT panel record are not correspondence.
    """
    reference_panel = Path(reference_panel)
    if not reference_panel.is_file():
        raise FileNotFoundError(reference_panel)
    index = Path(str(reference_panel) + ".tbi")
    if not index.is_file():
        raise FileNotFoundError(index)
    wrong_contigs = sorted({key[0] for key in raw_keys if key[0] != contig})
    if wrong_contigs:
        raise ValueError(
            f"raw key set for {contig!r} contains other contigs: {wrong_contigs!r}"
        )
    if not raw_keys:
        return set()

    positions = sorted({key[1] for key in raw_keys})
    region_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"{contig}.",
            suffix=".regions.tsv",
            dir=str(temp_dir) if temp_dir is not None else None,
            delete=False,
        ) as handle:
            region_path = Path(handle.name)
            for position in positions:
                handle.write(f"{contig}\t{position}\t{position}\n")
        completed = subprocess.run(
            [
                bcftools,
                "view",
                "-R",
                str(region_path),
                "-H",
                str(reference_panel),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return _panel_overlap_from_lines(
            completed.stdout.splitlines(),
            raw_keys,
            source=str(reference_panel),
        )
    finally:
        if region_path is not None:
            region_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class AlleleSetSummary:
    count: int
    sha256: str


def summarize_keys(keys: Iterable[VariantKey]) -> AlleleSetSummary:
    values = set(keys)
    return AlleleSetSummary(count=len(values), sha256=digest_keys(values))


def _examples(keys: Iterable[VariantKey], limit: int = 5) -> Sequence[str]:
    ordered = sorted(
        set(keys), key=lambda value: (value[0], value[1], value[2], value[3])
    )
    return [
        f"{chrom}:{position}:{ref}:{alt}"
        for chrom, position, ref, alt in ordered[:limit]
    ]


def tabix_probe(
    vcf_path: Path,
    expected_keys: Set[VariantKey],
    *,
    tabix: str = "tabix",
) -> str:
    """Prove that the index retrieves an expected exact allele."""
    vcf_path = Path(vcf_path)
    index_path = Path(str(vcf_path) + ".tbi")
    if not vcf_path.is_file():
        raise FileNotFoundError(vcf_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not expected_keys:
        raise ValueError("tabix probe requires at least one expected exact allele")
    probe = min(
        expected_keys, key=lambda value: (value[0], value[1], value[2], value[3])
    )
    chrom, position, _, _ = probe
    completed = subprocess.run(
        [tabix, str(vcf_path), f"{chrom}:{position}-{position}"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = _panel_overlap_from_lines(
        completed.stdout.splitlines(), {probe}, source=f"tabix:{vcf_path}"
    )
    if probe not in observed:
        raise ValueError(
            "tabix index did not retrieve expected exact allele "
            f"{chrom}:{position}:{probe[2]}:{probe[3]} from {vcf_path}"
        )
    return f"{chrom}:{position}:{probe[2]}:{probe[3]}"


def file_identity(path: Path) -> Mapping[str, object]:
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }



def verify_file_identity(
    identity: object,
    *,
    label: str,
    require_sha256: bool = True,
) -> Path:
    """Verify a manifest file identity against the current artifact."""
    if not isinstance(identity, dict):
        raise ValueError(f"invalid {label} identity")
    value = identity.get("path")
    if not isinstance(value, str) or not os.path.isabs(value):
        raise ValueError(f"invalid {label} path")
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    stat = path.stat()
    if (
        identity.get("size_bytes") != stat.st_size
        or identity.get("mtime_ns") != stat.st_mtime_ns
    ):
        raise ValueError(f"{label} metadata changed: {path}")
    if require_sha256:
        expected = identity.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid {label} SHA-256")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} content changed: {path}")
    return path


def build_contig_contract(
    *,
    contig: str,
    raw_keys: Set[VariantKey],
    panel_overlap_keys: Set[VariantKey],
    reference_panel: Path,
    beagle_vcf: Optional[Path] = None,
    tabix: str = "tabix",
) -> Mapping[str, object]:
    """Build and enforce one complete per-contig marker identity contract."""
    raw_keys = set(raw_keys)
    expected = set(panel_overlap_keys)
    if not expected.issubset(raw_keys):
        raise ValueError("panel-overlap set is not a subset of raw exact alleles")

    status: str
    observed: Set[VariantKey]
    probe: Optional[str]
    output_identity: Optional[Mapping[str, object]]
    index_identity: Optional[Mapping[str, object]]
    if not raw_keys:
        if beagle_vcf is not None:
            raise ValueError(
                f"{contig}: raw_empty contig must not have a Beagle output"
            )
        status, observed, probe = "raw_empty", set(), None
        output_identity = index_identity = None
    elif not expected:
        if beagle_vcf is not None:
            raise ValueError(
                f"{contig}: panel_absent contig must not have a Beagle output"
            )
        status, observed, probe = "panel_absent", set(), None
        output_identity = index_identity = None
    else:
        if beagle_vcf is None:
            raise ValueError(
                f"{contig}: {len(expected)} reference-panel alleles require "
                "Beagle output"
            )
        beagle_vcf = Path(beagle_vcf)
        beagle_keys = read_exact_keys(
            beagle_vcf, contig=contig, split_alt=False
        )
        observed = raw_keys & beagle_keys
        missing = expected - observed
        unexpected = observed - expected
        if missing or unexpected:
            raise ValueError(
                f"{contig}: raw∩panel != raw∩Beagle; "
                f"missing={len(missing)} {_examples(missing)!r}; "
                f"unexpected={len(unexpected)} {_examples(unexpected)!r}"
            )
        probe = tabix_probe(beagle_vcf, expected, tabix=tabix)
        status = "completed"
        output_identity = file_identity(beagle_vcf)
        index_identity = file_identity(Path(str(beagle_vcf) + ".tbi"))

    return {
        "schema": MARKER_CONTRACT_SCHEMA,
        "contig": contig,
        "status": status,
        "production_inputs_include_wes": False,
        "raw_exact_alleles": asdict(summarize_keys(raw_keys)),
        "raw_panel_exact_alleles": asdict(summarize_keys(expected)),
        "raw_beagle_exact_alleles": asdict(summarize_keys(observed)),
        "exact_identity_pass": observed == expected,
        "tabix_probe": probe,
        "reference_panel": {
            "path": str(Path(reference_panel).resolve()),
            "index": str(Path(str(reference_panel) + ".tbi").resolve()),
        },
        "beagle_vcf": output_identity,
        "beagle_index": index_identity,
    }


def write_sha256sums(
    root: Path,
    artifacts: Sequence[Path],
    *,
    filename: str = "OUTPUT_SHA256SUMS.txt",
) -> Path:
    """Write checksums with paths relative to root; never overwrite."""
    root = Path(root).resolve()
    destination = root / filename
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    seen = set()
    for path in artifacts:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"artifact is outside checksum root: {resolved}"
            ) from error
        if relative.as_posix() in seen:
            raise ValueError(f"duplicate checksum artifact: {relative}")
        seen.add(relative.as_posix())
        rows.append((relative.as_posix(), sha256_file(resolved)))
    with destination.open("x") as handle:
        for relative, digest in sorted(rows):
            handle.write(f"{digest}  {relative}\n")
    return destination


def write_completion(
    root: Path,
    *,
    manifest_path: Path,
    checksum_path: Path,
    filename: str = "RUN_COMPLETE.json",
) -> Path:
    """Bind a final completion marker to the manifest and checksum ledger."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    checksum_path = Path(checksum_path).resolve()
    for path in (manifest_path, checksum_path):
        path.relative_to(root)
    destination = root / filename
    payload = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_file(manifest_path),
        "checksum_file": str(checksum_path.relative_to(root)),
        "checksum_file_sha256": sha256_file(checksum_path),
    }
    with destination.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def verify_complete_bundle(root: Path) -> Mapping[str, object]:
    """Verify completion bindings and every path in the checksum ledger."""
    root = Path(root).resolve()
    completion_path = root / "RUN_COMPLETE.json"
    try:
        completion = json.loads(completion_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid completion file: {completion_path}: {error}"
        ) from error
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
    ):
        raise ValueError(f"invalid completion contract: {completion_path}")

    def safe_member(value: object, label: str) -> Path:
        if not isinstance(value, str) or not value or os.path.isabs(value):
            raise ValueError(f"invalid {label} path in completion contract")
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} escapes bundle root: {value}") from error
        return candidate

    manifest = safe_member(completion.get("manifest"), "manifest")
    checksums = safe_member(completion.get("checksum_file"), "checksum_file")
    if sha256_file(manifest) != completion.get("manifest_sha256"):
        raise ValueError("manifest checksum does not match RUN_COMPLETE.json")
    if sha256_file(checksums) != completion.get("checksum_file_sha256"):
        raise ValueError("checksum ledger does not match RUN_COMPLETE.json")

    checked = 0
    for line_number, line in enumerate(
        checksums.read_text().splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid checksum row {checksums}:{line_number}")
        expected_digest, relative = parts
        artifact = safe_member(relative, "artifact")
        if sha256_file(artifact) != expected_digest:
            raise ValueError(f"artifact checksum mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("checksum ledger is empty")
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "verified",
        "artifact_count": checked,
        "manifest": str(manifest),
        "checksum_file": str(checksums),
    }



def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError(f"invalid {label} relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes run root: {value}") from error
    return path


def validate_v2_3_run(root: Path) -> Mapping[str, object]:
    """Validate v2.3 completion plus its semantic routing contracts."""
    root = Path(root).resolve()
    verified = verify_complete_bundle(root)
    manifest_path = Path(verified["manifest"])
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != RUN_CONTRACT_SCHEMA
        or manifest.get("production_inputs_include_wes") is not False
        or manifest.get("evaluation_uses_wes") is not False
        or manifest.get("impute_effective") is not False
        or manifest.get("anchor_score") != "p_nonref"
    ):
        raise ValueError(f"invalid v2.3 run manifest: {manifest_path}")

    threshold = manifest.get("min_anchor_probability")
    if (
        manifest.get("status") != "outputs_ready_for_checksum"
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("invalid v2.3 routing threshold/status")
    params = manifest.get("beagle_parameters")
    if not isinstance(params, dict):
        raise ValueError("v2.3 run manifest lacks Beagle parameters")
    iterations = params.get("ITERATIONS")
    seed = params.get("SEED")
    modelscale = params.get("MODEL_SCALE")
    maxlr = params.get("MAX_LR")
    if (
        params.get("IMPUTE") is not False
        or params.get("GPROBS") is not True
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(modelscale, bool)
        or not isinstance(modelscale, (int, float))
        or float(modelscale) <= 0
        or isinstance(maxlr, bool)
        or not isinstance(maxlr, (int, float))
        or float(maxlr) <= 0
    ):
        raise ValueError("invalid v2.3 Beagle parameter contract")

    verify_file_identity(manifest.get("raw_vcf"), label="raw VCF")
    verify_file_identity(manifest.get("raw_vcf_index"), label="raw VCF index")
    code = manifest.get("code")
    if not isinstance(code, list) or len(code) != 4:
        raise ValueError("v2.3 manifest must bind exactly four source files")
    code_paths = []
    for index, identity in enumerate(code):
        path = verify_file_identity(identity, label=f"source file {index}")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("v2.3 source file is outside immutable run root") from error
        code_paths.append(path)
    if len(set(code_paths)) != len(code_paths):
        raise ValueError("duplicate v2.3 source identity")
    executables = manifest.get("executables")
    if not isinstance(executables, dict) or set(executables) != {
        "beagle_jar", "java", "bgzip", "tabix"
    }:
        raise ValueError("invalid v2.3 executable identity table")
    for name, identity in executables.items():
        verify_file_identity(identity, label=name)

    reference = manifest.get("reference_ledger")
    if not isinstance(reference, dict):
        raise ValueError("v2.3 run manifest lacks reference ledger")
    reference_root = Path(str(reference.get("root", ""))).resolve()
    reference_sums = verify_reference_ledger(reference_root)
    requested_ledger = params.get("REFERENCE_LEDGER")
    if (
        not isinstance(requested_ledger, str)
        or Path(requested_ledger).resolve() != reference_root
    ):
        raise ValueError("Beagle parameter reference ledger differs from manifest")
    for name in ("manifest", "reference_sha256sums", "output_sha256sums"):
        path = verify_file_identity(
            reference.get(name), label=f"reference ledger {name}"
        )
        try:
            path.relative_to(reference_root)
        except ValueError as error:
            raise ValueError("reference-ledger identity escapes its root") from error

    status_path = _safe_relative(
        root, manifest.get("contig_status_file"), "contig status"
    )
    if sha256_file(status_path) != manifest.get(
        "contig_status_file_sha256"
    ):
        raise ValueError("contig status digest differs from run manifest")
    status = json.loads(status_path.read_text())
    if (
        status.get("schema") != "sparcal_beagle_contig_status_v2_3"
        or status.get("production_inputs_include_wes") is not False
    ):
        raise ValueError("invalid v2.3 contig status")
    expected = status.get("expected_contigs")
    rows = status.get("contigs")
    if (
        not isinstance(expected, list)
        or not isinstance(rows, list)
        or len(expected) != len(set(expected))
    ):
        raise ValueError("invalid v2.3 expected-contig table")
    by_contig = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid v2.3 contig row")
        contig = row.get("chromosome")
        if contig in by_contig:
            raise ValueError(f"duplicate v2.3 contig row: {contig}")
        by_contig[contig] = row
    if set(by_contig) != set(expected):
        raise ValueError("v2.3 contig rows do not match expected contigs")

    panel_rows = manifest.get("reference_panels")
    if not isinstance(panel_rows, list) or len(panel_rows) != len(expected):
        raise ValueError("reference-panel provenance does not match contigs")
    panel_by_contig = {}
    for panel_row in panel_rows:
        if not isinstance(panel_row, dict):
            raise ValueError("invalid reference-panel provenance row")
        contig = panel_row.get("contig")
        if contig in panel_by_contig:
            raise ValueError(f"duplicate reference-panel contig: {contig}")
        panel_by_contig[contig] = panel_row
    if set(panel_by_contig) != set(expected):
        raise ValueError("reference-panel provenance contigs are incomplete")
    for contig in expected:
        panel_row = panel_by_contig[contig]
        for label, field in (("panel", "panel"), ("panel index", "panel_index")):
            identity = panel_row.get(field)
            path = verify_file_identity(
                identity,
                label=f"{contig} reference {label}",
                require_sha256=False,
            )
            logical_path = identity.get("path") if isinstance(identity, dict) else None
            expected_digest = reference_sums.get(logical_path)
            if (
                expected_digest is None
                or not isinstance(identity, dict)
                or identity.get("sha256") != expected_digest
            ):
                raise ValueError(
                    f"{contig} reference {label} differs from frozen ledger"
                )

    raw_total = 0
    completed = []
    for contig in expected:
        row = by_contig[contig]
        marker = row.get("marker_contract")
        if (
            not isinstance(marker, dict)
            or marker.get("schema") != MARKER_CONTRACT_SCHEMA
            or marker.get("contig") != contig
            or marker.get("production_inputs_include_wes") is not False
            or marker.get("exact_identity_pass") is not True
            or marker.get("status") != row.get("status")
        ):
            raise ValueError(f"{contig}: invalid v2.3 marker contract")
        raw = marker.get("raw_exact_alleles")
        panel = marker.get("raw_panel_exact_alleles")
        beagle = marker.get("raw_beagle_exact_alleles")
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("count"), int)
            and isinstance(item.get("sha256"), str)
            for item in (raw, panel, beagle)
        ):
            raise ValueError(f"{contig}: invalid allele-set summaries")
        raw_total += raw["count"]
        if (
            panel["count"] != beagle["count"]
            or panel["sha256"] != beagle["sha256"]
        ):
            raise ValueError(f"{contig}: marker identity digest/count mismatch")
        state = row.get("status")
        if state == "completed":
            if panel["count"] <= 0 or not marker.get("tabix_probe"):
                raise ValueError(f"{contig}: incomplete completed marker contract")
            completed.append(contig)
        elif state == "panel_absent":
            if raw["count"] <= 0 or panel["count"] != 0:
                raise ValueError(f"{contig}: invalid panel_absent contract")
        elif state == "raw_empty":
            if raw["count"] != 0 or panel["count"] != 0:
                raise ValueError(f"{contig}: invalid raw_empty contract")
        else:
            raise ValueError(f"{contig}: invalid v2.3 state {state!r}")
    if raw_total != status.get("raw_record_count"):
        raise ValueError("v2.3 raw total does not match contig status")

    quality = manifest.get("quality_filter")
    if not isinstance(quality, str) or not quality:
        raise ValueError("v2.3 manifest has invalid quality filter")
    router_path = (
        root / "output_VCFs" / "beagle" / quality / "MANIFEST.json"
    )
    router = json.loads(router_path.read_text())
    if (
        router.get("schema") != "sparcal_beagle_evidence_v2_2"
        or router.get("production_inputs_include_wes") is not False
        or router.get("raw_record_count") != raw_total
        or router.get("anchor_score") != "p_nonref"
        or router.get("min_anchor_probability") != threshold
        or router.get("hard_anchor_route") != "beagle_nonref_highconf"
        or "all_panel_present.vcf.gz"
        not in router.get("compatibility", {})
    ):
        raise ValueError("invalid v2.3 biological router manifest")
    status_digest = router.get(
        "upstream_contig_status_sha256",
        router.get("contig_status_file_sha256"),
    )
    if status_digest != sha256_file(status_path):
        raise ValueError("router is not bound to the v2.3 contig status")
    counts = router.get("counts")
    route_names = (
        "beagle_nonref_highconf",
        "beagle_ref_highconf",
        "beagle_uncertain",
        "panel_absent",
    )
    if (
        not isinstance(counts, dict)
        or counts.get("total") != raw_total
        or sum(int(counts.get(name, -1)) for name in route_names)
        != raw_total
    ):
        raise ValueError("v2.3 router counts do not conserve raw alleles")
    if sorted(manifest.get("completed_beagle_contigs", [])) != sorted(
        completed
    ):
        raise ValueError("run manifest completed-contig list is inconsistent")
    return {
        "schema": RUN_CONTRACT_SCHEMA,
        "status": "verified",
        "raw_record_count": raw_total,
        "expected_contig_count": len(expected),
        "completed_contig_count": len(completed),
        "router_manifest": str(router_path),
    }

def _parse_sha256sum_file(path: Path) -> Mapping[str, str]:
    values = {}
    for line_number, line in enumerate(
        Path(path).read_text().splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1]:
            raise ValueError(f"invalid SHA-256 row {path}:{line_number}")
        digest, filename = parts
        if filename in values:
            raise ValueError(f"duplicate SHA-256 path in {path}: {filename}")
        values[filename] = digest
    return values


def verify_reference_ledger(
    root: Path, *, require_bound_completion: bool = True
) -> Mapping[str, str]:
    """Verify the frozen full-panel ledger and current file identities."""
    root = Path(root).resolve()
    if (root / "RUN_COMPLETE").read_text() != "complete\n":
        raise ValueError(f"reference ledger is not complete: {root}")
    if (root / "RUN_COMPLETE.json").is_file():
        verify_complete_bundle(root)
    elif require_bound_completion:
        raise ValueError(
            f"reference ledger lacks bound RUN_COMPLETE.json: {root}"
        )
    manifest_values = {}
    for line_number, line in enumerate(
        (root / "MANIFEST.txt").read_text().splitlines(), start=1
    ):
        if "=" not in line:
            raise ValueError(
                f"invalid reference manifest row {line_number}: {line!r}"
            )
        key, value = line.split("=", 1)
        manifest_values[key] = value
    if (
        manifest_values.get("schema")
        != "sparcal_beagle_reference_ledger_v1"
        or manifest_values.get("production_inputs_include_wes") != "false"
        or manifest_values.get("artifact_count") != "88"
    ):
        raise ValueError(f"invalid reference ledger manifest: {root}")

    output_sums = _parse_sha256sum_file(
        root / "OUTPUT_SHA256SUMS.txt"
    )
    for filename, expected in output_sums.items():
        output_path = Path(filename)
        if not output_path.is_absolute():
            output_path = root / output_path
        if sha256_file(output_path) != expected:
            raise ValueError(
                f"reference-ledger output checksum mismatch: {output_path}"
            )

    reference_sums = _parse_sha256sum_file(
        root / "REFERENCE_SHA256SUMS.txt"
    )
    expected_paths = [
        line
        for line in (root / "REFERENCE_PATHS.txt").read_text().splitlines()
        if line
    ]
    if (
        len(expected_paths) != 88
        or len(set(expected_paths)) != 88
        or set(expected_paths) != set(reference_sums)
    ):
        raise ValueError("reference path/checksum ledger is not a complete 88-file set")

    metadata_rows = {}
    metadata_lines = (
        root / "FILE_METADATA.tsv"
    ).read_text().splitlines()
    if not metadata_lines or metadata_lines[0] != (
        "path\tsize_bytes\tmtime_epoch"
    ):
        raise ValueError("invalid reference metadata header")
    for line_number, line in enumerate(metadata_lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"invalid reference metadata row {line_number}"
            )
        path, size_text, mtime_text = fields
        if path in metadata_rows:
            raise ValueError(f"duplicate reference metadata path: {path}")
        metadata_rows[path] = (int(size_text), int(mtime_text))
    if set(metadata_rows) != set(expected_paths):
        raise ValueError("reference metadata/path ledger mismatch")

    for filename in expected_paths:
        path = Path(filename)
        stat = path.stat()
        expected_size, expected_mtime = metadata_rows[filename]
        if stat.st_size != expected_size or int(stat.st_mtime) != expected_mtime:
            raise ValueError(
                f"reference artifact changed since hashing: {filename}"
            )
    return reference_sums
