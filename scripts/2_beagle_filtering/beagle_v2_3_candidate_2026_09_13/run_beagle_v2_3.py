#!/usr/bin/env python3
"""Run the isolated, fail-closed SPARCAL Beagle v2.3 candidate.

This runner reuses the current biological routing implementation while adding
complete raw/panel/Beagle marker identity, real index queries, and a
cryptographically bound run bundle. It writes only to a fresh section root and
never uses WES as an inference input.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(PARENT / "run_beagle.py", "_sparcal_beagle_v2_3_base")
CONTRACTS = _load(HERE / "contracts.py", "_sparcal_beagle_v2_3_contracts")
RUN_SCHEMA = CONTRACTS.RUN_CONTRACT_SCHEMA


def _metadata_identity(path: Path) -> Mapping[str, object]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class BeagleV23Pipeline(BASE.BeaglePipeline):
    """The v2.2 computation plus the stricter v2.3 evidence contract."""

    def __init__(self, *args, **kwargs):
        self._raw_keys_by_contig = {}
        self._panel_keys_by_contig = {}
        self._requested_params = {}
        self._reference_sums = {}
        self._reference_ledger_root = None
        super().__init__(*args, **kwargs)

    def chromosome_has_reference_marker(self, input_vcf, chromosome):
        raw_keys = CONTRACTS.read_exact_keys(
            Path(input_vcf), contig=chromosome, split_alt=False
        )
        reference_panel = Path(self.get_1000genome_reference(chromosome))
        panel_keys = CONTRACTS.query_reference_overlap(
            raw_keys,
            reference_panel,
            contig=chromosome,
            bcftools="bcftools",
            temp_dir=Path(self.output_dirs["log_dir"]),
        )
        self._raw_keys_by_contig[chromosome] = raw_keys
        self._panel_keys_by_contig[chromosome] = panel_keys
        return bool(panel_keys), len(raw_keys)

    def run_beagle_command(
        self,
        input_vcf: str,
        output_prefix: str,
        chromosome: str,
        params: Dict,
        log_file: str,
    ):
        """Run GL refinement with every influential parameter explicit."""
        if params["IMPUTE"]:
            raise ValueError(
                "Beagle 4.1 impute has no effect for the gl+ref first pass; "
                "the v2.3 runner refuses this misleading setting"
            )
        reference_panel = self.get_1000genome_reference(chromosome)
        beagle_output = f"{output_prefix}.temp.vcf.gz"
        java = os.path.join(BASE.PATH_CONFIG["APPS_DIR"], BASE.PATH_CONFIG["JAVA"])
        jar = os.path.join(
            BASE.PATH_CONFIG["APPS_DIR"], BASE.PATH_CONFIG["BEAGLE_JAR"]
        )
        command = [
            java,
            f"-Xmx{params['MEMORY']}",
            "-jar",
            jar,
            f"gl={input_vcf}",
            f"ref={reference_panel}",
            f"chrom={chromosome}",
            f"out={output_prefix}.temp",
            "impute=false",
            f"modelscale={params['MODEL_SCALE']}",
            f"maxlr={params['MAX_LR']}",
            f"nthreads={params['THREADS']}",
            f"gprobs={'true' if params['GPROBS'] else 'false'}",
            f"niterations={params['ITERATIONS']}",
        ]
        if params.get("SEED") is not None:
            command.append(f"seed={params['SEED']}")
        if params.get("GENETIC_MAP"):
            map_path = str(params["GENETIC_MAP"]).format(
                chrom=chromosome,
                chrom_number=chromosome.removeprefix("chr"),
            )
            if not Path(map_path).is_file():
                raise FileNotFoundError(f"genetic map not found: {map_path}")
            command.append(f"map={map_path}")

        with open(log_file, "x") as log:
            log.write(f"Command: {' '.join(command)}\n\n")
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        final_output = f"{output_prefix}.vcf.gz"
        self.merge_vcf_fields(input_vcf, beagle_output, final_output)
        os.remove(beagle_output)
        Path(f"{beagle_output}.tbi").unlink(missing_ok=True)

    def process_chromosome(self, chromosome: str, params: Dict) -> Dict:
        result = super().process_chromosome(chromosome, params)
        if result.get("status") == "failed":
            return result
        raw_keys = self._raw_keys_by_contig[chromosome]
        panel_keys = self._panel_keys_by_contig[chromosome]
        output = (
            Path(result["output_file"])
            if result.get("output_file") is not None
            else None
        )
        contract = CONTRACTS.build_contig_contract(
            contig=chromosome,
            raw_keys=raw_keys,
            panel_overlap_keys=panel_keys,
            reference_panel=Path(
                self.get_1000genome_reference(chromosome)
            ),
            beagle_vcf=output,
            tabix=BASE.PATH_CONFIG["TABIX"],
        )
        if contract["status"] != result["status"]:
            raise ValueError(
                f"{chromosome}: execution status {result['status']!r} "
                f"does not match marker contract {contract['status']!r}"
            )
        result["raw_panel_exact_allele_count"] = contract[
            "raw_panel_exact_alleles"
        ]["count"]
        result["raw_panel_exact_allele_sha256"] = contract[
            "raw_panel_exact_alleles"
        ]["sha256"]
        result["raw_beagle_exact_allele_count"] = contract[
            "raw_beagle_exact_alleles"
        ]["count"]
        result["raw_beagle_exact_allele_sha256"] = contract[
            "raw_beagle_exact_alleles"
        ]["sha256"]
        result["marker_contract"] = contract
        return result

    def _write_contig_status_provenance(
        self,
        results: Sequence[Mapping],
        raw_contract,
    ) -> Path:
        path = (
            Path(self.output_dirs["output_vcf_dir"])
            / "BEAGLE_CONTIG_STATUS.json"
        )
        for result in results:
            contract = result.get("marker_contract")
            if (
                not isinstance(contract, dict)
                or contract.get("schema") != CONTRACTS.MARKER_CONTRACT_SCHEMA
                or contract.get("exact_identity_pass") is not True
            ):
                raise ValueError(
                    f"{result.get('chromosome')}: missing v2.3 marker contract"
                )
        input_vcf = (
            Path(self.output_dirs["input_vcf_dir"])
            / "merged_sorted_gt.vcf.gz"
        )
        payload = {
            "schema": "sparcal_beagle_contig_status_v2_3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset_name,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "production_inputs_include_wes": False,
            "raw_vcf": CONTRACTS.file_identity(input_vcf.resolve()),
            "raw_vcf_index": CONTRACTS.file_identity(
                Path(str(input_vcf.resolve()) + ".tbi")
            ),
            "raw_sample_id": raw_contract.sample_id,
            "raw_record_count": raw_contract.record_count,
            "expected_contigs": list(self.regions),
            "status_counts": {
                status: sum(
                    result["status"] == status for result in results
                )
                for status in ("completed", "panel_absent", "raw_empty")
            },
            "contigs": [dict(result) for result in results],
        }
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def _reference_provenance(self) -> Sequence[Mapping[str, object]]:
        rows = []
        for contig in self.regions:
            panel = Path(self.get_1000genome_reference(contig))
            index = Path(str(panel) + ".tbi")
            panel_key = str(panel)
            index_key = str(index)
            if (
                panel_key not in self._reference_sums
                or index_key not in self._reference_sums
            ):
                raise ValueError(
                    f"{contig}: reference panel is absent from frozen ledger"
                )
            panel_identity = dict(_metadata_identity(panel))
            panel_identity["sha256"] = self._reference_sums[panel_key]
            index_identity = dict(_metadata_identity(index))
            index_identity["sha256"] = self._reference_sums[index_key]
            rows.append(
                {
                    "contig": contig,
                    "panel": panel_identity,
                    "panel_index": index_identity,
                    "raw_panel_exact_alleles": {
                        "count": len(self._panel_keys_by_contig[contig]),
                        "sha256": CONTRACTS.digest_keys(
                            self._panel_keys_by_contig[contig]
                        ),
                    },
                }
            )
        return rows

    def _write_final_bundle(
        self,
        *,
        params: Mapping[str, object],
        metrics_file: Path,
        contig_status_path: Path,
        routing_mode: str,
        completed: Sequence[Mapping],
    ) -> Path:
        section_root = Path(self.output_base).resolve()
        manifest_path = section_root / "V2_3_MANIFEST.json"
        raw_vcf = (
            Path(self.output_dirs["input_vcf_dir"])
            / "merged_sorted_gt.vcf.gz"
        ).resolve()
        raw_index = Path(str(raw_vcf) + ".tbi")
        source_files = (
            Path(__file__).resolve(),
            (HERE / "contracts.py").resolve(),
            (PARENT / "run_beagle.py").resolve(),
            (PARENT / "beagle_evidence_router.py").resolve(),
        )
        manifest = {
            "schema": RUN_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "outputs_ready_for_checksum",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "dataset": self.dataset_name,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "anchor_score": self.anchor_score,
            "min_anchor_probability": self.min_anchor_probability,
            "beagle_parameters": dict(params),
            "impute_effective": False,
            "routing_mode": routing_mode,
            "completed_beagle_contigs": [
                result["chromosome"] for result in completed
            ],
            "raw_vcf": CONTRACTS.file_identity(raw_vcf),
            "raw_vcf_index": CONTRACTS.file_identity(raw_index),
            "contig_status_file": str(
                contig_status_path.resolve().relative_to(section_root)
            ),
            "contig_status_file_sha256": CONTRACTS.sha256_file(
                contig_status_path
            ),
            "metrics_file": str(
                metrics_file.resolve().relative_to(section_root)
            ),
            "reference_panels": self._reference_provenance(),
            "code": [
                CONTRACTS.file_identity(path) for path in source_files
            ],
            "executables": {
                "beagle_jar": CONTRACTS.file_identity(
                    Path(BASE.PATH_CONFIG["APPS_DIR"])
                    / BASE.PATH_CONFIG["BEAGLE_JAR"]
                ),
                "java": CONTRACTS.file_identity(
                    Path(BASE.PATH_CONFIG["APPS_DIR"])
                    / BASE.PATH_CONFIG["JAVA"]
                ),
                "bgzip": CONTRACTS.file_identity(
                    Path(BASE.PATH_CONFIG["BGZIP"])
                ),
                "tabix": CONTRACTS.file_identity(
                    Path(BASE.PATH_CONFIG["TABIX"])
                ),
            },
            "reference_ledger": {
                "root": str(self._reference_ledger_root),
                "manifest": CONTRACTS.file_identity(
                    self._reference_ledger_root / "MANIFEST.txt"
                ),
                "reference_sha256sums": CONTRACTS.file_identity(
                    self._reference_ledger_root
                    / "REFERENCE_SHA256SUMS.txt"
                ),
                "output_sha256sums": CONTRACTS.file_identity(
                    self._reference_ledger_root
                    / "OUTPUT_SHA256SUMS.txt"
                ),
            },
        }
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        artifacts = [manifest_path]
        for key in ("output_vcf_dir", "log_dir", "metrics_dir"):
            root = Path(self.output_dirs[key])
            artifacts.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        checksum_path = CONTRACTS.write_sha256sums(
            section_root, artifacts
        )
        return CONTRACTS.write_completion(
            section_root,
            manifest_path=manifest_path,
            checksum_path=checksum_path,
        )

    def run_pipeline(
        self, custom_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        params = {
            **BASE.DEFAULT_PARAMS,
            "MAX_LR": 5000,
            "GENETIC_MAP": None,
            "REFERENCE_LEDGER": None,
        }
        if custom_params:
            params.update(custom_params)
        self._requested_params = dict(params)
        if not params.get("REFERENCE_LEDGER"):
            raise ValueError(
                "v2.3 requires --reference-ledger-root so every panel "
                "artifact has a frozen full-content SHA-256"
            )
        self._reference_ledger_root = Path(
            params["REFERENCE_LEDGER"]
        ).resolve()
        params["REFERENCE_LEDGER"] = str(self._reference_ledger_root)
        self._reference_sums = CONTRACTS.verify_reference_ledger(
            self._reference_ledger_root
        )
        input_vcf = (
            Path(self.output_dirs["input_vcf_dir"])
            / "merged_sorted_gt.vcf.gz"
        )
        if not input_vcf.is_file():
            raise FileNotFoundError(f"input VCF not found: {input_vcf}")
        if not Path(str(input_vcf) + ".tbi").is_file():
            raise FileNotFoundError(f"input VCF index not found: {input_vcf}.tbi")
        self._claim_fresh_output_tree(params)

        results = []
        with ThreadPoolExecutor(
            max_workers=min(
                params["PARALLEL_CHROMOSOMES"], len(self.regions)
            )
        ) as executor:
            futures = {
                executor.submit(
                    self.process_chromosome, chromosome, params
                ): chromosome
                for chromosome in self.regions
            }
            with tqdm(
                total=len(self.regions), desc="Processing chromosomes"
            ) as progress:
                for future in as_completed(futures):
                    chromosome = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(
                            {
                                "chromosome": chromosome,
                                "status": "failed",
                                "error": str(error),
                            }
                        )
                    progress.update(1)

        metrics = pd.DataFrame(results)
        name = f"{self.dataset_name}"
        if self.section_id:
            name += f"_{self.section_id}"
        metrics_file = (
            Path(self.output_dirs["metrics_dir"])
            / f"{name}_beagle_metrics.csv"
        )
        metrics.to_csv(metrics_file, index=False, mode="x")
        valid = {"completed", "panel_absent", "raw_empty"}
        failed = int((~metrics["status"].isin(valid)).sum())
        if failed:
            raise RuntimeError(
                f"Beagle failed on {failed} chromosome(s); partial outputs "
                "were retained for audit and this run cannot resume in place"
            )

        validated, raw_contract = self._validate_contig_results(
            results, input_vcf
        )
        status_path = self._write_contig_status_provenance(
            validated, raw_contract
        )
        completed = [
            result
            for result in validated
            if result["status"] == "completed"
        ]
        router = BASE._load_beagle_router()
        if completed:
            routing_counts = router.route_vcf(
                input_vcf,
                [Path(result["output_file"]) for result in completed],
                Path(self.output_dirs["output_vcf_dir"]),
                min_anchor_probability=self.min_anchor_probability,
                anchor_score=self.anchor_score,
                bgzip=BASE.PATH_CONFIG["BGZIP"],
                tabix=BASE.PATH_CONFIG["TABIX"],
                allow_existing_output_dir=True,
                expected_contigs=self.regions,
                contig_status_path=status_path,
            )
            routing_mode = "beagle_with_v2_3_marker_contract"
        else:
            routing_counts = self._route_without_beagle(
                input_vcf,
                Path(self.output_dirs["output_vcf_dir"]),
                raw_contract,
                status_path,
            )
            routing_mode = "all_panel_absent_v2_3_contract"

        completion = self._write_final_bundle(
            params=params,
            metrics_file=metrics_file,
            contig_status_path=status_path,
            routing_mode=routing_mode,
            completed=completed,
        )
        CONTRACTS.validate_v2_3_run(Path(self.output_base))
        print(
            "Routing counts: "
            + ", ".join(
                f"{status}={routing_counts.get(status, 0):,}"
                for status in router.ROUTES
            )
        )
        print(f"Verified v2.3 completion: {completion}")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(BASE.DATASET_CONFIGS),
    )
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--section-root", required=True)
    parser.add_argument("--reference-ledger-root", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--memory", default="10g")
    parser.add_argument("--parallel-chromosomes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modelscale", type=float, default=2.0)
    parser.add_argument("--maxlr", type=float, default=5000)
    parser.add_argument(
        "--genetic-map",
        default=None,
        help=(
            "Optional per-contig template; placeholders chrom and "
            "chrom_number are supported"
        ),
    )
    parser.add_argument(
        "--anchor-score",
        choices=("p_nonref", "called_gp"),
        default="p_nonref",
    )
    parser.add_argument(
        "--min-anchor-probability", type=float, default=0.90
    )
    args = parser.parse_args()
    pipeline = BeagleV23Pipeline(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        section_root=args.section_root,
        anchor_score=args.anchor_score,
        min_anchor_probability=args.min_anchor_probability,
    )
    pipeline.run_pipeline(
        {
            "THREADS": args.threads,
            "MEMORY": args.memory,
            "PARALLEL_CHROMOSOMES": args.parallel_chromosomes,
            "ITERATIONS": args.iterations,
            "SEED": args.seed,
            "MODEL_SCALE": args.modelscale,
            "MAX_LR": args.maxlr,
            "GENETIC_MAP": args.genetic_map,
            "REFERENCE_LEDGER": args.reference_ledger_root,
            "IMPUTE": False,
        }
    )


if __name__ == "__main__":
    main()
