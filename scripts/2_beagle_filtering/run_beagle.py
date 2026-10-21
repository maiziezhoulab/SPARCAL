import importlib.util
import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Mapping, Optional, Sequence, Tuple
from tqdm import tqdm
import gzip
import argparse
import tempfile
import json
from datetime import datetime, timezone

# Configuration dictionaries from mpileup_pipeline.py
REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    },
    "TUMOR": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    }
}

# DLPFC_PREDEDUP=1 -> write to a SEPARATE tree (data/dlpfc_prededup) so the
# post-dedup results under data/dlpfc are left untouched. Default: unchanged.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc_prededup/{section_id}" if _PREDEDUP else "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC"
    },
    "OVAR_P5": {
        # GRCh38, chr prefix. Beagle: genome_build=GRCh38 (reference != TUMOR),
        # regions chr1..chr22 (DLPFC entry) — matches the chr-prefixed mpileup VCFs.
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "DLPFC"
    },
    "NCCE": {
        # GRCh38, chr-prefixed SpaceRanger BAM/VCF; DLPFC is used here only
        # for Beagle's chr1..chr22 region list and GRCh38 1KGP panel selection.
        "base_path": "/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs",
        "output_dir": "data/ncce/{section_id}",
        "has_sections": True,
        "section_ids": ["6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C", "11A", "11B", "11C", "14A", "14B", "14C"],
        "reference": "DLPFC"
    }
}

THOUSAND_GENOME_CONFIGS = {
    "GRCh38": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "pattern": "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    # example of 1kG for chrom 1:
    # /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/CCDG_14151_B01_GRM_WGS_2020-08-05_1.filtered.shapeit2-duohmm-phased.vcf.gz
    "hg19": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19/",
        "pattern": "hg19_chr{chrom}.vcf.gz"
    }
}

# Default Parameters
DEFAULT_PARAMS = {
    "THREADS": 24,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 0,
    "IMPUTE": False,
    "GPROBS": True,
    # None preserves Beagle's historical default. Paper ablations should pass
    # an explicit seed so repeated runs are exactly attributable.
    "SEED": None,
    # Each Beagle process is configured to use the full thread and memory
    # allocation.  Launching one process per chromosome would therefore
    # oversubscribe a SLURM job by up to 22-fold.
    "PARALLEL_CHROMOSOMES": 1
}

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "BEAGLE_JAR": "beagle.27Jul16.86a.jar",
    "JAVA": "src/jdk-11.0.2/bin/java",
    "THOUSAND_GENOME_DIR": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix"
}


_BEAGLE_ROUTER = None


def _load_beagle_router():
    """Load the colocated authoritative router, including from code snapshots."""
    global _BEAGLE_ROUTER
    if _BEAGLE_ROUTER is not None:
        return _BEAGLE_ROUTER
    router_path = Path(__file__).resolve().with_name("beagle_evidence_router.py")
    if not router_path.is_file():
        raise FileNotFoundError(
            f"authoritative Beagle evidence router not found beside runner: {router_path}"
        )
    module_name = "_sparcal_beagle_evidence_router"
    spec = importlib.util.spec_from_file_location(module_name, router_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Beagle evidence router: {router_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _BEAGLE_ROUTER = module
    return module


class BeaglePipeline:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0",
                 section_id: str = None, section_root: str = None,
                 min_anchor_gp: Optional[float] = None,
                 anchor_score: str = "p_nonref",
                 min_anchor_probability: float = 0.90):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.section_root = os.path.abspath(section_root) if section_root else None
        if min_anchor_gp is not None:
            anchor_score = "called_gp"
            min_anchor_probability = min_anchor_gp
        if anchor_score not in {"p_nonref", "called_gp"}:
            raise ValueError(f"unknown anchor_score: {anchor_score}")
        if not 0.0 <= min_anchor_probability <= 1.0:
            raise ValueError("min_anchor_probability must be between 0 and 1")
        self.anchor_score = anchor_score
        self.min_anchor_probability = min_anchor_probability
        self.validate_dataset_config()
        self.setup_paths()
        self.setup_environment()
        
    def validate_dataset_config(self):
        """Validate dataset configuration and section ID if required."""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        if dataset_config["has_sections"]:
            if not self.section_id:
                raise ValueError(f"Dataset {self.dataset_name} requires a section_id")
            if "section_ids" in dataset_config:
                if self.section_id not in dataset_config["section_ids"]:
                    raise ValueError(f"Invalid section_id for {self.dataset_name}. "
                                  f"Valid section IDs are: {dataset_config['section_ids']}")
    
    def setup_paths(self):
        """Setup paths based on dataset configuration."""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        reference_config = REFERENCE_CONFIGS[dataset_config["reference"]]
        
        # Set chromosome format and regions
        self.chr_prefix = reference_config["chr_prefix"]
        self.regions = reference_config["regions"]
        
        # Set output directory
        if self.section_root:
            self.output_base = self.section_root
        elif dataset_config["has_sections"]:
            self.output_base = os.path.join(
                PATH_CONFIG["PROJECT_DIR"],
                dataset_config["output_dir"].format(section_id=self.section_id)
            )
        else:
            self.output_base = os.path.join(
                PATH_CONFIG["PROJECT_DIR"],
                dataset_config["output_dir"]
            )
        
        # Setup output directories with quality filter
        self.output_dirs = {
            "input_vcf_dir": os.path.join(self.output_base, "output_VCFs/mpileup_multi_bam", 
                                        self.quality_filter),
            "output_vcf_dir": os.path.join(self.output_base, "output_VCFs/beagle", 
                                         self.quality_filter),
            "log_dir": os.path.join(self.output_base, "logs/beagle", 
                                  self.quality_filter),
            "metrics_dir": os.path.join(self.output_base, "metrics/beagle", 
                                     self.quality_filter)
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _claim_fresh_output_tree(self, params: Dict) -> Path:
        """Atomically claim an empty run tree before any per-contig write."""
        monitored = [
            Path(self.output_dirs["output_vcf_dir"]),
            Path(self.output_dirs["log_dir"]),
            Path(self.output_dirs["metrics_dir"]),
        ]
        existing = sorted(
            path for root in monitored for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        if existing:
            examples = ", ".join(str(path) for path in existing[:5])
            raise FileExistsError(
                f"refusing to reuse a non-empty Beagle run tree; found "
                f"{len(existing)} artifact(s), including: {examples}"
            )
        marker = Path(self.output_dirs["output_vcf_dir"]) / "RUN_STARTED.json"
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset_name,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "anchor_score": self.anchor_score,
            "min_anchor_probability": self.min_anchor_probability,
            "beagle_parameters": params,
        }
        with marker.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return marker

    def setup_environment(self):
        """Setup environment variables for the pipeline."""
        apps_dir = PATH_CONFIG['APPS_DIR']
        os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

    def get_1000genome_reference(self, chromosome: str) -> str:
        """Get the appropriate 1000 Genome reference file for a chromosome."""
        # Determine genome build based on dataset
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        reference_name = dataset_config['reference']
        
        # Select appropriate 1000 Genomes configuration
        genome_build = "hg19" if reference_name == "TUMOR" else "GRCh38"
        genome_config = THOUSAND_GENOME_CONFIGS[genome_build]
        
        # Format chromosome name correctly
        if genome_build == "hg19":
            # For hg19, remove 'chr' prefix as 1000G files use just numbers
            chrom = chromosome.replace('chr', '')
        else:
            # For GRCh38, keep chromosome format as is
            chrom = chromosome
            
        reference_pattern = os.path.join(
            genome_config["base_path"],
            genome_config["pattern"].format(chrom=chrom)
        )
        
        if not os.path.exists(reference_pattern):
            raise FileNotFoundError(
                f"1000 Genome reference not found: {reference_pattern}\n"
                f"Genome build: {genome_build}, Chromosome: {chromosome}"
            )
        
        return reference_pattern

    def merge_vcf_fields(self, original_vcf: str, beagle_vcf: str, output_vcf: str):
        """Merge the FORMAT and INFO fields from original VCF with Beagle's output."""
        # Ensure both VCFs are indexed
        for vcf in [original_vcf, beagle_vcf]:
            if not os.path.exists(vcf + '.tbi'):
                subprocess.run(['tabix', '-p', 'vcf', vcf], check=True)
        
        # Prepare bcftools command
        cmd = [
            'bcftools', 'annotate',
            '-a', original_vcf,
            '-c', 'INFO/DP,INFO/I16,INFO/QS,INFO/SGB,INFO/RPB,INFO/MQB,INFO/MQSB,INFO/BQB,INFO/MQ0F,FORMAT/GQ,FORMAT/BAF,FORMAT/PL',
            '-O', 'z',
            '-o', output_vcf,
            beagle_vcf
        ]
        
        subprocess.run(cmd, check=True)
        subprocess.run(['tabix', '-p', 'vcf', output_vcf], check=True)

    def run_beagle_command(self, input_vcf: str, output_prefix: str, 
                          chromosome: str, params: Dict, log_file: str):
        """Run Beagle with specified parameters."""
        reference_panel = self.get_1000genome_reference(chromosome)
        beagle_output = f"{output_prefix}.temp.vcf.gz"
        
        cmd = [
            os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["JAVA"]),
            f"-Xmx{params['MEMORY']}",
            "-jar",
            os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["BEAGLE_JAR"]),
            f"gl={input_vcf}",
            f"ref={reference_panel}",
            f"chrom={chromosome}",
            f"out={output_prefix}.temp",
            f"impute={'true' if params['IMPUTE'] else 'false'}",
            f"modelscale={params['MODEL_SCALE']}",
            f"nthreads={params['THREADS']}",
            f"gprobs={'true' if params['GPROBS'] else 'false'}",
            f"niterations={params['ITERATIONS']}"
        ]
        if params.get("SEED") is not None:
            cmd.append(f"seed={params['SEED']}")
        
        with open(log_file, 'w') as log:
            log.write(f"Command: {' '.join(cmd)}\n\n")
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        
        # Merge original fields with Beagle output
        final_output = f"{output_prefix}.vcf.gz"
        self.merge_vcf_fields(input_vcf, beagle_output, final_output)
        
        # Cleanup temporary file
        os.remove(beagle_output)
        if os.path.exists(f"{beagle_output}.tbi"):
            os.remove(f"{beagle_output}.tbi")

    def chromosome_has_reference_marker(self, input_vcf: str, chromosome: str) -> Tuple[bool, int]:
        """Return whether a contig has a target marker shared with 1000 Genomes.

        Beagle 4.1 exits unsuccessfully without producing a VCF if all raw calls
        on a contig are absent from the reference panel.  That is a normal sparse
        panel outcome.  Query only the target positions from the indexed reference
        and require an exact CHROM/POS/REF/ALT match before launching Beagle.
        """
        target_variants = set()
        normalized_chromosome = chromosome.removeprefix('chr')
        with gzip.open(input_vcf, 'rt') as handle:
            for line in handle:
                if line.startswith('#'):
                    continue
                fields = line.rstrip('\n').split('\t')
                if fields[0].removeprefix('chr') == normalized_chromosome:
                    target_variants.add((fields[1], fields[3], fields[4]))

        if not target_variants:
            return False, 0

        reference_panel = self.get_1000genome_reference(chromosome)
        region_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                prefix=f'{chromosome}_',
                suffix='.regions',
                dir=self.output_dirs['log_dir'],
                delete=False,
            ) as regions:
                region_file = regions.name
                for position in sorted({variant[0] for variant in target_variants}, key=int):
                    regions.write(f'{chromosome}\t{position}\t{position}\n')

            reference_records = subprocess.run(
                ['bcftools', 'view', '-R', region_file, '-H', reference_panel],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for line in reference_records.splitlines():
                fields = line.split('\t')
                if (fields[1], fields[3], fields[4]) in target_variants:
                    return True, len(target_variants)
            return False, len(target_variants)
        finally:
            if region_file and os.path.exists(region_file):
                os.remove(region_file)

    def process_chromosome(self, chromosome: str, params: Dict) -> Dict:
        """Process a single chromosome with Beagle."""
        start_time = time.time()
        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"], "merged_sorted_gt.vcf.gz")
        output_prefix = os.path.join(self.output_dirs["output_vcf_dir"], f"{chromosome}")
        log_file = os.path.join(self.output_dirs["log_dir"], f"{chromosome}.log")

        has_reference_marker, raw_marker_count = self.chromosome_has_reference_marker(input_vcf, chromosome)
        if not has_reference_marker:
            if raw_marker_count:
                message = (
                    f"No target markers on {chromosome} overlap the 1000 Genomes reference "
                    f"({raw_marker_count} raw calls); skipped Beagle.\n"
                )
                status = "panel_absent"
                route_expectation = "all_raw_alleles_panel_absent"
            else:
                message = f"No input markers on {chromosome}; skipped Beagle.\n"
                status = "raw_empty"
                route_expectation = "no_raw_records"
            with open(log_file, 'w') as log:
                log.write(message)
            return {
                "chromosome": chromosome,
                "duration": time.time() - start_time,
                "status": status,
                "raw_marker_count": raw_marker_count,
                "has_reference_marker": False,
                "beagle_required": False,
                "route_expectation": route_expectation,
                "output_file": None,
                "message": message.strip(),
            }

        try:
            self.run_beagle_command(input_vcf, output_prefix, chromosome, params, log_file)
            duration = time.time() - start_time

            beagle_output = f"{output_prefix}.vcf.gz"
            if not os.path.exists(beagle_output):
                raise FileNotFoundError(f"Expected output file not found: {beagle_output}")

            return {
                "chromosome": chromosome,
                "duration": duration,
                "status": "completed",
                "raw_marker_count": raw_marker_count,
                "has_reference_marker": True,
                "beagle_required": True,
                "route_expectation": "beagle_evidence_required",
                "output_file": beagle_output,
            }
            
        except Exception as e:
            return {
                "chromosome": chromosome,
                "duration": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            }


    def _validate_contig_results(
        self,
        results: Sequence[Mapping],
        raw_vcf: Path,
    ) -> Tuple[List[Dict], object]:
        """Validate one explicit, evidence-consistent result per expected contig.

        A completed contig is not accepted merely because a file exists: the
        authoritative router additionally requires data records for that exact
        contig. ``raw_empty`` and ``panel_absent`` remain distinct audited states.
        """
        router = _load_beagle_router()
        raw_keys_by_contig = {}

        def observe_raw(fields, line_number):
            raw_keys_by_contig.setdefault(fields[0], set()).add(
                router.variant_key(fields)
            )

        raw_contract = router._scan_single_sample_vcf(raw_vcf, observe_raw)
        expected = tuple(self.regions)
        expected_set = set(expected)
        unexpected_raw = sorted(set(raw_contract.data_contigs) - expected_set)
        if unexpected_raw:
            raise ValueError(
                "raw VCF contains contigs outside the configured Beagle run: "
                + ", ".join(unexpected_raw)
            )

        by_contig = {}
        for result in results:
            chromosome = result.get("chromosome")
            if chromosome not in expected_set:
                raise ValueError(f"unexpected chromosome result: {chromosome!r}")
            if chromosome in by_contig:
                raise ValueError(f"duplicate chromosome result: {chromosome}")
            by_contig[chromosome] = result
        missing = [chromosome for chromosome in expected if chromosome not in by_contig]
        if missing:
            raise ValueError("missing chromosome result(s): " + ", ".join(missing))

        validated = []
        for chromosome in expected:
            result = dict(by_contig[chromosome])
            status = result.get("status")
            raw_count = int(raw_contract.records_by_contig.get(chromosome, 0))
            try:
                reported_raw_count = int(result.get("raw_marker_count"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{chromosome}: missing or invalid raw_marker_count"
                ) from error
            if reported_raw_count != raw_count:
                raise ValueError(
                    f"{chromosome}: recorded raw_marker_count={reported_raw_count} "
                    f"does not match raw VCF count={raw_count}"
                )

            output_file = result.get("output_file")
            if status == "completed":
                if raw_count == 0 or result.get("has_reference_marker") is not True:
                    raise ValueError(
                        f"{chromosome}: completed status lacks raw/reference-panel evidence"
                    )
                if not output_file:
                    raise ValueError(f"{chromosome}: completed status has no output_file")
                output_path = Path(output_file)
                if not output_path.is_file():
                    raise FileNotFoundError(
                        f"{chromosome}: completed Beagle output is missing: {output_path}"
                    )
                if not Path(str(output_path) + ".tbi").is_file():
                    raise FileNotFoundError(
                        f"{chromosome}: completed Beagle index is missing: {output_path}.tbi"
                    )

                beagle_keys = set()

                def observe_beagle(fields, line_number):
                    beagle_keys.add(router.variant_key(fields))

                beagle_contract = router._scan_single_sample_vcf(
                    output_path, observe_beagle
                )
                if beagle_contract.sample_id != raw_contract.sample_id:
                    raise ValueError(
                        f"{chromosome}: raw/Beagle sample ID mismatch: "
                        f"raw={raw_contract.sample_id!r}, "
                        f"Beagle={beagle_contract.sample_id!r}"
                    )
                data_contigs = set(beagle_contract.data_contigs)
                if data_contigs != {chromosome}:
                    raise ValueError(
                        f"{chromosome}: completed Beagle output contains data for "
                        f"{sorted(data_contigs)!r}, expected only {chromosome!r}"
                    )
                exact_overlap_count = len(
                    raw_keys_by_contig.get(chromosome, set()) & beagle_keys
                )
                if exact_overlap_count == 0:
                    raise ValueError(
                        f"{chromosome}: completed Beagle output has no exact "
                        "raw-allele overlap"
                    )
                result["beagle_record_count"] = beagle_contract.record_count
                result["exact_raw_beagle_overlap_count"] = exact_overlap_count
                result["beagle_sample_id"] = beagle_contract.sample_id
            elif status == "raw_empty":
                if raw_count != 0 or result.get("has_reference_marker") is not False:
                    raise ValueError(f"{chromosome}: invalid raw_empty status")
                if output_file is not None:
                    raise ValueError(f"{chromosome}: raw_empty status has an output_file")
            elif status == "panel_absent":
                if raw_count == 0 or result.get("has_reference_marker") is not False:
                    raise ValueError(f"{chromosome}: invalid panel_absent status")
                if output_file is not None:
                    raise ValueError(f"{chromosome}: panel_absent status has an output_file")
            else:
                raise ValueError(f"{chromosome}: invalid chromosome status {status!r}")
            validated.append(result)
        return validated, raw_contract

    def _write_contig_status_provenance(
        self,
        results: Sequence[Mapping],
        raw_contract,
    ) -> Path:
        """Write the immutable per-contig execution/routing contract."""
        path = Path(self.output_dirs["output_vcf_dir"]) / "BEAGLE_CONTIG_STATUS.json"
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset_name,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "production_inputs_include_wes": False,
            "raw_sample_id": raw_contract.sample_id,
            "raw_record_count": raw_contract.record_count,
            "expected_contigs": list(self.regions),
            "status_counts": {
                status: sum(result["status"] == status for result in results)
                for status in ("completed", "panel_absent", "raw_empty")
            },
            "contigs": [dict(result) for result in results],
        }
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def _route_without_beagle(
        self,
        raw_vcf: Path,
        output_dir: Path,
        raw_contract,
        contig_status_path: Path,
    ) -> Mapping[str, int]:
        """Build an all-panel-absent route bundle without a fake Beagle VCF."""
        router = _load_beagle_router()
        schema = router.ROUTER_SCHEMAS[self.anchor_score]
        route_files = {
            route: output_dir / f"all_{route}.vcf" for route in router.ROUTES
        }
        route_files.update({
            "filtered_in": output_dir / "all_filtered_in.vcf",
            "filtered_out": output_dir / "all_filtered_out.vcf",
            "panel_discordant": output_dir / "all_panel_discordant.vcf",
            "panel_present": output_dir / "all_panel_present.vcf",
        })
        generated = [
            output_dir / "beagle_evidence_audit.tsv.gz",
            output_dir / "beagle_evidence_summary.tsv",
            output_dir / "MANIFEST.json",
            output_dir / "RUN_COMPLETE",
        ]
        for path in route_files.values():
            generated.extend((
                path,
                Path(str(path) + ".gz"),
                Path(str(path) + ".gz.tbi"),
            ))
        conflicts = [path for path in generated if path.exists()]
        if conflicts:
            raise FileExistsError(
                "refusing to overwrite route outputs: "
                + ", ".join(str(path) for path in conflicts[:5])
            )

        handles = {name: path.open("wt") for name, path in route_files.items()}
        counts = {route: 0 for route in router.ROUTES}
        audit_path = output_dir / "beagle_evidence_audit.tsv.gz"
        audit_columns = (
            "schema", "chrom", "pos", "ref", "alt", "raw_gt", "beagle_gt",
            "beagle_gp", "beagle_called_gp", "beagle_p_nonref", "beagle_gp_entropy",
            "beagle_gp_margin", "beagle_af", "gt_transition", "route", "hard_anchor",
        )
        try:
            with gzip.open(audit_path, "wt") as audit, router.open_text(raw_vcf) as raw_handle:
                audit.write("\t".join(audit_columns) + "\n")
                for line_number, line in enumerate(raw_handle, start=1):
                    if line.startswith("#"):
                        if line.startswith("#CHROM"):
                            for handle in handles.values():
                                handle.writelines(router._meta_headers(
                                    schema,
                                    self.anchor_score,
                                    self.min_anchor_probability,
                                ))
                        for handle in handles.values():
                            handle.write(line)
                        continue
                    fields = line.rstrip("\n").split("\t")
                    router._validate_vcf_data_fields(fields, raw_vcf, line_number)
                    raw_gt = router.normalize_gt(
                        router.extract_format_value(fields[8], fields[9], "GT")
                    )
                    routed_line, annotations = router._annotate_line(
                        line, "panel_absent", raw_gt, None, schema
                    )
                    handles["panel_absent"].write(routed_line)
                    for compatibility in router.compatibility_outputs("panel_absent"):
                        handles[compatibility].write(routed_line)
                    counts["panel_absent"] += 1
                    audit.write("\t".join((
                        schema, fields[0], fields[1], fields[3], fields[4], raw_gt or ".",
                        annotations["BEAGLE_GT"], annotations["BEAGLE_GP"],
                        annotations["BEAGLE_CALLED_GP"], annotations["BEAGLE_P_NONREF"],
                        annotations["BEAGLE_GP_ENTROPY"], annotations["BEAGLE_GP_MARGIN"],
                        annotations["BEAGLE_AF"], annotations["BEAGLE_GT_TRANSITION"],
                        "panel_absent", "0",
                    )) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

        if counts["panel_absent"] != raw_contract.record_count:
            raise AssertionError("all-panel-absent routing did not conserve raw variants")
        for path in route_files.values():
            subprocess.run([PATH_CONFIG["BGZIP"], "-f", str(path)], check=True)
            subprocess.run(
                [PATH_CONFIG["TABIX"], "-f", "-p", "vcf", str(path) + ".gz"],
                check=True,
            )

        with (output_dir / "beagle_evidence_summary.tsv").open("x") as handle:
            handle.write(
                "schema\troute\tcount\tanchor_score\tmin_anchor_probability\thard_anchor\n"
            )
            for route in router.ROUTES:
                handle.write(
                    f"{schema}\t{route}\t{counts[route]}\t{self.anchor_score}\t"
                    f"{self.min_anchor_probability:g}\t"
                    f"{int(route == 'beagle_nonref_highconf')}\n"
                )
            handle.write(
                f"{schema}\ttotal\t{raw_contract.record_count}\t{self.anchor_score}\t"
                f"{self.min_anchor_probability:g}\t.\n"
            )

        manifest = {
            "schema": schema,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "production_inputs_include_wes": False,
            "routing_mode": "all_panel_absent_no_beagle",
            "raw_vcf": str(raw_vcf.resolve()),
            "raw_vcf_sha256": router._sha256(raw_vcf),
            "raw_sample_id": raw_contract.sample_id,
            "raw_record_count": raw_contract.record_count,
            "beagle_vcfs": [],
            "beagle_vcf_count": 0,
            "beagle_sample_id": None,
            "beagle_covered_contigs": [],
            "expected_contigs": list(self.regions),
            "exact_overlap_count": 0,
            "contig_status_file": str(contig_status_path.resolve()),
            "contig_status_file_sha256": router._sha256(contig_status_path),
            "anchor_score": self.anchor_score,
            "min_anchor_probability": self.min_anchor_probability,
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
            "counts": {
                **counts,
                "total": raw_contract.record_count,
                "beagle_records": 0,
            },
            "router_source": str(Path(router.__file__).resolve()),
            "router_source_sha256": router._sha256(Path(router.__file__).resolve()),
        }
        with (output_dir / "MANIFEST.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (output_dir / "RUN_COMPLETE").open("x") as handle:
            handle.write("complete\n")
        return counts

    def run_pipeline(self, custom_params: Optional[Dict] = None) -> pd.DataFrame:
        """Run the complete Beagle pipeline."""
        params = DEFAULT_PARAMS.copy()
        if custom_params:
            params.update(custom_params)
        
        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"], "merged_sorted_gt.vcf.gz")
        if not os.path.exists(input_vcf):
            raise FileNotFoundError(f"Input VCF not found: {input_vcf}")
        self._claim_fresh_output_tree(params)
        
        # Process chromosomes in parallel
        results = []
        with ThreadPoolExecutor(
            max_workers=min(params["PARALLEL_CHROMOSOMES"], len(self.regions))
        ) as executor:
            future_to_chrom = {
                executor.submit(
                    self.process_chromosome,
                    chrom,
                    params
                ): chrom for chrom in self.regions
            }
            
            with tqdm(total=len(self.regions), desc="Processing chromosomes") as pbar:
                for future in as_completed(future_to_chrom):
                    chrom = future_to_chrom[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "chromosome": chrom,
                            "status": "failed",
                            "error": str(e)
                        })
                    pbar.update(1)
        
        # Persist per-contig execution metrics before route validation so a
        # fail-closed run still leaves an audit trail in its fresh output tree.
        metrics_df = pd.DataFrame(results)
        metrics_file = os.path.join(
            self.output_dirs["metrics_dir"],
            f"{self.dataset_name}_beagle_metrics.csv",
        )
        if self.section_id:
            metrics_file = os.path.join(
                self.output_dirs["metrics_dir"],
                f"{self.dataset_name}_{self.section_id}_beagle_metrics.csv",
            )
        metrics_df.to_csv(metrics_file, index=False, mode="x")

        valid_statuses = {"completed", "panel_absent", "raw_empty"}
        failed_count = int((~metrics_df["status"].isin(valid_statuses)).sum())
        if failed_count:
            print(
                f"ERROR: Beagle failed on {failed_count} chromosome(s); "
                "combined filtered VCFs were not written."
            )
            raise RuntimeError(
                f"Beagle failed on {failed_count} chromosome(s); partial outputs "
                "were retained for audit and the run cannot be resumed in place"
            )

        validated, raw_contract = self._validate_contig_results(
            results, Path(input_vcf)
        )
        contig_status_path = self._write_contig_status_provenance(
            validated, raw_contract
        )
        completed = [
            result for result in validated if result["status"] == "completed"
        ]

        print("\nRouting raw calls with the versioned Beagle evidence router...")
        router = _load_beagle_router()
        if completed:
            routing_counts = router.route_vcf(
                Path(input_vcf),
                [Path(result["output_file"]) for result in completed],
                Path(self.output_dirs["output_vcf_dir"]),
                min_anchor_probability=self.min_anchor_probability,
                anchor_score=self.anchor_score,
                bgzip=PATH_CONFIG["BGZIP"],
                tabix=PATH_CONFIG["TABIX"],
                allow_existing_output_dir=True,
                expected_contigs=self.regions,
                contig_status_path=contig_status_path,
            )
            routing_mode = "beagle_with_explicit_contig_statuses"
        else:
            routing_counts = self._route_without_beagle(
                Path(input_vcf),
                Path(self.output_dirs["output_vcf_dir"]),
                raw_contract,
                contig_status_path,
            )
            routing_mode = "all_panel_absent_no_beagle"

        print(
            f"Hard-anchor VCF created with {self.anchor_score}>="
            f"{self.min_anchor_probability:g}: "
            f"{self.output_dirs['output_vcf_dir']}/all_filtered_in.vcf.gz"
        )
        print("Routing counts: " + ", ".join(
            f"{status}={routing_counts.get(status, 0):,}"
            for status in router.ROUTES
        ))

        complete_marker = Path(self.output_dirs["output_vcf_dir"]) / "RUN_COMPLETE.json"
        with complete_marker.open("x") as handle:
            json.dump({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "metrics_file": str(Path(metrics_file).resolve()),
                "contig_status_file": str(contig_status_path.resolve()),
                "routing_mode": routing_mode,
                "expected_contigs": list(self.regions),
                "completed_beagle_contigs": [
                    result["chromosome"] for result in completed
                ],
                "production_inputs_include_wes": False,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return metrics_df

def main():
    parser = argparse.ArgumentParser(description="Beagle Pipeline for SNV Processing")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--threads", type=int, default=DEFAULT_PARAMS["THREADS"],
                      help="Number of threads to use")
    parser.add_argument("--memory", default=DEFAULT_PARAMS["MEMORY"],
                      help="Memory allocation for Beagle (e.g., '20g')")
    parser.add_argument("--parallel-chromosomes", type=int,
                      default=DEFAULT_PARAMS["PARALLEL_CHROMOSOMES"],
                      help="Number of concurrent Beagle processes (default: 1)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter directory (default: baseQ0mapQ0)")
    parser.add_argument("--section-root", default=None,
                      help="Override the section output root for an isolated run")
    parser.add_argument("--anchor-score", choices=("p_nonref", "called_gp"),
                      default="p_nonref",
                      help="Evidence score for hard germline anchors (default: p_nonref)")
    parser.add_argument("--min-anchor-probability", type=float, default=0.90,
                      help="Minimum anchor probability (default: 0.90)")
    parser.add_argument("--min-anchor-gp", type=float, default=None,
                      help=("Deprecated v1 ablation alias: use called-genotype GP "
                            "at this threshold instead of the default P(nonref)"))
    parser.add_argument("--iterations", type=int, default=DEFAULT_PARAMS["ITERATIONS"],
                      help="Beagle niterations value (default: 0, historical SPARCAL setting)")
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMS["SEED"],
                      help="Explicit Beagle random seed for reproducible ablations")
    parser.add_argument("--impute", action="store_true", default=DEFAULT_PARAMS["IMPUTE"],
                      help="Enable marker imputation (off by default; routing uses observed raw alleles)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BeaglePipeline(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        section_root=args.section_root,
        min_anchor_gp=args.min_anchor_gp,
        anchor_score=args.anchor_score,
        min_anchor_probability=args.min_anchor_probability,
    )
    
    # Set custom parameters
    custom_params = {
        "THREADS": args.threads,
        "MEMORY": args.memory,
        "PARALLEL_CHROMOSOMES": args.parallel_chromosomes,
        "ITERATIONS": args.iterations,
        "SEED": args.seed,
        "IMPUTE": args.impute,
    }
    
    # Run pipeline
    print(f"\nStarting Beagle pipeline for {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Using chromosome format: {pipeline.chr_prefix}N")
    print(f"Quality filter: {args.quality_filter}")
    
    result_df = pipeline.run_pipeline(custom_params=custom_params)
    
    # Print summary
    print("\nBeagle Pipeline Summary:")
    print(f"Total chromosomes processed: {len(result_df)}")
    print(f"Successful runs: {len(result_df[result_df['status'] == 'completed'])}")
    print(f"Failed runs: {len(result_df[result_df['status'] == 'failed'])}")
    if len(result_df[result_df['status'] == 'failed']) > 0:
        print("\nFailed chromosomes:")
        failed_df = result_df[result_df['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"Chromosome {row['chromosome']}: {row['error']}")
    if 'duration' in result_df.columns:
        print(f"Average processing time: {result_df['duration'].mean():.2f} seconds")

if __name__ == "__main__":
    main()

# usage on P4_tumor, baseQ0mapQ0
# python scripts/filtering/run_beagle.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0

# Run beagle on DLPFC 151507
# python scripts/filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ0mapQ0 --section_id 151507
