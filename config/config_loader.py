"""
SPARCAL Config Loader
=====================
Central configuration loader for the SPARCAL pipeline.
All Python scripts import this module instead of maintaining their own
hardcoded REFERENCE_CONFIGS / DATASET_CONFIGS / PATH_CONFIG dicts.

Usage:
    from config_loader import load_config

    cfg = load_config("P4_TUMOR")          # loads configs/P4_TUMOR.yaml
    cfg = load_config("P4_TUMOR", "1")     # also resolves section-specific paths

    # Access fields:
    cfg.reference_fasta          # /data/.../genome.fa
    cfg.chr_prefix               # "chr" or ""
    cfg.regions                  # ["chr1", ..., "chr22"] or ["1", ..., "22"]
    cfg.project_dir              # /data/maiziezhou_lab/leiy4/snv_calling
    cfg.output_dir               # resolved per-section output dir
    cfg.bam_pattern              # resolved glob pattern for BAMs
    cfg.tool("samtools")         # /data/.../apps/samtools
    cfg.calicost_tumor_purity    # path to loh_estimator_tumor_prop.tsv (somatic only)
    ...
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Locate config directory (sibling of this file, or overridden by env var)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
CONFIG_DIR = os.environ.get("SPARCAL_CONFIG_DIR", _DEFAULT_CONFIG_DIR)


# ---------------------------------------------------------------------------
# Dataclass that wraps the parsed YAML for convenient attribute access
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Structured access to a dataset's YAML configuration."""

    # ---- raw YAML dict (kept for any ad-hoc access) ----
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    # ---- resolved at load time ----
    dataset_name: str = ""
    section_id: Optional[str] = None

    # -- tissue type --
    tissue_type: str = ""              # "germline" or "somatic"

    # -- reference genome --
    reference_fasta: str = ""
    chr_prefix: str = ""
    regions: List[str] = field(default_factory=list)

    # -- paths --
    project_dir: str = ""
    apps_dir: str = ""
    output_dir: str = ""               # resolved with section_id
    bam_base_path: str = ""
    bam_pattern: str = ""              # resolved with section_id

    # -- spatial data --
    spatial_dir: str = ""
    position_file: str = ""
    scale_factor_file: str = ""
    image_file: str = ""
    has_header: bool = False
    in_tissue_column: int = 1

    # -- 1000 Genomes --
    thousand_genome_base: str = ""
    thousand_genome_pattern: str = ""

    # -- Beagle --
    beagle_jar: str = ""
    java_path: str = ""
    beagle_threads: int = 24
    beagle_memory: str = "10g"

    # -- quality filter --
    base_quality: int = 0
    mapping_quality: int = 0

    # -- CalicoST (somatic only) --
    calicost_base_dir: str = ""
    calicost_num_clones: int = 3
    calicost_tumor_purity: str = ""    # resolved per-section
    calicost_clone_labels: str = ""    # resolved per-section
    calicost_cnv_segments: str = ""    # resolved per-section

    # -- classifier --
    classifier_model_type: str = "neural_network"
    max_training_samples: int = 90000

    # -- SLURM --
    slurm_account: str = ""
    slurm_partition: str = ""
    conda_env: str = "snv_caller_new"

    # -- section metadata --
    section_ids: List[str] = field(default_factory=list)

    # -- tools cache (private) --
    _tools: Dict[str, str] = field(default_factory=dict, repr=False)

    def tool(self, name: str) -> str:
        """Return the absolute path for a CLI tool (samtools, bcftools, etc.)."""
        if name in self._tools:
            return self._tools[name]
        # fallback: assume it lives in apps_dir
        return os.path.join(self.apps_dir, name)

    @property
    def quality_filter_str(self) -> str:
        return f"baseQ{self.base_quality}mapQ{self.mapping_quality}"

    @property
    def is_somatic(self) -> bool:
        return self.tissue_type == "somatic"

    @property
    def is_germline(self) -> bool:
        return self.tissue_type == "germline"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_template(template: str, section_id: Optional[str] = None, **kwargs) -> str:
    """Replace {section_id} and any other placeholders in a string."""
    if section_id is not None:
        template = template.replace("{section_id}", str(section_id))
    for k, v in kwargs.items():
        template = template.replace(f"{{{k}}}", str(v))
    return template


def _resolve_section_specific(mapping: Dict, section_id: str, fallback: str = "") -> str:
    """Look up section_id in a dict; return the value or fallback."""
    return mapping.get(str(section_id), fallback)


def _build_calicost_subdir(cfg_calicost: Dict, section_id: Optional[str]) -> str:
    """
    Build the CalicoST results subdirectory path for a given section.
    Pattern: {calicost_base_dir}/{folder_name}/calicost/clone{N}_rectangle0_w1.0/
    """
    base = cfg_calicost.get("base_dir", "")
    if not base:
        return ""
    num_clones = cfg_calicost.get("num_clones", 3)
    folder_map = cfg_calicost.get("section_folder_map", {})
    folder = _resolve_section_specific(folder_map, section_id, "")
    if not folder:
        return ""
    return os.path.join(base, folder, "calicost",
                        f"clone{num_clones}_rectangle0_w1.0")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(dataset_name: str, section_id: Optional[str] = None) -> PipelineConfig:
    """
    Load and resolve a dataset YAML config.

    Parameters
    ----------
    dataset_name : str
        Name matching a YAML file in the configs/ directory
        (e.g. "DLPFC", "P4_TUMOR", "DCIS").
    section_id : str, optional
        Section identifier. Required for datasets with multiple sections.

    Returns
    -------
    PipelineConfig
        Fully resolved configuration dataclass.
    """
    yaml_path = os.path.join(CONFIG_DIR, f"{dataset_name.upper()}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)

    # ---- validate section_id ----
    sections_cfg = raw.get("sections", {})
    section_ids = [str(s) for s in sections_cfg.get("ids", [])]
    if section_ids and section_id is None:
        # Don't raise — caller might just want metadata without section resolution
        pass
    if section_id is not None and section_ids and str(section_id) not in section_ids:
        raise ValueError(
            f"Section '{section_id}' not in allowed list {section_ids} "
            f"for dataset {dataset_name}"
        )

    # ---- reference ----
    ref_cfg = raw.get("reference", {})
    chr_prefix = ref_cfg.get("chr_prefix", "")
    if chr_prefix:
        regions = [f"{chr_prefix}{i}" for i in range(1, 23)]
    else:
        regions = [str(i) for i in range(1, 23)]

    # ---- paths ----
    paths_cfg = raw.get("paths", {})
    project_dir = paths_cfg.get("project_dir", "")
    apps_dir = paths_cfg.get("apps_dir", "")

    # ---- tools ----
    tools_cfg = raw.get("tools", {})
    tools = {}
    for tool_name, tool_val in tools_cfg.items():
        if os.path.isabs(str(tool_val)):
            tools[tool_name] = str(tool_val)
        else:
            tools[tool_name] = os.path.join(apps_dir, str(tool_val))

    # ---- BAM / input ----
    input_cfg = raw.get("input", {})
    bam_base = input_cfg.get("bam_base_path", "")
    bam_pattern_raw = input_cfg.get("bam_pattern", "")
    bam_pattern = _resolve_template(bam_pattern_raw, section_id)

    # ---- output ----
    output_cfg = raw.get("output", {})
    output_dir_raw = output_cfg.get("dir_pattern", "")
    output_dir = _resolve_template(output_dir_raw, section_id)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_dir, output_dir)

    # ---- spatial data ----
    spatial_cfg = raw.get("spatial", {})
    spatial_dir_raw = spatial_cfg.get("dir", "")
    position_file_raw = spatial_cfg.get("position_file", "")
    scale_file_raw = spatial_cfg.get("scale_factor_file", "")
    image_file_raw = spatial_cfg.get("image_file", "")

    # Handle per-section spatial file overrides
    section_overrides = spatial_cfg.get("section_overrides", {})
    if section_id and str(section_id) in section_overrides:
        sec_ov = section_overrides[str(section_id)]
        position_file_raw = sec_ov.get("position_file", position_file_raw)
        scale_file_raw = sec_ov.get("scale_factor_file", scale_file_raw)
        image_file_raw = sec_ov.get("image_file", image_file_raw)

    spatial_dir = _resolve_template(spatial_dir_raw, section_id)
    position_file = _resolve_template(position_file_raw, section_id)
    scale_factor_file = _resolve_template(scale_file_raw, section_id)
    image_file = _resolve_template(image_file_raw, section_id)

    # ---- 1000 Genomes ----
    tg_cfg = raw.get("thousand_genomes", {})

    # ---- Beagle ----
    beagle_cfg = raw.get("beagle", {})

    # ---- quality filter ----
    qf_cfg = raw.get("quality_filter", {})

    # ---- CalicoST (somatic only) ----
    calicost_cfg = raw.get("calicost", {})
    calicost_subdir = _build_calicost_subdir(calicost_cfg, section_id)
    calicost_tumor_purity = ""
    calicost_clone_labels = ""
    calicost_cnv_segments = ""
    if calicost_subdir:
        calicost_tumor_purity = os.path.join(calicost_subdir, "loh_estimator_tumor_prop.tsv")
        calicost_clone_labels = os.path.join(calicost_subdir, "clone_labels.tsv")
        calicost_cnv_segments = os.path.join(calicost_subdir, "cnv_seglevel.tsv")

    # ---- classifier ----
    clf_cfg = raw.get("classifier", {})

    # ---- SLURM ----
    slurm_cfg = raw.get("slurm", {})

    # ---- build config object ----
    return PipelineConfig(
        raw=raw,
        dataset_name=dataset_name.upper(),
        section_id=section_id,
        tissue_type=raw.get("tissue_type", "germline"),
        reference_fasta=ref_cfg.get("fasta", ""),
        chr_prefix=chr_prefix,
        regions=regions,
        project_dir=project_dir,
        apps_dir=apps_dir,
        output_dir=output_dir,
        bam_base_path=bam_base,
        bam_pattern=bam_pattern,
        spatial_dir=spatial_dir,
        position_file=position_file,
        scale_factor_file=scale_factor_file,
        image_file=image_file,
        has_header=spatial_cfg.get("has_header", False),
        in_tissue_column=spatial_cfg.get("in_tissue_column", 1),
        thousand_genome_base=tg_cfg.get("base_path", ""),
        thousand_genome_pattern=tg_cfg.get("pattern", ""),
        beagle_jar=beagle_cfg.get("jar", ""),
        java_path=beagle_cfg.get("java", ""),
        beagle_threads=beagle_cfg.get("threads", 24),
        beagle_memory=beagle_cfg.get("memory", "10g"),
        base_quality=qf_cfg.get("base_quality", 0),
        mapping_quality=qf_cfg.get("mapping_quality", 0),
        calicost_base_dir=calicost_cfg.get("base_dir", ""),
        calicost_num_clones=calicost_cfg.get("num_clones", 3),
        calicost_tumor_purity=calicost_tumor_purity,
        calicost_clone_labels=calicost_clone_labels,
        calicost_cnv_segments=calicost_cnv_segments,
        classifier_model_type=clf_cfg.get("model_type", "neural_network"),
        max_training_samples=clf_cfg.get("max_training_samples", 90000),
        slurm_account=slurm_cfg.get("account", "maiziezhou_lab_acc"),
        slurm_partition=slurm_cfg.get("partition", "batch_gpu"),
        conda_env=slurm_cfg.get("conda_env", "snv_caller_new"),
        section_ids=section_ids,
        _tools=tools,
    )


def load_all_sections(dataset_name: str) -> List[PipelineConfig]:
    """
    Convenience: load configs for every section in a dataset.
    Returns a list of PipelineConfig, one per section.
    """
    # First load without section to get section_ids
    meta = load_config(dataset_name)
    if not meta.section_ids:
        return [meta]
    return [load_config(dataset_name, sid) for sid in meta.section_ids]