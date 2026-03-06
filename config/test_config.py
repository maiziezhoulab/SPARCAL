#!/usr/bin/env python3
"""Quick smoke test for config_loader + all YAML configs."""

import sys
sys.path.insert(0, "/home/claude")

from config_loader import load_config, load_all_sections

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def test_dataset(name, expected_tissue, expected_sections, expected_prefix, expected_build):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Load without section
    cfg = load_config(name)
    assert cfg.tissue_type == expected_tissue, f"tissue_type: got {cfg.tissue_type}, expected {expected_tissue}"
    print(f"  {PASS} tissue_type = {cfg.tissue_type}")

    assert cfg.chr_prefix == expected_prefix, f"chr_prefix: got '{cfg.chr_prefix}', expected '{expected_prefix}'"
    print(f"  {PASS} chr_prefix = '{cfg.chr_prefix}'")

    assert len(cfg.regions) == 22, f"regions count: {len(cfg.regions)}"
    print(f"  {PASS} regions = {cfg.regions[0]}..{cfg.regions[-1]} (22 total)")

    assert cfg.section_ids == expected_sections, f"sections: {cfg.section_ids}"
    print(f"  {PASS} section_ids = {cfg.section_ids}")

    assert cfg.reference_fasta.endswith("genome.fa"), f"ref fasta: {cfg.reference_fasta}"
    print(f"  {PASS} reference_fasta = {cfg.reference_fasta}")

    build = cfg.raw.get("reference", {}).get("build", "")
    assert build == expected_build, f"build: {build}"
    print(f"  {PASS} build = {build}")

    assert cfg.project_dir == "/data/maiziezhou_lab/leiy4/snv_calling"
    print(f"  {PASS} project_dir OK")

    assert cfg.conda_env == "snv_caller_new"
    print(f"  {PASS} conda_env = {cfg.conda_env}")

    assert cfg.quality_filter_str == "baseQ0mapQ0"
    print(f"  {PASS} quality_filter_str = {cfg.quality_filter_str}")

    # Test section-specific loading
    sid = expected_sections[0]
    cfg_s = load_config(name, sid)
    assert cfg_s.section_id == sid
    assert "{section_id}" not in cfg_s.output_dir, f"Unresolved template in output_dir: {cfg_s.output_dir}"
    assert "{section_id}" not in cfg_s.bam_pattern, f"Unresolved template in bam_pattern: {cfg_s.bam_pattern}"
    print(f"  {PASS} section '{sid}' resolved:")
    print(f"      output_dir  = {cfg_s.output_dir}")
    print(f"      bam_pattern = {cfg_s.bam_pattern}")

    # Tools
    assert cfg_s.tool("samtools").endswith("samtools")
    assert cfg_s.tool("bcftools").endswith("bcftools")
    print(f"  {PASS} tools: samtools={cfg_s.tool('samtools')}")

    # 1kG
    assert cfg_s.thousand_genome_base != ""
    assert "/lio/lfs" not in cfg_s.thousand_genome_base, "Still contains /lio/lfs!"
    print(f"  {PASS} 1kG base = {cfg_s.thousand_genome_base}")

    # Somatic-specific
    if expected_tissue == "somatic":
        assert cfg_s.calicost_tumor_purity.endswith("loh_estimator_tumor_prop.tsv"), \
            f"calicost purity: {cfg_s.calicost_tumor_purity}"
        assert cfg_s.calicost_clone_labels.endswith("clone_labels.tsv")
        assert cfg_s.calicost_cnv_segments.endswith("cnv_seglevel.tsv")
        assert "clone3_rectangle0_w1.0" in cfg_s.calicost_tumor_purity
        print(f"  {PASS} CalicoST paths resolved:")
        print(f"      tumor_purity = {cfg_s.calicost_tumor_purity}")
        print(f"      clone_labels = {cfg_s.calicost_clone_labels}")
        print(f"      cnv_segments = {cfg_s.calicost_cnv_segments}")
    else:
        assert cfg_s.calicost_tumor_purity == ""
        print(f"  {PASS} No CalicoST (germline dataset)")

    # Spatial
    if cfg_s.position_file:
        print(f"  {PASS} spatial: position_file = {cfg_s.position_file}")
    if cfg_s.spatial_dir:
        assert "{section_id}" not in cfg_s.spatial_dir
        print(f"  {PASS} spatial_dir = {cfg_s.spatial_dir}")

    # load_all_sections
    all_cfgs = load_all_sections(name)
    assert len(all_cfgs) == len(expected_sections)
    print(f"  {PASS} load_all_sections returned {len(all_cfgs)} configs")

    print(f"\n  {PASS} ALL TESTS PASSED for {name}")


# ---- Run tests ----
test_dataset("DLPFC",    "germline", 
             ["151507","151508","151509","151510",
              "151669","151670","151671","151672",
              "151673","151674","151675","151676"],
             "", "GRCh38")

test_dataset("P4_TUMOR", "somatic", ["1","2"], "chr", "hg19")
test_dataset("P6_TUMOR", "somatic", ["1","2"], "chr", "hg19")
test_dataset("DCIS",     "somatic", ["1","2"], "",    "GRCh38")

print(f"\n{'='*60}")
print(f"  {PASS}{PASS}{PASS} ALL DATASETS PASSED {PASS}{PASS}{PASS}")
print(f"{'='*60}")