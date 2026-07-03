#!/usr/bin/env python3
"""
run_generate_final_vcf.py

Post-processing step: generate publishable VCF outputs from SPARCAL pipeline results.

Handles two upstream spatial filter structures
----------------------------------------------
Basic (run_spatial_snv_filter.py):
    {spatial_filter_dir}/
        all_filtered_variants.txt    ← union of all passing variants
        {barcode}.txt                ← per-barcode variant lists

Enhanced (run_spatial_snv_filter_enhanced.py):
    {spatial_filter_dir}/
        all_variant_scores.txt
        somatic/
            somatic_variants.txt     ← union of somatic variants
            somatic_variants.vcf.gz
            denovo/                  ← per-barcode txt files (denovo somatic)
                {barcode}.txt
            defined/                 ← per-barcode txt files (defined somatic)
                {barcode}.txt
        germline/ ...
        ambiguous/ ...

Pass the base directory in either case via --spatial_filter_dir.
Structure is auto-detected; --classification selects which set to emit (default: somatic).

Two output types
----------------
Type 1 — final_vcf_per_spot/{barcode}.vcf.gz
    Per-barcode VCF retaining all FORMAT fields (GT:AD:DP:GQ:PL), filtered to
    spatially-confirmed variants of the chosen classification.
    INFO enriched with spatial filter scores and NN predictions when available.

Type 2 — final_snv_profile.vcf.gz
    1000-Genomes-style multi-sample VCF.
    INFO: SPOT_COUNT, SPOT_FREQ, DP_MEAN, AF_MEAN, RACE,
          GERMLINE_SCORE, SOMATIC_SCORE, NN_CLASS, NN_HOMO, NN_HETERO, NN_NOVAR.
    FORMAT: GT only (0/1 = carrying spot, ./. = absent).

Usage
-----
  python run_generate_final_vcf.py \\
      --dataset P4_TUMOR \\
      --section_id 1 \\
      --quality_filter baseQ0mapQ0 \\
      --spatial_filter_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0
"""

import os
import sys
import gzip
import glob
import argparse
import logging
import subprocess
import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARCAL_VERSION = "1.0.0"

PROJECT_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
BGZIP    = "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip"
BCFTOOLS = "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools"

DATASET_SUBDIRS: Dict[str, str] = {
    "P4_TUMOR": "P4_tumor",
    "P6_TUMOR": "P6_tumor",
    "DLPFC":    "dlpfc",
    "DCIS":     "dcis",
}

# Output INFO key names
_INFO_GERMLINE_SCORE = "GERMLINE_SCORE"
_INFO_SOMATIC_SCORE  = "SOMATIC_SCORE"
_INFO_NN_CLASS  = "NN_CLASS"
_INFO_NN_HOMO   = "NN_HOMO"
_INFO_NN_HETERO = "NN_HETERO"
_INFO_NN_NOVAR  = "NN_NOVAR"

# Source INFO keys in neural_network_predictions.vcf.gz
_SRC_NN_CLASS  = "NEURAL_NETWORK_CLASS"
_SRC_NN_HOMO   = "NEURAL_NETWORK_HOMO"
_SRC_NN_HETERO = "NEURAL_NETWORK_HETERO"
_SRC_NN_NOVAR  = "NEURAL_NETWORK_NOVAR"

VALID_CLASSIFICATIONS = ["somatic", "germline", "ambiguous", "all"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_chrom(chrom: str) -> str:
    return chrom.replace("chr", "")


def format_chrom(chrom: str, use_prefix: bool = True) -> str:
    bare = normalize_chrom(chrom)
    return f"chr{bare}" if use_prefix else bare


def open_vcf(path: str):
    """Open a VCF or VCF.gz, returns (fh, is_proc). Falls back to zcat."""
    if path.endswith(".gz"):
        try:
            fh = gzip.open(path, "rt")
            fh.read(1)
            fh.seek(0)
            return fh, False
        except Exception:
            proc = subprocess.Popen(
                ["zcat", path], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            )
            return proc.stdout, True
    return open(path, "r"), False


def run_cmd(cmd: str) -> bool:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning(f"Command failed:\n  {cmd}\n  STDERR: {r.stderr.strip()}")
        return False
    return True


def parse_info_str(info_str: str) -> dict:
    result: dict = {}
    if not info_str or info_str == ".":
        return result
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = True
    return result


def build_info_str(info_dict: dict) -> str:
    parts = [k if v is True else f"{k}={v}" for k, v in info_dict.items()]
    return ";".join(parts) if parts else "."


def parse_format_sample(format_str: str, sample_str: str) -> dict:
    keys = format_str.split(":")
    vals = sample_str.split(":") if sample_str not in (".", "./.", "") else []
    return {k: (vals[i] if i < len(vals) else ".") for i, k in enumerate(keys)}


def sort_variant_key(v: Tuple) -> Tuple:
    chrom = v[0]
    try:
        return (0, int(chrom), int(v[1]))
    except ValueError:
        return (1, chrom, int(v[1]) if v[1].isdigit() else 0)


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def detect_filter_structure(base_dir: str) -> str:
    """
    Detect which spatial filter produced the output directory.

    Returns
    -------
    'enhanced'  if somatic/ subdirectory exists  (run_spatial_snv_filter_enhanced.py)
    'basic'     if all_filtered_variants.txt exists  (run_spatial_snv_filter.py)
    'unknown'   otherwise
    """
    if os.path.isdir(os.path.join(base_dir, "somatic")):
        return "enhanced"
    if os.path.isfile(os.path.join(base_dir, "all_filtered_variants.txt")):
        return "basic"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-barcode variant loading — enhanced filter
# ---------------------------------------------------------------------------

def _load_txt_dir(txt_dir: str) -> Dict[str, Set[Tuple]]:
    """
    Load per-barcode variant sets from a directory of {barcode}.txt files.
    Format: tab-separated chrom pos ref alt (no chr prefix, no header).
    """
    per_spot: Dict[str, Set] = {}
    for txt_file in glob.glob(os.path.join(txt_dir, "*.txt")):
        fname = os.path.basename(txt_file)
        # Skip any summary files
        if "_variants" in fname or "summary" in fname.lower():
            continue
        barcode = fname.replace(".txt", "")
        variants: Set[Tuple] = set()
        with open(txt_file) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    chrom, pos, ref, alt = parts[:4]
                    variants.add((normalize_chrom(chrom), pos, ref, alt))
        if variants:
            per_spot[barcode] = variants
    return per_spot


def load_per_spot_enhanced(
    base_dir: str,
    classification: str = "somatic",
) -> Dict[str, Set[Tuple]]:
    """
    Load per-barcode variant sets from the enhanced spatial filter output.

    Merges denovo and defined subdirectories for the chosen classification.
    Falls back to adjacent classifications when 'all' is requested.
    """
    classes = (
        ["somatic", "germline", "ambiguous"]
        if classification == "all"
        else [classification]
    )

    per_spot: Dict[str, Set] = {}
    for cls in classes:
        for subdir_name in ("denovo", "defined"):
            subdir = os.path.join(base_dir, cls, subdir_name)
            if not os.path.isdir(subdir):
                continue
            partial = _load_txt_dir(subdir)
            for barcode, variants in partial.items():
                if barcode in per_spot:
                    per_spot[barcode].update(variants)
                else:
                    per_spot[barcode] = set(variants)

    logger.info(
        f"Enhanced: loaded per-barcode variants for {len(per_spot)} barcodes "
        f"(classification={classification})"
    )
    return per_spot


def load_all_passing_enhanced(
    base_dir: str,
    classification: str = "somatic",
) -> Set[Tuple]:
    """
    Load the union of all passing variants from the enhanced filter output.

    Reads {classification}/{classification}_variants.txt.
    Falls back to deriving the union from per-barcode data if file absent.
    """
    classes = (
        ["somatic", "germline", "ambiguous"]
        if classification == "all"
        else [classification]
    )

    passing: Set[Tuple] = set()
    for cls in classes:
        variant_file = os.path.join(base_dir, cls, f"{cls}_variants.txt")
        if not os.path.exists(variant_file):
            logger.warning(f"Variant file not found: {variant_file}; skipping.")
            continue
        try:
            df = pd.read_csv(variant_file, sep="\t")
        except Exception as e:
            logger.warning(f"Could not read {variant_file}: {e}")
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        chrom_col = next(
            (cols_lower[k] for k in ("chrom", "chr", "chromosome") if k in cols_lower),
            None,
        )
        pos_col = next(
            (cols_lower[k] for k in ("pos", "position", "start") if k in cols_lower),
            None,
        )
        ref_col = next((cols_lower[k] for k in ("ref",) if k in cols_lower), None)
        alt_col = next((cols_lower[k] for k in ("alt",) if k in cols_lower), None)

        if chrom_col and pos_col and ref_col and alt_col:
            for _, row in df.iterrows():
                try:
                    passing.add((
                        normalize_chrom(str(row[chrom_col])),
                        str(int(float(str(row[pos_col])))),
                        str(row[ref_col]),
                        str(row[alt_col]),
                    ))
                except (ValueError, TypeError):
                    continue
        else:
            logger.warning(
                f"{variant_file} has unrecognised column names; cannot parse. "
                f"Columns found: {list(df.columns)}"
            )

    logger.info(
        f"Enhanced: loaded {len(passing)} unique passing variants "
        f"(classification={classification})"
    )
    return passing


# ---------------------------------------------------------------------------
# Per-barcode variant loading — basic filter
# ---------------------------------------------------------------------------

def load_per_spot_basic(base_dir: str) -> Dict[str, Set[Tuple]]:
    """Load per-barcode variant sets from the basic spatial filter output."""
    per_spot: Dict[str, Set] = {}
    for txt_file in glob.glob(os.path.join(base_dir, "*.txt")):
        fname = os.path.basename(txt_file)
        if fname == "all_filtered_variants.txt":
            continue
        barcode = fname.replace(".txt", "")
        variants: Set[Tuple] = set()
        with open(txt_file) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    chrom, pos, ref, alt = parts[:4]
                    variants.add((normalize_chrom(chrom), pos, ref, alt))
        if variants:
            per_spot[barcode] = variants
    logger.info(
        f"Basic: loaded per-barcode variants for {len(per_spot)} barcodes"
    )
    return per_spot


def load_all_passing_basic(base_dir: str) -> Set[Tuple]:
    """Load union of all passing variants from basic filter all_filtered_variants.txt."""
    summary_file = os.path.join(base_dir, "all_filtered_variants.txt")
    passing: Set[Tuple] = set()
    if not os.path.exists(summary_file):
        logger.error(f"all_filtered_variants.txt not found: {summary_file}")
        return passing
    with open(summary_file) as fh:
        for i, line in enumerate(fh):
            if i == 0 and line.startswith("Chrom"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                passing.add((normalize_chrom(parts[0]), parts[1], parts[2], parts[3]))
    logger.info(f"Basic: loaded {len(passing)} passing variants")
    return passing


# ---------------------------------------------------------------------------
# Fallback: derive per-spot sets from source VCFs
# ---------------------------------------------------------------------------

def derive_per_spot_from_vcfs(
    all_passing: Set[Tuple],
    snv_vcf_dir: str,
) -> Dict[str, Set[Tuple]]:
    """
    Re-derive per-barcode variant sets by scanning source VCFs.

    Used as a fallback when neither basic nor enhanced per-barcode files exist,
    or when per-barcode subdirectories are empty.

    Only variants that are in all_passing AND present in the barcode's source VCF
    are included.
    """
    per_spot: Dict[str, Set] = {}
    vcf_files = sorted(
        glob.glob(os.path.join(snv_vcf_dir, "*.vcf.gz"))
        + glob.glob(os.path.join(snv_vcf_dir, "*.vcf"))
    )
    # Exclude index files
    vcf_files = [f for f in vcf_files if not f.endswith(".tbi") and not f.endswith(".csi")]

    logger.info(
        f"Fallback: re-deriving per-spot variants from {len(vcf_files)} source VCFs..."
    )
    for vcf_path in vcf_files:
        barcode = os.path.basename(vcf_path).replace(".vcf.gz", "").replace(".vcf", "")
        fh, is_proc = open_vcf(vcf_path)
        variants: Set[Tuple] = set()
        try:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                key = (normalize_chrom(parts[0]), parts[1], parts[3], parts[4])
                if key in all_passing:
                    variants.add(key)
        finally:
            if not is_proc:
                fh.close()
        if variants:
            per_spot[barcode] = variants

    logger.info(
        f"Fallback: found {len(per_spot)} barcodes with at least one passing variant"
    )
    return per_spot


# ---------------------------------------------------------------------------
# Enrichment sources: spatial scores + NN predictions
# ---------------------------------------------------------------------------

def load_spatial_scores(
    scores_file: str,
) -> Dict[Tuple[str, str, str, str], dict]:
    """
    Load germline / somatic spatial filter scores from all_variant_scores.txt.
    Returns empty dict (+ warning) on any failure; never raises.
    """
    if not os.path.exists(scores_file):
        logger.warning(f"Spatial scores file not found (skipping): {scores_file}")
        return {}
    try:
        df = pd.read_csv(scores_file, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to read {scores_file}: {e}")
        return {}

    cols_lower = {c.lower(): c for c in df.columns}
    chrom_col = next(
        (cols_lower[k] for k in ("chrom", "chr", "chromosome") if k in cols_lower), None
    )
    pos_col  = next(
        (cols_lower[k] for k in ("pos", "position", "start") if k in cols_lower), None
    )
    ref_col  = next((cols_lower[k] for k in ("ref",) if k in cols_lower), None)
    alt_col  = next((cols_lower[k] for k in ("alt",) if k in cols_lower), None)
    g_col    = next(
        (cols_lower[k] for k in ("germline_score", "germline") if k in cols_lower), None
    )
    s_col    = next(
        (cols_lower[k] for k in ("somatic_score", "somatic") if k in cols_lower), None
    )

    if g_col is None and s_col is None:
        logger.warning(f"No score columns found in {scores_file}; skipping.")
        return {}
    if not (chrom_col and pos_col and ref_col and alt_col):
        logger.warning(f"Missing identifier columns in {scores_file}; skipping.")
        return {}

    scores: Dict[Tuple, dict] = {}
    for _, row in df.iterrows():
        try:
            key = (
                normalize_chrom(str(row[chrom_col])),
                str(int(float(str(row[pos_col])))),
                str(row[ref_col]),
                str(row[alt_col]),
            )
        except (ValueError, TypeError):
            continue
        entry: dict = {}
        if g_col is not None:
            try:
                entry[_INFO_GERMLINE_SCORE] = f"{float(row[g_col]):.6f}"
            except (ValueError, TypeError):
                pass
        if s_col is not None:
            try:
                entry[_INFO_SOMATIC_SCORE] = f"{float(row[s_col]):.6f}"
            except (ValueError, TypeError):
                pass
        if entry:
            scores[key] = entry

    logger.info(f"Loaded spatial scores for {len(scores)} variants from {scores_file}")
    return scores


def load_nn_scores(
    nn_vcf_path: str,
) -> Dict[Tuple[str, str, str, str], dict]:
    """
    Load neural network prediction scores from neural_network_predictions.vcf.gz.
    Remaps NEURAL_NETWORK_* INFO keys to the shorter NN_* names.
    Returns empty dict (+ warning) on any failure; never raises.
    """
    if not os.path.exists(nn_vcf_path):
        logger.warning(f"NN predictions VCF not found (skipping): {nn_vcf_path}")
        return {}

    nn_scores: Dict[Tuple, dict] = {}
    fh, is_proc = open_vcf(nn_vcf_path)
    try:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            key = (normalize_chrom(parts[0]), parts[1], parts[3], parts[4])
            info = parse_info_str(parts[7])
            entry: dict = {}
            if _SRC_NN_CLASS  in info: entry[_INFO_NN_CLASS]  = info[_SRC_NN_CLASS]
            if _SRC_NN_HOMO   in info: entry[_INFO_NN_HOMO]   = info[_SRC_NN_HOMO]
            if _SRC_NN_HETERO in info: entry[_INFO_NN_HETERO] = info[_SRC_NN_HETERO]
            if _SRC_NN_NOVAR  in info: entry[_INFO_NN_NOVAR]  = info[_SRC_NN_NOVAR]
            if entry:
                nn_scores[key] = entry
    finally:
        if not is_proc:
            fh.close()

    logger.info(f"Loaded NN scores for {len(nn_scores)} variants from {nn_vcf_path}")
    return nn_scores


def _enrich_info(
    info_str: str,
    variant_key: Tuple,
    spatial_scores: Dict,
    nn_scores: Dict,
) -> str:
    """Inject spatial and NN scores into an INFO string; never overwrites existing keys."""
    info = parse_info_str(info_str)
    for src in (spatial_scores.get(variant_key), nn_scores.get(variant_key)):
        if src:
            for k, v in src.items():
                if k not in info:
                    info[k] = v
    return build_info_str(info)


def _extra_info_header_lines() -> List[str]:
    return [
        f'##INFO=<ID={_INFO_GERMLINE_SCORE},Number=1,Type=Float,'
        f'Description="Composite germline spatial score from SPARCAL enhanced spatial filter">',
        f'##INFO=<ID={_INFO_SOMATIC_SCORE},Number=1,Type=Float,'
        f'Description="Composite somatic spatial score from SPARCAL enhanced spatial filter">',
        f'##INFO=<ID={_INFO_NN_CLASS},Number=1,Type=String,'
        f'Description="Neural network predicted zygosity class (homozygous/heterozygous/no_variance)">',
        f'##INFO=<ID={_INFO_NN_HOMO},Number=1,Type=Float,'
        f'Description="Neural network posterior probability: homozygous (1/1)">',
        f'##INFO=<ID={_INFO_NN_HETERO},Number=1,Type=Float,'
        f'Description="Neural network posterior probability: heterozygous (0/1)">',
        f'##INFO=<ID={_INFO_NN_NOVAR},Number=1,Type=Float,'
        f'Description="Neural network posterior probability: no variant (0/0)">',
    ]


# ---------------------------------------------------------------------------
# VCF parsing
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path: str) -> Tuple[List[str], List[dict]]:
    header_lines: List[str] = []
    records:      List[dict] = []
    fh, is_proc = open_vcf(vcf_path)
    try:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                header_lines.append(line)
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            chrom_raw = parts[0]
            records.append({
                "chrom_raw":  chrom_raw,
                "chrom":      normalize_chrom(chrom_raw),
                "pos":        parts[1],
                "id":         parts[2],
                "ref":        parts[3],
                "alt":        parts[4],
                "qual":       parts[5],
                "filter":     parts[6],
                "info":       parts[7],
                "format_str": parts[8] if len(parts) > 8 else "GT",
                "samples":    parts[9:] if len(parts) > 9 else [],
            })
    finally:
        if not is_proc:
            fh.close()
    return header_lines, records


def _parse_vaf_from_record(r: dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract (depth, vaf) from a record; prefers FORMAT:AD over INFO:DP/AF."""
    dp_val:  Optional[float] = None
    vaf_val: Optional[float] = None
    if r["format_str"] and r["samples"]:
        fmt = parse_format_sample(r["format_str"], r["samples"][0])
        if fmt.get("AD", ".") not in (".", ""):
            ad = fmt["AD"].split(",")
            if len(ad) >= 2:
                try:
                    rc, ac = float(ad[0]), float(ad[1])
                    if rc + ac > 0:
                        vaf_val = ac / (rc + ac)
                except ValueError:
                    pass
        if fmt.get("DP", ".") not in (".", ""):
            try:
                dp_val = float(fmt["DP"])
            except ValueError:
                pass
    info = parse_info_str(r["info"])
    if dp_val  is None and "DP" in info:
        try:    dp_val  = float(info["DP"])
        except ValueError: pass
    if vaf_val is None and "AF" in info:
        try:    vaf_val = float(info["AF"])
        except ValueError: pass
    return dp_val, vaf_val


# ---------------------------------------------------------------------------
# Type 1 — per-spot filtered VCFs
# ---------------------------------------------------------------------------

def generate_per_spot_vcfs(
    per_spot_variants: Dict[str, Set[Tuple]],
    snv_vcf_dir: str,
    output_dir: str,
    dataset: str,
    section_id: str,
    classification: str,
    spatial_scores: Dict,
    nn_scores: Dict,
    use_chr_prefix: bool = True,
) -> int:
    """
    Write per-barcode VCFs filtered to spatially-confirmed variants.

    For each barcode: read source VCF → keep only variants in the barcode's
    passing set → inject enrichment INFO → write to output_dir/{barcode}.vcf.gz.
    All original FORMAT fields are preserved verbatim.
    """
    os.makedirs(output_dir, exist_ok=True)
    today  = datetime.date.today().strftime("%Y%m%d")
    extra  = _extra_info_header_lines()
    n_ok   = 0
    n_skip = 0

    for barcode, allowed in per_spot_variants.items():
        vcf_path = os.path.join(snv_vcf_dir, f"{barcode}.vcf.gz")
        if not os.path.exists(vcf_path):
            vcf_path = os.path.join(snv_vcf_dir, f"{barcode}.vcf")
        if not os.path.exists(vcf_path):
            n_skip += 1
            continue

        headers, records = parse_vcf(vcf_path)
        kept = [r for r in records if (r["chrom"], r["pos"], r["ref"], r["alt"]) in allowed]
        if not kept:
            n_skip += 1
            continue

        temp = os.path.join(output_dir, f"{barcode}.vcf")
        with open(temp, "w") as out:
            injected = False
            for h in headers:
                if h.startswith("##fileformat"):
                    out.write(h + "\n")
                    out.write(f"##fileDate={today}\n")
                    out.write(f"##source=SPARCAL_v{SPARCAL_VERSION}\n")
                    out.write(
                        f"##SPARCALFilter=spatial_filter_enhanced,"
                        f"dataset={dataset},section={section_id},"
                        f"classification={classification}\n"
                    )
                    for eh in extra:
                        out.write(eh + "\n")
                    injected = True
                elif h.startswith("##fileDate") or h.startswith("##source=SNVMatrixGenerator"):
                    continue
                else:
                    out.write(h + "\n")
            if not injected:
                out.write("##fileformat=VCFv4.2\n")
                out.write(f"##fileDate={today}\n")
                out.write(f"##source=SPARCAL_v{SPARCAL_VERSION}\n")
                out.write(
                    f"##SPARCALFilter=spatial_filter_enhanced,"
                    f"dataset={dataset},section={section_id},"
                    f"classification={classification}\n"
                )
                for eh in extra:
                    out.write(eh + "\n")

            for r in kept:
                key  = (r["chrom"], r["pos"], r["ref"], r["alt"])
                einfo = _enrich_info(r["info"], key, spatial_scores, nn_scores)
                cout  = format_chrom(r["chrom"], use_chr_prefix)
                samp  = r["samples"][0] if r["samples"] else "./."
                out.write(
                    f"{cout}\t{r['pos']}\t{r['id']}\t{r['ref']}\t{r['alt']}\t"
                    f"{r['qual']}\t{r['filter']}\t{einfo}\t{r['format_str']}\t{samp}\n"
                )

        out_gz = os.path.join(output_dir, f"{barcode}.vcf.gz")
        if run_cmd(f"{BGZIP} -f {temp}"):
            run_cmd(f"{BCFTOOLS} index -t {out_gz}")
        n_ok += 1

    logger.info(f"Type 1: wrote {n_ok} barcodes ({n_skip} skipped).")
    return n_ok


# ---------------------------------------------------------------------------
# Type 2 — multi-sample SNV profile VCF
# ---------------------------------------------------------------------------

def generate_snv_profile(
    per_spot_variants: Dict[str, Set[Tuple]],
    all_passing: Set[Tuple],
    snv_vcf_dir: str,
    output_vcf: str,
    dataset: str,
    section_id: str,
    quality_filter: str,
    classification: str,
    spatial_scores: Dict,
    nn_scores: Dict,
    use_chr_prefix: bool = True,
    extra_barcodes: Optional[List[str]] = None,
) -> str:
    """
    Generate a 1000-Genomes-style multi-sample SNV profile VCF.

    INFO:   SPOT_COUNT, SPOT_FREQ, DP_MEAN, AF_MEAN, RACE,
            GERMLINE_SCORE, SOMATIC_SCORE, NN_CLASS, NN_HOMO, NN_HETERO, NN_NOVAR
    FORMAT: GT only (0/1 = carrying spot, ./. = absent)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_vcf)), exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")

    active = sorted(per_spot_variants.keys())
    if extra_barcodes:
        aset = set(active)
        active += [b for b in extra_barcodes if b not in aset]
    total = len(active)

    # ---- aggregate stats per variant ----
    agg: Dict[Tuple, dict] = {
        v: {"carrying": set(), "dp_vals": [], "af_vals": [], "race": "."}
        for v in all_passing
    }
    logger.info(f"Type 2: collecting aggregate stats from {total} barcodes...")
    for barcode in active:
        vcf_path = os.path.join(snv_vcf_dir, f"{barcode}.vcf.gz")
        if not os.path.exists(vcf_path):
            vcf_path = os.path.join(snv_vcf_dir, f"{barcode}.vcf")
        if not os.path.exists(vcf_path):
            continue
        allowed = per_spot_variants.get(barcode, set())
        _, records = parse_vcf(vcf_path)
        for r in records:
            key = (r["chrom"], r["pos"], r["ref"], r["alt"])
            if key not in all_passing or key not in allowed:
                continue
            va = agg[key]
            va["carrying"].add(barcode)
            dp, vaf = _parse_vaf_from_record(r)
            if dp  is not None: va["dp_vals"].append(dp)
            if vaf is not None: va["af_vals"].append(vaf)
            if va["race"] == ".":
                info_d = parse_info_str(r["info"])
                if "RACE" in info_d:
                    va["race"] = info_d["RACE"]

    # ---- sort and write ----
    to_write = [v for v in sorted(all_passing, key=sort_variant_key) if agg[v]["carrying"]]
    n_drop = len(all_passing) - len(to_write)
    if n_drop:
        logger.warning(f"  {n_drop} variants omitted (no source VCF found for any carrying barcode).")

    temp_vcf = output_vcf.replace(".vcf.gz", ".vcf")
    with open(temp_vcf, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write(f"##fileDate={today}\n")
        out.write(f"##source=SPARCAL_v{SPARCAL_VERSION}\n")
        out.write(f"##reference=dataset:{dataset},section:{section_id}\n")
        out.write(
            f"##SPARCALFilter=spatial_filter_enhanced,"
            f"qualityFilter={quality_filter},"
            f"classification={classification},"
            f"totalBarcodes={total}\n"
        )
        # INFO definitions
        for tag, typ, desc in [
            ("SPOT_COUNT", "Integer", "Number of Visium spots carrying this variant after spatial filtering"),
            ("SPOT_FREQ",  "Float",   "Fraction of profiled barcodes carrying this variant"),
            ("DP_MEAN",    "Float",   "Mean total read depth across spots carrying this variant"),
            ("AF_MEAN",    "Float",   "Mean variant allele fraction across spots carrying this variant"),
            ("RACE",       "String",  "SNV origin: defined (Beagle/1000G) or denovo (SPARCAL classifier)"),
        ]:
            out.write(f'##INFO=<ID={tag},Number=1,Type={typ},Description="{desc}">\n')
        for eh in _extra_info_header_lines():
            out.write(eh + "\n")
        out.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,'
            'Description="Genotype: 0/1 = variant present in spot, ./. = absent or not detected">\n'
        )
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(active) + "\n"
        )

        for v in to_write:
            chrom, pos, ref, alt = v
            va = agg[v]
            carrying   = va["carrying"]
            spot_count = len(carrying)
            spot_freq  = spot_count / total if total else 0.0
            dp_vals, af_vals = va["dp_vals"], va["af_vals"]
            dp_str = f"{sum(dp_vals)/len(dp_vals):.2f}" if dp_vals else "."
            af_str = f"{sum(af_vals)/len(af_vals):.4f}" if af_vals else "."

            info_d: dict = {
                "SPOT_COUNT": str(spot_count),
                "SPOT_FREQ":  f"{spot_freq:.6f}",
                "DP_MEAN":    dp_str,
                "AF_MEAN":    af_str,
                "RACE":       va["race"],
            }
            sp = spatial_scores.get(v)
            if sp: info_d.update(sp)
            nn = nn_scores.get(v)
            if nn: info_d.update(nn)

            gt_cols = ["0/1" if bc in carrying else "./." for bc in active]
            cout = format_chrom(chrom, use_chr_prefix)
            out.write(
                f"{cout}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t"
                f"{build_info_str(info_d)}\tGT\t" + "\t".join(gt_cols) + "\n"
            )

    out_gz = output_vcf
    if run_cmd(f"{BGZIP} -f {temp_vcf}"):
        run_cmd(f"{BCFTOOLS} index -t {out_gz}")
        logger.info(
            f"Type 2: written to {out_gz} "
            f"({len(to_write)} variants, {total} barcodes)"
        )
    else:
        logger.warning(f"Compression failed; plain VCF at {temp_vcf}")
    return out_gz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SPARCAL post-processing: generate final per-spot VCFs (Type 1) and "
            "multi-sample SNV profile VCF (Type 2). Handles both basic and enhanced "
            "spatial filter output structures. Pass the base output directory of "
            "the spatial filter step as --spatial_filter_dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASET_SUBDIRS) + [k.lower() for k in DATASET_SUBDIRS],
    )
    parser.add_argument("--section_id", required=True)
    parser.add_argument(
        "--spatial_filter_dir", required=True,
        help=(
            "Base output directory of the spatial filter step. "
            "For the enhanced filter this is typically: "
            "{project_dir}/data/{dataset}/{section_id}/spatial_filter_purity/{quality_filter}. "
            "For the basic filter: {project_dir}/data/{dataset}/{section_id}/"
            "spatial_analysis/{quality_filter}/filtered_snvs."
        ),
    )

    # Optional
    parser.add_argument("--quality_filter", default="baseQ0mapQ0")
    parser.add_argument("--project_dir", default=PROJECT_DIR)
    parser.add_argument(
        "--classification", default="somatic",
        choices=VALID_CLASSIFICATIONS,
        help="Which variant classification to emit (enhanced filter only; ignored for basic).",
    )
    parser.add_argument(
        "--snv_vcf_dir",
        help="Per-barcode source VCF directory (auto-derived when absent).",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory (auto-derived when absent).",
    )
    parser.add_argument(
        "--variant_scores_file",
        help=(
            "Path to all_variant_scores.txt. "
            "Auto-derived as {spatial_filter_dir}/all_variant_scores.txt (enhanced) or "
            "{spatial_filter_dir}/../all_variant_scores.txt (basic)."
        ),
    )
    parser.add_argument(
        "--nn_predictions_vcf",
        help="Path to neural_network_predictions.vcf.gz (auto-derived when absent).",
    )
    parser.add_argument("--no_chr_prefix", action="store_true")
    parser.add_argument(
        "--all_barcodes", action="store_true",
        help="Include all tissue barcodes as columns in Type 2 (requires --barcode_list).",
    )
    parser.add_argument("--barcode_list")
    parser.add_argument("--skip_type1", action="store_true")
    parser.add_argument("--skip_type2", action="store_true")

    args = parser.parse_args()

    dataset = args.dataset.upper()
    if dataset not in DATASET_SUBDIRS:
        logger.error(f"Unknown dataset: {dataset}")
        sys.exit(1)

    subdir   = DATASET_SUBDIRS[dataset]
    data_dir = os.path.join(args.project_dir, "data", subdir, args.section_id)

    snv_vcf_dir = args.snv_vcf_dir or os.path.join(
        data_dir, "output_VCFs", "BAM_filtered", args.quality_filter, "snv_vcf"
    )
    output_dir = args.output_dir or os.path.join(
        data_dir, "final_output", args.quality_filter
    )
    nn_vcf_path = args.nn_predictions_vcf or os.path.join(
        data_dir, "output_VCFs", "Classifier", args.quality_filter,
        "results", "neural_network_predictions.vcf.gz",
    )
    use_chr = not args.no_chr_prefix

    # ---- detect structure ----
    structure = detect_filter_structure(args.spatial_filter_dir)
    logger.info("=" * 60)
    logger.info("SPARCAL final VCF generation")
    logger.info(f"  Dataset        : {dataset}  section {args.section_id}")
    logger.info(f"  Filter output  : {args.spatial_filter_dir}  [{structure}]")
    logger.info(f"  Classification : {args.classification}")
    logger.info(f"  Per-spot VCFs  : {snv_vcf_dir}")
    logger.info(f"  Output dir     : {output_dir}")
    logger.info("=" * 60)

    if structure == "unknown":
        logger.error(
            f"Cannot recognise spatial filter output structure in: {args.spatial_filter_dir}\n"
            f"  Expected either a 'somatic/' subdirectory (enhanced filter)\n"
            f"  or an 'all_filtered_variants.txt' file (basic filter)."
        )
        sys.exit(1)

    if not os.path.isdir(snv_vcf_dir):
        logger.error(f"snv_vcf_dir does not exist: {snv_vcf_dir}")
        sys.exit(1)
    if args.all_barcodes and not args.barcode_list:
        logger.error("--all_barcodes requires --barcode_list.")
        sys.exit(1)

    # ---- load variant data ----
    if structure == "enhanced":
        per_spot    = load_per_spot_enhanced(args.spatial_filter_dir, args.classification)
        all_passing = load_all_passing_enhanced(args.spatial_filter_dir, args.classification)
        # spatial scores file is in the base dir directly
        scores_file = args.variant_scores_file or os.path.join(
            args.spatial_filter_dir, "all_variant_scores.txt"
        )
    else:  # basic
        per_spot    = load_per_spot_basic(args.spatial_filter_dir)
        all_passing = load_all_passing_basic(args.spatial_filter_dir)
        # basic filter: scores file one level up from filtered_snvs/
        scores_file = args.variant_scores_file or os.path.join(
            os.path.dirname(args.spatial_filter_dir.rstrip("/")),
            "all_variant_scores.txt",
        )

    if not all_passing:
        logger.error("No passing variants found. Exiting.")
        sys.exit(1)

    # Fallback: if per-barcode files were empty, re-derive from source VCFs
    if not per_spot:
        logger.warning(
            "No per-barcode files found in spatial filter output. "
            "Falling back to re-deriving per-spot assignments from source VCFs."
        )
        per_spot = derive_per_spot_from_vcfs(all_passing, snv_vcf_dir)

    if not per_spot:
        logger.error("Per-spot variant assignment is empty after all attempts. Exiting.")
        sys.exit(1)

    logger.info(
        f"  Passing variants : {len(all_passing)}"
    )
    logger.info(f"  Barcodes with passing variants: {len(per_spot)}")

    # ---- enrichment data ----
    spatial_scores = load_spatial_scores(scores_file)
    nn_scores      = load_nn_scores(nn_vcf_path)

    extra_barcodes = None
    if args.all_barcodes and args.barcode_list:
        extra_barcodes = []
        with open(args.barcode_list) as fh:
            for line in fh:
                bc = line.strip().split()[0]
                if bc:
                    extra_barcodes.append(bc)

    # ---- output name suffix for classification ----
    cls_suffix = f"_{args.classification}" if args.classification != "somatic" else ""

    # ---- Type 1 ----
    if not args.skip_type1:
        logger.info("--- Type 1: per-spot filtered VCFs ---")
        type1_dir = os.path.join(output_dir, f"final_vcf_per_spot{cls_suffix}")
        generate_per_spot_vcfs(
            per_spot, snv_vcf_dir, type1_dir,
            dataset, args.section_id, args.classification,
            spatial_scores, nn_scores, use_chr,
        )

    # ---- Type 2 ----
    if not args.skip_type2:
        logger.info("--- Type 2: SNV profile VCF ---")
        type2_vcf = os.path.join(output_dir, f"final_snv_profile{cls_suffix}.vcf.gz")
        generate_snv_profile(
            per_spot, all_passing, snv_vcf_dir, type2_vcf,
            dataset, args.section_id, args.quality_filter, args.classification,
            spatial_scores, nn_scores, use_chr, extra_barcodes,
        )

    logger.info("=" * 60)
    logger.info("Done.")
    if not args.skip_type1:
        logger.info(f"  Type 1  →  {os.path.join(output_dir, f'final_vcf_per_spot{cls_suffix}')}/")
    if not args.skip_type2:
        logger.info(f"  Type 2  →  {os.path.join(output_dir, f'final_snv_profile{cls_suffix}.vcf.gz')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()