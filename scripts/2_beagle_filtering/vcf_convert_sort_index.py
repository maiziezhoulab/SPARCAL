#!/usr/bin/env python3

import argparse
import subprocess
import os
from pathlib import Path

# Environment Configuration
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
}

def setup_environment() -> dict:
    """Setup environment variables to ensure tools are accessible."""
    print("Setting up environment...")
    
    # Add apps directory to PATH
    apps_dir = PATH_CONFIG['APPS_DIR']
    os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
    
    # Add to LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    # Verify required tools are available
    required_tools = ['bgzip', 'tabix']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run(['which', tool], check=True, capture_output=True, text=True)
            print(f"Found {tool} in PATH")
        except subprocess.CalledProcessError:
            missing_tools.append(tool)
    
    if missing_tools:
        raise RuntimeError(f"Required tools not found in PATH: {', '.join(missing_tools)}\n"
                         f"Please ensure they are installed in {apps_dir}")
    
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

def convert_vcf_chromosome_format(input_vcf: str, output_vcf: str, compress: bool = True):
    """
    Convert chromosome format in VCF file from '1' to 'chr1' style.
    """
    print(f"Converting chromosome format in {input_vcf}")
    
    # Store everything in memory for sorting
    header_lines = []
    data_lines = []
    chromosomes = set()
    
    # First pass: collect headers and data
    print("Reading VCF file...")
    with open(input_vcf, 'r') as infile:
        for line in infile:
            if line.startswith('#'):
                if not line.startswith('##contig='):
                    header_lines.append(line)
            else:
                fields = line.strip().split('\t')
                # Convert chromosome name
                if not fields[0].startswith('chr'):
                    if fields[0] == 'MT':
                        fields[0] = 'chrM'
                    else:
                        fields[0] = f'chr{fields[0]}'
                chromosomes.add(fields[0])
                # Store chromosome, position, and full line for sorting
                pos = int(fields[1])
                data_lines.append((fields[0], pos, fields))

    # Sort data lines by chromosome and position
    print("Sorting data...")
    data_lines.sort(key=lambda x: (x[0], x[1]))  # Sort by chromosome then position

    # Write sorted output with proper header
    print("Writing sorted output...")
    with open(output_vcf, 'w') as outfile:
        # Write original headers except contig lines
        for line in header_lines[:-1]:
            outfile.write(line)
            
        # Write contig lines for all chromosomes we encountered
        for chrom in sorted(chromosomes):
            outfile.write(f'##contig=<ID={chrom}>\n')
            
        # Write the #CHROM line
        outfile.write(header_lines[-1])
        
        # Write sorted data lines
        for _, _, fields in data_lines:
            outfile.write('\t'.join(fields) + '\n')

    print(f"Conversion and sorting complete. Output written to {output_vcf}")
    
    if compress:
        print("Compressing and indexing the converted VCF...")
        try:
            # Compress with bgzip
            subprocess.run(['bgzip', '-f', output_vcf], check=True)
            compressed_vcf = f"{output_vcf}.gz"
            print(f"Successfully compressed: {compressed_vcf}")
            
            # Index with tabix
            print("Creating index...")
            subprocess.run(['tabix', '-f', '-p', 'vcf', compressed_vcf], check=True)
            print(f"Successfully indexed: {compressed_vcf}.tbi")
            
        except subprocess.CalledProcessError as e:
            print(f"Error during compression/indexing: {str(e)}")
            raise

def setup_environment() -> dict:
    """Setup environment variables to ensure tools are accessible."""
    print("Setting up environment...")
    
    # Add apps directory to PATH
    apps_dir = PATH_CONFIG['APPS_DIR']
    os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
    
    # Add to LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    # Verify required tools are available
    required_tools = ['bgzip', 'tabix']  # Need both tools
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run(['which', tool], check=True, capture_output=True, text=True)
            print(f"Found {tool} in PATH")
        except subprocess.CalledProcessError:
            missing_tools.append(tool)
    
    if missing_tools:
        raise RuntimeError(f"Required tools not found in PATH: {', '.join(missing_tools)}\n"
                         f"Please ensure they are installed in {apps_dir}")
    
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

def main():
    parser = argparse.ArgumentParser(
        description="Convert chromosome identifiers in VCF file from '1' to 'chr1' format"
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input VCF file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output VCF file path. If not specified, will append .chr.vcf to input name'
    )
    
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress and index the output VCF'
    )
    
    args = parser.parse_args()
    
    # Setup environment first
    try:
        env = setup_environment()
        print("\nEnvironment variables set:")
        print(f"PATH: {env['PATH']}")
        print(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
    except RuntimeError as e:
        print(f"Environment setup failed: {str(e)}")
        return
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Set up output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}.chr.vcf")
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        convert_vcf_chromosome_format(args.input, output_path, compress=not args.no_compress)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()