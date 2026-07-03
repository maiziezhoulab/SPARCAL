import gzip
import subprocess
import os

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "BGZIP": "bgzip",
    "TABIX": "tabix",
}

def setup_environment() -> dict:
    apps_dir = PATH_CONFIG['APPS_DIR']
    os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {'PATH': os.environ['PATH'], 'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']}

def infer_gt_from_pl(input_vcf: str, output_vcf: str):
    """
    Infer genotypes from PL scores for a single VCF file.
    
    Args:
        input_vcf: Path to input VCF file (.vcf.gz)
        output_vcf: Path to output VCF file (without .gz extension)
    """
    with gzip.open(input_vcf, 'rt') as f_in, open(output_vcf, 'w') as f_out:
        for line in f_in:
            # Write header lines unchanged
            if line.startswith('#'):
                # Add GT format line before the #CHROM line
                if line.startswith('#CHROM'):
                    f_out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                f_out.write(line)
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            
            # Get PL values
            format_field = fields[8]
            sample_field = fields[9]
            
            if 'PL' not in format_field:
                continue
                
            # Parse PL scores
            pl_values = [int(x) for x in sample_field.split(':')[-1].split(',')]
            
            # Map PL index to genotype
            gt_map = {0: '0/0', 1: '0/1', 2: '1/1'}
            
            # Find index of minimum PL score
            min_pl_index = pl_values.index(min(pl_values))
            
            # Get corresponding genotype
            inferred_gt = gt_map[min_pl_index]
            
            # Update FORMAT and sample fields
            new_format = 'GT:' + format_field
            new_sample = f"{inferred_gt}:" + sample_field
            
            # Update fields and write
            fields[8] = new_format
            fields[9] = new_sample
            
            # Write modified line
            f_out.write('\t'.join(fields) + '\n')

def main():
    input_vcf = "/data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr.vcf.gz"
    output_vcf_temp = "/data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr_gt.vcf"
    output_vcf = output_vcf_temp + ".gz"
    
    setup_environment()
    # Process the file
    # print("Inferring GT fields...")
    # infer_gt_from_pl(input_vcf, output_vcf_temp)
    
    # Compress the output
    print("Compressing output file...")
    subprocess.run(['bgzip', '-f', output_vcf_temp])
    
    # Index the compressed output
    print("Indexing compressed file...")
    subprocess.run(['tabix', '-p', 'vcf', output_vcf])
    
    print("Done!")

if __name__ == "__main__":
    main()


    #usage
    #input: /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr.vcf.gz
    #output: /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr_gt.vcf.gz
    #!python scripts/calling/infer_gt_from_pl.py