import os
import matplotlib.pyplot as plt

def count_snvs(vcf_file, dataset_type):
    snv_counts = {}
    try:
        with open(vcf_file, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                chrom = fields[0]
                ref = fields[3]
                alt = fields[4]
                
                if dataset_type == 'self_filtered':
                    if 'SVTYPE=SNV' in fields[7]:
                        if chrom.isdigit():
                            chrom = 'chr' + chrom
                        elif chrom in ['X', 'Y']:
                            chrom = 'chr' + chrom
                        if len(ref) == 1 and len(alt) == 1:
                            if chrom not in snv_counts:
                                snv_counts[chrom] = 0
                            snv_counts[chrom] += 1
                elif dataset_type == 'gatk':
                    if chrom.isdigit():
                        chrom = 'chr' + chrom
                    elif chrom in ['X', 'Y']:
                        chrom = 'chr' + chrom
                    if len(ref) == 1 and len(alt) == 1:
                        if chrom not in snv_counts:
                            snv_counts[chrom] = 0
                        snv_counts[chrom] += 1
    except Exception as e:
        print(f"Error processing VCF file {vcf_file}: {e}")
    return snv_counts

def read_vcf_folder(folder, dataset_type):
    vcf_counts = {}
    for vcf_file in os.listdir(folder):
        if vcf_file.endswith(".vcf") or vcf_file.endswith(".vcf.gz"):
            file_path = os.path.join(folder, vcf_file)
            snv_counts = count_snvs(file_path, dataset_type)
            vcf_counts[vcf_file] = snv_counts
    return vcf_counts

def plot_histograms(vcf_counts1, vcf_counts2, output_dir):
    common_files = set(vcf_counts1.keys()).intersection(vcf_counts2.keys())
    
    chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    
    for chrom in chromosomes:
        snv_self = [vcf_counts1[vcf_file].get(chrom, 0) for vcf_file in common_files]
        snv_gatk = [vcf_counts2[vcf_file].get(chrom, 0) for vcf_file in common_files]

        plt.figure(figsize=(10, 6))
        
        max_snv = max(snv_self + snv_gatk)
        bins = range(0, max_snv + 2, 1)
        
        plt.hist(snv_self, bins=bins, alpha=0.5, label='Self Filtered', color='blue', edgecolor='black')
        plt.hist(snv_gatk, bins=bins, alpha=0.5, label='GATK', color='orange', edgecolor='black')
        
        plt.xlabel('Number of SNVs')
        plt.ylabel('Frequency')
        plt.title(f'SNV Counts Distribution on {chrom}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'snv_histogram_{chrom}.png'))
        plt.close()

def main(folder1, folder2, output_dir):
    print("Processing GATK result")
    vcf_counts1 = read_vcf_folder(folder1, 'gatk')
    print("Processing self filtered result")
    vcf_counts2 = read_vcf_folder(folder2, 'self_filtered')
    
    plot_histograms(vcf_counts1, vcf_counts2, output_dir)

if __name__ == "__main__":
    root = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/151509/'
    folder1 = os.path.join(root, 'output_VCFs/gatk/0')
    folder2 = os.path.join(root, 'output_VCFs/self_filtered/1')
    output_dir = os.path.join(root, 'results/figures/histogram')
    os.makedirs(output_dir, exist_ok=True)
    main(folder1, folder2, output_dir)
