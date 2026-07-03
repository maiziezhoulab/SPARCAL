import sys
import os

def load_reference_snvs(reference_txt):
    """Load the reference SNVs into a set for quick lookup."""
    reference_snvs = set()
    with open(reference_txt, 'r') as txtfile:
        # Skip the header
        next(txtfile)
        for line in txtfile:
            columns = line.strip().split('\t')
            chrom = columns[0]
            pos = columns[1]
            ref = columns[2]
            key = (chrom, pos, ref)
            reference_snvs.add(key)
    return reference_snvs

def filter_vcf(vcf_file, reference_snvs, output_file):
    """Filter the VCF file based on the reference SNVs."""
    with open(vcf_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith("#"):
                outfile.write(line)
            else:
                parts = line.strip().split("\t")
                chrom = parts[0]
                pos = parts[1]
                ref = parts[3]
                if (chrom, pos, ref) in reference_snvs:
                    outfile.write(line)

def main(reference_txt, vcf_directory, output_directory):
    """Main function to filter VCF files based on the reference SNVs."""
    # Load reference SNVs
    reference_snvs = load_reference_snvs(reference_txt)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process each VCF file in the directory
    for filename in os.listdir(vcf_directory):
        if filename.endswith(".vcf"):
            vcf_file = os.path.join(vcf_directory, filename)
            output_file = os.path.join(output_directory, filename)
            filter_vcf(vcf_file, reference_snvs, output_file)
            # print(f"Filtered {vcf_file} and saved to {output_file}")

if __name__ == "__main__":
    reference_txt = sys.argv[1]
    vcf_directory = sys.argv[2]
    output_directory = sys.argv[3]
    main(reference_txt, vcf_directory, output_directory)
