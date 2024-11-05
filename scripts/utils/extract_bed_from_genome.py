import pysam

# Path to your FASTA file
fasta_path = '/data/maiziezhou_lab/Softwares/refdata-gex-mm10-2020-A/refdata-gex-mm10-2020-A/fasta/genome.fa'
# Path for the output BED file
bed_path = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/reference/refdata-gex-mm10-2020-A.bed'

# Open the FASTA file
fasta_file = pysam.FastaFile(fasta_path)

# Open the BED file for writing
with open(bed_path, 'w') as bed_file:
    # Iterate over each reference in the FASTA file
    for ref in fasta_file.references:
        # Get the length of the reference
        length = fasta_file.get_reference_length(ref)
        # BED format start coordinates are 0-based, but we're asked to start from 1,
        # so we adjust the start position to 0 to follow BED conventions, if considering 1-based as the input instruction
        start_pos = 0
        end_pos = length
        # Write the chromosome, start position, and end position to the BED file
        bed_file.write(f"{ref}\t{start_pos}\t{end_pos}\n")

# Close the FASTA file
fasta_file.close()

print(f"BED file has been generated at: {bed_path}")
