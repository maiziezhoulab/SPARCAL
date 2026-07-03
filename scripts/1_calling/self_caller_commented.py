from pysam import FastaFile
from collections import OrderedDict 
import pandas as pd
import numpy as np
import pysam
from argparse import ArgumentParser

# Helper function to parse chromosome name and convert it into integer index if possible.
def parse_chrName(chromo_name):
	chromo_name = chromo_name.split("chr")[1]  # Removes 'chr' prefix to get the chromosome number/name
	try:
		int(chromo_name)
		return int(chromo_name) - 1  # Converts chromosome number to index (0-based)
	except ValueError:
		return chromo_name  # Returns the name as is if it is not a number

# Binary search function to find a position in a sorted list (used to search for positions efficiently).
def binary_search(lst, low, high, x):
	if high >= low:
		mid = (high + low) // 2  # Find the mid-point index
		if lst[mid] == x:
			return mid  # Return if the position matches
		elif lst[mid] > x:
			return binary_search(lst, low, mid - 1, x)  # Search in the lower half
		else:
			return binary_search(lst, mid + 1, high, x)  # Search in the upper half
	else:
		return -1  # Return -1 if not found

# Function to compare differences between read and reference sequences
def compare(pos, seqread, seqref, po, ref, re, qn, q):
	seqr = seqread.capitalize()  # Capitalize read sequence
	seqre = seqref.capitalize()  # Capitalize reference sequence
	dif = [i for i, (a1, a2) in enumerate(zip(seqr, seqre)) if a1 != a2]  # Find mismatches
	for i in dif:
		pos.append(int(po + i))  # Record the position of the mismatch
		ref.append(seqref[i:i + 1])  # Reference base
		re.append(seqread[i:i + 1])  # Read base
		qn.append(q)  # Read name (query name)

# Function to find duplicate elements in a list and return their frequency and indices.
def getDuplicatesWithInfo(listOfElems):
	dictOfElems = dict()
	index = 0
	for elem in listOfElems:
		if elem in dictOfElems:
			dictOfElems[elem][0] += 1  # Increment count if the element is already present
			dictOfElems[elem][1].append(index)  # Append the index of the element
		else:
			dictOfElems[elem] = [1, [index]]  # Add a new entry if it's the first occurrence
		index += 1    
	return dictOfElems

# Function to extract chromosome regions from a BED file
def extract_chromosome_regions(bed_file_path, chromo_number):
	region_start = []
	region_end = []

	# Open the BED file and read the regions
	with open(bed_file_path, 'r') as file:
		for line in file:
			parts = line.strip().split()  # Split the line by whitespace
			chromosome = parts[0]  # Extract chromosome name
			start = int(parts[1])  # Start position of the region
			end = int(parts[2])  # End position of the region
			
			# Only extract regions for the specified chromosome
			if chromosome == str(chromo_number):
				region_start.append(start)
				region_end.append(end)

	return region_start, region_end

# The main function for SNP calling
def gene(standard, chromo, bamfile, bedfile, header, out):
	# Load the reference genome sequence
	seqo = FastaFile(standard)
	tmp = seqo.fetch(chromo)  # Fetch the reference sequence for the target chromosome

	# Open the BAM file
	try:
		samfile = pysam.AlignmentFile(bamfile, "rb")  # Open BAM file in read-binary mode
	except OSError as e:
		print(f"Error opening BAM file: {e}")
		return

	# Lists to hold information about positions, reference bases, read sequences, and query names
	position = []
	reference = []
	readss = []
	qna = []
	chromo_name = ''

	# Iterate over reads in the BAM file
	for read in samfile.fetch(until_eof=True):
		seq = read.seq  # The sequence of the read
		pos = read.pos  # Start position of the read on the reference genome
		a = read.cigar  # CIGAR string (alignment details)
		a = list(map(list, a))  # Convert tuples in the CIGAR to lists for modification
		c = list(map(list, a))  # A copy of the CIGAR for tracking read positions
		readpo = 0  # Initialize read position offset
		chromo_name = read.reference_name  # The chromosome name of the read's reference
		
		if (read.reference_name == chromo) and (seq is not None):
			qn = read.qname  # The query name (read ID)
			
			# Loop through the CIGAR operations
			for q in range(len(a)):
				# If the CIGAR operation consumes reference positions (match, deletion, skip)
				if a[q][0] in [0, 2, 3]:
					a[q][0] = pos  # Set the reference position
					pos = pos + a[q][1]  # Update position based on the length of the operation
					a[q][1] = pos
				
				# If the CIGAR operation does not consume reference positions (insertion, clipping)
				if a[q][0] in [1, 4, 5]:
					a[q][0] = 0  # Set reference position to 0 (doesn't consume reference)
					a[q][1] = 0
				
				# Handle read coordinate changes similarly for read-consuming operations
				if c[q][0] in [0, 1, 4]:
					c[q][0] = readpo  # Set the read position
					readpo = readpo + c[q][1]  # Update read position
					c[q][1] = readpo
				
				if c[q][0] in [2, 3, 5]:
					c[q][0] = 0  # Set read position to 0 (not consuming read)
					c[q][1] = 0

				# If both reference and read segments have non-zero length
				if (a[q][1] != 0 and c[q][1] != 0):
					readseq = seq[c[q][0]:c[q][1]]  # Extract read sequence for the current segment
					refseq = tmp[a[q][0]:a[q][1]]  # Extract reference sequence for the current segment
					compare(position, readseq, refseq, a[q][0], reference, readss, qna, qn)  # Compare the two sequences

	samfile.close()  # Close the BAM file after processing

	# Combine results into columns and sort them based on positions
	result = np.column_stack((position, reference, readss, qna))
	res = sorted(result, key=lambda x: int(x[0]))

	# Filter results by BED file regions
	regions_start, regions_end = extract_chromosome_regions(bedfile, chromo)
	result_po = []
	result_ref = []
	result_re = []
	result_qn = []

	# For each variant, check if it falls within the regions specified by the BED file
	for a in range(len(res)):
		for i in range(len(regions_start)):
			if int(res[a][0]) > int(regions_start[i]) and int(res[a][0]) < int(regions_end[i]):
				result_po.append(int(res[a][0]))  # Store position
				result_ref.append(res[a][1])  # Reference base
				result_re.append(res[a][2])  # Read base
				result_qn.append(res[a][3])  # Query name



	# Remove duplicate positions from the list of variant positions (result_po)
	res = list(OrderedDict.fromkeys(result_po))

	# Reopen the BAM file for fetching reads
	samfile = pysam.AlignmentFile(bamfile, "rb")

	# Initialize lists to store new positions, reference, and read sequences
	newpo = []
	ref = []
	readsq = []

	# Iterate through all reads in the BAM file
	for read in samfile.fetch(until_eof=True):
		seq = read.seq  # The sequence of the read
		pos = read.pos  # Start position of the read on the reference genome
		a = read.cigar  # CIGAR string (alignment details)
		a = list(map(list, a))  # Convert CIGAR tuples to lists
		c = list(map(list, a))  # Another copy of the CIGAR list to track read positions
		readpo = 0  # Initialize read position offset
		
		# Check if the read is on the correct chromosome and has a sequence
		if (read.reference_name == chromo) and (seq is not None):
			qn = read.qname  # Get the read's query name (read identifier)
			
			# Iterate through the CIGAR operations
			for q in range(len(a)):
				# For operations that consume reference positions (match, deletion, skip)
				if a[q][0] in [0, 2, 3]:
					a[q][0] = pos  # Set the reference position
					pos = pos + a[q][1]  # Update the position based on the length of the operation
					a[q][1] = pos
				
				# For operations that don't consume reference positions (insertion, clipping)
				if a[q][0] in [1, 4, 5]:
					a[q][0] = 0  # Set reference-consuming operations to 0
					a[q][1] = 0
				
				# Handle read coordinates for read-consuming operations
				if c[q][0] in [0, 1, 4]:
					c[q][0] = readpo  # Set the read position
					readpo = readpo + c[q][1]  # Update the read position
					c[q][1] = readpo
				
				# For operations that don't consume read positions (e.g., deletions)
				if c[q][0] in [2, 3, 5]:
					c[q][0] = 0  # Set read-consuming operations to 0
					c[q][1] = 0

				# Check if both reference and read segments have non-zero length
				if (a[q][1] != 0 and c[q][1] != 0):
					# Extract the read and reference sequences based on positions
					readseq = seq[c[q][0]:c[q][1]]
					refseq = tmp[a[q][0]:a[q][1]]

					#====================================
					# 
					
					# This might be the time consuming part
					# 
					# 
					# ===================================
					# Compare the read sequence to the reference for each variant position
					for k in res:
						# Check if the variant position falls within the current reference range
						if k <= a[q][1] and k >= a[q][0]:
							newpo.append(k)  # Record the variant position
							# Record the corresponding base from the read sequence
							readsq.append(readseq[k - a[q][0]:k + 1 - a[q][0]])

	# Close the BAM file after processing
	samfile.close()

	# Count the number of alternate and reference alleles for each SNP position
	a = getDuplicatesWithInfo(newpo)  # Find duplicate positions and their occurrences

	# Initialize lists to store the final output data
	pos_new = []
	gt_new = []
	ref_count = []
	alt_count = []

	# Process each SNP position and count the allele occurrences
	for key, value in a.items():
		stri0 = ""  # String to track reference alleles
		stri1 = ""  # String to track alternate alleles
		cur_ref_alio_count = 0  # Counter for reference alleles
		cur_alt_alio_count = 0  # Counter for alternate alleles
		
		# Iterate through all occurrences of the current SNP position
		for i in value[1]:
			ind = binary_search(result_po, 0, len(result_po) - 1, int(key))  # Find the index of the position in result_po
			
			# Compare the reference and read bases
			if result_ref[ind].upper() != readsq[i].upper():
				stri1 += "1"  # Count as alternate allele
				cur_alt_alio_count += 1  # Increment alternate allele count
			if result_ref[ind].upper() == readsq[i].upper():
				stri0 += "0"  # Count as reference allele
				cur_ref_alio_count += 1  # Increment reference allele count
		
		# Determine the genotype based on the counts of reference and alternate alleles
		if len(stri1) != 0 and len(stri0) == 0:
			gt_new.append("1/1")  # Homozygous alternate
		else:
			gt_new.append("0/1")  # Heterozygous (reference/alternate)

		# Record the SNP position and allele counts
		pos_new.append(int(key))
		ref_count += [cur_ref_alio_count]
		alt_count += [cur_alt_alio_count]


	# Open the output file in append mode ('a') with UTF-8 encoding and no extra newlines
	with open(out, 'a', encoding='UTF8', newline='') as my_file:
		# Loop through all SNP positions stored in 'pos_new'
		for i in range(0, len(pos_new)):
			# Create an empty array of size 10 to hold the VCF format information for each SNP
			arr = np.empty(10, dtype=object)
			
			# Get the genotype for the current SNP position
			gt = gt_new[i]

			# Fill in the VCF fields in the array
			arr[0] = chromo  # Chromosome name
			arr[1] = pos_new[i] + 1  # SNP position (1-based, so add 1 to the 0-based index)
			arr[2] = "."  # ID field (dot used if no ID is assigned)
			arr[3] = result_ref[result_po.index(pos_new[i])]  # Reference base at this position
			arr[4] = result_re[result_po.index(pos_new[i])]  # Alternate base (from the read) at this position
			arr[5] = "20"  # Placeholder for quality score (adjust as needed)
			arr[6] = "PASS"  # Filter field (PASS indicates the variant passed quality checks)
			
			# INFO field: Includes the SV type (SNV) and counts of reference and alternate alleles
			arr[7] = "SVTYPE=SNV|" + str(ref_count[i]) + '|' + str(alt_count[i])
			
			# FORMAT field: Genotype and Phase Set (PS)
			arr[8] = "GT:PS"
			
			# Get the phase set (PS) value from the read name, removing prefixes and using the numeric part
			ps = result_qn[result_po.index(pos_new[i])].split('_')[0][2:]
			
			# Combine the genotype and phase set (PS) into the format field
			arr[9] = gt + ":" + ps
			
			# Write the array to the file as a tab-delimited string, with each item in 'arr' joined by tabs
			my_file.write('\t'.join(str(item) for item in arr) + '\n')


def main():
	parser = ArgumentParser(description="Script description.")
	parser.add_argument('--reference_seq')
	parser.add_argument('--chromosome', type=str)
	parser.add_argument('--bamfile')
	parser.add_argument('--bedfile')
	parser.add_argument('--header')
	parser.add_argument('--out')
	args = parser.parse_args()
	gene(args.reference_seq, args.chromosome, args.bamfile, args.bedfile, args.header, args.out)

if __name__ == "__main__":
	main()