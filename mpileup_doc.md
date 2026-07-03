DESCRIPTION
Generate text pileup output for one or multiple BAM files. Each input file produces a separate group of pileup columns in the output.

Note that there are two orthogonal ways to specify locations in the input file; via -r region and -l file. The former uses (and requires) an index to do random access while the latter streams through the file contents filtering out the specified regions, requiring no index. The two may be used in conjunction. For example a BED file containing locations of genes in chromosome 20 could be specified using -r 20 -l chr20.bed, meaning that the index is used to find chromosome 20 and then it is filtered for the regions listed in the bed file.

Unmapped reads are not considered and are always discarded. By default secondary alignments, QC failures and duplicate reads will be omitted, along with low quality bases and some reads in high depth regions. See the --ff, -Q and -d options for changing this.


Pileup Format
Pileup format consists of TAB-separated lines, with each line representing the pileup of reads at a single genomic position.
Several columns contain numeric quality values encoded as individual ASCII characters. Each character can range from “!” to “~” and is decoded by taking its ASCII value and subtracting 33; e.g., “A” encodes the numeric value 32.

The first three columns give the position and reference:

Chromosome name.
1-based position on the chromosome.
Reference base at this position (this will be “N” on all lines if -f/--fasta-ref has not been used).
The remaining columns show the pileup data, and are repeated for each input BAM file specified:

Number of reads covering this position.
Read bases. This encodes information on matches, mismatches, indels, strand, mapping quality, and starts and ends of reads.
For each read covering the position, this column contains:

If this is the first position covered by the read, a “^” character followed by the alignment's mapping quality encoded as an ASCII character.
A single character indicating the read base and the strand to which the read has been mapped:
Forward	Reverse	Meaning
. dot	, comma	Base matches the reference base
ACGTN	acgtn	Base is a mismatch to the reference base
>	<	Reference skip (due to CIGAR “N”)
*	*/#	Deletion of the reference base (CIGAR “D”)
Deleted bases are shown as “*” on both strands unless --reverse-del is used, in which case they are shown as “#” on the reverse strand.

If there is an insertion after this read base, text matching “\+[0-9]+[ACGTNacgtn*#]+”: a “+” character followed by an integer giving the length of the insertion and then the inserted sequence. Pads are shown as “*” unless --reverse-del is used, in which case pads on the reverse strand will be shown as “#”.
If there is a deletion after this read base, text matching “-[0-9]+[ACGTNacgtn]+”: a “-” character followed by the deleted reference bases represented similarly. (Subsequent pileup lines will contain “*” for this read indicating the deleted bases.)
If this is the last position covered by the read, a “$” character.
Base qualities, encoded as ASCII characters.
Alignment mapping qualities, encoded as ASCII characters. (Column only present when -s/--output-MQ is used.)
Comma-separated 1-based positions within the alignments, in the orientation shown in the input file. E.g., 5 indicates that it is the fifth base of the corresponding read that is mapped to this genomic position. (Column only present when -O/--output-BP is used.)
Additional comma-separated read field columns, as selected via --output-extra. The fields selected appear in the same order as in SAM: QNAME, FLAG, RNAME, POS, MAPQ (displayed numerically), RNEXT, PNEXT, followed by RLEN for unclipped read length.
Comma-separated 1-based positions within the alignments, in 5' to 3' orientation. E.g., 5 indicates that it is the fifth base of the corresponding read as produced by the sequencing instrument, that is mapped to this genomic position. (Column only present when --output-BP-5 is used.)

Additional read tag field columns, as selected via --output-extra. These columns are formatted as determined by --output-sep and --output-empty (comma-separated by default), and appear in the same order as the tags are given in --output-extra.
Any output column that would be empty, such as a tag which is not present or the filtered sequence depth is zero, is reported as "*". This ensures a consistent number of columns across all reported positions.


OPTIONS
-6, --illumina1.3+
Assume the quality is in the Illumina 1.3+ encoding.

-A, --count-orphans
Do not skip anomalous read pairs in variant calling. Anomalous read pairs are those marked in the FLAG field as paired in sequencing but without the properly-paired flag set.

-b, --bam-list FILE
List of input BAM files, one file per line [null]

-B, --no-BAQ
Disable base alignment quality (BAQ) computation. See BAQ below.

-C, --adjust-MQ INT
Coefficient for downgrading mapping quality for reads containing excessive mismatches. Mismatches are counted as a proportion of the number of aligned bases ("M", "X" or "=" CIGAR operations), along with their quality, to derive an upper-bound of the mapping quality. Original mapping qualities lower than this are left intact, while higher ones are capped at the new adjusted score.

The exact formula is complex and likely tuned to specific instruments and specific alignment tools, so this option is disabled by default (indicated as having a zero value). Variables in the formulae and their meaning are defined below.


Variable	Meaning / formula
M	The number of matching CIGAR bases (operation "M", "X" or "=").
X	The number of substitutions with quality >= 13.
SubQ	The summed quality of substitution bases included in X, capped at a maximum of quality 33 per mismatching base.
ClipQ	The summed quality of soft-clipped or hard-clipped bases. This has no minimum or maximum quality threshold per base. For hard-clipped bases the per-base quality is taken as 13.
T	SubQ - 10 * log10(M^X / X!) + ClipQ/5
Cap	MAX(0, INT * sqrt((INT - T) / INT))
Some notes on the impact of this.


As the number of mismatches increases, the mapping quality cap reduces, eventually resulting in discarded alignments.

High quality mismatches reduces the cap faster than low quality mismatches.

The starting INT value also acts as a hard cap on mapping quality, even when zero mismatches are observed.

Indels have no impact on the mapping quality.
The intent of this option is to work around aligners that compute a mapping quality using a local alignment without having any regard to the degree of clipping required or consideration of potential contamination or large scale insertions with respect to the reference. A record may align uniquely and have no close second match, but having a high number of mismatches may still imply that the reference is not the correct site.

However we do not recommend use of this parameter unless you fully understand the impact of it and have determined that it is appropriate for your sequencing technology.


-d, --max-depth INT
At a position, read maximally INT reads per input file. Setting this limit reduces the amount of memory and time needed to process regions with very high coverage. Passing zero for this option sets it to the highest possible value, effectively removing the depth limit. [8000]

Note that up to release 1.8, samtools would enforce a minimum value for this option. This no longer happens and the limit is set exactly as specified.

-E, --redo-BAQ
Recalculate BAQ on the fly, ignore existing BQ tags. See BAQ below.

-f, --fasta-ref FILE
The faidx-indexed reference file in the FASTA format. The file can be optionally compressed by bgzip. [null]

Supplying a reference file will enable base alignment quality calculation for all reads aligned to a reference in the file. See BAQ below.

-G, --exclude-RG FILE
Exclude reads from read groups listed in FILE (one @RG-ID per line)

-l, --positions FILE
BED or position list file containing a list of regions or sites where pileup or BCF should be generated. Position list files contain two columns (chromosome and position) and start counting from 1. BED files contain at least 3 columns (chromosome, start and end position) and are 0-based half-open.
While it is possible to mix both position-list and BED coordinates in the same file, this is strongly ill advised due to the differing coordinate systems. [null]

-q, --min-MQ INT
Minimum mapping quality for an alignment to be used [0]

-Q, --min-BQ INT
Minimum base quality for a base to be considered. [13]

Note base-quality 0 is used as a filtering mechanism for overlap removal which marks bases as having quality zero and lets the base quality filter remove them. Hence using --min-BQ 0 will make the overlapping bases reappear, albeit with quality zero.

-r, --region STR
Only generate pileup in region. Requires the BAM files to be indexed. If used in conjunction with -l then considers the intersection of the two requests. STR [all sites]

-R, --ignore-RG
Ignore RG tags. Treat all reads in one BAM as one sample.

--rf, --incl-flags STR|INT
Required flags: only include reads with any of the mask bits set [null]. Note this is implemented as a filter-out rule, rejecting reads that have none of the mask bits set. Hence this does not override the --excl-flags option.

--ff, --excl-flags STR|INT
Filter flags: skip reads with any of the mask bits set. This defaults to SECONDARY,QCFAIL,DUP. The option is not accumulative, so specifying e.g. --ff QCFAIL will reenable output of secondary and duplicate alignments. Note this does not override the --incl-flags option.

-x, --ignore-overlaps-removal, --disable-overlap-removal
Overlap detection and removal is enabled by default. This option turns it off.

When enabled, where the ends of a read-pair overlap the overlapping region will have one base selected and the duplicate base nullified by setting its phred score to zero. It will then be discarded by the --min-BQ option unless this is zero.

The quality values of the retained base within an overlap will be the summation of the two bases if they agree, or 0.8 times the higher of the two bases if they disagree, with the base nucleotide also being the higher confident call.

-X
Include customized index file as a part of arguments. See EXAMPLES section for sample of usage.


Output Options:

-o, --output FILE
Write pileup output to FILE, rather than the default of standard output.


-O, --output-BP
Output base positions on reads in orientation listed in the SAM file (left to right).

--output-BP-5
Output base positions on reads in their original 5' to 3' orientation.

-s, --output-MQ
Output mapping qualities encoded as ASCII characters.

--output-QNAME
Output an extra column containing comma-separated read names. Equivalent to --output-extra QNAME.

--output-extra STR
Output extra columns containing comma-separated values of read fields or read tags. The names of the selected fields have to be provided as they are described in the SAM Specification (pag. 6) and will be output by the mpileup command in the same order as in the document (i.e. QNAME, FLAG, RNAME,...) The names are case sensitive. Currently, only the following fields are supported:

QNAME, FLAG, RNAME, POS, MAPQ, RNEXT, PNEXT, RLEN

Anything that is not on this list is treated as a potential tag, although only two character tags are accepted. In the mpileup output, tag columns are displayed in the order they were provided by the user in the command line. Field and tag names have to be provided in a comma-separated string to the mpileup command. Tags with type B (byte array) type are not supported. An absent or unsupported tag will be listed as "*". E.g.

samtools mpileup --output-extra FLAG,QNAME,RG,NM in.bam

will display four extra columns in the mpileup output, the first being a list of comma-separated read names, followed by a list of flag values, a list of RG tag values and a list of NM tag values. Field values are always displayed before tag values.

--output-sep CHAR
Specify a different separator character for tag value lists, when those values might contain one or more commas (,), which is the default list separator. This option only affects columns for two-letter tags like NM; standard fields like FLAG or QNAME will always be separated by commas.

--output-empty CHAR
Specify a different 'no value' character for tag list entries corresponding to reads that don't have a tag requested with the --output-extra option. The default is *.

This option only applies to rows that have at least one read in the pileup, and only to columns for two-letter tags. Columns for empty rows will always be printed as *.


-M, --output-mods
Adds base modification markup into the sequence column. This uses the Mm and Ml auxiliary tags (or their uppercase equivalents). Any base in the sequence output may be followed by a series of strand code quality strings enclosed within square brackets where strand is "+" or "-", code is a single character (such as "m" or "h") or a ChEBI numeric in parentheses, and quality is an optional numeric quality value. For example a "C" base with possible 5mC and 5hmC base modification may be reported as "C[+m179+h40]".

Quality values are from 0 to 255 inclusive, representing a linear scale of probability 0.0 to 1.0 in 1/256ths increments. If quality values are absent (no Ml tag) these are omitted, giving an example string of "C[+m+h]".

Note the base modifications may be identified on the reverse strand, either due to the native ability for this detection by the sequencing instrument or by the sequence subsequently being reverse complemented. This can lead to modification codes, such as "m" meaning 5mC, being shown for their complementary bases, such as "G[-m50]".

When --output-mods is selected base modifications can appear on any base in the sequence output, including during insertions. This may make parsing the string more complex, so also see the --no-output-ins-mods and --no-output-ins options to simplify this process.


--no-output-ins
Do not output the inserted bases in the sequence column. Usually this is reported as "+length sequence", but with this option it becomes simply "+length". For example an insertion of AGT in a pileup column changes from "CCC+3AGTGCC" to "CCC+3GCC".

Specifying this option twice also removes the "+length" portion, changing the example above to "CCCGCC".

The purpose of this change is to simplify parsing using basic regular expressions, which traditionally cannot perform counting operations. It is particularly beneficial when used in conjunction with --output-mods as the syntax of the inserted sequence is adjusted to also report possible base modifications, but see also --no-output-ins-mods as an alternative.


--no-output-ins-mods
Outputs the inserted bases in the sequence, but excluding any base modifications. This only affects output when --output-mods is also used.


--no-output-del
Do not output deleted reference bases in the sequence column. Normally this is reported as "+length sequence", but with this option it becomes simply "+length". For example an deletion of 3 unknown bases (due to no reference being specified) would normally be seen in a column as e.g. "CCC-3NNNGCC", but will be reported as "CCC-3GCC" with this option.

Specifying this option twice also removes the "-length" portion, changing the example above to "CCCGCC".

The purpose of this change is to simplify parsing using basic regular expressions, which traditionally cannot perform counting operations. See also --no-output-ins.


--no-output-ends
Removes the “^” (with mapping quality) and “$” markup from the sequence column.


--reverse-del
Mark the deletions on the reverse strand with the character #, instead of the usual *.

-a
Output all positions, including those with zero depth.

-a -a, -aa
Output absolutely all positions, including unused reference sequences. Note that when used in conjunction with a BED file the -a option may sometimes operate as if -aa was specified if the reference sequence has coverage outside of the region specified in the BED file.

BAQ (Base Alignment Quality)

BAQ is the Phred-scaled probability of a read base being misaligned. It greatly helps to reduce false SNPs caused by misalignments. BAQ is calculated using the probabilistic realignment method described in the paper “Improving SNP discovery by base alignment quality”, Heng Li, Bioinformatics, Volume 27, Issue 8 <https://doi.org/10.1093/bioinformatics/btr076>

BAQ is applied to modify quality values before the -Q filtering happens and before the choice of which sequence to retain when removing overlaps.

BAQ is turned on when a reference file is supplied using the -f option. To disable it, use the -B option.

It is possible to store precalculated BAQ values in a SAM BQ:Z tag. Samtools mpileup will use the precalculated values if it finds them. The -E option can be used to make it ignore the contents of the BQ:Z tag and force it to recalculate the BAQ scores by making a new alignment.