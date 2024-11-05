#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 -d <bam_directory> [-n <num_files>]"
    echo "Options:"
    echo "  -d: Directory containing BAM files"
    echo "  -n: Number of random files to analyze (default: 5)"
    exit 1
}

# Parse command line arguments
while getopts "d:n:" opt; do
    case $opt in
        d) BAM_DIR="$OPTARG"
        ;;
        n) NUM_FILES="$OPTARG"
        ;;
        ?) usage
        ;;
    esac
done

# Check if BAM directory is provided
if [ -z "$BAM_DIR" ]; then
    echo "Error: BAM directory is required"
    usage
fi

# Set default number of files if not specified
NUM_FILES=${NUM_FILES:-5}

# Ensure directory exists
if [ ! -d "$BAM_DIR" ]; then
    echo "Error: Directory $BAM_DIR does not exist"
    exit 1
fi

echo "Analyzing non-zero read depth for $NUM_FILES random BAM files from $BAM_DIR"
echo "========================================================================"

# Get random BAM files
BAM_FILES=$(find "$BAM_DIR" -name "*.bam" | shuf -n "$NUM_FILES")

# Process each BAM file
for bam in $BAM_FILES; do
    echo -e "\nProcessing $(basename "$bam"):"
    echo "--------------------------------"
    
    # Calculate coverage metrics for non-zero regions
    echo "Coverage and depth statistics (excluding zero-depth regions):"
    samtools depth "$bam" | awk '
    {
        total_pos++
        if ($3 > 0) {
            covered_pos++
            sum += $3
            sumsq += ($3)^2
            depths[covered_pos] = $3
        }
    } 
    END {
        # Basic coverage statistics
        printf "Total positions analyzed: %d\n", total_pos
        printf "Positions with coverage: %d (%.2f%%)\n", covered_pos, (covered_pos/total_pos)*100
        
        if (covered_pos > 0) {
            # Calculate mean and standard deviation
            mean = sum/covered_pos
            variance = (sumsq/covered_pos) - (mean)^2
            stddev = sqrt(variance)
            
            # Calculate median
            asort(depths)
            if (covered_pos % 2) {
                median = depths[(covered_pos + 1)/2]
            } else {
                median = (depths[covered_pos/2] + depths[covered_pos/2 + 1])/2
            }
            
            # Calculate mode
            delete count
            max_count = 0
            for (i in depths) {
                count[depths[i]]++
                if (count[depths[i]] > max_count) {
                    max_count = count[depths[i]]
                    mode = depths[i]
                }
            }
            
            # Calculate quartiles
            q1_pos = int(covered_pos/4)
            q3_pos = int(3*covered_pos/4)
            q1 = depths[q1_pos]
            q3 = depths[q3_pos]
            iqr = q3 - q1
            
            # Print statistics
            printf "\nDepth Statistics (non-zero regions):\n"
            printf "  Mean depth: %.2f\n", mean
            printf "  Median depth: %.2f\n", median
            printf "  Mode depth: %d\n", mode
            printf "  Standard deviation: %.2f\n", stddev
            printf "  First quartile (Q1): %.2f\n", q1
            printf "  Third quartile (Q3): %.2f\n", q3
            printf "  Interquartile range (IQR): %.2f\n", iqr
            
            # Calculate skewness
            sum_cube = 0
            for (i in depths) {
                sum_cube += ((depths[i] - mean)^3)
            }
            skewness = (sum_cube/covered_pos)/(stddev^3)
            printf "  Skewness: %.2f\n", skewness
            
            # Print coverage thresholds
            printf "\nCoverage thresholds (among covered positions):\n"
            for (i in depths) {
                if (depths[i] >= 1) cov1++
                if (depths[i] >= 10) cov10++
                if (depths[i] >= 20) cov20++
                if (depths[i] >= 30) cov30++
            }
            printf "  ≥1X:  100.00%%\n"
            printf "  ≥10X: %.2f%%\n", (cov10/covered_pos)*100
            printf "  ≥20X: %.2f%%\n", (cov20/covered_pos)*100
            printf "  ≥30X: %.2f%%\n", (cov30/covered_pos)*100
        }
    }'
done