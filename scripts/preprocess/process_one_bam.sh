# Specify the path to the BAM file
bam_file=$1
filename=$(basename $1 .bam)
outdir=$2/process_by_batch/$filename
mkdir -p $outdir
rm $outdir/*.sam
# Iterate over each read in the BAM file
samtools view -h "$bam_file" | while IFS=$'\t' read -r -a fields; do
for field in "${fields[@]}"; do
    if [[ $field == CB:Z:* ]]; then
        cb_tag="${field#CB:Z:}"
        #echo "$cb_tag"

        # Join the array using tabs
        joined=$(printf "\t%s" "${fields[@]}")
        joined=${joined:1} # Remove leading tab

        # Write to a text file
        echo "$joined" >> $outdir/${cb_tag}.sam


        break
    fi
done
done


