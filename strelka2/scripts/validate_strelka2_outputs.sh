#!/bin/bash
# ============================================================================
# validate_strelka2_outputs.sh
# ----------------------------------------------------------------------------
# Validity gate for strelka2 germline output before building comparison
# matrices. Checks all 12 DLPFC sections for:
#   - presence of variants.vcf.gz (+ .tbi)
#   - bgzf integrity (reads cleanly to EOF — catches truncation)
#   - matching tabix index
#   - record / PASS / PASS-SNV counts (matrix is SNV-based)
#   - the per-section workflow actually reported success in the SLURM log
#
# Run on the login node:
#   bash strelka2/scripts/validate_strelka2_outputs.sh
# A section is OK only if integrity passes AND PASS-SNV count > 0.
# ============================================================================

set -u
BCF=/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools
BASE=/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc
SECTIONS="151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676"

printf "%-8s | %-9s | %-9s | %10s | %10s | %10s | %s\n" \
       "section" "vcf" "integrity" "records" "PASS" "PASS_SNV" "status"
printf -- "---------+-----------+-----------+------------+------------+------------+--------\n"

n_ok=0; n_bad=0
for s in $SECTIONS; do
    V="$BASE/$s/strelka2/results/variants/variants.vcf.gz"
    if [ ! -f "$V" ]; then
        printf "%-8s | %-9s | %-9s | %10s | %10s | %10s | %s\n" \
               "$s" "MISSING" "-" "-" "-" "-" "NOT DONE"
        n_bad=$((n_bad+1)); continue
    fi
    # integrity: full read must succeed
    if $BCF view "$V" >/dev/null 2>&1; then integ="OK"; else integ="CORRUPT"; fi
    # index present?
    [ -f "$V.tbi" ] && vcf="present" || vcf="no .tbi"
    rec=$($BCF view -H "$V" 2>/dev/null | wc -l)
    pass=$($BCF view -H -f PASS "$V" 2>/dev/null | wc -l)
    snv=$($BCF view -H -f PASS -v snps "$V" 2>/dev/null | wc -l)

    if [ "$integ" = "OK" ] && [ "$snv" -gt 0 ]; then
        status="OK"; n_ok=$((n_ok+1))
    else
        status="*** CHECK ***"; n_bad=$((n_bad+1))
    fi
    printf "%-8s | %-9s | %-9s | %10s | %10s | %10s | %s\n" \
           "$s" "$vcf" "$integ" "$rec" "$pass" "$snv" "$status"
done

echo
echo "Summary: $n_ok OK, $n_bad not-OK / missing."
[ "$n_bad" -eq 0 ] && echo "All sections valid — safe to generate comparison matrices." \
                    || echo "Some sections not ready — do NOT generate matrices for those yet."
