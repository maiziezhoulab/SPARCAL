"""P(in COSMIC | 1000G allele), stratified by panel AF, on chr1.
Measures how much common germline polymorphism the COSMIC Genome Screens Mutant
catalog contains. Run after extracting the two key files (see RESULT.md)."""
import sys
panel={}
for line in open(sys.argv[1]):
    k,af=line.rstrip("\n").split("\t")
    try: panel[k]=float(af.split(",")[0])
    except ValueError: panel[k]=0.0
tot=common=rare=0
for line in open(sys.argv[2]):
    k=line.strip(); tot+=1
    af=panel.get(k)
    if af is None: continue
    common+=af>=0.01; rare+=af<0.01
pc=sum(1 for v in panel.values() if v>=0.01)
print(f"COSMIC chr1 SNV alleles {tot:,}; 1000G-matched {common+rare:,} ({100*(common+rare)/tot:.2f}%)")
print(f"P(COSMIC|common 1000G) {100*common/pc:.2f}%   P(COSMIC|rare) {100*rare/(len(panel)-pc):.3f}%"
      f"   enrichment {(common/pc)/(rare/(len(panel)-pc)):.1f}x")
