import os, gzip, csv, math
from collections import defaultdict

BASE="/data/maiziezhou_lab/leiy4/snv_calling/cosmic_amb"
SAMPLES=["p4_tumor","p6_tumor","dcis1","dcis2"]
CLASSES=["defined","upv","somatic","ambiguous"]     # germline, UPV, somatic, unresolved
XMHC=("6",28_000_000,34_000_000)                     # paper's definition, GRCh37

def read(vcf):
    """yield (chrom, pos, NS) from a plain VCF."""
    out=[]
    with open(vcf) as f:
        for line in f:
            if line[0]=="#": continue
            p=line.split("\t",8)
            ns=0
            for kv in p[7].split(";"):
                if kv.startswith("NS="):
                    ns=int(kv[3:]); break
            out.append((p[0],int(p[1]),ns))
    return out

def in_x(c,p):
    return c==XMHC[0] and XMHC[1]<=p<XMHC[2]

rows=[]
for s in SAMPLES:
    for c in CLASSES:
        d=os.path.join(BASE,f"isec_{s}_{c}")
        hit=read(os.path.join(d,"0002.vcf"))     # in COSMIC
        mis=read(os.path.join(d,"0000.vcf"))     # not in COSMIC
        tot=len(hit)+len(mis)
        hx=sum(1 for c_,p_,_ in hit if in_x(c_,p_))
        mx=sum(1 for c_,p_,_ in mis if in_x(c_,p_))
        totx=hx+mx
        rate=100*len(hit)/tot
        rate_in  = 100*hx/totx if totx else float('nan')
        rate_out = 100*(len(hit)-hx)/(tot-totx) if tot-totx else float('nan')
        rows.append(dict(sample=s, cls=c, n=tot, hits=len(hit), rate=rate,
                         n_x=totx, hits_x=hx, rate_in=rate_in, rate_out=rate_out,
                         prom=(rate_in/rate_out if rate_out else float('nan')),
                         share_var_x=100*totx/tot,
                         share_hit_x=100*hx/len(hit) if hit else float('nan'),
                         med_ns_hit=sorted(x[2] for x in hit)[len(hit)//2] if hit else 0,
                         med_ns_mis=sorted(x[2] for x in mis)[len(mis)//2] if mis else 0))

W=csv.DictWriter(open(os.path.expanduser(os.environ["S"]+"/cosmic_xmhc_table.csv"),"w"),
                 fieldnames=list(rows[0].keys())); W.writeheader(); W.writerows(rows)

lab={"defined":"germline(1KGP)","upv":"UPV","somatic":"somatic","ambiguous":"unresolved"}
print("=== COSMIC hit rate, and xMHC concentration, per class ===")
print(f"{'sample':9}{'class':16}{'N':>9}{'hits':>7}{'rate%':>8}{'|':>3}{'N_xMHC':>8}{'%var':>7}{'%hits':>7}{'rate_in%':>10}{'rate_out%':>10}{'promin.':>9}{'medNS_hit':>10}{'medNS_miss':>11}")
for r in rows:
    print(f"{r['sample']:9}{lab[r['cls']]:16}{r['n']:>9,}{r['hits']:>7,}{r['rate']:>8.3f}{'|':>3}"
          f"{r['n_x']:>8,}{r['share_var_x']:>7.2f}{r['share_hit_x']:>7.2f}"
          f"{r['rate_in']:>10.2f}{r['rate_out']:>10.3f}{r['prom']:>9.2f}"
          f"{r['med_ns_hit']:>10}{r['med_ns_mis']:>11}")
