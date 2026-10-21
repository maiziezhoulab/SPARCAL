"""Resume one bounded update from serialized A/B, with strict cross-bundle validation."""
import argparse
from pathlib import Path
import numpy as np
from . import audit,contracts as ct,model


def advance(variant_bundle,prior_bundle,output):
    (v,s,ref,alt,pres,g,am),(p,_,bm)=audit.load_pair(variant_bundle,prior_bundle)
    k=am['iteration']+1
    if k<1 or k>3:raise ValueError('registered exchange permits next iterations 1..3')
    par=am['parameters']; ops=model.operators(ref,alt,p['valid'])
    z=model.step(v,v['base_logits'],v['weight'],p['anchor_prior'],p['valid'],ops,g,p['tumor_prior'],k,par['lambda_prior'],par['quota'],par['low_depth_budget'])
    ids=model.regions(z['tumor_prior'],s['barcode'],g,p['valid'])
    vv=dict(v); vv.update({key:z[key] for key in ('posterior','feedback_log_evidence','selected')}); vv['class_id']=z['posterior'].argmax(1).astype('uint8')
    pp=dict(p); pp.update(tumor_prior=z['tumor_prior'],spot_evidence=z['spot_evidence'],region_id=ids)
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    inputs=[ct.entry(Path(variant_bundle)/'manifest.json','previous_contract_A'),ct.entry(Path(prior_bundle)/'manifest.json','previous_contract_B'),ct.entry(Path(__file__),'exchange_code'),ct.entry(Path(audit.__file__),'strict_audit_code')]
    common=(am['section'],am['genome_build'],k,am['arm'])
    ma=ct.metadata(ct.A,*common,inputs,par,am['provenance']['calicost_participated'],optional=am['optional_evidence'])
    ct.write_a(out/'A',vv,s,ref,alt,pres,g,ma)
    mb=ct.metadata(ct.B,*common,[ct.entry(out/'A/manifest.json','updated_contract_A')]+inputs,par,am['provenance']['calicost_participated'],bm['source'],bm['optional_evidence'])
    ct.write_b(out/'B',pp,g,mb)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant-bundle',required=True,type=Path);p.add_argument('--prior-bundle',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(advance(a.variant_bundle,a.prior_bundle,a.output))

if __name__=='__main__':main()
