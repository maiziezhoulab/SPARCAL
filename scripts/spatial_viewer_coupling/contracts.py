"""Strict, immutable evidence bundles. No evaluator or label dependencies."""
from __future__ import annotations
import hashlib, json, re, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path('/data/maiziezhou_lab/leiy4/snv_calling')
CODE = ROOT/'scripts/spatial_viewer_coupling'
OUT = ROOT/'data/spatial_viewer_coupling_2026-09-16'
STATES = ('artifact','germline_het','germline_hom_alt','somatic_subclonal','somatic_clonal')
A = 'SPARCAL_variant_evidence_v1'
B = 'SPARCAL_spatial_prior_v1'
SOURCES = {'coverage-only prior','phase-aware posterior','none','user-curated'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def entry(path, role):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=sha(p),role=role)

def dump(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

def provenance(calicost=False, mode='bounded_frozen_multistate'):
    return dict(gt_free=True,gt_used=False,evidence_mode=mode,calicost_participated=bool(calicost),
        calicost_in_sample_descriptive_evidence=bool(calicost),independent_validation=False,input_version_independent=True)

def array_hash(arrays):
    h=hashlib.sha256()
    for k,v in sorted(arrays.items()):
        a=np.ascontiguousarray(v); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def checked_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def _require(condition,msg):
    if not condition: raise ValueError(msg)

def _fields(d, keys, name):
    _require(set(d)==set(keys),f'{name} fields mismatch: missing={set(keys)-set(d)}, extra={set(d)-set(keys)}')

def _arr(x,dtype,shape,name):
    _require(x.dtype==np.dtype(dtype) and x.shape==shape,f'{name}: dtype/shape {x.dtype}/{x.shape}')

def _strings(x,n,name):
    _require(x.dtype.kind=='U' and x.shape==(n,) and all(len(v)>0 for v in x),f'{name}: strings')

def csr_check(x,shape,counts=True):
    _require(sparse.isspmatrix_csr(x) and x.shape==shape and x.has_canonical_format,'invalid CSR')
    _require(x.dtype==np.dtype('uint32' if counts else 'uint8'),'CSR dtype')
    _require(np.all(x.data>0),'explicit zero/negative CSR entries')
    if not counts: _require(np.all(x.data==1),'binary CSR')

def validate_a(v,s,ref,alt,presence,graph):
    _fields(v,'variant_key chrom pos ref_allele alt_allele race posterior information reliability weight class_id selected eligible qc_flags base_logits feedback_log_evidence channels'.split(),'A variants')
    _fields(s,'barcode xy umi ad_available'.split(),'A spots')
    nv=len(v['variant_key']); ns=len(s['barcode'])
    for k in ('variant_key','chrom','ref_allele','alt_allele','race'): _strings(v[k],nv,k)
    _require(len(set(v['variant_key']))==nv,'duplicate allele keys')
    _arr(v['pos'],'int64',(nv,),'pos'); _require(np.all(v['pos']>0),'position')
    keys=np.array([f'{c}_{p}_{r}_{a}' for c,p,r,a in zip(v['chrom'],v['pos'],v['ref_allele'],v['alt_allele'])])
    _require(np.array_equal(keys,v['variant_key']),'allele-key mismatch')
    _require(all(r in 'ACGT' and a in 'ACGT' and len(r)==len(a)==1 and r!=a for r,a in zip(v['ref_allele'],v['alt_allele'])),'non-SNP allele')
    for k in ('posterior','base_logits','channels','feedback_log_evidence'):
        shape={'posterior':(nv,5),'base_logits':(nv,5),'channels':(nv,3,5),'feedback_log_evidence':(nv,)}[k]
        _arr(v[k],'float64',shape,k); _require(np.isfinite(v[k]).all(),f'nonfinite {k}')
    _require(np.all(v['posterior']>=0) and np.allclose(v['posterior'].sum(1),1,atol=1e-12),'posterior simplex')
    _require(np.all(np.abs(v['channels'])<=np.array([4,1,2])[None,:,None]+1e-12),'channel cap')
    _require(np.max(np.abs(v['feedback_log_evidence']),initial=0)<=.5+1e-12,'feedback cap')
    for k in ('information','reliability','weight'):
        _arr(v[k],'float64',(nv,),k); _require(np.isfinite(v[k]).all() and np.all((v[k]>=0)&(v[k]<=1)),k)
    _require(np.allclose(v['weight'],v['information']*v['reliability'],atol=1e-12),'weight product')
    _arr(v['class_id'],'uint8',(nv,),'class_id'); _require(np.array_equal(v['class_id'],v['posterior'].argmax(1)),'class mismatch')
    for k in ('selected','eligible'): _arr(v[k],'bool',(nv,),k)
    _arr(v['qc_flags'],'uint16',(nv,),'qc'); _require(np.all(v['qc_flags']<32),'unknown QC flag')
    _require(np.all(~v['selected']|v['eligible']),'selection outside universe')
    _require(not np.any(v['selected'] & ((v['qc_flags']&1)>0)),'xMHC selected')
    _strings(s['barcode'],ns,'barcode'); _require(len(set(s['barcode']))==ns,'duplicate barcode')
    _arr(s['xy'],'float64',(ns,2),'xy'); _require(np.isfinite(s['xy']).all(),'xy finite')
    _arr(s['umi'],'float64',(ns,),'umi'); _require(np.all(np.isnan(s['umi'])|((s['umi']>=0)&np.isfinite(s['umi']))),'umi')
    _arr(s['ad_available'],'bool',(ns,),'ad_available')
    for x in (ref,alt): csr_check(x,(ns,nv))
    csr_check(presence,(ns,nv),False); csr_check(graph,(ns,ns),False)
    _require(not np.any(graph.diagonal()),'self graph edge')
    _require((ref+alt)[~s['ad_available']].nnz==0,'counts in unavailable spot')
    _require((ref.astype(bool)+alt.astype(bool)).multiply(presence.astype(bool)).nnz==(ref.astype(bool)+alt.astype(bool)).nnz,'reads without record')

def validate_b(p,g):
    _fields(p,'barcode tumor_prior valid region_id anchor_prior spot_evidence'.split(),'B prior')
    n=len(p['barcode']); _strings(p['barcode'],n,'barcode'); _require(len(set(p['barcode']))==n,'duplicate barcode')
    for k in ('tumor_prior','anchor_prior','spot_evidence'):
        _arr(p[k],'float64',(n,),k); _require(np.isfinite(p[k]).all(),k)
    for k in ('tumor_prior','anchor_prior'): _require(np.all((p[k]>=.01)&(p[k]<=.99)),f'{k} bounds')
    _arr(p['valid'],'bool',(n,),'valid'); _arr(p['region_id'],'int32',(n,),'region_id')
    _require(np.all(p['region_id']>=-1),'region IDs'); _require(np.all(p['region_id'][~p['valid']]==-1),'invalid region member')
    csr_check(g,(n,n),False); _require(not np.any(g.diagonal()),'graph diagonal')

def metadata(schema,section,build,iteration,arm,inputs,parameters,calicost=False,source=None,optional=None):
    d=dict(schema=schema,section=section,genome_build=build,iteration=int(iteration),arm=arm,
      created_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,parameters=parameters,
      provenance=provenance(calicost),optional_evidence=optional or {'status':'absent','reason':'not supplied'},
      source_code=[entry(CODE/n,'source_code') for n in ('contracts.py','model.py','adapters.py','run_coupled_filter_to_viewer.py')],
      design_sha256=sha(CODE/'DESIGN.md'),elapsed_seconds=0.)
    if schema==B: d['source']=source
    return d

def seal(path,meta,start):
    p=Path(path); meta=dict(meta); meta['elapsed_seconds']=time.monotonic()-start
    meta['files']={str(x.relative_to(p)):sha(x) for x in sorted(p.iterdir()) if x.is_file()}
    dump(p/'manifest.json',meta)
    (p/'COMPLETE.sha256').write_text(sha(p/'manifest.json')+'\n')

def verify(path,inputs=True):
    p=Path(path); m=json.loads((p/'manifest.json').read_text())
    required='schema section genome_build iteration arm created_utc inputs files parameters provenance optional_evidence elapsed_seconds source_code design_sha256'.split()
    _fields(m,required+(['source'] if m.get('schema')==B else []),'manifest')
    _require(m['schema'] in (A,B),'schema'); _require(m['genome_build'] in ('GRCh37','GRCh38'),'build')
    _require(sha(p/'manifest.json')==(p/'COMPLETE.sha256').read_text().strip(),'manifest hash')
    pr=m['provenance']; _fields(pr,provenance().keys(),'provenance')
    _require(pr['gt_free'] is True and pr['gt_used'] is False and pr['independent_validation'] is False and pr['input_version_independent'] is True,'GT/provenance')
    _require(type(pr['calicost_participated']) is bool and pr['calicost_in_sample_descriptive_evidence']==pr['calicost_participated'],'CalicoST provenance')
    _require(isinstance(pr['evidence_mode'],str) and bool(pr['evidence_mode']),'evidence_mode')
    _require(bool(m['inputs']) and bool(m['source_code']),'missing input/code provenance')
    if m['schema']==B: _require(m['source'] in SOURCES and m['source']!='user-curated','prior source not admitted')
    op=m['optional_evidence']; _require(op.get('status') in ('present','absent'),'optional evidence status')
    if op['status']=='absent': _require(bool(op.get('reason')),'missing reason')
    else: _require(all(k in op for k in ('path','sha256','missing_policy','orientation')),'optional provenance')
    for rec in m['inputs']+m['source_code']:
        _fields(rec,('path','sha256','role'),'input record')
        _require(Path(rec['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}',rec['sha256']) is not None and bool(rec['role']),'input provenance')
        if inputs: _require(sha(rec['path'])==rec['sha256'],f'input drift: {rec["path"]}')
    _require(re.fullmatch('[0-9a-f]{64}',m['design_sha256']) is not None,'design hash')
    if inputs: _require(sha(CODE/'DESIGN.md')==m['design_sha256'],'design drift')
    actual={x.name for x in p.iterdir() if x.is_file()}-{'manifest.json','COMPLETE.sha256'}
    _require(actual==set(m['files']),'unmanifested payload')
    for name,h in m['files'].items():
        _require(Path(name).name==name and sha(p/name)==h,f'payload hash: {name}')
    return m

def write_a(path,v,s,ref,alt,presence,g,meta):
    start=time.monotonic(); validate_a(v,s,ref,alt,presence,g); p=Path(path); p.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(p/'variants.npz',**v); np.savez_compressed(p/'spots.npz',**s)
    for name,x in [('ref',ref),('alt',alt),('record_presence',presence),('graph',g)]: sparse.save_npz(p/(name+'.npz'),x)
    seal(p,meta,start)

def read_a(path,inputs=True):
    p=Path(path); m=verify(p,inputs); _require(m['schema']==A,'expected A')
    v=checked_npz(p/'variants.npz'); s=checked_npz(p/'spots.npz')
    mats=[sparse.load_npz(p/(n+'.npz')) for n in ('ref','alt','record_presence','graph')]
    validate_a(v,s,*mats); return v,s,*mats,m

def write_b(path,p,g,meta):
    start=time.monotonic(); validate_b(p,g); out=Path(path); out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'prior.npz',**p); sparse.save_npz(out/'graph.npz',g); seal(out,meta,start)

def read_b(path,barcodes=None,inputs=True):
    out=Path(path); m=verify(out,inputs); _require(m['schema']==B,'expected B')
    p=checked_npz(out/'prior.npz'); g=sparse.load_npz(out/'graph.npz'); validate_b(p,g)
    if barcodes is not None: _require(np.array_equal(p['barcode'],barcodes),'A/B barcode order')
    return p,g,m
