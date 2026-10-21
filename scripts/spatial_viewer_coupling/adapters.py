"""Read-only ingestion; raw AD extraction is not variant calling or filtering."""
import gzip, hashlib, io, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax, betaln, gammaln, logsumexp
from . import contracts as ct
from .model import channel_logits, priors, PARAMS, VIEWER
from scripts.spatial_filter_rebuild import config as frozen

BENCH=Path('/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking')
HARNESS=BENCH/'analysis/region_method_benchmark/snv_informed_regions_2026-09-12'
FROZEN=frozen.OUTPUT_ROOT
_MODELS=('molecular','spatial','calicost','integrated')
_KEYS=None

def _worker_init(keys):
    global _KEYS; _KEYS=keys

def _vcf(task):
    row,path=task; p=Path(path); raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    r={}; a={}; present=set(); malformed=0
    with gzip.open(io.BytesIO(raw),'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            q=line.rstrip().split('\t')
            if len(q)<10: raise ValueError(f'malformed VCF {path}')
            key=f'{q[0].removeprefix("chr")}_{q[1]}_{q[3]}_{q[4]}'
            col=_KEYS.get(key)
            if col is None: continue
            if col in present: raise ValueError(f'duplicate allele in {path}: {key}')
            present.add(col); fields=dict(zip(q[8].split(':'),q[9].split(':')))
            vals=fields.get('AD','').split(',')
            if len(vals)!=2 or any(not x.isdigit() for x in vals): malformed+=1; continue
            rv,av=map(int,vals)
            if rv:r[col]=rv
            if av:a[col]=av
    if malformed: raise ValueError(f'{path}: {malformed} candidate records without exact AD')
    return row,r,a,present,dict(path=str(p.resolve()),sha256=digest,role='raw_spot_vcf')

def _geometry(section):
    if section=='DCIS1': study=BENCH/'viewer/DCIS_1_SPARCAL_1000G_umi_dedup'
    else:
        study=BENCH/'viewer/dist/studies/ad_vaf_shadow_v1'/f'{section}_SPARCAL_somatic_ad_vaf_shadow_v1'
    # Pure harness geometry functions do not read tumor_groups. Never call load_section.
    if str(HARNESS) not in sys.path: sys.path.insert(0,str(HARNESS))
    from harness.data import _config,_positions,_coverage,_adjacency
    cfg=_config(study); pos,xy=_positions(study,cfg); bcs=pos.index.astype(str).to_numpy()
    nb=_adjacency(xy); rows=np.repeat(np.arange(len(nb)),[len(x) for x in nb]); cols=np.concatenate(nb)
    g=sparse.csr_matrix((np.ones(len(rows),np.uint8),(rows,cols)),shape=(len(bcs),len(bcs))); g.sort_indices()
    inputs=[ct.entry(next(study.glob('*.config')),'geometry_configuration')]
    for key in ('tissue_positions','scalefactors','spot_coverage'): inputs.append(ct.entry(study/cfg['files'][key],key))
    inputs.append(ct.entry(HARNESS/'harness/data.py','geometry_reader_code'))
    return dict(barcode=bcs.astype(str),xy=xy.astype(float),umi=_coverage(study,cfg,bcs).astype(float),ad_available=np.ones(len(bcs),bool)),g,inputs

def _exact_ad(section,keys,barcodes,workers):
    root=(ct.ROOT/'data/dcis1/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot') if section=='DCIS1' else Path(frozen.SECTIONS[section]['vcf_dir']); tasks=[]; available=np.zeros(len(barcodes),bool)
    for i,b in enumerate(barcodes):
        p=root/(b+'.vcf.gz')
        if p.exists(): tasks.append((i,str(p))); available[i]=True
    parts=[([],[],[]) for _ in range(3)]; ledger=[]
    with mp.Pool(workers,initializer=_worker_init,initargs=({k:i for i,k in enumerate(keys)},)) as pool:
        for j,(row,r,a,p,record) in enumerate(pool.imap_unordered(_vcf,tasks,chunksize=8)):
            for idx,vals in enumerate((r,a,{k:1 for k in p})):
                parts[idx][0].extend([row]*len(vals)); parts[idx][1].extend(vals); parts[idx][2].extend(vals.values())
            ledger.append(record)
            if j%500==0: print(f'{section}: exact AD {j}/{len(tasks)}',flush=True)
    mats=[]
    for idx,(rr,cc,vv) in enumerate(parts):
        x=sparse.csr_matrix((np.array(vv,dtype='uint8' if idx==2 else 'uint32'),(rr,cc)),shape=(len(barcodes),len(keys))); x.sort_indices(); mats.append(x)
    return *mats,available,sorted(ledger,key=lambda x:x['path'])

def _dcis_post(meta,ref,alt):
    y=np.asarray(alt.sum(0)).ravel(); n=np.asarray((ref+alt).sum(0)).ravel()
    grids=[([.001,.005,.01,.02,.05],20),([.3,.5,.7],8),([.9,.97,.99],12),([.02,.05,.1,.2],8),([.2,.3,.4,.5],8)]
    ll=[]
    for grid,k in grids:
        q=np.array(grid)[None,:]; aa=q*k; bb=(1-q)*k
        z=betaln(y[:,None]+aa,n[:,None]-y[:,None]+bb)-betaln(aa,bb)
        ll.append(logsumexp(z,axis=1)-np.log(len(grid)))
    return softmax(np.column_stack(ll)+np.log(priors(meta.race.astype(str).to_numpy())),axis=1)

def prepare(section,out,workers=4):
    start=time.monotonic(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    s,g,inputs=_geometry(section)
    if section!='DCIS1':
        score=FROZEN/section/'scores/all_model_scores.tsv.gz'
        cols=['variant_key','race','xmhc','rna_editing_change','local_density_10bp','vaf_context_reliability']+[f'{m}_p_{st}' for m in _MODELS for st in ct.STATES]
        frame=pd.read_csv(score,sep='\t',usecols=cols,low_memory=False); inputs.append(ct.entry(score,'frozen_multistate'))
        parts=frame.variant_key.astype(str).str.split('_',expand=True)
        ok=(parts.shape[1]==4)&parts[2].isin(list('ACGT'))&parts[3].isin(list('ACGT'))&parts[2].ne(parts[3])
        excluded=int((~ok).sum()); frame=frame.loc[ok].reset_index(drop=True)
        keys=frame.variant_key.astype(str).to_numpy(); ref,alt,pres,available,ledger=_exact_ad(section,keys,s['barcode'],workers)
        s['ad_available']=available
        mol,spa,cal=[frame[[f'{m}_p_{st}' for st in ct.STATES]].to_numpy(float) for m in _MODELS[:3]]
        ch=channel_logits(mol,spa,cal,frame.race.astype(str).to_numpy(),frame.vaf_context_reliability.to_numpy()>0)
        qc=frame.xmhc.to_numpy(bool).astype('uint16')| (frame.rna_editing_change.to_numpy(bool).astype('uint16')<<1)|((frame.local_density_10bp.to_numpy()>1).astype('uint16')<<2)
        baseline_sets={}
        for m in ['baseline']+[m+'_quality_balanced' for m in (*_MODELS,'consensus')]:
            path=FROZEN/section/'scores'/f'selected__{m}__matched.tsv.gz'; f=pd.read_csv(path,sep='\t',usecols=['variant_key'])
            baseline_sets[m]=f.variant_key.astype(str).tolist(); inputs.append(ct.entry(path,'frozen_selection'))
        build='GRCh37'
    else:
        cache=HARNESS/'data_cache/DCIS1'; frame=pd.read_csv(cache/'snv_meta.csv',low_memory=False).rename(columns={'key':'variant_key'})
        ok=frame.ref.isin(list('ACGT'))&frame.alt.isin(list('ACGT'))&frame.ref.ne(frame.alt); excluded=int((~ok).sum())
        rr=sparse.load_npz(cache/'ad_ref.npz').tocsr(); aa=sparse.load_npz(cache/'ad_alt.npz').tocsr()
        bcs=(cache/'barcodes.txt').read_text().splitlines()
        if bcs!=s['barcode'].tolist(): raise ValueError('DCIS barcode ordering')
        frame=frame.loc[ok].reset_index(drop=True); ref=rr[:,ok.to_numpy()].astype('uint32'); alt=aa[:,ok.to_numpy()].astype('uint32')
        keys=frame.variant_key.astype(str).to_numpy()
        exact_ref,exact_alt,pres,available,ledger=_exact_ad(section,keys,s['barcode'],workers)
        if (exact_ref!=ref).nnz or (exact_alt!=alt).nnz: raise ValueError('DCIS frozen AD does not match raw VCF')
        s['ad_available']=available
        inputs.extend(ct.entry(cache/n,'exact_AD_cache') for n in ('ad_ref.npz','ad_alt.npz','snv_meta.csv','barcodes.txt','ad_build_qc.json'))
        qcmeta=json.loads((cache/'ad_build_qc.json').read_text())
        missing=qcmeta.get('missing_vcfs',[]); s['ad_available']=~np.isin(s['barcode'],missing)
        mol=_dcis_post(frame,ref,alt); ch=channel_logits(mol,mol,mol,frame.race.astype(str).to_numpy(),np.zeros(len(frame),bool))
        chrom=frame.chrom.astype(str).str.removeprefix('chr').to_numpy(); pos=frame.pos.to_numpy()
        density=np.zeros(len(frame),bool)
        for c in np.unique(chrom):
            ix=np.flatnonzero(chrom==c); z=np.sort(pos[ix]); density[ix]=(np.searchsorted(z,pos[ix]+10,side='right')-np.searchsorted(z,pos[ix]-10,side='left'))>1
        editing=((frame.ref=='A')&(frame.alt=='G'))|((frame.ref=='T')&(frame.alt=='C'))
        qc=((chrom=='6')&(pos>=25000000)&(pos<=34000000)).astype('uint16')|(editing.to_numpy().astype('uint16')<<1)|(density.astype('uint16')<<2)
        baseline_sets={'baseline':frame.loc[frame.in_somatic.astype(bool)&frame.race.eq('denovo')&(np.asarray(alt.sum(0)).ravel()>0),'variant_key'].tolist()}
        build='GRCh38'
    ct.dump(out/'input_ledger.json',ledger+inputs)
    input_rec=[ct.entry(out/'input_ledger.json','all_inputs_path_sha256')]+inputs
    dep=np.asarray((ref+alt).sum(0)).ravel().astype(float); positive=np.asarray(alt.astype(bool).sum(0)).ravel()
    qc|=(dep<3).astype('uint16')<<3; qc|=(positive==1).astype('uint16')<<4
    info=(1-np.exp(-dep/10))*np.minimum(positive/3,1)
    reliability=np.clip(1-.5*((qc&2)>0)-.25*((qc&4)>0),.25,1); reliability[(qc&1)>0]=0
    parts=[k.split('_') for k in keys]; race=frame.race.fillna('unknown').astype(str).to_numpy()
    eligible=(race=='denovo')&(positive>0); base=np.log(priors(race))+np.minimum(np.asarray((ref+alt).astype(bool).sum(0)).ravel(),8)[:,None]/8*(ch[:,0]+ch[:,1])
    post=softmax(base,axis=1)
    v=dict(variant_key=keys.astype(str),chrom=np.array([x[0] for x in parts]),pos=np.array([int(x[1]) for x in parts],dtype='int64'),
      ref_allele=np.array([x[2] for x in parts]),alt_allele=np.array([x[3] for x in parts]),race=race.astype(str),posterior=post,
      information=info,reliability=reliability,weight=info*reliability,class_id=post.argmax(1).astype('uint8'),selected=np.zeros(len(keys),bool),
      eligible=eligible,qc_flags=qc,base_logits=base,feedback_log_evidence=np.zeros(len(keys)),channels=ch)
    keyidx={k:i for i,k in enumerate(keys)}
    for m,ks in baseline_sets.items():
        absent=set(ks)-keyidx.keys()
        if absent: raise ValueError(f'{section}/{m} has {len(absent)} unaligned selected alleles')
    baseline_ix=np.array([keyidx[k] for k in baseline_sets['baseline']],int); quota=len(baseline_ix); budget=int(np.sum(dep[baseline_ix]<3))
    ct.dump(out/'selections.json',baseline_sets)
    ct.dump(out/'adapter.json',dict(section=section,quota=quota,low_depth_budget=budget,excluded_non_snp_or_missing_allele=excluded,
        input_kind='dcis_ad_adapter' if section=='DCIS1' else 'frozen_multistate',n_spots=len(s['barcode']),n_variants=len(keys),
        record_presence_semantics='VCF_record_exists',elapsed_seconds=time.monotonic()-start))
    meta=ct.metadata(ct.A,section,build,-1,'ingested',input_rec,dict(PARAMS,quota=quota,low_depth_budget=budget),True if section!='DCIS1' else False)
    ct.write_a(out/'A',v,s,ref,alt,pres,g,meta)
    print(f'{section}: prepared V={len(keys)} S={len(s["barcode"])} quota={quota}',flush=True)
    return out
