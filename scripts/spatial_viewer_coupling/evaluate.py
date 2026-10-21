"""Post-freeze evaluator. This is the only new module permitted to read GT."""
import argparse, contextlib, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from . import contracts as ct, model, audit
from .adapters import BENCH, HARNESS, FROZEN, frozen

SEED=20260916
_ACTIVE_GUARD=False

def _audit(event,args):
    if _ACTIVE_GUARD and event=='open': raise PermissionError('numerical replay denies all file reads/writes, including GT')
sys.addaudithook(_audit)

@contextlib.contextmanager
def no_io():
    global _ACTIVE_GUARD
    _ACTIVE_GUARD=True
    try: yield
    finally: _ACTIVE_GUARD=False

def metrics(y,p):
    keep=np.isfinite(y)
    if not keep.any():return dict(f1=np.nan,precision=np.nan,recall=np.nan,mcc=np.nan,tp=0,fp=0,fn=0,tn=0,n_evaluable=0)
    y=np.asarray(y)[keep]>0; p=np.asarray(p)[keep].astype(bool)
    tp=int(np.sum(y&p)); fp=int(np.sum(~y&p)); fn=int(np.sum(y&~p)); tn=int(np.sum(~y&~p))
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    denom=math.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return dict(f1=f1,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,mcc=(tp*tn-fp*fn)/denom if denom else 0.,tp=tp,fp=fp,fn=fn,tn=tn,n_evaluable=int(keep.sum()))

def labels(section,barcodes):
    paths=[]
    if section=='P6_rep1': path=frozen.P6_PATHOLOGY_STUDY/'tumor_groups.csv'; profile='groundtruth'; label_kind='pathology'
    elif section=='DCIS1':path=BENCH/'viewer/DCIS_1_SPARCAL_1000G_umi_dedup/tumor_groups.csv'; profile='gt'; label_kind='pathology'
    elif section=='P4_rep1':path=BENCH/'viewer/dist/studies/P4_rep1_WES_verified_detectable/tumor_groups.csv'; profile='copykat(cnvsilver)'; label_kind='copyKAT_silver'
    else:return np.full(len(barcodes),np.nan),[], 'unavailable'
    frame=pd.read_csv(path,dtype=str); z=frame.profile.str.replace(' ','',regex=False).str.lower(); frame=frame.loc[z.eq(profile)]
    mapping=dict(zip(frame.barcode,frame.region_name)); y=np.zeros(len(barcodes))
    if section=='P4_rep1':y=np.array([{'Tumor':1.,'Normal':0.}.get(mapping.get(b),np.nan) for b in barcodes])
    else:y=np.array([float(b in mapping) for b in barcodes])
    if not np.any(y==1):raise ValueError(f'no positive labels for {section}: {profile}')
    return y,[ct.entry(path,'evaluation_only_GT')],label_kind

def wes_sets():
    f=pd.read_csv(frozen.WES_TRUTH,sep='\t'); key=f.chrom_hg19.astype(str).str.removeprefix('chr')+'_'+f.pos_hg19.astype(int).astype(str)+'_'+f.ref.str.upper()+'_'+f.alt.str.upper()
    return {p:set(key[f['sample']==p]) for p in ('P4','P6')},ct.entry(frozen.WES_TRUTH,'evaluation_only_WES')

def phase_for(root,s):
    path=root/'phase_native.npz'
    if not path.exists():return None
    z=ct.checked_npz(path); lookup={b:i for i,b in enumerate(z['barcodes'])}; ix=np.array([lookup.get(b,-1) for b in s['barcode']]); ok=ix>=0
    p=dict(probability=np.full(len(ix),.5),valid=np.zeros(len(ix),bool)); p['probability'][ok]=z['tumor_probability'][ix[ok]]; p['valid'][ok]=z['valid'][ix[ok]]
    return p

def swap_test(v,s,ref,alt,g,arm,phase,records,y,wes,quota,budget):
    base,w,anchor,valid,ops=model.initial(v,s,ref,alt,g,arm,phase); rows=[]; previous=anchor.copy()
    contexts={'true':dict(pathology=y,wes=np.isin(v['variant_key'],list(wes)))}
    rng=np.random.default_rng(SEED)
    contexts['complement']={k:1-x for k,x in contexts['true'].items()}
    contexts['permuted']={k:rng.permutation(x) for k,x in contexts['true'].items()}
    # Positive control ensures the guard actually blocks GT reads.
    try:
        with no_io():open(frozen.WES_TRUTH,'rb')
    except PermissionError:guard_pass=True
    else:raise AssertionError('I/O guard failed')
    for k,saved in records.items():
        expected=ct.array_hash(saved); hashes=[]
        for name,ctx in contexts.items():
            with no_io():
                z=model.step(v,base,w,anchor,valid,ops,g,previous,k,model.ARMS[arm][2],quota,budget)
                ids=model.regions(z['tumor_prior'],s['barcode'],g,valid)
                numerical={x:z[x] for x in ('posterior','selected','feedback_log_evidence','tumor_prior','spot_evidence')}; numerical['region_id']=ids
                digest=ct.array_hash(numerical)
            if digest!=expected:raise AssertionError(f'label-swap or frozen replay mismatch {arm} k{k} {name}')
            hashes.append(digest)
            rows.append(dict(arm=arm,iteration=k,label_context=name,context_sha256=ct.array_hash(ctx),inference_sha256=digest,
                frozen_sha256=expected,pass_exact=True,io_guard_pass=guard_pass,swap_f1=metrics(ctx['pathology'],ids>=0)['f1'],
                swap_wes_hits=int(np.sum(ctx['wes']*z['selected']))))
        previous=saved['tumor_prior']
    return rows

def block_ids(xy):
    distances=cKDTree(xy).query(xy,k=2)[0][:,1]; pitch=float(np.median(distances))
    tiles=np.floor((xy-xy.min(0))/(4*pitch)).astype(int); _,ids=np.unique(tiles,axis=0,return_inverse=True)
    return ids,pitch

def bootstrap_counts(y,preds,blocks,n=2000):
    names=list(preds); nb=int(blocks.max())+1; keep=np.isfinite(y); positive=np.nan_to_num(y)>0
    nblock=np.bincount(blocks,weights=keep,minlength=nb); npos=np.bincount(blocks,weights=positive&keep,minlength=nb)
    frac=np.divide(npos,nblock,out=np.zeros(nb),where=nblock>0); median=np.median(frac[nblock>0]); strata=[np.flatnonzero((nblock>0)&(frac<=median)),np.flatnonzero((nblock>0)&(frac>median))]
    rng=np.random.default_rng(SEED); res=np.zeros((n,nb))
    for ids in strata:
        if len(ids):
            sampled=rng.choice(ids,size=(n,len(ids)),replace=True)
            for i in range(n):res[i]+=np.bincount(sampled[i],minlength=nb)
    out={}
    for name in names:
        p=preds[name]; tp=np.bincount(blocks,weights=positive&p&keep,minlength=nb); fp=np.bincount(blocks,weights=(~positive)&p&keep,minlength=nb); fn=np.bincount(blocks,weights=positive&(~p)&keep,minlength=nb)
        a=res@tp; denom=2*a+res@(fp+fn); out[name]=np.divide(2*a,denom,out=np.zeros(n),where=denom>0)
    return out,res

def paired_summary(y,preds,xy):
    blocks,pitch=block_ids(xy); boot,_=bootstrap_counts(y,preds,blocks); rows=[]
    targets=['coverage_Coff_k3','coverage_Con_k3','phase_Con_k3']
    pairs=[(t,b) for t in targets for b in ('smooth_coverage','coverage_Coff_k0','baseline','consensus_quality_balanced') if t in preds and b in preds]
    pairs +=[(a,b) for a,b in [('consensus_quality_balanced','baseline'),('calicost_quality_balanced','molecular_quality_balanced'),('coverage_Coff_k3','coverage_Coff_identity_k3'),('coverage_Coff_k3','coverage_Coff_prioroff_k3')] if a in preds and b in preds]
    # Equal-size spatially ordered blocks yield a bijective null on irregular tissue.
    tile=np.floor((xy-xy.min(0))/(4*pitch)).astype(int); order=np.lexsort((xy[:,0],xy[:,1],tile[:,0],tile[:,1])); block_size=32
    nfull=len(order)//block_size; interior=order[:nfull*block_size].reshape(nfull,block_size); tail=order[nfull*block_size:]
    rng=np.random.default_rng(SEED); permuted=[]
    for _ in range(999):
        ix=np.arange(len(y)); ix[interior.ravel()]=interior[rng.permutation(nfull)].ravel(); ix[tail]=tail; permuted.append(ix)
    for a,b in pairs:
        da=metrics(y,preds[a])['f1']-metrics(y,preds[b])['f1']; d=boot[a]-boot[b]
        null=np.array([metrics(y[ix],preds[a])['f1']-metrics(y[ix],preds[b])['f1'] for ix in permuted])
        rows.append(dict(model=a,reference=b,delta_f1=da,ci_low=float(np.quantile(d,.025)),ci_high=float(np.quantile(d,.975)),win_probability=float(np.mean(d>0)),
            spatial_null_p=(1+int(np.sum(null>=da)))/1000,spatial_null_two_sided_p=(1+int(np.sum(np.abs(null)>=abs(da))))/1000,
            n_bootstrap=2000,n_permutation=999,n_blocks=int(blocks.max())+1,block_side_pitches=4,permutation='equal_32_spot_spatial_order_blocks_tail_fixed',null_fixed_tail=len(tail),null_mapping_collisions=0))
    return rows,boot,blocks

def wilson(k,n):
    if not n:return (np.nan,np.nan)
    z=1.959963984540054; d=1+z*z/n; center=(k/n+z*z/(2*n))/d; rad=z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
    return center-rad,center+rad

def evaluate_section(root,out,truths):
    section=root.name; prepared=root/'prepared'
    if not (root/'section_complete.json').exists():raise ValueError(f'incomplete inference section: {section}')
    v,s,ref,alt,pres,g,im=audit.load_variant(prepared/'A',deep=True); adapter=json.loads((prepared/'adapter.json').read_text()); y,gtinputs,labelkind=labels(section,s['barcode'])
    calls=json.loads((prepared/'selections.json').read_text()); lookup={k:i for i,k in enumerate(v['variant_key'])}; obs=alt.astype(bool).astype(float)
    scores={}; valids={}; burdens={}; call_masks={}; posterior_records={}; swaps=[]; consumed=[ct.entry(root/'section_complete.json','section_freeze')]+gtinputs
    for name,keys in calls.items():
        mask=np.isin(v['variant_key'],keys); burden=np.asarray(obs@mask.astype(float)).ravel(); scores[name]=burden; valids[name]=np.ones(len(y),bool); burdens[name]=burden; call_masks[name]=mask
    coverage=model.smooth(s['umi'],g); scores['smooth_coverage']=coverage; valids['smooth_coverage']=np.ones(len(y),bool); burdens['smooth_coverage']=coverage
    phase=phase_for(root,s)
    for arm in model.ARMS:
        ar=root/arm
        if not ar.exists():continue
        if (ar/'COMPLETE.sha256').read_text().strip()!=ct.sha(ar/'run_manifest.json'):raise ValueError('run seal mismatch')
        records={}
        for k in range(4):
            it=ar/f'iteration_{k:02d}'; ma=ct.verify(it/'A'); p,_,mb=ct.read_b(it/'B',s['barcode']); av=ct.checked_npz(it/'A/variants.npz')
            ct.validate_a(av,s,ref,alt,pres,g)
            audit.check_records(ma['inputs'])
            if int(av['selected'].sum())!=adapter['quota'] or np.sum(av['selected']&((av['qc_flags']&8)>0))>adapter['low_depth_budget']:raise ValueError('quota/QC budget drift')
            if not np.array_equal(av['variant_key'],v['variant_key']):raise ValueError('iteration allele reordering')
            name=f'{arm}_k{k}'; scores[name]=p['tumor_prior']; valids[name]=p['valid']; burdens[name]=np.asarray(obs@av['selected'].astype(float)).ravel(); call_masks[name]=av['selected']
            records[k]={x:av[x] for x in ('posterior','selected','feedback_log_evidence')}; records[k].update({x:p[x] for x in ('tumor_prior','spot_evidence','region_id')})
            consumed.extend([ct.entry(it/'A/manifest.json','frozen_A'),ct.entry(it/'B/manifest.json','frozen_B')])
            # Matched ALT-burden endpoint makes the L1/score change distinguishable from anchor transfer.
            scores[name+'_matched_burden']=burdens[name]; valids[name+'_matched_burden']=np.ones(len(y),bool); burdens[name+'_matched_burden']=burdens[name]
        swaps.extend(swap_test(v,s,ref,alt,g,arm,phase,records,y,truths.get(section[:2],set()),adapter['quota'],adapter['low_depth_budget']))
    pd.DataFrame(swaps).assign(section=section).to_csv(out/f'{section}_label_swap.csv',index=False)
    qc=[]; wes=[]; selected={}; eligible=set(v['variant_key'][v['eligible']]); silver=truths.get(section[:2],set())&eligible
    for name,mask in call_masks.items():
        selected[name]=set(v['variant_key'][mask]); flags=v['qc_flags'][mask]
        qc.append(dict(section=section,model=name,n_selected=int(mask.sum()),**{label:float(np.mean((flags&bit)>0)) for label,bit in [('xmhc',1),('editing',2),('dense_10bp',4),('dp_lt3',8),('singleton',16)]}))
        if section.startswith(('P4','P6')):
            hit=len(selected[name]&silver); lo,hi=wilson(hit,len(silver)); wes.append(dict(section=section,model=name,eligible_n=len(silver),hits=hit,recall=hit/len(silver) if silver else np.nan,ci_low=lo,ci_high=hi))
    grid=[]; boot_by_mode={}; default_preds={}; paired=[]; corr=[]; detector_scores={}
    for name,score in scores.items():
        # Pure posterior is already a spatial score; preserve its frozen ordering.
        is_posterior='_k' in name and not name.endswith('_matched_burden')
        raw=score if name=='smooth_coverage' or is_posterior else model.smooth(score,g)
        resid=model.residual(score,s['umi']) if name=='smooth_coverage' else model.smooth(model.residual(score,s['umi']),g)
        detector_scores[(name,'raw')]=raw; detector_scores[(name,'coverage_residual')]=resid
        b=burdens[name]; keep=np.isfinite(s['umi']); corr.append(dict(section=section,model=name,burden_umi_spearman=float(spearmanr(b[keep],s['umi'][keep]).statistic),score_umi_spearman=float(spearmanr(score[keep],s['umi'][keep]).statistic)))
    for mode in ('raw','coverage_residual'):
        preds={}
        for name in scores:
            score=detector_scores[(name,mode)]
            # Full grid at frozen endpoints and comparators; intermediate iterations use default only.
            grid_params=list(itertools.product((85,90,95),(70,80,90),(3,5,10),(.3,.4,.5)))
            if '_k1' in name or '_k2' in name:grid_params=[(90,80,5,.4)]
            for seed,grow,size,split in grid_params:
                ids=model.regions(score,s['barcode'],g,valids[name],seed,grow,size,split); pred=ids>=0
                rec=dict(section=section,model=name,mode=mode,seed=seed,grow=grow,min_size=size,split=split,n_regions=len(set(ids)-{-1}),predicted_n=int(pred.sum()),label_kind=labelkind,**metrics(y,pred)); grid.append(rec)
                if (seed,grow,size,split)==(90,80,5,.4):preds[name]=pred
        default_preds[mode]=preds
        if np.isfinite(y).any():
            rr,bb,blocks=paired_summary(y,preds,s['xy']); paired.extend(dict(section=section,mode=mode,**r) for r in rr); boot_by_mode[mode]=bb
    pd.DataFrame(grid).to_csv(out/f'{section}_detector_grid.csv',index=False)
    pd.DataFrame(paired).to_csv(out/f'{section}_paired_spatial.csv',index=False)
    pd.DataFrame(qc).to_csv(out/f'{section}_intrinsic_qc.csv',index=False)
    pd.DataFrame(wes).to_csv(out/f'{section}_wes.csv',index=False)
    pd.DataFrame(corr).to_csv(out/f'{section}_burden_umi.csv',index=False)
    np.savez_compressed(out/f'{section}_maps.npz',barcode=s['barcode'],xy=s['xy'],umi=s['umi'],labels=y,**{f'{mode}__{name}':p for mode,pp in default_preds.items() for name,p in pp.items()},**{f'score__{name}':x for name,x in scores.items()})
    ct.dump(out/f'{section}_manifest.json',dict(section=section,inputs=consumed,label_kind=labelkind,source_code=ct.entry(Path(__file__),'evaluator_code'),audit_code=ct.entry(Path(audit.__file__),'strict_audit_code'),
        n_label_swaps=len(swaps),all_swaps_pass=all(r['pass_exact'] for r in swaps),grid_size=81,hyperparameters_gt_fit=False,
        post_freeze=True,design_sha256=ct.sha(ct.CODE/'DESIGN.md')))
    print(f'evaluated {section}: {len(grid)} cells, {len(swaps)} label replay checks',flush=True)
    return dict(section=section,selected=selected,eligible=eligible,silver=silver,boot=boot_by_mode,grid=grid,y=y,xy=s['xy'],umi=s['umi'],burdens=burdens)

def summarize(results,out,truths):
    reps=[]; wes=[]
    by={x['section']:x for x in results}
    if {'P6_rep1','P6_rep2'}<=by.keys():
        a,b=by['P6_rep1'],by['P6_rep2']; eligible=a['silver']|b['silver']
        for modelname in sorted(a['selected'].keys()&b['selected'].keys()):
            x,z=a['selected'][modelname],b['selected'][modelname]; union=x|z; hits=len(union&eligible); lo,hi=wilson(hits,len(eligible))
            reps.append(dict(patient='P6',model=modelname,intersection=len(x&z),union=len(union),jaccard=len(x&z)/len(union)))
            wes.append(dict(patient='P6',model=modelname,eligible_n=len(eligible),hits=hits,recall=hits/len(eligible),ci_low=lo,ci_high=hi))
    pd.DataFrame(reps).to_csv(out/'replicate_jaccard.csv',index=False); pd.DataFrame(wes).to_csv(out/'patient_union_wes.csv',index=False)
    interactions=[]
    contexts=['P6_rep1','P4_rep1','DCIS1']
    if set(contexts)<=by.keys():
        for mode in ('raw','coverage_residual'):
            for reference in ('smooth_coverage','coverage_Coff_k0'):
                effects=[]; actual=[]
                for section in contexts:
                    obj=by[section]; boot=obj['boot'][mode]; effects.append(boot['coverage_Coff_k3']-boot[reference])
                    rows=[r for r in obj['grid'] if r['mode']==mode and (r['seed'],r['grow'],r['min_size'],r['split'])==(90,80,5,.4)]
                    table={r['model']:r['f1'] for r in rows}; actual.append(table['coverage_Coff_k3']-table[reference])
                delta=actual[0]-(actual[1]+actual[2])/2
                # Independent section draws, not accidental pairing of identically seeded samples.
                rng=np.random.default_rng(SEED+9); dist=effects[0]-(effects[1][rng.permutation(2000)]+effects[2][rng.permutation(2000)])/2
                interactions.append(dict(endpoint=mode+'_F1',model='coverage_Coff_k3',reference=reference,interaction=delta,ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),two_sided_bootstrap_p=min(1.,2*min((np.sum(dist<=0)+1)/2001,(np.sum(dist>=0)+1)/2001)),scope='section_conditional_P4_copyKAT_silver'))
        # Formal model×context difference in absolute burden–UMI correlation.
        effects=[]; actual=[]
        for si,section in enumerate(contexts):
            obj=by[section]; blocks,_=block_ids(obj['xy']); groups=[np.flatnonzero(blocks==i) for i in np.unique(blocks)]; rng=np.random.default_rng(SEED+31+si)
            x=obj['burdens']['coverage_Coff_k3']; z=obj['burdens']['coverage_Coff_k0']; u=obj['umi']; keep=np.isfinite(u)
            def effect(ix):return abs(spearmanr(x[ix],u[ix]).statistic)-abs(spearmanr(z[ix],u[ix]).statistic)
            actual.append(effect(np.flatnonzero(keep))); vals=[]
            for _ in range(2000):
                ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); ix=ix[keep[ix]]; vals.append(effect(ix))
            effects.append(np.array(vals))
        dist=effects[0]-(effects[1]+effects[2])/2
        interactions.append(dict(endpoint='absolute_burden_UMI_rho',model='coverage_Coff_k3',reference='coverage_Coff_k0',interaction=actual[0]-(actual[1]+actual[2])/2,ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),two_sided_bootstrap_p=min(1.,2*min((np.sum(dist<=0)+1)/2001,(np.sum(dist>=0)+1)/2001)),scope='section_conditional'))
    pd.DataFrame(interactions).to_csv(out/'context_interaction.csv',index=False)
    frames=[pd.DataFrame(x['grid']) for x in results]; allgrid=pd.concat(frames,ignore_index=True)
    default=allgrid[(allgrid.seed==90)&(allgrid.grow==80)&(allgrid.min_size==5)&np.isclose(allgrid['split'],.4)]
    default.to_csv(out/'headline.csv',index=False)
    sensitivity=[]
    for (section,mode),df in allgrid.groupby(['section','mode']):
        idx=['seed','grow','min_size','split']; pivot=df.pivot(index=idx,columns='model',values='f1')
        for reference in ('smooth_coverage','coverage_Coff_k0','baseline','consensus_quality_balanced'):
            if reference not in pivot or 'coverage_Coff_k3' not in pivot:continue
            d=(pivot['coverage_Coff_k3']-pivot[reference]).dropna()
            sensitivity.append(dict(section=section,mode=mode,model='coverage_Coff_k3',reference=reference,n_grid=len(d),min_delta=d.min(),median_delta=d.median(),max_delta=d.max(),fraction_positive=float((d>0).mean())))
    pd.DataFrame(sensitivity).to_csv(out/'grid_sensitivity.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default='registered_v1'); ap.add_argument('--evaluation-id',default='evaluation_v1'); ap.add_argument('--sections',nargs='+',default=['P6_rep1','P6_rep2','P4_rep1','DCIS1']); args=ap.parse_args()
    out=ct.OUT/args.evaluation_id; out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); truths,truth_input=wes_sets()
    results=[evaluate_section(ct.OUT/args.run_id/section,out,truths) for section in args.sections]
    summarize(results,out,truths)
    ct.dump(out/'evaluation_manifest.json',dict(source=ct.entry(Path(__file__),'evaluator_code'),audit_code=ct.entry(Path(audit.__file__),'strict_audit_code'),runtime={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},truth_input=truth_input,design=ct.entry(ct.CODE/'DESIGN.md','frozen_design'),run_id=args.run_id,sections=args.sections,elapsed_seconds=time.monotonic()-started,all_label_swap_tests_passed=True,files={p.name:ct.sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    (out/'COMPLETE.sha256').write_text(ct.sha(out/'evaluation_manifest.json')+'\n')

if __name__=='__main__':main()
