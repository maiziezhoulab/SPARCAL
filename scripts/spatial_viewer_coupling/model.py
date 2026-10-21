"""Bounded inference. GT is deliberately absent from this module's interface."""
import sys
import numpy as np
from scipy import sparse
from scipy.special import expit, softmax, logit
from scipy.stats import rankdata
from .contracts import STATES
VIEWER='/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer'
if VIEWER not in sys.path: sys.path.insert(0,VIEWER)
from sparcal_viewer import region_methods as kernels
from sparcal_viewer.region_integration import watershed_result
from sparcal_viewer.data import StudyData

PARAMS=dict(K=3,damping=.5,lambda_prior=.5,effective_observation_cap=8,molecular_cap=4.,spatial_cap=1.,calicost_cap=2.,viewer_cap=2.,convergence_tolerance=1e-4,seed=20260916)
ARMS={'coverage_Coff':('coverage-only prior',False,.5,False),
      'coverage_Con':('coverage-only prior',True,.5,False),
      'none_Coff':('none',False,.5,False),
      'coverage_Coff_prioroff':('coverage-only prior',False,0.,False),
      'coverage_Coff_identity':('coverage-only prior',False,.5,True),
      'phase_Con':('phase-aware posterior',True,.5,False)}

def neighbors(g): return [g.indices[g.indptr[i]:g.indptr[i+1]] for i in range(g.shape[0])]
def smooth(x,g): return kernels.coverage_prior(np.asarray(x,float),neighbors(g))
def priors(race): return np.where((race=='defined')[:,None],np.array([.08,.72,.16,.03,.01]),np.array([.72,.16,.02,.08,.02]))
def center(x): return x-x.mean(axis=1,keepdims=True)

def channel_logits(mol,spa,cal,race,context):
    q=priors(race); m=np.clip(center(np.log(np.maximum(mol,1e-12)/q)),-4,4)
    s=np.clip(center(np.log(np.maximum(spa,1e-12)/np.maximum(mol,1e-12))),-1,1)
    c=np.clip(center(np.log(np.maximum(cal,1e-12)/np.maximum(mol,1e-12))),-2,2)
    c[~context]=0
    return np.stack([m,s,c],axis=1)

def operators(ref,alt,valid):
    obs=alt.astype(bool).astype(float).multiply(valid[:,None]).tocsr()
    cov=(ref+alt).astype(bool).astype(float).multiply(valid[:,None]).tocsr()
    na=np.asarray(obs.sum(0)).ravel(); nd=np.asarray(cov.sum(0)).ravel()
    adiff=obs @ sparse.diags(np.divide(1.,na,out=np.zeros_like(na),where=na>0))-cov @ sparse.diags(np.divide(1.,nd,out=np.zeros_like(nd),where=nd>0))
    adiff=adiff @ sparse.diags(((na>0)&(nd>0)).astype(float))
    return obs,cov,adiff.tocsr(),np.minimum(nd,8)/8

def select(v,posterior,quota,low_budget):
    score=posterior[:,3:].sum(1)*v['weight']
    allowed=v['eligible'] & ((v['qc_flags']&1)==0)
    ids=np.flatnonzero(allowed); ids=ids[np.lexsort((v['variant_key'][ids],-score[ids]))]
    low=(v['qc_flags']&8)>0
    low_order=np.cumsum(low[ids]); ids=ids[(~low[ids])|(low_order<=low_budget)]
    if len(ids)<quota: raise ValueError(f'impossible matched quota {quota}, eligible {len(ids)}')
    selected=np.zeros(len(score),bool); selected[ids[:quota]]=True
    return selected

def initial(v,s,ref,alt,g,arm,phase=None):
    source,cal,lam,identity=ARMS[arm]; u=np.array(v['channels'],copy=True); weights=v['weight'].copy()
    if identity:
        perm=np.random.default_rng(PARAMS['seed']).permutation(len(weights)); u=u[perm]; weights=weights[perm]
    h=np.minimum(np.asarray((ref+alt).astype(bool).sum(0)).ravel(),8)/8
    base=np.log(priors(v['race']))+h[:,None]*(u[:,0]+u[:,1]+int(cal)*u[:,2])
    valid=s['ad_available'].copy()
    if source=='coverage-only prior':
        ranks=rankdata(smooth(s['umi'],g),method='average'); anchor=.1+.8*(ranks-.5)/len(ranks)
    elif source=='none': anchor=np.full(len(valid),.5)
    else:
        if phase is None: raise ValueError('phase arm requires declared native sidecar')
        anchor=np.clip(phase['probability'],.1,.9); valid &= phase['valid']; anchor[~valid]=.5
    return base,weights,anchor,valid,operators(ref,alt,valid)

def step(v,base,weights,anchor,valid,ops,g,previous,iteration,lam,quota,low_budget):
    obs,cov,diff,h=ops
    e=np.zeros(len(weights)) if iteration==0 else lam*h*np.tanh(np.asarray(diff.T@(2*previous-1)).ravel())
    post=softmax(base+e[:,None]*np.array([0.,0.,0.,1.,1.]),axis=1)
    w=weights*v['eligible']; den=np.asarray(obs@w).ravel()
    signal=np.divide(np.asarray(obs@(w*post[:,3:].sum(1))).ravel(),den,out=np.full(len(den),.5),where=den>0)
    ev=2*smooth(np.minimum(den,8)/8*(2*signal-1),g)
    proposed=expit(logit(anchor)+ev)
    nxt=anchor.copy() if iteration==0 else .5*previous+.5*proposed
    nxt[~valid]=anchor[~valid]
    return dict(posterior=post,feedback_log_evidence=e,selected=select(v,post,quota,low_budget),
                tumor_prior=nxt,spot_evidence=ev,weighted_burden=np.asarray(obs@(w*post[:,3:].sum(1))).ravel())

def regions(score,barcodes,g,valid=None,seed=90,grow=80,size=5,split=.4):
    result=watershed_result(np.asarray(score,float),np.asarray(score,float),barcodes,neighbors(g),StudyData._watershed_basins,
        seed_pct=seed,grow_pct=grow,min_size=size,split_depth=split,eligible_mask=valid)
    ids=np.full(len(barcodes),-1,np.int32); mapping={b:i for i,b in enumerate(barcodes)}
    for j,group in enumerate(result['regions']):
        for b in group: ids[mapping[b]]=j
    return ids

def residual(x,umi):
    u=np.asarray(umi,float).copy(); u[~np.isfinite(u)]=np.nanmedian(u) if np.isfinite(u).any() else 0
    design=np.column_stack([np.ones(len(u)),u]); return kernels._residual(np.asarray(x,float),design)
