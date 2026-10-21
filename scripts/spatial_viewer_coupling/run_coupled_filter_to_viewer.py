#!/usr/bin/env python3
"""Immutable additive experiment; run as python -m scripts.spatial_viewer_coupling.run_coupled_filter_to_viewer."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from . import contracts as ct, model
from .adapters import prepare, FROZEN, frozen


def phase_sidecar(section,out):
    cfg=frozen.SECTIONS[section]; parsed=Path(cfg['clones']).parent.parent/'parsed_inputs'
    side=out/'phase_native.npz'
    cmd=[sys.executable,str(ct.ROOT/'scripts/viewer_phase_aware/build_from_calicost.py'),str(parsed),str(cfg['purity']),str(cfg['clones']),str(cfg['cnv']),str(side),
        '--confirm-label-free','--acknowledge-in-sample-calicost-evidence','--model-name',f'coupling_{section}']
    print('PHASE', ' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
    return side

def run_arm(section,prepared,out,arm,phase_path=None):
    started=time.monotonic(); v,s,ref,alt,pres,g,im=ct.read_a(prepared/'A')
    adapter=json.loads((prepared/'adapter.json').read_text()); quota=adapter['quota']; budget=adapter['low_depth_budget']
    source,cal,lam,identity=model.ARMS[arm]; phase=None; op={'status':'absent','reason':'phase not used by this arm'}
    inputs=[ct.entry(prepared/'A/manifest.json','contract_A'),ct.entry(prepared/'adapter.json','adapter_quotas')]
    if source=='phase-aware posterior':
        z=ct.checked_npz(phase_path); pr=json.loads(str(z['provenance']))
        if pr['production_inputs_include_ground_truth'] or pr['evaluation_uses_ground_truth'] or not pr['posterior_is_frozen']: raise ValueError('unsafe phase provenance')
        idx={b:i for i,b in enumerate(z['barcodes'])}; indexes=np.array([idx.get(b,-1) for b in s['barcode']]); found=indexes>=0
        phase=dict(probability=np.full(len(indexes),.5),valid=np.zeros(len(indexes),bool))
        phase['probability'][found]=z['tumor_probability'][indexes[found]]; phase['valid'][found]=z['valid'][indexes[found]]
        inputs.append(ct.entry(phase_path,'native_phase_posterior'))
        for p,h in pr['inputs'].items():
            if ct.sha(p)!=h: raise ValueError(f'phase input drift {p}')
            inputs.append(dict(path=p,sha256=h,role='phase_source'))
        inputs.extend(ct.entry(ct.ROOT/'scripts/viewer_phase_aware'/f,'phase_code') for f in ('build_from_calicost.py','phase_aware_evidence.py'))
        op=dict(status='present',path=str(phase_path.resolve()),sha256=ct.sha(phase_path),missing_policy='explicit_prior_marginalization',orientation=pr['phase_cn_orientation'])
    for name in ('region_methods.py','region_integration.py','data.py','config.py'):
        inputs.append(ct.entry(Path(model.VIEWER)/'sparcal_viewer'/name,'viewer_kernel_code'))
    base,w,anchor,valid,ops=model.initial(v,s,ref,alt,g,arm,phase)
    previous=anchor.copy(); old_selected=None; diagnostics=[]; out.mkdir(parents=True,exist_ok=False)
    for k in range(4):
        tick=time.monotonic(); result=model.step(v,base,w,anchor,valid,ops,g,previous,k,lam,quota,budget)
        ids=model.regions(result['tumor_prior'],s['barcode'],g,valid)
        p=dict(barcode=s['barcode'],tumor_prior=result['tumor_prior'],valid=valid,region_id=ids,anchor_prior=anchor,spot_evidence=result['spot_evidence'])
        vv=dict(v); vv.update(posterior=result['posterior'],base_logits=base,feedback_log_evidence=result['feedback_log_evidence'],selected=result['selected'],class_id=result['posterior'].argmax(1).astype('uint8'))
        if identity:
            permutation=np.random.default_rng(model.PARAMS['seed']).permutation(len(w))
            for field in ('information','reliability','weight','channels'): vv[field]=v[field][permutation]
        params=dict(model.PARAMS,lambda_prior=lam,calicost_channel=cal,identity_destroyed=identity,quota=quota,low_depth_budget=budget)
        iteration=out/f'iteration_{k:02d}'; iteration.mkdir()
        current_inputs=list(inputs)
        if k: current_inputs.append(ct.entry(out/f'iteration_{k-1:02d}/B/manifest.json','previous_contract_B'))
        ma=ct.metadata(ct.A,section,im['genome_build'],k,arm,current_inputs,params,cal or source=='phase-aware posterior',optional=op)
        ct.write_a(iteration/'A',vv,s,ref,alt,pres,g,ma)
        mb=ct.metadata(ct.B,section,im['genome_build'],k,arm,[ct.entry(iteration/'A/manifest.json','updated_contract_A')]+current_inputs,params,cal or source=='phase-aware posterior',source,op)
        ct.write_b(iteration/'B',p,g,mb)
        np.savez_compressed(iteration/'diagnostic_scores.npz',weighted_burden=result['weighted_burden'])
        delta=float(np.max(np.abs(result['tumor_prior']-previous))); sel=result['selected']
        diag=dict(iteration=k,prior_sup_change=delta,fixed_point_error_bound=delta/.25 if k else None,
            converged=bool(k>0 and delta<=1e-4),selected_n=int(sel.sum()),low_depth_n=int(np.sum(sel&((v['qc_flags']&8)>0))),
            matched_jaccard_previous=float(np.sum(sel&old_selected)/np.sum(sel|old_selected)) if old_selected is not None else None,
            max_abs_feedback=float(np.max(np.abs(result['feedback_log_evidence']))),elapsed_seconds=time.monotonic()-tick)
        ct.dump(iteration/'iteration_manifest.json',dict(diagnostics=diag,A=ct.entry(iteration/'A/manifest.json','frozen_A'),B=ct.entry(iteration/'B/manifest.json','frozen_B'),scores=ct.entry(iteration/'diagnostic_scores.npz','spot_scores'),label_swap_status='pending_post_freeze_evaluator'))
        diagnostics.append(diag); previous=result['tumor_prior']; old_selected=sel
        print(section,arm,diag,flush=True)
    ct.dump(out/'run_manifest.json',dict(section=section,arm=arm,freeze_iteration=3,iterations=diagnostics,elapsed_seconds=time.monotonic()-started,
        inputs=inputs,gt_free=True,label_swap_status='pending_post_freeze_evaluator'))
    (out/'COMPLETE.sha256').write_text(ct.sha(out/'run_manifest.json')+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--section',required=True,choices=['P6_rep1','P6_rep2','P4_rep1','DCIS1'])
    p.add_argument('--run-id',default='registered_v1'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--skip-phase',action='store_true'); p.add_argument('--prepared',type=Path)
    a=p.parse_args(); root=ct.OUT/a.run_id/a.section; root.mkdir(parents=True,exist_ok=False)
    prepared=a.prepared or root/'prepared'
    if a.prepared is None: prepare(a.section,prepared,a.workers)
    if a.prepare_only:return
    for arm in model.ARMS:
        if arm=='phase_Con' or (a.section=='DCIS1' and arm=='coverage_Con'):continue
        run_arm(a.section,prepared,root/arm,arm)
    if a.section!='DCIS1' and not a.skip_phase:
        phase=phase_sidecar(a.section,root); run_arm(a.section,prepared,root/'phase_Con','phase_Con',phase)
    # Recheck all consumed raw inputs once; per-bundle readers check the ledger hash.
    ledger=json.loads((prepared/'input_ledger.json').read_text())
    for rec in ledger:
        if ct.sha(rec['path'])!=rec['sha256']: raise ValueError(f'input changed during run: {rec["path"]}')
    ct.dump(root/'section_complete.json',dict(section=a.section,gt_free=True,inputs_rehashed=len(ledger),label_swap_status='pending_post_freeze_evaluator'))

if __name__=='__main__':main()
