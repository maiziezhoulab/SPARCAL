"""Export a new label-free viewer study; never edits an existing profile."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from . import contracts as ct,audit,model
from .adapters import BENCH
from sparcal_viewer.region_integration import load_tumor_evidence_sidecar,phase_aware_viewer_regions
from sparcal_viewer.data import StudyData


def export(run_root,section,out,arm='coverage_Coff',iteration=3):
    import yaml
    root=Path(run_root)/section; it=root/arm/f'iteration_{iteration:02d}'
    (v,s,ref,alt,pres,g,am),(p,_,bm)=audit.load_pair(it/'A',it/'B')
    dest=Path(out)/f'{section}_{arm}_k{iteration}'; dest.mkdir(parents=True,exist_ok=False)
    source=BENCH/'viewer/DCIS_1_SPARCAL_1000G_umi_dedup' if section=='DCIS1' else BENCH/'viewer/dist/studies/ad_vaf_shadow_v1'/f'{section}_SPARCAL_somatic_ad_vaf_shadow_v1'
    config_path=next(source.glob('*.config')); cfg=yaml.safe_load(config_path.read_text()); inputs=[ct.entry(config_path,'geometry_configuration'),ct.entry(it/'A/manifest.json','A'),ct.entry(it/'B/manifest.json','B')]
    for name in ('tissue_positions','scalefactors','hires_image','spot_coverage'):
        src=source/cfg['files'][name]; target=dest/src.name; shutil.copyfile(src,target); inputs.append(ct.entry(src,name)); cfg['files'][name]=target.name
    mask=v['selected']; matrix=alt[:,mask].astype(bool).astype('uint8').toarray()
    pd.DataFrame(matrix,index=s['barcode'],columns=v['variant_key'][mask]).to_pickle(dest/'snv_matrix.pkl')
    cfg['project']='SPARCAL coupling'; cfg['study']=dest.name; cfg['files']['snv_matrix']='snv_matrix.pkl'
    for name in ('tumor_groups','tumor_centers','variant_groups'):cfg['files'][name]=name+'.csv'
    groups=[('CoupledFrozen',f'Region_{int(r)+1}',str(b)) for b,r in zip(p['barcode'],p['region_id']) if r>=0]
    pd.DataFrame(groups,columns=['profile','region_name','barcode']).to_csv(dest/'tumor_groups.csv',index=False)
    pd.DataFrame(columns=['profile','region_name','barcode']).to_csv(dest/'tumor_centers.csv',index=False)
    pd.DataFrame(columns=['profile','region_name','group_name','group_type','snv_key']).to_csv(dest/'variant_groups.csv',index=False)
    cfg['files']['tumor_evidence']='tumor_evidence.npz'
    n=len(p['barcode']); zeros=np.zeros(n); no=np.zeros(n,bool)
    provenance=dict(production_inputs_include_ground_truth=False,evaluation_uses_ground_truth=False,posterior_is_frozen=True,
        input_contracts=inputs,source=bm['source'],calicost_participated=am['provenance']['calicost_participated'],
        independent_validation=False,component_interpretation='spatial_log_prior=anchor_logit; molecular_log_bf=net_damped_update; remaining components absent',
        contract_A=str((it/'A/manifest.json').resolve()),contract_B=str((it/'B/manifest.json').resolve()),
        compatibility_matrix_semantics='ALT>0; per-variant posterior and exact REF/ALT remain in contract A')
    np.savez_compressed(dest/'tumor_evidence.npz',schema=np.asarray('SPARCAL_tumor_evidence_v1'),model_name=np.asarray(dest.name),provenance=np.asarray(json.dumps(provenance,sort_keys=True)),
        barcodes=p['barcode'],valid=p['valid'],phase_available=no,purity_available=no,tumor_probability=p['tumor_prior'],information=p['valid'].astype(float),
        phase_cnv_log_bf=zeros,purity_log_prior=zeros,somatic_vaf_log_bf=zeros,molecular_log_bf=logit(p['tumor_prior'])-logit(p['anchor_prior']),spatial_log_prior=logit(p['anchor_prior']))
    (dest/(dest.name+'.config')).write_text(yaml.safe_dump(cfg,sort_keys=False))
    loaded=load_tumor_evidence_sidecar(dest/'tumor_evidence.npz',s['barcode']); predictions=[]
    for component in ('tumor_probability','molecular_log_bf'):
        result=phase_aware_viewer_regions(loaded,s['barcode'],model.neighbors(g),StudyData._watershed_basins,display_component=component,seed_pct=90,grow_pct=80,min_size=5,split_depth=.4)
        predictions.append({b for group in result['regions'] for b in group})
    expected=set(p['barcode'][p['region_id']>=0])
    if predictions[0]!=expected or predictions[1]!=expected:raise AssertionError('viewer export changed frozen region membership')
    from sparcal_viewer.config import load_config
    study=StudyData(load_config(str(dest/(dest.name+'.config'))))
    actual=study.detect_tumor_regions('phase_aware',seed_pct=90,grow_pct=80,min_size=5,split_depth=.4)
    if {b for group in actual['regions'] for b in group}!=expected:raise AssertionError('full viewer study disagrees with frozen B')
    ct.dump(dest/'export_manifest.json',dict(full_study_load_pass=True,section=section,arm=arm,iteration=iteration,inputs=inputs,source_code=ct.entry(Path(__file__),'export_code'),
        source_contract_A=str((it/'A').resolve()),source_contract_B=str((it/'B').resolve()),gt_free=True,viewer_roundtrip_regions_equal=True,component_switch_regions_equal=True,
        files={x.name:ct.sha(x) for x in sorted(dest.iterdir()) if x.is_file()}))
    (dest/'COMPLETE.sha256').write_text(ct.sha(dest/'export_manifest.json')+'\n')
    print(dest,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='registered_v1');p.add_argument('--output-name',default='viewer_studies');p.add_argument('--sections',nargs='+',default=['P6_rep1','P6_rep2','P4_rep1','DCIS1']);a=p.parse_args()
    for section in a.sections:export(ct.OUT/a.run_id,section,ct.OUT/a.output_name)
if __name__=='__main__':main()
