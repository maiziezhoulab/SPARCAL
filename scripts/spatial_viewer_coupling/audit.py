"""Cross-contract and transitive ledger validation for exchange/evaluation."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from . import contracts as ct


def check_records(records,deep=False):
    for rec in records:
        p=Path(rec['path'])
        if p.name=='tumor_groups.csv' or p.name.startswith('STmut_Table_S1'):
            raise ValueError(f'GT input rejected: {p}')
        if ct.sha(p)!=rec['sha256']:raise ValueError(f'input hash mismatch: {p}')
        if deep and rec['role']=='all_inputs_path_sha256':check_records(json.loads(p.read_text()))


def load_variant(path,deep=False):
    for name in ('ref','alt','record_presence','graph'):
        sparse.load_npz(Path(path)/(name+'.npz')).check_format(full_check=True)
    data=ct.read_a(path); v,s,r,a,p,g,m=data
    check_records(m['inputs'],deep)
    if m['iteration']>=0:
        par=m['parameters']
        required={'quota','low_depth_budget','lambda_prior','calicost_channel','identity_destroyed'}
        if not required<=par.keys():raise ValueError('missing scoring parameters')
        if par['lambda_prior'] not in (0.,.5) or par.get('damping')!=.5 or par.get('K')!=3:raise ValueError('unregistered update parameters')
        if type(par['quota']) is not int or type(par['low_depth_budget']) is not int or not 0<=par['low_depth_budget']<=par['quota']:raise ValueError('invalid quota/QC budget')
        if int(v['selected'].sum())!=par['quota']:raise ValueError('matched quota drift')
        if int(np.sum(v['selected']&((v['qc_flags']&8)>0)))>par['low_depth_budget']:raise ValueError('DP budget exceeded')
        if np.max(abs(v['feedback_log_evidence']),initial=0)>par['lambda_prior']+1e-12:raise ValueError('declared prior bound exceeded')
    return data


def load_pair(a_path,b_path,deep=False):
    sparse.load_npz(Path(b_path)/'graph.npz').check_format(full_check=True)
    a=load_variant(a_path,deep); v,s,ref,alt,pres,g,am=a; p,bg,bm=ct.read_b(b_path,s['barcode'])
    check_records(bm['inputs'],deep)
    for key in ('section','genome_build','arm','iteration'):
        if am[key]!=bm[key]:raise ValueError(f'A/B {key} mismatch')
    if (g!=bg).nnz:raise ValueError('A/B graph mismatch')
    if np.any(p['valid']&~s['ad_available']):raise ValueError('unobserved spot admitted')
    if np.max(abs(p['spot_evidence']),initial=0)>2+1e-12:raise ValueError('viewer channel cap exceeded')
    if not any(rec['role']=='updated_contract_A' and rec['sha256']==ct.sha(Path(a_path)/'manifest.json') for rec in bm['inputs']):raise ValueError('B not linked to this A')
    return a,(p,bg,bm)
