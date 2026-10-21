import copy, json, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import softmax
from . import contracts as c, model as m

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=c.OUT); self.root=Path(self.tmp.name)
        n=12; q=7; rng=np.random.default_rng(3)
        self.r=sparse.csr_matrix(rng.integers(0,5,(n,q),dtype='uint32')); self.a=sparse.csr_matrix(rng.integers(0,3,(n,q),dtype='uint32'))
        self.p=(self.r+self.a).astype(bool).astype('uint8'); self.g=sparse.csr_matrix((np.ones(22,'uint8'),(np.r_[np.arange(11),np.arange(1,12)],np.r_[np.arange(1,12),np.arange(11)])),shape=(n,n))
        self.s=dict(barcode=np.array([f'bc{i}' for i in range(n)]),xy=np.column_stack([np.arange(n),np.zeros(n)]).astype(float),umi=np.arange(n,dtype=float),ad_available=np.ones(n,bool))
        ch=rng.uniform(-.5,.5,(q,3,5)); post=softmax(ch[:,0],axis=1)
        self.v=dict(variant_key=np.array([f'1_{i+1}_A_C' for i in range(q)]),chrom=np.full(q,'1'),pos=np.arange(1,q+1,dtype='int64'),ref_allele=np.full(q,'A'),alt_allele=np.full(q,'C'),race=np.full(q,'denovo'),posterior=post,information=np.full(q,.8),reliability=np.ones(q),weight=np.full(q,.8),class_id=post.argmax(1).astype('uint8'),selected=np.zeros(q,bool),eligible=np.ones(q,bool),qc_flags=np.zeros(q,'uint16'),base_logits=ch[:,0],feedback_log_evidence=np.zeros(q),channels=ch)
    def tearDown(self):self.tmp.cleanup()
    def meta(self):return c.metadata(c.A,'toy','GRCh37',0,'test',[c.entry(c.CODE/'DESIGN.md','design')],m.PARAMS)
    def write(self):c.write_a(self.root/'A',self.v,self.s,self.r,self.a,self.p,self.g,self.meta())
    def test_roundtrip(self):
        self.write(); v,s,r,a,*_=c.read_a(self.root/'A'); self.assertEqual(c.array_hash(v),c.array_hash(self.v)); self.assertEqual((r!=self.r).nnz,0); self.assertEqual((a!=self.a).nnz,0)
    def test_corruption(self):
        self.write(); path=self.root/'A/variants.npz'; path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaises(ValueError):c.read_a(self.root/'A')
    def test_missing_field(self):
        del self.v['weight']
        with self.assertRaises(ValueError):self.write()
    def test_gt_provenance(self):
        self.write(); path=self.root/'A/manifest.json'; z=json.loads(path.read_text()); z['provenance']['gt_free']=False; path.write_text(json.dumps(z)); (path.parent/'COMPLETE.sha256').write_text(c.sha(path))
        with self.assertRaises(ValueError):c.read_a(self.root/'A')
    def test_absent_provenance(self):
        self.write(); path=self.root/'A/manifest.json'; z=json.loads(path.read_text()); del z['provenance']; path.write_text(json.dumps(z)); (path.parent/'COMPLETE.sha256').write_text(c.sha(path))
        with self.assertRaises(ValueError):c.read_a(self.root/'A')
    def test_order_mismatch(self):
        p=dict(barcode=self.s['barcode'],tumor_prior=np.full(12,.5),valid=np.ones(12,bool),region_id=np.full(12,-1,'int32'),anchor_prior=np.full(12,.5),spot_evidence=np.zeros(12))
        md=c.metadata(c.B,'toy','GRCh37',0,'test',[c.entry(c.CODE/'DESIGN.md','design')],m.PARAMS,source='none'); c.write_b(self.root/'B',p,self.g,md)
        with self.assertRaises(ValueError):c.read_b(self.root/'B',p['barcode'][::-1])
    def test_caps_contraction_and_quota(self):
        base,w,anchor,valid,ops=m.initial(self.v,self.s,self.r,self.a,self.g,'coverage_Coff')
        rng=np.random.default_rng(7)
        for _ in range(40):
            x=rng.uniform(.1,.9,12); y=rng.uniform(.1,.9,12)
            a=m.step(self.v,base,w,anchor,valid,ops,self.g,x,1,.5,3,0); b=m.step(self.v,base,w,anchor,valid,ops,self.g,y,1,.5,3,0)
            self.assertLessEqual(np.max(abs(a['tumor_prior']-b['tumor_prior'])),.75*np.max(abs(x-y))+1e-12)
            self.assertLessEqual(np.max(abs(a['feedback_log_evidence'])),.5); self.assertEqual(a['selected'].sum(),3)
    def test_prior_off_and_missing(self):
        base,w,anchor,valid,ops=m.initial(self.v,self.s,self.r,self.a,self.g,'none_Coff')
        a=m.step(self.v,base,w,anchor,valid,ops,self.g,anchor,1,0,3,0); b=m.step(self.v,base,w,anchor,valid,ops,self.g,anchor*.7,2,0,3,0)
        np.testing.assert_array_equal(a['posterior'],b['posterior']); np.testing.assert_array_equal(a['selected'],b['selected'])
        self.s['ad_available'][0]=False
        with self.assertRaises(ValueError):self.write()
    def test_fail_impossible_quota(self):
        self.v['qc_flags'][:]=1
        with self.assertRaises(ValueError):m.select(self.v,self.v['posterior'],3,0)

if __name__=='__main__':unittest.main()
