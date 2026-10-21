"""Hand-computable metric/null/guard tests; no biological labels are read."""
import unittest
import numpy as np
from .evaluate import metrics,wilson,bootstrap_counts,no_io

class EvaluatorTests(unittest.TestCase):
    def test_metrics_and_missing_labels(self):
        z=metrics(np.array([1,1,0,0,np.nan]),np.array([1,0,1,0,1],bool))
        self.assertEqual(z['f1'],.5); self.assertEqual(z['mcc'],0); self.assertEqual(z['n_evaluable'],4)
    def test_missing_is_not_zero_accuracy(self):
        self.assertTrue(np.isnan(metrics(np.array([np.nan]),np.array([True]))['f1']))
    def test_wilson(self):
        a,b=wilson(0,5); self.assertAlmostEqual(a,0); self.assertAlmostEqual(b,.4344824648,places=8)
        a,b=wilson(5,5); self.assertAlmostEqual(a,.5655175352,places=8); self.assertAlmostEqual(b,1)
    def test_paired_identical_zero(self):
        y=np.tile([0,1],12); p=(y==1); blocks=np.repeat(np.arange(6),4)
        boot,_=bootstrap_counts(y,{'a':p,'b':p.copy()},blocks,n=100)
        np.testing.assert_array_equal(boot['a']-boot['b'],np.zeros(100)); np.testing.assert_array_equal(boot['a'],np.ones(100))
    def test_runtime_guard(self):
        with self.assertRaises(PermissionError):
            with no_io():open('/etc/hosts')

if __name__=='__main__':unittest.main()
