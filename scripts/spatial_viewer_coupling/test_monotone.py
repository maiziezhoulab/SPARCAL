"""Regression tests for the representation artifact and paired spatial null."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from . import model, evaluate as legacy, monotone_controls as controls
from .evaluate_monotone import null_f1, permutation_indices, replay_processing, check_evaluation_records
from . import contracts as ct, audit


class MonotoneTests(unittest.TestCase):
    def setUp(self):
        n = 120
        self.g = sparse.diags([np.ones(n-1), np.ones(n-1)], [-1, 1], shape=(n, n), format='csr', dtype=np.uint8)
        self.u = np.random.default_rng(43).lognormal(6, 1.8, n)
        self.c = model.smooth(self.u, self.g)

    def test_nulls_preserve_ties_and_ranks(self):
        for c in [self.c, np.ones(30), np.repeat([0., 1., 10.], [21, 7, 2]), np.array([0., 100.])]:
            nulls, _ = controls.coverage_controls(c)
            for value in nulls.values():
                np.testing.assert_array_equal(rankdata(value), rankdata(c))

    def test_spline_is_declared_interpolant(self):
        nulls, spec = controls.coverage_controls(self.c)
        self.assertEqual(spec['knot_probabilities'], [0., .1, .25, .5, .75, .9, 1.])
        self.assertFalse(spec['gt_fit'])
        self.assertAlmostEqual(nulls['spline_coverage'].min(), 0.)
        self.assertAlmostEqual(nulls['spline_coverage'].max(), 1.)
        with self.assertRaises(ValueError):
            controls.coverage_controls(np.array([-1., 2.]))

    def test_shared_posterior_null_path_and_coverage_smoothing(self):
        for name in ['smooth_coverage', 'rank_coverage', 'spline_coverage', 'coverage_Coff_k0']:
            actual = controls.detector_scores(name, self.c, self.u, self.g)
            np.testing.assert_array_equal(actual['raw'], self.c)
            np.testing.assert_array_equal(actual['coverage_residual'], model.smooth(model.residual(self.c, self.u), self.g))
        self.assertFalse(np.allclose(actual['coverage_residual'], model.residual(self.c, self.u)))

    def test_burden_path_explicit_and_preserved(self):
        result = controls.detector_scores('baseline', self.c, self.u, self.g)
        np.testing.assert_array_equal(result['raw'], model.smooth(self.c, self.g))
        self.assertEqual(controls.transform_chain('baseline')['raw_chain'], 'H(x)')
        self.assertEqual(controls.transform_chain('coverage_Coff_k0')['raw_chain'], 'x')

    def test_affine_rank_prior_residual_is_same_representation(self):
        r = rankdata(self.c)
        prior = .1+.8*(r-.5)/len(r)
        a = controls.detector_scores('rank_coverage', r, self.u, self.g)['coverage_residual']
        b = controls.detector_scores('coverage_Coff_k0', prior, self.u, self.g)['coverage_residual']
        np.testing.assert_allclose(b, .8/len(r)*a, atol=1e-14)
        np.testing.assert_array_equal(rankdata(prior), rankdata(self.c))

    def test_label_context_replay_is_exact(self):
        _, _, _, rows = replay_processing({'smooth_coverage': self.c, 'baseline': np.arange(120.)}, self.u, self.g, np.tile([0., 1.], 60))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['pass_exact'] and r['io_guard_pass'] for r in rows))
        self.assertEqual(len({r['numerical_sha256'] for r in rows}), 1)
        with self.assertRaises(ValueError):
            controls.prepare_scores({'smooth_coverage': self.c+1}, self.u, self.g)

    def test_evaluator_gt_provenance_does_not_relax_inference_guard(self):
        with tempfile.TemporaryDirectory(dir=ct.OUT/'logs') as folder:
            p = Path(folder)/'tumor_groups.csv'
            p.write_text('synthetic fixture, no real labels\n')
            rec = ct.entry(p, 'evaluation_only_GT')
            check_evaluation_records([rec])
            with self.assertRaises(ValueError):
                audit.check_records([rec])
            with self.assertRaises(ValueError):
                check_evaluation_records([{**rec, 'role': 'inference'}])
            p.write_text('tampered\n')
            with self.assertRaises(ValueError):
                check_evaluation_records([rec])

    def test_vectorized_null_matches_scalar_metrics_with_missing_labels(self):
        xy = np.column_stack([np.arange(120)%10, np.arange(120)//10])
        indices, tail = permutation_indices(xy, n=12)
        self.assertEqual(tail, 24)
        for ix in indices:
            np.testing.assert_array_equal(np.sort(ix), np.arange(120))
        y = np.tile([0., 1., np.nan], 40)
        preds = dict(a=np.arange(120)%7 < 3, b=np.arange(120)%5 < 2)
        actual = null_f1(y, preds, indices)
        expected = np.array([[legacy.metrics(y[ix], p)['f1'] for p in preds.values()] for ix in indices])
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
