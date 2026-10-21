import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / "clustering_benchmark/beagle_default_dlpfc_2026_09_17/experiment.py"
spec = importlib.util.spec_from_file_location("dlpfc_final",path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestComparison(unittest.TestCase):
    def scores(self):
        return pd.DataFrame([
            dict(section=s,method=m,seed=r,status="ok",ari=.4+(i%2)*.01,nmi=.5+(i%2)*.02)
            for s in module.SECTIONS for i,m in enumerate(module.METHODS) for r in range(10)
        ])

    def test_bin_boundary_and_presence(self):
        matrix=pd.DataFrame([[4,0,2,1],[0,7,0,2]],index=["spot_b","spot_a"],
                            columns=["1_249999","1_250000","1_250001","2_5"])
        expected=pd.DataFrame([[1,1,1],[0,1,1]],index=matrix.index,
                              columns=["1_0","1_250000","2_0"],dtype=np.int32)
        pd.testing.assert_frame_equal(module.bin_matrix(matrix),expected)

    def test_positive_paired_gain(self):
        result,_=module.summarize_scores(self.scores())
        for value in result.values():
            self.assertAlmostEqual(value["mean_delta_ari"],.01)
            self.assertAlmostEqual(value["mean_delta_nmi"],.02)
            self.assertEqual(value["sections_improved"],12)
            self.assertTrue(value["supports_observed_representation_gain"])
            self.assertFalse(value["variant_level_germline_accuracy_proven"])

    def test_identical_inputs_do_not_establish_representation_gain(self):
        result,_=module.summarize_scores(self.scores(),True)
        self.assertFalse(result["unbinned"]["supports_observed_representation_gain"])

    def test_candidate_failure_remains_visible(self):
        scores=self.scores()
        i=scores.index[(scores.section==module.SECTIONS[0]) &
                       (scores.method==module.METHODS[1]) & (scores.seed==0)][0]
        scores.loc[i,["status","ari","nmi"]]=["numerical_failure",None,None]
        result,_=module.summarize_scores(scores)
        r=result["unbinned"]
        self.assertEqual(r["candidate_only_failures"],1)
        self.assertEqual(r["candidate_successes"],119)
        self.assertEqual(r["minimum_paired_successes"],9)
        self.assertFalse(r["supports_observed_representation_gain"])

    def test_missing_section_is_not_averaged_away(self):
        scores=self.scores()
        mask=scores.section==module.SECTIONS[0]
        scores.loc[mask,["status","ari","nmi"]]=["numerical_failure",None,None]
        result,_=module.summarize_scores(scores)
        self.assertIsNone(result["unbinned"]["mean_delta_ari"])
        self.assertFalse(result["unbinned"]["supports_observed_representation_gain"])

    def test_duplicate_and_unknown_grid_rejected(self):
        scores=self.scores()
        scores.loc[scores.seed==9,"seed"]=8
        with self.assertRaisesRegex(ValueError,"grid"):
            module.summarize_scores(scores)
        scores=self.scores()
        scores.loc[scores.section==module.SECTIONS[0],"section"]="unexpected"
        with self.assertRaisesRegex(ValueError,"grid"):
            module.summarize_scores(scores)

    def test_nonfinite_success_rejected(self):
        scores=self.scores()
        scores.loc[0,"ari"]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            module.summarize_scores(scores)

    def test_failed_run_cannot_supply_score(self):
        scores=self.scores()
        scores.loc[0,"status"]="numerical_failure"
        with self.assertRaisesRegex(ValueError,"cannot supply"):
            module.summarize_scores(scores)


if __name__ == "__main__": unittest.main()
