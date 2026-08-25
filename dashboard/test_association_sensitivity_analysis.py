import unittest

import numpy as np
import pandas as pd

from dashboard.logic.association_sensitivity_analysis import (
    _adjust_model_p_values,
    _attach_source_variables,
    _fit_candidate,
)


class AssociationSensitivityAnalysisTest(unittest.TestCase):
    def test_source_variables_distinguish_collection_and_stratum(self):
        data = pd.DataFrame(
            {
                "#Subject": ["COBEN-1", "HC2-2", "pre-LBD2-3"],
            }
        )

        result = _attach_source_variables(data)

        self.assertEqual(
            result["Clinical collection"].tolist(),
            ["NINR", "NU20", "NU20"],
        )
        self.assertEqual(
            result["Ascertainment stratum"].tolist(),
            ["COBEN", "HC/HC2", "pre-LBD/pre-LBD2"],
        )

    def test_age_adjusted_candidate_model_recovers_feature_effect(self):
        rng = np.random.default_rng(42)
        n = 80
        feature = rng.normal(size=n)
        age = rng.normal(70, 6, size=n)
        diagnosis = np.where(np.arange(n) % 2 == 0, "HC", "preDLB")
        outcome = (
                1.25 * feature
                + 0.08 * age
                + 0.4 * (diagnosis == "preDLB")
                + rng.normal(scale=0.25, size=n)
        )
        data = pd.DataFrame(
            {
                "Person ID": [f"P{i}" for i in range(n)],
                "Diagnosis": diagnosis,
                "Example feature": feature,
                "updrs": outcome,
                "#Gender": rng.integers(0, 2, size=n),
                "#Education": rng.normal(14, 2, size=n),
                "#Age": age,
            }
        )
        candidate = {
            "Clinical outcome": "UPDRS",
            "Feature family": "Example",
            "Redundancy cluster": "example-C01",
            "Representative feature": "Example feature",
        }
        variant = {
            "numeric_covariates": ("gender", "education", "age"),
            "categorical_covariates": (),
        }

        result, slopes = _fit_candidate(data, candidate, variant)

        self.assertEqual(result["Main model status"], "ok")
        self.assertGreater(result["Feature beta per SD"], 0.9)
        self.assertLess(result["Feature p"], 0.001)
        self.assertEqual(len(slopes), 2)

    def test_fdr_is_separate_by_model_variant(self):
        data = pd.DataFrame(
            {
                "Dataset": ["d"] * 4,
                "Analysis subset": ["All collections"] * 4,
                "Model": ["primary", "primary", "age", "age"],
                "Clinical outcome": ["UPDRS"] * 4,
                "Feature p": [0.01, 0.04, 0.02, 0.8],
                "Interaction p": [0.03, 0.5, 0.01, 0.9],
            }
        )

        result = _adjust_model_p_values(data)

        self.assertAlmostEqual(result.loc[0, "Feature FDR p"], 0.02)
        self.assertAlmostEqual(result.loc[2, "Feature FDR p"], 0.04)


if __name__ == "__main__":
    unittest.main()
