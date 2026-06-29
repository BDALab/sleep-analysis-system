import unittest

import numpy as np
import pandas as pd

from dashboard.logic.classification_grouped_statistics import (
    FoldwiseCovariateResidualizer,
)
from dashboard.logic.covariates import CovariateController


class CovariateResidualizationTest(unittest.TestCase):
    def test_controller_keeps_constant_feature_exactly_zero(self):
        features = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0],
                "variable": [1.0, 2.0, 4.0, 8.0],
            }
        )
        covariates = pd.DataFrame({"education": [10, 12, 14, 16]})

        adjusted = CovariateController().fit_transform(
            features,
            covariates,
        )

        self.assertTrue((adjusted["constant"] == 0.0).all())

    def test_foldwise_residualizer_keeps_constant_feature_exactly_zero(self):
        values = np.array(
            [
                [1.0, 1.0, 10.0],
                [1.0, 2.0, 12.0],
                [1.0, 4.0, 14.0],
                [1.0, 8.0, 16.0],
            ]
        )
        transformer = FoldwiseCovariateResidualizer(n_covariates=1)

        adjusted = transformer.fit_transform(values)

        self.assertTrue(np.array_equal(adjusted[:, 0], np.zeros(4)))


if __name__ == "__main__":
    unittest.main()
