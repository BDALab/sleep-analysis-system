import unittest

from dashboard.logic.analysis_preparation import _selected_covariates_for_scenario
from dashboard.logic.covariates import covariate_scenario_code_key


class AnalysisPreparationCovariateSelectionTest(unittest.TestCase):
    scenario = {
        "key": "predlb-vs-hc",
        "label": "MCI-LB vs HC",
        "positive_codes": (3,),
        "negative_codes": (0,),
    }

    def test_selects_by_diagnosis_codes_not_display_label(self):
        scenario_code_key = covariate_scenario_code_key((3,), (0,))
        verification = {
            "tests": [
                self._test_row(scenario_code_key, "preDLB vs HC", "age", False),
                self._test_row(scenario_code_key, "preDLB vs HC", "gender", True),
                self._test_row(scenario_code_key, "preDLB vs HC", "education", True),
            ]
        }

        selected = _selected_covariates_for_scenario(verification, self.scenario)

        self.assertEqual(selected, ["gender", "education"])

    def test_rejects_missing_scenario_instead_of_silently_selecting_none(self):
        verification = {
            "tests": [
                self._test_row("2-vs-0", "MCI-AD vs HC", "age", False),
                self._test_row("2-vs-0", "MCI-AD vs HC", "gender", False),
                self._test_row("2-vs-0", "MCI-AD vs HC", "education", True),
            ]
        }

        with self.assertRaisesRegex(ValueError, "no records for prepared scenario"):
            _selected_covariates_for_scenario(verification, self.scenario)

    def test_rejects_incomplete_verification(self):
        scenario_code_key = covariate_scenario_code_key((3,), (0,))
        verification = {
            "tests": [
                self._test_row(scenario_code_key, "MCI-LB vs HC", "age", False),
                self._test_row(scenario_code_key, "MCI-LB vs HC", "education", True),
            ]
        }

        with self.assertRaisesRegex(ValueError, r"missing tests: \['gender'\]"):
            _selected_covariates_for_scenario(verification, self.scenario)

    @staticmethod
    def _test_row(scenario_code_key, scenario_label, covariate, control_recommended):
        return {
            "scenario_code_key": scenario_code_key,
            "scenario": scenario_label,
            "covariate": covariate,
            "control_recommended": control_recommended,
        }


if __name__ == "__main__":
    unittest.main()
