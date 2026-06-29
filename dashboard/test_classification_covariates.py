import unittest

from dashboard.logic.classification_covariates import (
    build_scenario_covariate_mapping,
    resolve_adjustment_columns,
    validate_scenario_covariate_mapping,
)

SCENARIOS = (
    ((3,), (0,)),
    ((3, 2), (0,)),
    ((2,), (0,)),
)


class ClassificationCovariatesTest(unittest.TestCase):
    def test_builds_complete_scenario_specific_mapping(self):
        preparation = {
            "scenarios": [
                {
                    "positive_codes": [3],
                    "negative_codes": [0],
                    "selected_covariates": ["gender", "education"],
                },
                {
                    "positive_codes": [3, 2],
                    "negative_codes": [0],
                    "selected_covariates": ["education"],
                },
                {
                    "positive_codes": [2],
                    "negative_codes": [0],
                    "selected_covariates": ["education"],
                },
            ]
        }

        mapping = build_scenario_covariate_mapping(preparation, SCENARIOS)

        self.assertEqual(mapping[((3,), (0,))], ("gender", "education"))
        self.assertEqual(mapping[((3, 2), (0,))], ("education",))
        self.assertEqual(mapping[((2,), (0,))], ("education",))

    def test_rejects_missing_scenario_instead_of_using_no_covariates(self):
        with self.assertRaisesRegex(ValueError, "Missing scenarios"):
            validate_scenario_covariate_mapping(
                {
                    ((3,), (0,)): ("gender", "education"),
                    ((3, 2), (0,)): ("education",),
                },
                SCENARIOS,
            )

    def test_rejects_unknown_covariate(self):
        mapping = {
            ((3,), (0,)): ("gender", "education", "site"),
            ((3, 2), (0,)): ("education",),
            ((2,), (0,)): ("education",),
        }

        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_scenario_covariate_mapping(mapping, SCENARIOS)

    def test_resolves_only_selected_adjustment_columns(self):
        columns = ["feature", "#Age", "#Gender", "#Education"]

        resolved = resolve_adjustment_columns(
            selected_covariates=("gender", "education"),
            available_columns=columns,
            covariate_columns={
                "age": "#Age",
                "gender": "#Gender",
                "education": "#Education",
            },
        )

        self.assertEqual(resolved, ["#Gender", "#Education"])

    def test_rejects_missing_selected_adjustment_column(self):
        with self.assertRaisesRegex(ValueError, "missing selected adjustment columns"):
            resolve_adjustment_columns(
                selected_covariates=("education",),
                available_columns=["feature", "#Age", "#Gender"],
                covariate_columns={
                    "age": "#Age",
                    "gender": "#Gender",
                    "education": "#Education",
                },
            )


if __name__ == "__main__":
    unittest.main()
