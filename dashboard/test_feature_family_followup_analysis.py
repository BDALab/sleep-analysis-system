import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dashboard.logic.feature_family_followup_analysis import (
    _fit_gee,
    _focused_plot_specs,
    _person_id,
    _plot_focused_association,
)


class FeatureFamilyFollowupTest(unittest.TestCase):
    def test_visit_codes_share_underlying_person(self):
        cases = {
            "HC-10": "HC-10",
            "HC2-10": "HC-10",
            "HC3-10": "HC-10",
            "pre-LBD-102": "pre-LBD-102",
            "pre-LBD2-102": "pre-LBD-102",
            "pre-LBD3-102": "pre-LBD-102",
            "COBEN-111": "COBEN-111",
        }
        for subject, expected in cases.items():
            with self.subTest(subject=subject):
                self.assertEqual(_person_id(subject), expected)

    def test_gee_recovers_positive_clustered_association(self):
        rng = np.random.default_rng(12)
        person = np.repeat([f"P{i}" for i in range(30)], 2)
        x = rng.normal(size=len(person))
        person_effect = np.repeat(rng.normal(scale=0.6, size=30), 2)
        y = 1.5 * x + person_effect + rng.normal(scale=0.2, size=len(x))

        result = _fit_gee(
            y,
            pd.DataFrame({"Feature (per SD)": x}),
            person,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cluster_count"], 30)
        self.assertGreater(
            result["coefficients"]["Feature (per SD)"],
            1.0,
        )
        self.assertLess(result["p_values"]["Feature (per SD)"], 0.001)

    def test_focused_plot_specs_are_scenario_specific(self):
        combined = _focused_plot_specs(
            "dataset-clinical-acc",
            "predlb-mci-vs-hc",
        )
        pre_dlb = _focused_plot_specs(
            "dataset-clinical",
            "predlb-vs-hc",
        )
        pre_dlb_extended = _focused_plot_specs(
            "dataset-clinical-acc",
            "predlb-vs-hc",
        )

        self.assertEqual(len(combined), 3)
        self.assertEqual(len(pre_dlb), 5)
        self.assertEqual(len(pre_dlb_extended), 3)
        self.assertTrue(
            all(
                spec["scenario"] == "predlb-vs-hc"
                for _, spec in pre_dlb
            )
        )

    def test_focused_plot_writes_png_and_pdf(self):
        rng = np.random.default_rng(9)
        person_ids = [f"P{i}" for i in range(24)]
        diagnosis = np.repeat(["HC", "preDLB"], 12)
        feature = rng.normal(size=24)
        outcome = (
                2.0
                + 0.8 * feature
                + 1.2 * (diagnosis == "preDLB")
                + rng.normal(scale=0.25, size=24)
        )
        analysis_df = pd.DataFrame(
            {
                "Person ID": person_ids,
                "Diagnosis": diagnosis,
                "diagnosis_code": np.where(diagnosis == "HC", 0, 3),
                "Example feature": feature,
                "rbdq": outcome,
            }
        )
        candidate = {
            "Clinical outcome": "RBDq",
            "Feature family": "Awakenings",
            "Redundancy cluster": "awakenings-C01",
            "Representative feature": "Example feature",
            "Pooled n": 24,
            "Pooled Spearman rho": 0.6,
            "Pooled FDR p": 0.01,
        }
        spec = {
            "outcome": "RBDq",
            "feature": "Example feature",
            "title": "Example association",
            "x_label": "Example feature",
        }

        with TemporaryDirectory() as temporary_directory:
            result = _plot_focused_association(
                analysis_df=analysis_df,
                candidate=candidate,
                adjusted={"FDR p": 0.02},
                interaction={"FDR p": 0.3},
                selected_covariates=(),
                spec=spec,
                output_dir=Path(temporary_directory),
                position=1,
                scenario_label="preDLB vs HC",
            )

            self.assertTrue(result["png_path"].exists())
            self.assertTrue(result["pdf_path"].exists())
            self.assertGreater(result["png_path"].stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
