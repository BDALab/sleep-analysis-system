import os
import unittest
from unittest.mock import patch

from dashboard.logic.xgboost_runtime import (
    configure_xgboost_params,
    xgboost_device,
    xgboost_threads,
)


class XGBoostRuntimeTest(unittest.TestCase):
    @patch("dashboard.logic.xgboost_runtime.platform.system", return_value="Darwin")
    def test_auto_device_uses_cpu_on_macos(self, _):
        with patch.dict(os.environ, {"GENEACTIV_XGB_DEVICE": "auto"}):
            self.assertEqual(xgboost_device(), "cpu")

    @patch("dashboard.logic.xgboost_runtime.platform.system", return_value="Darwin")
    def test_cuda_is_rejected_on_macos(self, _):
        with patch.dict(os.environ, {"GENEACTIV_XGB_DEVICE": "cuda"}):
            with self.assertRaisesRegex(ValueError, "unavailable on macOS"):
                xgboost_device()

    def test_configure_replaces_windows_gpu_settings(self):
        with patch.dict(
                os.environ,
                {
                    "GENEACTIV_XGB_DEVICE": "cpu",
                    "GENEACTIV_XGB_THREADS": "4",
                },
        ):
            configured = configure_xgboost_params(
                {
                    "seed": 17,
                    "gpu_id": 0,
                    "predictor": "gpu_predictor",
                    "use_label_encoder": False,
                }
            )

        self.assertEqual(configured["device"], "cpu")
        self.assertEqual(configured["tree_method"], "hist")
        self.assertEqual(configured["n_jobs"], 4)
        self.assertEqual(configured["random_state"], 17)
        self.assertNotIn("gpu_id", configured)
        self.assertNotIn("predictor", configured)
        self.assertNotIn("use_label_encoder", configured)

    def test_thread_override(self):
        with patch.dict(os.environ, {"GENEACTIV_XGB_THREADS": "3"}):
            self.assertEqual(xgboost_threads(), 3)


if __name__ == "__main__":
    unittest.main()
