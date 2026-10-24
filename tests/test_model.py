"""
test_model.py
-------------
Unit tests for model training, persistence, and prediction.
Trains a lightweight model on synthetic data so no real data files are needed.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd

# Override MODEL_DIR before importing model module
TEST_MODEL_DIR = "/tmp/aavail_test_models"
os.environ["MODEL_DIR_OVERRIDE"] = TEST_MODEL_DIR

import src.model as model_module

model_module.MODEL_DIR = TEST_MODEL_DIR


def _make_synthetic_df(n: int = 120) -> pd.DataFrame:
    """Generate a synthetic daily revenue DataFrame."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "revenue": rng.uniform(10_000, 20_000, n),
        "purchases": rng.integers(50, 200, n).astype(float),
        "views": rng.integers(500, 2000, n).astype(float),
    })


class TestBuildFeatures(unittest.TestCase):
    def test_output_has_lag_columns(self):
        df = _make_synthetic_df()
        featured = model_module.build_features(df)
        for lag in [1, 7, 14]:
            self.assertIn(f"revenue_lag_{lag}", featured.columns)

    def test_no_nulls_after_build(self):
        df = _make_synthetic_df()
        featured = model_module.build_features(df)
        self.assertFalse(featured.isnull().any().any())


class TestTrainAndPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)
        df = _make_synthetic_df()
        cls.result = model_module.train(df, country="test_country")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEST_MODEL_DIR, ignore_errors=True)

    def test_model_file_created(self):
        self.assertTrue(os.path.exists(self.result["model_path"]))

    def test_mae_metrics_present(self):
        self.assertIn("mae_train", self.result)
        self.assertIn("mae_test", self.result)
        self.assertGreaterEqual(self.result["mae_test"], 0)

    def test_predict_returns_float(self):
        artefact = model_module.load_model("test_country")
        feature_cols = artefact["feature_cols"]
        features = {col: 1.0 for col in feature_cols}
        prediction = model_module.predict(features, country="test_country")
        self.assertIsInstance(prediction, float)

    def test_predict_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            model_module.predict({}, country="nonexistent_country")


if __name__ == "__main__":
    unittest.main()
