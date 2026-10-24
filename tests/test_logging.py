"""
test_logging.py
---------------
Unit tests for the logger module.
Writes to a temporary log file so production logs are never touched.
"""

import os
import json
import tempfile
import unittest
from src.logger import log_prediction


class TestLogPrediction(unittest.TestCase):
    def setUp(self):
        # Use a temp file so production logs are never modified
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        )
        self.log_path = self.tmp.name
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.log_path)

    def _write_and_read(self, **kwargs) -> dict:
        log_prediction(log_file=self.log_path, **kwargs)
        with open(self.log_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertTrue(lines, "Log file is empty after writing.")
        return json.loads(lines[-1])

    def test_log_file_created(self):
        log_prediction(
            country="all",
            target_date="2024-03-15",
            features={"revenue_lag_1": 15000.0},
            prediction=15342.87,
            runtime_ms=12.4,
            log_file=self.log_path,
        )
        self.assertTrue(os.path.exists(self.log_path))

    def test_log_entry_has_required_keys(self):
        entry = self._write_and_read(
            country="all",
            target_date="2024-03-15",
            features={"revenue_lag_1": 15000.0},
            prediction=15342.87,
            runtime_ms=12.4,
        )
        for key in ("timestamp", "country", "target_date", "features",
                    "predicted_revenue", "runtime_ms"):
            self.assertIn(key, entry, f"Missing key: {key}")

    def test_log_entry_values_correct(self):
        entry = self._write_and_read(
            country="united_kingdom",
            target_date="2024-06-01",
            features={"revenue_lag_1": 9999.0},
            prediction=10500.5,
            runtime_ms=8.1,
        )
        self.assertEqual(entry["country"], "united_kingdom")
        self.assertEqual(entry["target_date"], "2024-06-01")
        self.assertAlmostEqual(entry["predicted_revenue"], 10500.5, places=2)

    def test_multiple_entries_appended(self):
        for i in range(3):
            log_prediction(
                country="all",
                target_date=f"2024-01-0{i+1}",
                features={},
                prediction=float(i * 1000),
                runtime_ms=5.0,
                log_file=self.log_path,
            )
        with open(self.log_path, "r") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
