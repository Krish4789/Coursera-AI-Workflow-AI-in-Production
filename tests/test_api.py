"""
test_api.py
-----------
Unit tests for the Flask /health and /predict endpoints.
Uses unittest.mock to isolate the model layer.
"""

import json
import unittest
from unittest.mock import patch
from src.app import app


VALID_PAYLOAD = {
    "country": "all",
    "target_date": "2024-03-15",
    "features": {
        "revenue_lag_1": 15000.0,
        "revenue_lag_7": 14200.0,
        "revenue_lag_14": 13800.0,
        "revenue_roll7_mean": 14500.0,
        "revenue_roll7_std": 400.0,
        "day_of_week": 4,
        "month": 3,
        "day_of_year": 74,
    },
}


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("src.app.log_prediction")
    @patch("src.app.predict", return_value=15342.87)
    def test_valid_payload_returns_200(self, mock_predict, mock_log):
        resp = self.client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("predicted_revenue", body)
        self.assertEqual(body["country"], "all")
        mock_predict.assert_called_once()
        mock_log.assert_called_once()

    def test_missing_body_returns_400(self):
        resp = self.client.post("/predict", data="", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_missing_fields_returns_400(self):
        resp = self.client.post(
            "/predict",
            data=json.dumps({"country": "all"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Missing required fields", resp.get_json()["error"])

    def test_invalid_features_type_returns_400(self):
        bad = {**VALID_PAYLOAD, "features": "not_a_dict"}
        resp = self.client.post(
            "/predict",
            data=json.dumps(bad),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    @patch("src.app.predict", side_effect=FileNotFoundError("No model found"))
    def test_missing_model_returns_404(self, _):
        resp = self.client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
