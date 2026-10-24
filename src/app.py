"""
app.py
------
Flask REST API for AAVAIL revenue prediction.

Endpoints
---------
GET  /health          — liveness check
POST /predict         — revenue prediction for a country + date
"""

import time
from flask import Flask, request, jsonify
from src.model import predict
from src.logger import log_prediction

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Expected JSON body
    ------------------
    {
        "country":     "united_kingdom",   // or "all"
        "target_date": "2024-03-15",
        "features": {
            "revenue_lag_1":      15000.0,
            "revenue_lag_7":      14200.0,
            "revenue_lag_14":     13800.0,
            "revenue_roll7_mean": 14500.0,
            "revenue_roll7_std":  400.0,
            "day_of_week":        4,
            "month":              3,
            "day_of_year":        74
        }
    }

    Returns
    -------
    {
        "country":           "united_kingdom",
        "target_date":       "2024-03-15",
        "predicted_revenue": 15342.87,
        "runtime_ms":        12.4
    }
    """
    body = request.get_json(silent=True)

    # --- Input validation ---
    if not body:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [f for f in ("country", "target_date", "features") if f not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    if not isinstance(body["features"], dict):
        return jsonify({"error": "'features' must be a JSON object."}), 400

    country = body["country"]
    target_date = body["target_date"]
    features = body["features"]

    # --- Prediction ---
    start = time.perf_counter()
    try:
        predicted = predict(features, country=country)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    runtime_ms = (time.perf_counter() - start) * 1000

    # --- Logging ---
    log_prediction(
        country=country,
        target_date=target_date,
        features=features,
        prediction=predicted,
        runtime_ms=runtime_ms,
    )

    return jsonify({
        "country": country,
        "target_date": target_date,
        "predicted_revenue": round(predicted, 4),
        "runtime_ms": round(runtime_ms, 3),
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
