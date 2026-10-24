"""
logger.py
---------
Structured logging utility for the AAVAIL prediction API.
Tracks: timestamp, country, target_date, predicted_value, runtime_ms.
Writes to logs/predict.log (JSON-lines format for easy parsing).
"""

import os
import json
import time
import logging
from datetime import datetime
from functools import wraps

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "predict.log")


def _get_file_logger(log_file: str = LOG_FILE) -> logging.Logger:
    """Return a logger that writes JSON-lines to log_file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("aavail.predict")
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def log_prediction(
    country: str,
    target_date: str,
    features: dict,
    prediction: float,
    runtime_ms: float,
    log_file: str = LOG_FILE,
) -> None:
    """
    Write a single prediction event as a JSON-line entry.

    Parameters
    ----------
    country     : country the prediction was made for
    target_date : date string the prediction targets
    features    : input feature dict sent to the model
    prediction  : model output value
    runtime_ms  : end-to-end request latency in milliseconds
    log_file    : override log destination (used in tests)
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "country": country,
        "target_date": target_date,
        "features": features,
        "predicted_revenue": round(prediction, 4),
        "runtime_ms": round(runtime_ms, 3),
    }
    _get_file_logger(log_file).info(json.dumps(entry))


def timed(func):
    """Decorator that injects elapsed_ms into the wrapped function's return dict."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        if isinstance(result, dict):
            result["runtime_ms"] = round(elapsed, 3)
        return result
    return wrapper
