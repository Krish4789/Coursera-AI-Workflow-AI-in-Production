"""
model.py
--------
Training pipeline and prediction interface for AAVAIL daily revenue
forecasting. Supports per-country models and an aggregated 'all' model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TARGET = "revenue"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling-window features from a daily time-series DataFrame.
    Expects columns: date, revenue (and optionally purchases, views).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    for lag in [1, 7, 14]:
        df[f"revenue_lag_{lag}"] = df[TARGET].shift(lag)

    df["revenue_roll7_mean"] = df[TARGET].shift(1).rolling(7).mean()
    df["revenue_roll7_std"] = df[TARGET].shift(1).rolling(7).std()

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear

    # Optional auxiliary metrics
    for col in ["purchases", "views"]:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)

    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (everything except date and target)."""
    return [c for c in df.columns if c not in ("date", TARGET)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, country: str = "all") -> dict:
    """
    Train a GradientBoostingRegressor pipeline for the given country slice.

    Parameters
    ----------
    df      : daily aggregated DataFrame (output of data_ingestion)
    country : label used to name the saved model file

    Returns
    -------
    dict with keys: model_path, mae_train, mae_test
    """
    featured = build_features(df)
    feature_cols = get_feature_columns(featured)

    X = featured[feature_cols].values
    y = featured[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    mae_train = mean_absolute_error(y_train, pipeline.predict(X_train))
    mae_test = mean_absolute_error(y_test, pipeline.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_{country}.pkl")
    joblib.dump({"pipeline": pipeline, "feature_cols": feature_cols}, model_path)

    return {"model_path": model_path, "mae_train": mae_train, "mae_test": mae_test}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_model(country: str = "all") -> dict:
    """Load a saved model artefact for the given country."""
    model_path = os.path.join(MODEL_DIR, f"model_{country}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for country='{country}'. "
            f"Expected at: {model_path}"
        )
    return joblib.load(model_path)


def predict(features: dict, country: str = "all") -> float:
    """
    Generate a revenue prediction for a single observation.

    Parameters
    ----------
    features : dict mapping feature names to scalar values
    country  : which model to use ('all' or a specific country name)

    Returns
    -------
    Predicted revenue as a float.
    """
    artefact = load_model(country)
    pipeline: Pipeline = artefact["pipeline"]
    feature_cols: list = artefact["feature_cols"]

    # Build input array in the correct column order
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])
    prediction = pipeline.predict(X)[0]
    return float(prediction)
