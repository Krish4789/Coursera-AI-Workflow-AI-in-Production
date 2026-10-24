"""
data_ingestion.py
-----------------
Loads raw JSON/CSV AAVAIL data logs and aggregates daily time-series
metrics (revenue, purchases, views) per country.
"""

import os
import json
import glob
import pandas as pd


METRICS = ["revenue", "purchases", "views"]


def load_json_logs(data_dir: str) -> pd.DataFrame:
    """Load all JSON log files from data_dir into a single DataFrame."""
    records = []
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
            # Support both a list of records or a single dict
            records.extend(data if isinstance(data, list) else [data])
    return pd.DataFrame(records)


def load_csv_logs(data_dir: str) -> pd.DataFrame:
    """Load all CSV log files from data_dir into a single DataFrame."""
    frames = [
        pd.read_csv(p)
        for p in glob.glob(os.path.join(data_dir, "*.csv"))
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, parse dates, drop nulls in key fields."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

    # Keep only rows that have at least one metric column present
    present_metrics = [m for m in METRICS if m in df.columns]
    if not present_metrics:
        raise ValueError(f"Dataset must contain at least one of: {METRICS}")

    df = df.dropna(subset=["date"])
    for m in present_metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)

    return df


def aggregate_daily(df: pd.DataFrame, country: str = None) -> pd.DataFrame:
    """
    Aggregate metrics to daily granularity.

    Parameters
    ----------
    df      : cleaned DataFrame with 'date' and optional 'country' columns
    country : if provided, filter to that country before aggregating;
              if None, aggregate across all countries

    Returns
    -------
    DataFrame indexed by date with summed metric columns.
    """
    if country and "country" in df.columns:
        df = df[df["country"].str.lower() == country.lower()]

    present_metrics = [m for m in METRICS if m in df.columns]
    daily = (
        df.groupby("date")[present_metrics]
        .sum()
        .sort_index()
        .reset_index()
    )
    return daily


def load_and_prepare(data_dir: str, country: str = None) -> pd.DataFrame:
    """
    End-to-end pipeline: load → clean → aggregate.

    Parameters
    ----------
    data_dir : directory containing raw JSON/CSV files
    country  : optional country filter

    Returns
    -------
    Daily aggregated DataFrame ready for feature engineering / modelling.
    """
    json_df = load_json_logs(data_dir)
    csv_df = load_csv_logs(data_dir)

    frames = [f for f in [json_df, csv_df] if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No JSON or CSV files found in '{data_dir}'.")

    raw = pd.concat(frames, ignore_index=True)
    cleaned = clean_dataframe(raw)
    return aggregate_daily(cleaned, country=country)
