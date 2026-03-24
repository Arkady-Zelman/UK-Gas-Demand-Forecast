"""Baseline models for comparison.

Any ML model should beat these comfortably, otherwise you're over-engineering.
We use two baselines:

1. Seasonal naive — "same day last year" (or closest available day).
   This is the dumbest plausible forecast in energy and is surprisingly
   hard to beat in summer when demand is flat.

2. Linear regression on the same feature set — shows how much of the
   lift comes from XGBoost's non-linearity vs the features themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.engineer import FEATURE_COLS


def seasonal_naive_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "demand_mcm",
) -> np.ndarray:
    """Predict each test day using the same calendar day from the most recent
    training year.  Falls back to same-month average if no exact match."""
    train = train_df[["date", target]].copy()
    train["date"] = pd.to_datetime(train["date"])
    train["md"] = train["date"].dt.strftime("%m-%d")

    # Build a lookup: month-day -> most recent year's value
    lookup = train.sort_values("date").drop_duplicates("md", keep="last")
    lookup = lookup.set_index("md")[target]

    # Monthly fallback
    train["month"] = train["date"].dt.month
    monthly_avg = train.groupby("month")[target].mean()

    test = test_df[["date"]].copy()
    test["date"] = pd.to_datetime(test["date"])
    test["md"] = test["date"].dt.strftime("%m-%d")
    test["month"] = test["date"].dt.month

    preds = test["md"].map(lookup)
    # Fill gaps (e.g. Feb 29) with monthly average
    missing = preds.isna()
    if missing.any():
        preds[missing] = test.loc[missing, "month"].map(monthly_avg)

    return preds.values


class LinearBaseline:
    """Ridge regression on the same features — isolates the value of
    XGBoost's non-linear splits from the feature engineering itself."""

    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_cols = list(FEATURE_COLS)

    def fit(self, df: pd.DataFrame, target: str = "demand_mcm") -> "LinearBaseline":
        X = self.scaler.fit_transform(df[self.feature_cols].values)
        self.model.fit(X, df[target].values)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.feature_cols].values)
        return self.model.predict(X)
