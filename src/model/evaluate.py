"""Evaluation metrics and residual diagnostics.

We lead with RMSE and MAE because they're stable at all demand levels.
MAPE is included for intuition but should be treated with caution —
it blows up when actual demand is low (summer troughs near 130-140 mcm/d),
and a 5% MAPE on a 300 mcm/d winter day means 15 mcm, while 5% on a
140 mcm/d summer day means only 7 mcm.  The absolute error matters more
for operational decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in percent.  Caution: unstable when y_true contains small values."""
    return float(mean_absolute_percentage_error(y_true, y_pred)) * 100


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression metrics for demand forecasting."""
    return {
        "rmse_mcm": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mae_mcm": float(np.mean(np.abs(y_true - y_pred))),
        "n_samples": len(y_true),
    }


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Detailed residual breakdown for diagnostics."""
    residuals = y_true - y_pred
    out = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
        "pct_error": np.where(y_true != 0, (residuals / y_true) * 100, 0),
    })
    if dates is not None:
        out.insert(0, "date", dates.values)
    return out


def residual_acf(residuals: np.ndarray, nlags: int = 21) -> np.ndarray:
    """Compute the autocorrelation function of residuals.

    Significant autocorrelation at low lags suggests the model is missing
    temporal structure — e.g. if it under-predicts Monday it probably
    under-predicts Tuesday too.
    """
    n = len(residuals)
    r = residuals - residuals.mean()
    c0 = np.sum(r ** 2) / n
    acf = np.array([np.sum(r[:n - k] * r[k:]) / (n * c0) for k in range(nlags + 1)])
    return acf


def fold_summary(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(fold_metrics)
    df.index = [f"Fold {i+1}" for i in range(len(df))]
    mean_row = df.mean(numeric_only=True)
    mean_row.name = "Mean"
    std_row = df.std(numeric_only=True)
    std_row.name = "Std"
    return pd.concat([df, mean_row.to_frame().T, std_row.to_frame().T])
