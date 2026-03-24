"""Walk-forward validation aligned to Gas Year boundaries.

Each test fold is a full Gas Year the model has never seen.  We run XGBoost
alongside two baselines (seasonal naive, linear regression) so we can report
the *lift* over naive approaches — not just the raw error.

Important: this is a 1-step-ahead backtest.  Test-row demand lags use actual
prior-day values, which is the correct framing for "given yesterday's outturn,
what is tomorrow's demand?"  It does NOT simulate the compounding error of a
recursive multi-day forecast — that's a different (harder) problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.config import get
from src.features.engineer import FEATURE_COLS, gas_year as compute_gas_year
from src.model.xgboost_model import GasDemandXGB
from src.model.baselines import seasonal_naive_predict, LinearBaseline
from src.model.evaluate import compute_metrics

logger = logging.getLogger(__name__)

GY_START_MONTH: int = get("gas_year.start_month", 10)
N_FOLDS: int = get("backtest.n_folds", 3)
MIN_TRAIN_YEARS: int = get("backtest.min_train_years", 2)


@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: dict[str, float]
    predictions: pd.DataFrame
    baseline_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BacktestResult:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def metrics_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row = {"fold": f.fold, **f.metrics}
            rows.append(row)
        df = pd.DataFrame(rows)
        mean = df.select_dtypes(include="number").mean()
        mean["fold"] = "Mean"
        std = df.select_dtypes(include="number").std()
        std["fold"] = "Std"
        return pd.concat(
            [df, pd.DataFrame([mean, std])], ignore_index=True
        )

    @property
    def all_predictions(self) -> pd.DataFrame:
        return pd.concat(
            [f.predictions for f in self.folds], ignore_index=True
        )

    @property
    def baseline_comparison(self) -> pd.DataFrame:
        """Side-by-side RMSE comparison: XGBoost vs baselines per fold."""
        rows = []
        for f in self.folds:
            row = {
                "fold": f.fold,
                "xgboost_rmse": f.metrics["rmse_mcm"],
                "xgboost_mape": f.metrics["mape_pct"],
            }
            for name, bm in f.baseline_metrics.items():
                row[f"{name}_rmse"] = bm["rmse_mcm"]
                row[f"{name}_mape"] = bm["mape_pct"]
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        mean = df.select_dtypes(include="number").mean()
        mean["fold"] = "Mean"
        return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


def walk_forward_xgb(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    min_train_years: int = MIN_TRAIN_YEARS,
    target: str = "demand_mcm",
) -> BacktestResult:
    """Expanding-window walk-forward validation with XGBoost + baselines."""
    df = df.copy()
    df["_gy"] = compute_gas_year(df["date"])
    unique_gy = sorted(df["_gy"].unique())

    if len(unique_gy) < min_train_years + 1:
        raise ValueError(
            f"Need at least {min_train_years + 1} gas years, got {len(unique_gy)}"
        )

    test_gys = unique_gy[min_train_years:]
    if n_folds < len(test_gys):
        test_gys = test_gys[-n_folds:]

    result = BacktestResult()

    for i, test_gy in enumerate(test_gys):
        train_df = df[df["_gy"] < test_gy]
        test_df = df[df["_gy"] == test_gy]

        if train_df.empty or test_df.empty:
            logger.warning("Skipping fold %d — empty split", i + 1)
            continue

        y_true = test_df[target].values

        # -- XGBoost --
        xgb = GasDemandXGB()
        xgb.fit(train_df, target=target)
        xgb_preds = xgb.predict(test_df)
        xgb_metrics = compute_metrics(y_true, xgb_preds)

        # -- Baselines --
        baseline_metrics = {}

        naive_preds = seasonal_naive_predict(train_df, test_df, target)
        baseline_metrics["seasonal_naive"] = compute_metrics(y_true, naive_preds)

        try:
            lr = LinearBaseline()
            lr.fit(train_df, target)
            lr_preds = lr.predict(test_df)
            baseline_metrics["linear"] = compute_metrics(y_true, lr_preds)
        except Exception as exc:
            logger.warning("Linear baseline failed on fold %d: %s", i + 1, exc)

        pred_df = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_true,
            "predicted": xgb_preds,
            "fold": i + 1,
        })

        fold_result = FoldResult(
            fold=i + 1,
            train_start=train_df["date"].min().date(),
            train_end=train_df["date"].max().date(),
            test_start=test_df["date"].min().date(),
            test_end=test_df["date"].max().date(),
            metrics=xgb_metrics,
            predictions=pred_df,
            baseline_metrics=baseline_metrics,
        )
        result.folds.append(fold_result)

        # Log with context
        naive_rmse = baseline_metrics.get("seasonal_naive", {}).get("rmse_mcm", 0)
        logger.info(
            "Fold %d | GY %d | XGB RMSE=%.1f | Naive RMSE=%.1f | Lift=%.0f%%",
            i + 1, test_gy, xgb_metrics["rmse_mcm"], naive_rmse,
            (1 - xgb_metrics["rmse_mcm"] / naive_rmse) * 100 if naive_rmse else 0,
        )

    return result
