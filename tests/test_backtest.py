"""Tests for walk-forward backtest and evaluation modules."""

import numpy as np
import pandas as pd
import pytest

from src.model.evaluate import compute_metrics, residual_analysis, fold_summary


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([100, 200, 300])
        m = compute_metrics(y, y)
        assert m["rmse_mcm"] == 0.0
        assert m["mape_pct"] == 0.0
        assert m["r2"] == 1.0

    def test_imperfect_prediction(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse_mcm"] > 0
        assert m["mape_pct"] > 0
        assert m["r2"] < 1.0

    def test_sample_count(self):
        y = np.array([1, 2, 3, 4, 5])
        m = compute_metrics(y, y)
        assert m["n_samples"] == 5


class TestResidualAnalysis:
    def test_columns_present(self):
        y_true = np.array([100, 200])
        y_pred = np.array([105, 195])
        df = residual_analysis(y_true, y_pred)
        assert "residual" in df.columns
        assert "abs_residual" in df.columns
        assert "pct_error" in df.columns

    def test_with_dates(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        df = residual_analysis(
            np.array([100, 200]), np.array([100, 200]), dates=dates
        )
        assert "date" in df.columns


class TestFoldSummary:
    def test_aggregation(self):
        folds = [
            {"rmse_mcm": 10.0, "mape_pct": 5.0, "r2": 0.95},
            {"rmse_mcm": 12.0, "mape_pct": 6.0, "r2": 0.93},
            {"rmse_mcm": 11.0, "mape_pct": 5.5, "r2": 0.94},
        ]
        df = fold_summary(folds)
        assert "Mean" in df.index
        assert "Std" in df.index
        assert abs(df.loc["Mean", "rmse_mcm"] - 11.0) < 0.01
