"""XGBoost demand model.

This is the production model — everything else (LSTM, linear baseline)
is for comparison.  We chose XGBoost over LightGBM/CatBoost because it
handles the feature mix (continuous weather + categorical calendar) well
and the hyperparameters are easy to regularise without extensive tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

from src.config import get
from src.features.engineer import FEATURE_COLS

logger = logging.getLogger(__name__)


def _default_params() -> dict[str, Any]:
    return {
        "n_estimators": get("model.xgboost.n_estimators", 300),
        "max_depth": get("model.xgboost.max_depth", 4),
        "learning_rate": get("model.xgboost.learning_rate", 0.05),
        "subsample": get("model.xgboost.subsample", 0.8),
        "colsample_bytree": get("model.xgboost.colsample_bytree", 0.7),
        "min_child_weight": get("model.xgboost.min_child_weight", 10),
        "reg_alpha": get("model.xgboost.reg_alpha", 0.1),
        "reg_lambda": get("model.xgboost.reg_lambda", 5.0),
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }


class GasDemandXGB:
    """Wrapper around XGBRegressor tuned for daily gas demand."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or _default_params()
        self.model = XGBRegressor(**self.params)
        self.feature_cols = list(FEATURE_COLS)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target: str = "demand_mcm",
        early_stopping_rounds: int = 20,
        eval_df: pd.DataFrame | None = None,
    ) -> "GasDemandXGB":
        X = df[self.feature_cols].values
        y = df[target].values

        fit_kw: dict[str, Any] = {}
        if eval_df is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            fit_kw["eval_set"] = [(eval_df[self.feature_cols].values, eval_df[target].values)]
            fit_kw["verbose"] = False

        self.model.fit(X, y, **fit_kw)
        self.is_fitted = True
        logger.info("Trained on %d samples, %d features", len(X), len(self.feature_cols))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols].values)

    def feature_importance(self) -> pd.DataFrame:
        fi = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        })
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)

    def tune(
        self,
        df: pd.DataFrame,
        target: str = "demand_mcm",
        n_splits: int = 3,
    ) -> dict[str, Any]:
        """Grid search over key hyperparameters with TimeSeriesSplit CV."""
        X = df[self.feature_cols].values
        y = df[target].values

        param_grid = {
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
        }
        base = XGBRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            colsample_bytree=self.params.get("colsample_bytree", 0.7),
            min_child_weight=self.params.get("min_child_weight", 10),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 5.0),
            objective="reg:squarederror",
            n_jobs=-1, random_state=42,
        )
        gs = GridSearchCV(base, param_grid, cv=TimeSeriesSplit(n_splits=n_splits),
                          scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        self.is_fitted = True
        logger.info("Best: %s (RMSE=%.2f)", gs.best_params_, -gs.best_score_)
        return gs.best_params_

    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> "GasDemandXGB":
        self.model.load_model(str(path))
        self.is_fitted = True
        return self
