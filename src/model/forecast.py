"""Generate demand forecasts from live weather data.

Uses the trained XGBoost model with recent demand history for lag features.
Prediction intervals come from empirical quantiles of the walk-forward
residuals, which is more honest than assuming Gaussian errors (the residual
distribution is typically slightly right-skewed in winter due to cold snaps).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.weather_client import WeatherClient
from src.data import cache
from src.features.engineer import (
    FEATURE_COLS,
    HDD_BASE,
    CDD_BASE,
    _COVID_START,
    _COVID_END,
    wind_chill,
    gas_year,
    gas_quarter,
)
from src.features.holidays import is_bank_holiday_series
from src.model.xgboost_model import GasDemandXGB

logger = logging.getLogger(__name__)


def _build_forecast_features(
    weather_fc: pd.DataFrame,
    recent_demand: pd.DataFrame,
    recent_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build the feature matrix for the forecast horizon.

    Splices recent history with forecast weather so that lag and rolling
    features have valid values for the first few forecast days.
    """
    recent_demand = recent_demand.copy().sort_values("date")
    recent_weather = recent_weather.copy().sort_values("date")
    weather_fc = weather_fc.copy().sort_values("date")

    for frame in [weather_fc, recent_demand, recent_weather]:
        frame["date"] = pd.to_datetime(frame["date"])

    history = recent_demand[["date", "demand_mcm"]].merge(
        recent_weather[["date", "temp_mean", "temp_min", "temp_max", "windspeed_max"]],
        on="date", how="inner",
    )

    fc_rows = [{
        "date": row["date"], "demand_mcm": np.nan,
        "temp_mean": row["temp_mean"], "temp_min": row["temp_min"],
        "temp_max": row["temp_max"], "windspeed_max": row["windspeed_max"],
    } for _, row in weather_fc.iterrows()]

    combined = pd.concat([history, pd.DataFrame(fc_rows)], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    combined["hdd"] = (HDD_BASE - combined["temp_mean"]).clip(lower=0)
    combined["cdd"] = (combined["temp_mean"] - CDD_BASE).clip(lower=0)
    combined["wind_chill"] = wind_chill(combined["temp_mean"], combined["windspeed_max"])
    combined["effective_hdd"] = (HDD_BASE - combined["wind_chill"]).clip(lower=0)

    combined["demand_lag_1"] = combined["demand_mcm"].shift(1)
    combined["demand_lag_7"] = combined["demand_mcm"].shift(7)
    combined["temp_lag_1"] = combined["temp_mean"].shift(1)
    combined["temp_lag_2"] = combined["temp_mean"].shift(2)
    combined["temp_lag_3"] = combined["temp_mean"].shift(3)
    combined["hdd_lag_1"] = combined["hdd"].shift(1)
    combined["temp_roll_7"] = combined["temp_mean"].rolling(7).mean()
    combined["demand_roll_7"] = combined["demand_mcm"].rolling(7).mean()

    combined["day_of_week"] = combined["date"].dt.dayofweek
    combined["month"] = combined["date"].dt.month
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype(int)
    combined["is_bank_holiday"] = is_bank_holiday_series(combined["date"])

    combined["gas_year"] = gas_year(combined["date"])
    combined["gas_quarter"] = gas_quarter(combined["date"])
    combined["is_winter"] = combined["gas_quarter"].isin([1, 2]).astype(int)
    combined["days_since_start"] = (combined["date"] - combined["date"].min()).dt.days
    combined["is_covid"] = (
        (combined["date"] >= _COVID_START) & (combined["date"] <= _COVID_END)
    ).astype(int)

    return combined[combined["demand_mcm"].isna()].reset_index(drop=True)


def generate_forecast(
    model: GasDemandXGB,
    backtest_residuals: np.ndarray | None = None,
    recent_history_days: int = 30,
) -> pd.DataFrame | None:
    """Pull live weather forecast and generate demand predictions.

    Prediction intervals use empirical quantiles of the backtest residuals
    (5th/95th percentile) rather than assuming a Gaussian distribution.
    """
    wc = WeatherClient()
    weather_fc = wc.get_forecast()
    if weather_fc is None or weather_fc.empty:
        logger.error("Could not fetch weather forecast")
        return None

    demand_cache = cache.load("nts_demand")
    weather_cache = cache.load("weather_historical")
    if demand_cache is None or weather_cache is None:
        logger.error("No cached historical data for lag computation")
        return None

    demand_cache["date"] = pd.to_datetime(demand_cache["date"])
    weather_cache["date"] = pd.to_datetime(weather_cache["date"])

    cutoff = demand_cache["date"].max()
    recent_demand = demand_cache[
        demand_cache["date"] > cutoff - pd.Timedelta(days=recent_history_days)
    ]
    recent_weather = weather_cache[
        weather_cache["date"] > cutoff - pd.Timedelta(days=recent_history_days)
    ]

    fc_features = _build_forecast_features(weather_fc, recent_demand, recent_weather)
    if fc_features.empty:
        return None

    for col in FEATURE_COLS:
        if col not in fc_features.columns:
            fc_features[col] = 0
    fc_features[FEATURE_COLS] = fc_features[FEATURE_COLS].ffill().fillna(0)

    preds = model.predict(fc_features)

    result = pd.DataFrame({"date": fc_features["date"].values, "forecast_mcm": preds})

    for wcol in ["temp_mean", "temp_min", "temp_max", "windspeed_max", "hdd"]:
        if wcol in fc_features.columns:
            result[wcol] = fc_features[wcol].values

    # Empirical prediction intervals from backtest residuals
    if backtest_residuals is not None and len(backtest_residuals) > 20:
        q05 = float(np.percentile(backtest_residuals, 5))
        q95 = float(np.percentile(backtest_residuals, 95))
        result["lower_mcm"] = result["forecast_mcm"] + q05
        result["upper_mcm"] = result["forecast_mcm"] + q95
    else:
        # Fallback: ±10%
        result["lower_mcm"] = result["forecast_mcm"] * 0.9
        result["upper_mcm"] = result["forecast_mcm"] * 1.1

    return result
