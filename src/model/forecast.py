"""Generate demand forecasts from live weather data.

Prediction is recursive: we predict one day at a time and feed each day's
output back as the demand_lag_1 / demand_lag_7 / demand_roll_7 for the
next day.  This matters because without it, demand lags go stale after
day 1 and the 16-day strip isn't a true sequential forecast — it's just
day-1 repeated with frozen inputs.

Prediction intervals come from empirical quantiles of the walk-forward
residuals (not Gaussian).
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


def _build_combined_frame(
    weather_fc: pd.DataFrame,
    recent_demand: pd.DataFrame,
    recent_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Splice recent history with forecast weather and compute all
    features that do NOT depend on demand (weather, calendar, etc.).

    Demand-dependent features (lags, rolling avg) are left for the
    recursive loop in generate_forecast because each day's prediction
    must feed into the next day's inputs.

    Returns (combined_frame, first_forecast_index).
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

    fc_start = int(combined["demand_mcm"].isna().idxmax())

    # -- Weather-derived features (fully known for the entire horizon) --
    combined["hdd"] = (HDD_BASE - combined["temp_mean"]).clip(lower=0)
    combined["cdd"] = (combined["temp_mean"] - CDD_BASE).clip(lower=0)
    combined["wind_chill"] = wind_chill(combined["temp_mean"], combined["windspeed_max"])
    combined["effective_hdd"] = (HDD_BASE - combined["wind_chill"]).clip(lower=0)

    # Temperature lags/rolls — all based on weather data, no demand dependency
    combined["temp_lag_1"] = combined["temp_mean"].shift(1)
    combined["temp_lag_2"] = combined["temp_mean"].shift(2)
    combined["temp_lag_3"] = combined["temp_mean"].shift(3)
    combined["hdd_lag_1"] = combined["hdd"].shift(1)
    combined["temp_roll_7"] = combined["temp_mean"].rolling(7).mean()

    # Calendar / structural features
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

    # Seed demand lags for the historical portion only
    combined["demand_lag_1"] = combined["demand_mcm"].shift(1)
    combined["demand_lag_7"] = combined["demand_mcm"].shift(7)
    combined["demand_roll_7"] = combined["demand_mcm"].rolling(7).mean()

    return combined, fc_start


def generate_forecast(
    model: GasDemandXGB,
    backtest_residuals: np.ndarray | None = None,
    recent_history_days: int = 30,
) -> pd.DataFrame | None:
    """Pull live weather forecast and predict demand recursively.

    Each forecast day is predicted individually.  The prediction is written
    back into the demand series so the next day's lag features reflect it,
    producing a proper day-ahead chain rather than stale-lag extrapolation.
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

    combined, fc_start = _build_combined_frame(weather_fc, recent_demand, recent_weather)

    if fc_start >= len(combined):
        return None

    # --- Recursive day-ahead prediction ---
    # For each forecast day, update demand lags from the growing prediction
    # series, predict, then write the result back before moving to day+1.
    for i in range(fc_start, len(combined)):
        if i >= 1:
            combined.at[i, "demand_lag_1"] = combined.at[i - 1, "demand_mcm"]
        if i >= 7:
            combined.at[i, "demand_lag_7"] = combined.at[i - 7, "demand_mcm"]

        # Rolling 7-day demand average using the 7 days preceding today
        roll_start = max(0, i - 7)
        roll_vals = combined.loc[roll_start : i - 1, "demand_mcm"].dropna()
        combined.at[i, "demand_roll_7"] = roll_vals.mean() if len(roll_vals) > 0 else 0.0

        # Ensure all feature columns exist and fill remaining gaps
        for col in FEATURE_COLS:
            if col not in combined.columns:
                combined[col] = 0
        row = combined.loc[[i], FEATURE_COLS].fillna(0)

        pred = float(model.predict(row)[0])
        combined.at[i, "demand_mcm"] = pred

    # --- Assemble output ---
    fc = combined.iloc[fc_start:].reset_index(drop=True)

    result = pd.DataFrame({
        "date": fc["date"].values,
        "forecast_mcm": fc["demand_mcm"].values,
    })

    for wcol in ["temp_mean", "temp_min", "temp_max", "windspeed_max", "hdd"]:
        if wcol in fc.columns:
            result[wcol] = fc[wcol].values

    if backtest_residuals is not None and len(backtest_residuals) > 20:
        q05 = float(np.percentile(backtest_residuals, 5))
        q95 = float(np.percentile(backtest_residuals, 95))
        result["lower_mcm"] = result["forecast_mcm"] + q05
        result["upper_mcm"] = result["forecast_mcm"] + q95
    else:
        result["lower_mcm"] = result["forecast_mcm"] * 0.9
        result["upper_mcm"] = result["forecast_mcm"] * 1.1

    return result
