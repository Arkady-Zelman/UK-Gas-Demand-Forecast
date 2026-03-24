"""Feature engineering for the weather-demand model.

The core insight is simple: UK gas demand is fundamentally a heating problem.
HDD (Heating Degree Days) explains ~70-80% of daily variation on its own.
Everything else — wind chill, lags, calendar effects — is refinement on top
of that base relationship.

We also flag the COVID lockdown period (Mar 2020 – Jun 2021) as a structural
break.  Commercial/industrial demand dropped ~15-20% during restrictions, and
the model needs to know that rather than attributing it to weather.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get
from src.features.holidays import is_bank_holiday_series

HDD_BASE: float = get("weather.hdd_base", 15.5)
CDD_BASE: float = get("weather.cdd_base", 22.0)
GAS_YEAR_START_MONTH: int = get("gas_year.start_month", 10)

# Fixed reference date for the trend feature.  Must be the same in training
# and inference — using df.min() would silently break when the forecast
# frame starts from a different date than the training set.
TREND_EPOCH = pd.Timestamp(get("date_range.start", "2020-10-01"))

# COVID lockdown windows — these materially distorted demand patterns
_COVID_START = pd.Timestamp("2020-03-23")
_COVID_END = pd.Timestamp("2021-06-21")


def gas_year(dates: pd.Series) -> pd.Series:
    """GY label: 2024 means the year starting Oct 2024."""
    return dates.dt.year - (dates.dt.month < GAS_YEAR_START_MONTH).astype(int)


def gas_quarter(dates: pd.Series) -> pd.Series:
    """Gas year quarter: Q1=Oct-Dec, Q2=Jan-Mar, Q3=Apr-Jun, Q4=Jul-Sep."""
    month = dates.dt.month
    return pd.Series(
        np.select(
            [month.isin([10, 11, 12]), month.isin([1, 2, 3]),
             month.isin([4, 5, 6]), month.isin([7, 8, 9])],
            [1, 2, 3, 4],
            default=0,
        ),
        index=dates.index,
    )


def wind_chill(temp: pd.Series, wind_kmh: pd.Series) -> pd.Series:
    """JAG/TI wind chill formula.  Only applies when wind >= 4.8 km/h and temp <= 10 C."""
    wc = (
        13.12
        + 0.6215 * temp
        - 11.37 * wind_kmh.clip(lower=4.8) ** 0.16
        + 0.3965 * temp * wind_kmh.clip(lower=4.8) ** 0.16
    )
    mask = (wind_kmh < 4.8) | (temp > 10)
    return wc.where(~mask, temp)


def build_features(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge demand + weather and build the full feature matrix.

    Returns a DataFrame ready for modelling — target is ``demand_mcm``,
    features are everything in ``FEATURE_COLS``.  Lag warm-up rows are dropped.
    """
    demand = demand.copy()
    weather = weather.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    df = demand.merge(weather, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Degree days — the primary demand driver
    df["hdd"] = (HDD_BASE - df["temp_mean"]).clip(lower=0)
    df["cdd"] = (df["temp_mean"] - CDD_BASE).clip(lower=0)

    # Wind chill amplifies heating demand on cold windy days
    df["wind_chill"] = wind_chill(df["temp_mean"], df["windspeed_max"])
    df["effective_hdd"] = (HDD_BASE - df["wind_chill"]).clip(lower=0)

    # Autoregressive lags — demand has strong day-to-day persistence
    df["demand_lag_1"] = df["demand_mcm"].shift(1)
    df["demand_lag_7"] = df["demand_mcm"].shift(7)
    df["temp_lag_1"] = df["temp_mean"].shift(1)
    df["temp_lag_2"] = df["temp_mean"].shift(2)
    df["temp_lag_3"] = df["temp_mean"].shift(3)
    df["hdd_lag_1"] = df["hdd"].shift(1)

    # Rolling context — smooths out day-to-day noise
    df["temp_roll_7"] = df["temp_mean"].rolling(7).mean()
    df["demand_roll_7"] = df["demand_mcm"].rolling(7).mean()

    # Calendar effects
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_bank_holiday"] = is_bank_holiday_series(df["date"])

    # Gas year structure
    df["gas_year"] = gas_year(df["date"])
    df["gas_quarter"] = gas_quarter(df["date"])
    df["is_winter"] = df["gas_quarter"].isin([1, 2]).astype(int)

    # Structural trend anchored to a fixed epoch so the value is identical
    # in training and live inference (never use df.min() — it drifts)
    df["days_since_start"] = (df["date"] - TREND_EPOCH).dt.days

    # COVID regime — lockdowns suppressed commercial/industrial demand
    df["is_covid"] = ((df["date"] >= _COVID_START) & (df["date"] <= _COVID_END)).astype(int)

    # Drop warm-up rows where lags/rolls are NaN
    df = df.dropna(subset=[
        "demand_lag_1", "demand_lag_7", "temp_lag_3",
        "temp_roll_7", "demand_roll_7",
    ])

    return df.reset_index(drop=True)


FEATURE_COLS = [
    "temp_mean", "temp_min", "temp_max", "windspeed_max",
    "hdd", "cdd", "wind_chill", "effective_hdd",
    "demand_lag_1", "demand_lag_7",
    "temp_lag_1", "temp_lag_2", "temp_lag_3", "hdd_lag_1",
    "temp_roll_7", "demand_roll_7",
    "day_of_week", "month", "is_weekend", "is_bank_holiday",
    "gas_quarter", "is_winter",
    "days_since_start",
    "is_covid",
]
