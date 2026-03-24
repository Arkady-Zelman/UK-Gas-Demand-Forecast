"""Client for the Open-Meteo weather API — historical + forecast.

Historical: https://archive-api.open-meteo.com/v1/archive
Forecast:   https://api.open-meteo.com/v1/forecast
Auth:       NONE REQUIRED — fully free and open.

Returns population-weighted UK-average daily weather for representative cities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import get

logger = logging.getLogger(__name__)

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 30
MAX_DAYS_PER_REQUEST = 365

DAILY_VARS = "temperature_2m_mean,temperature_2m_min,temperature_2m_max,wind_speed_10m_max"


@dataclass
class City:
    name: str
    lat: float
    lon: float
    weight: float


def _load_cities() -> list[City]:
    raw = get("weather.cities", [])
    return [City(**c) for c in raw]


class WeatherClient:
    """Fetch daily weather from Open-Meteo and produce a UK-weighted average."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.cities = _load_cities()

    def _fetch_city_historical(
        self, city: City, start: date, end: date
    ) -> pd.DataFrame | None:
        """Fetch historical daily weather for one city, chunked by year."""
        frames: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(
                chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end
            )
            params = {
                "latitude": city.lat,
                "longitude": city.lon,
                "start_date": str(chunk_start),
                "end_date": str(chunk_end),
                "daily": DAILY_VARS,
                "timezone": "Europe/London",
            }
            try:
                resp = self.session.get(
                    HISTORICAL_URL, params=params, timeout=TIMEOUT
                )
                resp.raise_for_status()
                data = resp.json()
                daily = data.get("daily", {})
                df = pd.DataFrame(daily)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning(
                    "Open-Meteo historical fetch failed for %s: %s",
                    city.name, exc,
                )
            chunk_start = chunk_end + timedelta(days=1)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _fetch_city_forecast(self, city: City) -> pd.DataFrame | None:
        """Fetch the 16-day daily forecast for one city."""
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "daily": DAILY_VARS,
            "timezone": "Europe/London",
            "forecast_days": 16,
        }
        try:
            resp = self.session.get(FORECAST_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            df = pd.DataFrame(daily)
            return df if not df.empty else None
        except Exception as exc:
            logger.warning(
                "Open-Meteo forecast fetch failed for %s: %s", city.name, exc
            )
            return None

    def _weighted_average(
        self, city_frames: list[tuple[City, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Compute population-weighted average across cities."""
        total_weight = sum(c.weight for c, _ in city_frames)

        merged: pd.DataFrame | None = None
        for city, df in city_frames:
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"])
            w = city.weight / total_weight
            for col in [
                "temperature_2m_mean",
                "temperature_2m_min",
                "temperature_2m_max",
                "wind_speed_10m_max",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce") * w

            if merged is None:
                merged = df
            else:
                merged = merged.set_index("time").add(
                    df.set_index("time"), fill_value=0
                ).reset_index()

        if merged is None:
            return pd.DataFrame()

        merged = merged.rename(columns={
            "time": "date",
            "temperature_2m_mean": "temp_mean",
            "temperature_2m_min": "temp_min",
            "temperature_2m_max": "temp_max",
            "wind_speed_10m_max": "windspeed_max",
        })
        return merged[["date", "temp_mean", "temp_min", "temp_max", "windspeed_max"]]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_historical(
        self,
        start: str | date = "2020-10-01",
        end: str | date | None = None,
    ) -> pd.DataFrame | None:
        """Population-weighted UK-average historical daily weather."""
        start_dt = date.fromisoformat(str(start))
        end_dt = date.fromisoformat(str(end)) if end else date.today() - timedelta(days=5)

        city_frames: list[tuple[City, pd.DataFrame]] = []
        for city in self.cities:
            df = self._fetch_city_historical(city, start_dt, end_dt)
            if df is not None and not df.empty:
                city_frames.append((city, df))

        if not city_frames:
            return None

        result = self._weighted_average(city_frames)
        for col in ["temp_mean", "temp_min", "temp_max", "windspeed_max"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result.sort_values("date").reset_index(drop=True)

    def get_forecast(self) -> pd.DataFrame | None:
        """Population-weighted UK-average 16-day daily weather forecast."""
        city_frames: list[tuple[City, pd.DataFrame]] = []
        for city in self.cities:
            df = self._fetch_city_forecast(city)
            if df is not None and not df.empty:
                city_frames.append((city, df))

        if not city_frames:
            return None

        result = self._weighted_average(city_frames)
        for col in ["temp_mean", "temp_min", "temp_max", "windspeed_max"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result.sort_values("date").reset_index(drop=True)
