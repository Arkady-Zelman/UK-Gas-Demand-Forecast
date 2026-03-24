"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    build_features,
    gas_year,
    gas_quarter,
    wind_chill,
    FEATURE_COLS,
    HDD_BASE,
)
from src.features.holidays import uk_bank_holidays, is_bank_holiday_series


class TestGasYear:
    def test_october_is_new_gas_year(self):
        dates = pd.Series(pd.to_datetime(["2024-10-01"]))
        assert gas_year(dates).iloc[0] == 2024

    def test_september_is_previous_gas_year(self):
        dates = pd.Series(pd.to_datetime(["2024-09-30"]))
        assert gas_year(dates).iloc[0] == 2023

    def test_january_is_previous_gas_year(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15"]))
        assert gas_year(dates).iloc[0] == 2023


class TestGasQuarter:
    def test_q1_is_oct_dec(self):
        dates = pd.Series(pd.to_datetime(["2024-11-15"]))
        assert gas_quarter(dates).iloc[0] == 1

    def test_q2_is_jan_mar(self):
        dates = pd.Series(pd.to_datetime(["2024-02-15"]))
        assert gas_quarter(dates).iloc[0] == 2

    def test_q3_is_apr_jun(self):
        dates = pd.Series(pd.to_datetime(["2024-05-15"]))
        assert gas_quarter(dates).iloc[0] == 3

    def test_q4_is_jul_sep(self):
        dates = pd.Series(pd.to_datetime(["2024-08-15"]))
        assert gas_quarter(dates).iloc[0] == 4


class TestWindChill:
    def test_low_wind_returns_actual_temp(self):
        temp = pd.Series([5.0])
        wind = pd.Series([3.0])
        result = wind_chill(temp, wind)
        assert result.iloc[0] == 5.0

    def test_high_temp_returns_actual(self):
        temp = pd.Series([15.0])
        wind = pd.Series([20.0])
        result = wind_chill(temp, wind)
        assert result.iloc[0] == 15.0

    def test_cold_windy_lowers_temp(self):
        temp = pd.Series([0.0])
        wind = pd.Series([30.0])
        result = wind_chill(temp, wind)
        assert result.iloc[0] < 0.0


class TestBankHolidays:
    def test_christmas_is_holiday(self):
        from datetime import date
        bh = uk_bank_holidays(2024, 2024)
        assert date(2024, 12, 25) in bh

    def test_series_detection(self):
        dates = pd.Series(pd.to_datetime(["2024-12-25", "2024-12-24"]))
        result = is_bank_holiday_series(dates)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0


class TestBuildFeatures:
    @pytest.fixture
    def sample_data(self):
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        demand = pd.DataFrame({
            "date": dates,
            "demand_mcm": np.random.uniform(150, 300, n),
        })
        weather = pd.DataFrame({
            "date": dates,
            "temp_mean": np.random.uniform(-2, 15, n),
            "temp_min": np.random.uniform(-5, 10, n),
            "temp_max": np.random.uniform(0, 20, n),
            "windspeed_max": np.random.uniform(5, 40, n),
        })
        return demand, weather

    def test_output_has_all_feature_cols(self, sample_data):
        demand, weather = sample_data
        df = build_features(demand, weather)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_hdd_is_non_negative(self, sample_data):
        demand, weather = sample_data
        df = build_features(demand, weather)
        assert (df["hdd"] >= 0).all()

    def test_no_nans_in_features(self, sample_data):
        demand, weather = sample_data
        df = build_features(demand, weather)
        for col in FEATURE_COLS:
            assert not df[col].isna().any(), f"NaN found in {col}"

    def test_warm_up_rows_dropped(self, sample_data):
        demand, weather = sample_data
        df = build_features(demand, weather)
        assert len(df) < 30
