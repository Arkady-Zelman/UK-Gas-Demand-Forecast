"""Tests for data client modules."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data.demand_client import DemandClient, kwh_to_mcm, PUBOB_IDS


class TestKwhToMcm:
    def test_positive_value(self):
        result = kwh_to_mcm(11.16e6)
        assert abs(result - 1.0) < 0.01

    def test_zero(self):
        assert kwh_to_mcm(0) == 0

    def test_series(self):
        s = pd.Series([0, 11.16e6, 22.32e6])
        result = kwh_to_mcm(s)
        assert abs(result.iloc[1] - 1.0) < 0.01
        assert abs(result.iloc[2] - 2.0) < 0.01


class TestPubobRegistry:
    def test_nts_demand_actual_exists(self):
        assert "NTS_Demand_Actual" in PUBOB_IDS
        assert PUBOB_IDS["NTS_Demand_Actual"] == "PUBOB637"

    def test_ndm_ldz_count(self):
        ndm = [k for k in PUBOB_IDS if k.startswith("NDM_")]
        assert len(ndm) == 13

    def test_dm_ldz_count(self):
        dm = [k for k in PUBOB_IDS if k.startswith("DM_")]
        assert len(dm) == 13


class TestDemandClient:
    @patch("src.data.demand_client.DemandClient._fetch_csv")
    def test_get_demand_nts(self, mock_fetch):
        csv_data = pd.DataFrame({
            "Applicable For": ["01/01/2024", "02/01/2024"],
            "Data Item": ["Demand, Actual, NTS"] * 2,
            "Value": [250.0, 260.0],
        })
        mock_fetch.return_value = csv_data

        client = DemandClient()
        result = client.get_demand("NTS", "2024-01-01", "2024-01-02")

        assert result is not None
        assert "demand_mcm" in result.columns
        assert len(result) == 2

    def test_get_demand_unknown_type(self):
        client = DemandClient()
        result = client.get_demand("UNKNOWN", "2024-01-01", "2024-01-02")
        assert result is None
