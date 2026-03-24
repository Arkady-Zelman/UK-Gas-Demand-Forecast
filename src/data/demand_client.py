"""Client for the National Gas Transmission Data Portal — demand data only.

Endpoint: https://data.nationalgas.com/api/find-gas-data-download
Auth:     NONE REQUIRED — fully public, open data policy.

Fetches daily gas demand (mcm/d) by Publication Object IDs (PUBOBs).
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import get

logger = logging.getLogger(__name__)

BASE_URL = "https://data.nationalgas.com/api/find-gas-data-download"
TIMEOUT = 45
MAX_DAYS_PER_REQUEST = 365

MCM_TO_GWH: float = get("project.mcm_to_gwh", 11.16)

PUBOB_IDS = {
    "NTS_Demand_Actual": "PUBOB637",
    "NTS_Demand_Forecast": "PUBOB28",

    # Residential (NDM) demand by LDZ — 13 zones (kWh, needs conversion)
    "NDM_EA": "PUBOB3755", "NDM_EM": "PUBOB3756", "NDM_NE": "PUBOB3757",
    "NDM_NO": "PUBOB3758", "NDM_NT": "PUBOB3759", "NDM_NW": "PUBOB3760",
    "NDM_SC": "PUBOB3761", "NDM_SE": "PUBOB3762", "NDM_SO": "PUBOB3763",
    "NDM_SW": "PUBOB3764", "NDM_WN": "PUBOB3765", "NDM_WS": "PUBOB3766",
    "NDM_WM": "PUBOB3767",

    # Industrial (DM) demand by LDZ — 13 zones (kWh, needs conversion)
    "DM_EA": "PUBOB3742", "DM_EM": "PUBOB3743", "DM_NE": "PUBOB3744",
    "DM_NO": "PUBOB3745", "DM_NT": "PUBOB3746", "DM_NW": "PUBOB3747",
    "DM_SC": "PUBOB3748", "DM_SE": "PUBOB3749", "DM_SO": "PUBOB3750",
    "DM_SW": "PUBOB3751", "DM_WN": "PUBOB3752", "DM_WS": "PUBOB3753",
    "DM_WM": "PUBOB3754",
}

NDM_LDZ_IDS = [k for k in PUBOB_IDS if k.startswith("NDM_")]
DM_LDZ_IDS = [k for k in PUBOB_IDS if k.startswith("DM_")]


def kwh_to_mcm(kwh: float | pd.Series) -> float | pd.Series:
    """Convert kWh to mcm.  1 mcm ~ MCM_TO_GWH * 1e6 kWh."""
    return kwh / (MCM_TO_GWH * 1e6)


class DemandClient:
    """Client for National Gas demand CSV download.  No authentication."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def _fetch_csv(
        self,
        pubob_ids: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Fetch CSV for a list of PUBOB IDs over a date range."""
        params = {
            "applicableFor": "Y",
            "dateFrom": f"{start}T00:00:00",
            "dateTo": f"{end}T23:59:59",
            "dateType": "GASDAY",
            "latestFlag": "Y",
            "ids": ",".join(pubob_ids),
            "type": "CSV",
        }
        try:
            resp = self.session.get(BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            if "text/html" in resp.headers.get("content-type", ""):
                return None
            df = pd.read_csv(io.StringIO(resp.text))
            if df.empty:
                return None
            df.columns = df.columns.str.strip()
            return df
        except Exception as exc:
            logger.warning("National Gas CSV fetch failed: %s", exc)
            return None

    def _fetch_chunked(
        self,
        pubob_ids: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Fetch in yearly chunks to stay within the API's row limit."""
        frames: list[pd.DataFrame] = []
        chunk_start = start

        while chunk_start <= end:
            chunk_end = min(
                chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end
            )
            df = self._fetch_csv(pubob_ids, chunk_start, chunk_end)
            if df is not None and not df.empty:
                frames.append(df)
            chunk_start = chunk_end + timedelta(days=1)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the standard CSV into a clean daily DataFrame."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["Applicable For"], dayfirst=True)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_demand(
        self,
        demand_type: str = "NTS",
        start: str | date = "2020-10-01",
        end: str | date | None = None,
    ) -> pd.DataFrame | None:
        """Fetch daily demand (mcm/d).

        demand_type: "NTS" (total actual), "FORECAST", "NDM" (residential),
                     "DM" (industrial).
        """
        dtype = demand_type.upper()
        if dtype == "NTS":
            ids = [PUBOB_IDS["NTS_Demand_Actual"]]
        elif dtype == "FORECAST":
            ids = [PUBOB_IDS["NTS_Demand_Forecast"]]
        elif dtype in ("LDZ", "NDM"):
            ids = [PUBOB_IDS[k] for k in NDM_LDZ_IDS]
        elif dtype in ("DM", "INDUSTRIAL"):
            ids = [PUBOB_IDS[k] for k in DM_LDZ_IDS]
        else:
            logger.warning("Unknown demand type: %s", demand_type)
            return None

        start_dt = date.fromisoformat(str(start))
        end_dt = date.fromisoformat(str(end)) if end else date.today()

        df = self._fetch_chunked(ids, start_dt, end_dt)
        if df is None:
            return None

        df = self._to_daily(df)

        if dtype in ("LDZ", "NDM", "DM", "INDUSTRIAL"):
            out = df.groupby("date", as_index=False)["Value"].sum()
            out = out.rename(columns={"Value": "demand_mcm"})
            out["demand_mcm"] = kwh_to_mcm(out["demand_mcm"])
        else:
            out = df[["date", "Value"]].rename(columns={"Value": "demand_mcm"})

        out = out.dropna(subset=["demand_mcm"])
        return out.sort_values("date").reset_index(drop=True)

    def get_all_demand(
        self,
        start: str | date = "2020-10-01",
        end: str | date | None = None,
    ) -> pd.DataFrame | None:
        """Fetch NTS total, NDM, and DM demand and return a merged DataFrame.

        Columns: date, demand_mcm, ndm_mcm, dm_mcm.
        """
        nts = self.get_demand("NTS", start, end)
        ndm = self.get_demand("NDM", start, end)
        dm = self.get_demand("DM", start, end)

        if nts is None:
            return None

        out = nts.copy()
        if ndm is not None:
            ndm = ndm.rename(columns={"demand_mcm": "ndm_mcm"})
            out = out.merge(ndm[["date", "ndm_mcm"]], on="date", how="left")
        if dm is not None:
            dm = dm.rename(columns={"demand_mcm": "dm_mcm"})
            out = out.merge(dm[["date", "dm_mcm"]], on="date", how="left")

        return out.sort_values("date").reset_index(drop=True)
