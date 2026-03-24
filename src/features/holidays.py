"""UK bank holiday calendar for feature engineering."""

from __future__ import annotations

from datetime import date

import pandas as pd

try:
    import holidays as _hol

    def uk_bank_holidays(start_year: int, end_year: int) -> set[date]:
        """Return a set of UK bank holiday dates across the given year range."""
        bh: set[date] = set()
        for yr in range(start_year, end_year + 1):
            bh.update(_hol.UnitedKingdom(years=yr).keys())
        return bh

except ImportError:
    _STATIC_HOLIDAYS: dict[int, list[str]] = {
        2020: ["2020-01-01","2020-04-10","2020-04-13","2020-05-08",
               "2020-05-25","2020-08-31","2020-12-25","2020-12-28"],
        2021: ["2021-01-01","2021-04-02","2021-04-05","2021-05-03",
               "2021-05-31","2021-08-30","2021-12-27","2021-12-28"],
        2022: ["2022-01-03","2022-04-15","2022-04-18","2022-05-02",
               "2022-06-02","2022-06-03","2022-08-29","2022-12-26","2022-12-27"],
        2023: ["2023-01-02","2023-04-07","2023-04-10","2023-05-01",
               "2023-05-08","2023-05-29","2023-08-28","2023-12-25","2023-12-26"],
        2024: ["2024-01-01","2024-03-29","2024-04-01","2024-05-06",
               "2024-05-27","2024-08-26","2024-12-25","2024-12-26"],
        2025: ["2025-01-01","2025-04-18","2025-04-21","2025-05-05",
               "2025-05-26","2025-08-25","2025-12-25","2025-12-26"],
        2026: ["2026-01-01","2026-04-03","2026-04-06","2026-05-04",
               "2026-05-25","2026-08-31","2026-12-25","2026-12-28"],
    }

    def uk_bank_holidays(start_year: int, end_year: int) -> set[date]:
        bh: set[date] = set()
        for yr in range(start_year, end_year + 1):
            for ds in _STATIC_HOLIDAYS.get(yr, []):
                bh.add(date.fromisoformat(ds))
        return bh


def is_bank_holiday_series(dates: pd.Series) -> pd.Series:
    """Return a boolean Series indicating UK bank holidays."""
    years = dates.dt.year
    bh = uk_bank_holidays(int(years.min()), int(years.max()))
    return dates.dt.date.isin(bh).astype(int)
