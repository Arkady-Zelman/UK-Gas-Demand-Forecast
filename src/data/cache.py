"""Local file cache for API data.

Saves DataFrames as Parquet in data/cache/ with mtime-based staleness,
so the model can load cached data instantly and only hit APIs when refreshing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import get

logger = logging.getLogger(__name__)

CACHE_DIR = Path(get("data_paths.cache", "data/cache"))


def _cache_path(component: str) -> Path:
    safe = component.lower().replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def save(component: str, df: pd.DataFrame) -> None:
    """Write a component DataFrame to the Parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(component)
    df.to_parquet(path, index=False)
    logger.info("Cached %s -> %s (%d rows)", component, path.name, len(df))


def load(component: str, max_age_hours: float | None = None) -> pd.DataFrame | None:
    """Load cached data if it exists and is fresh enough."""
    path = _cache_path(component)
    if not path.exists():
        return None

    if max_age_hours is not None:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=max_age_hours):
            logger.info("Cache stale for %s (older than %sh)", component, max_age_hours)
            return None

    try:
        df = pd.read_parquet(path)
        logger.info("Loaded %s from cache (%d rows)", component, len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to read cache for %s: %s", component, exc)
        return None


def age_hours(component: str) -> float | None:
    """Return cache age in hours, or None if no cache file exists."""
    path = _cache_path(component)
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600


def clear(component: str | None = None) -> None:
    """Delete cached data for one component or all components."""
    if component:
        path = _cache_path(component)
        if path.exists():
            path.unlink()
    else:
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
