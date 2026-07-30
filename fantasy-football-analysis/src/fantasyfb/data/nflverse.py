"""Loaders for nflverse historical stats, cached locally as parquet.

nfl_data_py pulls from the nflverse-data GitHub releases on every call, which
is slow and network-dependent. We cache each (function, seasons) result to
``data/cache/`` so repeated app runs and analyses are fast and work offline
after the first fetch.
"""

from __future__ import annotations

import pandas as pd

from fantasyfb.config import CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str, seasons: list[int]) -> "pd.io.common.Path":
    season_key = "-".join(str(s) for s in sorted(seasons))
    return CACHE_DIR / f"{name}_{season_key}.parquet"


def _load_cached(name: str, seasons: list[int], fetch_fn, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(name, seasons)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = fetch_fn(seasons)
    df.to_parquet(path, index=False)
    return df


def load_weekly_stats(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per-player, per-week raw stats (regular season + playoffs)."""
    import nfl_data_py as nfl

    return _load_cached(
        "weekly_stats",
        seasons,
        lambda s: nfl.import_weekly_data(s, downcast=True),
        refresh=refresh,
    )


def load_seasonal_stats(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per-player, per-season aggregated stats."""
    import nfl_data_py as nfl

    return _load_cached(
        "seasonal_stats",
        seasons,
        lambda s: nfl.import_seasonal_data(s),
        refresh=refresh,
    )


def load_schedules(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Game schedules/results, used to derive matchup and defense context."""
    import nfl_data_py as nfl

    return _load_cached(
        "schedules",
        seasons,
        lambda s: nfl.import_schedules(s),
        refresh=refresh,
    )


def load_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly rosters, used to map player_id -> name/team/position/status."""
    import nfl_data_py as nfl

    return _load_cached(
        "rosters",
        seasons,
        lambda s: nfl.import_weekly_rosters(s),
        refresh=refresh,
    )
