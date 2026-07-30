"""Client for the (keyless) Sleeper public API.

Used for current player metadata (team, position, injury/practice status)
that's fresher than nflverse's roster snapshots. See
https://docs.sleeper.com/ for the underlying endpoints.
"""

from __future__ import annotations

import json

import pandas as pd
import requests

from fantasyfb.config import CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PLAYERS_CACHE_PATH = CACHE_DIR / "sleeper_players.json"

_BASE_URL = "https://api.sleeper.app/v1"


def get_nfl_state() -> dict:
    """Current NFL week/season, per Sleeper."""
    resp = requests.get(f"{_BASE_URL}/state/nfl", timeout=15)
    resp.raise_for_status()
    return resp.json()


def load_players(refresh: bool = False) -> pd.DataFrame:
    """All active/inactive NFL players Sleeper knows about.

    This endpoint returns ~5MB of JSON and Sleeper asks that it be called
    sparingly (at most once every few hours), so we always cache it locally
    and only re-fetch when ``refresh=True``.
    """
    if _PLAYERS_CACHE_PATH.exists() and not refresh:
        raw = json.loads(_PLAYERS_CACHE_PATH.read_text())
    else:
        resp = requests.get(f"{_BASE_URL}/players/nfl", timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        _PLAYERS_CACHE_PATH.write_text(json.dumps(raw))

    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index.name = "sleeper_id"
    return df.reset_index()


def load_trending_players(add_drop: str = "add", hours: int = 24, limit: int = 25) -> pd.DataFrame:
    """Players trending up (adds) or down (drops) across Sleeper leagues."""
    resp = requests.get(
        f"{_BASE_URL}/players/nfl/trending/{add_drop}",
        params={"lookback_hours": hours, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json())
