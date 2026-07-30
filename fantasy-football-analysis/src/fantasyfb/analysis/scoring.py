"""Fantasy point calculation from raw nflverse weekly stats.

We compute points from raw stat columns rather than relying on nflverse's
built-in ``fantasy_points`` / ``fantasy_points_ppr`` columns so that any
scoring settings (standard / half-PPR / PPR / a league's custom rules) can be
applied consistently across the whole app.
"""

from __future__ import annotations

import pandas as pd

from fantasyfb.config import SCORING_PRESETS

# Raw stat columns this calculation depends on, and the default value to use
# if a column is missing from a given dataset (e.g. older seasons).
_REQUIRED_COLUMNS = {
    "passing_yards": 0,
    "passing_tds": 0,
    "interceptions": 0,
    "passing_2pt_conversions": 0,
    "rushing_yards": 0,
    "rushing_tds": 0,
    "rushing_fumbles_lost": 0,
    "rushing_2pt_conversions": 0,
    "receptions": 0,
    "receiving_yards": 0,
    "receiving_tds": 0,
    "receiving_fumbles_lost": 0,
    "receiving_2pt_conversions": 0,
    "sack_fumbles_lost": 0,
    "special_teams_tds": 0,
}


def resolve_scoring(scoring: str | dict) -> dict:
    """Accept either a preset name ("ppr", "half_ppr", "standard") or a
    fully custom scoring dict, and return the scoring dict to use."""
    if isinstance(scoring, str):
        try:
            return SCORING_PRESETS[scoring]
        except KeyError:
            raise ValueError(
                f"Unknown scoring preset '{scoring}'. "
                f"Choose from {list(SCORING_PRESETS)} or pass a custom dict."
            )
    return scoring


def compute_fantasy_points(df: pd.DataFrame, scoring: str | dict = "ppr") -> pd.Series:
    """Compute fantasy points per row of a weekly stats dataframe.

    Parameters
    ----------
    df: weekly stats dataframe (as returned by
        ``fantasyfb.data.nflverse.load_weekly_stats``).
    scoring: a preset name or a custom scoring dict (see
        ``fantasyfb.config.SCORING_PRESETS`` for the expected keys).
    """
    rules = resolve_scoring(scoring)

    stats = {}
    for col, default in _REQUIRED_COLUMNS.items():
        stats[col] = df[col] if col in df.columns else default

    fumbles_lost = (
        stats["rushing_fumbles_lost"]
        + stats["receiving_fumbles_lost"]
        + stats["sack_fumbles_lost"]
    )
    two_pt_conversions = (
        stats["passing_2pt_conversions"]
        + stats["rushing_2pt_conversions"]
        + stats["receiving_2pt_conversions"]
    )

    points = (
        stats["passing_yards"] * rules["pass_yd"]
        + stats["passing_tds"] * rules["pass_td"]
        + stats["interceptions"] * rules["pass_int"]
        + stats["rushing_yards"] * rules["rush_yd"]
        + stats["rushing_tds"] * rules["rush_td"]
        + stats["receptions"] * rules["rec"]
        + stats["receiving_yards"] * rules["rec_yd"]
        + stats["receiving_tds"] * rules["rec_td"]
        + fumbles_lost * rules["fumble_lost"]
        + two_pt_conversions * rules["two_pt"]
        + stats["special_teams_tds"] * rules.get("return_td", 6)
    )
    return points.astype(float).rename("fantasy_points")
