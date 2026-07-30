"""Phase 1: player scoring trends, consistency, and matchup difficulty.

Everything here operates on a weekly stats dataframe that already has a
``fantasy_points`` column (see ``fantasyfb.analysis.scoring``).
"""

from __future__ import annotations

import pandas as pd

from fantasyfb.config import BOOM_BUST_THRESHOLDS

PLAYER_KEY = "player_id"
GAME_KEYS = ["season", "week"]


def add_position_finish_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``position_rank``: the player's fantasy-points rank among their
    position that week (1 = best). Ties broken by points, no rank gaps."""
    df = df.copy()
    df["position_rank"] = (
        df.groupby(GAME_KEYS + ["position"])["fantasy_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return df


def add_rolling_average(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Add ``rolling_avg_points``: trailing mean fantasy points over the last
    ``window`` games played (including the current game), per player,
    ordered by season/week. A player's bye week (no row) doesn't count
    against the window since we only roll over games actually played.
    """
    df = df.sort_values(GAME_KEYS).copy()
    df["rolling_avg_points"] = df.groupby(PLAYER_KEY)["fantasy_points"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    return df


def add_boom_bust(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean ``is_boom`` / ``is_bust`` columns using position-specific
    point thresholds from ``fantasyfb.config.BOOM_BUST_THRESHOLDS``."""
    df = df.copy()
    boom = df["position"].map(lambda p: BOOM_BUST_THRESHOLDS.get(p, {}).get("boom"))
    bust = df["position"].map(lambda p: BOOM_BUST_THRESHOLDS.get(p, {}).get("bust"))
    df["is_boom"] = df["fantasy_points"] >= boom
    df["is_bust"] = df["fantasy_points"] <= bust
    return df


def player_consistency(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Per player (optionally per season), summarize scoring consistency.

    Coefficient of variation (``cv`` = std / mean) is the headline
    consistency metric: lower means a steadier, more predictable scorer.
    Requires ``fantasy_points`` and, for boom/bust rates, ``is_boom`` /
    ``is_bust`` columns (see ``add_boom_bust``).
    """
    group_cols = group_cols or [PLAYER_KEY, "player_display_name", "position"]
    has_boom_bust = {"is_boom", "is_bust"}.issubset(df.columns)

    agg = {"fantasy_points": ["count", "mean", "std", "min", "max"]}
    grouped = df.groupby(group_cols).agg(agg)
    grouped.columns = ["games", "mean_points", "std_points", "min_points", "max_points"]
    grouped = grouped.reset_index()

    grouped["cv"] = grouped["std_points"] / grouped["mean_points"].replace(0, pd.NA)

    if has_boom_bust:
        rates = df.groupby(group_cols)[["is_boom", "is_bust"]].mean()
        rates.columns = ["boom_rate", "bust_rate"]
        grouped = grouped.merge(rates.reset_index(), on=group_cols)

    return grouped.sort_values("mean_points", ascending=False)


def points_allowed_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Fantasy points each defense has allowed to each position, per week.

    ``opponent_team`` in the weekly stats is the defense a given player's
    stat line was produced *against*, so summing a position's points by
    (season, week, opponent_team) gives that defense's points allowed.
    """
    return (
        df.groupby(GAME_KEYS + ["opponent_team", "position"])["fantasy_points"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team", "fantasy_points": "points_allowed"})
    )


def defense_matchup_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Season-long matchup difficulty: for each (season, position, team),
    average points allowed per week and its rank among all 32 teams that
    season (1 = allows the fewest = toughest matchup, 32 = easiest).
    """
    allowed = points_allowed_by_position(df)
    season_avg = (
        allowed.groupby(["season", "position", "team"])["points_allowed"]
        .mean()
        .reset_index()
    )
    season_avg["matchup_rank"] = season_avg.groupby(["season", "position"])[
        "points_allowed"
    ].rank(method="min", ascending=True)
    return season_avg.sort_values(["season", "position", "matchup_rank"])
