import pandas as pd
import pytest

from fantasyfb.analysis.trends import (
    add_boom_bust,
    add_position_finish_rank,
    add_rolling_average,
    defense_matchup_rank,
    player_consistency,
    points_allowed_by_position,
)


def make_weekly_df():
    # Two RBs across 3 weeks of one season, facing two different defenses.
    rows = [
        # player A: steady ~15 pts
        dict(season=2023, week=1, player_id="A", player_display_name="Player A", position="RB", opponent_team="NYJ", fantasy_points=15),
        dict(season=2023, week=2, player_id="A", player_display_name="Player A", position="RB", opponent_team="BUF", fantasy_points=14),
        dict(season=2023, week=3, player_id="A", player_display_name="Player A", position="RB", opponent_team="NYJ", fantasy_points=16),
        # player B: boom/bust ~ swings between 2 and 28
        dict(season=2023, week=1, player_id="B", player_display_name="Player B", position="RB", opponent_team="BUF", fantasy_points=28),
        dict(season=2023, week=2, player_id="B", player_display_name="Player B", position="RB", opponent_team="NYJ", fantasy_points=2),
        dict(season=2023, week=3, player_id="B", player_display_name="Player B", position="RB", opponent_team="BUF", fantasy_points=27),
    ]
    return pd.DataFrame(rows)


def test_add_position_finish_rank():
    df = add_position_finish_rank(make_weekly_df())
    week1 = df[df["week"] == 1].set_index("player_id")
    # Player B (28 pts) should outrank Player A (15 pts) in week 1.
    assert week1.loc["B", "position_rank"] == 1
    assert week1.loc["A", "position_rank"] == 2


def test_add_rolling_average_uses_prior_games_only_current_player():
    df = add_rolling_average(make_weekly_df(), window=2)
    a = df[df["player_id"] == "A"].sort_values("week")
    # window=2: week1 -> just itself (15), week2 -> mean(15,14)=14.5, week3 -> mean(14,16)=15
    assert a["rolling_avg_points"].tolist() == pytest.approx([15.0, 14.5, 15.0])


def test_add_boom_bust_thresholds():
    df = add_boom_bust(make_weekly_df())
    b = df[df["player_id"] == "B"].sort_values("week").reset_index(drop=True)
    # RB boom threshold=20, bust threshold=5 (see config.BOOM_BUST_THRESHOLDS)
    assert b.loc[0, "is_boom"] and not b.loc[0, "is_bust"]  # 28 pts
    assert b.loc[1, "is_bust"] and not b.loc[1, "is_boom"]  # 2 pts
    assert b.loc[2, "is_boom"]  # 27 pts


def test_player_consistency_flags_volatile_player():
    df = add_boom_bust(make_weekly_df())
    summary = player_consistency(df).set_index("player_display_name")
    # Player B should have far higher coefficient of variation (volatility)
    # than the steady Player A.
    assert summary.loc["Player B", "cv"] > summary.loc["Player A", "cv"]
    assert summary.loc["Player A", "games"] == 3


def test_points_allowed_by_position_sums_correctly():
    allowed = points_allowed_by_position(make_weekly_df())
    # Week 1: NYJ faced Player A (15 pts), BUF faced Player B (28 pts).
    week1 = allowed[allowed["week"] == 1].set_index("team")
    assert week1.loc["NYJ", "points_allowed"] == 15
    assert week1.loc["BUF", "points_allowed"] == 28


def test_defense_matchup_rank_toughest_is_rank_one():
    ranks = defense_matchup_rank(make_weekly_df())
    ranks = ranks.set_index("team")
    # NYJ allowed (15+2)/2=8.5 avg vs BUF's (14+28)/2=21 avg -> NYJ is tougher.
    assert ranks.loc["NYJ", "matchup_rank"] < ranks.loc["BUF", "matchup_rank"]
