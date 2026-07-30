import pandas as pd
import pytest

from fantasyfb.analysis.rankings import (
    add_vbd,
    backtest_rank_correlation,
    build_draft_board,
    compute_replacement_baselines,
    project_player_value,
    tier_by_gap,
    tier_by_kmeans,
)


def _seasonal_row(player_id, season, position, games, per_game_points, name=None):
    return dict(
        player_id=player_id,
        season=season,
        player_display_name=name or player_id,
        games=games,
        total_points=games * per_game_points,
        per_game_points=per_game_points,
        position=position,
    )


def test_project_player_value_recency_weighting_matches_hand_computed_average():
    # A single QB, alone at the position each season, so the positional mean
    # equals the player's own value and shrinkage becomes a no-op -- this
    # isolates the recency-weighting math.
    seasonal_df = pd.DataFrame(
        [
            _seasonal_row("QB1", 2021, "QB", games=10, per_game_points=6),
            _seasonal_row("QB1", 2022, "QB", games=10, per_game_points=8),
            _seasonal_row("QB1", 2023, "QB", games=10, per_game_points=10),
        ]
    )
    projected = project_player_value(seasonal_df, as_of_season=2023)
    row = projected.set_index("player_id").loc["QB1"]

    # RECENCY_WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.2}; weights already sum to 1.
    expected_ppg = 0.5 * 10 + 0.3 * 8 + 0.2 * 6
    assert row["projected_ppg"] == pytest.approx(expected_ppg)
    assert row["projected_games"] == pytest.approx(10)
    assert row["position"] == "QB"


def test_project_player_value_shrinks_low_sample_outlier_toward_positional_mean():
    # Three WRs in the same season: two normal-volume 10 ppg scorers, and one
    # extreme outlier who only played 1 game. The outlier's shrunk value
    # should land much closer to the positional mean than to its raw value.
    seasonal_df = pd.DataFrame(
        [
            _seasonal_row("WR_LOW", 2023, "WR", games=1, per_game_points=40),
            _seasonal_row("WR_A", 2023, "WR", games=17, per_game_points=10),
            _seasonal_row("WR_B", 2023, "WR", games=17, per_game_points=10),
        ]
    )
    projected = project_player_value(seasonal_df, as_of_season=2023)
    low = projected.set_index("player_id").loc["WR_LOW"]

    positional_mean = (40 + 10 + 10) / 3  # 20.0
    credibility = 1 / (1 + 6)  # MIN_GAMES_CREDIBILITY default = 6
    expected = credibility * 40 + (1 - credibility) * positional_mean

    assert low["projected_ppg"] == pytest.approx(expected)
    assert low["projected_ppg"] < 40  # pulled well below its raw average...
    assert abs(low["projected_ppg"] - positional_mean) < abs(low["projected_ppg"] - 40)  # ...and closer to the mean than to itself


def test_compute_replacement_baselines_pools_rb_wr_te_to_same_value():
    league_size = 2
    roster_slots = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1}

    def _pos_rows(position, points):
        return [
            dict(player_id=f"{position}{i}", position=position, projected_points=p)
            for i, p in enumerate(points)
        ]

    projected = pd.DataFrame(
        _pos_rows("QB", [100, 90, 80, 70])
        + _pos_rows("RB", [50, 45, 10, 5])
        + _pos_rows("WR", [48, 40, 9, 4])
        + _pos_rows("TE", [30, 20, 3, 1])
    )

    baselines = compute_replacement_baselines(
        projected, league_size=league_size, roster_slots=roster_slots
    )

    # QB: dedicated baseline = rank (2*1 + 1) = 3rd QB by projected_points -> 80.
    assert baselines["QB"] == pytest.approx(80)

    # RB/WR/TE: leftovers below each position's own 2*1=2 starter cutoff are
    # {10,5}, {9,4}, {3,1} -> pooled desc = [10,9,5,4,3,1]; FLEX rank
    # (2*1+1=3) in that pool -> 5. Same value applies to all three positions.
    assert baselines["RB"] == pytest.approx(5)
    assert baselines["WR"] == pytest.approx(5)
    assert baselines["TE"] == pytest.approx(5)


def _vbd_df(vbd_values, position="RB"):
    return pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(len(vbd_values))],
            "position": position,
            "vbd": vbd_values,
        }
    )


def test_tier_by_gap_breaks_at_engineered_large_gap():
    # Sorted desc: 50, 48, 45 | 20, 18, 15, 5, 3 -- one clear outsized gap
    # (45 -> 20) should force a new tier there; the smaller gap (15 -> 5)
    # should not.
    df = _vbd_df([50, 48, 45, 20, 18, 15, 5, 3])
    tiered = tier_by_gap(df)

    tiers = tiered.set_index("player_id")["tier"]
    assert tiers.loc[["P0", "P1", "P2"]].tolist() == [1, 1, 1]
    assert tiers.loc[["P3", "P4", "P5", "P6", "P7"]].nunique() == 1
    assert tiers.loc["P3"] > tiers.loc["P2"]


def test_tier_by_kmeans_matches_tier_by_gap_on_clearly_clustered_data():
    # Three tight, widely separated clusters of 3 players each.
    values = [90, 88, 85, 50, 48, 45, 10, 8, 5]
    df = _vbd_df(values)

    gap_tiers = tier_by_gap(df).set_index("player_id")["tier"]
    kmeans_tiers = tier_by_kmeans(df, k_range=(3, 6)).set_index("player_id")["tier"]

    # Same grouping of players into tiers (tier *labels* both run 1=best, so
    # they should match directly on this clean fixture).
    assert (gap_tiers.sort_index() == kmeans_tiers.sort_index()).all()


def test_tier_by_kmeans_falls_back_to_gap_on_small_pool():
    df = _vbd_df([10, 5, 1])  # fewer than 2 * k_range[0] players
    kmeans_tiers = tier_by_kmeans(df, k_range=(3, 6))
    gap_tiers = tier_by_gap(df)
    assert kmeans_tiers["tier"].tolist() == gap_tiers["tier"].tolist()


def test_backtest_rank_correlation_matches_hand_computed_spearman():
    # Train season 2022: WR1/WR2 are normal-volume, WR3 is a 1-game outlier
    # whose raw ppg (22) briefly out-scores WR1 (20), but shrinkage (K=6)
    # pulls WR3's blended value back below WR1 -- so the model's projected
    # order (WR1, WR3, WR2) differs from the naive raw-ppg order (WR3, WR1, WR2).
    seasonal_df = pd.DataFrame(
        [
            _seasonal_row("WR1", 2022, "WR", games=17, per_game_points=20),
            _seasonal_row("WR2", 2022, "WR", games=17, per_game_points=15),
            _seasonal_row("WR3", 2022, "WR", games=1, per_game_points=22),
            # Holdout season 2023 actual finish exactly matches the model's
            # predicted order (WR1 > WR3 > WR2), not the naive order.
            _seasonal_row("WR1", 2023, "WR", games=10, per_game_points=25),
            _seasonal_row("WR3", 2023, "WR", games=10, per_game_points=10),
            _seasonal_row("WR2", 2023, "WR", games=10, per_game_points=5),
        ]
    )

    result = backtest_rank_correlation(
        seasonal_df, train_seasons=[2022], holdout_season=2023, min_games=6
    )
    row = result.set_index("position").loc["WR"]

    # Model ranks: WR1 (~19.74) > WR3 (~19.43) > WR2 (~16.04); actual holdout
    # ranks: WR1 > WR3 > WR2 -- perfect agreement, Spearman rho = 1.0.
    assert row["n_players"] == 3
    assert row["spearman_model"] == pytest.approx(1.0)

    # Naive (raw last-season ppg) ranks: WR3 (22) > WR1 (20) > WR2 (15) --
    # WR1/WR3 swapped vs. actual -> rho = 1 - 6*sum(d^2)/(n^3-n) = 1 - 12/24 = 0.5.
    assert row["spearman_naive"] == pytest.approx(0.5)

    # All holdout players meet the games floor, so the floor-imputed variant
    # is identical to the dropped variant here.
    assert row["spearman_model_floor_imputed"] == pytest.approx(1.0)
    assert row["spearman_naive_floor_imputed"] == pytest.approx(0.5)


def test_backtest_rank_correlation_floor_imputes_missing_holdout_players():
    # WR4 posts a great train season but is injured/absent in the holdout
    # season (falls below min_games there). The floor-imputed variant should
    # still count WR4 (assigned the worst holdout rank), while the dropped
    # variant should simply exclude them.
    seasonal_df = pd.DataFrame(
        [
            _seasonal_row("WR1", 2022, "WR", games=17, per_game_points=20),
            _seasonal_row("WR2", 2022, "WR", games=17, per_game_points=15),
            _seasonal_row("WR4", 2022, "WR", games=17, per_game_points=18),
            _seasonal_row("WR1", 2023, "WR", games=10, per_game_points=20),
            _seasonal_row("WR2", 2023, "WR", games=10, per_game_points=10),
            _seasonal_row("WR4", 2023, "WR", games=2, per_game_points=5),  # below min_games
        ]
    )

    result = backtest_rank_correlation(
        seasonal_df, train_seasons=[2022], holdout_season=2023, min_games=6
    )
    row = result.set_index("position").loc["WR"]

    assert row["n_players"] == 2  # WR4 dropped from the "dropped" variant
    assert row["spearman_model"] == pytest.approx(1.0)  # WR1 > WR2 both sides
    # Floor-imputed variant includes all 3 model-projected players.
    assert not pd.isna(row["spearman_model_floor_imputed"])


@pytest.mark.network
def test_build_draft_board_end_to_end_on_real_data():
    """Validation step for Phase 2: run the full pipeline on real cached
    nflverse data and check internal consistency (no K/DST rows, and
    tier/vbd/rank_position ordering agree within each position)."""
    pytest.importorskip("nfl_data_py")
    from fantasyfb.data.nflverse import load_weekly_stats

    weekly = load_weekly_stats([2022, 2023, 2024])
    board = build_draft_board(weekly, as_of_season=2023)

    assert not board.empty
    assert not set(board["position"]) & {"K", "DST"}

    for position, group in board.groupby("position"):
        by_tier_then_vbd = group.sort_values(["tier", "vbd"], ascending=[True, False])
        assert by_tier_then_vbd["rank_position"].is_monotonic_increasing
