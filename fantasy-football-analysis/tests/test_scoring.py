import pandas as pd
import pytest

from fantasyfb.analysis.scoring import compute_fantasy_points, resolve_scoring


def _row(**overrides):
    base = {
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
    }
    base.update(overrides)
    return base


def test_resolve_scoring_preset():
    assert resolve_scoring("ppr")["rec"] == 1.0
    assert resolve_scoring("standard")["rec"] == 0.0


def test_resolve_scoring_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_scoring("nonsense")


def test_resolve_scoring_passthrough_dict():
    custom = {"rec": 0.5}
    assert resolve_scoring(custom) is custom


def test_qb_stat_line_ppr():
    # 300 pass yds, 3 pass TD, 1 INT, 20 rush yds, 1 fumble lost.
    df = pd.DataFrame(
        [_row(passing_yards=300, passing_tds=3, interceptions=1, rushing_yards=20, rushing_fumbles_lost=1)]
    )
    points = compute_fantasy_points(df, "ppr")
    # 300*0.04 + 3*4 + 1*-2 + 20*0.1 + 1*-2 = 12 + 12 - 2 + 2 - 2 = 22
    assert points.iloc[0] == pytest.approx(22.0)


def test_receptions_only_affect_ppr_not_standard():
    df = pd.DataFrame([_row(receptions=8, receiving_yards=80)])
    ppr_points = compute_fantasy_points(df, "ppr").iloc[0]
    standard_points = compute_fantasy_points(df, "standard").iloc[0]
    half_ppr_points = compute_fantasy_points(df, "half_ppr").iloc[0]

    assert standard_points == pytest.approx(8.0)  # 80 * 0.1
    assert half_ppr_points == pytest.approx(12.0)  # + 8 * 0.5
    assert ppr_points == pytest.approx(16.0)  # + 8 * 1.0


def test_missing_columns_default_to_zero():
    df = pd.DataFrame([{"passing_yards": 100}])
    points = compute_fantasy_points(df, "ppr")
    assert points.iloc[0] == pytest.approx(4.0)


@pytest.mark.network
def test_matches_nflverse_ppr_column_on_real_data():
    """Validation step for Phase 0: our from-scratch PPR calc should match
    nflverse's own precomputed fantasy_points_ppr column on real data."""
    nfl = pytest.importorskip("nfl_data_py")
    df = nfl.import_weekly_data([2023], downcast=True)
    df = df[df["season_type"] == "REG"]

    computed = compute_fantasy_points(df, "ppr")
    diff = (computed - df["fantasy_points_ppr"]).abs()

    # Allow tiny floating point / rounding slack; nflverse's own column is
    # computed the same way from the same raw stats.
    assert diff.max() < 0.5
    assert diff.mean() < 0.05
