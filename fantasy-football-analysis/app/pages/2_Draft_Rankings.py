import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import streamlit as st

from fantasyfb.analysis.rankings import (
    backtest_rank_correlation,
    build_draft_board,
    label_sleeper_buzz,
    seasonal_player_summary,
)
from fantasyfb.config import (
    DEFAULT_LEAGUE_SIZE,
    DEFAULT_SEASONS,
    MIN_GAMES_CREDIBILITY,
    RANKED_POSITIONS,
)
from fantasyfb.data.nflverse import load_weekly_stats
from fantasyfb.data.sleeper import load_players, load_trending_players

st.set_page_config(page_title="Draft Rankings", page_icon="🏆", layout="wide")
st.title("🏆 Draft Rankings")

# Wider, fixed season range used for the rolling backtest below -- deliberately
# independent of whatever seasons the sidebar picks for the board itself, since
# a meaningful backtest needs more history than a single 3-season board.
BACKTEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
ROLLING_HOLDOUTS = [
    ([2019, 2020, 2021], 2022),
    ([2019, 2020, 2021, 2022], 2023),
    ([2019, 2020, 2021, 2022, 2023], 2024),
]

st.warning(
    "**Read this before drafting from this board.**\n\n"
    "- **Positions:** scoped to **QB / RB / WR / TE only**. nflverse's weekly "
    "stats contain zero K or DST rows (only incidental FB/P rows from trick "
    "plays), so there's no real data to rank kickers or defenses against here.\n"
    "- **\"Projected points\" is NOT a real projections model.** There is no "
    "projections system in this app yet (that's a future phase). What you see "
    "is a from-scratch, recency-weighted blend of each player's own scoring "
    "history across seasons, with shrinkage toward the positional mean for "
    "low-sample seasons. Treat it as an informed historical baseline, not a "
    "forecast.\n"
    "- **No real ADP (average draft position) data is shown**, because there "
    "is no free, reliable ADP API: nfl_data_py's draft data is the NFL entry "
    "draft, not fantasy ADP, and scraping a third-party ADP site is fragile "
    "and a ToS risk. Instead, a **Sleeper 'trending adds' cross-reference** "
    "is shown below -- explicitly a different and weaker signal than ADP "
    "(recent add velocity, not where people actually draft someone)."
)

with st.sidebar:
    seasons = st.multiselect(
        "Seasons to project from",
        options=list(range(2019, 2025)),
        default=DEFAULT_SEASONS,
        help="The board projects the season AFTER the most recent season selected here.",
    )
    scoring = st.selectbox("Scoring", options=["ppr", "half_ppr", "standard"], index=0)
    league_size = st.number_input(
        "League size", min_value=4, max_value=20, value=DEFAULT_LEAGUE_SIZE, step=1
    )
    tier_method = st.radio("Tier method", options=["gap", "kmeans"], index=0, horizontal=True)
    position_filter = st.selectbox("Position filter", options=["All"] + RANKED_POSITIONS)

if not seasons:
    st.warning("Select at least one season.")
    st.stop()

as_of_season = max(seasons)


@st.cache_data(show_spinner="Building draft board...")
def get_draft_board(seasons, scoring, league_size, tier_method, as_of_season):
    weekly = load_weekly_stats(seasons)
    return build_draft_board(
        weekly,
        as_of_season=as_of_season,
        scoring=scoring,
        league_size=league_size,
        tier_method=tier_method,
    )


@st.cache_data(show_spinner="Running rolling backtest across 2019-2024...")
def get_backtest_results(scoring, min_games):
    weekly = load_weekly_stats(BACKTEST_SEASONS)
    seasonal = seasonal_player_summary(weekly, scoring)
    frames = []
    for train_seasons, holdout_season in ROLLING_HOLDOUTS:
        result = backtest_rank_correlation(
            seasonal, train_seasons=train_seasons, holdout_season=holdout_season, min_games=min_games
        )
        result.insert(0, "holdout_season", holdout_season)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


board = get_draft_board(seasons, scoring, league_size, tier_method, as_of_season)

st.caption(
    f"Board built from seasons {sorted(seasons)} (scoring: {scoring}, league size: "
    f"{league_size}, tier method: {tier_method}) -- projecting for the "
    f"{as_of_season + 1} season."
)

filtered_board = board if position_filter == "All" else board[board["position"] == position_filter]

st.subheader("Draft board")

_TIER_COLORS = [
    "rgba(31, 119, 180, 0.16)",
    "rgba(44, 160, 44, 0.16)",
    "rgba(255, 127, 14, 0.16)",
    "rgba(148, 103, 189, 0.16)",
    "rgba(214, 39, 40, 0.16)",
    "rgba(23, 190, 207, 0.16)",
    "rgba(188, 189, 34, 0.16)",
]


def _style_by_tier(row):
    color = _TIER_COLORS[(int(row["tier"]) - 1) % len(_TIER_COLORS)]
    return [f"background-color: {color}"] * len(row)


display_cols = [
    "rank_overall",
    "tier",
    "position",
    "player_display_name",
    "projected_points",
    "vbd",
]
display_df = (
    filtered_board[display_cols]
    .rename(columns={"player_display_name": "player"})
    .sort_values("rank_overall")
    .reset_index(drop=True)
)

styled = display_df.style.apply(_style_by_tier, axis=1).format(
    {"projected_points": "{:.1f}", "vbd": "{:.1f}"}
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=560)
st.caption(
    "Rows are shaded by tier (a new tier starts wherever the drop-off in VBD "
    "to the next player is unusually large for that position). Click column "
    "headers to sort."
)

st.divider()

with st.expander("💬 Sleeper Buzz -- NOT the same as ADP"):
    st.caption(
        "Sleeper's public API has no draft-position data, only 'trending adds' "
        "(recent add velocity across Sleeper leagues). That's a real-time "
        "popularity signal, not average draft position -- treat it as a "
        "supplementary nudge, never as a substitute for ADP."
    )
    try:
        trending = load_trending_players()
        players_meta = load_players()[["player_id", "full_name"]]
        trending = trending.merge(players_meta, on="player_id", how="left")
        buzz_board = label_sleeper_buzz(filtered_board, trending, top_n=24)
        buzzing = buzz_board[buzz_board["sleeper_buzz"]].sort_values("rank_overall")

        if buzzing.empty:
            st.write("No player currently on this board matches Sleeper's trending-adds list.")
        else:
            st.dataframe(
                buzzing[
                    ["rank_overall", "tier", "position", "player_display_name", "buzz_note"]
                ].rename(columns={"player_display_name": "player"}),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.error(
            f"Could not fetch live Sleeper trending data right now ({exc}). "
            "Buzz cross-reference is unavailable this session -- the draft "
            "board above is unaffected."
        )

st.divider()

with st.expander("📊 Model validation (rolling backtest)"):
    st.caption(
        "For each holdout season below, the model is projected from ONLY the "
        "seasons strictly before it (a real rolling backtest, not the sidebar's "
        "board seasons), then Spearman-correlated against each position's "
        "actual per-game finish that season -- compared against a naive "
        "baseline (plain prior-season per-game-points rank). 'Floor-imputed' "
        "variants assign players who missed the games floor in the holdout "
        "season (injuries, busts) the worst rank at their position, instead of "
        "dropping them -- the honest way to look at this, since dropping them "
        "hides total misses."
    )
    try:
        backtest_df = get_backtest_results(scoring, MIN_GAMES_CREDIBILITY)
        st.dataframe(
            backtest_df.style.format(
                {
                    "spearman_model": "{:.2f}",
                    "spearman_naive": "{:.2f}",
                    "spearman_model_floor_imputed": "{:.2f}",
                    "spearman_naive_floor_imputed": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        avg_model_dropped = backtest_df["spearman_model"].mean()
        avg_naive_dropped = backtest_df["spearman_naive"].mean()
        avg_model_imputed = backtest_df["spearman_model_floor_imputed"].mean()
        avg_naive_imputed = backtest_df["spearman_naive_floor_imputed"].mean()

        beats_dropped = avg_model_dropped > avg_naive_dropped
        beats_imputed = avg_model_imputed > avg_naive_imputed

        st.markdown(
            f"**Averaged across all {len(ROLLING_HOLDOUTS)} holdout seasons and all positions:** "
            f"model = {avg_model_dropped:.3f} vs. naive = {avg_naive_dropped:.3f} "
            f"({'model beats' if beats_dropped else 'model does NOT beat'} the naive baseline, "
            "games-floor-dropped variant); "
            f"model = {avg_model_imputed:.3f} vs. naive = {avg_naive_imputed:.3f} "
            f"({'model beats' if beats_imputed else 'model does NOT beat'} the naive baseline, "
            "floor-imputed variant)."
        )
        st.caption(
            "Caveat: rookies are structurally excluded from this backtest on "
            "both sides -- a pure-history model has zero prior seasons to "
            "project a first-year player from, so anyone debuting in a holdout "
            "season never appears in either the model's projection or the "
            "naive baseline. This validates ranking of returning players only."
        )
    except Exception as exc:
        st.error(f"Could not run the backtest right now ({exc}).")
