import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import plotly.express as px
import streamlit as st

from fantasyfb.analysis.scoring import compute_fantasy_points
from fantasyfb.analysis.trends import (
    add_boom_bust,
    add_position_finish_rank,
    add_rolling_average,
    defense_matchup_rank,
    player_consistency,
)
from fantasyfb.config import DEFAULT_SEASONS, FANTASY_POSITIONS
from fantasyfb.data.nflverse import load_weekly_stats

st.set_page_config(page_title="Player Trends", page_icon="📈", layout="wide")
st.title("📈 Player Stats & Trends")

with st.sidebar:
    seasons = st.multiselect("Seasons", options=[2022, 2023, 2024], default=DEFAULT_SEASONS)
    scoring = st.selectbox("Scoring", options=["ppr", "half_ppr", "standard"], index=0)
    position = st.selectbox("Position", options=["All"] + [p for p in FANTASY_POSITIONS if p != "DST"])
    rolling_window = st.slider("Rolling average window (games)", min_value=2, max_value=8, value=4)

if not seasons:
    st.warning("Select at least one season.")
    st.stop()


@st.cache_data(show_spinner="Loading weekly stats...")
def get_data(seasons, scoring, rolling_window):
    df = load_weekly_stats(seasons)
    df = df[df["season_type"] == "REG"]
    df["fantasy_points"] = compute_fantasy_points(df, scoring)
    df = add_position_finish_rank(df)
    df = add_boom_bust(df)
    df = add_rolling_average(df, window=rolling_window)
    return df


full_df = get_data(seasons, scoring, rolling_window)

df = full_df if position == "All" else full_df[full_df["position"] == position]

players = sorted(df["player_display_name"].dropna().unique())
default_player = "Patrick Mahomes" if "Patrick Mahomes" in players else (players[0] if players else None)

selected = st.selectbox("Player", options=players, index=players.index(default_player) if default_player else 0)

if not selected:
    st.stop()

player_df = df[df["player_display_name"] == selected].sort_values(["season", "week"])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{selected} — weekly fantasy points ({scoring})")
    plot_df = player_df.assign(game=lambda d: d["season"].astype(str) + " wk" + d["week"].astype(str))
    fig = px.bar(plot_df, x="game", y="fantasy_points", color="season", title=None)
    fig.add_scatter(
        x=plot_df["game"],
        y=plot_df["rolling_avg_points"],
        mode="lines",
        name=f"{rolling_window}-game rolling avg",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Consistency")
    consistency = player_consistency(player_df)
    if not consistency.empty:
        row = consistency.iloc[0]
        st.metric("Games played", int(row["games"]))
        st.metric("Mean points", f"{row['mean_points']:.1f}")
        st.metric("Coefficient of variation", f"{row['cv']:.2f}" if row["cv"] == row["cv"] else "n/a")
        st.metric("Boom rate", f"{row['boom_rate']:.0%}")
        st.metric("Bust rate", f"{row['bust_rate']:.0%}")

st.caption(
    "Coefficient of variation (std / mean) summarizes week-to-week volatility: "
    "lower means steadier scoring. Boom/bust rates use position-specific point "
    "thresholds (see fantasyfb.config.BOOM_BUST_THRESHOLDS)."
)

st.divider()
st.subheader("Weekly log")
st.dataframe(
    player_df[
        [
            "season",
            "week",
            "recent_team",
            "opponent_team",
            "fantasy_points",
            "position_rank",
            "rolling_avg_points",
            "is_boom",
            "is_bust",
        ]
    ].sort_values(["season", "week"], ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Matchup difficulty by defense")
st.caption(
    "Average fantasy points a defense has allowed per week to a position, "
    "and its rank across the league that season (1 = toughest matchup)."
)
matchup_position = st.selectbox(
    "Position for matchup difficulty",
    options=[p for p in FANTASY_POSITIONS if p != "DST"],
    key="matchup_position",
)
matchup_season = st.selectbox("Season", options=seasons, key="matchup_season")

matchup_df = defense_matchup_rank(full_df[full_df["season"] == matchup_season])
matchup_df = matchup_df[matchup_df["position"] == matchup_position]
st.dataframe(
    matchup_df[["team", "points_allowed", "matchup_rank"]].sort_values("matchup_rank"),
    use_container_width=True,
    hide_index=True,
)
