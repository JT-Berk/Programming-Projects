"""Phase 2: value-based draft rankings, tiering, and backtest validation.

Scope note: nflverse's weekly stats contain zero K/DST rows (only incidental
FB/P rows from trick plays), so this module -- and the draft board it
produces -- is scoped to ``fantasyfb.config.RANKED_POSITIONS`` (QB/RB/WR/TE).

There is no real projections system yet (that's a later phase). The
"projection" here is a recency-weighted, shrinkage-adjusted blend of a
player's own scoring history -- see ``project_player_value`` for the exact
formula and ``README.md`` / the Phase 2 plan for the rationale.

Sleeper "buzz" (``label_sleeper_buzz``) is deliberately never called or
treated as ADP anywhere in this module: it is Sleeper's trending-*adds*
signal (recent add velocity across Sleeper leagues), not draft position, and
there is no free/reliable ADP source to compare against.
"""

from __future__ import annotations

import re

import pandas as pd
from scipy.stats import spearmanr

from fantasyfb.analysis.scoring import compute_fantasy_points
from fantasyfb.analysis.trends import PLAYER_KEY
from fantasyfb.config import (
    DEFAULT_LEAGUE_SIZE,
    DEFAULT_ROSTER_SLOTS,
    MIN_GAMES_CREDIBILITY,
    RANKED_POSITIONS,
    RECENCY_WEIGHTS,
    REPLACEMENT_RANK_OFFSET,
    TIER_GAP_STD_MULTIPLIER,
)


def seasonal_player_summary(weekly_df: pd.DataFrame, scoring: str | dict = "ppr") -> pd.DataFrame:
    """Collapse weekly stats to one row per player/season.

    Regular season only (postseason games are extra opportunity a full-season
    starter gets that a bench player doesn't, which would distort per-game
    averages) and scoped to ``RANKED_POSITIONS`` -- a player-season with no
    ranked-position rows at all (a pure K/DST/FB/P player) is dropped
    entirely, since this module doesn't rank those positions.
    """
    df = weekly_df[weekly_df["season_type"] == "REG"].copy()
    df = df[df["position"].isin(RANKED_POSITIONS)]
    df["fantasy_points"] = compute_fantasy_points(df, scoring)

    summary = (
        df.groupby([PLAYER_KEY, "season"])
        .agg(
            player_display_name=("player_display_name", "first"),
            games=("fantasy_points", "count"),
            total_points=("fantasy_points", "sum"),
        )
        .reset_index()
    )
    summary["per_game_points"] = summary["total_points"] / summary["games"]

    # Position-of-record: the mode of that player-season's weekly position
    # (already filtered to RANKED_POSITIONS above), in case a player has an
    # incidental alternate label on a handful of weeks.
    position_of_record = (
        df.groupby([PLAYER_KEY, "season"])["position"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    summary = summary.merge(position_of_record, on=[PLAYER_KEY, "season"])

    return summary.sort_values(["season", PLAYER_KEY]).reset_index(drop=True)


def _project_value_core(
    df: pd.DataFrame, weights: dict[int, float], min_games_credibility: float
) -> pd.DataFrame:
    """Shared blending logic for both ``project_player_value`` and the
    backtest's train-season projections.

    ``df`` must already be restricted to the seasons under consideration and
    carry a ``seasons_ago`` column (0 = most recent season in that set),
    filtered to ``seasons_ago`` values present in ``weights``.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                PLAYER_KEY,
                "player_display_name",
                "position",
                "projected_ppg",
                "projected_games",
                "projected_points",
            ]
        )

    df = df.copy()
    # Shrink each season's per-game points toward that season+position's mean
    # -- low-sample seasons (a handful of games) regress hard toward the
    # positional average; a full healthy season barely moves.
    positional_mean = df.groupby(["season", "position"])["per_game_points"].transform("mean")
    credibility = df["games"] / (df["games"] + min_games_credibility)
    df["shrunk_ppg"] = credibility * df["per_game_points"] + (1 - credibility) * positional_mean

    # Recency weights, renormalized per player over whichever seasons they
    # actually have -- a player missing an older season isn't zero-padded.
    df["raw_weight"] = df["seasons_ago"].map(weights)
    weight_sum = df.groupby(PLAYER_KEY)["raw_weight"].transform("sum")
    df["blend_weight"] = df["raw_weight"] / weight_sum

    df["_w_ppg"] = df["blend_weight"] * df["shrunk_ppg"]
    df["_w_games"] = df["blend_weight"] * df["games"]

    grouped = (
        df.groupby(PLAYER_KEY)
        .agg(
            player_display_name=("player_display_name", "first"),
            projected_ppg=("_w_ppg", "sum"),
            projected_games=("_w_games", "sum"),
        )
        .reset_index()
    )
    grouped["projected_games"] = grouped["projected_games"].clip(upper=17)
    grouped["projected_points"] = grouped["projected_ppg"] * grouped["projected_games"]

    # Position-of-record: the player's most recent qualifying season's
    # position -- never blended across a position change.
    most_recent = df.loc[
        df.groupby(PLAYER_KEY)["seasons_ago"].idxmin(), [PLAYER_KEY, "position"]
    ]
    grouped = grouped.merge(most_recent, on=PLAYER_KEY)

    return grouped.sort_values("projected_points", ascending=False).reset_index(drop=True)


def project_player_value(
    seasonal_df: pd.DataFrame,
    as_of_season: int,
    weights: dict[int, float] = RECENCY_WEIGHTS,
    min_games_credibility: float = MIN_GAMES_CREDIBILITY,
) -> pd.DataFrame:
    """Project each returning player's value for the season after
    ``as_of_season``, from seasons up to and including ``as_of_season``.

    Rookies (players with zero qualifying seasons at or before
    ``as_of_season``) are excluded outright, not zero-filled -- a
    pure-history model has nothing to project them from.
    """
    df = seasonal_df[seasonal_df["season"] <= as_of_season].copy()
    df["seasons_ago"] = as_of_season - df["season"]
    df = df[df["seasons_ago"].isin(weights)]

    result = _project_value_core(df, weights, min_games_credibility)
    result["season"] = as_of_season + 1
    return result


def _project_value_from_seasons(
    seasonal_df: pd.DataFrame,
    train_seasons: list[int],
    weights: dict[int, float] = RECENCY_WEIGHTS,
    min_games_credibility: float = MIN_GAMES_CREDIBILITY,
) -> pd.DataFrame:
    """Same blend as ``project_player_value``, but driven by an explicit list
    of training seasons rather than derived from a single ``as_of_season``
    (used by ``backtest_rank_correlation`` to build a train-only projection).
    """
    most_recent_train = max(train_seasons)
    df = seasonal_df[seasonal_df["season"].isin(train_seasons)].copy()
    df["seasons_ago"] = most_recent_train - df["season"]
    df = df[df["seasons_ago"].isin(weights)]

    return _project_value_core(df, weights, min_games_credibility)


def _value_at_rank(df_sorted_desc: pd.DataFrame, rank: int, value_col: str = "projected_points"):
    """The value at 1-indexed ``rank`` in a descending-sorted dataframe,
    clamped to the last row if the pool is smaller than ``rank`` (small
    leagues/pools shouldn't raise)."""
    idx = max(min(rank, len(df_sorted_desc)) - 1, 0)
    return df_sorted_desc.iloc[idx][value_col]


def compute_replacement_baselines(
    projected_df: pd.DataFrame,
    league_size: int = DEFAULT_LEAGUE_SIZE,
    roster_slots: dict[str, int] = DEFAULT_ROSTER_SLOTS,
    flex_positions: tuple[str, ...] = ("RB", "WR", "TE"),
) -> dict[str, float]:
    """Replacement-level ``projected_points`` per position.

    QB gets a dedicated baseline (the last startable QB across the league,
    plus one). RB/WR/TE share a single pooled flex baseline: players left
    over after each position's own dedicated-starter slots are pooled
    together and re-ranked, so real positional scarcity -- not an arbitrary
    RB/WR/TE split -- determines how much of the flex replacement level each
    position contributes.
    """
    baselines: dict[str, float] = {}

    qb = projected_df[projected_df["position"] == "QB"].sort_values(
        "projected_points", ascending=False
    )
    qb_rank = league_size * roster_slots["QB"] + REPLACEMENT_RANK_OFFSET
    baselines["QB"] = _value_at_rank(qb, qb_rank)

    leftovers = []
    for pos in flex_positions:
        pos_df = projected_df[projected_df["position"] == pos].sort_values(
            "projected_points", ascending=False
        )
        starter_cutoff = league_size * roster_slots[pos]
        leftovers.append(pos_df.iloc[starter_cutoff:])
    pooled = pd.concat(leftovers).sort_values("projected_points", ascending=False)

    flex_rank = league_size * roster_slots["FLEX"] + REPLACEMENT_RANK_OFFSET
    flex_baseline = _value_at_rank(pooled, flex_rank)
    for pos in flex_positions:
        baselines[pos] = flex_baseline

    return baselines


def add_vbd(projected_df: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    """Add ``vbd`` = projected_points minus that player's positional
    replacement baseline."""
    df = projected_df.copy()
    df["vbd"] = df["projected_points"] - df["position"].map(baselines)
    return df


def tier_by_gap(
    position_df: pd.DataFrame, std_multiplier: float = TIER_GAP_STD_MULTIPLIER
) -> pd.DataFrame:
    """Tier a single position's players by gaps in ``vbd``: a new tier starts
    wherever the drop to the next-ranked player exceeds
    ``mean(gaps) + std_multiplier * std(gaps)``. Deterministic, no
    hyperparameter to tune per position."""
    df = position_df.sort_values("vbd", ascending=False).reset_index(drop=True)
    if len(df) <= 1:
        df["tier"] = 1
        return df

    gaps = (-df["vbd"].diff()).iloc[1:]  # positive drop from each player to the next
    threshold = gaps.mean() + std_multiplier * gaps.std(ddof=0)

    tiers = [1]
    current_tier = 1
    for gap in gaps:
        if gap > threshold:
            current_tier += 1
        tiers.append(current_tier)
    df["tier"] = tiers
    return df


def tier_by_kmeans(position_df: pd.DataFrame, k_range: tuple[int, int] = (3, 6)) -> pd.DataFrame:
    """Tier a single position's players with 1-D KMeans on ``vbd``, picking
    k in ``k_range`` by silhouette score. Falls back to ``tier_by_gap`` when
    there are too few players to meaningfully cluster (fewer than
    ``2 * k_range[0]``)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = position_df.sort_values("vbd", ascending=False).reset_index(drop=True)
    k_min, k_max = k_range
    n = len(df)
    if n < 2 * k_min:
        return tier_by_gap(df)

    X = df[["vbd"]].to_numpy()
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return tier_by_gap(df)

    df["_cluster"] = best_labels
    # Relabel clusters 1..k in descending mean-vbd order so tier 1 is always
    # the best group, regardless of KMeans' arbitrary cluster ids.
    cluster_order = df.groupby("_cluster")["vbd"].mean().sort_values(ascending=False).index
    rank_map = {cluster: i + 1 for i, cluster in enumerate(cluster_order)}
    df["tier"] = df["_cluster"].map(rank_map)
    return df.drop(columns="_cluster")


def build_draft_board(
    weekly_df: pd.DataFrame,
    as_of_season: int,
    scoring: str | dict = "ppr",
    league_size: int = DEFAULT_LEAGUE_SIZE,
    roster_slots: dict[str, int] = DEFAULT_ROSTER_SLOTS,
    tier_method: str = "gap",
) -> pd.DataFrame:
    """Run the full pipeline -- seasonal summary, projection, replacement
    baselines, VBD, per-position tiering -- to a single ranked draft board
    with ``rank_overall`` (by vbd) and ``rank_position`` columns."""
    tier_fns = {"gap": tier_by_gap, "kmeans": tier_by_kmeans}
    if tier_method not in tier_fns:
        raise ValueError(f"Unknown tier_method '{tier_method}'; choose from {list(tier_fns)}.")
    tier_fn = tier_fns[tier_method]

    seasonal = seasonal_player_summary(weekly_df, scoring)
    projected = project_player_value(seasonal, as_of_season)
    baselines = compute_replacement_baselines(projected, league_size, roster_slots)
    board = add_vbd(projected, baselines)

    tiered_positions = []
    for pos in RANKED_POSITIONS:
        pos_df = board[board["position"] == pos]
        if pos_df.empty:
            continue
        tiered = tier_fn(pos_df)
        tiered["rank_position"] = (
            tiered["vbd"].rank(method="min", ascending=False).astype(int)
        )
        tiered_positions.append(tiered)

    board = pd.concat(tiered_positions, ignore_index=True)
    board["rank_overall"] = board["vbd"].rank(method="min", ascending=False).astype(int)
    return board.sort_values("rank_overall").reset_index(drop=True)


_NAME_SUFFIXES = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")
# Candidate name columns a caller's trending dataframe might carry. Sleeper's
# raw trending-adds endpoint returns ONLY its own player_id + add count, with
# no name field -- a caller wanting name matching must first merge in
# fantasyfb.data.sleeper.load_players() to attach one of these.
_TRENDING_NAME_COLUMNS = ["full_name", "player_name", "search_full_name", "player_display_name", "name"]


def _normalize_name(name) -> str:
    if pd.isna(name):
        return ""
    normalized = re.sub(r"[^\w\s]", "", str(name).lower().strip())
    normalized = _NAME_SUFFIXES.sub("", normalized)
    return normalized.strip()


def label_sleeper_buzz(
    board_df: pd.DataFrame, trending_df: pd.DataFrame, top_n: int = 24
) -> pd.DataFrame:
    """Cross-reference the draft board against Sleeper's trending-*adds*
    signal (recent add velocity across Sleeper leagues).

    This is explicitly NOT average draft position (ADP) -- there is no free,
    reliable ADP source, and "buzz" (who's being added right now) is a
    different signal than "where people draft someone." Never label this ADP.

    Known limitation: Sleeper's trending-adds endpoint
    (``fantasyfb.data.sleeper.load_trending_players``) returns only Sleeper's
    own ``player_id`` (a different ID space from nflverse's ``player_id``)
    plus an add count -- no player name. There is no reliable join key
    between the two ID spaces, so if ``trending_df`` doesn't already carry a
    name column (e.g. merged in from ``fantasyfb.data.sleeper.load_players``
    upstream), buzz can't be matched and every row is labeled accordingly
    (TODO: a maintained nflverse-id <-> sleeper-id crosswalk would remove
    this limitation).
    """
    df = board_df.copy()
    df["sleeper_buzz"] = False

    name_col = next((c for c in _TRENDING_NAME_COLUMNS if c in trending_df.columns), None)
    if name_col is None:
        df["buzz_note"] = (
            "Sleeper buzz unavailable: trending data has no name column to "
            "join against nflverse player_id (different ID spaces, no clean "
            "join key)."
        )
        return df

    trending_names = set(trending_df.head(top_n)[name_col].map(_normalize_name)) - {""}
    normalized_board_names = df["player_display_name"].map(_normalize_name)
    matched = normalized_board_names.isin(trending_names)

    df.loc[matched, "sleeper_buzz"] = True
    df["buzz_note"] = "No recent Sleeper trending-add signal."
    df.loc[matched, "buzz_note"] = "Trending add on Sleeper recently (current buzz, not ADP)."
    return df


def backtest_rank_correlation(
    seasonal_df: pd.DataFrame,
    train_seasons: list[int],
    holdout_season: int,
    min_games: int = MIN_GAMES_CREDIBILITY,
) -> pd.DataFrame:
    """Rolling-backtest validation: project value from ``train_seasons``
    only, then Spearman-correlate the projected rank against each position's
    *actual* per-game finish rank in ``holdout_season``.

    Reports, per position:
    - ``spearman_model`` / ``spearman_naive``: the recency+shrinkage model's
      rank correlation vs. a naive baseline (plain most-recent-train-season
      per-game rank), among only the holdout players who met ``min_games``
      in the holdout season ("dropped" variant -- players who missed the
      games floor, e.g. injuries/busts, are excluded).
    - ``spearman_model_floor_imputed`` / ``spearman_naive_floor_imputed``:
      the same correlations, but instead of dropping players who missed the
      games floor, they're assigned the worst actual rank at their position
      in the holdout season -- this keeps total misses (injuries, busts,
      players who didn't play at all) from inflating accuracy by hiding them.

    Caveat: rookies are structurally excluded from this backtest on both
    sides -- a pure-history model has no prior seasons to project a rookie
    from, so anyone debuting in ``holdout_season`` never appears in the
    train-season projection or the naive baseline to begin with. This
    backtest validates ranking of *returning* players only.
    """
    model_proj = _project_value_from_seasons(seasonal_df, train_seasons, RECENCY_WEIGHTS, min_games)

    most_recent_train = max(train_seasons)
    naive = seasonal_df[seasonal_df["season"] == most_recent_train][
        [PLAYER_KEY, "position", "per_game_points"]
    ].copy()

    holdout = seasonal_df[seasonal_df["season"] == holdout_season][
        [PLAYER_KEY, "position", "games", "per_game_points"]
    ].copy()
    holdout_qualified = holdout[holdout["games"] >= min_games].copy()

    rows = []
    for pos in RANKED_POSITIONS:
        model_pos = model_proj[model_proj["position"] == pos].copy()
        naive_pos = naive[naive["position"] == pos].copy()
        holdout_pos = holdout_qualified[holdout_qualified["position"] == pos].copy()

        if holdout_pos.empty or model_pos.empty:
            continue

        holdout_pos["actual_rank"] = holdout_pos["per_game_points"].rank(
            ascending=False, method="min"
        )
        worst_rank = holdout_pos["actual_rank"].max()

        model_pos["model_rank"] = model_pos["projected_ppg"].rank(ascending=False, method="min")
        naive_pos["naive_rank"] = naive_pos["per_game_points"].rank(ascending=False, method="min")

        actual_lookup = holdout_pos[[PLAYER_KEY, "actual_rank"]]

        model_dropped = model_pos.merge(actual_lookup, on=PLAYER_KEY, how="inner")
        naive_dropped = naive_pos.merge(actual_lookup, on=PLAYER_KEY, how="inner")

        model_imputed = model_pos.merge(actual_lookup, on=PLAYER_KEY, how="left")
        model_imputed["actual_rank"] = model_imputed["actual_rank"].fillna(worst_rank)
        naive_imputed = naive_pos.merge(actual_lookup, on=PLAYER_KEY, how="left")
        naive_imputed["actual_rank"] = naive_imputed["actual_rank"].fillna(worst_rank)

        def _corr(frame, rank_col):
            if len(frame) < 2:
                return float("nan")
            return spearmanr(frame[rank_col], frame["actual_rank"]).correlation

        rows.append(
            dict(
                position=pos,
                n_players=len(model_dropped),
                spearman_model=_corr(model_dropped, "model_rank"),
                spearman_naive=_corr(naive_dropped, "naive_rank"),
                spearman_model_floor_imputed=_corr(model_imputed, "model_rank"),
                spearman_naive_floor_imputed=_corr(naive_imputed, "naive_rank"),
            )
        )

    return pd.DataFrame(rows)
