"""Shared constants and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# nflverse's weekly player-stats release lags the live season considerably,
# so "current" here means the most recent season with complete published
# data, not the calendar year.
CURRENT_SEASON = 2024
DEFAULT_SEASONS = [2022, 2023, 2024]

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

# Standard 12-team league roster: how many starters per position, used by
# rankings (replacement level) and the lineup optimizer.
DEFAULT_LEAGUE_SIZE = 12
DEFAULT_ROSTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,  # RB/WR/TE
    "K": 1,
    "DST": 1,
}

# Named scoring presets. Values are points per unit unless noted.
# Weekly fantasy-point thresholds used to flag "boom" (great) and "bust"
# (poor) weeks per position. These are rough, commonly-used PPR cutoffs
# roughly aligned with weekly QB1/RB1/WR1/TE1-level and replacement-level
# scoring; they're a heuristic, not a statistically derived boundary.
BOOM_BUST_THRESHOLDS = {
    "QB": {"boom": 25, "bust": 10},
    "RB": {"boom": 20, "bust": 5},
    "WR": {"boom": 20, "bust": 5},
    "TE": {"boom": 15, "bust": 3},
    "K": {"boom": 12, "bust": 4},
    "DST": {"boom": 12, "bust": 2},
}

SCORING_PRESETS = {
    "standard": {
        "pass_yd": 0.04,
        "pass_td": 4,
        "pass_int": -2,
        "rush_yd": 0.1,
        "rush_td": 6,
        "return_td": 6,
        "rec": 0.0,
        "rec_yd": 0.1,
        "rec_td": 6,
        "fumble_lost": -2,
        "two_pt": 2,
    },
    "half_ppr": {
        "pass_yd": 0.04,
        "pass_td": 4,
        "pass_int": -2,
        "rush_yd": 0.1,
        "rush_td": 6,
        "return_td": 6,
        "rec": 0.5,
        "rec_yd": 0.1,
        "rec_td": 6,
        "fumble_lost": -2,
        "two_pt": 2,
    },
    "ppr": {
        "pass_yd": 0.04,
        "pass_td": 4,
        "pass_int": -2,
        "rush_yd": 0.1,
        "rush_td": 6,
        "return_td": 6,
        "rec": 1.0,
        "rec_yd": 0.1,
        "rec_td": 6,
        "fumble_lost": -2,
        "two_pt": 2,
    },
}
