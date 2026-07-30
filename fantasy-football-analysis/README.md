# Fantasy Football Analysis

A data-driven fantasy football toolkit: player trends, draft rankings, a
weekly start/sit optimizer, and a trade analyzer, all built on real historical
NFL data.

## Stack

- **Data**: [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (nflverse)
  for historical weekly/seasonal stats and schedules; the
  [Sleeper API](https://docs.sleeper.com/) for current player/roster metadata.
- **Analysis**: pandas / numpy, scikit-learn (clustering for draft tiers,
  regression for projections).
- **App**: Streamlit, multipage.

Data is pulled live (and cached locally under `data/cache/` as parquet/JSON,
gitignored) — no API keys required for either source.

## Project layout

```
fantasy-football-analysis/
├── app/
│   ├── Home.py                 # Streamlit entry point / overview
│   └── pages/                  # one page per feature area
├── src/fantasyfb/
│   ├── config.py                # scoring settings, constants
│   ├── data/
│   │   ├── nflverse.py          # weekly/seasonal stats + schedules, cached
│   │   └── sleeper.py           # player metadata / league client
│   └── analysis/
│       ├── scoring.py            # fantasy point calculation
│       ├── trends.py             # Phase 1: consistency & trend metrics
│       ├── rankings.py           # Phase 2: value-based draft rankings/tiers
│       ├── projections.py        # Phase 3: weekly projections
│       └── trade.py              # Phase 4: trade value/fairness
├── tests/
└── data/cache/                   # gitignored local cache
```

## Roadmap

Each phase ships a working, testable increment before the next one starts.

### Phase 0 — Foundation (data layer + app scaffold)
- `nflverse.py`: load weekly/seasonal player stats and schedules, cached
  locally so repeated runs don't re-download.
- `sleeper.py`: pull current player metadata (names, teams, positions,
  injury status) from Sleeper's public API.
- `scoring.py`: configurable scoring settings (PPR / half-PPR / standard)
  applied consistently across every downstream feature.
- Streamlit app shell (`Home.py`) with navigation to each feature page.
- **Validation**: spot-check computed fantasy points against known season
  leaders (e.g. 2023 PPR leaders) before building anything on top.

### Phase 1 — Player stats & trends analysis
- Rolling scoring averages, week-over-week volatility (consistency score),
  boom/bust rate, and positional benchmarking (e.g. weekly finish rank).
- Matchup difficulty using opponent defense's fantasy points allowed by
  position.
- Streamlit page: pick a player, see scoring trend chart + consistency
  metrics + upcoming matchup context.
- **Validation**: cross-check a handful of well-known player trends against
  public record (e.g. a player's known bust/boom weeks).

### Phase 2 — Draft rankings & tools
- Scoped to QB/RB/WR/TE only: nflverse's weekly stats contain zero K/DST
  rows, so there's no real data to rank those positions against.
- "Projected points" is a from-scratch, recency-weighted blend of each
  player's own scoring history across seasons, with shrinkage toward the
  positional mean for low-sample seasons — an informed historical baseline,
  not a real projections model (that's a future phase).
- Value-based drafting (VBD): replacement-level baselines per position,
  with RB/WR/TE sharing one pooled flex baseline (leftover players below
  each position's own starter cutoff are pooled and re-ranked together)
  instead of an arbitrary flex-share split.
- Tiering via position-level gaps in VBD (default) or KMeans clustering
  (toggle-able alternative).
- **No real ADP data**: there's no free, reliable fantasy ADP API (nfl_data_py's
  draft data is the NFL entry draft, not fantasy ADP), and scraping a
  third-party ADP site is fragile/ToS-risky. Instead, a Sleeper "trending
  adds" cross-reference is shown, explicitly labeled as a different, weaker
  signal than ADP.
- **Validation**: a rolling backtest — project from prior seasons only,
  Spearman-correlate against each holdout season's actual finish, per
  position — reported against a naive last-season-rank baseline (so the
  model has to prove it adds signal) with both a games-floor-dropped and a
  floor-imputed variant (the latter keeps injuries/busts who miss the
  holdout games floor from being silently excluded and inflating accuracy).
  Result so far: the model beats the naive baseline on the dropped variant,
  but **not** on the honest floor-imputed variant — its multi-season blend
  doesn't discount older data enough to catch players already declining,
  while a plain last-season rank is more current and reacts to that decline
  first. A real caveat, not swept under the rug — see the Draft Rankings
  page's validation expander for the live numbers.

### Phase 3 — Weekly start/sit & lineup optimizer
- Weekly projection model (rolling performance + matchup adjustment).
- Lineup optimizer: given a roster and league scoring/roster settings, pick
  the highest-expected-value valid lineup.
- **Validation**: backtest projections against actual outcomes for prior
  weeks/seasons (mean absolute error), reported before the optimizer ships.

### Phase 4 — Trade analyzer
- Trade value calculator built on Phase 2/3 outputs (rest-of-season value,
  not just next week).
- Multi-player trade support with a fairness/lopsidedness score.
- **Validation**: sanity-check against a few real, well-known historical
  trades where the "winner" is not controversial.

## Setup

```bash
cd fantasy-football-analysis
pip install -r requirements.txt
streamlit run app/Home.py
```

Run tests with:

```bash
pytest tests/
```
