import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

st.set_page_config(page_title="Fantasy Football Analysis", page_icon="🏈", layout="wide")

st.title("🏈 Fantasy Football Analysis")
st.write(
    "A data-driven fantasy football toolkit built on real historical NFL "
    "data from nflverse and current player metadata from Sleeper."
)

st.subheader("Roadmap")
st.markdown(
    """
| Phase | Feature | Status |
|---|---|---|
| 0 | Data layer + app scaffold | ✅ Done |
| 1 | Player stats & trends analysis | ✅ Done |
| 2 | Draft rankings & tools | 🔜 Planned |
| 3 | Weekly start/sit & lineup optimizer | 🔜 Planned |
| 4 | Trade analyzer | 🔜 Planned |
    """
)

st.info("Use the sidebar to open a feature page.")

st.subheader("Data sources")
st.markdown(
    """
- **nflverse** (`nfl_data_py`): historical weekly/seasonal player stats and
  schedules. Cached locally under `data/cache/` after the first fetch.
- **Sleeper API**: current player metadata (team, position, injury status).
  No API key required for either source.
    """
)
