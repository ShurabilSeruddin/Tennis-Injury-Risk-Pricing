"""
Interactive premium estimator.

Run with:  streamlit run app/streamlit_app.py

Requires src/run_pipeline.py to have been run first, so that the fitted model
and pricing parameters exist in outputs/models/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import pricing as P
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

st.set_page_config(page_title="Tennis Injury Premium Estimator", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(C.MODELS / "logit_pipeline.joblib")
    params = joblib.load(C.MODELS / "pricing_params.joblib")
    return model, params


try:
    model, params = load_artifacts()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.run_pipeline` first.")
    st.stop()

sev = params["severity"]

st.title("Tennis Injury Risk — Premium Estimator")
st.caption(
    "One-season indemnity cover paying a fixed benefit each time the player "
    "fails to complete or take the court for a scheduled tour match. "
    "Fitted on 163,918 ATP and WTA player-matches, 2010–2025."
)

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Player profile")
    tour = st.selectbox("Tour", ["ATP", "WTA"])
    age = st.slider("Age", 16, 42, 27)
    rank = st.slider("Singles ranking", 1, 500, 60)
    height = st.slider("Height (cm)", 155, 210, 185)
    hand = st.selectbox("Playing hand", ["R", "L"])

    st.subheader("Workload and history")
    matches_28 = st.slider("Matches, last 28 days", 0, 20, 5)
    matches_90 = st.slider("Matches, last 90 days", 0, 45, 15)
    matches_365 = st.slider("Matches, last 365 days", 0, 100, 50)
    minutes_28 = st.slider("Court minutes, last 28 days", 0, 1500, 500, step=25)
    days_since = st.slider("Days since last match", 0, 200, 7)
    prior_365 = st.slider("Non-completions, last 365 days", 0, 6, 0)
    prior_career = st.slider("Non-completions, career", 0, 40, 2)
    career_matches = st.slider("Career tour matches", 5, 1200, 250)

    st.subheader("Match context")
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    level = st.selectbox(
        "Tournament tier", ["tour", "masters", "grand_slam", "finals", "team", "olympics"]
    )
    round_ord = st.slider("Round (0 = RR, 7 = final)", 0, 7, 3)
    surface_switch = st.checkbox("Surface change since last match")

    st.subheader("Pricing assumptions")
    matches_season = st.slider(
        "Expected matches this season", 5, 90, int(params["matches_per_season_median"])
    )
    mean_cost = st.slider("Mean claim cost (£)", 500, 6000, int(sev["mean_severity"]), step=100)
    loading = st.slider("Expense loading", 0.0, 0.40, C.EXPENSE_LOADING, step=0.01)

# --------------------------------------------------------------------------
chronic = max(minutes_28 / 28.0, 1e-6)
acwr = float(np.clip((minutes_28 / 28.0) / max((minutes_28 * 3) / 90.0, 1e-6), 0, 4))

row = pd.DataFrame(
    [
        {
            "age": age,
            "age_c_sq": (age - 26.0) ** 2,
            "height": height,
            "log_rank": np.log1p(rank),
            "log_rank_points": np.log1p(max(8000 / max(rank, 1), 1)),
            "matches_28d": matches_28,
            "matches_90d": matches_90,
            "matches_365d": matches_365,
            "minutes_28d": minutes_28,
            "acwr": acwr,
            "days_since_last": days_since,
            "prev_match_minutes": 100.0,
            "career_matches": career_matches,
            "years_on_tour": career_matches / 45.0,
            "prior_events_365d": prior_365,
            "prior_events_career": prior_career,
            "round_ord": round_ord,
            "match_in_event": min(round_ord, 4),
            "surface_switch": int(surface_switch),
            "is_qualifier": 0,
            "is_wildcard": 0,
            "tour": tour,
            "surface": surface,
            "level": level,
            "hand": hand,
        }
    ]
)

freq = float(model.predict_proba(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1][0])
pp = float(P.pure_premium(np.array([freq]), matches_season, mean_cost)[0])
op = float(P.office_premium(np.array([pp]), loading)[0])
sev_scaled = dict(sev)
sev_scaled["mean_severity"] = mean_cost
sev_scaled["var_severity"] = sev["var_severity"] * (mean_cost / sev["mean_severity"]) ** 2
sd = float(P.aggregate_sd(np.array([freq]), matches_season, sev_scaled)[0])

with right:
    st.subheader("Estimated premium")
    a, b, c = st.columns(3)
    a.metric("Per-match event probability", f"{freq:.2%}",
             delta=f"{freq / 0.0184 - 1:+.0%} vs tour average")
    b.metric("Expected events this season", f"{freq * matches_season:.2f}")
    c.metric("Annual office premium", f"£{op:,.0f}")

    st.write("")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": [
                    "Expected number of events, E[N]",
                    "Mean cost per event, E[X]",
                    "Pure premium",
                    f"Expense loading ({loading:.0%})",
                    "Office premium",
                    "SD of annual aggregate loss",
                ],
                "Value": [
                    f"{freq * matches_season:.3f}",
                    f"£{mean_cost:,.0f}",
                    f"£{pp:,.0f}",
                    f"£{op - pp:,.0f}",
                    f"£{op:,.0f}",
                    f"£{sd:,.0f}",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        f"**Volatility check.** The standard deviation of this player's annual "
        f"loss (£{sd:,.0f}) is {sd / max(pp, 1):.1f}× the pure premium. For a "
        f"low-frequency risk this is expected, and it is why an individual "
        f"policy needs either a risk loading or a large enough book to "
        f"diversify."
    )

    st.warning(
        "**Not a quotable price.** The event target is a proxy for injury, not "
        "a medical diagnosis, and the mean claim cost is an assumption with no "
        "empirical basis in this data — it is one of the two largest drivers of "
        "the premium. Model AUC is 0.651: this separates rating classes, it "
        "does not predict individuals. See docs/METHODOLOGY.md §7."
    )

st.divider()
st.caption(
    "Data: Tennis Abstract / Jeff Sackmann ATP and WTA databases via community "
    "archive, CC BY-NC-SA 4.0. Non-commercial use only."
)
