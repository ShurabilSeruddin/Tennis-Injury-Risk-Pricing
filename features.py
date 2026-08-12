"""
Stage 2 - feature engineering.

The governing rule in this module: every feature attached to a player-match row
must be computable strictly BEFORE that match starts. Court time in the match
itself, the result, and anything downstream of the outcome are all off limits,
because the target is whether the player completes this match.

Rolling windows are built with shift(1) inside each player's history so that
the current match never contributes to its own predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _rolling_by_time(g: pd.DataFrame, value_col: str, days: int) -> pd.Series:
    """
    Sum `value_col` over the trailing `days` days, excluding the current row.

    Implemented as a time-indexed rolling sum minus the current value, which
    keeps the window calendar-based (a player who plays 5 matches in 10 days
    is loaded differently to one who plays 5 in 10 weeks).
    """
    s = g.set_index("tourney_date")[value_col]
    total = s.rolling(f"{days}D", closed="both").sum()
    return (total - s).values


def _rolling_count(g: pd.DataFrame, days: int) -> np.ndarray:
    s = g.set_index("tourney_date")["_one"]
    total = s.rolling(f"{days}D", closed="both").sum()
    return (total - s).values


def build_features(pm: pd.DataFrame) -> pd.DataFrame:
    """Attach workload, recovery, experience and injury-history features."""
    df = pm.sort_values(["player_id", "tourney_date", "match_num"]).copy()
    df["_one"] = 1.0

    parts = []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.copy()

        # --- workload: how much tennis recently -------------------------
        for w in C.WINDOWS:
            g[f"matches_{w}d"] = _rolling_count(g, w)
            g[f"minutes_{w}d"] = _rolling_by_time(g, "minutes", w)

        # --- recovery: how long since the last match --------------------
        prev_date = g["tourney_date"].shift(1)
        g["days_since_last"] = (g["tourney_date"] - prev_date).dt.days
        g["prev_match_minutes"] = g["minutes"].shift(1).fillna(0.0)

        # --- experience: cumulative matches on tour ---------------------
        g["career_matches"] = np.arange(len(g), dtype=float)
        first = g["tourney_date"].iloc[0]
        g["years_on_tour"] = (g["tourney_date"] - first).dt.days / 365.25

        # --- injury history: prior non-completions ----------------------
        g["prior_events_365d"] = _rolling_by_time(g, "injury_event", 365)
        g["prior_events_career"] = g["injury_event"].cumsum() - g["injury_event"]

        # --- surface transition -----------------------------------------
        g["prev_surface"] = g["surface"].shift(1)
        g["surface_switch"] = (
            (g["prev_surface"].notna()) & (g["prev_surface"] != g["surface"])
        ).astype(int)

        # --- within-tournament fatigue ----------------------------------
        g["match_in_event"] = g.groupby("tourney_id").cumcount()

        # --- secondary target: extended absence AFTER this match --------
        # Used only to validate that the primary target behaves like injury.
        next_date = g["tourney_date"].shift(-1)
        gap = (next_date - g["tourney_date"]).dt.days
        g["gap_to_next"] = gap
        g["layoff_event"] = (
            (gap >= C.LAYOFF_DAYS) & (gap <= C.LAYOFF_RETURN_WINDOW)
        ).astype(int)
        # Last observed match for a player is right-censored: we cannot tell a
        # season-ending injury from a retirement from the sport, so exclude.
        g.loc[gap.isna(), "layoff_event"] = np.nan

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)

    # --- burn-in ---------------------------------------------------------
    # A player's first appearance in the window has no history, so their
    # workload features are structurally zero rather than genuinely low. We
    # require at least one prior year of observed play before a row is usable.
    out["has_history"] = (out["days_since_last"].notna()) & (out["career_matches"] >= 5)

    # --- derived / transformed -------------------------------------------
    out["log_rank"] = np.log1p(out["rank"])
    out["log_rank_points"] = np.log1p(out["rank_points"])
    # Age is centred before squaring. Raw age and age^2 correlate at r=0.995,
    # which inflates their variance inflation factors past 100 and makes the
    # two coefficients uninterpretable individually. Centring removes almost
    # all of that correlation while leaving the fitted curve identical.
    out["age_c"] = out["age"] - 26.0
    out["age_c_sq"] = out["age_c"] ** 2
    # Acute:chronic workload ratio, standard in sports science: recent load
    # relative to what the athlete is conditioned for. Values well above 1
    # indicate a spike.
    chronic = (out["minutes_90d"] / 90.0).replace(0, np.nan)
    out["acwr"] = (out["minutes_28d"] / 28.0) / chronic
    out["acwr"] = out["acwr"].clip(upper=4).fillna(1.0)
    out["days_since_last"] = out["days_since_last"].clip(upper=365)
    out["is_qualifier"] = (out["entry"].astype(str) == "Q").astype(int)
    out["is_wildcard"] = (out["entry"].astype(str) == "WC").astype(int)

    return out.drop(columns=["_one"])


# Feature set after collinearity screening (see docs/METHODOLOGY.md s.4.2).
# Removed and why:
#   load_per_day_28 - an exact rescaling of minutes_28d (VIF = inf)
#   minutes_90d     - VIF ~39 against matches_90d; matches_90d is preferred
#                     because it does not depend on the minutes imputation
#                     that fills 42% of WTA rows
#   age_sq          - replaced by age_c_sq, squared about a centred age
#   best_of         - r = 0.60 with the Grand Slam indicator, since ATP
#                     Slams are the only best-of-five events in the data;
#                     the level and tour dummies already carry the effect
NUMERIC_FEATURES = [
    "age", "age_c_sq", "height", "log_rank", "log_rank_points",
    "matches_28d", "matches_90d", "matches_365d",
    "minutes_28d", "acwr",
    "days_since_last", "prev_match_minutes",
    "career_matches", "years_on_tour",
    "prior_events_365d", "prior_events_career",
    "round_ord", "match_in_event",
    "surface_switch", "is_qualifier", "is_wildcard",
]

CATEGORICAL_FEATURES = ["tour", "surface", "level", "hand"]

TARGET = "injury_event"


def model_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Select analysis-ready rows and columns."""
    keep = (
        ["player_id", "player_name", "season", "tourney_date", TARGET, "layoff_event"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )
    df = feat.loc[feat["has_history"], keep].copy()
    # Height and hand are genuinely missing for some lower-ranked players.
    df["height"] = df["height"].fillna(df.groupby("tour")["height"].transform("median"))
    df["hand"] = df["hand"].fillna("U").replace({"nan": "U"})
    # Unranked players (protected ranking, wildcards) get a conservative
    # placeholder rather than being dropped.
    df["log_rank"] = df["log_rank"].fillna(np.log1p(500))
    df["log_rank_points"] = df["log_rank_points"].fillna(0.0)
    df = df.dropna(subset=["age"])
    return df.reset_index(drop=True)
