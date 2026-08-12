"""
Stage 1 - ingestion and cleaning.

Reads raw ATP and WTA match files, harmonises the two tours onto a common
schema, and explodes each match into two player-match rows (one per
participant). Nothing here looks forward in time; feature construction and
target labelling happen in features.py.
"""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Tournament level harmonisation
# --------------------------------------------------------------------------
# ATP and WTA use different level codes for what is economically the same
# tier of event. Collapsing them lets a single model span both tours.
LEVEL_MAP = {
    # ATP
    "G": "grand_slam",     # Grand Slam
    "M": "masters",        # Masters 1000
    "A": "tour",           # ATP Tour 500/250
    "F": "finals",         # Tour Finals
    "D": "team",           # Davis Cup
    "O": "olympics",
    # WTA
    "PM": "masters",       # Premier Mandatory -> WTA 1000
    "P": "masters",        # Premier -> WTA 1000/500 boundary
    "I": "tour",           # International -> WTA 250
    "W": "team",           # Fed / BJK Cup
    "C": "tour",
}

ROUND_ORDER = {
    "RR": 0, "BR": 0, "R128": 1, "R64": 2, "R32": 3,
    "R16": 4, "QF": 5, "SF": 6, "F": 7,
}


def ensure_raw_data() -> None:
    """Clone the archive of tour match files if it is not already present."""
    if C.ARCHIVE_DIR.exists():
        return
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Cloning match archive into {C.ARCHIVE_DIR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", C.ARCHIVE_REPO, str(C.ARCHIVE_DIR)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _load_tour(tour: str) -> pd.DataFrame:
    """Load and concatenate one tour's yearly match files."""
    sub = C.ARCHIVE_DIR / tour.lower()
    frames = []
    for year in range(C.SEASON_START, C.SEASON_END + 1):
        path = sub / f"{tour.lower()}_matches_{year}.csv"
        if not path.exists():
            print(f"  ! missing {path.name}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No match files found for {tour} in {sub}")
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour.upper()
    return df


def _parse_outcome(score: pd.Series) -> pd.DataFrame:
    """
    Classify how a match ended, from the free-text score string.

    Returns three mutually exclusive-ish flags:
      is_ret  - loser retired mid-match
      is_wo   - walkover / withdrawal before play, or default
      is_incomplete - either of the above

    Note on directionality: in this data source the non-completing player is
    always recorded as the loser. That is a definitional artefact, not a
    finding, and it is what lets us attach the event to a specific player.
    """
    s = score.fillna("").astype(str).str.upper()
    is_ret = s.str.contains(r"\bRET\b", regex=True)
    is_wo = s.str.contains(r"W/O|WALKOVER|\bDEF\b", regex=True)
    # A handful of rows are blank or say "UNFINISHED"; treat as unusable, not
    # as an event, to avoid manufacturing positives out of missing data.
    unusable = (s.str.strip() == "") | s.str.contains("UNFINISHED|UNKNOWN")
    return pd.DataFrame(
        {
            "is_ret": is_ret & ~unusable,
            "is_wo": is_wo & ~unusable,
            "score_unusable": unusable,
        }
    )


def _sets_played(score: pd.Series) -> pd.Series:
    """Count completed sets in a score string - used to impute missing minutes."""
    pat = re.compile(r"\d+-\d+")
    return score.fillna("").astype(str).map(lambda x: len(pat.findall(x)))


def load_matches() -> pd.DataFrame:
    """Load, clean and harmonise both tours into one match-level frame."""
    ensure_raw_data()
    df = pd.concat([_load_tour("atp"), _load_tour("wta")], ignore_index=True)

    # --- basic cleaning -------------------------------------------------
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df[df["tourney_date"].notna()].copy()
    df["season"] = df["tourney_date"].dt.year

    # Surface arrives with inconsistent casing ('clay' vs 'Clay') and blanks.
    df["surface"] = (
        df["surface"].astype(str).str.strip().str.title().replace({"Nan": np.nan, "": np.nan})
    )
    df["surface"] = df["surface"].fillna("Unknown")
    # Carpet is a near-extinct surface (<0.3% of rows); folding it into Hard
    # avoids a category with too few events to estimate a coefficient on.
    df["surface"] = df["surface"].replace({"Carpet": "Hard"})

    df["level"] = df["tourney_level"].map(LEVEL_MAP).fillna("tour")
    df["round_ord"] = df["round"].map(ROUND_ORDER).fillna(3).astype(int)
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)

    outcome = _parse_outcome(df["score"])
    df = pd.concat([df, outcome], axis=1)
    df = df[~df["score_unusable"]].copy()

    # --- impute missing court time --------------------------------------
    # `minutes` is missing for ~10% of ATP and ~42% of WTA rows. Dropping
    # those rows would bias the sample toward better-covered (bigger) events,
    # so instead we impute from sets played, fitted separately per tour and
    # per best-of format, and carry a flag so the model can learn any residual
    # difference between observed and imputed load.
    df["n_sets"] = _sets_played(df["score"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    # Guard against absurd values (data entry errors: 0 or >6 hours).
    df.loc[(df["minutes"] <= 10) | (df["minutes"] > 400), "minutes"] = np.nan
    df["minutes_imputed"] = df["minutes"].isna()

    med = (
        df.dropna(subset=["minutes"])
        .groupby(["tour", "best_of", "n_sets"])["minutes"]
        .median()
        .rename("min_med")
    )
    df = df.merge(med, on=["tour", "best_of", "n_sets"], how="left")
    global_med = df["minutes"].median()
    df["minutes"] = df["minutes"].fillna(df["min_med"]).fillna(global_med)
    df = df.drop(columns=["min_med"])

    return df


def to_player_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each match into two player-match rows.

    This is the modelling unit: one row = one player contesting one match, and
    one unit of exposure. The event flag attaches to the player who failed to
    complete, which by construction is the loser.
    """
    common = [
        "tour", "season", "tourney_id", "tourney_name", "tourney_date",
        "match_num", "surface", "level", "round", "round_ord", "best_of",
        "minutes", "minutes_imputed", "is_ret", "is_wo", "draw_size",
    ]

    def side(prefix: str, is_winner: bool) -> pd.DataFrame:
        out = df[common].copy()
        out["player_id"] = df[f"{prefix}_id"]
        out["player_name"] = df[f"{prefix}_name"]
        out["age"] = pd.to_numeric(df[f"{prefix}_age"], errors="coerce")
        out["height"] = pd.to_numeric(df[f"{prefix}_ht"], errors="coerce")
        out["hand"] = df[f"{prefix}_hand"]
        out["rank"] = pd.to_numeric(df[f"{prefix}_rank"], errors="coerce")
        out["rank_points"] = pd.to_numeric(df[f"{prefix}_rank_points"], errors="coerce")
        out["entry"] = df[f"{prefix}_entry"]
        out["won"] = int(is_winner)
        # Only the losing player can carry a non-completion event.
        out["event_ret"] = (~is_winner) & df["is_ret"]
        out["event_wo"] = (~is_winner) & df["is_wo"]
        return out

    pm = pd.concat([side("winner", True), side("loser", False)], ignore_index=True)
    pm["injury_event"] = (pm["event_ret"] | pm["event_wo"]).astype(int)

    pm["player_id"] = pm["player_id"].astype(str)
    pm = pm[pm["player_id"].notna() & (pm["player_id"] != "nan")].copy()

    # A walkover means no tennis was played, so it contributes no court time
    # to the player's subsequent workload history.
    pm.loc[pm["event_wo"], "minutes"] = 0.0

    pm = pm.sort_values(["player_id", "tourney_date", "match_num"]).reset_index(drop=True)
    return pm


if __name__ == "__main__":
    matches = load_matches()
    print(f"matches loaded: {len(matches):,}")
    panel = to_player_matches(matches)
    print(f"player-match rows: {len(panel):,}")
    print(f"event rate: {panel['injury_event'].mean():.3%}")
