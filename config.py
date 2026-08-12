"""
Central configuration for the tennis injury-risk pricing project.

Every tunable assumption lives here so that the write-up, the notebook and the
Streamlit app all read from one source of truth. If a number appears in the
report, it should be traceable to this file or to a fitted model artefact.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"

for _p in (DATA_RAW, DATA_PROCESSED, OUT_FIG, OUT_TAB, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data sourcing
# --------------------------------------------------------------------------
# The canonical public tennis dataset (JeffSackmann/tennis_atp and
# JeffSackmann/tennis_wta) was taken offline by its author. See
# docs/METHODOLOGY.md section 2 for the provenance audit. We therefore pull
# from a community archive that preserves the identical schema.
ARCHIVE_REPO = "https://github.com/Aneeshers/tennis-sackmann-archive.git"
ARCHIVE_DIR = DATA_RAW / "tennis-sackmann-archive"

SEASON_START = 2010
SEASON_END = 2025  # inclusive; 2026 excluded as a partial season

# --------------------------------------------------------------------------
# Target definition
# --------------------------------------------------------------------------
# Primary target: the player failed to complete a scheduled match, recorded in
# the score field as RET (retired mid-match) or W/O (walkover / withdrawal).
# Secondary target: an extended absence from the tour, used to sanity-check
# that the primary target behaves like an injury signal rather than noise.
LAYOFF_DAYS = 90          # gap length that counts as an extended absence
LAYOFF_RETURN_WINDOW = 540  # must return within this window, else treated as
                            # career exit / right-censored rather than injury

# Rolling workload windows (days)
WINDOWS = (28, 90, 365)

# --------------------------------------------------------------------------
# Train / test protocol
# --------------------------------------------------------------------------
# Primary protocol is a temporal holdout: fit on the past, test on the future.
# A random split is ALSO reported, purely to demonstrate the leakage it causes.
TEMPORAL_SPLIT_YEAR = 2022  # train <= 2022, test 2023-2025
CV_FOLDS = 5
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Actuarial assumptions
# --------------------------------------------------------------------------
# All of these are assumptions, not findings. They are flagged as such in the
# report and every one of them is stress-tested in src/pricing.py.
MEAN_CLAIM_COST_GBP = 2_000.0   # average cost of one injury claim
SEVERITY_CV = 1.20              # coefficient of variation of claim severity
                                # (lognormal), reflecting a long right tail
EXPENSE_LOADING = 0.15          # insurer expense + profit margin (15%)
RISK_LOADING_ALT = 0.20         # alternative loading for sensitivity testing

# Policy exposure basis: cover is written for one season, and the number of
# matches a player contests is the exposure measure.
# The naive per-player median (5.7 matches) is dominated by qualifiers who
# appear once and never return. That is not the risk being underwritten. The
# rating basis is instead the match-weighted median: the number of matches
# played by the player who contests the typical match on tour. Computed in
# run_pipeline.py, this constant is the fallback only.
DEFAULT_MATCHES_PER_SEASON = 31

BOOTSTRAP_N = 400               # bootstrap replicates for premium intervals
